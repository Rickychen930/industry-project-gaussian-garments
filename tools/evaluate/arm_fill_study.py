"""Could a learned fill (the LaMa already on disk) make the erased arm read as an arm?

Variants on the ARM hole only (a crop around the arm tube, longer side 128 / 192 / 256 px), applied to
what the live fill produced (`out`, the tube):

  tube        the live fill (the reference)
  lama        LaMa fills the tube region (the room is what it paints: it was made to remove objects)
  lama_gated  `lama`, kept only where its colour is within `SKIN_CHROMA` of the tube's (else the tube stays)
  lama_hint   LaMa sees `out` (the skin-coloured tube drawn) and re-synthesises only blotches (~40 %) of
              the tube: the stripe prior tells it what to paint, the blotches give it texture to invent

    nice -n 15 .venv/bin/python -m tools.evaluate.arm_fill_study metrics 00176_upper   # synthetic arm metrics + timing
    nice -n 15 .venv/bin/python -m tools.evaluate.arm_fill_study figure                # docs/figures/fill-quality-4.jpg
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch

from tools.evaluate import fill_quality as fq
from tools.evaluate.arm_visibility import CACHE as VIS_CACHE
from tools.evaluate.arm_visibility import tube_pixels
from tryon.paths import BASELINE, REPO_ROOT
from tryon.removal.remover import ClothingRemover

SIDES = (128, 192, 256)
CONTEXT = 12
FIGURE_VIDEOS = ("00176_upper", "00170_upper", "00122_outer", "00152_outer")
_model = None


def lama_model():
    global _model
    if _model is None:
        _model = torch.jit.load(str(ClothingRemover.LAMA_CHECKPOINT), map_location="cpu").eval()
    return _model


def lama_on(frame, mask, side, context=CONTEXT):
    """`frame` with the `mask` pixels filled by LaMa run on the mask's crop (longer side `side`)."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return frame.copy()
    h, w = mask.shape
    y0, y1, x0, x1 = max(rows[0] - context, 0), min(rows[-1] + 1 + context, h), max(cols[0] - context, 0), min(cols[-1] + 1 + context, w)
    crop, hole = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    scale = side / max(crop.shape[:2])
    run_h, run_w = max(8, round(crop.shape[0] * scale / 8) * 8), max(8, round(crop.shape[1] * scale / 8) * 8)
    rgb = cv2.cvtColor(cv2.resize(crop, (run_w, run_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    small = cv2.resize(hole.astype(np.float32), (run_w, run_h), interpolation=cv2.INTER_LINEAR) > 0
    with torch.no_grad():
        out = lama_model()(torch.from_numpy(rgb).permute(2, 0, 1)[None], torch.from_numpy(small.astype(np.float32))[None, None])
    out = np.clip(out[0].permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)
    result = frame.copy()
    result[y0:y1, x0:x1][hole] = out[hole]
    return result


def blotches(shape, seed, share=0.4, sigma=3.0):
    """A random blob mask covering about `share` of the pixels."""
    noise = cv2.GaussianBlur(np.random.default_rng(seed).standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return noise > np.quantile(noise, 1.0 - share)


def apply(variant, base, plain_source, tube, side, seed=0):
    """`base` (the live fill) with the arm `tube` redone by `variant`; `plain_source` is the frame LaMa fills from."""
    if variant == "tube" or not tube.any():
        return base
    if variant == "lama":
        return lama_on(plain_source, tube, side)
    if variant == "lama_gated":
        out = lama_on(plain_source, tube, side)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = np.median(cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)[tube], axis=0)
        near = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2]) < ClothingRemover.SKIN_CHROMA
        result = base.copy()
        result[tube & near] = out[tube & near]
        return result
    if variant == "lama_hint":
        blot = tube & blotches(tube.shape, seed)
        return lama_on(base, blot, side, context=CONTEXT) if blot.any() else base
    raise ValueError(variant)


LAMA_VARIANTS = ("tube", "lama", "lama_gated", "lama_hint")


def metrics(video):
    """Synthetic arm cases of `fill_quality` (truth known): the live fill, then each LaMa variant on the arm tube."""
    samples = fq.load(video)
    remover = ClothingRemover(schp_checkpoint=fq.ATR_CHECKPOINT) if hasattr(fq, "ATR_CHECKPOINT") else None
    if remover is None:
        from tryon.removal.parsing_schp import ATR_CHECKPOINT

        remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.KEEP_INNER_LAYER = False
    rows = []
    for sample in samples:
        spared = fq._eroded(np.isin(sample["labels"], (14, 15)) & ~sample["clothing"], 5)
        for name, hole, tube, keep, points in fq.arm_cases(sample, remover.ARM_TUBE_WIDTH):
            base = fq.run_fill(remover, sample, hole, tube, keep, "labels", points)
            truth_hole = hole & ~keep
            fill_hole = truth_hole & ~spared
            region = tube & hole & ~keep
            for side in SIDES:
                for variant in LAMA_VARIANTS:
                    if variant == "tube" and side != SIDES[0]:
                        continue
                    out = apply(variant, base, sample["image"], region, side)
                    for kind, scored in (("armfull", truth_hole), ("arm", fill_hole)):
                        if scored.sum() >= fq.MIN_HOLE:
                            rows.append({"variant": variant, "side": side, "kind": kind, "pixels": int(scored.sum()), **fq.hole_scores(sample["image"], out, scored)})
    table = {}
    for variant in LAMA_VARIANTS:
        for side in SIDES:
            if variant == "tube" and side != SIDES[0]:
                continue
            entry = {f"{kind}_{key}": fq.mean_of(rows, key, variant=variant, side=side, kind=kind) for kind in ("armfull", "arm") for key in ("psnr", "ssim", "de76", "seam")}
            table[f"{variant}@{side}" if variant != "tube" else "tube"] = entry
    (BASELINE / f"arm-fill-lama-{video}.json").write_text(json.dumps(table, indent=1))
    for name, entry in table.items():
        print(f"{name:18s}" + "".join(f"{key}={value:7.3f} " for key, value in entry.items()))
    return table


def timing(video):
    """Min of 10 LaMa runs per crop side on a typical arm tube of `video`."""
    import pickle

    samples = pickle.load(open(VIS_CACHE / f"{video}.pkl", "rb"))
    sample = max(samples, key=lambda s: tube_pixels(s, ClothingRemover.ARM_TUBE_WIDTH).sum())
    tube = tube_pixels(sample, ClothingRemover.ARM_TUBE_WIDTH)
    rows, cols = np.flatnonzero(tube.any(axis=1)), np.flatnonzero(tube.any(axis=0))
    print("tube box", rows[-1] - rows[0] + 1, "x", cols[-1] - cols[0] + 1)
    lama_on(sample["frame"], tube, 128)
    out = {}
    for side in SIDES:
        runs = []
        for _ in range(10):
            started = time.perf_counter()
            lama_on(sample["frame"], tube, side)
            runs.append((time.perf_counter() - started) * 1000)
        out[side] = min(runs)
        print(f"side {side}: min of 10 = {out[side]:.0f} ms")
    return out


def figure(path, frames=3):
    """real | tube | LaMa variants of the arm crop, `frames` frames x the videos of FIGURE_VIDEOS."""
    import pickle

    sheet_rows = []
    for video in FIGURE_VIDEOS:
        samples = pickle.load(open(VIS_CACHE / f"{video}.pkl", "rb"))
        ranked = sorted(range(len(samples)), key=lambda k: -tube_pixels(samples[k], ClothingRemover.ARM_TUBE_WIDTH).sum())
        for k in sorted(ranked[:frames]):
            s = samples[k]
            tube = tube_pixels(s, ClothingRemover.ARM_TUBE_WIDTH)
            if not tube.any():
                continue
            rows, cols = np.flatnonzero(tube.any(axis=1)), np.flatnonzero(tube.any(axis=0))
            window = (slice(max(rows[0] - 20, 0), rows[-1] + 20), slice(max(cols[0] - 20, 0), cols[-1] + 20))
            tiles = [("real", s["frame"])] + [(v, apply(v, s["erased"], s["frame"], tube, 192)) for v in ("tube", "lama", "lama_gated", "lama_hint")]
            crops = []
            for label, tile in tiles:
                crop = tile[window].copy()
                factor = 300 / crop.shape[0]
                crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
                cv2.putText(crop, f"{video} f{s['t']} {label}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                crops.append(crop)
            sheet_rows.append(np.hstack(crops))
    width = max(r.shape[1] for r in sheet_rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 3, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in sheet_rows])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return sheet.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["metrics", "timing", "figure"])
    parser.add_argument("video", nargs="?", default="00176_upper")
    args = parser.parse_args()
    if args.command == "metrics":
        metrics(args.video)
    elif args.command == "timing":
        timing(args.video)
    else:
        print(figure(REPO_ROOT / "docs" / "figures" / "fill-quality-4.jpg"))


if __name__ == "__main__":
    main()
