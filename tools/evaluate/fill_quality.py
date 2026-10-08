"""How natural is the ERASED region? A ground-truth test and a no-ground-truth test of the fill.

The fit audit named the erased real clothing as the weakest thing on screen: grey patches at the
shoulder and armpit, bare arms filled with one flat pale colour, shorts erased into grey. There is
no photograph of "what is under the shirt", so two measurements stand in for it.

SYNTHETIC (ground truth). On the release videos' photo panels, holes with the shapes of the real
erase masks are cut where the pixels are known, the live fill (`ClothingRemover.remove`, with the
causally learned background plate and the fitted body, as `LiveSession._erase` calls it) runs on
them, and the result is scored against the original pixels in the hole:

  arm     the live arm stroke (0.45 shoulder widths, hand kept) on an arm that wears nothing: the
          truth is the bare arm and the room beside it
  graft   the frame's real erase mask (torso, shoulder, sleeve, without the strokes) shifted to a
          place free of clothing that straddles the person and the room: the truth is whatever the
          shifted hole covers (legs, arm, floor, wall)

  psnr, ssim (local, over the hole), dE76 (CIELAB, mean), seam (mean |gradient of the fill -
  gradient of the truth| in the 3 px band around the hole's edge: what a visible seam adds).

REAL (no ground truth). The live masks on the same frames; the fill under the arm tube
(`ARM_TUBE_WIDTH` strokes) against the bare skin the same person shows in that frame (SCHP arm
labels outside the hole, else the face): `skin_de` the CIELAB distance of the two mean colours,
`texture` the Laplacian variance of the fill over that of the visible skin (1 = as much grain; a flat
fill is ~0). The wearer's own lower body kept: `lower_erased` is the share of the parser's lower-body pixels (shorts, trousers) the mask covers.

Variants are class-attribute settings of `ClothingRemover`, interleaved frame by frame so a load
spike hits all of them alike; `baseline` is the committed fill.

    .venv/bin/python -m tools.evaluate.fill_quality prepare 00176_upper     # one video per process
    .venv/bin/python -m tools.evaluate.fill_quality score 00176_upper --variants baseline,skin_model
    .venv/bin/python -m tools.evaluate.fill_quality report                  # json from the cache
    .venv/bin/python -m tools.evaluate.fill_quality figure                  # docs/figures/fill-quality.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import panel_frames, write_json
from tools.evaluate.fit_audit import boundary, delta_e76, lab_of, laplacian_variance, psnr
from tryon.paths import BASELINE, REPO_ROOT, SCRATCH

VIDEOS = ("00152_outer", "00170_upper", "00134_upper", "10006_outer", "00190_upper", "00176_upper", "00122_outer")
CORE = VIDEOS[:5]
HEIGHT = 870
FRAMES = 150
PARSE_STEP = 3  # the plate and the parser see every third frame, as a slow machine would
EVAL_FRAMES = (36, 54, 72, 90, 108, 126, 144)
CACHE = Path(SCRATCH) / "fill_quality"
SEAM_BAND = 3
MIN_HOLE = 300
MIN_SKIN = 150  # px of visible skin a reference needs

# What a variant changes: ClothingRemover class attributes (see the remover's docstrings).
VARIANTS = {
    "baseline": {"LABEL_FILL": False},  # the committed fill before the label-aware one
    "labels_flat": {"LABEL_FILL": True, "SKIN_MODEL": False, "GRAIN": 0.0},
    "labels": {"LABEL_FILL": True},
    "labels_nograin": {"LABEL_FILL": True, "GRAIN": 0.0},
    "labels_w18": {"LABEL_FILL": True, "ARM_TUBE_WIDTH": 0.18},
    "labels_w26": {"LABEL_FILL": True, "ARM_TUBE_WIDTH": 0.26},
    "labels_w30": {"LABEL_FILL": True, "ARM_TUBE_WIDTH": 0.30},
    "baseline_tubebody": {"_body": "tube"},
    "labels_tubebody": {"LABEL_FILL": True, "_body": "tube"},
}


# ============================================================== pure metric helpers


def ssim_over(truth, out, region):
    """Mean local SSIM (7 px window, per channel) over `region`, computed on the region's bounding box."""
    from skimage.metrics import structural_similarity

    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    window = (slice(max(rows[0] - 4, 0), rows[-1] + 5), slice(max(cols[0] - 4, 0), cols[-1] + 5))
    _, ssim_map = structural_similarity(truth[window], out[window], channel_axis=2, full=True, data_range=255, win_size=7)
    return float(ssim_map.mean(axis=2)[region[window]].mean())


def gradient_magnitude(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def seam_score(truth, out, hole, band=SEAM_BAND):
    """Mean | |grad out| - |grad truth| | in the `band` px each side of the hole's edge."""
    outline = boundary(hole)
    ring = cv2.dilate(outline.astype(np.uint8), np.ones((2 * band + 1,) * 2, np.uint8)).astype(bool)
    if not ring.any():
        return 0.0
    return float(np.abs(gradient_magnitude(out)[ring] - gradient_magnitude(truth)[ring]).mean())


def hole_scores(truth, out, hole):
    """psnr / ssim / dE76 / seam of `out` against `truth` over the boolean `hole`."""
    return {
        "psnr": psnr(out[hole], truth[hole]),
        "ssim": ssim_over(truth, out, hole),
        "de76": float(delta_e76(out[hole], truth[hole]).mean()),
        "seam": seam_score(truth, out, hole),
    }


def mean_lab_distance(bgr_a, bgr_b):
    """CIE76 distance of the mean colours of two (N, 3) BGR pixel sets."""
    a = lab_of(np.asarray(bgr_a, np.float32).mean(axis=0)[None].astype(np.uint8))[0]
    b = lab_of(np.asarray(bgr_b, np.float32).mean(axis=0)[None].astype(np.uint8))[0]
    return float(np.linalg.norm(a - b))


# ============================================================== preparing one video


def _eroded(mask, size):
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def prepare(video, frames=FRAMES):
    """Parses every PARSE_STEP-th frame, learns the plate causally, and keeps, for the EVAL_FRAMES, what
    `LiveSession._erase` has when it calls the remover (image, keypoints, parse, the live erase mask, the
    body, the plate). Cached as a pickle."""
    from tools.evaluate.tracking_jitter import landmarks_for
    from tryon.app.person import torso_visible, track_step
    from tryon.body.fitter import BodyFitter
    from tryon.removal.background import BackgroundPlate
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover, arm_strokes
    from tryon.app.pipeline import ERASE_NEAR_BODY
    import collections

    from tryon.removal import background

    background.USE_GL = False
    images = panel_frames(video, HEIGHT, frames)
    points, _ = landmarks_for(video, HEIGHT, frames)
    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    plate, fitter, errors = BackgroundPlate(), None, collections.deque(maxlen=10)
    samples = []
    for t in range(0, len(images), PARSE_STEP):
        image, p = images[t], points[t]
        p = None if np.isnan(p).any() else np.asarray(p[:, :3], np.float64)
        parse = remover.parse_clothing(image, keypoints=p)
        body = None
        if p is not None and torso_visible(p, tracking=fitter is not None):
            if fitter is None:
                fitter = BodyFitter()
                fitter.fit(p)
            else:
                track_step(fitter, p, errors, HEIGHT)
            fill = background.body_fill(fitter, image.shape)
            silhouette = background.body_silhouette(fitter, image.shape, fill=fill)
            body = silhouette | arm_strokes(p, image.shape)
            near = background.body_silhouette(fitter, image.shape, grow=ERASE_NEAR_BODY, fill=fill) | arm_strokes(p, image.shape)
        else:
            fitter = None
        if t in EVAL_FRAMES and body is not None:
            samples.append({
                "t": t, "image": image.copy(), "points": p, "labels": parse.labels, "clothing": parse.clothing,
                "mask": parse.mask & near, "body": body, "silhouette": silhouette, "plate": (plate.plate.copy(), plate.count.copy()),
            })
        if parse.labels is not None:
            plate.update(image, parse.labels != 0)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{video}.pkl", "wb") as handle:
        pickle.dump(samples, handle)
    return samples


def load(video):
    with open(CACHE / f"{video}.pkl", "rb") as handle:
        return pickle.load(handle)


def plate_of(state):
    from tryon.removal.background import BackgroundPlate

    plate = BackgroundPlate()
    plate.plate, plate.count = state[0].copy(), state[1].copy()
    return plate


# ============================================================== the cases


def one_arm(points, arm):
    """Keypoints with only arm `arm` (0 left, 1 right) visible, for `arm_strokes`."""
    out = points.copy()
    other = (13, 15) if arm == 0 else (14, 16)
    out[list(other), 2] = 0.0
    return out


def arm_cases(sample, tube_width):
    """Holes shaped as one arm's live stroke, on arms that wear nothing: [(name, hole, arm tube, keep)]. The
    tube (`ARM_TUBE_WIDTH` strokes, what the fill dresses in skin) is for `tube_width`."""
    from tryon.removal.remover import arm_strokes, hand_discs

    p, shape = sample["points"], sample["image"].shape
    labels = sample["labels"]
    clothed = cv2.dilate((np.isin(labels, (4, 7))).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    skin = np.isin(labels, (14, 15))
    keep = hand_discs(p, shape)
    out = []
    for arm in (0, 1):
        q = one_arm(p, arm)
        stroke = arm_strokes(q, shape) & ~clothed  # the part clear of the real shirt: its truth is known
        tube = arm_strokes(q, shape, width_fraction=tube_width) & stroke
        if stroke.sum() < MIN_HOLE or (arm_strokes(q, shape, width_fraction=0.22) & stroke & skin).sum() < 0.25 * max(tube.sum(), 1):
            continue
        out.append((f"arm{arm}", stroke, tube, keep))
    return out


def graft_cases(sample, count=2):
    """The frame's real erase mask without the strokes, shifted to clothing-free places that straddle
    the person and the room: [(name, hole)]."""
    from tryon.removal.remover import arm_strokes

    p, labels, mask = sample["points"], sample["labels"], sample["mask"]
    shape = sample["image"].shape
    base = mask & ~arm_strokes(p, shape)
    if base.sum() < 4 * MIN_HOLE:
        return []
    ys, xs = np.nonzero(base)
    clothed = cv2.dilate((mask | sample["clothing"]).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    person = (labels != 0) & ~np.isin(labels, (4, 7))
    # work at a quarter of the frame: the search is a grid of shifts
    scale = 0.25
    def small(m):
        return cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)

    base_s, clothed_s, person_s = small(base), small(clothed), small(person)
    sy, sx = np.nonzero(base_s)
    height, width = base_s.shape
    found = []
    rng = np.random.default_rng(sample["t"])
    for dy, dx in rng.integers(-int(0.45 * height), int(0.45 * height) + 1, (400, 2)):
        ty, tx = sy + dy, sx + dx
        if ty.min() < 0 or tx.min() < 0 or ty.max() >= height or tx.max() >= width:
            continue
        if clothed_s[ty, tx].any():
            continue
        found.append((abs(float(person_s[ty, tx].mean()) - 0.5), int(dy), int(dx)))
    found.sort()
    out = []
    for k, (_, dy, dx) in enumerate(found[:count]):
        shift = np.float32([[1, 0, dx / scale], [0, 1, dy / scale]])
        hole = cv2.warpAffine(base.astype(np.uint8), shift, (shape[1], shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
        if hole.sum() >= MIN_HOLE:
            out.append((f"graft{k}", hole))
    return out


# ============================================================== scoring


def apply_variant(remover, name):
    for key, value in VARIANTS[name].items():
        if not key.startswith("_"):
            setattr(remover, key, value)


def reset_variant(remover):
    for key in {k for v in VARIANTS.values() for k in v if not k.startswith("_")}:
        if key in remover.__dict__:
            delattr(remover, key)


def run_fill(remover, sample, hole, arms, keep, variant="baseline"):
    """The live `LiveSession._erase` call, for `hole`. `_body: tube` hands the plate the fitted body plus
    the narrow arm tube instead of the body plus the 0.45-wide strokes (what `LiveSession._erase` builds)."""
    from tryon.removal.remover import ClothingParse

    parse = ClothingParse(hole, sample["labels"], sample["clothing"])
    plate = plate_of(sample["plate"])
    body = sample["body"]
    if VARIANTS[variant].get("_body") == "tube" and arms is not None:
        body = sample["silhouette"] | np.asarray(arms, bool)
    return remover.remove(sample["image"], hole, arms=arms, keep=keep, plate=plate, body=body, parse=parse)


def synthetic_rows(remover, samples, variants):
    rows = []
    for sample in samples:
        # grafted holes: no strokes, no tint
        grafts = [(name, hole, None, None) for name, hole in graft_cases(sample)]
        for variant in variants:
            reset_variant(remover)
            apply_variant(remover, variant)
            cases = arm_cases(sample, remover.ARM_TUBE_WIDTH) + grafts
            for name, hole, tube, keep in cases:
                started = time.perf_counter()
                out = run_fill(remover, sample, hole, tube, keep, variant)
                took = (time.perf_counter() - started) * 1000
                truth_hole = hole & ~(keep if keep is not None else np.zeros_like(hole))
                if truth_hole.sum() < MIN_HOLE:
                    continue
                rows.append({"t": sample["t"], "case": name, "kind": "arm" if name.startswith("arm") else "graft", "variant": variant,
                             "pixels": int(truth_hole.sum()), "ms": took, **hole_scores(sample["image"], out, truth_hole)})
        reset_variant(remover)
    return rows


def real_rows(remover, samples, variants):
    from tryon.removal.remover import arm_strokes, hand_discs

    rows = []
    for sample in samples:
        p, shape, labels = sample["points"], sample["image"].shape, sample["labels"]
        mask = sample["mask"]
        tube = arm_strokes(p, shape, width_fraction=0.22)  # the reference region, whatever a variant draws
        keep = hand_discs(p, shape)
        visible = ~cv2.dilate(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        arm_skin = _eroded(np.isin(labels, (14, 15)) & visible & ~sample["clothing"], 5)
        face = _eroded(labels == 11, 5)
        reference = arm_skin if arm_skin.sum() >= MIN_SKIN else face
        if reference.sum() < MIN_SKIN:
            continue
        under = _eroded(mask & tube & ~keep, 5)
        if under.sum() < MIN_HOLE // 2:
            continue
        lower = np.isin(labels, (5, 6, 12, 13))
        rows_below = np.arange(shape[0])[:, None] > p[[23, 24], 1].mean() if min(p[23, 2], p[24, 2]) > 0.5 else None
        gray = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2GRAY)
        skin_texture = laplacian_variance(gray, reference)
        for variant in variants:
            reset_variant(remover)
            apply_variant(remover, variant)
            live_tube = arm_strokes(p, shape, width_fraction=remover.ARM_TUBE_WIDTH)
            out = run_fill(remover, sample, mask, live_tube, keep, variant)
            out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            row = {"t": sample["t"], "variant": variant, "pixels": int(under.sum()),
                   "skin_de": mean_lab_distance(out[under], sample["image"][reference]),
                   "texture": laplacian_variance(out_gray, under) / max(skin_texture, 1e-6),
                   }
            if lower.sum() > 0 and rows_below is not None:
                below = lower & rows_below
                row["lower_erased"] = float((mask & below).sum() / max(below.sum(), 1)) if below.sum() > 200 else float("nan")
            rows.append(row)
        reset_variant(remover)
    return rows


def mean_of(rows, key, **match):
    chosen = [r for r in rows if all(r.get(k) == v for k, v in match.items()) and np.isfinite(r.get(key, np.nan))]
    if not chosen:
        return float("nan")
    return float(np.average([r[key] for r in chosen], weights=[max(r.get("pixels", 1), 1) for r in chosen]))


def summarise(rows):
    out = {}
    for variant in sorted({r["variant"] for r in rows}):
        entry = {}
        for kind in ("arm", "graft"):
            for key in ("psnr", "ssim", "de76", "seam"):
                entry[f"{kind}_{key}"] = mean_of(rows, key, variant=variant, kind=kind)
        entry["ms"] = float(np.mean([r["ms"] for r in rows if r["variant"] == variant and "ms" in r]))
        for key in ("skin_de", "texture", "lower_erased"):
            entry[key] = mean_of(rows, key, variant=variant, kind=None)
        out[variant] = entry
    return out


def score(video, variants):
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover

    samples = load(video)
    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.KEEP_INNER_LAYER = False
    synthetic = synthetic_rows(remover, samples, variants)
    real = real_rows(remover, samples, variants)
    for row in real:
        row["kind"] = None
    return {"video": video, "synthetic": synthetic, "real": real}


# ============================================================== the figure

FIGURE_VIDEOS = ("00176_upper", "00152_outer", "00122_outer", "00170_upper")
FIGURE_FRAMES_PER_VIDEO = 3
CROP_PAD = 24
CROP_HEIGHT = 330


def erased_crop(remover, sample, variant):
    """The live erase of this frame under `variant`, as the crop of the mask's bounding box."""
    from tryon.removal.remover import arm_strokes, hand_discs

    tube = arm_strokes(sample["points"], sample["image"].shape, width_fraction=remover.ARM_TUBE_WIDTH)
    reset_variant(remover)
    apply_variant(remover, variant)
    out = run_fill(remover, sample, sample["mask"], tube, hand_discs(sample["points"], sample["image"].shape), variant)
    reset_variant(remover)
    return out


def figure(variants, path):
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover

    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.KEEP_INNER_LAYER = False
    sheet_rows = []
    for video in FIGURE_VIDEOS:
        samples = load(video)
        picks = [samples[i] for i in np.linspace(1, len(samples) - 2, FIGURE_FRAMES_PER_VIDEO).round().astype(int)]
        for sample in picks:
            mask = sample["mask"]
            rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            if rows.size == 0:
                continue
            height, width = mask.shape
            window = (slice(max(rows[0] - CROP_PAD, 0), min(rows[-1] + CROP_PAD, height)), slice(max(cols[0] - CROP_PAD, 0), min(cols[-1] + CROP_PAD, width)))
            tiles = [sample["image"]] + [erased_crop(remover, sample, v) for v in variants]
            crops = []
            for k, tile in enumerate(tiles):
                crop = tile[window].copy()
                crop = cv2.resize(crop, None, fx=CROP_HEIGHT / crop.shape[0], fy=CROP_HEIGHT / crop.shape[0], interpolation=cv2.INTER_AREA)
                label = f"{video} f{sample['t']} " + ("real" if k == 0 else variants[k - 1])
                cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                crops.append(crop)
            sheet_rows.append(np.hstack(crops))
    pad = max(r.shape[1] for r in sheet_rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 3, 0, pad - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in sheet_rows])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return sheet.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "score", "report", "figure"])
    parser.add_argument("video", nargs="?")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    if args.command == "prepare":
        samples = prepare(args.video)
        print(args.video, "prepared", len(samples), "frames")
    elif args.command == "score":
        result = score(args.video, args.variants.split(","))
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"score_{args.video}.json").write_text(json.dumps(result))
        print(json.dumps(summarise(result["synthetic"] + result["real"]), indent=1))
    elif args.command == "figure":
        print(figure(args.variants.split(","), REPO_ROOT / "docs" / "figures" / "fill-quality.jpg"))
    elif args.command == "report":
        rows = []
        for video in CORE:
            path = CACHE / f"score_{video}.json"
            if path.exists():
                data = json.loads(path.read_text())
                rows += data["synthetic"] + data["real"]
        table = summarise(rows)
        write_json(BASELINE / "fill-quality.json", table)
        keys = list(next(iter(table.values())))
        print(f"{'variant':20s}" + "".join(f"{k:>11s}" for k in keys))
        for variant, entry in table.items():
            print(f"{variant:20s}" + "".join(f"{entry[k]:11.3f}" for k in keys))


if __name__ == "__main__":
    main()
