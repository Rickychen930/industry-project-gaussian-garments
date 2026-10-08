"""How much of the erased ARM does the live garment leave uncovered? (the "arms look like sticks" question)

The fill draws a bare arm as a tapered tube (`arm_limb`) wherever the wearer's own sleeve is erased. A
viewer only sees the part of that tube the rendered garment does not cover. For each video the live
pipeline (`LiveSession`, the launcher's settings) runs over the first frames; at every sampled frame the
remover's call is recorded (frame, erase mask, arm strokes, hands, parse, plate, body) and the garment's
alpha is kept, so that

    tube      = arm_limb tube & erase mask & ~hands & the fill changed the pixel
    visible   = tube & (garment alpha <= 0.5)

and the share `visible / tube` is the table. The recorded samples are pickled (SCRATCH/arm_visibility)
for `tools.evaluate.arm_fill_study`, which tries learned / textured fills on the same holes.

    nice -n 15 .venv/bin/python -m tools.evaluate.arm_visibility collect 00176_upper     # one video per process
    .venv/bin/python -m tools.evaluate.arm_visibility report
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tryon.paths import BASELINE, GARMENTS, REPO_ROOT, SCRATCH

CACHE = Path(SCRATCH) / "arm_visibility"
CORE = ("00152_outer", "00122_outer", "00170_upper", "00176_upper", "10006_outer", "00190_upper")
FRAMES = 86
FIRST, EVERY = 26, 12  # sampled frames: the cloth has settled by FIRST
ALPHA_VISIBLE = 0.5
CHANGED = 6  # a tube pixel counts when the fill moved it by more than this (max channel)


def tube_pixels(sample, width_fraction):
    """The pixels of the drawn arm tube of `sample`: the tapered tube inside the erase mask, away from the hands."""
    from tryon.removal.remover import ClothingRemover, arm_limb

    points = sample["points"]
    if points is None:
        return np.zeros(sample["mask"].shape, bool)
    tube, _, _ = arm_limb(points, sample["mask"].shape, width_fraction, ClothingRemover.ARM_MUSCLE)
    changed = np.abs(sample["erased"].astype(np.int16) - sample["frame"].astype(np.int16)).max(axis=2) > CHANGED
    return tube & sample["mask"] & ~sample["keep"] & changed


def visibility(sample, width_fraction):
    """(tube pixels, visible tube pixels) of one sample."""
    tube = tube_pixels(sample, width_fraction)
    visible = tube & (sample["alpha"] <= ALPHA_VISIBLE * 255)
    return int(tube.sum()), int(visible.sum())


def collect(video):
    import mediapipe as mp

    from tools.evaluate import vs_registration as vr
    from tools.evaluate.reference_frame import find_panel
    from tryon.app import pipeline
    from tryon.app.person import landmarks_from
    from tryon.garment.renderer import GarmentRenderer
    from tryon.paths import RELEASE_VIDEOS
    from tryon.removal.remover import ClothingRemover

    GarmentRenderer.GPU_RASTER, GarmentRenderer.STUDENT_SHADING, GarmentRenderer.APPEARANCE_ASYNC = True, True, False
    session = pipeline.LiveSession(garment=str(GARMENTS / video), orientation_regressor=str(REPO_ROOT / "models" / "pose_regressor.pt"),
                                   remove_clothing=True, appearance=True, headless=True)
    remover, calls = session.remover, []
    real_remove = remover.remove

    def recording(frame, mask=None, **kwargs):
        out = real_remove(frame, mask, **kwargs)
        plate = kwargs.get("plate")
        calls.append({"frame": frame.copy(), "erased": out.copy(), "mask": np.asarray(mask).copy(), "kwargs": kwargs,
                      "plate": (plate.plate.copy(), plate.count.copy()) if plate is not None else None})
        return out

    remover.remove = recording
    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{video}.mp4"))
    box, state = [None], {"read": 0}

    def upcoming(pose):
        ok, full = capture.read()
        if not ok or state["read"] >= FRAMES:
            return None
        if box[0] is None:
            box[0] = find_panel(full, "left")
        y0, y1, x0, x1 = box[0]
        scale = vr.HEIGHT / (y1 - y0)
        panel = cv2.resize(full[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
        t = state["read"]
        state["read"] += 1
        session.prefetch(panel, points)
        return t, panel, points

    samples = []
    original_render, seen = session.renderer.render, {}

    def render(fitter, size, **kwargs):
        kwargs.setdefault("dt", 1.0 / 25.0)
        seen["rgba"] = original_render(fitter, size, **kwargs)
        return seen["rgba"]

    session.renderer.render = render
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=pipeline.POSE_COMPLEXITY) as pose:
        item = upcoming(pose)
        while item is not None:
            t, panel, points = item
            item = upcoming(pose)
            calls.clear()
            seen.clear()
            session.process(panel, points, t / 25.0, lambda *_a, **_k: None, panel.copy())
            if t < FIRST or (t - FIRST) % EVERY or not calls or "rgba" not in seen:
                continue
            call = calls[-1]
            kw = call["kwargs"]
            parse = kw.get("parse")
            samples.append({"t": t, "frame": call["frame"], "erased": call["erased"], "mask": call["mask"], "alpha": seen["rgba"][..., 3].copy(),
                            "points": None if parse is None or parse.points is None else np.asarray(parse.points).copy(),
                            "arms": kw.get("arms"), "keep": kw.get("keep"), "body": kw.get("body"), "plate": call["plate"],
                            "labels": parse.labels if parse is not None else None, "clothing": parse.clothing if parse is not None else None})
            if samples[-1]["points"] is None and points is not None:
                samples[-1]["points"] = np.asarray(points, np.float64)[:, :3].copy()
            print(video, "frame", t, flush=True)
    session.finish()
    if hasattr(session.regression, "close"):
        try:
            session.regression.close()
        except Exception:  # noqa: BLE001 - a worker still answering a prefetched frame nobody reads
            session.regression._process.kill()
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{video}.pkl", "wb") as handle:
        pickle.dump(samples, handle)
    row = summarise(video, samples, ClothingRemover.ARM_TUBE_WIDTH)
    write_json(CACHE / f"{video}.json", row)
    print(json.dumps(row))
    return row


def summarise(video, samples, width_fraction):
    counts = [visibility(s, width_fraction) for s in samples]
    shares = [v / t for t, v in counts if t > 0]
    return {"video": video, "frames": len(samples), "tube_px_mean": float(np.mean([t for t, _ in counts])) if counts else 0.0,
            "visible_px_mean": float(np.mean([v for _, v in counts])) if counts else 0.0,
            "visible_share_pooled": float(sum(v for _, v in counts) / max(sum(t for t, _ in counts), 1)),
            "visible_share_per_frame": [round(s, 3) for s in shares]}


def report():
    rows = [json.loads((CACHE / f"{v}.json").read_text()) for v in CORE if (CACHE / f"{v}.json").exists()]
    write_json(BASELINE / "arm-visibility.json", rows)
    print(f"{'video':14s}{'frames':>7s}{'tube px':>9s}{'visible px':>11s}{'visible %':>10s}  per frame")
    for r in rows:
        print(f"{r['video']:14s}{r['frames']:7d}{r['tube_px_mean']:9.0f}{r['visible_px_mean']:11.0f}{100 * r['visible_share_pooled']:10.1f}  {r['visible_share_per_frame']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "report"])
    parser.add_argument("video", nargs="?")
    args = parser.parse_args()
    collect(args.video) if args.command == "collect" else report()


if __name__ == "__main__":
    main()
