"""One recorded webcam session through the live pipeline, with numbers: a fixed benchmark.

    .venv/bin/python -m tools.evaluate.session_benchmark ~/tryon-data/captures/session-....mp4 --garment 00190_upper --out docs/baseline/session-<name>.json

A recording has no ground truth, so it measures what can be checked without one:
frames dressed, fitted body vs MediaPipe keypoints, shoulder width facing and
away from the camera, how much of the wearer's own clothing stays uncovered, and
per-frame time. Run with nothing else busy: stage times are the machine's too.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json

FRONT_BACK_DEGREES = 35.0
SHOULDER_BAND = (0.74, 1.35)


def _plain(value):
    """JSON form of the numpy scalars in the per-frame records."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def percentiles(values, points=(50, 90, 95, 99)):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if values.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(values, p)), 2) for p in points}


def stage_medians(text):
    """{stage: median ms} from the pipeline's `--profile` line in `text`."""
    for line in text.splitlines():
        if line.startswith("median ms a frame:"):
            # "mediapipe 14  erase start 3  render 78": the names can hold a space
            return {name.strip(): float(value) for name, value in re.findall(r"([a-z][a-z ]*?)\s+(\d+(?:\.\d+)?)(?=\s|$)", line.split(":", 1)[1])}
    return {}


def run_session(source, garment, max_frames=0, max_width=960):
    """Per-frame records, the stage medians and the seconds of one run of the live pipeline over `source`."""
    import torch

    import tryon.app.pipeline as pipeline
    from tryon.body.landmarks import landmark_yaw
    from tryon.garment.renderer import GarmentRenderer
    from tryon.paths import GARMENTS
    from tryon.removal.remover import ClothingRemover

    GarmentRenderer.GPU_RASTER = True
    GarmentRenderer.APPEARANCE_ASYNC = True
    GarmentRenderer.STUDENT_SHADING = True
    current = {}
    records = []

    original_remove = ClothingRemover.remove

    def remove(self, frame, mask=None, *args, **kwargs):
        current["mask"] = None if mask is None else mask.copy()
        return original_remove(self, frame, mask, *args, **kwargs)

    original_composite = pipeline.composite_garment

    def composite(frame, rgba):
        mask = current.get("mask")
        alpha = rgba[..., 3] >= 25
        current["alpha_area"] = int(alpha.sum())
        columns = np.flatnonzero(alpha.any(axis=0))
        current["garment_width"] = int(columns[-1] - columns[0] + 1) if columns.size else 0
        current["uncovered"] = None if mask is None or mask.shape != alpha.shape or not mask.any() else float((mask & ~alpha).sum() / mask.sum())
        original_composite(frame, rgba)

    original_process = pipeline.LiveSession.process

    def process(self, frame, points, stamp, stage, untouched=None):
        current.clear()
        out = original_process(self, frame, points, stamp, stage, untouched)
        status, reprojection, alpha = out
        ratio = facing = profile_angle = None
        if self.fitter is not None and points is not None and min(points[11, 2], points[12, 2]) > 0.5:
            with torch.no_grad():
                pixels = self.fitter._project(self.fitter._forward().joints[0, 16:18]).numpy()
            wanted = float(np.linalg.norm(points[11, :2] - points[12, :2]))
            yaw = landmark_yaw(points)
            if wanted > 4 and yaw is not None:
                away = abs((np.degrees(yaw) + 180.0) % 360.0 - 180.0)
                profile_angle = min(away, 180.0 - away)  # 0 facing the camera or away, 90 side-on
                facing = profile_angle <= FRONT_BACK_DEGREES
                ratio = float(np.linalg.norm(pixels[0] - pixels[1])) / wanted
        mask = current.get("mask")
        person_columns = np.flatnonzero(mask.any(axis=0)) if mask is not None and mask.any() else np.zeros(0)
        records.append(dict(status=status, reprojection=None if reprojection != reprojection else float(reprojection),
                            dressed=alpha is not None, garment_pixels=current.get("alpha_area"), uncovered=current.get("uncovered"),
                            shoulder_ratio=ratio, front_or_back=facing, profile_angle=profile_angle,
                            garment_width=current.get("garment_width"),
                            person_width=int(person_columns[-1] - person_columns[0] + 1) if person_columns.size else None))
        return out

    ClothingRemover.remove = remove
    pipeline.composite_garment = composite
    pipeline.LiveSession.process = process
    started = time.perf_counter()
    text = io.StringIO()
    try:
        with redirect_stdout(text):
            pipeline.run(str(source), max_frames=max_frames, headless=True, garment=str(GARMENTS / garment),
                         orientation_regressor="models/pose_regressor.pt", remove_clothing=True, appearance=True,
                         max_width=max_width, profile=True)
    finally:
        ClothingRemover.remove = original_remove
        pipeline.composite_garment = original_composite
        pipeline.LiveSession.process = original_process
    return records, stage_medians(text.getvalue()), time.perf_counter() - started


def summarise(records, stages, seconds):
    total = len(records)
    statuses = collections.Counter(r["status"] for r in records)
    ratios = [r["shoulder_ratio"] for r in records if r["front_or_back"]]
    outside = [x for x in ratios if x < SHOULDER_BAND[0] or x > SHOULDER_BAND[1]]
    pixels = [r["garment_pixels"] for r in records if r["garment_pixels"]]
    return {
        "frames": total,
        "dressed_share": round(sum(r["dressed"] for r in records) / max(total, 1), 4),
        "status_counts": dict(statuses),
        "reprojection_px": percentiles([r["reprojection"] for r in records]),
        "front_or_back_frames": len(ratios),
        "shoulder_ratio_outside_band": len(outside),
        "shoulder_band": list(SHOULDER_BAND),
        "uncovered_clothing_share": percentiles([r["uncovered"] for r in records if r["dressed"]], (50, 90)),
        "garment_pixels_median": int(np.median(pixels)) if pixels else 0,
        "stage_median_ms": stages,
        "wall_seconds": round(seconds, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a recorded session (.mp4)")
    parser.add_argument("--garment", default="00190_upper")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--out", default=None, help="write the summary here (JSON)")
    parser.add_argument("--records", default=None, help="also write the per-frame records here (JSON)")
    args = parser.parse_args()
    records, stages, seconds = run_session(args.source, args.garment, args.max_frames, args.max_width)
    summary = summarise(records, stages, seconds)
    summary["source"] = Path(args.source).name
    summary["garment"] = args.garment
    print(json.dumps(summary, indent=2))
    if args.out:
        write_json(args.out, summary)
    if args.records:
        write_json(args.records, json.loads(json.dumps(records, default=_plain)))


if __name__ == "__main__":
    main()
