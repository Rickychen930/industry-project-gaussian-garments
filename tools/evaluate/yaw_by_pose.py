"""Signed yaw of the tracked body against the poses of a guided recording.

    .venv/bin/python -m tools.evaluate.yaw_by_pose docs/baseline/session-<name>-records.json

`session_benchmark --records` logs, for every frame, the yaw of the fitted body (`body_yaw`) and the yaw
MediaPipe's landmarks give (`mediapipe_yaw`), both in degrees and signed (0 facing the camera, 180 back to it).
`tools.record_session` asks for 12 poses in turn after a 5 s get-ready. The recording does not store when each
pose started, and cutting it into 12 equal slices put the left-side turn in the wrong slice, so the yaw trace is
summarised as a whole: its signed extremes, when they happen, and the stretches with no yaw. A real turn has no
ground truth; the trace is read by hand against the prompt order (left side, back, right side).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tools.record_session import FPS
from tryon.paths import BASELINE

OUTPUT = BASELINE / "yaw-by-pose.json"


def wrap(angle: float) -> float:
    """Degrees wrapped into [-180, 180)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def agreement(records: list[dict]) -> dict:
    """How far the fitted body's yaw is from MediaPipe's, over the frames that have both."""
    gaps = [abs(wrap(r["body_yaw"] - r["mediapipe_yaw"])) for r in records
            if r.get("body_yaw") is not None and r.get("mediapipe_yaw") is not None]
    if not gaps:
        return {"frames": 0}
    return {"frames": len(gaps), "median_gap_degrees": float(np.median(gaps)),
            "p90_gap_degrees": float(np.percentile(gaps, 90)),
            "share_over_90_degrees": float(np.mean(np.array(gaps) > 90.0))}


def missing_runs(records: list[dict], key: str = "body_yaw", fps: float = FPS, minimum_seconds: float = 1.0) -> list[dict]:
    """Stretches of at least `minimum_seconds` in which the frame has no `key` yaw."""
    runs, start = [], None
    for index, record in enumerate(list(records) + [{key: 0.0}]):
        missing = record.get(key) is None
        if missing and start is None:
            start = index
        if not missing and start is not None:
            if (index - start) / fps >= minimum_seconds:
                runs.append({"from_seconds": round(start / fps, 2), "to_seconds": round((index - 1) / fps, 2)})
            start = None
    return runs


def trace(records: list[dict], key: str = "body_yaw", fps: float = FPS) -> dict:
    """Signed extremes of a yaw trace and when they happened, plus the frames near a back view (over 135 degrees)."""
    values = [(i, wrap(r[key])) for i, r in enumerate(records) if r.get(key) is not None]
    if not values:
        return {"frames_with_yaw": 0}
    lowest = min(values, key=lambda v: v[1])
    highest = max(values, key=lambda v: v[1])
    return {"frames_with_yaw": len(values),
            "lowest_degrees": lowest[1], "lowest_at_seconds": round(lowest[0] / fps, 2),
            "highest_degrees": highest[1], "highest_at_seconds": round(highest[0] / fps, 2),
            "frames_over_135_degrees": sum(1 for _, v in values if abs(v) > 135.0)}


def summarise(records: list[dict]) -> dict:
    return {"assumption": "none about pose timing: the recording stores no pose start times, so the yaw trace is "
                          "summarised as a whole (extremes, gaps) and read against the prompt order by hand",
            "frames": len(records),
            "agreement": agreement(records),
            "body_yaw": {**trace(records, "body_yaw"), "missing_runs": missing_runs(records, "body_yaw")},
            "mediapipe_yaw": {**trace(records, "mediapipe_yaw"), "missing_runs": missing_runs(records, "mediapipe_yaw")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", help="per-frame records from session_benchmark --records")
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text())
    summary = summarise(records)
    write_json(OUTPUT, summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
