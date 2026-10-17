"""Does the garment turn with the wearer? Widths against the angle to the camera, on recorded sessions.

Reads the per-frame records that `session_benchmark --records` writes. For every frame where the wearer's
angle to the camera is known (`profile_angle`: 0 facing the camera or away, 90 side-on) it takes the width
in pixels of the person mask and of the drawn garment, groups the frames in angle bins and reports the
median widths and their ratio. A garment that turns with the wearer is narrower side-on than head-on, like
the wearer, and keeps a similar width ratio in every bin.

    python -m tools.evaluate.turn_profile records1.json records2.json --out docs/baseline/turn-profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tryon import paths

BINS = ((0, 20, "head-on (0 to 20 degrees)"), (20, 70, "oblique (20 to 70)"), (70, 91, "side-on (70 to 90)"))
OUTPUT = paths.BASELINE / "turn-profile.json"
MIN_FRAMES = 20


def summarise(records: list[dict]) -> dict:
    usable = [r for r in records if r.get("dressed") and r.get("profile_angle") is not None
              and r.get("garment_width") and r.get("person_width")]
    rows = []
    for low, high, name in BINS:
        sel = [r for r in usable if low <= r["profile_angle"] < high]
        if len(sel) < MIN_FRAMES:
            rows.append({"bin": name, "frames": len(sel)})
            continue
        garment = float(np.median([r["garment_width"] for r in sel]))
        person = float(np.median([r["person_width"] for r in sel]))
        rows.append({"bin": name, "frames": len(sel), "garment_width_px": garment, "person_width_px": person,
                     "ratio": garment / person})
    angles = [r["profile_angle"] for r in usable]
    return {"frames": len(records), "usable_frames": len(usable),
            "angle_range_degrees": [float(min(angles)), float(max(angles))] if angles else None, "bins": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=Path, help="per-frame record files from session_benchmark")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    records = []
    for path in args.records:
        with open(path) as handle:
            records += json.load(handle)
    result = summarise(records)
    result["sources"] = [p.name for p in args.records]
    write_json(args.out, result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
