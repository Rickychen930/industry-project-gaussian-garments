"""Fit SMPL-X to every frame of the authors' trajectory for each garment.

`tools/train/shading_student.py` samples its training poses from
`docs/baseline/trajectory-fits-<name>.npz`; until now only the garments
evaluated with `tools/evaluate/cloth_trajectory.py` had a full one
(the others held the single scan frame). This fills them all, with the
same fitter and cache format (`tools.evaluate.cloth_trajectory.fit_frames`,
cold 500 / warm 150 `fit_mesh` iterations); frames already cached are
kept.

    .venv/bin/python -m tools.data.fit_trajectories [00122_outer,00136_outer ...] [--stride 1]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from tryon.paths import BASELINE, GARMENTS


def main():
    from tools.evaluate.cloth_trajectory import fit_frames
    from tools.data.fit_scan_pose import trajectory_for

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("garments", nargs="?", default=None, help="comma-separated names (default: every checkpoint)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warm-iterations", type=int, default=150)
    args = parser.parse_args()
    names = args.garments.split(",") if args.garments else sorted(p.name for p in GARMENTS.iterdir() if (p / "stage1").is_dir())
    for name in names:
        trajectory = trajectory_for(GARMENTS / name)
        if trajectory is None:
            print(f"{name}: no trajectory, skipped", flush=True)
            continue
        with open(trajectory, "rb") as handle:
            data = pickle.load(handle)
        frames = list(range(0, len(data["obstacle"]), args.stride))
        cache = BASELINE / f"trajectory-fits-{name}.npz"
        fits = fit_frames(data, frames, cache, args.iterations, args.warm_iterations, verbose=False)
        residual = sum(float(fits[k]["residual_m"]) for k in frames) / len(frames)
        print(f"{name}: {len(frames)} frames fitted, mean residual {residual * 100:.2f} cm -> {cache.name}", flush=True)


if __name__ == "__main__":
    main()
