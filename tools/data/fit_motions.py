"""Fit SMPL-X to the bodies of the authors' other simulations, as extra
training poses for the shading students: docs/baseline/motion-fits/
<source>-<outfit>-<motion>.npz, every STRIDE-th frame.

Each garment's own trajectory is one motion on one body. The authors'
other simulation sets add, with the SMPL-X body of every frame:

  mg  multigarment-outfits/  15 CMU motions, each on 4 body shapes
                             (1.67-1.75 m tall): one take per motion and
                             body shape
  rs  resize/                4 more motions on 10 bodies of other sizes
  ob  one-by-one-draping/    1 more motion (15_09)

`tools/train/shading_student.py --with-motions` trains on all of them but
the motions in `TEST_MOTIONS`, which `tools/evaluate/shading_students.py`
keeps unseen. Fitted with the trajectory fitter
(`cloth_trajectory.fit_frames`); files already fitted are kept.

    .venv/bin/python -m tools.data.fit_motions [--stride 17] [--sources mg,rs,ob]
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

from tryon.paths import BASELINE, TRAJECTORIES

MOTION_FITS = BASELINE / "motion-fits"
SOURCES = {"mg": "multigarment-outfits", "rs": "resize", "ob": "one-by-one-draping"}


def body_key(obstacle):
    """The body shape of a take: its first frame's height and width (cm)."""
    first = np.asarray(obstacle[0])
    return round(float(np.ptp(first[:, 1])) * 100), round(float(np.ptp(first[:, 0])) * 100)


def takes(source):
    """(outfit, motion, path) of one take per motion and body shape."""
    chosen, seen = [], set()
    for path in sorted(glob.glob(str(TRAJECTORIES / SOURCES[source] / "*" / "*.pkl"))):
        outfit, motion = Path(path).parent.name, Path(path).stem
        if source == "ob" and any(m == motion for _, m, _ in chosen):
            continue  # the same body and motion, with 1 to 5 garments
        with open(path, "rb") as handle:
            key = (motion, body_key(pickle.load(handle)["obstacle"]))
        if key not in seen:
            seen.add(key)
            chosen.append((outfit, motion, path))
    return chosen


def main():
    from tools.evaluate.cloth_trajectory import fit_frames

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=17)
    parser.add_argument("--sources", default="mg,rs,ob")
    args = parser.parse_args()
    MOTION_FITS.mkdir(parents=True, exist_ok=True)
    for source in args.sources.split(","):
        for outfit, motion, path in takes(source):
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            frames = list(range(0, len(data["obstacle"]), args.stride))
            fits = fit_frames(data, frames, MOTION_FITS / f"{source}-{outfit}-{motion}.npz", 500, 150, verbose=False)
            residual = 100 * np.mean([fits[k]["residual_m"] for k in frames])
            print(f"{source}-{outfit}-{motion}: {len(frames)} poses, residual {residual:.2f} cm", flush=True)


if __name__ == "__main__":
    main()
