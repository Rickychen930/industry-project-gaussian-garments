"""Recover the scan-frame SMPL-X pose of a Gaussian-Garments checkpoint
from the authors' trajectory file and cache it next to the checkpoint.

    .venv/bin/python -m tools.data.fit_scan_pose ~/tryon-data/garments/00152_outer

The trajectory files (README link, `individual-garments/<name>_Take<N>.pkl`)
hold `obstacle` (frames, 10475, 3): the SMPL-X body per frame, and
`pred` (frames, V, 3): the simulated garment; frame 0 is the template
pose (pred[0] equals stage1/template_uv.obj to 0.06 cm median). Fitting
SMPL-X to obstacle[0] gives the pose the garment was scanned in, which
`scan_pose.ScanPoseBinder.bind` otherwise has to guess.
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

from tryon import paths

TRAJECTORIES = paths.TRAJECTORIES / "individual-garments"
CACHE_NAME = "scan_pose.npz"
# Median distance (m) from the trajectory's first garment frame to the
# checkpoint's template above which `obstacle[0]` is NOT the template's body:
# 0.0-0.7 mm for 11 garments, 33-218 mm for the four that need refining.
MISMATCH_LIMIT = 0.01


def trajectory_for(checkpoint_dir):
    matches = sorted(glob.glob(str(TRAJECTORIES / f"{Path(checkpoint_dir).name}_Take*.pkl")))
    return Path(matches[0]) if matches else None


def template_mismatch(checkpoint_dir, pred0):
    """Median distance (m) from the vertices of the trajectory's first frame
    `pred0` to the nearest vertex of the checkpoint's template mesh."""
    from scipy.spatial import cKDTree

    from tryon.garment.checkpoint import load_template

    template = np.asarray(load_template(Path(checkpoint_dir) / "stage1" / "template_uv.obj").vertices, np.float64)
    return float(np.median(cKDTree(template).query(np.asarray(pred0, np.float64))[0]))


def scan_pose_path(checkpoint_dir):
    return Path(checkpoint_dir) / "stage1" / CACHE_NAME


def fit_scan_pose(checkpoint_dir, iterations=500, verbose=True):
    """Fit and cache; returns the dict saved (or the cached one)."""
    import torch

    from tryon.body.fitter import BodyFitter

    out = scan_pose_path(checkpoint_dir)
    if out.exists():
        return dict(np.load(out))
    trajectory = trajectory_for(checkpoint_dir)
    if trajectory is None:
        return None
    with open(trajectory, "rb") as handle:
        data = pickle.load(handle)
    body = np.asarray(data["obstacle"][0], np.float64)
    template = np.asarray(data["pred"][0], np.float64)
    fitter = BodyFitter()
    residual = fitter.fit_mesh(body, iterations=iterations)
    with torch.no_grad():
        fitted = fitter._forward().vertices[0].numpy()  # _forward applies mesh_offset
        joints = fitter._forward().joints[0].numpy()
    result = {
        "betas": fitter.betas.detach().numpy().ravel(),
        "body_pose": fitter.body_pose.detach().numpy().ravel(),
        "global_orient": fitter.global_orient.detach().numpy().ravel(),
        "offset": fitter.mesh_offset.detach().numpy().ravel(),
        "residual_m": np.float64(residual),
        "trajectory": str(trajectory),
        "template_mismatch_m": np.float64(template_mismatch(checkpoint_dir, template)),
    }
    np.savez(out, **result)
    if verbose:
        per_vertex = np.linalg.norm(fitted - body, axis=1)
        print(f"{trajectory.name}: residual mean {residual * 100:.2f} cm, max {per_vertex.max() * 100:.2f} cm")
        for name, sh, el, wr in (("L", 16, 18, 20), ("R", 17, 19, 21)):
            upper = joints[el] - joints[sh]
            fore = joints[wr] - joints[el]
            def ang(v):
                return np.degrees(np.arccos(-v[1] / np.linalg.norm(v)))
            print(f"  {name} upper arm {ang(upper):.0f} deg from vertical, forearm {ang(fore):.0f} deg; z tilt upper {np.degrees(np.arcsin(upper[2] / np.linalg.norm(upper))):+.0f} deg")
        print(f"  template garment bbox y {template[:, 1].min():.2f}..{template[:, 1].max():.2f}, body y {body[:, 1].min():.2f}..{body[:, 1].max():.2f}")
        if result["template_mismatch_m"] > MISMATCH_LIMIT:
            print(f"  WARNING: the trajectory's first frame is {result['template_mismatch_m'] * 100:.1f} cm (median) from this template, so this pose is for a different body; "
                  f"run: python -m tools.data.refine_scan_pose {checkpoint_dir}  (or fit_scan_pose --refine)")
        print(f"  saved {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refine", action="store_true", help="after fitting, refine the pose against the template when the trajectory's first frame is not it (MISMATCH_LIMIT)")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        if args.force and scan_pose_path(checkpoint).exists():
            scan_pose_path(checkpoint).unlink()
        result = fit_scan_pose(checkpoint, args.iterations)
        if result is None:
            print(f"{checkpoint}: no trajectory file in {TRAJECTORIES}")
        elif args.refine and "refined" not in result and float(result.get("template_mismatch_m", 0.0)) > MISMATCH_LIMIT:
            from tools.data.refine_scan_pose import refine_and_write

            print(f"{checkpoint}: template is {float(result['template_mismatch_m']) * 100:.1f} cm from the trajectory's first frame; refining")
            refine_and_write(checkpoint)


if __name__ == "__main__":
    main()
