"""Refine a checkpoint's scan pose against its OWN template garment.

`tools/data/fit_scan_pose.py` fits SMPL-X to the trajectory file's
`obstacle[0]`, which is right when the trajectory's `pred[0]` IS the
checkpoint's template (00152_outer: 0.07 cm). On four checkpoints it is
not (00122_outer 9, 00148_upper 32, 00176_outer 10, 10004_outer 12 cm
mean, non-rigid), so that body sits in a different drape and the binding
is off before any tracking (docs/LOG.md, "All 12 upper-body
garments"). This starts from that pose and moves it until the template
garment sits around the body:

  ICP, one-sided: every template vertex is paired with its nearest body
  vertex (re-paired every `repair` steps) and the squared distance is
  minimised, truncated at `truncate` metres so loose fabric (a hem, an
  open flap) does not drag the body; pairs whose garment vertex lies
  INSIDE the body (behind the body vertex's normal) cost 4x, since a
  garment cannot be inside its wearer. Shape stays (the scan subject's
  betas); pose, root orientation and offset move, with a prior toward the
  starting pose.

The result is written as `scan_pose.npz` (the trajectory fit is kept as
`scan_pose_trajectory.npz`), with `refined=1` and the before/after
garment-to-body distances.

    .venv/bin/python -m tools.data.refine_scan_pose ~/tryon-data/garments/00122_outer [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch



def refine(checkpoint, iterations=300, learning_rate=0.01, truncate=0.10, repair=10, prior=50.0, shape=False, shape_prior=1.0, coarse=0, coarse_truncate=0.5, verbose=True):
    from scipy.spatial import cKDTree

    from tryon.garment.checkpoint import load_template
    from tryon.garment.skinning import _vertex_normals
    from tryon.body.fitter import BodyFitter

    checkpoint = Path(checkpoint)
    source = checkpoint / "stage1" / "scan_pose_trajectory.npz"
    if not source.exists():
        source = checkpoint / "stage1" / "scan_pose.npz"
    scan = dict(np.load(source))
    garment = torch.tensor(np.asarray(load_template(checkpoint / "stage1" / "template_uv.obj").vertices, np.float32))

    fitter = BodyFitter()
    with torch.no_grad():
        fitter.betas[:] = torch.tensor(scan["betas"], dtype=torch.float32).reshape(1, -1)
        fitter.body_pose[:] = torch.tensor(scan["body_pose"], dtype=torch.float32).reshape(1, -1)
        fitter.global_orient[:] = torch.tensor(scan["global_orient"], dtype=torch.float32).reshape(1, -1)
    offset = torch.tensor(np.asarray(scan["offset"], np.float32).reshape(1, 3), requires_grad=True)
    start_pose = fitter.body_pose.detach().clone()
    start_orient = fitter.global_orient.detach().clone()
    faces = np.asarray(fitter.faces)

    def distances(body_np):
        d, index = cKDTree(body_np).query(garment.numpy())
        return d, index

    with torch.no_grad():
        body0 = (fitter._forward().vertices[0] + offset).numpy()
    before, _ = distances(body0)
    start_betas = fitter.betas.detach().clone()
    parameters = [fitter.body_pose, fitter.global_orient, offset] + ([fitter.betas] if shape else [])
    for p in parameters:
        p.requires_grad_(True)
    optimiser = torch.optim.Adam(parameters, learning_rate)
    index = normals = None
    for step in range(iterations):
        body = fitter._forward().vertices[0] + offset
        if step % repair == 0:
            body_np = body.detach().numpy().astype(np.float64)
            _d, index = distances(body_np)
            normals = torch.tensor(_vertex_normals(body_np, faces)[index], dtype=torch.float32)
        paired = body[index]
        delta = garment - paired
        distance = delta.norm(dim=1)
        inside = (delta * normals).sum(dim=1) < 0.0
        weight = torch.where(inside, torch.full_like(distance, 4.0), torch.ones_like(distance))
        # coarse-to-fine: a wide cut-off for the first `coarse` steps, so a
        # garment that starts far from the body (00148_upper: 27 cm median)
        # is pulled in before loose fabric is excluded
        keep = distance < (coarse_truncate if step < coarse else truncate)
        data = (weight * distance**2)[keep].mean()
        reg = prior * (((fitter.body_pose - start_pose) ** 2).mean() + ((fitter.global_orient - start_orient) ** 2).mean())
        if shape:
            reg = reg + shape_prior * ((fitter.betas - start_betas) ** 2).mean()
        loss = data + reg
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if verbose and step % 50 == 0:
            print(f"  step {step}: data {float(data) * 1e4:.2f} cm^2, inside {float(inside.float().mean()):.1%}", flush=True)
    with torch.no_grad():
        body1 = (fitter._forward().vertices[0] + offset).numpy()
    after, _ = distances(body1)
    result = {
        "betas": fitter.betas.detach().numpy(),
        "body_pose": fitter.body_pose.detach().numpy(),
        "global_orient": fitter.global_orient.detach().numpy(),
        "offset": offset.detach().numpy().reshape(3),
        "residual_m": np.asarray(scan.get("residual_m", np.nan)),
        "trajectory": scan.get("trajectory", np.asarray("")),
        "refined": np.asarray(1),
        "garment_to_body_before_m": np.asarray(np.median(before)),
        "garment_to_body_after_m": np.asarray(np.median(after)),
    }
    return result, before, after


def refine_and_write(checkpoint, iterations=500, coarse=200, **options):
    """Refine with the command line's defaults (300 fine + 200 coarse steps) and
    replace stage1/scan_pose.npz (the trajectory fit is kept as
    scan_pose_trajectory.npz); returns (before, after) distances."""
    result, before, after = refine(checkpoint, iterations=iterations, coarse=coarse, **options)
    stage1 = Path(checkpoint) / "stage1"
    if not (stage1 / "scan_pose_trajectory.npz").exists():
        shutil.copy(stage1 / "scan_pose.npz", stage1 / "scan_pose_trajectory.npz")
    np.savez(stage1 / "scan_pose.npz", **result)
    print(f"wrote {stage1 / 'scan_pose.npz'} (trajectory fit kept as scan_pose_trajectory.npz)")
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shape", action="store_true", help="let the body shape move too (prior toward the trajectory's)")
    parser.add_argument("--coarse", type=int, default=200, help="steps at a 50 cm cut-off before the fine 10 cm one")
    args = parser.parse_args()
    result, before, after = refine(args.checkpoint, iterations=args.iterations + args.coarse, shape=args.shape, coarse=args.coarse)
    print(f"garment-to-body median {np.median(before) * 100:.2f} -> {np.median(after) * 100:.2f} cm, "
          f"p90 {np.quantile(before, 0.9) * 100:.2f} -> {np.quantile(after, 0.9) * 100:.2f} cm")
    if args.dry_run:
        return
    stage1 = Path(args.checkpoint) / "stage1"
    if not (stage1 / "scan_pose_trajectory.npz").exists():
        shutil.copy(stage1 / "scan_pose.npz", stage1 / "scan_pose_trajectory.npz")
    np.savez(stage1 / "scan_pose.npz", **result)
    print(f"wrote {stage1 / 'scan_pose.npz'} (trajectory fit kept as scan_pose_trajectory.npz)")


if __name__ == "__main__":
    main()
