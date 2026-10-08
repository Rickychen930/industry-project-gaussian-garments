"""Does the garment follow a wearer of another build, and by how much? (docs/baseline/size-ablation.json)

Binds a garment to bodies whose first two shape coefficients are moved by +-2 (SMPL-X betas are in
standard deviations) from the neutral body, same (zero) pose, under each `ScanPoseBinder.SIZE_MODES`,
and reports the garment's torso width at chest height (sleeve vertices left out, zero-pose rest)
over the body's chest width. The garment should widen with a wider body, not by more than the body.

    .venv/bin/python -m tools.evaluate.size_generality [garment] [--out j.json]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from tools.evaluate._common import write_json
from tryon.paths import GARMENTS

CHEST_JOINT = 9  # spine3
SLAB_M = 0.02
SHAPES = {"neutral": (0, 0), "b0+2": (2, 0), "b0-2": (-2, 0), "b1+2": (0, 2), "b1-2": (0, -2), "both+2": (2, 2), "both-2": (-2, -2)}
MODES = ("current", "no_chest", "widen", "none", "shrink", "damped", "height_y")


def chest_width(fitter):
    with torch.no_grad():
        saved = fitter.body_pose.clone(), fitter.global_orient.clone()
        fitter.body_pose.zero_()
        fitter.global_orient.zero_()
        out = fitter._forward()
        fitter.body_pose[:], fitter.global_orient[:] = saved
    v = out.vertices[0].numpy().astype(np.float64)
    y = float(out.joints[0, CHEST_JOINT, 1])
    from tryon.garment.skinning import SkinnedGarment

    arm = fitter.model.lbs_weights.detach().numpy()[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
    band = (~arm) & (np.abs(v[:, 1] - y) < SLAB_M)
    return float(np.ptp(v[band, 0])), y, float(v[:, 1].max() - v[:, 1].min())


def garment_width(renderer, y):
    rest = renderer._binding.rest
    keep = ~renderer.sleeve_vertices & (np.abs(rest[:, 1] - y) < SLAB_M)
    return float(np.ptp(rest[keep, 0])) if keep.any() else float("nan")


def measure(garment="00152_outer"):
    from tryon.body.fitter import BodyFitter
    from tryon.garment.renderer import GarmentRenderer

    renderer = GarmentRenderer(GARMENTS / garment, dynamics=False)
    result = {}
    for mode in MODES:
        renderer.SIZE_MODE = mode
        rows = {}
        for name, (b0, b1) in SHAPES.items():
            fitter = BodyFitter()
            with torch.no_grad():
                fitter.betas.zero_()
                fitter.betas[0, 0], fitter.betas[0, 1] = float(b0), float(b1)
            renderer.bind(fitter)
            body_w, y, height = chest_width(fitter)
            garment_w = garment_width(renderer, y)
            rows[name] = {"body_width_m": body_w, "body_height_m": height, "garment_width_m": garment_w,
                          "garment_over_body": garment_w / body_w, "size_ratio": renderer._size_ratio, "width_ratio": renderer._width_ratio}
        base = rows["neutral"]
        for name, row in rows.items():
            row["body_width_vs_neutral"] = row["body_width_m"] / base["body_width_m"]
            row["garment_width_vs_neutral"] = row["garment_width_m"] / base["garment_width_m"]
            row["elasticity"] = float(np.log(row["garment_width_vs_neutral"]) / np.log(row["body_width_vs_neutral"])) if abs(row["body_width_vs_neutral"] - 1) > 1e-3 else None
        result[mode] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("garment", nargs="?", default="00152_outer")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = measure(args.garment)
    for mode, rows in result.items():
        print(mode)
        for name, r in rows.items():
            print(f"  {name:8} body {r['body_width_vs_neutral']:.3f}x garment {r['garment_width_vs_neutral']:.3f}x (garment/body {r['garment_over_body']:.3f}) height ratio {r['size_ratio']:.3f}")
    if args.out:
        write_json(args.out, {"garment": args.garment, "modes": result})


if __name__ == "__main__":
    main()
