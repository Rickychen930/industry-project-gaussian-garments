"""Why does a loose shirt's hem hang below the real one? Measured per frame on recorded bodies.

    .venv/bin/python -m tools.evaluate.hem_study diagnose 00152_outer --cache DIR [--set Class.ATTR=value ...] [--out j.json]

`diagnose` replays the bodies recorded by `open_front record` and, per scored frame, takes
  - the hem offset: lowest row of our garment minus lowest row of the real shirt (SCHP), over the real
    shirt's height (`placement_gap.decompose`), and the torso width ratio;
  - the pose: how far the arms are raised (`GarmentRenderer.arm_raise`) and how far a thigh is lifted (angle
    between the hip-to-knee direction and straight down along the spine);
  - the garment's hem in 3D (the lowest 15 % of the body vertices at rest): its height below the pelvis along the
    spine, posed and carried by skinning only, over the garment height (rest value subtracted), and the largest
    ratio of the straight distance from a hem vertex to the top ring over its rest geodesic (> 1 is what a
    long-range attachment would act on);
and prints the correlations of the hem offset with each, and the hem vertices' skinning weight shares.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tools.evaluate.open_front import CACHE, RecordedFitter, load_record, replay

HEM_FRACTION = 0.15
TOP_FRACTION = 0.03
THIGH_JOINTS = ((1, 4), (2, 5))  # hip, knee of each leg
SPINE_JOINTS = (0, 12)  # pelvis, neck


def hem_vertices(rest, body_mask, fraction=HEM_FRACTION):
    """Indices of the lowest `fraction` of the non-sleeve vertices (by rest height)."""
    ids = np.flatnonzero(body_mask)
    cut = np.quantile(rest[ids, 1], fraction)
    return ids[rest[ids, 1] <= cut]


def top_ring(rest, body_mask, fraction=TOP_FRACTION):
    ids = np.flatnonzero(body_mask)
    cut = np.quantile(rest[ids, 1], 1.0 - fraction)
    return ids[rest[ids, 1] >= cut]


def thigh_lift(joints):
    """Largest angle (degrees) between a hip-to-knee direction and straight down the spine."""
    up = joints[SPINE_JOINTS[1]] - joints[SPINE_JOINTS[0]]
    up = up / max(np.linalg.norm(up), 1e-9)
    angles = []
    for hip, knee in THIGH_JOINTS:
        leg = joints[knee] - joints[hip]
        leg = leg / max(np.linalg.norm(leg), 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(-leg @ up, -1.0, 1.0))))
    return float(max(angles))


def pose_features(fitter, renderer):
    joints = fitter.joints(indices=list(range(22))).detach().numpy().astype(np.float64)
    raised, up = renderer.arm_raise(fitter)
    return joints, raised, up


def diagnose(name, cache=CACHE, every=5, assignments=()):
    from tryon.body.fitter import BodyFitter
    from tryon.garment.cloth import geodesic_from
    from tryon.garment.renderer import GarmentRenderer
    from tryon.garment.skinning import SkinnedGarment
    from tryon.paths import GARMENTS

    meshes = {}
    result = replay(name, every, False, assignments, cache, keep_meshes=meshes)
    data = load_record(name, cache)
    renderer = GarmentRenderer(GARMENTS / name)
    fitter = RecordedFitter(BodyFitter().faces)
    fitter.load(data, 0)
    renderer.bind(fitter)
    binding = renderer._binding
    rest = binding.physical_rest
    body_mask = ~np.asarray(renderer.sleeve_vertices, bool)
    hem, ring = hem_vertices(rest, body_mask), top_ring(rest, body_mask)
    height = float(np.ptp(rest[body_mask, 1]))
    geodesic, anchor_of = geodesic_from(len(rest), renderer.scan_binder.edges, rest, ring)
    weights = binding.weights
    share = {
        "leg": float(weights[hem][:, [1, 2, 4, 5, 7, 8, 10, 11]].sum(axis=1).mean()),
        "pelvis": float(weights[hem][:, 0].mean()),
        "spine": float(weights[hem][:, [3, 6, 9]].sum(axis=1).mean()),
        "arm": float(weights[hem][:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1).mean()),
    }
    scored = np.flatnonzero(data["kind"])
    index = {int(t): int(i) for t, i in zip(data["t"], scored)}
    rows = []

    def drop(vertices, pelvis, up):
        """How far the hem hangs below the pelvis along the spine, over the garment's height."""
        return float(((pelvis - vertices[hem]) @ up).mean()) / height

    for row in result["frames"]:
        frame = row["frame"]
        fitter.load(data, index[frame])
        joints, raised, up = pose_features(fitter, renderer)
        mesh, _body, skin = meshes[frame]
        pelvis = joints[0]
        straight = np.linalg.norm(skin[hem] - skin[anchor_of[hem]], axis=1)
        ratio = float((straight / np.maximum(geodesic[hem], 1e-9)).max())
        straight_cloth = np.linalg.norm(mesh[hem] - mesh[anchor_of[hem]], axis=1)
        rows.append({"frame": frame, "hem_offset": row["split_hem_offset"], "torso_width_ratio": row["split_torso_width_ratio"],
                     "iou": row["iou"], "arm_raise": raised, "thigh_lift_deg": thigh_lift(joints),
                     "hem_drop_skin": drop(skin, pelvis, up), "hem_drop_cloth": drop(mesh, pelvis, up), "lra_ratio_max": ratio,
                     "lra_ratio_p95": float(np.percentile(straight / np.maximum(geodesic[hem], 1e-9), 95)),
                     "lra_ratio_cloth_max": float((straight_cloth / np.maximum(geodesic[hem], 1e-9)).max())})
    keys = ("arm_raise", "thigh_lift_deg", "hem_drop_skin", "hem_drop_cloth", "lra_ratio_max")
    corr = {k: float(np.corrcoef([r["hem_offset"] for r in rows], [r[k] for r in rows])[0, 1]) for k in keys}
    summary = {"hem_offset_mean": float(np.mean([r["hem_offset"] for r in rows])), "weights_hem": share, "corr_hem_offset": corr,
               "lra_ratio_max_over_clip": float(max(r["lra_ratio_max"] for r in rows)),
               "iou": result["summary"]["iou"]}
    return {"video": name, "frames": rows, "summary": summary}


CANDIDATE_KEYS = ("split_hem_offset", "split_torso_width_ratio", "iou", "split_extra", "split_missed", "inside_share", "cloth_step_ms")


def candidates(name, variants, cache=CACHE, out=None):
    """Replay `name` under each variant ({tag: [assignments]}); one row per variant of `CANDIDATE_KEYS`
    (means over the scored frames). `out`: a json path for the rows and the per-frame results."""
    rows, frames = {}, {}
    for tag, assignments in variants.items():
        result = replay(name, 5, False, assignments, cache)
        rows[tag] = {k: result["summary"].get(k, float("nan")) for k in CANDIDATE_KEYS}
        frames[tag] = result["frames"]
        print(name, tag, {k: round(v, 4) for k, v in rows[tag].items()}, flush=True)
    if out:
        write_json(out, {"video": name, "summary": rows, "frames": frames})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["diagnose", "candidates"])
    parser.add_argument("video")
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--set", dest="assignments", action="append", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--variant", action="append", default=[], help="candidates: tag=Class.ATTR=value;Class.ATTR=value (empty after = for the defaults)")
    args = parser.parse_args()
    if args.command == "candidates":
        variants = {}
        for item in args.variant:
            tag, _, rest = item.partition("=")
            variants[tag] = [a for a in rest.split(";") if a]
        candidates(args.video, variants, Path(args.cache), args.out)
        return
    result = diagnose(args.video, Path(args.cache), args.every, args.assignments)
    print(args.video, result["summary"], flush=True)
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
