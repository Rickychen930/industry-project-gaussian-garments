"""The cloth against the authors' simulation: how close does our garment
posing get to ContourCraft's when our body IS the scan body?

    .venv/bin/python -m tools.evaluate.cloth_trajectory ~/tryon-data/garments/00152_outer

The trajectory file (`individual-garments/<name>_Take<N>.pkl`, see
tools/data/fit_scan_pose.py) carries `obstacle[k]` (the SMPL-X body, our
BodyFitter's own topology) and `pred[k]` (the garment ContourCraft
simulated on it) for 160 frames at 25 fps; pred[0] is the template. For
every `stride`-th frame:

1. SMPL-X is fitted to obstacle[k] (`BodyFitter.fit_mesh`, warm-started
   from the previous frame; all fits cached in one npz so reruns are
   instant), residual reported per frame.
2. `GarmentRenderer` is driven by a fitter set to those fits, rigid LBS
   (`dynamics=False`) and the live cloth (`dynamics=True`: `LightCloth`
   springs plus body collision, stepped once per fitted frame with
   dt = stride / 25 s; a step longer than `light_cloth.MAX_DT` restarts
   the springs from the skinned pose).
3. Per frame: mean / p90 vertex distance (cm) to pred[k], over all
   vertices, the sleeves (`renderer.sleeve_vertices`) and the rest. Two
   reference floors: pred[k] against the template carried rigidly by the
   TRUE body (`bind.GarmentBinding` on obstacle - nearest-body-vertex LBS
   with no fit in the loop), i.e. how far the reference cloth moves away
   from rigid following; and pred[k] against pred[0] unmoved.

The renderer's mesh is trimesh's UV-split order (8208 vertices); the
reference's is the obj's (7972). They index the same faces in the same
order (`AppearanceNet.__init__`), which gives the map used here.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tools.data.fit_scan_pose import scan_pose_path, trajectory_for
from tryon.paths import GARMENTS

REFERENCE_FPS = 25.0


def reference_vertex_map(renderer):
    """(8208,) -> index into the obj's 7972-vertex order."""
    from tryon.garment.texels import read_template

    template = read_template(renderer.checkpoint_dir / "stage1" / "template_uv.obj")
    reference_faces = np.asarray(template["faces"])
    to_ref = np.full(len(renderer.rest_vertices), -1, np.int64)
    to_ref[renderer.faces.ravel()] = reference_faces.ravel()
    if (to_ref < 0).any():
        raise ValueError("renderer mesh has vertices on no face; cannot map to the reference order")
    if not np.allclose(renderer.rest_vertices, template["vertices"][to_ref], atol=1e-5):
        raise ValueError("renderer rest mesh does not match the template through the face map")
    return to_ref, len(template["vertices"])


def to_reference_order(vertices, to_ref, count):
    """Average the UV-split duplicates back onto the reference vertex."""
    summed = np.zeros((count, 3))
    np.add.at(summed, to_ref, np.asarray(vertices, np.float64))
    return summed / np.bincount(to_ref, minlength=count)[:, None]


def fit_frames(trajectory, frames, cache, iterations, warm_iterations, seed=None, verbose=True):
    """Fit SMPL-X to obstacle[k] for every k in `frames`, cached in `cache`.
    Returns {k: {betas, body_pose, global_orient, offset, residual_m}}."""
    import torch

    from tryon.body.fitter import BodyFitter

    fits = {}
    if cache.exists():
        data = np.load(cache)
        for i, k in enumerate(data["frames"]):
            fits[int(k)] = {key: data[key][i] for key in ("betas", "body_pose", "global_orient", "offset", "residual_m")}
    missing = [k for k in frames if k not in fits]
    if missing:
        fitter = BodyFitter()
        warm = False
        if seed is not None:
            with torch.no_grad():
                fitter.betas[:] = torch.tensor(np.asarray(seed["betas"], np.float32))[None]
                fitter.body_pose[:] = torch.tensor(np.asarray(seed["body_pose"], np.float32))[None]
                fitter.global_orient[:] = torch.tensor(np.asarray(seed["global_orient"], np.float32))[None]
            warm = True
        obstacle = trajectory["obstacle"]
        for k in missing:
            # Warm start: the fitter keeps its parameters between calls, and
            # `fit_mesh` re-solves the offset from the centroids itself.
            nearest = [j for j in fits if j < k]
            if nearest:
                j = max(nearest)
                with torch.no_grad():
                    fitter.betas[:] = torch.tensor(np.asarray(fits[j]["betas"], np.float32))[None]
                    fitter.body_pose[:] = torch.tensor(np.asarray(fits[j]["body_pose"], np.float32))[None]
                    fitter.global_orient[:] = torch.tensor(np.asarray(fits[j]["global_orient"], np.float32))[None]
                warm = True
            started = time.perf_counter()
            residual = fitter.fit_mesh(np.asarray(obstacle[k], np.float64), iterations=warm_iterations if warm else iterations)
            fits[k] = {
                "betas": fitter.betas.detach().numpy().ravel().copy(),
                "body_pose": fitter.body_pose.detach().numpy().ravel().copy(),
                "global_orient": fitter.global_orient.detach().numpy().ravel().copy(),
                "offset": fitter.mesh_offset.detach().numpy().ravel().copy(),
                "residual_m": np.float64(residual),
            }
            warm = True
            if verbose:
                print(f"  fit frame {k:3d}: residual {residual * 100:.2f} cm ({time.perf_counter() - started:.1f} s)")
            keys = sorted(fits)
            np.savez(
                cache,
                frames=np.array(keys),
                **{key: np.stack([fits[j][key] for j in keys]) for key in ("betas", "body_pose", "global_orient", "offset", "residual_m")},
            )
    return fits


def set_fitter(fitter, fit):
    import torch

    with torch.no_grad():
        fitter.betas[:] = torch.tensor(np.asarray(fit["betas"], np.float32))[None]
        fitter.body_pose[:] = torch.tensor(np.asarray(fit["body_pose"], np.float32))[None]
        fitter.global_orient[:] = torch.tensor(np.asarray(fit["global_orient"], np.float32))[None]
    fitter.mesh_offset = torch.tensor(np.asarray(fit["offset"], np.float32))[None]


def signed_distance(points, body, body_faces):
    """Signed distance (m) of `points` to the body: nearest body vertex,
    sign from trimesh's vertex normal (negative = inside)."""
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(np.asarray(body, np.float64), np.asarray(body_faces), process=False)
    normals = np.asarray(mesh.vertex_normals)
    d, nearest = cKDTree(mesh.vertices).query(np.asarray(points, np.float64))
    sign = np.sign(np.einsum("ij,ij->i", np.asarray(points) - mesh.vertices[nearest], normals[nearest]))
    return d * np.where(sign == 0, 1.0, sign)


def inside_fraction(vertices, body_vertices, body_faces, sleeve):
    """Fraction of garment vertices INSIDE the body (negative `signed_distance`).
    ContourCraft's "% penetrating vertices" (utils/metrics.py, broken in their
    repo) and the quantity its collision loss minimises; the reference's own
    `pred` has ~1% (the floor of the metric)."""
    inside = signed_distance(vertices, body_vertices, body_faces) < 0
    return {"all": float(inside.mean()), "sleeve": float(inside[sleeve].mean()), "rest": float(inside[~sleeve].mean())}


def distances(ours, reference, sleeve):
    d = np.linalg.norm(np.asarray(ours) - np.asarray(reference), axis=1) * 100.0
    out = {}
    for name, mask in (("all", np.ones(len(d), bool)), ("sleeve", sleeve), ("rest", ~sleeve)):
        out[name] = {"mean": float(d[mask].mean()), "p90": float(np.percentile(d[mask], 90))}
    return out


def evaluate(checkpoint, trajectory_path, frames, fits_cache, iterations, warm_iterations, verbose=True):
    import torch

    from tryon.garment.skinning import GarmentBinding
    from tryon.body.fitter import BodyFitter
    from tryon.garment.cloth import MAX_DT
    from tryon.garment.renderer import GarmentRenderer

    with open(trajectory_path, "rb") as handle:
        trajectory = pickle.load(handle)
    obstacle = np.asarray(trajectory["obstacle"], np.float64)
    pred = np.asarray(trajectory["pred"], np.float64)
    obstacle_faces = np.asarray(trajectory["obstacle_faces"])
    stride = frames[1] - frames[0] if len(frames) > 1 else 1
    dt = stride / REFERENCE_FPS

    seed = dict(np.load(scan_pose_path(checkpoint))) if scan_pose_path(checkpoint).exists() else None
    if verbose:
        print(f"fitting {len(frames)} frames of {trajectory_path.name} (cache {fits_cache})")
    fits = fit_frames(trajectory, frames, fits_cache, iterations, warm_iterations, seed=seed, verbose=verbose)

    renderers = {
        "lbs": GarmentRenderer(checkpoint, dynamics=False),
        "dynamics": GarmentRenderer(checkpoint, dynamics=True),
    }
    to_ref, count = reference_vertex_map(renderers["lbs"])
    driver = BodyFitter()  # scale / translation: the 2D camera, unused in 3D
    set_fitter(driver, fits[frames[0]])
    for renderer in renderers.values():
        renderer.bind(driver)
    sleeve_fine = renderers["lbs"].sleeve_vertices
    sleeve = np.zeros(count, bool)
    sleeve[to_ref[sleeve_fine]] = True
    ratio = renderers["lbs"]._size_ratio
    if verbose:
        print(f"height ratio wearer / scan body: {ratio:.4f} (bind uses scan_pose: {renderers['lbs'].scan_pose_used}); "
              f"sleeve vertices {int(sleeve.sum())} of {count}")
        print(f"cloth dt per fitted frame: {dt:.3f} s ({'springs run' if dt <= MAX_DT else 'over MAX_DT: the springs restart every frame'})")

    # Reference floor: the template carried rigidly by the TRUE body.
    floor = GarmentBinding.build(pred[0], obstacle[0])

    rows = []
    for k in frames:
        set_fitter(driver, fits[k])
        row = {"frame": k, "fit_residual_cm": float(fits[k]["residual_m"] * 100), "time_s": {}}
        for name, renderer in renderers.items():
            started = time.perf_counter()
            mesh = renderer.posed_mesh(driver, dt=dt) if name == "dynamics" else renderer.posed_mesh(driver)
            row["time_s"][name] = time.perf_counter() - started
            ours = to_reference_order(mesh, to_ref, count)
            row[name] = distances(ours, pred[k], sleeve)
            row[name]["inside"] = inside_fraction(ours, obstacle[k], obstacle_faces, sleeve)
        row["floor_rigid_true_body"] = distances(floor.apply(obstacle[k]), pred[k], sleeve)
        row["reference_inside"] = inside_fraction(pred[k], obstacle[k], obstacle_faces, sleeve)
        row["floor_template_unmoved"] = distances(pred[0], pred[k], sleeve)
        with torch.no_grad():
            body = driver.vertices()
        row["body_fit_vs_obstacle_cm"] = float(np.linalg.norm(body - obstacle[k], axis=1).mean() * 100)
        rows.append(row)
        if verbose:
            print(format_row(row))

    methods = ("lbs", "dynamics", "floor_rigid_true_body", "floor_template_unmoved")
    average = {
        m: {part: {stat: float(np.mean([r[m][part][stat] for r in rows])) for stat in ("mean", "p90")} for part in ("all", "sleeve", "rest")}
        for m in methods
    }
    average["inside"] = {
        m: {part: float(np.mean([r[m]["inside"][part] for r in rows])) for part in ("all", "sleeve", "rest")} for m in renderers
    }
    average["inside"]["reference"] = {part: float(np.mean([r["reference_inside"][part] for r in rows])) for part in ("all", "sleeve", "rest")}
    average["inside_max"] = {m: {part: float(np.max([r[m]["inside"][part] for r in rows])) for part in ("all", "sleeve", "rest")} for m in renderers}
    average["fit_residual_cm"] = float(np.mean([r["fit_residual_cm"] for r in rows]))
    average["time_s"] = {name: float(np.mean([r["time_s"][name] for r in rows])) for name in renderers}
    return {
        "checkpoint": str(checkpoint),
        "trajectory": str(trajectory_path),
        "frames": frames,
        "stride": stride,
        "dt_s": dt,
        "max_dt_s": float(MAX_DT),
        "height_ratio": float(ratio),
        "sleeve_vertices": int(sleeve.sum()),
        "reference_vertices": int(count),
        "per_frame": rows,
        "average": average,
    }


def format_row(row):
    def cell(m, part):
        return f"{row[m][part]['mean']:5.1f}/{row[m][part]['p90']:5.1f}"

    return (f"{row['frame']:5d} {row['fit_residual_cm']:6.2f}  "
            f"{cell('lbs', 'all')} {cell('lbs', 'sleeve')} {cell('lbs', 'rest')} | "
            f"{cell('dynamics', 'all')} {cell('dynamics', 'sleeve')} {cell('dynamics', 'rest')} | "
            f"{cell('floor_rigid_true_body', 'all')} {cell('floor_rigid_true_body', 'sleeve')} | "
            f"{cell('floor_template_unmoved', 'all')}")


HEADER = ("frame    fit   LBS all     sleeve      rest    | DYN all     sleeve      rest    | "
          "rigid-true-body all sleeve | template unmoved\n"
          "        (cm)  mean/p90 cm")


def print_table(result):
    print(HEADER)
    for row in result["per_frame"]:
        print(format_row(row))
    avg = {"frame": -1, "fit_residual_cm": result["average"]["fit_residual_cm"], **result["average"]}
    print("  avg " + format_row(avg)[6:])
    t = result["average"]["time_s"]
    print(f"mean time/frame: lbs {t['lbs'] * 1000:.0f} ms, cloth {t['dynamics'] * 1000:.0f} ms; dt {result['dt_s']:.3f} s (MAX_DT {result['max_dt_s']:.2f} s)")
    inside = result["average"].get("inside")
    if inside:
        worst = result["average"]["inside_max"]
        print("inside the body, % of vertices, mean over frames (max frame): all / sleeve / rest")
        for m in ("lbs", "dynamics"):
            print(f"  {m:10s} " + "  ".join(f"{inside[m][p] * 100:5.1f} ({worst[m][p] * 100:5.1f})" for p in ("all", "sleeve", "rest")))
        print(f"  {'reference':10s} " + "  ".join(f"{inside['reference'][p] * 100:5.1f}" + " " * 8 for p in ("all", "sleeve", "rest")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=str(GARMENTS / "00152_outer"))
    parser.add_argument("--trajectory", default=None, help="the pkl; default: found next to the others by checkpoint name")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--frames", type=int, default=None, help="use only the first N reference frames")
    parser.add_argument("--iterations", type=int, default=500, help="fit_mesh iterations for a cold frame")
    parser.add_argument("--warm-iterations", type=int, default=150, help="fit_mesh iterations for a warm-started frame")
    parser.add_argument("--fits", default=None, help="npz cache of the per-frame body fits")
    parser.add_argument("--out", default=None, help="json results")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    trajectory = Path(args.trajectory).expanduser() if args.trajectory else trajectory_for(checkpoint)
    if trajectory is None or not trajectory.exists():
        raise SystemExit(f"no trajectory file for {checkpoint.name}")
    baseline = Path(__file__).resolve().parents[2] / "docs" / "baseline"
    fits_cache = Path(args.fits) if args.fits else baseline / f"trajectory-fits-{checkpoint.name}.npz"
    out = Path(args.out) if args.out else baseline / f"dynamics-vs-trajectory-{checkpoint.name}.json"
    with open(trajectory, "rb") as handle:
        total = len(pickle.load(handle)["obstacle"])
    if args.frames:
        total = min(total, args.frames)
    frames = list(range(0, total, args.stride))

    result = evaluate(checkpoint, trajectory, frames, fits_cache, args.iterations, args.warm_iterations)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print()
    print_table(result)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
