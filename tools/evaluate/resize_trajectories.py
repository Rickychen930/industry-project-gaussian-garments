"""The garment's size handling against the authors' "resize" simulations: is a
garment bound on ITS scan body and posed on ANOTHER body still on that
body, and does the height-only scale (`SkinnedGarment.scale`) leave it
inside the body as girth changes?

    .venv/bin/python -m tools.evaluate.resize_trajectories --garment 10006_outer

The data (`~/tryon-data/trajectories/resize/<folder>/<take>.pkl`,
inspected 25 Sep): ten folders o0100..o0280, each ONE OUTFIT - two to
four garments from different scanned subjects (`garment_names`, e.g.
o0100 = 00170_lower + 00190_upper + 10006_outer, o0170 = 00152_lower +
00176_upper + 00152_outer + 10004_outer) - simulated by the authors on
four motion takes (13_10, 13_15, 13_17, 13_27; 402 frames each).
`pred[k]` stacks the garments in `garment_names` order (vertex counts
equal each checkpoint's stage1/template_uv.obj, and they sum exactly to
pred's width); `obstacle[k]` is the SMPL-X body (10,475 vertices, our
BodyFitter's topology). Frame 0 is NOT the template pose - the outfit is
already on the moving body (nearest template vertex 5-45 cm away) - so
every frame is a test of posing the template onto a different body.
Every (folder, take) pair is its own body (measured by the SMPL-X fit
below, 25 Sep: heights 1.53-1.89 m, chest girth 62-136 cm, i.e. girth
ratios 0.54-1.48 to the scan bodies), so betas[0], zero-pose height and
chest width / depth / girth are reported per folder and take.

Per (folder, take) that contains `--garment`, and per frame in
`--frames` (default 0, 100, 200, 300):

1. SMPL-X is fitted to `obstacle[k]` (`BodyFitter.fit_mesh`, warm-started
   along the take; every fit cached in docs/baseline/resize-fits.npz).
2. `GarmentRenderer.bind` on a fitter set to the frame-0 fit, rigid LBS,
   exactly the live path with the checkpoint's scan pose cached
   (`ScanPoseBinder.bind`: bound in the template frame on the scan body,
   then `SkinnedGarment.scale` by the wearer's zero-pose height over the
   scan body's). The bound garment's zero-pose rest is then re-scaled per
   `--scale-modes` about the pelvis:
     none         ratio 1 - the control (the template at the scan
                  subject's size);
     height       the live path: x, y, z by the height ratio;
     anisotropic  x by the chest-WIDTH ratio, y by the height ratio, z by
                  the chest-DEPTH ratio;
     girth        x and z by the chest-GIRTH (slab perimeter) ratio, y by
                  the height ratio.
   Every mode is posed on the frame-k fit and compared with `pred[k]`'s
   slice for the garment: mean / p90 vertex distance (cm), all / sleeves
   (`renderer.sleeve_vertices`) / rest.
3. Garment-inside-body fraction: the signed distance of every garment
   vertex to `obstacle[k]` (nearest body vertex, trimesh vertex normal;
   negative = inside), for each mode and, as the floor of the metric, for
   `pred[k]` itself.

Chest measurements are taken on the ZERO-pose body of the fitted betas:
the non-arm vertices within 2 cm of the spine3 joint's height (joint 9);
width = x extent, depth = z extent, girth = convex-hull perimeter in xz.
The ratios are wearer over scan body (`stage1/scan_pose.npz` betas).

Writes docs/baseline/resize-eval-<garment>.json and prints one line per
(folder, take), averaged over the frames.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from tools.evaluate.cloth_trajectory import distances, reference_vertex_map, set_fitter, signed_distance, to_reference_order
from tools.evaluate._common import write_json
from tryon.paths import GARMENTS, TRAJECTORIES

RESIZE = TRAJECTORIES / "resize"
FRAMES = (0, 100, 200, 300)
MODES = ("none", "height", "anisotropic", "girth")
CHEST_JOINT = 9  # SMPL-X spine3
CHEST_SLAB_M = 0.02
FIT_KEYS = ("betas", "body_pose", "global_orient", "offset", "residual_m")


# --- data ---------------------------------------------------------------------

def template_vertex_count(garment):
    from tryon.garment.texels import read_template

    return len(read_template(GARMENTS / garment / "stage1" / "template_uv.obj")["vertices"])


def inventory(root, cache):
    """{folder: {take: {"garments": [...], "frames": n}}}, from every pkl
    once (160 MB each) and then from `cache`."""
    if cache.exists():
        with open(cache) as handle:
            return json.load(handle)
    out = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        out[folder.name] = {}
        for pkl in sorted(folder.glob("*.pkl")):
            with open(pkl, "rb") as handle:
                data = pickle.load(handle)
            out[folder.name][pkl.stem] = {"garments": list(data["garment_names"]), "frames": int(len(data["obstacle"])), "vertices": int(data["pred"].shape[1])}
            del data
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache, out)
    return out


def garment_slice(names, garment):
    """[start, stop) of `garment` in pred's vertex axis, from the
    checkpoints' template vertex counts in `garment_names` order."""
    start = 0
    for name in names:
        count = template_vertex_count(name)
        if name == garment:
            return start, start + count
        start += count
    raise KeyError(garment)


# --- body ---------------------------------------------------------------------

_probe = None


def probe():
    global _probe
    if _probe is None:
        from tryon.body.fitter import BodyFitter

        _probe = BodyFitter()
    return _probe


def zero_pose_measurements(betas):
    """Height and chest width / depth / girth (m) of the zero-pose body
    with these betas."""
    import torch
    from scipy.spatial import ConvexHull

    from tryon.garment.skinning import SkinnedGarment

    fitter = probe()
    with torch.no_grad():
        fitter.betas[:] = torch.tensor(np.asarray(betas, np.float32))[None]
        fitter.body_pose.zero_()
        fitter.global_orient.zero_()
        fitter.mesh_offset = None
        output = fitter._forward()
        vertices = output.vertices[0].numpy().astype(np.float64)
        chest_y = float(output.joints[0, CHEST_JOINT, 1])
    weights = fitter.model.lbs_weights.detach().numpy()
    is_arm = weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
    slab = vertices[(~is_arm) & (np.abs(vertices[:, 1] - chest_y) < CHEST_SLAB_M)]
    return {
        "height": float(vertices[:, 1].max() - vertices[:, 1].min()),
        "chest_width": float(slab[:, 0].max() - slab[:, 0].min()),
        "chest_depth": float(slab[:, 2].max() - slab[:, 2].min()),
        "chest_girth": float(ConvexHull(slab[:, [0, 2]]).area),  # 2D hull: area is the perimeter
    }


class FitCache:
    """docs/baseline/resize-fits.npz: one entry per folder/take/frame."""

    def __init__(self, path):
        self.path = path
        self.fits = {}
        if path.exists():
            data = np.load(path)
            for i, key in enumerate(data["keys"]):
                self.fits[str(key)] = {name: data[name][i] for name in FIT_KEYS}

    def get(self, folder, take, frame):
        return self.fits.get(f"{folder}/{take}/{frame}")

    def put(self, folder, take, frame, fit):
        self.fits[f"{folder}/{take}/{frame}"] = fit
        keys = sorted(self.fits)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, keys=np.array(keys), **{name: np.stack([self.fits[k][name] for k in keys]) for name in FIT_KEYS})


def fit_frames(cache, folder, take, obstacle, frames, iterations, warm_iterations, verbose=True):

    from tryon.body.fitter import BodyFitter

    fits = {}
    fitter = None
    for k in frames:
        fit = cache.get(folder, take, k)
        if fit is None:
            if fitter is None:
                fitter = BodyFitter()
            warm = [j for j in fits if j < k]
            if warm:
                set_fitter(fitter, fits[max(warm)])
            started = time.perf_counter()
            residual = fitter.fit_mesh(np.asarray(obstacle[k], np.float64), iterations=warm_iterations if warm else iterations)
            fit = {
                "betas": fitter.betas.detach().numpy().ravel().copy(),
                "body_pose": fitter.body_pose.detach().numpy().ravel().copy(),
                "global_orient": fitter.global_orient.detach().numpy().ravel().copy(),
                "offset": fitter.mesh_offset.detach().numpy().ravel().copy(),
                "residual_m": np.float64(residual),
            }
            cache.put(folder, take, k, fit)
            if verbose:
                print(f"  fit {folder}/{take} frame {k:3d}: residual {residual * 100:.2f} cm ({time.perf_counter() - started:.1f} s)")
        fits[k] = fit
    return fits


# --- metrics ------------------------------------------------------------------

def inside_stats(points, body, body_faces, sleeve):
    s = signed_distance(points, body, body_faces)
    return {
        "inside": float((s < 0).mean()),
        "inside_sleeve": float((s[sleeve] < 0).mean()) if sleeve.any() else float("nan"),
        "inside_rest": float((s[~sleeve] < 0).mean()),
        "median_signed_cm": float(np.median(s) * 100),
    }


def scale_factors(mode, wearer, scan):
    h = wearer["height"] / scan["height"]
    if mode == "none":
        return np.ones(3)
    if mode == "height":
        return np.array([h, h, h])
    if mode == "anisotropic":
        return np.array([wearer["chest_width"] / scan["chest_width"], h, wearer["chest_depth"] / scan["chest_depth"]])
    if mode == "girth":
        g = wearer["chest_girth"] / scan["chest_girth"]
        return np.array([g, h, g])
    raise ValueError(mode)


# --- evaluation ---------------------------------------------------------------

def evaluate(garment, takes, frames, modes, fits_cache, inventory_cache, iterations, warm_iterations, verbose=True):

    from tryon.garment.skinning import SkinnedGarment
    from tryon.body.fitter import BodyFitter
    from tryon.garment.renderer import GarmentRenderer

    checkpoint = GARMENTS / garment
    scan_betas = np.load(checkpoint / "stage1" / "scan_pose.npz")["betas"]
    scan = zero_pose_measurements(scan_betas)
    scan["betas0"] = float(scan_betas[0])
    if verbose:
        print(f"{garment}: scan body betas0 {scan['betas0']:+.2f} height {scan['height']:.3f} m "
              f"chest width {scan['chest_width'] * 100:.1f} depth {scan['chest_depth'] * 100:.1f} girth {scan['chest_girth'] * 100:.1f} cm")

    listing = inventory(RESIZE, inventory_cache)
    sequences = [(folder, take) for folder in sorted(listing) for take in sorted(listing[folder])
                 if garment in listing[folder][take]["garments"] and (not takes or take in takes)]
    if not sequences:
        raise SystemExit(f"{garment} is in no resize outfit")
    if verbose:
        print(f"{len(sequences)} sequences hold {garment}: " + ", ".join(f"{f}/{t}" for f, t in sequences))

    renderer = GarmentRenderer(checkpoint, dynamics=False)
    # These bodies are TRUE shapes from the authors' simulations, so the
    # chest ratio measures the wearer and widening is valid here; the live
    # default is off because a monocular fit no longer estimates girth.
    renderer.WIDEN_TO_WEARER = True
    to_ref, count = reference_vertex_map(renderer)
    cache = FitCache(fits_cache)
    driver = BodyFitter()
    rows = []
    for folder, take in sequences:
        with open(RESIZE / folder / f"{take}.pkl", "rb") as handle:
            data = pickle.load(handle)
        names = list(data["garment_names"])
        start, stop = garment_slice(names, garment)
        obstacle = {k: np.asarray(data["obstacle"][k], np.float64) for k in frames}
        pred = {k: np.asarray(data["pred"][k][start:stop], np.float64) for k in frames}
        body_faces = np.asarray(data["obstacle_faces"])
        del data
        if pred[frames[0]].shape[0] != count:
            raise ValueError(f"{garment} slice has {pred[frames[0]].shape[0]} vertices, template {count}")

        fits = fit_frames(cache, folder, take, obstacle, frames, iterations, warm_iterations, verbose=verbose)
        set_fitter(driver, fits[frames[0]])
        renderer.bind(driver)
        if not renderer.scan_pose_used:
            raise RuntimeError("bind did not use the scan pose")
        sleeve = np.zeros(count, bool)
        sleeve[to_ref[renderer.sleeve_vertices]] = True
        binding = renderer._binding
        rest0 = binding.rest / renderer._size_ratio
        physical0 = binding.physical_rest / renderer._size_ratio
        wearer = zero_pose_measurements(fits[frames[0]]["betas"])
        wearer["betas0"] = float(fits[frames[0]]["betas"][0])
        factors = {mode: scale_factors(mode, wearer, scan) for mode in modes}
        # The live path's own ratio, for the record: it is the height mode.
        assert np.isclose(renderer._size_ratio, factors["height"][1] if "height" in factors else wearer["height"] / scan["height"], atol=1e-4)

        row = {
            "folder": folder, "take": take, "garment_slice": [start, stop], "outfit": names,
            "wearer": wearer, "height_ratio": float(renderer._size_ratio),
            "factors": {m: f.tolist() for m, f in factors.items()},
            "sleeve_vertices": int(sleeve.sum()),
            "per_frame": [],
        }
        for k in frames:
            set_fitter(driver, fits[k])
            frame = {"frame": k, "fit_residual_cm": float(fits[k]["residual_m"] * 100), "modes": {}}
            for mode in modes:
                f = factors[mode]
                posed = SkinnedGarment(rest0 * f, binding.weights, physical_rest=physical0 * f).apply(driver)
                posed = to_reference_order(posed, to_ref, count)
                frame["modes"][mode] = {"error": distances(posed, pred[k], sleeve), **inside_stats(posed, obstacle[k], body_faces, sleeve)}
            frame["reference"] = inside_stats(pred[k], obstacle[k], body_faces, sleeve)
            row["per_frame"].append(frame)
        row["average"] = average(row["per_frame"], modes)
        rows.append(row)
        if verbose:
            print(format_row(row, modes))
    return {
        "garment": garment, "checkpoint": str(checkpoint), "frames": list(frames), "modes": list(modes),
        "scan_body": scan, "chest_joint": CHEST_JOINT, "chest_slab_m": CHEST_SLAB_M,
        "sequences": rows,
    }


def average(per_frame, modes):
    out = {"fit_residual_cm": float(np.mean([f["fit_residual_cm"] for f in per_frame])),
           "reference_inside": float(np.mean([f["reference"]["inside"] for f in per_frame]))}
    for mode in modes:
        out[mode] = {
            "error": {part: {stat: float(np.mean([f["modes"][mode]["error"][part][stat] for f in per_frame])) for stat in ("mean", "p90")}
                      for part in ("all", "sleeve", "rest")},
            "inside": float(np.mean([f["modes"][mode]["inside"] for f in per_frame])),
            "inside_sleeve": float(np.nanmean([f["modes"][mode]["inside_sleeve"] for f in per_frame])),
            "inside_rest": float(np.mean([f["modes"][mode]["inside_rest"] for f in per_frame])),
        }
    return out


def header(modes):
    cells = "  ".join(f"{m[:6]:>6}: all mean/p90 slv rest in%" for m in modes)
    return (f"{'folder':6} {'take':5} {'b0':>5} {'hgt':>5} {'width':>5} {'depth':>5} {'girth':>5} | {'ratio h':>7} {'w':>5} {'d':>5} {'g':>5} | "
            f"{cells} | ref in%  fit cm")


def format_row(row, modes):
    w, a = row["wearer"], row["average"]
    f = row["factors"]
    ratios = (f"{row['height_ratio']:7.3f} {f['anisotropic'][0] if 'anisotropic' in f else float('nan'):5.3f} "
              f"{f['anisotropic'][2] if 'anisotropic' in f else float('nan'):5.3f} {f['girth'][0] if 'girth' in f else float('nan'):5.3f}")
    cells = []
    for m in modes:
        e = a[m]["error"]
        cells.append(f"{'':>6}  {e['all']['mean']:4.1f}/{e['all']['p90']:4.1f} {e['sleeve']['mean']:4.1f} {e['rest']['mean']:4.1f} {a[m]['inside'] * 100:3.0f}%")
    return (f"{row['folder']:6} {row['take']:5} {w['betas0']:+5.2f} {w['height']:5.2f} {w['chest_width'] * 100:5.1f} {w['chest_depth'] * 100:5.1f} {w['chest_girth'] * 100:5.1f} | "
            f"{ratios} | " + "  ".join(cells) + f" | {a['reference_inside'] * 100:5.1f}%  {a['fit_residual_cm']:4.2f}")


def print_table(result):
    modes = result["modes"]
    s = result["scan_body"]
    print(f"scan body of {result['garment']}: betas0 {s['betas0']:+.2f} height {s['height']:.2f} m chest width {s['chest_width'] * 100:.1f} depth {s['chest_depth'] * 100:.1f} girth {s['chest_girth'] * 100:.1f} cm")
    print(header(modes))
    for row in result["sequences"]:
        print(format_row(row, modes))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--garment", default="10006_outer")
    parser.add_argument("--takes", nargs="*", default=None, help="restrict to these takes (default: all that hold the garment)")
    parser.add_argument("--frames", type=int, nargs="*", default=list(FRAMES))
    parser.add_argument("--scale-modes", nargs="*", default=list(MODES), choices=MODES)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warm-iterations", type=int, default=200)
    parser.add_argument("--fits", default=None, help="npz cache of the body fits (default docs/baseline/resize-fits.npz)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    baseline = Path(__file__).resolve().parents[2] / "docs" / "baseline"
    fits_cache = Path(args.fits) if args.fits else baseline / "resize-fits.npz"
    out = Path(args.out) if args.out else baseline / f"resize-eval-{args.garment}.json"
    result = evaluate(args.garment, args.takes, tuple(args.frames), tuple(args.scale_modes), fits_cache,
                      baseline / "resize-inventory.json", args.iterations, args.warm_iterations)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print()
    print_table(result)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
