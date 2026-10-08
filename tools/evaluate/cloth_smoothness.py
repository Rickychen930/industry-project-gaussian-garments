"""Cloth quality, offline and fast: distance to the authors' ContourCraft
simulation plus how smooth the garment's motion is.

    .venv/bin/python -m tools.evaluate.cloth_smoothness 00152_outer 00170_upper --stride 5

The skinned targets (the renderer's rigid LBS on the cached SMPL-X fits of
`tools.evaluate.cloth_trajectory`), the fitted bodies, and the reference
`pred` are cached once per garment as `<cache>/<name>.npz` for every one of the
160 frames; a cloth variant is then replayed on `frames[::stride]` with
dt = stride / 25 s, so a variant costs seconds rather than a render.

Metrics (all in the reference's vertex order, cm; lower is better):
  distance  mean / p90 vertex distance to pred[k] (all / sleeves / rest),
            after `GarmentRenderer._limit_strain` like the live path.
  jitter    mean norm of the second difference of the displacement
            d = x - skinned target over the replayed frames (the body's own
            motion removed), per frame^2; for the reference d = pred - target.
  rough     mean norm of the uniform graph Laplacian of d over the mesh
            faces: how far each vertex's displacement is from its
            neighbours' mean; the reference's value is the natural floor.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tools.evaluate.cloth_trajectory import (
    REFERENCE_FPS,
    reference_vertex_map,
    set_fitter,
    to_reference_order,
)
from tools.data.fit_scan_pose import trajectory_for
from tryon.paths import GARMENTS

BASELINE = Path(__file__).resolve().parents[2] / "docs" / "baseline"
CACHE = Path("/tmp/tryon-cloth-seq")


def build_sequence(name, cache_dir=CACHE):
    """Cache (and return) the replay inputs of one garment; see module doc."""
    from tryon.body.fitter import BodyFitter
    from tryon.garment.renderer import GarmentRenderer

    path = Path(cache_dir) / f"{name}.npz"
    if path.exists():
        return load_sequence(path)
    checkpoint = GARMENTS / name
    with open(trajectory_for(checkpoint), "rb") as handle:
        trajectory = pickle.load(handle)
    pred = np.asarray(trajectory["pred"], np.float64)
    fits = np.load(BASELINE / f"trajectory-fits-{name}.npz")
    renderer = GarmentRenderer(checkpoint, dynamics=False)
    driver = BodyFitter()
    keys = ("betas", "body_pose", "global_orient", "offset")
    fit = lambda i: {k: fits[k][i] for k in keys}  # noqa: E731
    set_fitter(driver, fit(0))
    renderer.bind(driver)
    to_ref, count = reference_vertex_map(renderer)
    from tryon.garment.texels import read_template

    template = read_template(renderer.checkpoint_dir / "stage1" / "template_uv.obj")
    edges = renderer.scan_binder.edges
    rest = renderer._binding.physical_rest
    limit = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1) * (1.0 + renderer.STRAIN_LIMIT)
    targets, bodies = [], []
    for i, k in enumerate(fits["frames"]):
        set_fitter(driver, fit(i))
        targets.append(renderer.posed_mesh(driver))
        bodies.append(np.asarray(driver.vertices(), np.float32))
    sleeve = np.zeros(count, bool)
    sleeve[to_ref[renderer.sleeve_vertices]] = True
    n = len(targets)
    np.savez(
        path,
        targets=np.stack(targets).astype(np.float32), bodies=np.stack(bodies), pred=pred[:n].astype(np.float32),
        body_faces=np.asarray(driver.faces), to_ref=to_ref, sleeve=sleeve, ref_faces=np.asarray(template["faces"]),
        edges=edges, strain_limit=limit, count=np.array(count),
    )
    return load_sequence(path)


def load_sequence(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def laplacian_matrix(faces, n):
    """Row-normalised uniform graph Laplacian (I - D^-1 A) over the faces' edges."""
    import scipy.sparse as sp

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    deg = np.asarray(a.sum(axis=1)).ravel()
    return sp.identity(n) - sp.diags(1.0 / np.maximum(deg, 1)) @ a


def simulate(seq, stride, make_cloth, limit_strain=True, filter_target=None):
    """Replay frames[::stride]; returns (frames, x (T, V, 3) in renderer order, targets)."""
    from tryon.garment.renderer import _strain_passes, GarmentRenderer

    frames = list(range(0, len(seq["targets"]), stride))
    dt = stride / REFERENCE_FPS
    cloth = make_cloth()
    cloth.edges = seq["edges"]
    edges, limit = seq["edges"], seq["strain_limit"]
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(seq["targets"][0])).astype(np.float64), 1.0)
    out, tg = [], []
    for k in frames:
        target = seq["targets"][k].astype(np.float64)
        if filter_target is not None:
            target = filter_target(target, dt)
        x = cloth.step(target, dt, seq["bodies"][k].astype(np.float64), seq["body_faces"])
        x = np.array(x, np.float64)
        if limit_strain:
            _strain_passes(x, edges, limit, count, int(GarmentRenderer.STRAIN_ITERATIONS))
        out.append(x)
        tg.append(target)
    return frames, np.stack(out), np.stack(tg)


def metrics(seq, frames, xs, targets):
    to_ref = seq["to_ref"]
    n_ref = len(seq["pred"][0])
    sleeve = seq["sleeve"]
    pred = seq["pred"][frames].astype(np.float64)
    L = laplacian_matrix(seq["ref_faces"], n_ref)
    ours = np.stack([to_reference_order(x, to_ref, n_ref) for x in xs])
    skin = np.stack([to_reference_order(t, to_ref, n_ref) for t in targets])
    out = {}
    d = np.linalg.norm(ours - pred, axis=2) * 100.0
    out["distance"] = {
        part: {"mean": float(d[:, m].mean()) if d[:, m].size else float("nan"),
               "p90": float(np.percentile(d[:, m], 90)) if d[:, m].size else float("nan")}
        for part, m in (("all", slice(None)), ("sleeve", sleeve), ("rest", ~sleeve))
    }
    for label, pos in (("ours", ours), ("reference", pred)):
        disp = pos - skin  # (T, V, 3)
        acc = disp[2:] - 2 * disp[1:-1] + disp[:-2]
        rough = np.stack([L @ disp[t] for t in range(len(disp))])
        raw = pos[2:] - 2 * pos[1:-1] + pos[:-2]
        out[label] = {
            "jitter_cm": float(np.linalg.norm(acc, axis=2).mean() * 100),
            "raw_accel_cm": float(np.linalg.norm(raw, axis=2).mean() * 100),
            "rough_cm": float(np.linalg.norm(rough, axis=2).mean() * 100),
            "disp_cm": float(np.linalg.norm(disp, axis=2).mean() * 100),
        }
    return out


def evaluate(names, stride, make_cloth, **kw):
    per = {}
    for name in names:
        seq = SEQUENCES.get(name) or SEQUENCES.setdefault(name, build_sequence(name))
        frames, xs, tg = simulate(seq, stride, make_cloth, **kw)
        per[name] = metrics(seq, frames, xs, tg)
    return per


SEQUENCES = {}


def summarise(per):
    """Mean over garments of the headline numbers."""
    def avg(f):
        return float(np.nanmean([f(m) for m in per.values()]))
    return {
        "dist_all": avg(lambda m: m["distance"]["all"]["mean"]),
        "dist_p90": avg(lambda m: m["distance"]["all"]["p90"]),
        "dist_sleeve": avg(lambda m: m["distance"]["sleeve"]["mean"]),
        "dist_rest": avg(lambda m: m["distance"]["rest"]["mean"]),
        "jitter": avg(lambda m: m["ours"]["jitter_cm"]),
        "rough": avg(lambda m: m["ours"]["rough_cm"]),
        "ref_jitter": avg(lambda m: m["reference"]["jitter_cm"]),
        "ref_rough": avg(lambda m: m["reference"]["rough_cm"]),
    }


def format_summary(label, s):
    return (f"{label:28s} dist {s['dist_all']:.3f} (p90 {s['dist_p90']:.2f} sleeve {s['dist_sleeve']:.2f} rest {s['dist_rest']:.2f}) "
            f"jitter {s['jitter']:.3f} (ref {s['ref_jitter']:.3f}) rough {s['rough']:.3f} (ref {s['ref_rough']:.3f})")


def main():
    from tryon.garment.cloth import LightCloth

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--stride", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = {}
    for stride in args.stride:
        started = time.perf_counter()
        per = evaluate(args.names, stride, LightCloth)
        # the skinned (no cloth) pass-through for the same frames
        class Skin:
            def step(self, target, dt, body, faces):
                return target

        per_skin = evaluate(args.names, stride, Skin)
        result[f"stride{stride}"] = {"light_cloth": per, "skin_only": per_skin,
                                     "summary": {"light_cloth": summarise(per), "skin_only": summarise(per_skin)}}
        print(f"stride {stride} (dt {stride / REFERENCE_FPS:.2f} s), {time.perf_counter() - started:.0f} s")
        for name in args.names:
            print(f"  {name}: " + format_summary("light", summarise({name: per[name]})))
        print(format_summary("MEAN light_cloth", summarise(per)))
        print(format_summary("MEAN skin_only", summarise(per_skin)))
    if args.out:
        write_json(Path(args.out), result)


if __name__ == "__main__":
    main()
