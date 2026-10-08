"""A static, data-driven RELAXED REST SHAPE per garment, learned from the
authors' ContourCraft trajectory and validated on held-out time blocks.

    .venv/bin/python -m tools.evaluate.relaxed_rest validate 00152_outer 00170_upper --stride 5
    .venv/bin/python -m tools.evaluate.relaxed_rest write 00152_outer 00170_upper

The template carried by skinning is systematically not where the simulated
worn garment sits (gravity droop, how it hangs, fabric settling): on
00152_outer 4.5 cm of the 7.7 cm skin-to-simulation distance is the same
displacement in every frame. For each vertex we express pred[k] - skin[k]
in the vertex's own blended skinning frame, o_k = A_k^-1 (pred_k - skin_k)
with A_k the 3x3 part of the vertex's blended joint transform (so it does
not depend on the body pose), and take the mean over frames:

    rest_relaxed = template + mean_k o_k       (zero-pose frame, scan size)

`GarmentRenderer.RELAXED_REST = "mean"` binds to it, reading the offset from
`<checkpoint>/stage1/relaxed_rest.npz` (`checkpoint.load_relaxed_rest`).

`validate` fits the offset on a block of the trajectory's frames and scores
the distance to pred[k] (cm, reference vertex order) on the frames it did
not see: two halves (fit 0..n/2, test the rest, and the reverse) and four
blocks. These are the SAME garment on the SAME motion and body in different
TIME blocks. `heldout_motion` scores the offset on the authors' resize
simulations of the same garment (other motions, other body sizes).
`write` stores the offset fitted on all frames.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from tools.data.fit_scan_pose import trajectory_for
from tools.evaluate._common import write_json
from tools.evaluate.cloth_smoothness import simulate
from tools.evaluate.cloth_trajectory import reference_vertex_map, set_fitter, to_reference_order
from tryon.garment.checkpoint import relaxed_rest_path
from tryon.paths import BASELINE, GARMENTS

REFERENCE_FPS = 25.0
HEM_FRACTION = 0.20  # lowest share of the non-sleeve vertices (template height) counted as the hem
REGIONS = ("all", "sleeve", "hem", "torso")


class Setup:
    """One garment on its trajectory: the renderer bound on the first fit, the
    per-frame bodies, pred, and the per-frame offsets o_k."""

    def __init__(self, name):
        from tryon.body.fitter import BodyFitter
        from tryon.garment.renderer import GarmentRenderer
        from tryon.garment.texels import read_template

        self.name = name
        checkpoint = GARMENTS / name
        with open(trajectory_for(checkpoint), "rb") as handle:
            pred = np.asarray(pickle.load(handle)["pred"], np.float64)
        self.fits = np.load(BASELINE / f"trajectory-fits-{name}.npz")
        self.n = len(self.fits["frames"])
        self.pred = pred[: self.n]
        self.renderer = GarmentRenderer(checkpoint, dynamics=False)
        self.driver = BodyFitter()
        self.set_frame(0)
        self.renderer.bind(self.driver)
        self.binding = self.renderer._binding
        self.factors = np.asarray(self.renderer._size_applied, np.float64)
        self.to_ref, self.count = reference_vertex_map(self.renderer)
        template = read_template(checkpoint / "stage1" / "template_uv.obj")
        self.ref_faces = np.asarray(template["faces"])
        height = np.asarray(template["vertices"])[:, 1]
        sleeve = np.zeros(self.count, bool)
        sleeve[self.to_ref[self.renderer.sleeve_vertices]] = True
        cut = np.percentile(height[~sleeve], HEM_FRACTION * 100)
        hem = ~sleeve & (height <= cut)
        self.masks = {"all": np.ones(self.count, bool), "sleeve": sleeve, "hem": hem, "torso": ~sleeve & ~hem}
        self.edges = self.renderer.scan_binder.edges

    def set_frame(self, i):
        keys = ("betas", "body_pose", "global_orient", "offset")
        set_fitter(self.driver, {k: self.fits[k][i] for k in keys})

    def blend(self):
        """(V, 3, 3) linear part of every vertex's blended joint transform at the driver's pose."""
        transforms = self.driver.joint_transforms().detach().numpy().astype(np.float64)
        return np.einsum("nj,jab->nab", self.binding.weights, transforms)[:, :3, :3]

    def collect(self):
        """Per-frame offsets o_k (T, V, 3) in the zero-pose frame at the scan size, the
        rigid skin, bodies, in renderer vertex order."""
        offsets, skin, bodies = [], [], []
        for i in range(self.n):
            self.set_frame(i)
            x = self.binding.apply(self.driver)
            d = self.pred[i][self.to_ref] - x
            offsets.append(np.linalg.solve(self.blend(), d[:, :, None])[:, :, 0] / self.factors)
            skin.append(x)
            bodies.append(np.asarray(self.driver.vertices(), np.float32))
        self.offsets = np.stack(offsets).astype(np.float32)
        self.skin = np.stack(skin).astype(np.float32)
        self.bodies = np.stack(bodies)
        return self

    def smooth_duplicates(self, offset):
        """UV-split duplicates of one vertex take the same offset (closed seams)."""
        return to_reference_order(offset, self.to_ref, self.count)[self.to_ref]

    def fit_offset(self, frames, statistic="mean"):
        """Offset of `frames` (mean or median over them), UV-duplicates merged."""
        o = self.offsets[list(frames)].astype(np.float64)
        o = np.median(o, axis=0) if statistic == "median" else o.mean(axis=0)
        return self.smooth_duplicates(o)

    def relaxed_binding(self, offset):
        """The bound garment (already scaled) with `offset` (zero frame, scan size) applied."""
        return self.binding.relaxed(np.asarray(offset) * self.factors)

    def targets(self, binding, frames):
        """Strain-limited skinned targets of `binding` on `frames`, and the strain limits."""
        from tryon.garment.renderer import _strain_passes

        rest = binding.physical_rest
        e = self.edges
        limit = np.linalg.norm(rest[e[:, 1]] - rest[e[:, 0]], axis=1) * (1.0 + self.renderer.STRAIN_LIMIT)
        count = np.maximum(np.bincount(e.ravel(), minlength=len(rest)).astype(np.float64), 1.0)
        out = []
        for i in frames:
            self.set_frame(i)
            x = binding.apply(self.driver)
            _strain_passes(x, e, limit, count, int(self.renderer.STRAIN_ITERATIONS))
            out.append(x)
        return np.stack(out), limit

    def variant(self, binding, frames, stride, make_cloth):
        """Positions in reference order for `frames` ([::stride] of the sequence),
        carrying `binding`; make_cloth None = skin only."""
        frames = list(frames)
        targets, limit = self.targets(binding, frames)
        if make_cloth is None:
            xs = targets
        else:
            seq = {"targets": targets, "bodies": self.bodies[frames], "body_faces": np.asarray(self.driver.faces),
                   "edges": self.edges, "strain_limit": limit}
            _, xs, _ = simulate(seq, 1, make_cloth)  # stride 1 over the given frames; dt below
        return np.stack([to_reference_order(x, self.to_ref, self.count) for x in xs])

    def distances(self, positions, frames):
        """(T, V) distance (cm) to pred[frames]."""
        return np.linalg.norm(positions - self.pred[list(frames)], axis=2) * 100.0


def pooled(distance_blocks, masks):
    """{region: {mean, p90}} over the (T, V) blocks pooled."""
    out = {}
    for region in REGIONS:
        d = np.concatenate([b[:, masks[region]].ravel() for b in distance_blocks])
        out[region] = {"mean": float(d.mean()), "p90": float(np.percentile(d, 90))} if d.size else {"mean": float("nan"), "p90": float("nan")}
    return out


def schemes(n):
    """{scheme: [(train_frames, test_frames)]}: blocked time splits."""
    def blocks(k):
        edges = np.linspace(0, n, k + 1).astype(int)
        return [np.arange(edges[i], edges[i + 1]) for i in range(k)]

    out = {}
    for k, label in ((2, "halves"), (4, "quarters")):
        parts = blocks(k)
        out[label] = [(np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i]) for i in range(k)]
    return out


def validate(setup, stride, makers, statistic="mean", region_scale=None):
    """Held-out distance of every variant on every scheme: {scheme: {variant: {region: stats}}}.
    Variants: `<cloth>` (template) and `<cloth>+relaxed`, cloth in `makers` ({label: class or None})."""
    all_frames = list(range(0, setup.n, stride))
    dt = stride / REFERENCE_FPS
    base = setup.binding
    result = {}
    cache = {}
    for scheme, folds in schemes(setup.n).items():
        blocks = {}
        for train, test in folds:
            offset = setup.fit_offset(train, statistic)
            if region_scale:
                offset = offset * region_scale(setup)[:, None]
            relaxed = setup.relaxed_binding(offset)
            keep = [j for j, k in enumerate(all_frames) if k in set(test.tolist())]
            for label, make in makers.items():
                for tag, binding in (("", base), ("+relaxed", relaxed)):
                    if tag == "" and (label in cache):
                        positions = cache[label]
                    else:
                        positions = _replay(setup, binding, all_frames, dt, make)
                        if tag == "":
                            cache[label] = positions
                    blocks.setdefault(label + tag, []).append(setup.distances(positions[keep], [all_frames[j] for j in keep]))
        result[scheme] = {v: pooled(b, setup.masks) for v, b in blocks.items()}
    return result


def _replay(setup, binding, frames, dt, make_cloth):
    from tryon.garment.renderer import _strain_passes

    targets, limit = setup.targets(binding, frames)
    if make_cloth is None:
        xs = targets
    else:
        cloth = make_cloth()
        cloth.edges = setup.edges
        count = np.maximum(np.bincount(setup.edges.ravel(), minlength=targets.shape[1]).astype(np.float64), 1.0)
        xs = []
        for j, k in enumerate(frames):
            x = np.array(cloth.step(targets[j].astype(np.float64), dt, setup.bodies[k].astype(np.float64), np.asarray(setup.driver.faces)), np.float64)
            _strain_passes(x, setup.edges, limit, count, int(setup.renderer.STRAIN_ITERATIONS))
            xs.append(x)
    return np.stack([to_reference_order(x, setup.to_ref, setup.count) for x in xs])


def format_table(name, result, scheme):
    lines = [f"{name} [{scheme}] held-out distance to pred, cm (mean / p90)"]
    for variant, regions in result[scheme].items():
        lines.append(f"  {variant:18s} " + "  ".join(f"{r} {regions[r]['mean']:5.2f}/{regions[r]['p90']:5.2f}" for r in REGIONS))
    return "\n".join(lines)


def make_makers():
    from tryon.garment.cloth import LightCloth, SmoothCloth

    return {"skin": None, "light": LightCloth, "smooth": SmoothCloth}


def command_validate(args):
    makers = make_makers()
    out = json.load(open(args.out)) if args.out and Path(args.out).exists() and args.resume else {}
    for name in args.names:
        if name in out:
            continue
        started = time.perf_counter()
        setup = Setup(name).collect()
        out[name] = validate(setup, args.stride, makers, args.statistic)
        print(f"{name}  ({time.perf_counter() - started:.0f} s)")
        for scheme in out[name]:
            print(format_table(name, out[name], scheme), flush=True)
        if args.out:
            write_json(Path(args.out), out)
    return out


def command_write(args):
    for name in args.names:
        setup = Setup(name).collect()
        offset = setup.fit_offset(range(setup.n), args.statistic)
        path = relaxed_rest_path(GARMENTS / name)
        np.savez(path, offset=offset.astype(np.float32), frames=setup.n, statistic=args.statistic)
        print(f"{name}: wrote {path} (mean |offset| {np.linalg.norm(offset, axis=1).mean() * 100:.2f} cm)")


def command_heldout(args):
    """The offset (fitted on the garment's own trajectory, all frames) on the authors' RESIZE
    simulations of the same garment: other motions (CMU 13_10/15/17/27), other body sizes,
    scored on the SMPL-X fits of `docs/baseline/motion-fits/rs-<outfit>-<motion>.npz`
    (stride 17), rigid skin (the frames are 0.68 s apart: no cloth)."""
    from tools.evaluate.resize_trajectories import RESIZE, garment_slice

    with open(BASELINE / "resize-inventory.json") as handle:
        listing = json.load(handle)
    out = {}
    for name in args.names:
        sequences = [(f, t) for f in sorted(listing) for t in sorted(listing[f]) if name in listing[f][t]["garments"]]
        if not sequences:
            print(f"{name}: in no resize outfit")
            continue
        setup = Setup(name).collect()
        offset = setup.fit_offset(range(setup.n), args.statistic)
        own_fits = setup.fits
        blocks = {"skin": [], "skin+relaxed": []}
        for folder, take in sequences:
            fits = np.load(BASELINE / "motion-fits" / f"rs-{folder}-{take}.npz")
            with open(RESIZE / folder / f"{take}.pkl", "rb") as handle:
                data = pickle.load(handle)
            start, stop = garment_slice(list(data["garment_names"]), name)
            frames = [int(k) for k in fits["frames"]]
            pred = np.stack([np.asarray(data["pred"][k][start:stop], np.float64) for k in frames])
            del data
            setup.fits = fits
            setup.set_frame(0)
            setup.renderer.bind(setup.driver)
            setup.binding = setup.renderer._binding
            setup.factors = np.asarray(setup.renderer._size_applied, np.float64)
            idx = list(range(len(frames)))
            for tag, binding in (("skin", setup.binding), ("skin+relaxed", setup.relaxed_binding(offset))):
                targets, _ = setup.targets(binding, idx)
                ours = np.stack([to_reference_order(x, setup.to_ref, setup.count) for x in targets])
                blocks[tag].append(np.linalg.norm(ours - pred, axis=2) * 100.0)
        setup.fits = own_fits
        out[name] = {tag: pooled(b, setup.masks) for tag, b in blocks.items()}
        out[name]["sequences"] = [f"{f}/{t}" for f, t in sequences]
        print(name, f"({len(sequences)} sequences)", flush=True)
        for tag in blocks:
            print(f"  {tag:14s} " + "  ".join(f"{r} {out[name][tag][r]['mean']:5.2f}/{out[name][tag][r]['p90']:5.2f}" for r in REGIONS), flush=True)
        if args.out:
            write_json(Path(args.out), out)


BETA_SHIFTS = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2))


def command_generality(args):
    """The relaxed rest on wearers of other shapes: the first two betas shifted by +-2 (every
    combination), garment bound on that wearer (the live path: scan pose, height scale) and posed
    on three trajectory poses. Per wearer: share of garment vertices INSIDE the body (signed
    distance, `cloth_trajectory.signed_distance`) and the stretch of the posed edges over the
    template's physical rest edge lengths, raw and after the strain limit, relaxed against not."""
    import torch

    from tools.evaluate.cloth_trajectory import signed_distance
    from tryon.garment.renderer import _strain_passes

    out = {}
    for name in args.names:
        setup = Setup(name).collect()
        offset = setup.fit_offset(range(setup.n), args.statistic)
        poses = [setup.n // 4, setup.n // 2, 3 * setup.n // 4]
        e = setup.edges
        rows = {}
        for shift in BETA_SHIFTS:
            stats = {"plain": [], "relaxed": []}
            for i in poses:
                setup.set_frame(i)
                with torch.no_grad():
                    setup.driver.betas[0, 0] += shift[0]
                    setup.driver.betas[0, 1] += shift[1]
                setup.renderer.bind(setup.driver)
                plain = setup.renderer._binding
                factors = np.asarray(setup.renderer._size_applied, np.float64)
                relaxed = plain.relaxed(offset * factors)
                body = np.asarray(setup.driver.vertices(), np.float64)
                faces = np.asarray(setup.driver.faces)
                reference = np.linalg.norm(plain.physical_rest[e[:, 1]] - plain.physical_rest[e[:, 0]], axis=1)
                count = np.maximum(np.bincount(e.ravel(), minlength=len(plain.rest)).astype(np.float64), 1.0)
                for tag, binding in (("plain", plain), ("relaxed", relaxed)):
                    x = binding.apply(setup.driver)
                    raw = np.linalg.norm(x[e[:, 1]] - x[e[:, 0]], axis=1) / reference
                    limit = np.linalg.norm(binding.physical_rest[e[:, 1]] - binding.physical_rest[e[:, 0]], axis=1) * (1.0 + setup.renderer.STRAIN_LIMIT)
                    y = np.array(x)
                    _strain_passes(y, e, limit, count, int(setup.renderer.STRAIN_ITERATIONS))
                    limited = np.linalg.norm(y[e[:, 1]] - y[e[:, 0]], axis=1) / reference
                    stats[tag].append({
                        "inside": float((signed_distance(y, body, faces) < 0).mean()),
                        "stretch_max_raw": float(raw.max()), "stretch_p99_raw": float(np.percentile(raw, 99)),
                        "stretch_max_limited": float(limited.max()), "stretch_p99_limited": float(np.percentile(limited, 99)),
                    })
            rows[f"{shift[0]:+d},{shift[1]:+d}"] = {tag: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for tag, v in stats.items()}
        out[name] = rows
        for key, row in rows.items():
            print(f"{name} betas{key}: inside {row['plain']['inside'] * 100:5.1f}% -> {row['relaxed']['inside'] * 100:5.1f}%   "
                  f"stretch max raw {row['plain']['stretch_max_raw']:5.2f} -> {row['relaxed']['stretch_max_raw']:5.2f}  "
                  f"p99 raw {row['plain']['stretch_p99_raw']:4.2f} -> {row['relaxed']['stretch_p99_raw']:4.2f}  "
                  f"max limited {row['plain']['stretch_max_limited']:4.2f} -> {row['relaxed']['stretch_max_limited']:4.2f}", flush=True)
        if args.out:
            write_json(Path(args.out), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for label, fn in (("validate", command_validate), ("write", command_write), ("heldout", command_heldout), ("generality", command_generality)):
        p = sub.add_parser(label)
        p.add_argument("names", nargs="+")
        p.add_argument("--statistic", choices=("mean", "median"), default="mean")
        p.add_argument("--stride", type=int, default=5)
        p.add_argument("--out", default=None)
        p.add_argument("--resume", action="store_true", help="keep the garments already in --out")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
