"""Open-front garments (jackets, outer shirts): where the front panels go wrong, and which fix helps.

`record` runs `vs_registration.evaluate` once per video and keeps, per renderer
call, the fitted body and the real shirt mask (the body logic is the expensive
part and identical across cloth variants). `replay` drives a `GarmentRenderer`
with those bodies under any class-attribute variant and scores IoU, 3D fit and
cloth-step cost. `pieces` diagnoses per cloth piece.

    .venv/bin/python -m tools.evaluate.open_front record 00122_outer [--cache DIR]
    .venv/bin/python -m tools.evaluate.open_front replay 00122_outer [--rigid] [--set Class.ATTR=value ...] [--out j.json]
    .venv/bin/python -m tools.evaluate.open_front pieces 00122_outer [--frames 30,80,140]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from tools.evaluate._common import iou, write_json

CACHE = Path("/tmp/gg/openfront")
OPEN_FRONT = ("00122_outer", "00136_outer", "00148_outer", "00152_outer", "00176_outer", "10004_outer", "10006_outer")
CLOSED = ("00170_upper", "00190_upper", "00134_upper")
ALPHA_MASK = 25


# ------------------------------------------------------------------ record


def record(name, every=5, cache=CACHE):
    """One live-pipeline pass (NLF body, GL, the class defaults): the renderer's calls and the real shirt masks."""
    from tools.evaluate import vs_registration as vr
    from tryon.garment.renderer import GarmentRenderer

    GarmentRenderer.GPU_RASTER = True
    calls, state = [], {"depth": 0, "gen": 0, "last": None}
    original_posed, original_render = GarmentRenderer.posed_mesh, GarmentRenderer.render

    def snapshot(kind, fitter, dt, size=None):
        import torch

        if state["last"] is not id(fitter):
            state["last"], state["gen"] = id(fitter), state["gen"] + 1
        with torch.no_grad():
            calls.append({"kind": kind, "gen": state["gen"], "dt": dt, "size": size,
                          "jt": fitter.joint_transforms().detach().numpy().astype(np.float32),
                          "verts": np.asarray(fitter.vertices(), np.float32),
                          "scale": float(fitter.scale.detach()), "trans": fitter.translation.detach().numpy().ravel().astype(np.float64),
                          "betas": fitter.betas.detach().numpy().ravel().astype(np.float32)})

    def posed(self, fitter, dt=None):
        if state["depth"] == 0:
            snapshot("step", fitter, dt)
        return original_posed(self, fitter, dt)

    def render(self, fitter, size, *args, **kwargs):
        snapshot("render", fitter, kwargs.get("dt"), tuple(size))
        state["depth"] += 1
        try:
            return original_render(self, fitter, size, *args, **kwargs)
        finally:
            state["depth"] -= 1

    GarmentRenderer.posed_mesh, GarmentRenderer.render = posed, render
    vr.REGRESSOR, vr.REGRESS_EVERY, vr.REGRESS_LAG = "nlf", 1, 0
    capture = []
    vr.CAPTURE = capture
    try:
        result = vr.evaluate(name, every)
    finally:
        vr.CAPTURE = None
        GarmentRenderer.posed_mesh, GarmentRenderer.render = original_posed, original_render
    Path(cache).mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack([np.asarray(c[k]) for c in calls]) for k in ("jt", "verts", "scale", "trans", "betas", "gen")}
    arrays["dt"] = np.array([c["dt"] or 0.0 for c in calls], np.float64)
    truth = np.stack([c[3] for c in capture])
    np.savez_compressed(Path(cache) / f"{name}.npz", kind=np.array([c["kind"] == "render" for c in calls]),
                        size=np.array([c["size"] or (0, 0) for c in calls]), t=np.array([c[0] for c in capture]),
                        truth=np.packbits(truth, axis=-1), truth_width=truth.shape[-1], reported=np.array([r["ours"] for r in result["frames"]]), **arrays)
    print(name, "recorded", len(calls), "calls", len(capture), "scored frames; live IoU", result["summary"]["ours"], flush=True)


# ------------------------------------------------------------------ replay


class RecordedFitter:
    """What `GarmentRenderer` asks of a `BodyFitter`, answered from a recorded call."""

    def __init__(self, faces):
        self._faces = self.faces = np.asarray(faces)
        self._shape = None
        self.model = SimpleNamespace(faces=self._faces)

    def load(self, data, i):
        import torch

        self._jt, self._verts = data["jt"][i].astype(np.float64), data["verts"][i]
        if self._shape is None:
            from tryon.body.fitter import BodyFitter

            self._shape = BodyFitter()
        with torch.no_grad():
            self._shape.betas[:] = torch.tensor(np.asarray(data["betas"][i], np.float32))[None]
            self._rest_joints = self._shape.joints(indices=list(range(55))).numpy().astype(np.float64)
        self.scale = torch.tensor(float(data["scale"][i]))
        self.translation = torch.tensor(np.asarray(data["trans"][i], np.float32))
        self.betas = torch.tensor(np.asarray(data["betas"][i], np.float32))[None]

    def joints(self, indices=None):
        """Posed joints (the rest joints of the recorded shape, carried by the recorded transforms)."""
        import torch

        if indices is None:
            raise NotImplementedError("pass the joint indices")
        homogeneous = np.concatenate([self._rest_joints[list(indices)], np.ones((len(indices), 1))], axis=1)
        return torch.tensor(np.einsum("nab,nb->na", self._jt[list(indices)], homogeneous)[:, :3])

    def joint_transforms(self):
        import torch

        return torch.tensor(self._jt)

    def vertices(self):
        return self._verts.astype(np.float64)

    def _forward(self):
        import torch

        return SimpleNamespace(vertices=torch.tensor(self._verts)[None])


def set_attributes(assignments):
    """`module.Class.ATTR=value` strings applied as class attributes; returns the undo list."""
    undo = []
    for item in assignments:
        target, value = item.split("=", 1)
        owner_path, attribute = target.rsplit(".", 1)
        module_path, class_name = owner_path.rsplit(".", 1)
        owner = getattr(importlib.import_module(module_path), class_name)
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = getattr(importlib.import_module(module_path), value) if hasattr(importlib.import_module(module_path), value) else value
        undo.append((owner, attribute, getattr(owner, attribute, None)))
        setattr(owner, attribute, parsed)
    return undo


def load_record(name, cache=CACHE):
    data = dict(np.load(Path(cache) / f"{name}.npz"))
    data["truth"] = np.unpackbits(data["truth"], axis=-1)[..., : int(data["truth_width"])].astype(bool)
    return data


def replay(name, every=5, rigid=False, assignments=(), cache=CACHE, keep_meshes=None, keep_rgba=None):
    """Score a variant on the recorded bodies. Returns {"frames": [...], "summary": {...}}."""
    from tools.evaluate.fit_audit import signed_distance
    from tools.evaluate.placement_gap import decompose
    from tryon.body.fitter import BodyFitter
    from tryon.garment.renderer import GarmentRenderer
    from tryon.paths import GARMENTS

    GarmentRenderer.GPU_RASTER = True
    undo = set_attributes(assignments)
    try:
        data = load_record(name, cache)
        renderer = GarmentRenderer(GARMENTS / name, dynamics=not rigid)
        fitter = RecordedFitter(BodyFitter().faces)
        generation, scored, rows, step_ms = None, 0, [], []
        original_posed = renderer.posed_mesh
        last = {}

        cloth_class = renderer.CLOTH
        original_step = cloth_class.step

        def timed_step(self, *args, **kwargs):
            started = time.perf_counter()
            out = original_step(self, *args, **kwargs)
            step_ms.append(1000 * (time.perf_counter() - started))
            return out

        cloth_class.step = timed_step

        def posed(f, dt=None):
            mesh = original_posed(f, dt)
            last["mesh"] = np.array(mesh, np.float64)
            return mesh

        renderer.posed_mesh = posed
        for i in range(len(data["kind"])):
            if data["gen"][i] != generation:
                generation = data["gen"][i]
                fitter = RecordedFitter(fitter._faces)
                renderer._binding = None
            fitter.load(data, i)
            if not data["kind"][i]:
                renderer.posed_mesh(fitter, float(data["dt"][i]))
                continue
            rgba = renderer.render(fitter, tuple(int(x) for x in data["size"][i]), dt=float(data["dt"][i]))
            ours = rgba[..., 3] > ALPHA_MASK
            truth = data["truth"][scored]
            row = {"frame": int(data["t"][scored]), "iou": iou(ours, truth)}
            split = decompose(ours, truth) if ours.any() and truth.any() else None
            if split:
                row.update({f"split_{k}": v for k, v in split.items()})
            signed = signed_distance(last["mesh"], fitter.vertices(), np.asarray(fitter._faces))
            row["inside_share"] = float((signed < 0).mean())
            if keep_rgba is not None and row["frame"] in keep_rgba:
                keep_rgba[row["frame"]] = rgba
            if keep_meshes is not None:
                keep_meshes[row["frame"]] = (last["mesh"], fitter.vertices(), renderer._binding.apply(fitter))
            rows.append(row)
            scored += 1
    finally:
        cloth_class.step = original_step
        for owner, attribute, old in reversed(undo):
            setattr(owner, attribute, old)
    summary = {k: float(np.mean([r[k] for r in rows if k in r])) for k in rows[0] if k != "frame"}
    summary["cloth_step_ms"] = float(np.median(step_ms)) if step_ms else 0.0
    return {"video": name, "frames": rows, "summary": summary}


# ------------------------------------------------------------------ pieces


def piece_labels(faces, count):
    """Connected component of every garment vertex over the cloth mesh's faces (seam faces excluded by the caller)."""
    faces = np.asarray(faces)
    both = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = sp.coo_matrix((np.ones(len(both)), (both[:, 0], both[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def strip(name, frames, assignments, out, cache=CACHE, every=5, before=()):
    """Before / after strip: per frame the photo, the garment drawn by the class defaults and by `assignments`,
    each with the real shirt's contour (green) and ours (red). Written to `out` (jpg)."""
    import cv2

    from tools.evaluate._common import panel_frames

    frames = list(frames)
    versions = {}
    for tag, setting in (("before", tuple(before)), ("after", tuple(assignments))):
        versions[tag] = dict.fromkeys(frames)
        replay(name, every, False, setting, cache, keep_rgba=versions[tag])
    truth = load_record(name, cache)
    index = {int(t): k for k, t in enumerate(truth["t"])}
    photos = panel_frames(name, 870, max(frames) + 1, step=every)
    rows = []
    for frame in frames:
        photo = photos[frame // every]
        tiles = [photo.copy()]
        for tag in ("before", "after"):
            rgba = versions[tag][frame]
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            composite = np.clip(photo.astype(np.float32) * (1.0 - alpha) + rgba[..., :3].astype(np.float32), 0, 255).astype(np.uint8)
            mask = rgba[..., 3] > ALPHA_MASK
            for contour_mask, colour in ((truth["truth"][index[frame]], (0, 220, 0)), (mask, (0, 0, 255))):
                contours, _ = cv2.findContours(contour_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(composite, contours, -1, colour, 2)
            tiles.append(composite)
        for tile, label in zip(tiles, (f"{name} f{frame} photo", "before", "after")):
            cv2.putText(tile, label, (8, 28), 0, 0.8, (255, 255, 255), 2)
        rows.append(cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows))


def piece_report(name, frames, rigid=False, assignments=(), cache=CACHE):
    """Per cloth piece (connected component of the faces that are drawn), on `frames`: distance of its vertices
    to the fitted body (cloth and skin target, mm, signed, median / p90), the cloth's displacement from the
    skin target (cm), its skinning weight on the arm joints, and the same at the garment's rest."""
    from tools.evaluate.fit_audit import signed_distance
    from tryon.garment.renderer import GarmentRenderer
    from tryon.garment.skinning import SkinnedGarment
    from tryon.paths import GARMENTS

    meshes = {}
    replay(name, 5, rigid, assignments, cache, keep_meshes=meshes)
    renderer = GarmentRenderer(GARMENTS / name)
    data = load_record(name, cache)
    # pieces: components over faces never culled for the sleeve seam; bind once on
    # the first recorded body (the seam belongs to the bound garment)
    from tryon.body.fitter import BodyFitter

    fitter = RecordedFitter(BodyFitter().faces)
    fitter.load(data, 0)
    renderer.bind(fitter)
    drawn = renderer.faces[~renderer._seam_faces]
    labels = piece_labels(drawn, len(renderer.rest_vertices))
    weights = renderer._binding.weights
    arm = weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1)
    sizes = np.bincount(labels)
    pieces = [p for p in np.argsort(-sizes) if sizes[p] >= 50]
    rest = renderer._binding.physical_rest
    rows = []
    for frame in frames:
        mesh, body, skin = meshes[min(meshes, key=lambda k: abs(k - frame))]
        faces = np.asarray(fitter._faces)
        cloth_d = 1000 * signed_distance(mesh, body, faces)
        skin_d = 1000 * signed_distance(skin, body, faces)
        for p in pieces:
            m = labels == p
            rows.append({"frame": frame, "piece": int(p), "vertices": int(m.sum()), "arm_weight": float(arm[m].mean()),
                         "cloth_mm_p50": float(np.median(cloth_d[m])), "cloth_mm_p90": float(np.percentile(cloth_d[m], 90)),
                         "skin_mm_p50": float(np.median(skin_d[m])), "skin_mm_p90": float(np.percentile(skin_d[m], 90)),
                         "drift_cm_p50": float(100 * np.median(np.linalg.norm(mesh[m] - skin[m], axis=1))),
                         "drift_cm_p90": float(100 * np.percentile(np.linalg.norm(mesh[m] - skin[m], axis=1), 90)),
                         "centroid_height_m": float(rest[m, 1].mean())})
    return rows


COLUMNS = ("iou", "split_extra", "split_missed", "split_hem_offset", "split_torso_width_ratio", "inside_share")


def compare(tags, cache=CACHE):
    """Table of `replay --out rep_<tag>_<video>.json` results: per video and the open-front / closed means."""
    import json

    cache = Path(cache)
    results = {tag: {} for tag in tags}
    for tag in tags:
        for path in sorted(cache.glob(f"rep_{tag}_*.json")):
            results[tag][path.stem[len(f"rep_{tag}_"):]] = json.loads(path.read_text())["summary"]
    videos = sorted({v for r in results.values() for v in r})
    lines = ["variant | video | " + " | ".join(COLUMNS)]
    for tag in tags:
        for video in videos:
            if video in results[tag]:
                lines.append(f"{tag} | {video} | " + " | ".join(f"{results[tag][video].get(c, float('nan')):.4f}" for c in COLUMNS))
        for label, group in (("open-front mean", OPEN_FRONT), ("closed mean", CLOSED)):
            rows = [results[tag][v] for v in group if v in results[tag]]
            if rows:
                lines.append(f"{tag} | {label} ({len(rows)}) | " + " | ".join(f"{np.mean([r.get(c, float('nan')) for r in rows]):.4f}" for c in COLUMNS))
    return "\n".join(lines)


def main():
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["record", "replay", "pieces", "compare", "strip"])
    parser.add_argument("video", help="the video, or for `compare` the comma list of variant tags")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--rigid", action="store_true")
    parser.add_argument("--set", dest="assignments", action="append", default=[], help="module.Class.ATTR=value")
    parser.add_argument("--out", default=None)
    parser.add_argument("--frames", default="30,80,140")
    parser.add_argument("--before", action="append", default=[], help="strip: module.Class.ATTR=value of the before column (default: the class defaults)")
    args = parser.parse_args()
    CACHE = Path(args.cache)
    if args.command == "record":
        record(args.video, args.every, CACHE)
        return
    if args.command == "strip":
        strip(args.video, [int(x) for x in args.frames.split(",")], args.assignments, args.out, CACHE, args.every, args.before)
        return
    if args.command == "compare":
        print(compare(args.video.split(","), CACHE))
        return
    if args.command == "pieces":
        for row in piece_report(args.video, [int(x) for x in args.frames.split(",")], args.rigid, args.assignments, CACHE):
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
        return
    result = replay(args.video, args.every, args.rigid, args.assignments, CACHE)
    print(args.video, {k: round(v, 4) for k, v in result["summary"].items() if not k.startswith("split_")}, flush=True)
    if args.out:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
