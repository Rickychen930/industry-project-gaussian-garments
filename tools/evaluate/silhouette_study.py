"""How much of the garment's over-drawn silhouette is the Gaussians' soft fringe rather than geometry.

`vs_registration` scores our alpha > 25 against the real shirt, and a splat's
smooth falloff makes that a low bar. `collect` runs the live path (as
`placement_gap.collect`) and re-renders the same splats on each scored frame
under rasteriser settings (the arguments handed to the rasteriser are
intercepted; tryon/ is unchanged):
  scale xM   every Gaussian's scale times M
  op^G       per-splat opacity to the power G
  ewa0       without the +0.3 px^2 low-pass of `screen_covariance`
  nofloat    without Gaussians farther than 1 cm from the template surface
Each is scored at the `THRESHOLDS` alpha levels: IoU, extra / missed pixels as a
share of the real shirt, and composite guards (interior holes, edge softness).

    .venv/bin/python -m tools.evaluate.silhouette_study collect [videos] [--every 20]
    .venv/bin/python -m tools.evaluate.silhouette_study report [--out docs/baseline/silhouette-study.json]
    .venv/bin/python -m tools.evaluate.silhouette_study checkpoint [garments]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import iou, panel_frames, write_json
from tryon.paths import GARMENTS, SCRATCH

VIDEOS = ("00152_outer", "00170_upper", "00134_upper", "10006_outer", "00190_upper")
THRESHOLDS = (25, 50, 100, 128, 180, 230)
BASE_THRESHOLD = 25  # vs_registration
# name -> (scale multiplier, opacity power, drop the 0.3 low-pass, drop the floaters)
VARIANTS = {
    "base": (1.0, 1.0, False, False),
    "scale0.9": (0.9, 1.0, False, False),
    "scale0.8": (0.8, 1.0, False, False),
    "scale0.7": (0.7, 1.0, False, False),
    "op^1.5": (1.0, 1.5, False, False),
    "op^2": (1.0, 2.0, False, False),
    "scale0.9+op^1.5": (0.9, 1.5, False, False),
    "ewa0": (1.0, 1.0, True, False),
    "nofloat": (1.0, 1.0, False, True),
}
CACHE = Path(SCRATCH) / "silhouette_study"
FIGURE_VIDEOS = ("00152_outer", "00170_upper")
FLOATER_DISTANCE = 0.01  # metres from the template surface
HOLE_ALPHA = 0.9
INTERIOR_ERODE = 5
CLOSE_KERNEL = 25


# ------------------------------------------------------------------ floaters


def surface_distance(points, vertices, faces, per_face=200, seed=0):
    """Distance of each point to the mesh surface, by a KD-tree over a dense random sampling of it."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    count = len(faces) * per_face
    face = np.repeat(np.arange(len(faces)), per_face)
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    samples = tri[face, 0] + u[:, None] * (tri[face, 1] - tri[face, 0]) + v[:, None] * (tri[face, 2] - tri[face, 0])
    return cKDTree(np.vstack([samples, vertices])).query(points)[0]


def floaters(renderer):
    """(bool mask over the renderer's Gaussians farther than FLOATER_DISTANCE from the template at rest, the distances)."""
    rest = renderer.rest_vertices
    placed = renderer.splats(rest, full=True, colours=False)
    distance = surface_distance(np.asarray(placed["xyz"]), rest, renderer.faces)
    return distance > FLOATER_DISTANCE, distance


# ------------------------------------------------------------------ one render


def _prepare_without_low_pass(prepare):
    def wrapped(splats, view, intrinsics, tile):
        pixels, depth, valid, cov2d, _determinant, colours, opacity, _radius = prepare(splats, view, intrinsics, tile)
        cov2d = cov2d.copy()
        cov2d[:, 0, 0] -= 0.3
        cov2d[:, 1, 1] -= 0.3
        determinant = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
        valid = valid & (determinant > 1e-12) & (cov2d[:, 0, 0] > 0) & (cov2d[:, 1, 1] > 0)
        radius = tile * np.sqrt(np.maximum(np.maximum(cov2d[:, 0, 0], cov2d[:, 1, 1]), 0.0))
        return pixels, depth, valid, cov2d, determinant, colours, opacity, radius

    return wrapped


def draw(call, setting, drop):
    """(premultiplied BGR float, alpha float) of the intercepted rasteriser call under `setting`."""
    from tryon.rendering import splat_gl

    mult, power, no_low_pass, no_floaters = setting
    splats, view, intrinsics, size, occluder = call
    splats = dict(splats)
    if mult != 1.0:
        splats["scaling"] = np.asarray(splats["scaling"]) + np.log(mult)
    if power != 1.0:
        splats["opacity_alpha"] = np.asarray(splats["opacity_alpha"]) ** power
    if no_floaters:
        splats["keep"] = np.asarray(splats["keep"], bool) & ~drop
    original = splat_gl._prepare
    if no_low_pass:
        splat_gl._prepare = _prepare_without_low_pass(original)
    try:
        return splat_gl.render_splats_gl(splats, view, intrinsics, size, occluder=occluder)
    finally:
        splat_gl._prepare = original


# ------------------------------------------------------------------ the numbers of one render


def interior_region(base_alpha):
    """The garment's filled region (closing of the baseline mask), pulled in from its edge: where nothing should be see-through."""
    mask = (np.round(base_alpha * 255) > BASE_THRESHOLD).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL)))
    return cv2.erode(closed, np.ones((2 * INTERIOR_ERODE + 1,) * 2, np.uint8)).astype(bool)


def edge_numbers(alpha, mask):
    """(mean |grad alpha| per pixel, width in px of the 0.1-0.9 transition) in a band around the mask's boundary."""
    mask8 = mask.astype(np.uint8)
    band = cv2.dilate(mask8, np.ones((7, 7), np.uint8)).astype(bool) & ~cv2.erode(mask8, np.ones((7, 7), np.uint8)).astype(bool)
    if not band.any():
        return 0.0, 0.0
    gy, gx = np.gradient(alpha.astype(np.float32))
    perimeter = max(int((mask & ~cv2.erode(mask8, np.ones((3, 3), np.uint8)).astype(bool)).sum()), 1)
    transition = int(((alpha > 0.1) & (alpha < HOLE_ALPHA) & band).sum())
    return float(np.hypot(gx, gy)[band].mean()), transition / perimeter


def score(alpha, shirt, region):
    """Per threshold {iou, extra, missed}, plus the guards, for one alpha plane."""
    a8 = np.round(np.clip(alpha, 0, 1) * 255).astype(np.int32)
    area = max(int(shirt.sum()), 1)
    out = {}
    for threshold in THRESHOLDS:
        mask = a8 > threshold
        out[str(threshold)] = {"iou": iou(mask, shirt), "extra": float((mask & ~shirt).sum() / area),
                               "missed": float((shirt & ~mask).sum() / area), "area_ratio": float(mask.sum() / area)}
    mask25 = a8 > BASE_THRESHOLD
    grad, width = edge_numbers(alpha, mask25)
    out["holes"] = float((alpha[region] < HOLE_ALPHA).mean()) if region.any() else 0.0
    out["edge_gradient"], out["edge_width_px"] = grad, width
    out["garment_area"] = int(mask25.sum())
    return out


class Collector(list):
    """CAPTURE list: on every scored frame re-render the intercepted call under all the VARIANTS."""

    def __init__(self, drop, figure_frame):
        super().__init__()
        self.call = None
        self.drop = drop
        self.frames, self.figure = [], {}
        self.figure_frame = figure_frame

    def append(self, item):
        super().append(item)
        t, _raw, ours, shirt = item
        call, self.call = self.call, None
        if call is None:
            return
        rendered = {name: draw(call, setting, self.drop) for name, setting in VARIANTS.items()}
        region = interior_region(rendered["base"][1])
        record = {"frame": int(t), "shirt_area": int(shirt.sum()),
                  "metric_pixels_differ": int((ours != (np.round(rendered["base"][1] * 255) > BASE_THRESHOLD)).sum()),
                  "variants": {name: score(alpha, shirt, region) for name, (_image, alpha) in rendered.items()}}
        self.frames.append(record)
        if len(self.frames) == self.figure_frame:
            self.figure = {"t": int(t), "shirt": shirt.copy(), **{
                name: (np.clip(image * 255, 0, 255).astype(np.uint8), np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8))
                for name, (image, alpha) in rendered.items()}}
        print(f"  frame {t}: base IoU@25 {record['variants']['base']['25']['iou']:.3f}", flush=True)


def collect(name, every=20):
    from tools.evaluate import vs_registration as vr
    from tryon.garment.renderer import GarmentRenderer
    from tryon.rendering import splat_gl

    GarmentRenderer.GPU_RASTER = True
    vr.REGRESSOR, vr.REGRESS_EVERY, vr.REGRESS_LAG = "nlf", 1, 0
    drop = floaters(GarmentRenderer(GARMENTS / name, dynamics=False))[0]
    collector = Collector(drop, figure_frame=3)
    real = splat_gl.render_splats_gl

    def intercept(splats, view, intrinsics, size, tile=3.0, bgra8=False, occluder=None, margin=0.03):
        collector.call = (splats, view, intrinsics, size, occluder)
        return real(splats, view, intrinsics, size, tile=tile, bgra8=bgra8, occluder=occluder, margin=margin)

    splat_gl.render_splats_gl = intercept
    vr.CAPTURE = collector
    try:
        result = vr.evaluate(name, every)
    finally:
        splat_gl.render_splats_gl = real
        vr.CAPTURE = None
    CACHE.mkdir(parents=True, exist_ok=True)
    write_json(CACHE / f"{name}.json", {"video": name, "every": every, "reported_ours": result["summary"]["ours"],
                                        "floaters": int(drop.sum()), "gaussians": int(len(drop)), "frames": collector.frames})
    if name in FIGURE_VIDEOS and collector.figure:
        figure_data = collector.figure
        panel = panel_frames(name, vr.HEIGHT, figure_data["t"] + 1, step=max(figure_data["t"], 1))[-1]
        arrays = {"panel": panel, "t": figure_data["t"], "shirt": figure_data["shirt"]}
        for key in VARIANTS:
            arrays[f"{key}/image"], arrays[f"{key}/alpha"] = figure_data[key]
        np.savez_compressed(CACHE / f"{name}_figure.npz", **arrays)
    print(name, "reported IoU", result["summary"]["ours"], "frames", len(collector.frames), flush=True)


# ------------------------------------------------------------------ report


def _mean(records, variant, threshold, key):
    return float(np.mean([f["variants"][variant][str(threshold)][key] for f in records]))


def aggregate(per_video):
    """{variant: {thresholds: {t: {iou, extra, missed, ...}}, guards}} as the mean over the videos (each video's frames first)."""
    out = {}
    for variant in VARIANTS:
        row = {"thresholds": {}}
        for threshold in THRESHOLDS:
            cell = {key: float(np.mean([_mean(frames, variant, threshold, key) for frames in per_video.values()]))
                    for key in ("iou", "extra", "missed", "area_ratio")}
            cell["per_video_iou"] = {v: _mean(f, variant, threshold, "iou") for v, f in per_video.items()}
            row["thresholds"][str(threshold)] = cell
        for key in ("holes", "edge_gradient", "edge_width_px", "garment_area"):
            row[key] = float(np.mean([np.mean([f["variants"][variant][key] for f in frames]) for frames in per_video.values()]))
        out[variant] = row
    return out


def held_out(per_video, pool):
    """Every 3-video train / 2-video test split: pick the (variant, threshold) from `pool` best on train, score it on test."""
    names = list(per_video)
    splits = []
    for train in combinations(names, 3):
        test = [n for n in names if n not in train]
        table = {(v, t): float(np.mean([_mean(per_video[n], v, t, "iou") for n in train])) for v, t in pool}
        chosen = max(table, key=table.get)
        baseline = float(np.mean([_mean(per_video[n], "base", BASE_THRESHOLD, "iou") for n in test]))
        picked = float(np.mean([_mean(per_video[n], chosen[0], chosen[1], "iou") for n in test]))
        splits.append({"train": list(train), "test": test, "chosen": list(chosen), "test_iou_chosen": picked, "test_iou_baseline": baseline})
    return {"splits": splits, "mean_gain": float(np.mean([s["test_iou_chosen"] - s["test_iou_baseline"] for s in splits])),
            "splits_with_gain": int(sum(s["test_iou_chosen"] > s["test_iou_baseline"] for s in splits))}


def figure(path, picks):
    rows = []
    for name in FIGURE_VIDEOS:
        file = CACHE / f"{name}_figure.npz"
        if not file.exists():
            continue
        data = np.load(file)
        shirt = data["shirt"]
        ys, xs = np.nonzero(shirt | (data["base/alpha"] > 25))
        y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
        row = []
        for variant in picks:
            alpha = data[f"{variant}/alpha"].astype(np.float32)[..., None] / 255
            out = np.clip(data[f"{variant}/image"].astype(np.float32) + (1 - alpha) * data["panel"].astype(np.float32), 0, 255).astype(np.uint8)
            outline = cv2.findContours(shirt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(out, outline, -1, (0, 255, 0), 1)  # the real shirt's outline
            crop = out[y0:y1, x0:x1]
            crop = cv2.resize(crop, (int(crop.shape[1] * 520 / crop.shape[0]), 520))
            cv2.putText(crop, f"{name} {variant}", (6, 22), 0, 0.6, (255, 255, 255), 3)
            cv2.putText(crop, f"{name} {variant}", (6, 22), 0, 0.6, (0, 0, 0), 1)
            row.append(crop)
        rows.append(np.hstack(row))
    if rows:
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def report(out, figure_path, picks=None):
    per_video = {}
    for name in VIDEOS:
        file = CACHE / f"{name}.json"
        if file.exists():
            per_video[name] = json.loads(file.read_text())["frames"]
    table = aggregate(per_video)
    pool = [(v, t) for v in VARIANTS for t in THRESHOLDS]
    ranking = sorted(((table[v]["thresholds"][str(t)]["iou"], v, t) for v, t in pool), reverse=True)
    result = {"videos": list(per_video), "frames": {v: len(f) for v, f in per_video.items()}, "table": table,
              "best_by_variant": {v: max((table[v]["thresholds"][str(t)]["iou"], t) for t in THRESHOLDS) for v in VARIANTS},
              "held_out_any_setting": held_out(per_video, pool),
              "held_out_footprint_only_at_25": held_out(per_video, [(v, BASE_THRESHOLD) for v in VARIANTS]),
              "held_out_threshold_only": held_out(per_video, [("base", t) for t in THRESHOLDS]),
              "top5": ranking[:5]}
    checks = {name: json.loads((CACHE / f"{name}.json").read_text()) for name in per_video}
    result["floaters"] = {n: {"floaters": c["floaters"], "gaussians": c["gaussians"]} for n, c in checks.items()}
    result["checkpoints"] = {n: checkpoint(n) for n in FIGURE_VIDEOS}
    write_json(out, result)
    picks = picks or ["base", *[v for _, v, _ in ranking if v != "base"][:2]]
    figure(figure_path, picks)
    return result


# ------------------------------------------------------------------ the checkpoint itself


def checkpoint(name):
    from tryon.garment.renderer import GarmentRenderer

    renderer = GarmentRenderer(GARMENTS / name, dynamics=False)
    opacity = renderer.appearance["opacity_alpha"]
    scale = np.exp(renderer.log_scaling)
    mask, distance = floaters(renderer)

    def histogram(values, bins):
        counts, edges = np.histogram(values, bins=bins)
        return {"edges": [round(float(e), 3) for e in edges], "counts": [int(c) for c in counts]}

    return {"garment": name, "gaussians": int(len(opacity)),
            "opacity_histogram": histogram(opacity, np.linspace(0, 1, 11)),
            "opacity_below_0.05": float((opacity < 0.05).mean()), "opacity_below_0.5": float((opacity < 0.5).mean()),
            "opacity_percentiles_5_50_95": [float(x) for x in np.percentile(opacity, [5, 50, 95])],
            "log10_scale_histogram_per_axis": [histogram(np.log10(scale[:, k]), np.linspace(-4, 0, 17)) for k in range(3)],
            "log10_scale_median_per_axis": [float(np.median(np.log10(scale[:, k]))) for k in range(3)],
            "mesh_extent": [float(x) for x in np.ptp(renderer.rest_vertices, axis=0)],
            "floaters_over_1cm": int(mask.sum()), "floater_share": float(mask.mean()),
            "floaters_over_2cm": float((distance > 0.02).mean()),
            "distance_percentiles_50_90_99": [float(x) for x in np.percentile(distance, [50, 90, 99])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "report", "checkpoint"])
    parser.add_argument("videos", nargs="?", default=",".join(VIDEOS))
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--out", default="docs/baseline/silhouette-study.json")
    parser.add_argument("--figure", default="docs/figures/silhouette-study.jpg")
    parser.add_argument("--picks", default=None, help="comma list of the variants to show in the figure")
    args = parser.parse_args()
    if args.command == "collect":
        for name in args.videos.split(","):
            collect(name, args.every)
    elif args.command == "checkpoint":
        for name in args.videos.split(","):
            print(json.dumps(checkpoint(name)))
    else:
        result = report(args.out, args.figure, args.picks.split(",") if args.picks else None)
        print(json.dumps({k: result[k] for k in ("best_by_variant", "top5")}, indent=1))


if __name__ == "__main__":
    main()
