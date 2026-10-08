"""Where the 0.73 -> 0.83 gap to the authors' registration is lost: placement bias, frame noise, or shape.

`collect` runs `vs_registration.evaluate` (NLF regressor, as the README number) and keeps, per scored
frame, our garment mask O, the real shirt mask T (SCHP) and the registration's garment mask R in the
left panel. `analyse` then, on those masks only:

  1. finds per frame the similarity (log scale, dx, dy, rotation about O's centroid) that maximises
     IoU(warp(O), T) (and, separately, vs R): the ceiling of ANY 2D placement fix;
  2. fits ONE constant transform per video on the odd scored frames and scores it on the even ones
     (and the reverse), and a pooled one on 4 videos scored on the 5th;
  3. gives the mean / std / lag-1 autocorrelation of the per-frame optimum (bias vs noise vs drift);
  4. splits the error left after alignment: area ratio, extra / missed pixels by region, row widths.

    .venv/bin/python -m tools.evaluate.placement_gap collect [videos] [--cache DIR]
    .venv/bin/python -m tools.evaluate.placement_gap analyse [videos] [--cache DIR] [--out j.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from tools.evaluate._common import iou, write_json

VIDEOS = ("00152_outer", "00170_upper", "00134_upper", "10006_outer", "00190_upper")
CACHE = Path("/tmp/gg/placement")
SCALE = 0.5  # masks are aligned at half resolution
PANEL_HEIGHT = 870  # vs_registration.HEIGHT


class _Capture(list):
    """CAPTURE list that also remembers the registration mask warped into the panel for each frame."""

    def __init__(self):
        super().__init__()
        self.theirs = []
        self.last = None

    def append(self, item):
        super().append(item)
        self.theirs.append(self.last)


def collect(name, every=5, size_mode="current", relaxed_rest=False):
    from tools.evaluate import vs_registration as vr
    from tryon.garment.renderer import GarmentRenderer

    GarmentRenderer.GPU_RASTER = True
    GarmentRenderer.SIZE_MODE = size_mode
    GarmentRenderer.RELAXED_REST = "mean" if relaxed_rest else None
    binds = []
    original_bind = GarmentRenderer.bind

    def bind(self, fitter):
        original_bind(self, fitter)
        binds.append((float(self._size_history[0]), [float(x) for x in self._size_applied]))

    GarmentRenderer.bind = bind
    vr.REGRESSOR, vr.REGRESS_EVERY, vr.REGRESS_LAG = "nlf", 1, 0
    capture = _Capture()
    original_warp, original_masks = vr.warp, vr.registration_masks
    garment = {}

    def masks(region, *args, **kwargs):
        person, reg_garment = original_masks(region, *args, **kwargs)
        garment["mask"] = reg_garment
        return person, reg_garment

    def warp(mask, *args, **kwargs):
        out = original_warp(mask, *args, **kwargs)
        if mask is garment.get("mask"):
            capture.last = out
        return out

    vr.warp, vr.registration_masks = warp, masks
    vr.CAPTURE = capture
    try:
        result = vr.evaluate(name, every)
    finally:
        vr.warp, vr.registration_masks, vr.CAPTURE = original_warp, original_masks, None
        GarmentRenderer.bind = original_bind
    CACHE.mkdir(parents=True, exist_ok=True)
    times = np.array([c[0] for c in capture])
    ours = np.stack([c[2] for c in capture])
    truth = np.stack([c[3] for c in capture])
    theirs = np.stack([t if t is not None else np.zeros_like(c[3]) for t, c in zip(capture.theirs, capture, strict=True)])
    np.savez_compressed(CACHE / f"{name}.npz", t=times, ours=ours, truth=truth, theirs=theirs,
                        reported=np.array([[r["ours"], r["theirs"]] for r in result["frames"]]),
                        binds=np.array([[b[0], *b[1]] for b in binds]))
    print(name, "collected", len(times), "frames", result["summary"]["ours"], result["summary"]["theirs"])


# ----------------------------------------------------------------- alignment


def _small(mask):
    return cv2.resize(mask.astype(np.uint8), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)


def _matrix(centre, params, rotate=True):
    log_s, dx, dy, theta = params if rotate else (*params, 0.0)
    s = np.exp(log_s)
    c, sn = s * np.cos(theta), s * np.sin(theta)
    cx, cy = centre
    return np.array([[c, -sn, cx - c * cx + sn * cy + dx], [sn, c, cy - sn * cx - c * cy + dy]], np.float32)


def _warp_float(mask_f, centre, params, rotate=True):
    h, w = mask_f.shape
    return cv2.warpAffine(mask_f, _matrix(centre, params, rotate), (w, h), flags=cv2.INTER_LINEAR)


def _soft_iou(warped, target):
    inter = np.minimum(warped, target).sum()
    union = np.maximum(warped, target).sum()
    return inter / union if union else 1.0


def _centre(mask):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)


def _height(mask):
    ys = np.nonzero(mask.any(axis=1))[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else 1.0


def _prepare(ours, target):
    o, t = _small(ours), _small(target)
    return o.astype(np.float32), t.astype(np.float32), _centre(o), _height(t)


def align_frame(ours, target, rotate=True):
    """Best (log s, dx, dy[, theta]) about O's centroid for IoU(warp(O), target); dx, dy in full-res pixels."""
    o, t, centre, _ = _prepare(ours, target)
    if o.sum() == 0 or t.sum() == 0:
        return None
    tc = _centre(t)
    start = [0.5 * np.log(t.sum() / o.sum()), tc[0] - centre[0], tc[1] - centre[1], 0.0]
    n = 4 if rotate else 3
    best = None
    for init in ([0.0, 0.0, 0.0, 0.0], start):
        res = minimize(lambda p: -_soft_iou(_warp_float(o, centre, p, rotate), t), init[:n],
                       method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400,
                                                      "initial_simplex": _simplex(init[:n])})
        if best is None or res.fun < best.fun:
            best = res
    params = np.array(best.x, float)
    params[1:3] /= SCALE
    return params


def _simplex(x0):
    x0 = np.asarray(x0, float)
    steps = [0.05, 8.0, 8.0, 0.03][: len(x0)]
    return np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])


def apply(ours, params, rotate=True, unit=None):
    """Hard mask after the transform (`params` with dx, dy in full-res pixels; `unit`: scale dx, dy by it)."""
    p = np.array(params, float)
    if unit is not None:
        p[1:3] *= unit
    p[1:3] *= SCALE
    o = _small(ours).astype(np.float32)
    warped = _warp_float(o, _centre(o), p, rotate) > 0.5
    return cv2.resize(warped.astype(np.uint8), (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def hard_iou(ours, params, target, rotate=True):
    o, t = _small(ours), _small(target)
    p = np.array(params, float)
    p[1:3] *= SCALE
    warped = _warp_float(o.astype(np.float32), _centre(o), p, rotate) > 0.5
    return iou(warped, t.astype(bool))


def fit_constant(frames, rotate=True):
    """One transform (log s, dx/height, dy/height[, theta]) maximising the mean soft IoU over `frames`
    [(ours, target)]; dx, dy are fractions of each frame's O height so the unit follows the person."""
    prepared = [_prepare(o, t) for o, t in frames]
    prepared = [p for p in prepared if p[0].sum() and p[1].sum()]
    heights = [_height(o) for o, _ in ((p[0], 0) for p in prepared)]

    def cost(p):
        total = 0.0
        for (o, t, centre, _), h in zip(prepared, heights, strict=True):
            q = np.array(p, float)
            q[1:3] *= h
            total += _soft_iou(_warp_float(o, centre, q, rotate), t)
        return -total / len(prepared)

    n = 4 if rotate else 3
    simplex = np.vstack([np.zeros(n)] + [np.eye(n)[i] * [0.04, 0.02, 0.02, 0.03][i] for i in range(n)])
    res = minimize(cost, np.zeros(n), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300, "initial_simplex": simplex})
    return np.array(res.x, float)


def hard_constant(ours, constant, target, rotate=True):
    p = np.array(constant, float)
    p[1:3] *= _height(_small(ours)) / SCALE
    return hard_iou(ours, p, target, rotate)


# ------------------------------------------------------------ decomposition


def decompose(ours, truth):
    """Leftover error between two aligned masks (fractions of the real shirt's area unless noted)."""
    ys, xs = np.nonzero(truth)
    if len(ys) == 0 or ours.sum() == 0:
        return None
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    height, width = bottom - top + 1, right - left + 1
    rows = np.arange(truth.shape[0])[:, None]
    cols = np.arange(truth.shape[1])[None, :]
    rel_y, rel_x = (rows - top) / height, (cols - left) / width
    extra, missed = ours & ~truth, truth & ~ours
    area = truth.sum()
    out = {"area_ratio": float(ours.sum() / area), "extra": float(extra.sum() / area), "missed": float(missed.sum() / area)}
    arm = np.abs(rel_x - 0.5) > 0.3  # outside the central 60 % of the width: sleeves / arms
    regions = {"torso_upper": (rel_y < 0.4) & ~arm, "torso_mid": (rel_y >= 0.4) & (rel_y < 0.8) & ~arm,
               "hem": (rel_y >= 0.8) & ~arm, "sleeve_left": arm & (rel_x < 0.5), "sleeve_right": arm & (rel_x >= 0.5)}
    for key, region in regions.items():
        out[f"extra@{key}"] = float((extra & region).sum() / area)
        out[f"missed@{key}"] = float((missed & region).sum() / area)
    # row widths (torso rows) and vertical extents of the two masks, in fractions of the real height
    band = (rel_y[:, 0] >= 0.45) & (rel_y[:, 0] < 0.75)
    wo, wt = ours.sum(axis=1)[band], truth.sum(axis=1)[band]
    out["torso_width_ratio"] = float(wo.sum() / max(wt.sum(), 1))
    oy = np.nonzero(ours.any(axis=1))[0]
    out["top_offset"] = float((oy.min() - top) / height)
    out["hem_offset"] = float((oy.max() - bottom) / height)
    out["horizontal_extent_ratio"] = float((np.ptp(np.nonzero(ours.any(axis=0))[0]) + 1) / width)
    return out


def lag1(series):
    x = np.asarray(series, float)
    x = x - x.mean()
    denom = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / denom) if denom > 1e-12 and len(x) > 2 else float("nan")


# ----------------------------------------------------------------- analysis


def load(name):
    data = np.load(CACHE / f"{name}.npz")
    keep = [i for i in range(len(data["t"])) if data["ours"][i].any()]
    return {k: data[k][keep] for k in ("t", "ours", "truth", "theirs")}


def analyse_video(name, rotate=True):
    d = load(name)
    n = len(d["t"])
    result = {"video": name, "frames": n}
    for tag, target in (("truth", d["truth"]), ("theirs", d["theirs"])):
        before = [iou(d["ours"][i], target[i]) for i in range(n)]
        params = [align_frame(d["ours"][i], target[i], rotate) for i in range(n)]
        after = [hard_iou(d["ours"][i], params[i], target[i], rotate) for i in range(n)]
        after_notrot = [hard_iou(d["ours"][i], align_frame(d["ours"][i], target[i], False)[:3], target[i], False) for i in range(n)]
        pairs = [(d["ours"][i], target[i]) for i in range(n)]
        held = []
        for fit_set, test_set in ((range(1, n, 2), range(0, n, 2)), (range(0, n, 2), range(1, n, 2))):
            constant = fit_constant([pairs[i] for i in fit_set], rotate)
            held += [hard_constant(*pairs[i][:1], constant, pairs[i][1], rotate) for i in test_set]
        full = fit_constant(pairs, rotate)
        stats = {"iou_before": float(np.mean(before)), "iou_per_frame_aligned_no_rotation": float(np.mean(after_notrot)),
                 "iou_per_frame_aligned": float(np.mean(after)), "iou_constant_heldout_odd_even": float(np.mean(held)),
                 "constant_transform_all_frames": {"log_s": float(full[0]), "dx_per_height": float(full[1]), "dy_per_height": float(full[2]), "theta_rad": float(full[3] if rotate else 0)}}
        arr = np.array(params, float)
        heights = np.array([_height(_small(o)) / SCALE for o in d["ours"]])
        garment_height = np.array([_height(t) for t in target])
        norm = np.column_stack([arr[:, 0], arr[:, 1] / garment_height, arr[:, 2] / garment_height, arr[:, 3] if rotate else np.zeros(n)])
        stats["optimal_transform"] = {key: {"mean": float(norm[:, j].mean()), "std": float(norm[:, j].std()), "lag1": lag1(norm[:, j]),
                                            "abs_mean_over_std": float(abs(norm[:, j].mean()) / max(norm[:, j].std(), 1e-9))}
                                      for j, key in enumerate(("log_s", "dx_per_garment_height", "dy_per_garment_height", "theta_rad"))}
        stats["optimal_transform_dx_dy_per_panel_height_mean"] = [float(arr[:, 1].mean() / PANEL_HEIGHT), float(arr[:, 2].mean() / PANEL_HEIGHT)]
        del heights
        if tag == "truth":
            aligned = [apply(d["ours"][i], params[i], rotate) for i in range(n)]
            raw = [decompose(d["ours"][i], target[i]) for i in range(n)]
            fixed = [decompose(aligned[i], target[i]) for i in range(n)]
            stats["leftover_before"] = _mean_dict(raw)
            stats["leftover_after_alignment"] = _mean_dict(fixed)
        result[tag] = stats
    result["theirs_vs_truth"] = float(np.mean([iou(d["theirs"][i], d["truth"][i]) for i in range(n)]))
    return result


def _mean_dict(rows):
    rows = [r for r in rows if r]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def pooled_heldout(names, rotate=True, scale_only=False):
    """Fit one constant on the other videos' frames, score the held-out video (vs real shirt)."""
    data = {n: load(n) for n in names}
    out = {}
    for held in names:
        fit = [(o, t) for n in names if n != held for o, t in zip(data[n]["ours"][::2], data[n]["truth"][::2], strict=True)]
        constant = fit_constant(fit, rotate)
        if scale_only:
            constant = np.array([constant[0], 0, 0, 0])[: len(constant)]
        d = data[held]
        before = np.mean([iou(o, t) for o, t in zip(d["ours"], d["truth"], strict=True)])
        after = np.mean([hard_constant(o, constant, t, rotate) for o, t in zip(d["ours"], d["truth"], strict=True)])
        out[held] = {"before": float(before), "after": float(after), "constant": [float(x) for x in constant]}
    return out


def main():
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "analyse"])
    parser.add_argument("videos", nargs="?", default=",".join(VIDEOS))
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--out", default=None)
    parser.add_argument("--size-mode", default="current", help="GarmentRenderer.SIZE_MODE of the collected run")
    parser.add_argument("--relaxed-rest", action="store_true", help="collect with GarmentRenderer.RELAXED_REST='mean'")
    args = parser.parse_args()
    CACHE = Path(args.cache)
    names = args.videos.split(",")
    if args.command == "collect":
        for name in names:
            collect(name, args.every, args.size_mode, args.relaxed_rest)
        return
    results = [analyse_video(n) for n in names]
    for r in results:
        print(r["video"], r["frames"], {k: round(v, 4) for k, v in r["truth"].items() if k.startswith("iou")},
              {k: round(v, 4) for k, v in r["theirs"].items() if k.startswith("iou")}, flush=True)
    report = {"per_video": results}
    if len(names) > 1:
        report["pooled_heldout_video"] = pooled_heldout(names)
        report["pooled_heldout_video_scale_only"] = pooled_heldout(names, scale_only=True)
    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
