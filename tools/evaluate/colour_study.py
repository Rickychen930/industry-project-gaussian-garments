"""Why the rendered garment colours are off the real video's, and what could fix it live.

The fit audit found the jackets 19-20 levels low in R and B. This measures where
the error comes from and scores colour corrections against the real garment
pixels while the corrections themselves use only information available live.

    .venv/bin/python -m tools.evaluate.colour_study collect VIDEO [--max-frames N]   # slow: live pipeline
    .venv/bin/python -m tools.evaluate.colour_study report [videos]                  # json + figure from cache

`collect` (one video per process: memory) caches per scored frame the real panel,
SCHP labels, shirt mask, body silhouette and the garment rendered with student
shading, static colours and the 55M teacher. `report` reads the cache.
Pure helpers at the top are checked by tests/test_tools_colour_study.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tools.evaluate.fit_audit import ALPHA_MASK, CACHE as AUDIT_CACHE, ERODE_PX, VIDEOS, delta_e76, lab_of, psnr
from tryon.paths import BASELINE, GARMENTS, REPO_ROOT

CACHE = Path(AUDIT_CACHE).parent / "colour_study"
FIGURES = REPO_ROOT / "docs" / "figures"
EVERY = 10
TEACHER_EVERY = 20
MIN_REGION = 200
# ATR labels of the scene that exist live: background, face, arms (skin), legs
SCENE_BACKGROUND, SCENE_SKIN = (0,), (11, 12, 13, 14, 15)

# ============================================================== pure helpers


def straight_colours(rgba):
    """(H, W, 3) float32 BGR of the render with the premultiplication undone, and alpha in 0..1."""
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3].astype(np.float32) / np.maximum(alpha, 1e-3)[..., None], 0, 255), alpha


def overlap_region(rgba, shirt, erode=ERODE_PX):
    """Our garment mask AND the real shirt mask, eroded `erode` px, where alpha >= 0.9 (fit_audit's region)."""
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    closed = (rgba[..., 3] > ALPHA_MASK) & shirt
    return cv2.erode(closed.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)).astype(bool) & (alpha >= 0.9)


def channel_gain_offset(ours, real):
    """Least-squares per-channel (gain, offset) with real ~ gain * ours + offset ((N, 3) arrays)."""
    ours, real = np.asarray(ours, np.float64), np.asarray(real, np.float64)
    spread = ours.var(axis=0)
    covariance = ((ours - ours.mean(0)) * (real - real.mean(0))).mean(axis=0)
    gain = np.where(spread > 1e-6, covariance / np.maximum(spread, 1e-6), 1.0)
    return gain, real.mean(axis=0) - gain * ours.mean(axis=0)


def mean_gain(ours, real):
    """Per-channel ratio of the means: the multiplicative white-balance / exposure difference."""
    return np.asarray(real, np.float64).mean(axis=0) / np.maximum(np.asarray(ours, np.float64).mean(axis=0), 1e-6)


def luminance(bgr):
    """Rec. 601 luma of BGR values (..., 3)."""
    bgr = np.asarray(bgr, np.float32)
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def metrics(corrected_px, real_px):
    """dE76 (mean) and PSNR of (N, 3) uint8-range pixels against the real ones."""
    corrected = np.clip(corrected_px, 0, 255)
    return {"de76": float(delta_e76(corrected.astype(np.uint8), real_px.astype(np.uint8)).mean()),
            "psnr": psnr(corrected, real_px)}


# ============================================================== collecting (the live pipeline)


def _variant_renders(renderer, original_render, fitter, size, mesh, dt, with_teacher):
    """The same posed mesh drawn with the checkpoint's static colours and (optionally) the teacher."""
    import contextlib

    @contextlib.contextmanager
    def pinned_mesh():
        saved = renderer.posed_mesh
        renderer.posed_mesh = lambda _fitter, _dt=None: mesh
        try:
            yield
        finally:
            renderer.posed_mesh = saved

    out = {}
    with pinned_mesh():
        renderer.appearance_enabled = False
        try:
            out["static"] = original_render(fitter, size, dt=dt)
        finally:
            renderer.appearance_enabled = True
        if with_teacher:
            from tryon.garment.appearance import AppearanceNet

            student_net, student_flag = renderer._appearance_net, renderer.STUDENT_SHADING
            renderer.STUDENT_SHADING = False
            if getattr(renderer, "_teacher_net", None) is None:
                renderer._teacher_net = AppearanceNet(renderer.checkpoint_dir, faces=renderer.faces, half=True)
            renderer._appearance_net = renderer._teacher_net
            try:
                out["teacher"] = original_render(fitter, size, dt=dt)
            finally:
                renderer._appearance_net, renderer.STUDENT_SHADING = student_net, student_flag
    return out


def collect(name, every=EVERY, max_frames=0, teacher_every=TEACHER_EVERY):
    """Run the live pipeline over one clip and cache the frames the colour study needs (see the module doc)."""
    import mediapipe as mp

    from tools.evaluate import vs_registration as vr
    from tools.evaluate.fit_audit import person_masks, stop_regressor
    from tools.evaluate.reference_frame import find_panel
    from tryon.app import pipeline
    from tryon.app.person import landmarks_from
    from tryon.garment.renderer import GarmentRenderer
    from tryon.paths import RELEASE_VIDEOS
    from tryon.removal.background import body_silhouette
    from tryon.removal.parsing_schp import ATR_CHECKPOINT, LIP_CHECKPOINT, SCHPParser

    GarmentRenderer.GPU_RASTER, GarmentRenderer.STUDENT_SHADING, GarmentRenderer.APPEARANCE_ASYNC = True, True, False
    session = pipeline.LiveSession(garment=str(GARMENTS / name), orientation_regressor=str(REPO_ROOT / "models" / "pose_regressor.pt"),
                                   remove_clothing=True, appearance=True, headless=True)
    parser, lip = SCHPParser(ATR_CHECKPOINT), SCHPParser(LIP_CHECKPOINT, size=473)
    renderer = session.renderer
    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total, max_frames) if max_frames else total

    seen = {}
    original_render, original_posed = renderer.render, renderer.posed_mesh

    def posed(fitter, dt=None):
        mesh = original_posed(fitter, dt)
        seen["mesh"] = np.array(mesh, np.float64)
        return mesh

    def render(fitter, size, **kwargs):
        kwargs.setdefault("dt", 1.0 / fps)
        seen["rgba"] = original_render(fitter, size, **kwargs)
        seen["call"] = (fitter, size, kwargs["dt"])
        return seen["rgba"]

    renderer.posed_mesh, renderer.render = posed, render
    stage = lambda *_args, **_kwargs: None  # noqa: E731
    box, state, keep = [None], {"read": 0}, []

    def upcoming(pose):
        ok, full = capture.read()
        if not ok or state["read"] >= total:
            return None
        if box[0] is None:
            box[0] = find_panel(full, "left")
        y0, y1, x0, x1 = box[0]
        panel = cv2.resize(full[y0:y1, x0:x1], None, fx=vr.HEIGHT / (y1 - y0), fy=vr.HEIGHT / (y1 - y0), interpolation=cv2.INTER_AREA)
        points = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
        t = state["read"]
        state["read"] += 1
        session.prefetch(panel, points)
        return t, panel, points

    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=pipeline.POSE_COMPLEXITY) as pose:
        item = upcoming(pose)
        while item is not None:
            t, panel, points = item
            item = upcoming(pose)
            seen.clear()
            source = panel.copy()
            session.process(panel, points, t / fps, stage, source)
            rgba = seen.get("rgba")
            if t % every or rgba is None or not rgba[..., 3].any() or session.fitter is None:
                continue
            fitter, size, dt = seen["call"]
            student = rgba.copy()
            variants = _variant_renders(renderer, original_render, fitter, size, seen["mesh"], dt, with_teacher=t % teacher_every == 0)
            person, shirt = person_masks(parser, lip, source)
            keep.append({"t": t, "frame": source, "labels": parser.labels(source).astype(np.uint8), "shirt": shirt, "person": person,
                         "body": body_silhouette(session.fitter, source.shape, grow=0.0), "student": student, **variants})
            print(f"{name} frame {t}: kept {len(keep)} ({sorted(variants)})", flush=True)
    session.finish()
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack([k[key] for k in keep if key in k]) for key in ("frame", "labels", "shirt", "person", "body", "student", "static")}
    teacher = [k for k in keep if "teacher" in k]
    if teacher:
        arrays["teacher"], arrays["teacher_t"] = np.stack([k["teacher"] for k in teacher]), np.array([k["t"] for k in teacher])
    np.savez_compressed(CACHE / f"{name}.npz", t=np.array([k["t"] for k in keep]), **arrays)
    stop_regressor(session.regression)
    return len(keep)


# ============================================================== the study (from the cache)

STAGES = ("static", "student", "teacher")


def load(name):
    """The cached arrays of one video as a dict (npz members read out)."""
    with np.load(CACHE / f"{name}.npz") as cached:
        return {key: cached[key] for key in cached.files}


def frame_pixels(cache, k, stage):
    """(region mask, ours (N, 3) float BGR, real (N, 3) float BGR, straight colour image) of frame k of `stage`;
    None when the overlap is too small to score."""
    rgba = cache[stage][k] if stage != "teacher" else cache["teacher"][int(np.flatnonzero(cache["teacher_t"] == cache["t"][k])[0])]
    region = overlap_region(rgba, cache["shirt"][k])
    if region.sum() < MIN_REGION:
        return None
    colours, _ = straight_colours(rgba)
    return region, colours[region], cache["frame"][k][region].astype(np.float32), colours


def has_teacher(cache, k):
    return "teacher_t" in cache and bool((cache["teacher_t"] == cache["t"][k]).any())


def ssim_of(image, real, region):
    """Mean SSIM over the region (the fit_audit protocol: the bounding crop, the region's pixels averaged)."""
    from skimage.metrics import structural_similarity

    ys, xs = np.nonzero(region)
    window = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    if min(image[window].shape[:2]) < 7:
        return float("nan")
    _, ssim_map = structural_similarity(real[window], np.clip(image[window], 0, 255).astype(np.uint8), channel_axis=2, full=True, data_range=255)
    return float(ssim_map.mean(axis=2)[region[window]].mean())


def score(image, cache, k, region):
    """dE76 / PSNR / SSIM of a full colour image (H, W, 3 float BGR) against the real frame in `region`."""
    real = cache["frame"][k]
    out = metrics(image[region], real[region].astype(np.float32))
    out["ssim"] = ssim_of(image, real, region)
    return out


def nanmean_rows(rows, key):
    values = np.array([r[key] for r in rows if key in r], np.float64)
    return float(np.nanmean(values)) if values.size and np.isfinite(values).any() else float("nan")


def diagnose(names):
    """Study 1-3: per-channel offsets and their stability, the stage that adds the error, error vs shadow."""
    out = {"channels": {}, "stages": {}, "dark": {}}
    pooled = {stage: {"luma_diff": [], "real_luma": [], "edge": [], "body": []} for stage in ("static", "student")}
    for name in names:
        cache = load(name)
        rows = {stage: [] for stage in STAGES}
        for k in range(len(cache["t"])):
            for stage in STAGES:
                if stage == "teacher" and not has_teacher(cache, k):
                    continue
                got = frame_pixels(cache, k, stage)
                if got is None:
                    continue
                region, ours, real, colours = got
                gain, offset = channel_gain_offset(ours, real)
                row = {"k": k, "ours": ours.mean(0), "real": real.mean(0), "diff": ours.mean(0) - real.mean(0), "ratio": mean_gain(ours, real),
                       "ls_gain": gain, "ls_offset": offset, **score(colours, cache, k, region)}
                rows[stage].append(row)
                if stage in pooled:
                    edge = cv2.distanceTransform((cache[stage][k][..., 3] > ALPHA_MASK).astype(np.uint8), cv2.DIST_L2, 5)
                    body = cv2.distanceTransform((~cache["body"][k]).astype(np.uint8), cv2.DIST_L2, 5)
                    pooled[stage]["luma_diff"].append(luminance(ours) - luminance(real))
                    pooled[stage]["real_luma"].append(luminance(real))
                    pooled[stage]["edge"].append(edge[region])
                    pooled[stage]["body"].append(body[region])
        student = rows["student"]
        stack = lambda key: np.array([r[key] for r in student])  # noqa: E731
        out["channels"][name] = {
            "frames": len(student),
            "mean_diff_bgr": stack("diff").mean(0).tolist(), "mean_ratio_bgr": stack("ratio").mean(0).tolist(),
            "ratio_std_over_frames_bgr": stack("ratio").std(0).tolist(), "diff_std_over_frames_bgr": stack("diff").std(0).tolist(),
            "ls_gain_bgr": stack("ls_gain").mean(0).tolist(), "ls_gain_std_bgr": stack("ls_gain").std(0).tolist(),
            "ls_offset_bgr": stack("ls_offset").mean(0).tolist(), "ls_offset_std_bgr": stack("ls_offset").std(0).tolist(),
            "luma_ratio": float(luminance(stack("ours")).mean() / luminance(stack("real")).mean()),
            "luma_ratio_std": float((luminance(stack("ours")) / luminance(stack("real"))).std())}
        teacher_frames = {r["k"] for r in rows["teacher"]}
        out["stages"][name] = {stage: {key: nanmean_rows([r for r in rows[stage] if r["k"] in teacher_frames or stage != "teacher"], key)
                                       for key in ("de76", "psnr", "ssim")} | {"mean_diff_bgr": np.mean([r["diff"] for r in rows[stage] if r["k"] in teacher_frames], axis=0).tolist() if teacher_frames else None}
                               for stage in STAGES if rows[stage]}
        print(name, out["channels"][name]["mean_diff_bgr"], flush=True)
    for stage, data in pooled.items():
        luma_diff, real_luma = (np.concatenate(data[key]) for key in ("luma_diff", "real_luma"))
        edge, body = np.concatenate(data["edge"]), np.concatenate(data["body"])
        out["dark"][stage] = {
            "by_real_luma": _binned(luma_diff, real_luma, (0, 40, 80, 120, 160, 200, 256)),
            "by_edge_distance_px": _binned(luma_diff, edge, (0, 10, 20, 40, 80, 1e9)),
            "by_body_distance_px": _binned(luma_diff, body, (-1, 0.5, 10, 30, 1e9))}
    return out


def _binned(values, key, edges):
    """Mean (ours - real) luma and the share of pixels per bin of `key`."""
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (key >= low) & (key < high)
        rows.append({"range": [float(low), float(min(high, 1e6))], "share": float(inside.mean()),
                     "mean_luma_diff": float(values[inside].mean()) if inside.any() else float("nan")})
    return rows


# ============================================================== correction candidates

GAIN_DAMPING = 0.5  # exponent on the scene ratio
GAIN_LIMITS = (0.8, 1.25)
REINHARD_WEIGHT = 0.15
EMA_PER_SCORED_FRAME = 1.0 - 0.9**EVERY  # the live EMA of 0.9 a frame, as seen every EVERY-th frame
BLUR_SIGMAS = (0.5, 0.8, 1.2)
SHADOW_STRENGTHS = (0.0, 0.25, 0.5, 0.75)


def scene_pixels(frame, labels, garment_alpha, classes, margin=12):
    """BGR pixels of the live scene of the given ATR classes outside the garment (and `margin` px around it)."""
    free = ~cv2.dilate((garment_alpha > ALPHA_MASK).astype(np.uint8), np.ones((margin, margin), np.uint8)).astype(bool)
    return frame[np.isin(labels, classes) & free]


def scene_features(cache, k):
    """Live-available scene numbers of frame k: median luma and mean BGR of skin, of the background and of both."""
    alpha = cache["student"][k][..., 3]
    out = {}
    for key, classes in (("skin", SCENE_SKIN), ("bg", SCENE_BACKGROUND), ("both", SCENE_SKIN + SCENE_BACKGROUND)):
        pixels = scene_pixels(cache["frame"][k], cache["labels"][k], alpha, classes)
        out[key] = None if len(pixels) < MIN_REGION else {"luma": float(np.median(luminance(pixels))), "bgr": pixels.astype(np.float64).mean(0),
                                                           "lab_mean": lab_of(pixels[::7]).mean(0), "lab_std": lab_of(pixels[::7]).std(0)}
    return out


def smoothed(values, rate=EMA_PER_SCORED_FRAME):
    """EMA of a series, the first value as it is (the live smoothing, evaluated at the scored frames)."""
    out, state = [], None
    for value in values:
        state = np.asarray(value, np.float64) if state is None else rate * np.asarray(value, np.float64) + (1 - rate) * state
        out.append(state)
    return out


def scene_gain(feature, reference, damping=GAIN_DAMPING, limits=GAIN_LIMITS):
    """A damped, clipped multiplier from the ratio of the scene's median luma to the studio reference's."""
    return float(np.clip((feature["luma"] / reference) ** damping, *limits))


def reinhard(colours, region, scene, weight=REINHARD_WEIGHT):
    """Lab mean/std transfer from the scene to the garment, damped by `weight`, the garment's own statistics from its pixels."""
    lab = lab_of(np.clip(colours, 0, 255).astype(np.uint8))
    pixels = lab[region]
    mean, std = pixels.mean(0), np.maximum(pixels.std(0), 1e-3)
    moved = lab + weight * ((scene["lab_std"] / std - 1) * (lab - mean) + (scene["lab_mean"] - mean))
    moved[..., 0] = np.clip(moved[..., 0], 0, 100)
    return cv2.cvtColor(moved.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(moved.shape) * 255.0


def candidates_for(cache, features, references, picture=None):
    """{candidate: [score row per frame]} for one video; `references` are leave-one-out numbers (see `evaluate`)."""
    rows = {}
    pooled_ours, pooled_real = [], []
    frames = []
    for k in range(len(cache["t"])):
        got = frame_pixels(cache, k, "student")
        if got is not None:
            frames.append((k, *got))
            pooled_ours.append(got[1])
            pooled_real.append(got[2])
    video_gain = mean_gain(np.concatenate(pooled_ours), np.concatenate(pooled_real))
    video_affine = channel_gain_offset(np.concatenate(pooled_ours), np.concatenate(pooled_real))
    ours_all, real_all = np.concatenate(pooled_ours), np.concatenate(pooled_real)
    matrix = np.linalg.lstsq(np.c_[ours_all, np.ones(len(ours_all))], real_all, rcond=None)[0]
    luma_series = {key: smoothed([features[k][key]["luma"] if features[k][key] else np.nan for k in range(len(cache["t"]))]) for key in ("skin", "bg", "both")}
    bgr_series = {key: smoothed([features[k][key]["bgr"] if features[k][key] else (np.nan,) * 3 for k in range(len(cache["t"]))]) for key in ("skin", "bg", "both")}

    def add(label, k, image, region):
        rows.setdefault(label, []).append(score(image, cache, k, region))
        if picture is not None and k == picture["k"]:
            x, y, w, h = cv2.boundingRect((cache["student"][k][..., 3] > ALPHA_MASK).astype(np.uint8))
            picture["images"][label] = np.clip(image[y:y + h, x:x + w], 0, 255).astype(np.uint8)
            picture["real"], picture["box"] = cache["frame"][k][y:y + h, x:x + w], (x, y, w, h)

    for k, region, ours, real, colours in frames:
        add("student (live now)", k, colours, region)
        for key in ("skin", "bg", "both"):
            f = features[k][key]
            if f is None:
                add(f"scene luma gain ({key})", k, colours, region)
                add(f"scene luma gain ({key}, EMA)", k, colours, region)
                add(f"scene white balance ({key}, EMA)", k, colours, region)
                continue
            add(f"scene luma gain ({key})", k, colours * scene_gain(f, references["luma"][key]), region)
            ema_f = {"luma": luma_series[key][k]}
            add(f"scene luma gain ({key}, EMA)", k, colours * scene_gain(ema_f, references["luma"][key]), region)
            wb = np.clip((bgr_series[key][k] / np.maximum(references["bgr"][key], 1.0)) ** GAIN_DAMPING, *GAIN_LIMITS)
            add(f"scene white balance ({key}, EMA)", k, colours * wb, region)
        static_colours = frame_pixels(cache, k, "static")[3]
        add("static colours (no shading)", k, static_colours, region)
        for beta in SHADOW_STRENGTHS:  # the shading's SH change is linear in the colours: a blend of the two renders
            add(f"shadow strength {beta}", k, static_colours + beta * (colours - static_colours), region)
        if features[k]["both"] is not None:
            add("Reinhard Lab transfer (both, w=0.15)", k, reinhard(colours, region, features[k]["both"]), region)
        add("universal gain (other videos)", k, colours * references["universal_gain"], region)
        for sigma in BLUR_SIGMAS:
            add(f"blur sigma {sigma}", k, cv2.GaussianBlur(colours, (0, 0), sigma), region)
        # oracles: they use the real garment pixels, which do not exist live
        gain, offset = fit_gain_offset_moment(ours, real)
        add("ORACLE per-frame gain+offset", k, gain * colours + offset, region)
        add("ORACLE per-video gain", k, colours * video_gain, region)
        add("ORACLE per-video gain+offset", k, colours * video_affine[0] + video_affine[1], region)
        add("ORACLE per-video 3x3+offset", k, np.einsum("hwc,cd->hwd", colours, matrix[:3]) + matrix[3], region)
    return rows, video_gain


def fit_gain_offset_moment(ours, real):
    """fit_audit's per-frame gain-offset (moment matching), imported under a short name."""
    from tools.evaluate.fit_audit import fit_gain_offset

    return fit_gain_offset(ours, real)


def harmonizer_rows(cache, picture=None):
    """Score the repository's Harmonizer on the same frames: it sees the garment pasted over the frame (as
    live it sees the erased composite) and the garment mask, and its filter arguments are predicted per frame."""
    from tryon.app.harmonize import LiveHarmonizer

    rows = []
    for k in range(len(cache["t"])):
        got = frame_pixels(cache, k, "student")
        if got is None:
            continue
        region, _, _, colours = got
        rgba = cache["student"][k]
        alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)
        composite = np.clip(colours * alpha + cache["frame"][k].astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
        out = LiveHarmonizer(every=1, ema=1.0)(composite, rgba[..., 3] > ALPHA_MASK).astype(np.float32)
        rows.append(score(out, cache, k, region))
        if picture is not None and k == picture["k"]:
            x, y, w, h = picture["box"]
            picture["images"]["Harmonizer (per-frame arguments)"] = np.clip(out[y:y + h, x:x + w], 0, 255).astype(np.uint8)
    return rows


FIGURE_VIDEOS = ("00136_outer", "10006_outer", "00190_upper")


def evaluate(names, with_harmonizer=True, pictures=None):
    """Study 4: every candidate on every video; the means over videos, the per-video table."""
    caches = {name: load(name) for name in names}
    features = {name: [scene_features(cache, k) for k in range(len(cache["t"]))] for name, cache in caches.items()}
    video_luma = {n: {key: np.nanmean([f[key]["luma"] for f in features[n] if f[key]]) for key in ("skin", "bg", "both")} for n in names}
    video_bgr = {n: {key: np.nanmean([f[key]["bgr"] for f in features[n] if f[key]], axis=0) for key in ("skin", "bg", "both")} for n in names}
    gains = {}
    for n in names:  # the ratio of means each video would need: used only by the OTHER videos
        pairs = [frame_pixels(caches[n], k, "student") for k in range(len(caches[n]["t"]))]
        pairs = [p for p in pairs if p is not None]
        gains[n] = mean_gain(np.concatenate([p[1] for p in pairs]), np.concatenate([p[2] for p in pairs]))
    result, table, needed = {}, {}, {}
    for name in names:
        others = [n for n in names if n != name]
        references = {"luma": {key: float(np.mean([video_luma[n][key] for n in others])) for key in ("skin", "bg", "both")},
                      "bgr": {key: np.mean([video_bgr[n][key] for n in others], axis=0) for key in ("skin", "bg", "both")},
                      "universal_gain": np.mean([gains[n] for n in others], axis=0)}
        picture = {"k": len(caches[name]["t"]) // 2, "images": {}} if pictures is not None and name in FIGURE_VIDEOS else None
        rows, needed[name] = candidates_for(caches[name], features[name], references, picture)
        if picture is not None:
            pictures[name] = picture
        if with_harmonizer:
            rows["Harmonizer (per-frame arguments)"] = harmonizer_rows(caches[name], picture)
        for label, frame_rows in rows.items():
            table.setdefault(label, {})[name] = {key: nanmean_rows(frame_rows, key) for key in ("de76", "psnr", "ssim")}
        print(name, {label: round(table[label][name]["de76"], 2) for label in table}, flush=True)
    strengths = [f"shadow strength {b}" for b in SHADOW_STRENGTHS] + ["student (live now)"]
    table["shadow strength (leave-one-video-out)"] = {
        name: table[min(strengths, key=lambda label: np.mean([table[label][n]["de76"] for n in names if n != name]))][name] for name in names}
    for label, per_video in table.items():
        result[label] = {"per_video": per_video, "mean": {key: float(np.mean([v[key] for v in per_video.values()])) for key in ("de76", "psnr", "ssim")}}
    base = table["student (live now)"]
    for label, per_video in table.items():
        result[label]["worst_video_de76_change"] = float(max(per_video[n]["de76"] - base[n]["de76"] for n in names))
        result[label]["mean_de76_change"] = result[label]["mean"]["de76"] - float(np.mean([base[n]["de76"] for n in names]))
    result["_needed_gain_bgr"] = {n: g.tolist() for n, g in needed.items()}
    return result


def figure(pictures, labels, path, height=300):
    """Rows of crops, one per video: the real photo, then the render under each label, with the label above."""
    rows = []
    for name, picture in pictures.items():
        crops = [("real video", picture["real"])] + [(label, picture["images"][label]) for label in labels if label in picture["images"]]
        tiles = []
        for label, crop in crops:
            tile = cv2.resize(crop, (max(int(crop.shape[1] * height / crop.shape[0]), 1), height), interpolation=cv2.INTER_AREA)
            band = np.full((22, tile.shape[1], 3), 255, np.uint8)
            cv2.putText(band, label[:34], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([band, tile]))
        rows.append((name, np.hstack(tiles)))
    width = max(row.shape[1] for _, row in rows)
    sheet = np.vstack([np.hstack([row, np.full((row.shape[0], width - row.shape[1], 3), 255, np.uint8)]) for _, row in rows])
    for i, (name, _) in enumerate(rows):
        cv2.putText(sheet, name, (3, i * (height + 22) + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


FIGURE_LABELS = ("student (live now)", "static colours (no shading)", "scene luma gain (skin, EMA)", "ORACLE per-video gain")


def report(names):
    """Write docs/baseline/colour-study.json and docs/figures/colour-study.jpg from the cache."""
    names = [n for n in names if (CACHE / f"{n}.npz").exists()]
    pictures = {}
    data = {"videos": names, "diagnosis": diagnose(names), "candidates": evaluate(names, pictures=pictures)}
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure({n: pictures[n] for n in FIGURE_VIDEOS if n in pictures}, FIGURE_LABELS, FIGURES / "colour-study.jpg")
    write_json(BASELINE / "colour-study.json", data)
    print(json.dumps({label: round(v["mean"]["de76"], 2) for label, v in data["candidates"].items() if not label.startswith("_")}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("collect")
    one.add_argument("video", choices=VIDEOS)
    one.add_argument("--max-frames", type=int, default=0)
    one.add_argument("--no-teacher", action="store_true")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("videos", nargs="*", default=list(VIDEOS))
    args = parser.parse_args()
    if args.command == "collect":
        print(collect(args.video, max_frames=args.max_frames, teacher_every=10**9 if args.no_teacher else TEACHER_EVERY), "frames cached")
    else:
        report(args.videos)


if __name__ == "__main__":
    main()
