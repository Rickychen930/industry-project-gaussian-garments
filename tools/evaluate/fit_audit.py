"""Does the garment fit the person, move naturally and look right? One audit, six sides.

Drives the live pipeline (`tryon.app.pipeline.LiveSession`) frame by frame on the
release videos (left photo panel, 870 px tall as in `vs_registration`) and scores:

  A  body        PA-MPJPE on annotated views, torso reprojection, silhouette IoU,
                 height / shoulder / chest ratios vs scan and registration bodies
  B  fit in 3D   vertices inside the body, penetration, gap by region, torn faces,
                 strain; the authors' simulation is the reference
  C  silhouette  IoU vs the SCHP shirt, extra / missed split, boundary F-score,
                 Hausdorff-95
  D  motion      jitter of garment and silhouette vs the real shirt's, lag, dropouts
  E  texture     CIELAB dE76 with / without gain-offset, PSNR / SSIM, detail, holes
  F  sheet       real | composite | contour overlay, six frames per video

    .venv/bin/python -m tools.evaluate.fit_audit collect [videos]     # slow: runs the live pipeline
    .venv/bin/python -m tools.evaluate.fit_audit body                 # A1 on the annotated views
    .venv/bin/python -m tools.evaluate.fit_audit report [videos]      # json + figures from the cache

The pure metric helpers at the top are checked by `tests/test_tools_fit_audit.py`.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import iou, write_json
from tryon.paths import BASELINE, GARMENTS, REPO_ROOT, SCRATCH

VIDEOS = ("00152_outer", "00170_upper", "00134_upper", "10006_outer", "00190_upper",
          "00122_outer", "00136_outer", "00148_upper", "00176_upper")
CACHE = Path(SCRATCH) / "fit_audit"
EVERY = 10  # scored frames: every 10th
WINDOW_START, WINDOW_FRAMES = 50, 30  # consecutive frames for the motion metrics
ALPHA_MASK = 25  # vs_registration: alpha > 25 of 255 is "garment drawn"
ERODE_PX = 5  # the colour region stays this far from any edge
FIGURE_FRAMES = 6
TORSO_LANDMARKS = (11, 12, 23, 24)  # shoulders and hips
HEM_FRACTION = 0.2  # the lowest fifth of the non-sleeve rest mesh is "hem"
MAX_LAG = 5

# ============================================================== pure metric helpers


def lab_of(bgr):
    """CIELAB (L 0..100) of a uint8 BGR array of any shape (..., 3)."""
    bgr = np.asarray(bgr)
    flat = (bgr.reshape(-1, 1, 3).astype(np.float32)) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(bgr.shape)


def delta_e76(bgr_a, bgr_b):
    """CIE76 colour distance per pixel between two BGR arrays (..., 3)."""
    return np.linalg.norm(lab_of(bgr_a) - lab_of(bgr_b), axis=-1)


def boundary(mask):
    """The mask's one-pixel inner outline (outer contour and hole contours)."""
    mask = np.asarray(mask, bool)
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _distance_to(outline):
    return cv2.distanceTransform((~outline).astype(np.uint8), cv2.DIST_L2, 5)


def boundary_fscore(a, b, tolerance):
    """F-score of the outline of `a` against the outline of `b`: precision is the share of a's outline
    within `tolerance` px of b's, recall the share of b's within it of a's. Two empty masks agree (1.0)."""
    outline_a, outline_b = boundary(a), boundary(b)
    if not outline_a.any() and not outline_b.any():
        return 1.0
    if not outline_a.any() or not outline_b.any():
        return 0.0
    precision = float((_distance_to(outline_b)[outline_a] <= tolerance).mean())
    recall = float((_distance_to(outline_a)[outline_b] <= tolerance).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def hausdorff95(a, b):
    """95th-percentile Hausdorff distance (px) between the outlines of two masks; nan if either is empty."""
    outline_a, outline_b = boundary(a), boundary(b)
    if not outline_a.any() or not outline_b.any():
        return float("nan")
    return float(max(np.percentile(_distance_to(outline_b)[outline_a], 95), np.percentile(_distance_to(outline_a)[outline_b], 95)))


def second_difference_mean(series):
    """Mean norm of the second difference x[t+1] - 2 x[t] + x[t-1] of a (T, ...) series; the last axis is
    the coordinate when the series has more than one dimension (T, 2) or (T, V, 3), else the series is a scalar."""
    series = np.asarray(series, np.float64)
    if len(series) < 3:
        return float("nan")
    second = series[2:] - 2 * series[1:-1] + series[:-2]
    return float(np.abs(second).mean()) if series.ndim == 1 else float(np.linalg.norm(second, axis=-1).mean())


def penetration_stats(signed):
    """Share of points inside the body (signed distance < 0) and the depth (m) of those: mean and 95th percentile."""
    signed = np.asarray(signed, np.float64)
    inside = signed < 0
    depth = -signed[inside]
    return {"inside_share": float(inside.mean()) if signed.size else 0.0,
            "depth_mean": float(depth.mean()) if depth.size else 0.0,
            "depth_p95": float(np.percentile(depth, 95)) if depth.size else 0.0}


def gap_stats(signed, region):
    """Mean and 95th-percentile gap (m) of the points of `region` that are outside the body; nan if none."""
    outside = np.asarray(region, bool) & (np.asarray(signed) >= 0)
    if not outside.any():
        return {"mean": float("nan"), "p95": float("nan")}
    gaps = np.asarray(signed, np.float64)[outside]
    return {"mean": float(gaps.mean()), "p95": float(np.percentile(gaps, 95))}


def region_masks(rest_vertices, sleeve, hem_fraction=HEM_FRACTION):
    """(sleeve, torso, hem) boolean masks of a garment: the sleeve label, then the non-sleeve vertices
    split by height at rest (y up): the lowest `hem_fraction` of their range is the hem."""
    rest = np.asarray(rest_vertices, np.float64)
    sleeve = np.asarray(sleeve, bool)
    body = ~sleeve
    low, high = rest[body, 1].min(), rest[body, 1].max()
    fraction = (rest[:, 1] - low) / max(high - low, 1e-9)
    hem = body & (fraction < hem_fraction)
    return sleeve, body & ~hem, hem


def fit_gain_offset(ours, real):
    """Per-channel (gain, offset) so that gain * ours + offset has the mean and standard deviation of `real`
    ((N, 3) float arrays). Moment matching, not regression: unaffected by a small misregistration of the textures."""
    ours, real = np.asarray(ours, np.float64), np.asarray(real, np.float64)
    spread = ours.std(axis=0)
    gain = np.where(spread > 1e-6, real.std(axis=0) / np.maximum(spread, 1e-6), 1.0)
    return gain, real.mean(axis=0) - gain * ours.mean(axis=0)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak**2 / mse)


def laplacian_variance(gray, region):
    """Variance of the Laplacian inside `region`: the high-frequency energy, larger = sharper."""
    laplacian = cv2.Laplacian(np.asarray(gray, np.float32), cv2.CV_32F)
    return float(laplacian[np.asarray(region, bool)].var())


def best_lag(ours, real, max_lag=MAX_LAG):
    """Shift L (frames, sub-frame by a parabola through the peak) maximising the correlation of
    ours[t] with real[t - L]: L > 0 means ours comes L frames AFTER the real one. Series (T,) or (T, k),
    each overlap linearly detrended. Returns (lag, {shift: correlation})."""
    def detrended(x):
        t = np.arange(len(x))
        return x - np.stack([np.polyval(np.polyfit(t, x[:, k], 1), t) for k in range(x.shape[1])], axis=1)

    a, b = (np.asarray(v, np.float64) for v in (ours, real))
    a, b = (v[:, None] if v.ndim == 1 else v for v in (a, b))
    correlation = {}
    for shift in range(-max_lag, max_lag + 1):
        x, y = (a[shift:], b[: len(b) - shift]) if shift >= 0 else (a[:shift], b[-shift:])
        if len(x) < 6:
            continue
        x, y = detrended(x), detrended(y)
        values = [np.corrcoef(x[:, k], y[:, k])[0, 1] for k in range(a.shape[1]) if x[:, k].std() > 1e-9 and y[:, k].std() > 1e-9]
        if values:
            correlation[shift] = float(np.mean(values))
    if not correlation:
        return float("nan"), {}
    peak = max(correlation, key=correlation.get)
    lag = float(peak)
    if peak - 1 in correlation and peak + 1 in correlation:
        left, mid, right = correlation[peak - 1], correlation[peak], correlation[peak + 1]
        curve = left - 2 * mid + right
        if curve < 0:
            lag += 0.5 * (left - right) / curve
    return lag, correlation


def centroid_and_area(masks):
    """((T, 2) centroids in px, (T,) areas) of boolean masks; nan centroid for an empty mask."""
    centroids, areas = [], []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        areas.append(len(xs))
        centroids.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    return np.array(centroids), np.array(areas, np.float64)


def torso_frame(pelvis, spine, left_shoulder, right_shoulder):
    """(origin, 3x3 matrix whose columns are the lateral, up and forward axes) of the torso: up from the
    pelvis to the chest, lateral along the shoulders made orthogonal to it."""
    up = np.asarray(spine, np.float64) - pelvis
    up /= max(np.linalg.norm(up), 1e-9)
    lateral = np.asarray(right_shoulder, np.float64) - left_shoulder
    lateral -= up * float(lateral @ up)
    lateral /= max(np.linalg.norm(lateral), 1e-9)
    return np.asarray(pelvis, np.float64), np.stack([lateral, up, np.cross(lateral, up)], axis=1)


def procrustes_mpjpe(predicted, truth):
    """Mean joint error (same units) after the best rotation, scale and translation of `predicted` onto `truth`."""
    from tools.evaluate.real_fit import procrustes_align

    return float(np.linalg.norm(procrustes_align(np.asarray(predicted), np.asarray(truth)) - truth, axis=1).mean())


def nanmean(values):
    values = np.asarray(values, np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


# ============================================================== body shape (A4)

CHEST_SLAB = 0.02  # m either side of the chest joint
TORSO_JOINTS = (3, 6, 9)  # spine1..3: the vertices whose dominant joint is one of these are the torso's


def body_dimensions(vertices, model, torso_vertices):
    """Height, shoulder width and chest width (m) of an SMPL-X body in any pose: joints from the model's
    regressor, chest width = extent along the shoulder axis of the torso vertices within a slab around the
    chest joint (arms excluded by their skinning weights)."""
    vertices = np.asarray(vertices, np.float64)
    regressor = model.J_regressor.numpy().astype(np.float64)
    joints = regressor @ vertices
    left, right, chest = joints[16], joints[17], joints[9]
    axis = right - left
    shoulder = float(np.linalg.norm(axis))
    axis /= max(shoulder, 1e-9)
    near = torso_vertices & (np.abs(vertices[:, 1] - chest[1]) < CHEST_SLAB)
    along = (vertices[near] - chest) @ axis
    return {"height": float(np.ptp(vertices[:, 1])), "shoulder_width": shoulder,
            "chest_width": float(np.ptp(along)) if near.any() else float("nan")}


def shape_report(betas_by_source, registration_vertices):
    """Dimensions of the zero-pose body of each betas vector, and of the registration body as it stands."""
    import torch

    from tryon.body.fitter import BodyFitter

    fitter = BodyFitter()
    weights = fitter.model.lbs_weights.detach().numpy()
    torso = np.isin(weights.argmax(axis=1), TORSO_JOINTS)
    out = {}
    for source, betas in betas_by_source.items():
        with torch.no_grad():
            fitter.betas[:] = torch.tensor(np.asarray(betas, np.float32)).reshape(fitter.betas.shape)
            fitter.body_pose.zero_()
            fitter.global_orient.zero_()
        out[source] = body_dimensions(fitter.vertices(), fitter.model, torso)
    if registration_vertices is not None:
        out["registration"] = body_dimensions(registration_vertices, fitter.model, torso)
    return out


# ============================================================== one frame's numbers (collect)


def signed_distance(points, body_vertices, body_faces):
    """Signed distance (m) of `points` to the body mesh (negative inside), by open3d's ray-cast scene."""
    import open3d as o3d

    from tryon.garment.raycast import build_scene

    scene = build_scene([(body_vertices, body_faces)])
    return scene.compute_signed_distance(o3d.core.Tensor(np.ascontiguousarray(points, np.float32))).numpy().astype(np.float64)


def fit_numbers(mesh, body, body_faces, regions, strain_edges, strain_rest, torn):
    """Section B for one posed garment mesh on one body (lengths in mm)."""
    signed = signed_distance(mesh, body, body_faces)
    sleeve, torso, hem = regions
    penetration = penetration_stats(signed)
    out = {"inside_share": penetration["inside_share"], "pen_mean_mm": 1000 * penetration["depth_mean"],
           "pen_p95_mm": 1000 * penetration["depth_p95"], "inside_1cm_share": float((signed < -0.01).mean())}
    for key, region in (("sleeve", sleeve), ("torso", torso), ("hem", hem)):
        gap = gap_stats(signed, region)
        out[f"gap_{key}_mm"], out[f"gap_{key}_p95_mm"] = 1000 * gap["mean"], 1000 * gap["p95"]
        out[f"inside_{key}_share"] = float((signed[region] < 0).mean()) if region.any() else float("nan")
    if strain_edges is not None:
        length = np.linalg.norm(mesh[strain_edges[:, 1]] - mesh[strain_edges[:, 0]], axis=1)
        strain = length / np.maximum(np.linalg.norm(strain_rest[strain_edges[:, 1]] - strain_rest[strain_edges[:, 0]], axis=1), 1e-9) - 1
        out.update(strain_max=float(strain.max()), strain_p99=float(np.percentile(strain, 99)), strain_mean=float(strain.mean()))
    out["torn_share"] = float(torn.mean())
    return out


def torso_reprojection(fitter, raw, shirt_height):
    """Mean distance (px, and % of the shirt's height) between MediaPipe's visible shoulders / hips and the
    fitted body's own shoulder / hip joints in the image."""
    import torch

    from tryon.body.landmarks import JOINT_MAP

    with torch.no_grad():
        pixels = fitter._project(fitter._forward().joints[0, fitter._smplx_indices]).numpy()
    errors = [float(np.linalg.norm(pixels[k] - raw[index, :2])) for k, index in enumerate(sorted(JOINT_MAP))
              if index in TORSO_LANDMARKS and raw[index, 2] >= 0.3]
    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(errors)), 100 * float(np.mean(errors)) / max(shirt_height, 1)


def silhouette_numbers(ours, shirt, person, body_mask, reg_person):
    """Sections A3 and C for one frame: IoUs of the body and of the garment, the placement_gap split of the
    error, boundary F-scores and Hausdorff-95 (% of the shirt's height)."""
    from tools.evaluate.placement_gap import decompose

    ys = np.nonzero(shirt.any(axis=1))[0]
    height = float(ys.max() - ys.min() + 1) if len(ys) else 1.0
    out = {"iou": iou(ours, shirt), "f_2px": boundary_fscore(ours, shirt, 2), "f_5px": boundary_fscore(ours, shirt, 5),
           "hd95_px": hausdorff95(ours, shirt), "shirt_height_px": height,
           "body_iou_person": iou(body_mask, person), "body_precision": float((body_mask & person).sum() / max(body_mask.sum(), 1)),
           "bodygarment_iou_person": iou(body_mask | ours, person),
           "bodygarment_iou_registration": iou(body_mask | ours, reg_person),
           "registration_alignment_iou": iou(person, reg_person)}
    out["hd95_pct"] = 100 * out["hd95_px"] / height
    split = decompose(ours, shirt) if ours.any() and shirt.any() else None
    if split:
        out.update({f"split_{key}": value for key, value in split.items()})
    return out


def colour_numbers(rgba, source, shirt):
    """Section E for one frame: our render against the real pixels inside the eroded overlap of the two
    garment masks. Our rasteriser's colours are premultiplied; they are divided by alpha first."""
    from skimage.metrics import structural_similarity

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    ours25 = rgba[..., 3] > ALPHA_MASK
    out = {}
    closed = ours25 & shirt
    region = cv2.erode(closed.astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)).astype(bool) & (alpha >= 0.9)
    out["region_px"] = int(region.sum())
    if region.sum() < 200:
        return out
    colours = np.clip(rgba[..., :3].astype(np.float32) / np.maximum(alpha, 1e-3)[..., None], 0, 255)
    ours_px, real_px = colours[region], source[region].astype(np.float32)
    out["de76_raw"] = float(delta_e76(ours_px.astype(np.uint8), real_px.astype(np.uint8)).mean())
    mean_lab = float(np.linalg.norm(lab_of(ours_px.astype(np.uint8)).mean(axis=0) - lab_of(real_px.astype(np.uint8)).mean(axis=0)))
    out["de76_of_mean_colour"] = mean_lab
    for channel, name in enumerate("bgr"):
        out[f"mean_diff_{name}"] = float(ours_px[:, channel].mean() - real_px[:, channel].mean())
    out["luminance_ratio"] = float(ours_px.mean() / max(real_px.mean(), 1e-6))
    gain, offset = fit_gain_offset(ours_px, real_px)
    corrected_px = np.clip(gain * ours_px + offset, 0, 255)
    out["de76_corrected"] = float(delta_e76(corrected_px.astype(np.uint8), real_px.astype(np.uint8)).mean())
    out["gain_mean"], out["offset_mean"] = float(gain.mean()), float(offset.mean())
    out["psnr_raw"], out["psnr_corrected"] = psnr(ours_px, real_px), psnr(corrected_px, real_px)
    ys, xs = np.nonzero(region)
    window = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    for key, image in (("raw", colours.astype(np.uint8)), ("corrected", None)):
        crop = image[window].copy() if image is not None else colours[window].copy()
        if image is None:
            crop[region[window]] = corrected_px.astype(np.float32)
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        if min(crop.shape[:2]) >= 7:
            _, ssim_map = structural_similarity(source[window], crop, channel_axis=2, full=True, data_range=255)
            out[f"ssim_{key}"] = float(ssim_map.mean(axis=2)[region[window]].mean())
    gray_ours = cv2.cvtColor(colours.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_real = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    real_energy = laplacian_variance(gray_real, region)
    out["laplacian_ratio"] = laplacian_variance(gray_ours, region) / real_energy if real_energy > 0 else float("nan")
    return out


def edge_numbers_of(rgba):
    """Section E4 as `silhouette_study` defines it: mean |grad alpha| across the boundary of the alpha > 25
    mask, the width of its 0.1-0.9 transition, and the share of its filled interior with alpha < 0.9."""
    from tools.evaluate.silhouette_study import HOLE_ALPHA, edge_numbers, interior_region

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    mask = rgba[..., 3] > ALPHA_MASK
    if not mask.any():
        return {}
    gradient, width = edge_numbers(alpha, mask)
    region = interior_region(alpha)
    return {"edge_gradient": gradient, "edge_width_px": width,
            "holes": float((alpha[region] < HOLE_ALPHA).mean()) if region.any() else 0.0}


# ============================================================== driving the live pipeline


def reference_profile(renderer, name):
    """The authors' simulation on their registration body (trajectory frames), section B's reference:
    the same numbers `fit_numbers` gives, on `pred[k]` against `obstacle[k]`, plus the registration
    body itself (frame 0) for the size comparison. (None, None) without a trajectory file."""
    from tools.data.fit_scan_pose import trajectory_for
    from tools.evaluate.cloth_trajectory import reference_vertex_map

    path = trajectory_for(GARMENTS / name)
    if path is None:
        return None, None
    with open(path, "rb") as handle:
        trajectory = pickle.load(handle)
    to_ref, count = reference_vertex_map(renderer)
    sleeve = np.zeros(count, bool)
    sleeve[to_ref[renderer.sleeve_vertices]] = True
    rest = np.zeros((count, 3))
    rest[to_ref] = renderer.rest_vertices
    regions = region_masks(rest, sleeve)
    faces = np.asarray(trajectory["obstacle_faces"])
    frames = [k for k in (0, 40, 80, 120, 159) if k < len(trajectory["pred"])]
    rows = [fit_numbers(np.asarray(trajectory["pred"][k], np.float64), np.asarray(trajectory["obstacle"][k], np.float64), faces,
                        regions, None, None, np.zeros(1)) for k in frames]
    keys = [key for key in rows[0] if key != "torn_share"]
    mean = {key: nanmean([row[key] for row in rows]) for key in keys}
    mean["frames"] = frames
    return mean, np.asarray(trajectory["obstacle"][0], np.float64)


def person_masks(parser, lip, panel):
    """(person, real shirt) of the photo panel exactly as `vs_registration` makes them."""
    labels = parser.labels(panel)
    person, shirt = labels != 0, np.isin(labels, (4, 7))
    lip_labels = lip.labels(panel)
    coat, inner = lip_labels == 7, np.isin(lip_labels, (5, 10))
    if coat.sum() >= 0.5 * max(int((coat | inner).sum()), 1):
        shirt &= ~inner
    return person, shirt


def stop_regressor(regression):
    """Close the NLF worker; one still answering a prefetched frame nobody read is killed."""
    if regression is None or not hasattr(regression, "close"):
        return
    try:
        regression.close()
    except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired or a closed pipe: the worker is stuck mid-answer
        regression._process.kill()


def collect(name, every=EVERY, max_frames=0):
    """Run the live pipeline over the clip and write the cache: per-scored-frame numbers, the window arrays
    of the motion metrics, the six sheet frames."""
    import mediapipe as mp

    from tools.evaluate import vs_registration as vr
    from tools.evaluate.reference_frame import find_panel
    from tryon.app import pipeline
    from tryon.app.person import landmarks_from
    from tryon.garment.renderer import GarmentRenderer
    from tryon.paths import RELEASE_VIDEOS
    from tryon.removal.background import body_silhouette
    from tryon.removal.parsing_schp import ATR_CHECKPOINT, LIP_CHECKPOINT, SCHPParser

    # the launcher's flags: --gpu-render --appearance --student-shading (synchronous here: every frame is
    # shaded for its own pose, no worker lag); SmoothCloth is the class default
    GarmentRenderer.GPU_RASTER, GarmentRenderer.STUDENT_SHADING, GarmentRenderer.APPEARANCE_ASYNC = True, True, False
    session = pipeline.LiveSession(garment=str(GARMENTS / name), orientation_regressor=str(REPO_ROOT / "models" / "pose_regressor.pt"),
                                   remove_clothing=True, appearance=True, headless=True)
    parser, lip = SCHPParser(ATR_CHECKPOINT), SCHPParser(LIP_CHECKPOINT, size=473)
    renderer = session.renderer
    from tryon.garment.shading_student import ShadingStudent

    student_used = bool(GarmentRenderer.STUDENT_SHADING and ShadingStudent.exists(GARMENTS / name))
    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total, max_frames) if max_frames else total
    window = range(WINDOW_START, WINDOW_START + WINDOW_FRAMES)

    seen = {}
    original_render, original_posed = renderer.render, renderer.posed_mesh

    def posed(fitter, dt=None):
        mesh = original_posed(fitter, dt)
        seen["mesh"] = np.array(mesh, np.float64)
        return mesh

    def render(fitter, size, **kwargs):
        kwargs.setdefault("dt", 1.0 / fps)  # the clip's own interval, not the wall clock of a loaded machine
        seen["rgba"] = original_render(fitter, size, **kwargs)
        return seen["rgba"]

    renderer.posed_mesh, renderer.render = posed, render
    stage = lambda *_args, **_kwargs: None  # noqa: E731

    box, state = [None], {"read": 0}

    def upcoming(pose):
        ok, full = capture.read()
        if not ok or state["read"] >= total:
            return None
        if box[0] is None:
            box[0] = find_panel(full, "left")
        y0, y1, x0, x1 = box[0]
        scale = vr.HEIGHT / (y1 - y0)
        panel = cv2.resize(full[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
        t = state["read"]
        state["read"] += 1
        session.prefetch(panel, points)
        return t, full, panel, points

    dressed, records, sheet, betas = [], [], [], []
    windowed = {"ours": {}, "shirt": {}, "local": {}, "skin": {}}
    similarity = None
    static = {}
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=pipeline.POSE_COMPLEXITY) as pose:
        item = upcoming(pose)
        while item is not None:
            t, full, panel, points = item
            item = upcoming(pose)  # the next frame's regression runs while this one is drawn, as live
            seen.clear()
            source = panel.copy()
            status, _reprojection, garment_alpha = session.process(panel, points, t / fps, stage, source)
            rgba = seen.get("rgba")
            is_dressed = garment_alpha is not None and rgba is not None and bool(rgba[..., 3].any())
            dressed.append(is_dressed)
            scored = t % every == 0 or t in window
            if not scored:
                continue
            person, shirt = person_masks(parser, lip, source)
            y0, y1, x0, x1 = box[0]
            scale = vr.HEIGHT / (y1 - y0)
            region = cv2.resize(full[y0:y1, x1:], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            reg_person, reg_garment = vr.registration_masks(region)
            if similarity is None:
                similarity = vr.fit_similarity(reg_person, person)
            reg_person_w = vr.warp(reg_person, *similarity[1:], person.shape)
            fitter = session.fitter
            ours = (rgba[..., 3] > ALPHA_MASK) if is_dressed else np.zeros_like(shirt)
            record = {"frame": t, "dressed": is_dressed, "status": status}
            if fitter is not None:
                body_mask = body_silhouette(fitter, panel.shape, grow=0.0)
                record.update(silhouette_numbers(ours, shirt, person, body_mask, reg_person_w))
                if points is not None:
                    record["torso_px"], record["torso_pct"] = torso_reprojection(fitter, points, record["shirt_height_px"])
                betas.append(fitter.betas.detach().numpy().ravel().copy())
            else:
                record.update(silhouette_numbers(ours, shirt, person, np.zeros_like(shirt), reg_person_w))
            if is_dressed:
                body, faces = fitter.vertices().astype(np.float64), np.asarray(fitter.faces)
                if "regions" not in static:
                    static["regions"] = region_masks(renderer.rest_vertices, renderer.sleeve_vertices)
                binding = renderer._binding
                record.update(fit_numbers(seen["mesh"], body, faces, static["regions"], renderer.scan_binder.edges,
                                          binding.physical_rest, renderer.torn_faces(seen["mesh"])))
                record.update(colour_numbers(rgba, source, shirt))
                record.update(edge_numbers_of(rgba))
                if t in window:
                    joints = fitter.joints(indices=[0, 9, 16, 17]).detach().numpy().astype(np.float64)
                    origin, axes = torso_frame(joints[0], joints[1], joints[2], joints[3])
                    windowed["local"][t] = ((seen["mesh"] - origin) @ axes).astype(np.float32)
                    windowed["skin"][t] = ((np.asarray(binding.apply(fitter), np.float64) - origin) @ axes).astype(np.float32)
            if t in window:
                windowed["ours"][t], windowed["shirt"][t] = ours, shirt
            if t % every == 0:
                sheet.append({"t": t, "dressed": is_dressed, "iou": record["iou"], "panel": source, "composite": panel.copy(),
                              "alpha": rgba[..., 3].copy() if is_dressed else np.zeros(shirt.shape, np.uint8), "shirt": shirt,
                              "person": person})
            records.append(record)
            print(f"{name} frame {t}: dressed {is_dressed} IoU {record['iou']:.3f}", flush=True)
    session.finish()

    body_dims = None
    reference, registration_body = reference_profile(renderer, name)
    scan = dict(np.load(GARMENTS / name / "stage1" / "scan_pose.npz")) if (GARMENTS / name / "stage1" / "scan_pose.npz").exists() else None
    sources = {"fitted": np.median(betas, axis=0)} if betas else {}
    if scan is not None:
        sources["scan"] = scan["betas"]
    if sources:
        body_dims = shape_report(sources, registration_body)
        body_dims["garment_size_applied"] = [float(x) for x in np.ravel(getattr(renderer, "_size_applied", [np.nan]))]
    CACHE.mkdir(parents=True, exist_ok=True)
    keep = [s for s in sheet if s["dressed"]]
    picks = [keep[int(i)] for i in np.linspace(0, len(keep) - 1, min(FIGURE_FRAMES, len(keep))).round()] if keep else []
    np.savez_compressed(CACHE / f"{name}_sheet.npz", t=[s["t"] for s in picks], **{
        key: np.stack([s[key] for s in picks]) for key in ("panel", "composite", "alpha", "shirt", "person")} if picks else {})
    order = sorted(windowed["ours"])
    np.savez_compressed(CACHE / f"{name}_window.npz", t=order, ours=np.stack([windowed["ours"][t] for t in order]) if order else [],
                        shirt=np.stack([windowed["shirt"][t] for t in order]) if order else [],
                        local=np.stack([windowed["local"][t] for t in order if t in windowed["local"]]) if windowed["local"] else [],
                        skin=np.stack([windowed["skin"][t] for t in order if t in windowed["skin"]]) if windowed["skin"] else [],
                        local_t=sorted(windowed["local"]))
    write_json(CACHE / f"{name}.json", {"video": name, "fps": fps, "frames_total": len(dressed), "dressed": dressed, "records": records,
                                        "body_dimensions": body_dims, "reference": reference, "student_shading": student_used,
                                        "alignment_iou_frame0": similarity[0] if similarity else None})
    stop_regressor(session.regression)
    return len(records)


# ============================================================== A1: the annotated views


def annotated_root():
    """data/annotated of this checkout or of the repository it is a worktree of."""
    for base in (REPO_ROOT, *REPO_ROOT.parents):
        if (base / "data" / "annotated").is_dir():
            return base / "data" / "annotated"
    raise FileNotFoundError("data/annotated not found in the checkout or any parent directory")


def body_accuracy():
    """PA-MPJPE (mm) of the live body path - MediaPipe keypoints, NLF-S regression, `fit_from_regression` -
    on the 16 annotated 4D-DRESS photographs (8 take / camera views, 2 frames each) against their real SMPL-X
    ground truth, `real_fit`'s protocol (the 12 JOINT_MAP joints, Procrustes-aligned). NLF gets its box from
    MediaPipe and a 60-degree field of view, as live; and, as a second row, the true intrinsics."""
    import torch

    from tools.evaluate import real_fit
    from tryon.body.fitter import BodyFitter
    from tryon.body.landmarks import JOINT_MAP
    from tryon.body.regressor import NLF

    real_fit.DATA = annotated_root()
    nlf = NLF()
    order = sorted(JOINT_MAP)
    rows = []
    for take in ("Take1", "Take10"):
        for cam in real_fit.CAMERAS:
            for frame in ("00001", "00002"):
                image = cv2.imread(str(real_fit.DATA / take / cam / "rgb_images" / f"{frame}.png"))
                points = real_fit.landmarks_from_mediapipe(image)
                row = {"view": f"{take}/{cam}/{frame}", "cam": cam}
                if points is None:
                    rows.append({**row, "detected": False})
                    continue
                truth = real_fit.ground_truth_joints(take, frame)[[JOINT_MAP[i] for i in order]]
                intrinsics, _ = real_fit.load_camera(take, cam)
                for label, k in (("live_fov", None), ("true_intrinsics", intrinsics.astype(np.float32))):
                    prediction = nlf.predict(image, intrinsics=k, keypoints=points)
                    fitter = BodyFitter()
                    fitter.fit_from_regression(points, prediction)
                    with torch.no_grad():
                        row[f"mpjpe_mm_{label}"] = 1000 * procrustes_mpjpe(fitter.joints().numpy(), truth)
                rows.append({**row, "detected": True})
                print(row, flush=True)
    stop_regressor(nlf)
    found = [r for r in rows if r["detected"]]
    out = {"views": len(rows), "detected": len(found), "protocol": "MediaPipe(static, complexity 2) -> NLF-S (box from MediaPipe) -> fit_from_regression -> PA-MPJPE of the 12 JOINT_MAP joints",
           "rows": rows}
    for label in ("live_fov", "true_intrinsics"):
        out[f"pa_mpjpe_mm_{label}"] = float(np.mean([r[f"mpjpe_mm_{label}"] for r in found]))
        out[f"by_camera_mm_{label}"] = {cam: float(np.mean([r[f"mpjpe_mm_{label}"] for r in found if r["cam"] == cam])) for cam in real_fit.CAMERAS}
    recorded = json.loads((BASELINE / "body-regressors.json").read_text())
    out["recorded_docs_baseline"] = {key: recorded[key]["pa_mpjpe_mm_params_in_bodyfitter"] for key in ("nlf_s_cpu", "nlf_s_mps", "nlf_l_cpu", "nlf_l_mps", "multihmr_672_S_cpu")}
    CACHE.mkdir(parents=True, exist_ok=True)
    write_json(CACHE / "body.json", out)
    print({k: v for k, v in out.items() if k != "rows"})
    return out


# ============================================================== report


def _mean_of(records, key):
    return nanmean([r.get(key, float("nan")) for r in records])


BY_FRAME = ("torso_px", "torso_pct", "body_iou_person", "body_precision", "bodygarment_iou_person", "bodygarment_iou_registration",
            "registration_alignment_iou", "iou", "f_2px", "f_5px", "hd95_pct", "split_area_ratio", "split_extra", "split_missed",
            "split_torso_width_ratio", "split_top_offset", "split_hem_offset",
            "inside_share", "pen_mean_mm", "pen_p95_mm", "inside_1cm_share", "gap_sleeve_mm", "gap_torso_mm", "gap_hem_mm",
            "gap_sleeve_p95_mm", "gap_torso_p95_mm", "gap_hem_p95_mm", "inside_sleeve_share", "inside_torso_share", "inside_hem_share",
            "strain_max", "strain_p99", "torn_share", "de76_raw", "de76_of_mean_colour", "de76_corrected", "psnr_raw", "psnr_corrected",
            "ssim_raw", "ssim_corrected", "laplacian_ratio", "luminance_ratio", "mean_diff_b", "mean_diff_g", "mean_diff_r",
            "edge_gradient", "edge_width_px", "holes") + tuple(
    f"split_{kind}@{region}" for kind in ("extra", "missed") for region in ("torso_upper", "torso_mid", "hem", "sleeve_left", "sleeve_right"))


def motion_numbers(name):
    """Section D from the cached 30-frame window; {} if the clip had none."""
    path = CACHE / f"{name}_window.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    if data["ours"].ndim < 3 or len(data["ours"]) < 8:
        return {}
    ours, shirt = data["ours"].astype(bool), data["shirt"].astype(bool)
    out = {"window_frames": int(len(ours))}
    centroid_ours, area_ours = centroid_and_area(ours)
    centroid_real, area_real = centroid_and_area(shirt)
    live = np.isfinite(centroid_ours).all(axis=1)
    if live.all():
        out["centroid_jitter_ours_px"] = second_difference_mean(centroid_ours)
        out["centroid_jitter_real_px"] = second_difference_mean(centroid_real)
        out["area_jitter_ours_pct"] = 100 * second_difference_mean(area_ours) / area_ours.mean()
        out["area_jitter_real_pct"] = 100 * second_difference_mean(area_real) / area_real.mean()
        out["centroid_jitter_ratio"] = out["centroid_jitter_ours_px"] / max(out["centroid_jitter_real_px"], 1e-9)
        out["area_jitter_ratio"] = out["area_jitter_ours_pct"] / max(out["area_jitter_real_pct"], 1e-9)
        out["frame_iou_ours"] = nanmean([iou(ours[i], ours[i + 1]) for i in range(len(ours) - 1)])
        out["frame_iou_real"] = nanmean([iou(shirt[i], shirt[i + 1]) for i in range(len(shirt) - 1)])
        lag, correlation = best_lag(centroid_ours, centroid_real)
        out["lag_frames"], out["lag_peak_correlation"] = lag, max(correlation.values()) if correlation else float("nan")
    if len(data["local"]) >= 8 and data["local"].ndim == 3:
        out["vertex_jitter_cm"] = 100 * second_difference_mean(data["local"].astype(np.float64))
        out["vertex_jitter_skin_target_cm"] = 100 * second_difference_mean(data["skin"].astype(np.float64))
        out["vertex_jitter_cloth_over_skin"] = out["vertex_jitter_cm"] / max(out["vertex_jitter_skin_target_cm"], 1e-9)
    return out


def video_metrics(name):
    """The flat dict of one video's audit from its cache, or None."""
    from tools.evaluate.dropouts import summary

    path = CACHE / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    scored = [r for r in data["records"] if r["frame"] % EVERY == 0]
    dressed = [r for r in scored if r["dressed"]]
    out = {"scored_frames": len(scored), "dressed_scored_frames": len(dressed), "clip_frames": data["frames_total"]}
    for key in BY_FRAME:
        out[key] = _mean_of(dressed, key)
    out["iou_dressed"] = out.pop("iou")
    out["iou_all_frames"] = nanmean([r["iou"] if r["dressed"] else 0.0 for r in scored])
    flags = data["dressed"]
    drop = summary(flags)
    out["dropout_share"], out["dropout_longest_run"] = 1 - drop["dressed"], drop["longest_gap"]
    dims = data.get("body_dimensions") or {}
    for other in ("scan", "registration"):
        for quantity in ("height", "shoulder_width", "chest_width"):
            if "fitted" in dims and other in dims:
                out[f"fitted_over_{other}_{quantity}"] = dims["fitted"][quantity] / dims[other][quantity]
    for source in ("fitted", "scan", "registration"):
        for quantity in ("height", "shoulder_width", "chest_width"):
            if source in dims:
                out[f"dim_{source}_{quantity}_m"] = dims[source][quantity]
    if dims.get("garment_size_applied"):
        out["garment_size_applied"] = float(np.mean(dims["garment_size_applied"]))
    for key, value in (data.get("reference") or {}).items():
        if key != "frames":
            out[f"ref_{key}"] = value
    out.update(motion_numbers(name))
    out["student_shading"] = bool(data.get("student_shading"))
    return out


def shading_numbers(names):
    """E3: the student against the 55M teacher, from docs/baseline/shading-students.json (not retrained)."""
    recorded = json.loads((BASELINE / "shading-students.json").read_text())
    rows = {n: {"student_db": recorded[n]["unseen_motions"]["live"], "no_shading_db": recorded[n]["unseen_motions"]["static"],
                "student_heldout_db": recorded[n]["held_out"]["live"], "no_shading_heldout_db": recorded[n]["held_out"]["static"]}
            for n in names if n in recorded}
    rows["mean"] = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))} if rows else {}
    return rows


def sheet(name, path):
    """Section F: six frames, each real | our composite | contours (ours red, the real shirt green), cropped to the person."""
    file = CACHE / f"{name}_sheet.npz"
    if not file.exists():
        return False
    data = np.load(file)
    triplets, cell_height = [], 400
    for i, t in enumerate(data["t"]):
        panel, composite = data["panel"][i], data["composite"][i]
        ours, shirt, person = data["alpha"][i] > ALPHA_MASK, data["shirt"][i], data["person"][i]
        ys, xs = np.nonzero(person | ours)
        margin = int(0.04 * panel.shape[0])
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, panel.shape[0])
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, panel.shape[1])
        overlay = panel.copy()
        cv2.drawContours(overlay, cv2.findContours(shirt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 2)
        cv2.drawContours(overlay, cv2.findContours(ours.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 0, 255), 2)
        cells = []
        for image in (panel, composite, overlay):
            crop = image[y0:y1, x0:x1]
            cells.append(cv2.resize(crop, (int(crop.shape[1] * cell_height / crop.shape[0]), cell_height), interpolation=cv2.INTER_AREA))
        label = f"{name} f{t} IoU {iou(ours, shirt):.2f}"
        cv2.putText(cells[0], label, (6, 20), 0, 0.5, (255, 255, 255), 3)
        cv2.putText(cells[0], label, (6, 20), 0, 0.5, (0, 0, 0), 1)
        triplets.append(cv2.copyMakeBorder(np.hstack(cells), 0, 6, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    rows = []
    for i in range(0, len(triplets), 2):
        row = np.hstack(triplets[i:i + 2])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet_image = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
    if sheet_image.shape[1] > 1600:
        scale = 1600 / sheet_image.shape[1]
        sheet_image = cv2.resize(sheet_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def report(names, out, figures):
    """docs/baseline/fit-audit.json (every video and the mean) and the sheets, from the cache."""
    per_video = {}
    for name in names:
        metrics = video_metrics(name)
        if metrics is not None:
            per_video[name] = metrics
            sheet(name, Path(figures) / f"fit-audit-{name}.jpg")
    keys = sorted({k for m in per_video.values() for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    mean = {k: nanmean([m.get(k, float("nan")) for m in per_video.values()]) for k in keys}
    body = json.loads((CACHE / "body.json").read_text()) if (CACHE / "body.json").exists() else None
    result = {"protocol": {"pipeline": "tryon.app.pipeline.LiveSession, NLF body, GL render, SmoothCloth, synchronous student shading, SCHP clothing removal; left panel at 870 px",
                           "scored_frames": f"every {EVERY}th frame of the clip for A-C and E; D on {WINDOW_FRAMES} consecutive frames from frame {WINDOW_START}",
                           "truth": "SCHP ATR shirt (4, 7) minus the LIP inner layer, as vs_registration"},
              "videos": per_video, "mean": mean, "A1_annotated_views": body, "E3_shading_students": shading_numbers(per_video)}
    write_json(out, result)
    print(json.dumps({"mean": {k: round(v, 4) for k, v in mean.items()}}, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "body", "report"])
    parser.add_argument("videos", nargs="?", default=",".join(VIDEOS))
    parser.add_argument("--every", type=int, default=EVERY)
    parser.add_argument("--frames", type=int, default=0, help="collect: stop after this many frames (smoke test)")
    parser.add_argument("--out", default=str(BASELINE / "fit-audit.json"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "docs" / "figures"))
    args = parser.parse_args()
    names = [n for n in args.videos.split(",") if n]
    if args.command == "collect":
        for name in names:
            print(name, "scored", collect(name, args.every, args.frames), "frames", flush=True)
    elif args.command == "body":
        body_accuracy()
    else:
        report(names, args.out, args.figures)


if __name__ == "__main__":
    main()
