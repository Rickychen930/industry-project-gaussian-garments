"""Where is the quality lost against the reference? A three-level ablation on
ONE ground-truth frame of the checkpoint's own release video.

    .venv/bin/python -m tools.evaluate.reference_frame ~/tryon-data/garments/00152_outer

The release video (`~/tryon-data/release_videos/<name>.mp4`) shows the
real 4D-DRESS camera image ("Ground Truth" panel) next to the authors'
registration. The trajectory file (`individual-garments/<name>_Take<N>.pkl`)
holds their SMPL-X body (`obstacle[k]`) and their ContourCraft cloth
(`pred[k]`) in the scan world, and `stage1/cameras.json` the four
4D-DRESS cameras (OpenCV `[R|t]`, intrinsics for the 940x1280 release
images - the same convention `tryon/splat.render_splats` takes). So one
frame can be rendered through our pipeline at three levels, each
replacing one more stage with ours, and compared with the real photo:

  L1  their body + their cloth, OUR renderer   (pred[k] through
      `GarmentRenderer.splats` + `render_splats`; static SH, and with the
      checkpoint's appearance network)
  L2  their body, OUR cloth                     (`GarmentRenderer` bound and
      posed on a `BodyFitter` set to the cached SMPL-X fit of obstacle[k];
      rigid LBS and the cloth solver)
  L3  OUR tracking, OUR cloth                   (MediaPipe on the photo ->
      `BodyFitter.fit` -> the live path, `GarmentRenderer.render`)

The GT panel is a zoomed crop of the camera image, so the panel's pixel
grid is `s * K + (ox, oy)` for an unknown similarity: it is fitted by
maximising the silhouette IoU of the reference's own body + cloth
(obstacle[k] and pred[k]) projected with each camera against the SCHP
parser's person mask, and the camera with the best IoU is the one the
panel shows. The same IoU over the trajectory frames around k checks
which trajectory frame the video frame is (video frame 0 = trajectory
frame 0 on 00152_outer: 0.845, falling monotonically with k).

Metrics per level, over the garment region (union of the parser's shirt
mask on the photo and the render's alpha > 25): PSNR on that region,
SSIM and DISTS on the region's crop resized to 256x341 (TryOffDiff's
evaluation size; DISTS only when torchvision's VGG16 weights are already
cached - they are 528 MB and this tool does not download), the
silhouette IoU of alpha > 25 against the shirt mask, and the body
silhouette IoU of the level's body against the person mask. Each level
is composited over the photo directly (the real shirt underneath, as the
reference's own video implicitly does) and over the photo with the
shirt erased by `ClothingRemover` (SCHP + LaMa).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import iou as _iou
from tools.evaluate._common import write_json
from tools.evaluate.cloth_trajectory import REFERENCE_FPS, reference_vertex_map, set_fitter
from tools.data.fit_scan_pose import trajectory_for
from tryon.garment.raycast import occluded_from as _occluded_from
from tryon.paths import GARMENTS, RELEASE_VIDEOS, SCRATCH

ALPHA_FLOOR = 25  # alpha above which a garment pixel counts as drawn
EVAL_SIZE = (256, 341)  # (width, height) TryOffDiff evaluates at
SHIRT_CLASSES = (4, 7)  # ATR upper-clothes, dress
OCCLUSION_MARGIN = 0.03  # metres, as GarmentRenderer.occluded_by_body
VGG16_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "vgg16-397923af.pth"


# --- the ground-truth panel --------------------------------------------------


def read_video_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {path}")
    return frame


def _longest_run(flags):
    best, start = (0, 0), None
    for i, f in enumerate(np.r_[flags, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_panel(frame, side="left", white=240):
    """(y0, y1, x0, x1) of the photo panel on that half of the frame.

    Columns first, then rows inside them: the panel is narrower than half
    the frame on most release videos (00122_outer, 00170_upper), so a row
    test over the whole half saw the white margin and picked a band of
    the photo instead of the photo. Thresholds are loose (0.75) because a
    white shirt inside the photo is itself near-white."""
    width = frame.shape[1]
    nonwhite = frame.astype(np.int32).sum(axis=2) < 3 * white
    if side == "left":
        offset, region = 0, nonwhite[:, : width // 2 + width // 8]
    else:
        offset = width // 2 - width // 8
        region = nonwhite[:, offset:]
    x0, x1 = _longest_run(region.mean(axis=0) > 0.5)
    y0, y1 = _longest_run(region[:, x0:x1].mean(axis=1) > 0.75)
    x0, x1 = _longest_run(region[y0:y1].mean(axis=0) > 0.75)
    return y0, y1, x0 + offset, x1 + offset


# --- cameras, silhouettes, alignment -----------------------------------------


def camera_matrices(camera, similarity):
    """`view` (4x4 world-to-camera, OpenCV) and `intrinsics` for the
    panel's pixel grid: `pixel = s * K_normalised + (ox, oy)`."""
    s, ox, oy = similarity
    extrinsic = np.asarray(camera["extrinsics"], np.float64)
    view = np.eye(4)
    view[:3] = extrinsic
    k = np.asarray(camera["intrinsics"], np.float64)
    intrinsics = np.array([[k[0, 0] * s, 0.0, k[0, 2] * s + ox], [0.0, k[1, 1] * s, k[1, 2] * s + oy], [0.0, 0.0, 1.0]])
    return view, intrinsics


def project(points, view, intrinsics):
    cam = np.asarray(points, np.float64) @ view[:3, :3].T + view[:3, 3]
    z = np.maximum(cam[:, 2], 1e-6)
    return np.stack([intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2], intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]], 1), cam[:, 2]


def fill_silhouette(pixels, faces, size):
    """Boolean mask of the mesh's projection. One `fillConvexPoly` per
    triangle: `fillPoly` over many contours uses an even-odd rule, so the
    front and back of a body cancel and only edges are drawn."""
    height, width = size
    mask = np.zeros((height, width), np.uint8)
    for triangle in np.round(np.asarray(pixels)[faces]).astype(np.int32):
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask.astype(bool)


def iou(a, b):
    return _iou(a, b, empty=0.0)


def fit_similarity(meshes, camera, person, size, verbose=True):
    """(s, ox, oy) maximising the IoU of the union silhouette of `meshes`
    ((vertices, faces) pairs: the reference body and cloth) against the
    person mask, by coordinate descent from a bounding-box guess."""
    base_view, base_k = camera_matrices(camera, (1.0, 0.0, 0.0))
    pixels, depth = project(np.concatenate([v for v, _ in meshes]), base_view, base_k)
    if (depth <= 0).any():
        return None, 0.0
    ys, xs = np.where(person)
    x0, x1, y0, y1 = pixels[:, 0].min(), pixels[:, 0].max(), pixels[:, 1].min(), pixels[:, 1].max()
    s = 0.5 * ((xs.max() - xs.min()) / (x1 - x0) + (ys.max() - ys.min()) / (y1 - y0))
    current = np.array([s, xs.min() - x0 * s, ys.min() - y0 * s])

    def score(p):
        return iou(union_silhouette(meshes, camera, p, size), person)

    value = score(current)
    for ds, dp in ((0.04, 40), (0.02, 20), (0.01, 10), (0.005, 5), (0.0025, 2), (0.001, 1)):
        improved = True
        while improved:
            improved = False
            for d in ([ds, 0, 0], [-ds, 0, 0], [0, dp, 0], [0, -dp, 0], [0, 0, dp], [0, 0, -dp]):
                candidate = current + np.array(d)
                v = score(candidate)
                if v > value:
                    current, value, improved = candidate, v, True
    return tuple(float(v) for v in current), float(value)


def union_silhouette(meshes, camera, similarity, size):
    view, k = camera_matrices(camera, similarity)
    mask = np.zeros(size, bool)
    for vertices, faces in meshes:
        mask |= fill_silhouette(project(vertices, view, k)[0], faces, size)
    return mask


# --- rendering --------------------------------------------------------------


def occluded_from(xyz, camera_centre, body_vertices, body_faces, margin=OCCLUSION_MARGIN):
    """`GarmentRenderer.occluded_by_body` for an arbitrary camera position."""
    return _occluded_from(xyz, camera_centre, body_vertices, body_faces, margin)


def render_with_camera(renderer, mesh, view, intrinsics, size, body, appearance_net=None, body_occluder=None):
    """BGRA uint8 of `mesh` (renderer vertex order) seen by the real camera.
    `body`: (vertices, faces) hiding the splats behind it; `appearance_net`:
    run the checkpoint's shading with `body_occluder` (default `body`) in
    its ambient-occlusion bake."""
    from tryon.rendering.splat_cpu import render_splats
    from tryon.garment.appearance import merge

    splats = renderer.splats(mesh)
    if appearance_net is not None:
        result = appearance_net.forward(mesh, view, occluders=[body_occluder or body])
        splats = merge(splats, result, renderer._kept_indices)
    centre = -view[:3, :3].T @ view[:3, 3]
    keep = ~occluded_from(splats["xyz"], centre, *body)
    splats = {key: value[keep] for key, value in splats.items()}
    image, alpha = render_splats(splats, view, intrinsics, size)
    return np.dstack([(np.clip(image, 0, 1) * 255).astype(np.uint8), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])


def composite(frame, bgra):
    from tryon.app.compositing import composite_garment

    out = frame.copy()
    composite_garment(out, bgra)
    return out


# --- metrics ----------------------------------------------------------------


def ssim_gray(a, b):
    """Wang et al. 2004 on uint8 grayscale, Gaussian window 11 / sigma 1.5."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).mean())


class Dists:
    """DISTS (Ding et al. 2020) via the installed `DISTS_pytorch`, only when
    its VGG16 backbone is already on disk: `None` otherwise."""

    def __init__(self):
        self.model = None
        self.reason = None
        if not VGG16_WEIGHTS.exists():
            self.reason = f"torchvision VGG16 weights not cached ({VGG16_WEIGHTS}); not downloaded by this tool"
            return
        try:
            import torch
            from DISTS_pytorch import DISTS

            self.torch = torch
            self.model = DISTS().eval()
        except Exception as error:  # noqa: BLE001 - report, do not fail the ablation
            self.reason = f"{type(error).__name__}: {error}"

    def __call__(self, a, b):
        if self.model is None:
            return None
        def to(x):
            return self.torch.as_tensor(cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        with self.torch.no_grad():
            return float(self.model(to(a), to(b)))


def region_bbox(mask, margin=0.05):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return (max(int(ys.min() - margin * h), 0), int(ys.max() + margin * h) + 1, max(int(xs.min() - margin * w), 0), int(xs.max() + margin * w) + 1)


def garment_metrics(truth, ours, alpha, shirt, crop, dists):
    """Over the union of the shirt mask and the drawn garment."""
    drawn = alpha > ALPHA_FLOOR
    region = shirt | drawn
    diff = (truth.astype(np.float64) - ours.astype(np.float64))[region]
    mse = float((diff**2).mean()) if region.any() else float("nan")
    y0, y1, x0, x1 = crop
    def small(img):
        return cv2.resize(img[y0:y1, x0:x1], EVAL_SIZE, interpolation=cv2.INTER_AREA)
    a, b = small(truth), small(ours)
    return {
        "psnr_db": float(10 * np.log10(255.0**2 / max(mse, 1e-9))),
        "ssim": ssim_gray(a, b),
        "dists": dists(a, b),
        "silhouette_iou": iou(drawn, shirt),
        "drawn_pixels": int(drawn.sum()),
        "shirt_pixels": int(shirt.sum()),
        "region_pixels": int(region.sum()),
    }


# --- the ablation -----------------------------------------------------------


def evaluate(checkpoint, video, video_frame, trajectory_path, fits_cache, camera_name, trajectory_frame, settle, out_png, live_height=870,
             similarity_seed=None, auto_frame=False, verbose=True, dump_dir=None):
    import torch

    from tryon.garment.appearance import AppearanceNet
    from tryon.body.fitter import BodyFitter
    from tryon.app.person import landmarks_from
    from tryon.removal.parsing_schp import ATR_CHECKPOINT, SCHPParser
    from tryon.garment.renderer import GarmentRenderer

    started = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    frame = read_video_frame(video, video_frame)
    panel_box = find_panel(frame, "left")
    y0, y1, x0, x1 = panel_box
    panel = np.ascontiguousarray(frame[y0:y1, x0:x1])
    size = panel.shape[:2]
    log(f"video frame {video_frame}: GT panel rows {y0}..{y1}, cols {x0}..{x1} -> {size[1]}x{size[0]}")

    parser = SCHPParser(ATR_CHECKPOINT)
    labels = parser.labels(panel)
    person = labels != 0
    shirt = np.isin(labels, SHIRT_CLASSES)
    shirt_rows = shirt.any(axis=1)
    body_rows = ~shirt_rows  # head and legs: rows where the body silhouette is not hidden by the shirt
    log(f"parser: person {int(person.sum())} px, shirt {int(shirt.sum())} px, shirt rows {int(shirt_rows.sum())} of {size[0]}")

    with open(trajectory_path, "rb") as handle:
        trajectory = pickle.load(handle)
    obstacle = np.asarray(trajectory["obstacle"], np.float64)
    pred = np.asarray(trajectory["pred"], np.float64)
    obstacle_faces = np.asarray(trajectory["obstacle_faces"])
    with open(Path(checkpoint) / "stage1" / "cameras.json") as handle:
        cameras = json.load(handle)

    # Which camera, and how the panel crops it: the reference's own
    # silhouette (their body + their cloth) against the person mask.
    cloth_faces = np.asarray(trajectory["cloth_faces"])
    if similarity_seed is not None:
        # A later video frame: the crop is the one fitted on an earlier
        # frame, and the trajectory frame this photo shows is unknown (the
        # video and the trajectory differ in length) - search every frame.
        camera = cameras[camera_name]
        scores = [iou(union_silhouette([(obstacle[j], obstacle_faces), (pred[j], cloth_faces)], camera, similarity_seed, size), person) for j in range(len(obstacle))]
        found = int(np.argmax(scores))
        log(f"trajectory frame search at the seed crop: best {found} (IoU {scores[found]:.3f}; frame {trajectory_frame} scores {scores[trajectory_frame]:.3f})")
        if auto_frame:
            trajectory_frame = found
    reference = [(obstacle[trajectory_frame], obstacle_faces), (pred[trajectory_frame], cloth_faces)]
    candidates = {}
    for name, camera in cameras.items():
        if camera_name is not None and name != camera_name:
            continue
        similarity, value = fit_similarity(reference, camera, person, size, verbose)
        if similarity is None:
            log(f"camera {name}: body behind the camera")
            continue
        body_sil = union_silhouette(reference[:1], camera, similarity, size)
        candidates[name] = {"similarity": similarity, "iou_body_and_cloth": value, "iou_body_full": iou(body_sil, person),
                            "iou_body_head_legs": iou(body_sil[body_rows], person[body_rows])}
        log(f"camera {name}: s={similarity[0]:.4f} ox={similarity[1]:.1f} oy={similarity[2]:.1f}  IoU body+cloth {value:.3f}, "
            f"body {candidates[name]['iou_body_full']:.3f} (head+legs {candidates[name]['iou_body_head_legs']:.3f})")
    camera_name = max(candidates, key=lambda n: candidates[n]["iou_body_and_cloth"])
    similarity = candidates[camera_name]["similarity"]
    camera = cameras[camera_name]
    view, intrinsics = camera_matrices(camera, similarity)
    log(f"panel = camera {camera_name}")

    # Which trajectory frame the video frame is.
    alignment = []
    for j in range(max(trajectory_frame - 12, 0), min(trajectory_frame + 13, len(obstacle))):
        both = union_silhouette([(obstacle[j], obstacle_faces), (pred[j], cloth_faces)], camera, similarity, size)
        body_sil = union_silhouette([(obstacle[j], obstacle_faces)], camera, similarity, size)
        alignment.append({"trajectory_frame": j, "iou_body_and_cloth": iou(both, person), "iou_body_head_legs": iou(body_sil[body_rows], person[body_rows])})
    best_k = max(alignment, key=lambda r: r["iou_body_and_cloth"])["trajectory_frame"]
    log("trajectory frame vs this video frame (IoU body+cloth): " + " ".join(f"{r['trajectory_frame']}:{r['iou_body_and_cloth']:.3f}" for r in alignment) + f" -> best {best_k}, using {trajectory_frame}")

    k = trajectory_frame
    body_true = (obstacle[k], obstacle_faces)
    body_iou = {}
    body_iou["L1_their_body"] = {"full": candidates[camera_name]["iou_body_full"], "head_legs": candidates[camera_name]["iou_body_head_legs"]}

    # The shirt erased (the live path's clothing removal), for the second background.
    from tryon.removal.remover import ClothingRemover, arm_strokes, hand_discs

    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.fill = "lama" if ClothingRemover.LAMA_CHECKPOINT.exists() else "skin"
    import mediapipe as mp

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        points = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), size[1], size[0])
    if points is None:
        raise SystemExit("MediaPipe found no person on the GT panel")
    t0 = time.perf_counter()
    removal_mask = remover.clothing_mask(panel, keypoints=points)
    arms = arm_strokes(points, panel.shape, width_fraction=remover.ARM_TUBE_WIDTH)
    erased = remover.remove(panel, removal_mask, arms=arms, keep=hand_discs(points, panel.shape))
    log(f"clothing removal ({remover.fill}): {int(removal_mask.sum())} px in {time.perf_counter() - t0:.1f} s")

    renders = {}  # name -> BGRA
    timing = {}

    # L1: their body + their cloth, our renderer.
    renderer = GarmentRenderer(checkpoint, dynamics=False)
    to_ref, count = reference_vertex_map(renderer)
    mesh_theirs = pred[k][to_ref]  # 7972 -> 8208 order (duplicates share the vertex)
    net = AppearanceNet(checkpoint, faces=renderer.faces, half=True)
    t0 = time.perf_counter()
    renders["L1_static"] = render_with_camera(renderer, mesh_theirs, view, intrinsics, size, body_true)
    timing["L1_static"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    renders["L1_appearance"] = render_with_camera(renderer, mesh_theirs, view, intrinsics, size, body_true, appearance_net=net)
    timing["L1_appearance"] = time.perf_counter() - t0
    log(f"L1 rendered: static {timing['L1_static']:.1f} s, appearance {timing['L1_appearance']:.1f} s")

    # L2: their body (the cached SMPL-X fit of obstacle[k]), our cloth. The
    # garment is bound on frame 0's fit (the template pose) and the cloth
    # solver replays the fitted frames 1..k at 25 fps, as
    # tools/evaluate/cloth_trajectory.py does; at k = 0 it settles for
    # `settle` steps on the static pose instead.
    fits = np.load(fits_cache)
    def fit_of(j):
        return {key: fits[key][int(np.where(fits["frames"] == j)[0][0])] for key in ("betas", "body_pose", "global_orient", "offset", "residual_m")}
    driver = BodyFitter()
    set_fitter(driver, fit_of(k))
    with torch.no_grad():
        body_fit = (driver._forward().vertices[0].numpy().astype(np.float64), np.asarray(driver.model.faces))
    fit_vs_true_cm = float(np.linalg.norm(body_fit[0] - obstacle[k], axis=1).mean() * 100)
    silhouette = fill_silhouette(project(body_fit[0], view, intrinsics)[0], body_fit[1], size)
    body_iou["L2_fitted_body"] = {"full": iou(silhouette, person), "head_legs": iou(silhouette[body_rows], person[body_rows]), "fit_residual_cm": fit_vs_true_cm}
    l2 = {}
    for mode, dynamics in (("lbs", False), ("dynamics", True)):
        r = GarmentRenderer(checkpoint, dynamics=dynamics)
        t0 = time.perf_counter()
        set_fitter(driver, fit_of(0))
        r.bind(driver)
        if dynamics and k > 0:
            for j in range(1, k + 1):
                set_fitter(driver, fit_of(j))
                mesh = r.posed_mesh(driver, dt=1.0 / REFERENCE_FPS)
        else:
            set_fitter(driver, fit_of(k))
            mesh = r.posed_mesh(driver, dt=0.0)
            if dynamics:
                for _ in range(settle):
                    mesh = r.posed_mesh(driver, dt=1.0 / REFERENCE_FPS)
        timing[f"L2_{mode}_pose"] = time.perf_counter() - t0
        ours_ref = np.zeros((count, 3))
        np.add.at(ours_ref, to_ref, mesh)
        ours_ref /= np.bincount(to_ref, minlength=count)[:, None]
        l2[mode] = {"vertex_distance_to_pred_cm": float(np.linalg.norm(ours_ref - pred[k], axis=1).mean() * 100)}
        renders[f"L2_{mode}_static"] = render_with_camera(r, mesh, view, intrinsics, size, body_fit)
        net_r = AppearanceNet(checkpoint, faces=r.faces, half=True) if mode == "dynamics" else net
        renders[f"L2_{mode}_appearance"] = render_with_camera(r, mesh, view, intrinsics, size, body_fit, appearance_net=net_r)
        log(f"L2 {mode}: posed in {timing[f'L2_{mode}_pose']:.1f} s, {l2[mode]['vertex_distance_to_pred_cm']:.1f} cm from pred[{k}]")

    # L3: our tracking (MediaPipe -> BodyFitter.fit), our cloth - the live path.
    # The fitter is tuned for webcam-sized frames (`fit` starts the weak-
    # perspective scale at 250 and its camera stage can only move it a few
    # hundred), so the photo is fitted at `live_height` and the camera
    # scaled back up - weak perspective is linear in the image size.
    fit_by_size = {}
    for height in sorted({size[0], live_height, 640, 480}, reverse=True):
        f = height / size[0]
        img = cv2.resize(panel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1.0 else panel
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            pts = landmarks_from(pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), img.shape[1], img.shape[0])
        t0 = time.perf_counter()
        probe = BodyFitter()
        probe.fit(pts)
        fit_by_size[str(height)] = {"reprojection_px": float(probe.reprojection_error(pts)), "reprojection_px_at_panel_scale": float(probe.reprojection_error(pts) / f),
                                    "scale": float(probe.scale), "betas_abs_max": float(probe.betas.detach().abs().max()), "seconds": time.perf_counter() - t0}
        log(f"  live fit at height {height}: reprojection {fit_by_size[str(height)]['reprojection_px']:.1f} px ({fit_by_size[str(height)]['reprojection_px_at_panel_scale']:.1f} at panel scale), scale {float(probe.scale):.0f}")
        if height == live_height:
            live, live_factor = probe, f
    with torch.no_grad():
        live.scale.mul_(1.0 / live_factor)
        live.translation.mul_(1.0 / live_factor)
    timing["L3_fit"] = fit_by_size[str(live_height)]["seconds"]
    reprojection = fit_by_size[str(live_height)]["reprojection_px_at_panel_scale"]
    silhouette = fill_silhouette(live.projected_vertices(), np.asarray(live.faces), size)
    body_iou["L3_tracked_body"] = {"full": iou(silhouette, person), "head_legs": iou(silhouette[body_rows], person[body_rows]), "reprojection_px": reprojection,
                                   "fit_height": live_height, "fit_by_height": fit_by_size}
    l3 = {}
    for mode, dynamics in (("lbs", False), ("dynamics", True)):
        for shading in ("static", "appearance"):
            r = GarmentRenderer(checkpoint, dynamics=dynamics, appearance=shading == "appearance")
            t0 = time.perf_counter()
            r.bind(live)
            if dynamics:
                for _ in range(settle):
                    r.posed_mesh(live, dt=1.0 / REFERENCE_FPS)
            renders[f"L3_{mode}_{shading}"] = r.render(live, size)
            timing[f"L3_{mode}_{shading}"] = time.perf_counter() - t0
        l3[mode] = {"height_ratio": float(r._size_ratio), "width_ratio": float(r._width_ratio)}
    log(f"L3: fit {timing['L3_fit']:.1f} s, reprojection {reprojection:.1f} px, body IoU {body_iou['L3_tracked_body']['full']:.3f}")

    # Metrics, one crop for every level.
    union_all = shirt.copy()
    for bgra in renders.values():
        union_all |= bgra[..., 3] > ALPHA_FLOOR
    crop = region_bbox(union_all)
    dists = Dists()
    if dists.reason:
        log(f"DISTS unavailable: {dists.reason}")
    results = {}
    for name, bgra in renders.items():
        results[name] = {
            "over_photo": garment_metrics(panel, composite(panel, bgra), bgra[..., 3], shirt, crop, dists),
            "over_erased": garment_metrics(panel, composite(erased, bgra), bgra[..., 3], shirt, crop, dists),
        }
    # The photo itself with the shirt erased, and untouched: the floors.
    results["erased_only"] = {"over_erased": garment_metrics(panel, erased, np.zeros(size, np.uint8), shirt, crop, dists)}

    # The figure: GT | L1 static | L1 appearance | L2 | L3 (appearance, dynamics), same crop; second row over the erased photo.
    order = ["L1_static", "L1_appearance", "L2_dynamics_appearance", "L3_dynamics_appearance"]
    cy0, cy1, cx0, cx1 = crop
    tile_h = 900
    scale = tile_h / (cy1 - cy0)
    tile_w = int((cx1 - cx0) * scale)

    def tile(img, label):
        t = cv2.resize(img[cy0:cy1, cx0:cx1], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tile_w, 34), (0, 0, 0), -1)
        cv2.putText(t, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return t

    def caption(name, background):
        m = results[name][background]
        d = f" DISTS {m['dists']:.3f}" if m["dists"] is not None else ""
        return f"{name} PSNR {m['psnr_db']:.1f} SSIM {m['ssim']:.3f}{d} IoU {m['silhouette_iou']:.2f}"

    rows = [
        np.hstack([tile(panel, f"Ground truth (camera {camera_name})")] + [tile(composite(panel, renders[n]), caption(n, "over_photo")) for n in order]),
        np.hstack([tile(erased, "GT, shirt erased")] + [tile(composite(erased, renders[n]), caption(n, "over_erased")) for n in order]),
    ]
    figure = np.vstack(rows)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), figure)
    if dump_dir is not None:
        # full-resolution images, so other systems' outputs can be scored
        # on exactly the same frame (docs/REFERENCE_RUNS.md)
        dump = Path(dump_dir)
        dump.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dump / "ground_truth.png"), panel)
        cv2.imwrite(str(dump / "erased.png"), erased)
        for name in order:
            cv2.imwrite(str(dump / f"{name}_over_erased.png"), composite(erased, renders[name]))

    return {
        "checkpoint": str(checkpoint),
        "video": str(video),
        "video_frame": video_frame,
        "video_size": [int(frame.shape[1]), int(frame.shape[0])],
        "gt_panel_box_yyxx": [int(v) for v in panel_box],
        "panel_size": [int(size[1]), int(size[0])],
        "trajectory": str(trajectory_path),
        "trajectory_frame": k,
        "camera": camera_name,
        "camera_candidates": candidates,
        "panel_similarity": {"s": similarity[0], "ox": similarity[1], "oy": similarity[2]},
        "frame_alignment": alignment,
        "best_trajectory_frame": best_k,
        "clothing_removal_fill": remover.fill,
        "body_silhouette_iou": body_iou,
        "L2": l2,
        "L3": l3,
        "settle_steps": settle,
        "metrics": results,
        "metric_crop_yyxx": [int(v) for v in crop],
        "dists_available": dists.model is not None,
        "dists_unavailable_reason": dists.reason,
        "lpips": "not installed",
        "timing_s": timing,
        "total_s": time.perf_counter() - started,
        "figure": str(out_png),
    }


def print_table(result):
    body = result["body_silhouette_iou"]
    print(f"\ncamera {result['camera']}, trajectory frame {result['trajectory_frame']} (best by silhouette: {result['best_trajectory_frame']}), "
          f"panel similarity s={result['panel_similarity']['s']:.3f}")
    print("body silhouette IoU vs parser person mask (full / head+legs):")
    for name, v in body.items():
        extra = "".join(f", {k} {val:.2f}" for k, val in v.items() if isinstance(val, float) and k not in ("full", "head_legs"))
        print(f"  {name:18s} {v['full']:.3f} / {v['head_legs']:.3f}{extra}")
    print("\ngarment region, composited over the photo | over the erased photo")
    print(f"{'level':26s} {'PSNR':>6s} {'SSIM':>6s} {'DISTS':>6s} {'IoU':>5s} | {'PSNR':>6s} {'SSIM':>6s} {'DISTS':>6s} {'IoU':>5s}")
    for name, m in result["metrics"].items():
        cells = []
        for background in ("over_photo", "over_erased"):
            v = m.get(background)
            if v is None:
                cells.append(" " * 27)
                continue
            d = f"{v['dists']:6.3f}" if v["dists"] is not None else "   n/a"
            cells.append(f"{v['psnr_db']:6.2f} {v['ssim']:6.3f} {d} {v['silhouette_iou']:5.2f}")
        print(f"{name:26s} {cells[0]} | {cells[1]}")
    if result["dists_unavailable_reason"]:
        print(f"DISTS: {result['dists_unavailable_reason']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=str(GARMENTS / "00152_outer"))
    parser.add_argument("--video", default=None, help="the release video (default: <release videos>/<checkpoint name>.mp4)")
    parser.add_argument("--video-frame", type=int, default=0)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--trajectory-frame", type=int, default=0)
    parser.add_argument("--fits", default=None, help="npz cache of the per-frame body fits (tools/evaluate/cloth_trajectory.py)")
    parser.add_argument("--camera", default=None, help="camera id from cameras.json (default: the one whose projection matches the panel)")
    parser.add_argument("--settle", type=int, default=5, help="cloth solver steps at the frame's pose before rendering")
    parser.add_argument("--similarity", default=None, help="s,ox,oy of the panel crop from an earlier frame's json; with --camera, "
                        "the trajectory frame is searched over the whole sequence at that crop")
    parser.add_argument("--auto-frame", action="store_true", help="use the trajectory frame the search finds instead of --trajectory-frame")
    parser.add_argument("--live-height", type=int, default=870, help="frame height the live fit runs at (the webcam size the fitter is tuned for)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--png", default=str(SCRATCH / "reference_ablation.png"))
    parser.add_argument("--dump-dir", default=None, help="also write the ground truth, the erased photo and every level's composite at full resolution here")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    video = Path(args.video).expanduser() if args.video else RELEASE_VIDEOS / f"{checkpoint.name}.mp4"
    trajectory = Path(args.trajectory).expanduser() if args.trajectory else trajectory_for(checkpoint)
    baseline = Path(__file__).resolve().parents[2] / "docs" / "baseline"
    fits = Path(args.fits) if args.fits else baseline / f"trajectory-fits-{checkpoint.name}.npz"
    out = Path(args.out) if args.out else baseline / f"reference-frame-ablation-{checkpoint.name}.json"
    for path, what in ((video, "video"), (trajectory, "trajectory"), (fits, "fits cache")):
        if path is None or not path.exists():
            raise SystemExit(f"no {what} for {checkpoint.name}: {path}")
    seed = tuple(float(v) for v in args.similarity.split(",")) if args.similarity else None
    if seed is not None and args.camera is None:
        raise SystemExit("--similarity needs --camera")
    result = evaluate(checkpoint, video, args.video_frame, trajectory, fits, args.camera, args.trajectory_frame, args.settle, args.png, args.live_height,
                      similarity_seed=seed, auto_frame=args.auto_frame, dump_dir=args.dump_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print_table(result)
    print(f"saved {out} and {args.png} ({result['total_s']:.0f} s)")


if __name__ == "__main__":
    main()
