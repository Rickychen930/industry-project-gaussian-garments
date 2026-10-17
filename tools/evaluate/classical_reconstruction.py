"""Classical multi-view reconstruction of a garment from the four 4D-DRESS cameras, scored against the scan.

The registered project planned this pipeline: frames, feature matching, camera motion, triangulation
and surface reconstruction. For each take (one frame, four calibrated cameras about 90 degrees apart) it runs

1. SIFT features inside the person mask and a ratio-test match for every camera pair;
2. camera motion from the essential matrix (RANSAC), compared with the capture's calibration;
3. triangulation of the matches with the capture's calibration, kept if the reprojection error is small,
   scored by the distance of each point to the scan surface;
4. a visual hull (shape from silhouette) of the person and of the upper garment, carved from the four masks
   and meshed with marching cubes, scored against the scan's garment vertices (F-score at 2 cm).

The upper-garment masks come from the SCHP parser; the scan and its labels (label 3 is the upper garment)
are used only for scoring.

    python -m tools.evaluate.classical_reconstruction     # writes docs/baseline/classical-reconstruction.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from tools.evaluate._common import write_json
from tryon import paths
from tryon.removal.parsing_schp import SCHPParser

CAMERAS = ("0004", "0028", "0052", "0076")
UPPER_LABEL = 3          # scan label of the upper garment
ATR_UPPER_CLOTHES = 4    # SCHP ATR class of upper clothes
RATIO = 0.8
MAX_REPROJECTION_PX = 3.0
VOXEL = 0.02             # metres
THRESHOLD = 0.02         # F-score distance, metres
OUTPUT = paths.BASELINE / "classical-reconstruction.json"


def _project(points: np.ndarray, camera: dict) -> np.ndarray:
    cam = camera["extrinsics"] @ np.c_[points, np.ones(len(points))].T
    uv = camera["intrinsics"] @ cam
    return (uv[:2] / uv[2]).T


def _sift(gray: np.ndarray, mask: np.ndarray):
    return cv2.SIFT_create().detectAndCompute(gray, mask)


def _ratio_matches(a, b):
    (kp_a, des_a), (kp_b, des_b) = a, b
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    return (np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2),
            np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2))


def _rotation_error_deg(r_est: np.ndarray, r_true: np.ndarray) -> float:
    cos = (np.trace(r_est.T @ r_true) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _direction_error_deg(t_est: np.ndarray, t_true: np.ndarray) -> float:
    t_est, t_true = t_est.ravel(), t_true.ravel()
    cos = abs(t_est @ t_true) / (np.linalg.norm(t_est) * np.linalg.norm(t_true))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def camera_motion(pts_a, pts_b, cam_a, cam_b) -> dict:
    """Relative pose of camera b from a by the essential matrix, against the capture's calibration."""
    empty = {"matches": len(pts_a), "inliers": 0, "rotation_error_deg": None, "direction_error_deg": None}
    if len(pts_a) < 8:
        return empty
    k = cam_a["intrinsics"]
    essential, inliers = cv2.findEssentialMat(pts_a, pts_b, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None or essential.shape != (3, 3):
        return empty
    count, rot, trans, _ = cv2.recoverPose(essential, pts_a, pts_b, k, mask=inliers)
    r_a, t_a = cam_a["extrinsics"][:, :3], cam_a["extrinsics"][:, 3]
    r_b, t_b = cam_b["extrinsics"][:, :3], cam_b["extrinsics"][:, 3]
    r_true = r_b @ r_a.T
    t_true = t_b - r_true @ t_a
    return {"matches": len(pts_a), "inliers": int(count), "rotation_error_deg": _rotation_error_deg(rot, r_true),
            "direction_error_deg": _direction_error_deg(trans, t_true)}


def triangulate(pts_a, pts_b, cam_a, cam_b) -> np.ndarray:
    """3D points from matches with the capture's calibration, kept if both reprojection errors are small."""
    if len(pts_a) == 0:
        return np.zeros((0, 3))
    proj_a = cam_a["intrinsics"] @ cam_a["extrinsics"]
    proj_b = cam_b["intrinsics"] @ cam_b["extrinsics"]
    hom = cv2.triangulatePoints(proj_a, proj_b, pts_a.T.astype(np.float64), pts_b.T.astype(np.float64))
    points = (hom[:3] / hom[3]).T
    err_a = np.linalg.norm(_project(points, cam_a) - pts_a, axis=1)
    err_b = np.linalg.norm(_project(points, cam_b) - pts_b, axis=1)
    return points[(err_a < MAX_REPROJECTION_PX) & (err_b < MAX_REPROJECTION_PX)]


def visual_hull(masks: dict, cameras: dict, bounds: np.ndarray):
    """Occupied voxel centres (inside every mask once projected) and the grid axes."""
    axes = [np.arange(lo, hi, VOXEL) for lo, hi in zip(bounds[0], bounds[1])]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(grid), bool)
    for cam, mask in masks.items():
        uv = np.round(_project(grid, cameras[cam])).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        hit = np.zeros(len(grid), bool)
        hit[inside] = mask[uv[inside, 1], uv[inside, 0]] > 0
        keep &= hit
    return grid[keep], axes


def hull_surface(centres: np.ndarray, axes) -> np.ndarray:
    """Surface points of the hull by marching cubes on the occupancy grid (empty if the hull is empty)."""
    if len(centres) < 8:
        return np.zeros((0, 3))
    origin = np.array([a[0] for a in axes])
    volume = np.zeros([len(a) for a in axes], np.float32)
    index = np.round((centres - origin) / VOXEL).astype(int)
    volume[index[:, 0], index[:, 1], index[:, 2]] = 1.0
    verts, _, _, _ = marching_cubes(np.pad(volume, 1), 0.5)
    return (verts - 1) * VOXEL + origin


def f_score(surface: np.ndarray, truth: np.ndarray) -> dict:
    if len(surface) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0, "chamfer_cm": None}
    d_to_truth = cKDTree(truth).query(surface)[0]
    d_to_surface = cKDTree(surface).query(truth)[0]
    precision = float((d_to_truth < THRESHOLD).mean())
    recall = float((d_to_surface < THRESHOLD).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f,
            "chamfer_cm": float(100 * (d_to_truth.mean() + d_to_surface.mean()) / 2)}


def evaluate_take(take: Path, frame_index: int, parser: SCHPParser) -> dict:
    capture = take / "Capture"
    with open(capture / "cameras.pkl", "rb") as handle:
        cameras = pickle.load(handle)
    name = sorted((capture / CAMERAS[0] / "images").glob("capture-f*.png"))[frame_index].name
    number = name.split("-f")[1].split(".")[0]
    images = {c: cv2.imread(str(capture / c / "images" / name)) for c in CAMERAS}
    person = {c: (cv2.imread(str(capture / c / "masks" / f"mask-f{number}.png"), 0) > 0).astype(np.uint8) * 255
              for c in CAMERAS}
    garment = {c: ((parser.labels(images[c]) == ATR_UPPER_CLOTHES) & (person[c] > 0)).astype(np.uint8) * 255
               for c in CAMERAS}
    with open(take / "Meshes_pkl" / f"mesh-f{number}.pkl", "rb") as handle:
        scan = pickle.load(handle)["vertices"].astype(np.float64)
    with open(take / "Semantic" / "labels" / f"label-f{number}.pkl", "rb") as handle:
        labels = pickle.load(handle)["scan_labels"]
    truth_garment = scan[labels == UPPER_LABEL]
    features = {c: _sift(cv2.cvtColor(images[c], cv2.COLOR_BGR2GRAY), person[c]) for c in CAMERAS}

    motion, points = [], []
    for a, b in combinations(CAMERAS, 2):
        pts_a, pts_b = _ratio_matches(features[a], features[b])
        motion.append(camera_motion(pts_a, pts_b, cameras[a], cameras[b]))
        points.append(triangulate(pts_a, pts_b, cameras[a], cameras[b]))
    cloud = np.vstack(points)
    cloud_error = cKDTree(scan).query(cloud)[0] * 100 if len(cloud) else np.zeros(0)

    margin = 0.15
    person_hull = hull_surface(*visual_hull(person, cameras, np.array([scan.min(0) - margin, scan.max(0) + margin])))
    box = np.array([truth_garment.min(0) - 0.3, truth_garment.max(0) + 0.3])
    garment_hull = hull_surface(*visual_hull(garment, cameras, box))
    return {"take": f"{take.parent.parent.name}/{take.parent.name}/{take.name}",
            "motion": motion,
            "triangulated_points": int(len(cloud)),
            "triangulated_median_error_cm": float(np.median(cloud_error)) if len(cloud_error) else None,
            "person_hull": f_score(person_hull, scan),
            "garment_hull": f_score(garment_hull, truth_garment)}


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarise(rows: list[dict]) -> dict:
    motion = [m for r in rows for m in r["motion"]]
    return {
        "takes": len(rows),
        "pairs": len(motion),
        "pairs_with_pose": sum(m["rotation_error_deg"] is not None for m in motion),
        "median_camera_rotation_error_deg": _median([m["rotation_error_deg"] for m in motion]),
        "median_camera_direction_error_deg": _median([m["direction_error_deg"] for m in motion]),
        "median_triangulated_points": _median([r["triangulated_points"] for r in rows]),
        "median_triangulation_error_cm": _median([r["triangulated_median_error_cm"] for r in rows]),
        "person_hull_f_score": _median([r["person_hull"]["f_score"] for r in rows]),
        "garment_hull_f_score": _median([r["garment_hull"]["f_score"] for r in rows]),
        "garment_hull_precision": _median([r["garment_hull"]["precision"] for r in rows]),
        "garment_hull_recall": _median([r["garment_hull"]["recall"] for r in rows]),
        "garment_hull_chamfer_cm": _median([r["garment_hull"]["chamfer_cm"] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--takes", type=int, default=3, help="takes per subject")
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()
    schp = SCHPParser()
    takes = []
    for subject in ("00134", "00136"):
        takes += sorted((paths.DATA_ROOT / "4d-dress" / subject / "Inner").glob("Take*"))[: args.takes]
    rows = [evaluate_take(take, args.frame_index, schp) for take in takes]
    result = {"cameras": list(CAMERAS), "frame_index": args.frame_index, "voxel_m": VOXEL,
              "threshold_m": THRESHOLD, "rows": rows, "summary": summarise(rows)}
    write_json(Path(args.out), result)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
