"""How accurate is BodyFitter on a REAL photo, not a synthetic skeleton?

Every existing test in `tests/test_body_fitter.py` drives `BodyFitter` with
`_landmarks()` — hand-placed pixel coordinates for a made-up pose. That
proves the optimiser converges; it says nothing about MediaPipe's actual
detector noise, occlusion, or a real body's proportions. `data/annotated/`
holds exactly what is missing: real 4D-DRESS photographs with calibrated
cameras (`cameras.json`) and the real SMPL-X parameters those photos were
fit from (`smplx/*.pkl`) — genuine ground truth, not a synthetic stand-in.

This script:
  1. loads a real photo and its camera calibration
  2. runs MediaPipe pose on the real pixels (the actual detector, not a
     hand-typed landmark dict)
  3. fits BodyFitter from those detections
  4. compares the fitted body against the REAL SMPL-X ground truth two ways:
       - 2D: project both the fit's joints (its own weak-perspective camera)
         and the ground truth's joints (the REAL calibrated camera) into
         pixel space, compare
       - 3D: MPJPE after Procrustes alignment (root + scale + rotation),
         the field-standard protocol (see docs/PLAN.md, survey ref #8)

Run: .venv/bin/python -m tools.evaluate.real_fit
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.body.landmarks import JOINT_MAP, POSE

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "annotated"
CAMERAS = ("0004", "0028", "0052", "0076")


def load_camera(take, cam_id):
    with open(DATA / take / "cameras.json") as f:
        cams = json.load(f)
    c = cams[cam_id]
    return np.array(c["intrinsics"]), np.array(c["extrinsics"])


def project(points_world, intrinsics, extrinsics):
    """World (N,3) -> pixel (N,2) through a real calibrated camera."""
    R, t = extrinsics[:, :3], extrinsics[:, 3]
    cam = points_world @ R.T + t
    pixel_h = cam @ intrinsics.T
    return pixel_h[:, :2] / pixel_h[:, 2:3]


def ground_truth_joints(take, frame):
    """Real SMPL-X joints for this photo, straight from the model, no fitting."""
    with open(DATA / take / "smplx" / f"{frame}.pkl", "rb") as f:
        params = pickle.load(f)
    import smplx

    model = smplx.create(
        str(SMPLX_ROOT),
        model_type="smplx",
        gender="neutral",
        use_pca=True,
        num_pca_comps=12,
        num_betas=10,
    )
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(params["global_orient"], dtype=torch.float32)[None],
            body_pose=torch.tensor(params["body_pose"], dtype=torch.float32)[None],
            betas=torch.tensor(params["betas"], dtype=torch.float32)[None],
            transl=torch.tensor(params["transl"], dtype=torch.float32)[None],
            left_hand_pose=torch.tensor(params["left_hand_pose"], dtype=torch.float32)[None],
            right_hand_pose=torch.tensor(params["right_hand_pose"], dtype=torch.float32)[None],
            expression=torch.tensor(params["expression"], dtype=torch.float32)[None],
        )
    return out.joints[0].numpy()  # (55, 3), world space (model's own transl already applied)


def landmarks_from_mediapipe(image_bgr):
    with POSE.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return None
    h, w = image_bgr.shape[:2]
    return np.array(
        [[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks.landmark],
        np.float32,
    )


def procrustes_align(pred, gt):
    """Rotation + scale + translation aligning pred to gt (Umeyama). Returns aligned pred."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p0, g0 = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = S.sum() / (p0**2).sum()
    return (p0 @ R.T) * scale + mu_g


def evaluate_one(take, cam_id, frame):
    intrinsics, extrinsics = load_camera(take, cam_id)
    image = cv2.imread(str(DATA / take / cam_id / "rgb_images" / f"{frame}.png"))
    if image is None:
        return None

    points = landmarks_from_mediapipe(image)
    if points is None:
        return {"take": take, "cam": cam_id, "frame": frame, "detected": False}

    gt_joints = ground_truth_joints(take, frame)
    order = sorted(JOINT_MAP)
    smplx_indices = [JOINT_MAP[i] for i in order]
    gt_used = gt_joints[smplx_indices]
    gt_pixels = project(gt_used, intrinsics, extrinsics)

    fitter = BodyFitter()
    fitter.fit(points, iterations=200, camera_iterations=100)
    own_reproj = fitter.reprojection_error(points)

    with torch.no_grad():
        fitted_joints = fitter.joints().numpy()  # (13, 3) model space, weak-perspective camera

    aligned = procrustes_align(fitted_joints, gt_used)
    mpjpe_mm = float(np.linalg.norm(aligned - gt_used, axis=1).mean() * 1000)

    detected_used = points[order][:, :2]
    detector_vs_gt_px = float(np.linalg.norm(detected_used - gt_pixels, axis=1).mean())

    return {
        "take": take,
        "cam": cam_id,
        "frame": frame,
        "detected": True,
        "own_reprojection_px": float(own_reproj),
        "detector_vs_real_camera_px": detector_vs_gt_px,
        "mpjpe_pa_mm": mpjpe_mm,
    }


def main():
    results = []
    for take in ("Take1", "Take10"):
        for cam_id in CAMERAS:
            for frame in ("00001", "00002"):
                r = evaluate_one(take, cam_id, frame)
                if r is not None:
                    results.append(r)
                    print(r)

    ok = [r for r in results if r.get("detected")]
    missed = [r for r in results if not r.get("detected")]
    print()
    print(f"{len(ok)}/{len(results)} frames: MediaPipe detected a person")
    if missed:
        for r in missed:
            print(f"  MISSED: {r['take']} {r['cam']} {r['frame']}")
    if ok:
        reproj = [r["own_reprojection_px"] for r in ok]
        det_gt = [r["detector_vs_real_camera_px"] for r in ok]
        mpjpe = [r["mpjpe_pa_mm"] for r in ok]
        print(f"own reprojection error:      mean {np.mean(reproj):.1f}px  max {np.max(reproj):.1f}px")
        print(f"detector vs real camera:     mean {np.mean(det_gt):.1f}px  max {np.max(det_gt):.1f}px")
        print(f"3D MPJPE (Procrustes-aligned): mean {np.mean(mpjpe):.1f}mm  max {np.max(mpjpe):.1f}mm")

    out_path = ROOT / "docs" / "baseline" / "real-fit-evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
