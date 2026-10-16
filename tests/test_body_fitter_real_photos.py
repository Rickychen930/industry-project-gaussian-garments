"""BodyFitter against real photographs rather than synthetic landmarks.

`data/annotated/` holds 4D-DRESS captures with independent SMPL-X ground
truth. This file pins two findings from `tools/evaluate/real_fit.py`:

1. The cold fit's camera-only stage must run once per yaw hypothesis; a
   180-degree-turned torso is not a scaled copy of a forward-facing one.
2. Profile views (~65-68 degrees) are a known unsolved monocular
   ambiguity, pinned as a measured boundary.
"""

from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path

import numpy as np

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.body.landmarks import JOINT_MAP

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "annotated"
MODELS_PRESENT = SMPLX_ROOT.is_dir()
DATA_PRESENT = DATA.is_dir()

try:
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401

    DEPS_PRESENT = True
except ImportError:
    DEPS_PRESENT = False


def _load_camera(take, cam_id):
    with open(DATA / take / "cameras.json") as f:
        cams = json.load(f)
    c = cams[cam_id]
    return np.array(c["intrinsics"]), np.array(c["extrinsics"])


def _landmarks_from_real_photo(take, cam_id, frame):
    import cv2

    from tryon.body.landmarks import POSE

    image = cv2.imread(str(DATA / take / cam_id / "rgb_images" / f"{frame}.png"))
    with POSE.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return None
    h, w = image.shape[:2]
    return np.array(
        [[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks.landmark],
        np.float32,
    )


def _ground_truth_yaw(take, frame, cam_id):
    """The real yaw (radians) of the SMPL-X root, as camera `cam_id` sees it."""
    with open(DATA / take / "smplx" / f"{frame}.pkl", "rb") as f:
        params = pickle.load(f)
    from scipy.spatial.transform import Rotation

    world_R = Rotation.from_rotvec(params["global_orient"]).as_matrix()
    _intrinsics, extrinsics = _load_camera(take, cam_id)
    cam_R = extrinsics[:, :3] @ world_R
    _pitch, yaw, _roll = Rotation.from_matrix(cam_R).as_euler("xyz")
    return float(yaw)


@unittest.skipUnless(MODELS_PRESENT, f"no SMPL-X body models at {SMPLX_ROOT}")
@unittest.skipUnless(DATA_PRESENT, f"no annotated data at {DATA}")
@unittest.skipUnless(DEPS_PRESENT, "opencv/mediapipe not installed")
class RealPhotoFitTest(unittest.TestCase):
    def test_the_far_side_camera_now_converges(self):
        """Camera 0052 (yaw ~178 deg) converges; the camera is fitted per yaw hypothesis."""
        for take in ("Take1", "Take10"):
            with self.subTest(take=take):
                points = _landmarks_from_real_photo(take, "0052", "00001")
                self.assertIsNotNone(points, "MediaPipe did not detect a person")
                fitter = BodyFitter()
                fitter.fit(points, iterations=200, camera_iterations=100)
                error = fitter.reprojection_error(points)
                self.assertLess(
                    error, 25.0, f"{take} camera 0052: {error:.1f} px (was ~70-72 px)"
                )

    def test_a_profile_view_is_a_known_hard_case(self):
        """Cameras 0028/0076 (~65-68 deg): the true yaw is not a stable optimum.

        If a fix makes this fail its upper bound, tighten the bound.
        """
        take, cam_id, frame = "Take1", "0028", "00001"
        points = _landmarks_from_real_photo(take, cam_id, frame)
        self.assertIsNotNone(points, "MediaPipe did not detect a person")

        true_yaw = _ground_truth_yaw(take, frame, cam_id)
        self.assertGreater(
            abs(np.degrees(true_yaw)), 45.0, "fixture assumption: this should be a profile view"
        )

        from tryon.body.landmarks import _TORSO, landmarks_to_targets
        import torch

        targets, weights = landmarks_to_targets(points)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        order = sorted(JOINT_MAP)
        torso = torch.tensor([1.0 if idx in _TORSO else 0.0 for idx in order])
        torso_weights = weight_tensor * torso
        visible = weights > 0
        centre = torch.tensor(targets[visible].mean(axis=0), dtype=torch.float32)

        fitter = BodyFitter()
        with torch.no_grad():
            fitter.global_orient.zero_()
            fitter.global_orient[0, 1] = true_yaw
            fitter.scale.copy_(torch.tensor([250.0]))
            fitter.translation.copy_(centre)
        fitter._optimiser = None
        fitter._fit_camera(target_tensor, weight_tensor, torso_weights, 100)
        for _ in range(200):
            fitter._step(target_tensor, weight_tensor, 0.02, shape=True)

        converged_yaw = float(fitter.global_orient.detach().numpy()[0, 1])
        drift_degrees = abs(np.degrees(converged_yaw) - np.degrees(true_yaw))
        self.assertGreater(
            drift_degrees,
            10.0,
            f"started at the true yaw ({np.degrees(true_yaw):.0f} deg) and stayed within "
            f"{drift_degrees:.0f} deg - if this holds, the profile-view limitation this test "
            "documents may be fixed; tighten the assertion above instead of leaving this loose.",
        )
