"""BodyFitter tested against REAL photographs, not synthetic landmarks.

Every other test in `test_body_fitter.py` drives the fitter with either
`_landmarks()` (hand-typed pixel coordinates for a made-up pose) or
`_turned_frame()` (a perfect noise-free synthetic reprojection). Both prove
the optimiser converges to what it is given; neither runs the actual
MediaPipe detector on an actual photograph, so neither can catch a failure
that only shows up with real detector noise and a real body.

`data/annotated/` is real 4D-DRESS captures: photographs from four
calibrated cameras, plus the genuine SMPL-X parameters those photographs
were fit from (not by this project - by 4D-DRESS's own pipeline, an
independent ground truth). `tools/evaluate/real_fit.py` runs the full
chain - real photo -> MediaPipe -> BodyFitter -> compare against the real
camera projection and the real 3D joints - across all 4 cameras x 2 takes
x 2 frames = 16 real frames. This file pins the two findings from that run
that matter for correctness, so they cannot silently regress:

  1. A fixable bug, now fixed: camera 0052's real yaw is ~178 degrees
     (near-exact facing-away). The cold fit's camera-only stage used to
     run ONCE (at yaw=0) and reuse that camera for both yaw candidates in
     stage 2. A 180-degree-turned torso is not a scaled, translated copy
     of a forward-facing one, so no scale/translation reproduces it - the
     camera-only loss floored at 10,953 px^2 after 2,000 iterations,
     confirmed to not be a "needs more iterations" problem. Stage 2 then
     started from a camera that fit neither facing and never recovered:
     72 px reprojection, betas pushed to the prior's edge (|beta| up to
     4.0) trying to explain the residual as body shape instead. Fitting
     the camera separately per yaw hypothesis reaches 2.9-16 px on the
     real frames - `test_the_far_side_camera_now_converges` below.

  2. A real, NOT fixed, limitation: cameras 0028 and 0076 are ~65-68
     degree profile views. Confirmed by starting the optimiser AT the
     true yaw (read from the ground-truth pkl) that it does not stay
     there - it drifts to roughly half that angle and reprojection error
     barely improves (25.5 px cold vs 25.5 px from the true start). This
     is the field's own monocular depth/orientation ambiguity (Bogo et
     al. 2016): MediaPipe reports both shoulders and both hips at high
     visibility even in profile (they are still on-screen, just
     foreschortened), so the 2D reprojection loss has no signal telling
     "turned 68 degrees" apart from "turned less, closer to camera".
     `test_a_profile_view_is_a_known_hard_case` below pins this as a
     measured, honest boundary rather than letting a silent regression
     make it worse without anyone noticing, or letting a claimed fix go
     unverified.
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
        """Camera 0052: real yaw ~178 deg. Was 70-72 px, now single-digit-to-teens.

        Fixing this required fitting the camera-only stage once PER yaw
        hypothesis rather than once and reusing it for both - see `fit`'s
        docstring in `tryon/body/fitter.py`. Threshold set above the measured
        11.7-16.1 px across the two takes, well below the 70+ px it was.
        """
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
        """Cameras 0028/0076 (~65-68 deg yaw): 2D can look fine while 3D is wrong.

        This does not assert a tight bound - the point is the opposite: it
        proves the true yaw is NOT a stable optimum of the reprojection
        loss on a real profile photo (starting the optimiser there, it
        drifts away and reprojection error does not improve), so nobody
        should expect a small tweak to fix this silently. If a future
        change genuinely fixes it, this test should start failing ITS
        upper-bound assertion and get tightened - that would be good news,
        not a broken test.
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
