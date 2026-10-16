"""Fit a SMPL-X body to MediaPipe landmarks (SMPLify-style reprojection fit).

`BodyFitter` is stateful because a cold fit is too slow per frame: `fit`
cold-starts once, `track` warm-starts from the previous frame.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.body.landmarks import landmarks_to_targets


def _landmarks(shift_x=0.0, scale=1.0):
    """A plausible standing pose in pixels, as MediaPipe reports it."""
    from tryon.body.landmarks import POSE

    ids = POSE.PoseLandmark
    points = np.zeros((33, 3), np.float32)
    points[:, 2] = 1.0
    layout = {
        ids.NOSE: (320, 120),
        ids.LEFT_SHOULDER: (380, 200),
        ids.RIGHT_SHOULDER: (260, 200),
        ids.LEFT_ELBOW: (410, 280),
        ids.RIGHT_ELBOW: (230, 280),
        ids.LEFT_WRIST: (430, 360),
        ids.RIGHT_WRIST: (210, 360),
        ids.LEFT_HIP: (360, 360),
        ids.RIGHT_HIP: (280, 360),
        ids.LEFT_KNEE: (365, 480),
        ids.RIGHT_KNEE: (275, 480),
        ids.LEFT_ANKLE: (368, 600),
        ids.RIGHT_ANKLE: (272, 600),
    }
    centre = 320.0
    for landmark, (x, y) in layout.items():
        points[landmark.value] = (
            centre + (x - centre) * scale + shift_x,
            y * scale,
            1.0,
        )
    return points


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class BodyFitterTest(unittest.TestCase):
    def test_a_cold_fit_reduces_reprojection_error(self):
        fitter = BodyFitter()
        points = _landmarks()
        before = fitter.reprojection_error(points)
        fitter.fit(points, iterations=40)
        after = fitter.reprojection_error(points)
        self.assertLess(
            after,
            before * 0.6,
            f"fitting barely moved the body: {before:.1f} -> {after:.1f} px",
        )

    def test_tracking_keeps_up_with_a_moving_body(self):
        """3 iterations a frame must hold, or the live path cannot work."""
        fitter = BodyFitter()
        fitter.fit(_landmarks(), iterations=40)
        errors = []
        for frame in range(6):
            moved = _landmarks(shift_x=frame * 4.0)
            fitter.track(moved, iterations=3)
            errors.append(fitter.reprojection_error(moved))
        self.assertLess(
            max(errors),
            errors[0] * 2.0,
            f"tracking diverged as the body moved: {[round(e, 1) for e in errors]}",
        )

    def test_it_produces_a_posed_body_mesh(self):
        fitter = BodyFitter()
        fitter.fit(_landmarks(), iterations=10)
        vertices = fitter.vertices()
        self.assertEqual(vertices.shape[1], 3)
        self.assertGreater(len(vertices), 6000)
        self.assertTrue(np.isfinite(vertices).all())

    def test_the_pose_survives_a_missing_landmark(self):
        """A webcam loses a joint constantly; the fit must not explode."""
        from tryon.body.landmarks import POSE

        points = _landmarks()
        points[POSE.PoseLandmark.LEFT_WRIST.value, 2] = 0.0  # not visible
        fitter = BodyFitter()
        fitter.fit(points, iterations=20)
        self.assertTrue(np.isfinite(fitter.vertices()).all())

    def test_targets_ignore_landmarks_the_detector_cannot_see(self):
        from tryon.body.landmarks import POSE

        points = _landmarks()
        points[POSE.PoseLandmark.LEFT_ANKLE.value, 2] = 0.1
        visible_targets, visible_weights = landmarks_to_targets(points)
        order = sorted(__import__("tryon.body.fitter", fromlist=["JOINT_MAP"]).JOINT_MAP)
        ankle = order.index(POSE.PoseLandmark.LEFT_ANKLE.value)
        self.assertEqual(float(visible_weights[ankle]), 0.0)
        self.assertGreater(float(visible_weights.sum()), 0.0)
        self.assertEqual(len(visible_targets), len(visible_weights))


if __name__ == "__main__":
    unittest.main()




@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class DepthAmbiguityTest(unittest.TestCase):
    """A good 2D fit must not hide a body tipped over in depth.

    A leaning body projects almost like an upright one under weak
    perspective, so the orientation prior must make the fit pay for it.
    """

    def _upright_sequence_frame(self):
        import torch

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(180.0)
            truth.translation[:] = torch.tensor([[320.0, 260.0]])
            joints = truth._forward().joints[0, truth._smplx_indices]
            pixels = truth._project(joints).detach().numpy()
        from tryon.body.landmarks import JOINT_MAP

        points = np.zeros((33, 3), np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
        return points, joints.numpy()

    def test_an_upright_body_is_not_fitted_leaning_over(self):
        points, _truth = self._upright_sequence_frame()
        fitter = BodyFitter()
        fitter.fit(points, iterations=150)
        tilt = float(np.abs(fitter.global_orient.detach().numpy()).max())
        self.assertLess(
            tilt,
            0.35,
            f"the body was fitted with a {np.degrees(tilt):.0f} degree global tilt "
            "to an upright wearer",
        )

    def test_the_3d_joints_land_near_the_truth(self):
        """The check the 2D number cannot give: error in metres, not pixels."""
        import torch

        points, truth = self._upright_sequence_frame()
        fitter = BodyFitter()
        fitter.fit(points, iterations=150)
        with torch.no_grad():
            fitted = fitter._forward().joints[0, fitter._smplx_indices].numpy()
        error_mm = float(np.linalg.norm(fitted - truth, axis=1).mean() * 1000)
        self.assertLess(error_mm, 120.0, f"3D joint error {error_mm:.0f} mm")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class FacingAwayTest(unittest.TestCase):
    """A wearer seen from behind must be fitted turned around, not stalled.

    The prior constrains pitch and roll and leaves yaw free; the cold fit
    tries both facings.
    """

    def _turned_frame(self, yaw):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(180.0)
            truth.translation[:] = torch.tensor([[320.0, 260.0]])
            truth.global_orient[0, 1] = yaw
            joints = truth._forward().joints[0, truth._smplx_indices]
            pixels = truth._project(joints).detach().numpy()
        points = np.zeros((33, 3), np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
        return points

    def test_a_body_facing_away_is_fitted_facing_away(self):
        points = self._turned_frame(np.pi)
        fitter = BodyFitter()
        fitter.fit(points)
        yaw = float(fitter.global_orient.detach().numpy()[0, 1])
        self.assertLess(
            fitter.reprojection_error(points),
            8.0,
            f"back-facing body fitted at {fitter.reprojection_error(points):.1f} px",
        )
        self.assertGreater(
            abs(yaw), np.radians(150), f"yaw {np.degrees(yaw):.0f} deg, expected ~180"
        )

    def test_a_body_facing_the_camera_still_is(self):
        points = self._turned_frame(0.0)
        fitter = BodyFitter()
        fitter.fit(points)
        yaw = float(fitter.global_orient.detach().numpy()[0, 1])
        self.assertLess(abs(yaw), np.radians(30))

    def test_upright_is_still_enforced(self):
        """Freeing yaw must not free pitch: the body must not lean over."""
        points = self._turned_frame(0.0)
        fitter = BodyFitter()
        fitter.fit(points)
        pitch_roll = fitter.global_orient.detach().numpy()[0, [0, 2]]
        self.assertLess(float(np.abs(pitch_roll).max()), 0.35)


class TurningTest(unittest.TestCase):
    """A body turning through profile, the case a single frame cannot fit.

    Warm-started `track()` gets few iterations per frame, so the default
    iteration count must still follow a fast turn toward profile.
    """

    def _turning_sequence(self, frames, target_yaw_deg, noise_px=1.5, seed=0):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(180.0)
            truth.translation[:] = torch.tensor([[320.0, 260.0]])
        target_yaw = np.radians(target_yaw_deg)
        observations = []
        for frame in range(frames):
            yaw = target_yaw * frame / (frames - 1)
            with torch.no_grad():
                truth.global_orient.zero_()
                truth.global_orient[0, 1] = yaw
                joints = truth._forward().joints[0, truth._smplx_indices]
                pixels = truth._project(joints).detach().numpy()
            points = np.zeros((33, 3), np.float32)
            points[:, 2] = 1.0
            rng = np.random.default_rng(seed * 1000 + frame)
            for k, index in enumerate(sorted(JOINT_MAP)):
                points[index, :2] = pixels[k] + rng.normal(0, noise_px, 2)
            observations.append(points)
        return observations

    def _track_through(self, observations, iterations):
        fitter = BodyFitter()
        fitter.fit(observations[0], iterations=150, camera_iterations=80)
        for points in observations[1:]:
            fitter.track(points, iterations=iterations)
        return float(fitter.global_orient.detach().numpy()[0, 1])

    def test_more_iterations_follows_the_turn_better_than_the_default(self):
        """15 iterations a frame is no worse than 3, within one degree."""
        observations = self._turning_sequence(frames=30, target_yaw_deg=-68.0)
        default_yaw = self._track_through(observations, iterations=3)
        raised_yaw = self._track_through(observations, iterations=15)
        target = np.radians(-68.0)
        default_error = abs(default_yaw - target)
        raised_error = abs(raised_yaw - target)
        self.assertLessEqual(
            raised_error,
            default_error + np.radians(1.0),
            f"15 iterations ({np.degrees(raised_yaw):.0f} deg) did not track the turn "
            f"better than 3 ({np.degrees(default_yaw):.0f} deg); true yaw -68 deg",
        )

    def test_a_realistic_turn_speed_converges_with_the_live_default(self):
        """A 3-second turn reaches close to the true yaw at TRACK_ITERATIONS."""
        observations = self._turning_sequence(frames=90, target_yaw_deg=-68.0)
        final_yaw = self._track_through(observations, iterations=15)
        error_degrees = abs(np.degrees(final_yaw) - (-68.0))
        self.assertLess(
            error_degrees, 10.0, f"final yaw {np.degrees(final_yaw):.1f} deg, true -68"
        )


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class RefitTest(unittest.TestCase):
    """`BodyFitter.refit`: a cold fit that may only improve on the pose."""

    def _view(self, yaw):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(300.0)
            truth.translation[:] = torch.tensor([[320.0, 430.0]])
            truth.global_orient[0, 1] = yaw
            pixels = truth._project(truth._forward().joints[0, truth._smplx_indices]).numpy()
        points = np.zeros((33, 3), np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
        return points

    def test_a_cut_to_the_back_is_recovered(self):
        front, back = self._view(0.0), self._view(np.pi)
        fitter = BodyFitter()
        fitter.fit(front, orientation_hypotheses=(0.0,))
        fitter.track(back, iterations=15)
        stuck = fitter.reprojection_error(back)
        self.assertTrue(fitter.refit(back))
        yaw = float(fitter.global_orient.detach()[0, 1]) % (2 * np.pi)
        self.assertLess(abs(yaw - np.pi), np.radians(20), f"yaw {np.degrees(yaw):.0f} deg after the refit")
        self.assertLess(fitter.reprojection_error(back), stuck)

    def test_a_worse_refit_is_rolled_back(self):
        import torch

        front = self._view(0.0)
        fitter = BodyFitter()
        fitter.fit(front, orientation_hypotheses=(0.0,))
        before = [p.detach().clone() for p in (fitter.body_pose, fitter.global_orient, fitter.scale, fitter.translation)]
        # Force the refit to be worse: it may only use a wrong facing.
        original = fitter.fit
        fitter.fit = lambda points, orientation_hypotheses=None, **_: original(points, orientation_hypotheses=(np.pi,), iterations=5, camera_iterations=5)
        adopted = fitter.refit(front)
        self.assertFalse(adopted)
        for p, b in zip((fitter.body_pose, fitter.global_orient, fitter.scale, fitter.translation), before):
            self.assertTrue(torch.allclose(p.detach(), b), "the old pose was not restored")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class BoneDepthTest(unittest.TestCase):
    """Depth from MediaPipe's world landmarks (4th column, DEPTH_BONES)
    resolves an arm angled into the screen that 2D alone cannot."""

    def _frame_with_an_arm_into_depth(self):
        import torch

        from tryon.body.landmarks import JOINT_MAP, POSE

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(180.0)
            truth.translation[:] = torch.tensor([[320.0, 260.0]])
            # left shoulder rotated so the arm points toward the camera
            truth.body_pose[0, (16 - 1) * 3 : (16 - 1) * 3 + 3] = torch.tensor([0.0, -1.1, 0.0])
            joints = truth._forward().joints[0, truth._smplx_indices]
            pixels = truth._project(joints).detach().numpy()
        points = np.full((33, 4), np.nan, np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
            points[index, 3] = -float(joints[k, 2])  # MediaPipe: smaller z = closer
        ids = POSE.PoseLandmark
        order = sorted(JOINT_MAP)
        truth_dz = float(joints[order.index(ids.LEFT_WRIST.value), 2] - joints[order.index(ids.LEFT_ELBOW.value), 2])
        return points, truth_dz

    def _fitted_dz(self, fitter):
        import torch

        from tryon.body.landmarks import JOINT_MAP, POSE

        ids = POSE.PoseLandmark
        order = sorted(JOINT_MAP)
        with torch.no_grad():
            joints = fitter._forward().joints[0, fitter._smplx_indices].numpy()
        return float(joints[order.index(ids.LEFT_WRIST.value), 2] - joints[order.index(ids.LEFT_ELBOW.value), 2])

    def test_a_three_column_caller_is_unchanged(self):
        from tryon.body.landmarks import landmarks_to_depths

        _targets, weights = landmarks_to_depths(np.zeros((33, 3)))
        self.assertEqual(weights.sum(), 0.0)

    def test_the_depth_column_recovers_the_forearm_depth(self):
        """With exact depth targets the forearm depth is recovered."""
        import tryon.body.landmarks as landmarks

        points, truth_dz = self._frame_with_an_arm_into_depth()
        self.assertGreater(abs(truth_dz), 0.08, "the fixture's forearm really points into depth")
        shrink = landmarks.DEPTH_SHRINK
        landmarks.DEPTH_SHRINK = 1.0
        try:
            with_depth = BodyFitter()
            with_depth.fit(points, iterations=150)
        finally:
            landmarks.DEPTH_SHRINK = shrink
        without = BodyFitter()
        without.fit(points[:, :3], iterations=150)
        err_with = abs(self._fitted_dz(with_depth) - truth_dz)
        err_without = abs(self._fitted_dz(without) - truth_dz)
        self.assertLess(err_with, 0.06, f"forearm depth error {err_with * 100:.1f} cm with the depth term")
        self.assertLess(err_with, err_without, f"with {err_with * 100:.1f} cm vs without {err_without * 100:.1f} cm")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class TorsoDepthTest(unittest.TestCase):
    """The torso's facing from MediaPipe's world z (TORSO_DEPTH_BONES)."""

    def _profile_frame(self, yaw_deg):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(230.0)
            truth.translation[:] = torch.tensor([[320.0, 400.0]])
            truth.global_orient[0, 1] = np.radians(yaw_deg)
            joints = truth._forward().joints[0, truth._smplx_indices]
            pixels = truth._project(joints).detach().numpy()
        points = np.full((33, 4), np.nan, np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
            points[index, 3] = -float(joints[k, 2])  # MediaPipe: smaller z = closer
        return points

    @staticmethod
    def _shoulder_yaw(fitter):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        order = sorted(JOINT_MAP)
        with torch.no_grad():
            joints = fitter.joints().numpy()
        d = joints[order.index(11)] - joints[order.index(12)]
        return float(np.degrees(np.arctan2(-d[2], d[0])))

    def test_the_shoulder_depth_turns_the_body_sideways(self):
        for yaw in (75.0, -75.0):
            with self.subTest(yaw=yaw):
                points = self._profile_frame(yaw)
                with_depth = BodyFitter()
                with_depth.fit(points, iterations=150)
                error = abs(self._shoulder_yaw(with_depth) - yaw)
                self.assertLess(error, 12.0, f"shoulders {self._shoulder_yaw(with_depth):.0f} vs {yaw:.0f} degrees with the torso depth")

    def test_the_torso_depth_has_its_own_targets(self):
        from tryon.body.landmarks import DEPTH_BONES, TORSO_DEPTH_BONES, landmarks_to_depths

        targets, weights = landmarks_to_depths(self._profile_frame(75.0))
        self.assertEqual(len(targets), len(DEPTH_BONES) + len(TORSO_DEPTH_BONES))
        self.assertTrue((weights[len(DEPTH_BONES):] > 0).all())
        self.assertGreater(abs(targets[len(DEPTH_BONES)]), 0.2, "shoulders 75 degrees round are 0.3 m apart in depth")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class MeshFitTest(unittest.TestCase):
    """`fit_mesh`: recover pose and shape from a SMPL-X-topology mesh in
    3D (the scan-frame body in the Gaussian-Garments trajectory files)."""

    def test_recovers_a_posed_body_to_within_a_centimetre(self):
        import torch

        truth = BodyFitter()
        with torch.no_grad():
            truth.betas[0, 0] = 1.5
            truth.body_pose[0, (16 - 1) * 3 + 2] = -0.5  # left arm down
            truth.body_pose[0, (17 - 1) * 3 + 2] = 0.8  # right arm further down
            truth.body_pose[0, (18 - 1) * 3 + 1] = -0.6  # left elbow bent
            truth.global_orient[0, 1] = 0.3  # turned a little
            target = truth._forward().vertices[0].numpy() + np.array([0.2, 1.3, -0.1], np.float32)
        fitter = BodyFitter()
        residual = fitter.fit_mesh(target, iterations=400)
        self.assertLess(residual, 0.01, f"mean vertex residual {residual * 100:.2f} cm")
        with torch.no_grad():
            joints = fitter._forward().joints[0].numpy()  # _forward applies mesh_offset
            truth_joints = truth._forward().joints[0].numpy() + np.array([0.2, 1.3, -0.1], np.float32)
        self.assertLess(float(np.linalg.norm(joints[:22] - truth_joints[:22], axis=1).max()), 0.02)

    def test_rejects_a_mesh_of_the_wrong_size(self):
        fitter = BodyFitter()
        with self.assertRaises(ValueError):
            fitter.fit_mesh(np.zeros((100, 3)))


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class ProfileFacingTest(FacingAwayTest):
    """A wearer in pure profile: the cold fit also seeds the two profiles."""

    def test_a_body_in_profile_is_fitted_sideways(self):
        for yaw in (np.pi / 2, -np.pi / 2):
            points = self._turned_frame(yaw)
            fitter = BodyFitter()
            fitter.fit(points, iterations=150)
            fitted = float(fitter.global_orient[0, 1])
            error = abs((fitted - yaw + np.pi) % (2 * np.pi) - np.pi)
            self.assertLess(np.degrees(error), 25.0, f"yaw {np.degrees(yaw):.0f}: fitted {np.degrees(fitted):.0f} degrees")


class LandmarkYawTest(unittest.TestCase):
    """`landmarks.landmark_yaw`: the torso's facing from MediaPipe's world landmarks."""

    def _points(self, yaw_deg):
        points = np.zeros((33, 5), np.float32)
        a = np.radians(yaw_deg)
        for left, right, half in ((11, 12, 0.18), (23, 24, 0.12)):
            points[left, 4], points[left, 3] = half * np.cos(a), half * np.sin(a)
            points[right, 4], points[right, 3] = -half * np.cos(a), -half * np.sin(a)
        return points

    def test_yaw_is_recovered_in_the_fitters_convention(self):
        from tryon.body.landmarks import landmark_yaw, yaw_difference

        for deg in (-100, -16, 0, 69, 172):
            self.assertLess(np.degrees(yaw_difference(landmark_yaw(self._points(deg)), np.radians(deg))), 1.0)

    def test_old_four_column_landmarks_give_no_yaw(self):
        from tryon.body.landmarks import landmark_yaw

        self.assertIsNone(landmark_yaw(np.zeros((33, 4), np.float32)))

    def test_yaw_difference_wraps(self):
        from tryon.body.landmarks import yaw_difference

        self.assertAlmostEqual(np.degrees(yaw_difference(np.radians(175), np.radians(-175))), 10.0, places=4)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class FitFromRegressionTest(unittest.TestCase):
    """`BodyFitter.fit_from_regression`: an image regressor's body placed in
    the frame - pose, shape and facing kept, only the camera solved."""

    def _truth(self):
        import torch

        truth = BodyFitter()
        with torch.no_grad():
            truth.scale.fill_(300.0)
            truth.translation[:] = torch.tensor([[320.0, 430.0]])
            truth.global_orient[0, 1] = 0.6
            truth.body_pose[0, 47] = -0.8  # left elbow
            truth.betas[0, 0] = 1.0
        return truth

    def _prediction(self, truth):
        import torch

        with torch.no_grad():
            joints_2d = truth._project(truth.joints(list(range(22)))).numpy()
        return {"global_orient": truth.global_orient.detach().numpy()[0], "body_pose": truth.body_pose.detach().numpy()[0],
                "betas": truth.betas.detach().numpy()[0], "joints_2d": joints_2d}

    def _points(self, truth):
        import torch

        from tryon.body.landmarks import JOINT_MAP

        with torch.no_grad():
            pixels = truth._project(truth.joints()).numpy()
        points = np.zeros((33, 3), np.float32)
        points[:, 2] = 1.0
        for k, index in enumerate(sorted(JOINT_MAP)):
            points[index, :2] = pixels[k]
        return points

    def test_the_regressor_placement_recovers_the_camera(self):
        truth = self._truth()
        fitter = BodyFitter()
        fitter.fit_from_regression(self._points(truth), self._prediction(truth))
        self.assertAlmostEqual(float(fitter.scale), 300.0, delta=1.0)
        np.testing.assert_allclose(fitter.translation.detach().numpy(), [[320.0, 430.0]], atol=1.0)
        np.testing.assert_allclose(fitter.body_pose.detach().numpy(), truth.body_pose.detach().numpy(), atol=1e-6)
        self.assertLess(fitter.reprojection_error(self._points(truth)), 1.0)

    def test_joints_outside_the_image_do_not_move_the_placement(self):
        truth = self._truth()
        prediction = self._prediction(truth)
        # a wearer cut off at the waist: the regressor's legs are a guess,
        # here a wrong one, below the bottom of a 600 px frame
        legs = [4, 5, 7, 8, 10, 11]
        prediction["joints_2d"][legs, 1] += 300.0
        self.assertTrue((prediction["joints_2d"][legs, 1] > 600).all())
        prediction["image_size"] = (600, 640)
        fitter = BodyFitter()
        fitter.fit_from_regression(self._points(truth), prediction)
        self.assertAlmostEqual(float(fitter.scale), 300.0, delta=2.0)
        np.testing.assert_allclose(fitter.translation.detach().numpy(), [[320.0, 430.0]], atol=2.0)

    def test_the_keypoint_placement_keeps_the_pose(self):
        import tryon.body.fitter as bodyfit

        truth = self._truth()
        prediction = self._prediction(truth)
        del prediction["joints_2d"]
        fitter = BodyFitter()
        fitter.fit_from_regression(self._points(truth), prediction)
        np.testing.assert_allclose(fitter.body_pose.detach().numpy(), truth.body_pose.detach().numpy(), atol=1e-6)
        self.assertLess(fitter.reprojection_error(self._points(truth)), 3.0)
        self.assertEqual(bodyfit.REGRESSION_CAMERA, "regressor")
