"""ScanPoseBinder.size_factors: how the wearer/scan size ratios scale the garment."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.garment.scan_pose import ScanPoseBinder
from tryon.garment.skinning import SkinnedGarment


class SizeFactorsTest(unittest.TestCase):
    def factors(self, mode, height=1.2, chest=1.3, widen=False):
        return ScanPoseBinder.size_factors(mode, height, chest, widen)

    def test_current_scales_all_axes_by_the_height_ratio(self):
        np.testing.assert_allclose(self.factors("current"), [1.2, 1.2, 1.2])

    def test_current_widens_only_when_the_renderer_asks(self):
        np.testing.assert_allclose(self.factors("current", widen=True), [1.2 * 1.3, 1.2, 1.2 * 1.3])
        np.testing.assert_allclose(self.factors("current", chest=0.8, widen=True), [1.2, 1.2, 1.2])

    def test_no_chest_ignores_the_chest_ratio(self):
        np.testing.assert_allclose(self.factors("no_chest", widen=True), [1.2, 1.2, 1.2])

    def test_none_is_identity(self):
        np.testing.assert_allclose(self.factors("none"), [1, 1, 1])

    def test_shrink_never_grows_the_garment(self):
        np.testing.assert_allclose(self.factors("shrink", height=1.2), [1, 1, 1])
        np.testing.assert_allclose(self.factors("shrink", height=0.8), [0.8, 0.8, 0.8])

    def test_damped_is_the_square_root(self):
        np.testing.assert_allclose(self.factors("damped", height=1.44), [1.2, 1.2, 1.2])

    def test_height_y_leaves_the_width_alone(self):
        np.testing.assert_allclose(self.factors("height_y"), [1, 1.2, 1])

    def test_widen_never_narrows(self):
        np.testing.assert_allclose(self.factors("widen", chest=0.8), [1.2, 1.2, 1.2])

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.factors("bogus")


def _arm_down(theta):
    """A body with the shoulders `theta` degrees below horizontal."""
    fitter = BodyFitter()
    with torch.no_grad():
        fitter.body_pose[0, (16 - 1) * 3 + 2] = -np.radians(theta)
        fitter.body_pose[0, (17 - 1) * 3 + 2] = +np.radians(theta)
    return fitter


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class RefineArmsTest(unittest.TestCase):
    """`ScanPoseBinder.refine_arms`: arms bound at the wrong angle turn into the sleeves."""

    @classmethod
    def setUpClass(cls):
        truth = _arm_down(30.0)
        with torch.no_grad():
            body = truth._forward().vertices[0].numpy().astype(np.float64)
        weights = truth.model.lbs_weights.detach().numpy()
        cls.is_arm = weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
        cls.sleeve_points = body[cls.is_arm]  # a sleeve lying on the true arms
        cls.sleeve = np.ones(len(cls.sleeve_points), bool)

    def binder(self, mode):
        binder = ScanPoseBinder(np.zeros((1, 3)), np.zeros((0, 3), int), None)
        binder.REFINE_ARMS = mode
        return binder

    def distance(self, fitter):
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        return ScanPoseBinder.sleeve_arm_distance(body, self.is_arm, self.sleeve_points, self.sleeve)

    def test_arms_bound_at_the_wrong_angle_move_into_the_sleeves(self):
        fitter = _arm_down(75.0)
        before = self.distance(fitter)
        self.assertGreater(before, 0.05)
        self.assertTrue(self.binder("always").refine_arms(fitter, self.sleeve_points, self.sleeve))
        self.assertLess(self.distance(fitter), 0.5 * before)

    def test_arms_already_in_their_sleeves_are_left_alone(self):
        fitter = _arm_down(30.0)
        pose = fitter.body_pose.detach().clone()
        self.assertFalse(self.binder("always").refine_arms(fitter, self.sleeve_points, self.sleeve))
        self.assertTrue(torch.equal(fitter.body_pose, pose))

    def test_off_and_unflagged_do_nothing(self):
        for mode, flagged in ((None, True), ("refined", False)):
            fitter = _arm_down(75.0)
            pose = fitter.body_pose.detach().clone()
            self.assertFalse(self.binder(mode).refine_arms(fitter, self.sleeve_points, self.sleeve, flagged))
            self.assertTrue(torch.equal(fitter.body_pose, pose))

    def test_a_flagged_pose_is_refined_in_refined_mode(self):
        fitter = _arm_down(75.0)
        self.assertTrue(self.binder("refined").refine_arms(fitter, self.sleeve_points, self.sleeve, True))

    def test_poor_mode_waits_for_a_poor_fit(self):
        self.assertFalse(self.binder("poor").refine_arms(_arm_down(34.0), self.sleeve_points, self.sleeve))
        self.assertTrue(self.binder("poor").refine_arms(_arm_down(75.0), self.sleeve_points, self.sleeve))


if __name__ == "__main__":
    unittest.main()
