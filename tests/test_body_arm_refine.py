"""Arm refinement of a regressed pose toward MediaPipe's elbows and wrists (`BodyFitter._refine_arms`)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.body.regressor import _rotation, refine_arm


def _landmarks(fitter):
    """33 MediaPipe-like landmarks (torso, elbows, wrists) at the fitter's own projected joints."""
    with torch.no_grad():
        pixels = fitter._project(fitter.joints(list(range(22)))).numpy().astype(np.float64)
    points = np.zeros((33, 5), np.float64)
    for landmark, joint in ((11, 16), (12, 17), (13, 18), (14, 19), (15, 20), (16, 21), (23, 1), (24, 2)):
        points[landmark, :2], points[landmark, 2] = pixels[joint], 1.0
    return points, pixels


def _posed(pose):
    fitter = BodyFitter()
    with torch.no_grad():
        for joint, vector in pose.items():
            fitter.body_pose[0, (joint - 1) * 3: joint * 3] = torch.tensor(vector, dtype=torch.float32)
        fitter.scale.fill_(300.0)
        fitter.translation.copy_(torch.tensor([[400.0, 300.0]]))
    return fitter


class RotationTest(unittest.TestCase):
    def test_quarter_turn_about_z(self):
        np.testing.assert_allclose(_rotation(np.array([0.0, 0.0, np.pi / 2])) @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_zero_is_identity(self):
        np.testing.assert_array_equal(_rotation(np.zeros(3)), np.eye(3))


class RefineArmTest(unittest.TestCase):
    def test_recovers_a_known_elbow_and_wrist(self):
        frames = {"R_collar": np.eye(3), "shoulder": np.zeros(3), "o_elbow": np.array([0.0, -0.28, 0.0]), "o_wrist": np.array([0.0, -0.25, 0.0])}
        truth = np.array([0.4, 0.2, 0.9, 0.0, -1.3, 0.0])

        def project(a):
            r = _rotation(a[:3])
            elbow = r @ frames["o_elbow"]
            wrist = elbow + (r @ _rotation(a[3:])) @ frames["o_wrist"]
            return {k: v[:2] * (1.0, -1.0) * 300.0 + (400.0, 300.0) for k, v in (("elbow", elbow), ("wrist", wrist))}

        goal = project(truth)
        start = {**frames, "a_shoulder": truth[:3] + 0.5, "a_elbow": truth[3:] + [0.0, 0.5, 0.0]}
        shoulder, elbow, before, after = refine_arm(start, goal, 250.0, (300.0, np.array([400.0, 300.0])), -1.0)
        got = project(np.concatenate([shoulder, elbow]))
        self.assertLess(after, 0.3 * before)
        for key in goal:
            self.assertLess(np.linalg.norm(got[key] - goal[key]) / 250.0, 0.1)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class RefineArmsTest(unittest.TestCase):
    def test_a_perturbed_arm_goes_back_to_the_landmarks(self):
        truth = {16: [0.1, 0.3, -0.6], 18: [0.0, -1.5, 0.0], 17: [0.0, 0.0, 0.3], 19: [0.0, 1.0, 0.0]}
        wanted, pixels = _landmarks(_posed(truth))
        perturbed = _posed({**truth, 18: [0.0, -0.3, 0.0], 16: [0.1, 0.9, 0.3]})
        perturbed.ARM_GATE = 0.12  # this perturbation is a small one
        before = float(np.linalg.norm(_landmarks(perturbed)[1][20] - pixels[20]))
        touched = perturbed._refine_arms(wanted)
        after = float(np.linalg.norm(_landmarks(perturbed)[1][20] - pixels[20]))
        self.assertEqual(touched, ["left"])  # the right arm already agrees
        self.assertLess(after, 0.25 * before)
        self.assertLess(after / 250.0, 0.06)

    def test_arms_that_agree_are_not_touched(self):
        fitter = _posed({16: [0.1, 0.3, -0.6], 18: [0.0, -1.5, 0.0]})
        wanted, _ = _landmarks(fitter)
        pose = fitter.body_pose.clone()
        self.assertEqual(fitter._refine_arms(wanted), [])
        self.assertTrue(torch.equal(pose, fitter.body_pose))

    def test_swapped_wrists_are_left_alone(self):
        fitter = _posed({16: [0.1, 0.3, -0.6], 18: [0.0, -1.5, 0.0], 17: [0.0, -0.6, 0.3], 19: [0.0, 1.5, 0.0]})
        wanted, _ = _landmarks(fitter)
        wanted[[15, 16], :2] = wanted[[16, 15], :2]
        pose = fitter.body_pose.clone()
        self.assertEqual(fitter._refine_arms(wanted), [])
        self.assertTrue(torch.equal(pose, fitter.body_pose))

    def test_a_large_disagreement_is_fixed_with_the_hips_out_of_the_frame(self):
        truth = {16: [0.1, 0.3, -0.6], 18: [0.0, -1.5, 0.0], 17: [0.0, 0.0, 0.3], 19: [0.0, 1.0, 0.0]}
        wanted, pixels = _landmarks(_posed(truth))
        wanted = wanted.copy()
        wanted[[23, 24], 2] = 0.0  # the hips below the frame
        perturbed = _posed({**truth, 18: [0.0, -0.3, 0.0], 16: [0.1, 0.9, 0.3]})
        before = float(np.linalg.norm(_landmarks(perturbed)[1][20] - pixels[20]))
        touched = perturbed._refine_arms(wanted)
        after = float(np.linalg.norm(_landmarks(perturbed)[1][20] - pixels[20]))
        self.assertEqual(touched, ["left"])
        self.assertLess(after, 0.25 * before)

    def test_on_by_default_with_a_gate_that_leaves_small_disagreements_alone(self):
        self.assertTrue(BodyFitter.ARM_REFINE)
        self.assertGreaterEqual(BodyFitter.ARM_GATE, 0.3)


if __name__ == "__main__":
    unittest.main()
