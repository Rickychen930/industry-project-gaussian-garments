"""The synthetic orientation-training data generator,
checked for sanity before anything is trained on it.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.body.fitter import SMPLX_ROOT
from tryon.body.orientation_data import (
    N_BINS,
    _JOINT_ORDER,
    bin_center,
    make_dataset,
    sample_pose_and_keypoints,
    yaw_to_bin,
)


class YawBinTest(unittest.TestCase):
    def test_bin_zero_centred_on_zero_yaw(self):
        self.assertEqual(yaw_to_bin(0.0), 0)

    def test_bin_center_roundtrips(self):
        for bin_index in range(N_BINS):
            center = bin_center(bin_index)
            self.assertEqual(yaw_to_bin(center), bin_index)

    def test_bins_partition_the_full_circle(self):
        # Sweeping yaw across the whole range should visit every bin
        # exactly proportionally - no gaps, no bin unreachable.
        yaws = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
        seen_bins = {yaw_to_bin(y) for y in yaws}
        self.assertEqual(seen_bins, set(range(N_BINS)))

    def test_adjacent_yaws_are_same_or_adjacent_bin(self):
        # A small change in yaw must not jump more than one bin - the
        # binning itself must be monotonic/contiguous, not scrambled.
        yaws = np.linspace(-np.pi, np.pi - 1e-6, 2000)
        bins = [yaw_to_bin(y) for y in yaws]
        for a, b in zip(bins, bins[1:]):
            diff = min((b - a) % N_BINS, (a - b) % N_BINS)
            self.assertLessEqual(diff, 1)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class SamplePoseAndKeypointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tryon.body.fitter import BodyFitter

        # One fitter reused across every test in this class - constructing
        # one costs ~0.67s (loads the SMPL-X model from disk), and every
        # parameter `sample_pose_and_keypoints` touches is overwritten
        # before use, so sharing it across tests is safe.
        cls.fitter = BodyFitter()

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        features, yaw, label = sample_pose_and_keypoints(rng, fitter=self.fitter)
        self.assertEqual(features.shape, (3 * len(_JOINT_ORDER),))
        self.assertIsInstance(yaw, float)
        self.assertGreaterEqual(yaw, -np.pi)
        self.assertLess(yaw, np.pi)
        self.assertIn(label, range(N_BINS))

    def test_label_matches_yaw(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            _, yaw, label = sample_pose_and_keypoints(rng, fitter=self.fitter)
            self.assertEqual(label, yaw_to_bin(yaw))

    def test_visibility_channel_is_binary(self):
        rng = np.random.default_rng(2)
        features, _, _ = sample_pose_and_keypoints(rng, fitter=self.fitter)
        visibility = features[2 * len(_JOINT_ORDER) :]
        self.assertTrue(np.all((visibility == 0.0) | (visibility == 1.0)))

    def test_dropout_rate_matches_requested_probability_on_average(self):
        # Over many draws, the fraction of dropped (LEFT_WRIST) samples
        # should be close to its configured 0.12 probability - not exact
        # (this is a random process) but not wildly off either.
        from tryon.body.orientation_data import _DROPOUT_PROBABILITY

        left_wrist_landmark = 15
        row = _JOINT_ORDER.index(left_wrist_landmark)
        rng = np.random.default_rng(3)
        n = 2000
        dropped = 0
        for _ in range(n):
            features, _, _ = sample_pose_and_keypoints(rng, fitter=self.fitter)
            visibility = features[2 * len(_JOINT_ORDER) :]
            if visibility[row] == 0.0:
                dropped += 1
        observed_rate = dropped / n
        expected_rate = _DROPOUT_PROBABILITY[left_wrist_landmark]
        self.assertLess(abs(observed_rate - expected_rate), 0.03)

    def test_normalization_is_invariant_to_translation_and_scale(self):
        """The whole point of normalising by torso scale/translation: two
        bodies with the SAME orientation/pose but different screen
        position/size must produce nearly the same feature vector, so the
        classifier cannot cheat by learning screen placement instead of
        orientation (see the module docstring)."""
        import torch

        from tryon.body.fitter import BodyFitter
        from tryon.body.orientation_data import _normalize_features

        fitter = BodyFitter()
        with torch.no_grad():
            fitter.global_orient[0, 1] = 0.5
            joints = fitter._forward().joints[0, fitter._smplx_indices]

            fitter.scale.fill_(200.0)
            fitter.translation[:] = torch.tensor([[300.0, 250.0]])
            pixels_a = fitter._project(joints).numpy()

            fitter.scale.fill_(280.0)
            fitter.translation[:] = torch.tensor([[400.0, 180.0]])
            pixels_b = fitter._project(joints).numpy()

        visibility = np.ones(len(_JOINT_ORDER))
        features_a = _normalize_features(pixels_a, visibility, 200.0, [300.0, 250.0])
        features_b = _normalize_features(pixels_b, visibility, 280.0, [400.0, 180.0])
        np.testing.assert_allclose(features_a, features_b, atol=1e-4)

    def test_left_right_ordering_flips_across_a_half_turn(self):
        """A qualitative sanity check on the yaw-to-keypoint relationship
        itself (not just the binning): at yaw=0 the left shoulder should
        project to one side, and at yaw=pi (fully turned) the apparent
        left-right ordering in the image should flip - the actual visual
        cue MediaPipe-derived keypoints carry that lets orientation be
        inferred at all."""
        import torch

        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        with torch.no_grad():
            fitter.scale.fill_(200.0)
            fitter.translation[:] = torch.tensor([[300.0, 250.0]])

            fitter.global_orient.zero_()
            joints_front = fitter._forward().joints[0, fitter._smplx_indices]
            pixels_front = fitter._project(joints_front).numpy()

            fitter.global_orient[0, 1] = float(np.pi)
            joints_back = fitter._forward().joints[0, fitter._smplx_indices]
            pixels_back = fitter._project(joints_back).numpy()

        left_shoulder_row = _JOINT_ORDER.index(11)  # LEFT_SHOULDER
        right_shoulder_row = _JOINT_ORDER.index(12)  # RIGHT_SHOULDER

        front_left_minus_right_x = pixels_front[left_shoulder_row, 0] - pixels_front[right_shoulder_row, 0]
        back_left_minus_right_x = pixels_back[left_shoulder_row, 0] - pixels_back[right_shoulder_row, 0]
        # Turning 180 degrees must flip which side the (anatomically)
        # left shoulder projects to in the image.
        self.assertTrue(front_left_minus_right_x * back_left_minus_right_x < 0)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class MakeDatasetTest(unittest.TestCase):
    def test_shapes_and_determinism(self):
        features_a, yaws_a, labels_a = make_dataset(20, seed=42)
        features_b, yaws_b, labels_b = make_dataset(20, seed=42)
        self.assertEqual(features_a.shape, (20, 3 * len(_JOINT_ORDER)))
        self.assertEqual(yaws_a.shape, (20,))
        self.assertEqual(labels_a.shape, (20,))
        np.testing.assert_array_equal(features_a, features_b)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_different_seeds_give_different_data(self):
        features_a, _, _ = make_dataset(20, seed=1)
        features_b, _, _ = make_dataset(20, seed=2)
        self.assertFalse(np.allclose(features_a, features_b))


if __name__ == "__main__":
    unittest.main()
