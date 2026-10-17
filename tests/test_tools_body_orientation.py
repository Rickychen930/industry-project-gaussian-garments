"""The facing-angle helpers of tools/evaluate/body_orientation_4ddress.py, on synthetic joints."""

import unittest

import numpy as np

from tools.evaluate.body_orientation_4ddress import (
    body_facing,
    camera_to_model,
    facing_angle,
    root_relative_mpjpe,
    summarise,
    wrapped_difference,
)


class FacingAngleTest(unittest.TestCase):
    def test_left_on_the_cameras_right_is_zero_degrees(self):
        self.assertAlmostEqual(facing_angle([0.2, 0, 3], [-0.2, 0, 3]), 0.0)

    def test_left_on_the_cameras_left_is_a_half_turn(self):
        self.assertAlmostEqual(abs(facing_angle([-0.2, 0, 3], [0.2, 0, 3])), 180.0)

    def test_side_on_is_ninety_degrees(self):
        self.assertAlmostEqual(abs(facing_angle([0, 0, 3.2], [0, 0, 2.8])), 90.0)


class CameraToModelTest(unittest.TestCase):
    def test_flips_y_and_z_only(self):
        np.testing.assert_allclose(camera_to_model([[1.0, 2.0, 3.0]]), [[1.0, -2.0, -3.0]])

    def test_is_a_proper_rotation_so_distances_are_kept(self):
        a, b = np.array([0.0, 0.1, 3.0]), np.array([0.2, -0.4, 2.5])
        before = np.linalg.norm(a - b)
        after = np.linalg.norm(camera_to_model(a) - camera_to_model(b))
        self.assertAlmostEqual(before, after)


class WrappedDifferenceTest(unittest.TestCase):
    def test_wraps_across_the_seam(self):
        self.assertAlmostEqual(wrapped_difference(170.0, -170.0), -20.0)

    def test_equal_angles_give_zero(self):
        self.assertAlmostEqual(wrapped_difference(35.0, 35.0), 0.0)


class BodyFacingTest(unittest.TestCase):
    def test_uses_the_mean_of_shoulders_and_hips(self):
        joints = np.array([[0.2, 0, 3], [0.1, 0, 3], [-0.2, 0, 3], [-0.1, 0, 3]])
        self.assertAlmostEqual(body_facing(joints, (0, 1), (2, 3)), 0.0)


class RootRelativeTest(unittest.TestCase):
    def test_a_pure_shift_costs_nothing(self):
        truth = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.5, 0]])
        self.assertAlmostEqual(root_relative_mpjpe(truth + 5.0, truth, (0, 1)), 0.0)

    def test_a_turned_body_costs_something(self):
        truth = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0.3, 0.5, 0]])
        turned = truth * np.array([-1.0, 1.0, 1.0])
        self.assertGreater(root_relative_mpjpe(turned, truth, (0, 1)), 0.1)


class SummariseTest(unittest.TestCase):
    def test_counts_flips_over_ninety_degrees(self):
        rows = [
            {"detected": True, "facing_error_degrees": 10.0, "root_relative_mm": 50.0},
            {"detected": True, "facing_error_degrees": -170.0, "root_relative_mm": 150.0},
            {"detected": False},
        ]
        out = summarise(rows)
        self.assertEqual((out["views"], out["detected"], out["flips"]), (3, 2, 1))
        self.assertAlmostEqual(out["flip_share"], 0.5)
        self.assertAlmostEqual(out["root_relative_mm_mean"], 100.0)

    def test_no_detections_gives_counts_only(self):
        self.assertEqual(summarise([{"detected": False}]), {"views": 1, "detected": 0})


if __name__ == "__main__":
    unittest.main()
