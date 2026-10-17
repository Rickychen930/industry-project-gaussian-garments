"""Tests of the classical-pipeline tools on synthetic data (no 4D-DRESS data needed)."""
from __future__ import annotations

import unittest

import numpy as np

from tools.evaluate import bootstrap_iou, classical_factors, classical_reconstruction, turn_profile


def _camera(angle_deg: float, distance: float = 3.0, focal: float = 1000.0) -> dict:
    """A camera on a circle around the origin, looking at it (world to camera, 3x4 extrinsics)."""
    a = np.radians(angle_deg)
    rotation = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    translation = np.array([0.0, 0.0, distance])
    return {"intrinsics": np.array([[focal, 0, 640], [0, focal, 640], [0, 0, 1.0]]),
            "extrinsics": np.c_[rotation, translation]}


def _points(count: int = 200) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(-0.4, 0.4, (count, 3))


class ReconstructionTests(unittest.TestCase):
    def test_f_score_of_identical_surfaces_is_one(self):
        points = _points()
        score = classical_reconstruction.f_score(points, points)
        self.assertAlmostEqual(score["f_score"], 1.0)
        self.assertAlmostEqual(score["chamfer_cm"], 0.0)

    def test_f_score_of_an_empty_surface_is_zero(self):
        score = classical_reconstruction.f_score(np.zeros((0, 3)), _points())
        self.assertEqual(score["f_score"], 0.0)

    def test_triangulation_recovers_points_from_two_calibrated_views(self):
        points = _points()
        cam_a, cam_b = _camera(0), _camera(40)
        pts_a = classical_reconstruction._project(points, cam_a).astype(np.float32)
        pts_b = classical_reconstruction._project(points, cam_b).astype(np.float32)
        found = classical_reconstruction.triangulate(pts_a, pts_b, cam_a, cam_b)
        self.assertEqual(len(found), len(points))
        self.assertLess(np.abs(found - points).max(), 1e-2)

    def test_camera_motion_from_perfect_matches_is_close_to_the_truth(self):
        points = _points(400)
        cam_a, cam_b = _camera(0), _camera(30)
        pts_a = classical_reconstruction._project(points, cam_a).astype(np.float32)
        pts_b = classical_reconstruction._project(points, cam_b).astype(np.float32)
        motion = classical_reconstruction.camera_motion(pts_a, pts_b, cam_a, cam_b)
        self.assertLess(motion["rotation_error_deg"], 2.0)
        self.assertLess(motion["direction_error_deg"], 5.0)

    def test_camera_motion_with_too_few_matches_reports_no_pose(self):
        few = np.zeros((3, 2), np.float32)
        motion = classical_reconstruction.camera_motion(few, few, _camera(0), _camera(30))
        self.assertIsNone(motion["rotation_error_deg"])

    def test_visual_hull_of_a_full_mask_fills_the_box(self):
        cameras = {"a": _camera(0), "b": _camera(90)}
        masks = {name: np.full((1280, 1280), 255, np.uint8) for name in cameras}
        bounds = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
        centres, axes = classical_reconstruction.visual_hull(masks, cameras, bounds)
        self.assertEqual(len(centres), int(np.prod([len(a) for a in axes])))


class FactorTests(unittest.TestCase):
    def test_gamma_darkens_and_brightens(self):
        gray = np.full((8, 8), 128, np.uint8)
        self.assertLess(classical_factors._gamma(gray, 2.2).mean(), 128)
        self.assertGreater(classical_factors._gamma(gray, 0.45).mean(), 128)

    def test_every_perturbation_keeps_the_image_shape(self):
        gray = np.random.default_rng(0).integers(0, 255, (64, 64)).astype(np.uint8)
        for name, perturb in classical_factors.PERTURBATIONS.items():
            self.assertEqual(perturb(gray).shape, gray.shape, name)


class BootstrapTests(unittest.TestCase):
    def test_constant_iou_gives_a_zero_width_interval_and_the_exact_gap(self):
        videos = [{"frames": [{"ours": 0.7, "theirs": 0.8}] * 10} for _ in range(3)]
        result = bootstrap_iou.bootstrap(videos, resamples=200)
        self.assertAlmostEqual(result["mean"]["ours"], 0.7)
        low, high = result["frame_bootstrap"]["gap"]
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.1)

    def test_video_bootstrap_is_wider_when_videos_differ(self):
        videos = [{"frames": [{"ours": v, "theirs": 0.8}] * 10} for v in (0.6, 0.7, 0.8)]
        result = bootstrap_iou.bootstrap(videos, resamples=500)
        low, high = result["video_bootstrap"]["ours"]
        self.assertGreater(high - low, 0.05)


class TurnProfileTests(unittest.TestCase):
    def _records(self, angle: float, garment: int, person: int, count: int = 30) -> list[dict]:
        return [{"dressed": True, "profile_angle": angle, "garment_width": garment, "person_width": person}
                for _ in range(count)]

    def test_side_on_is_narrower_and_the_ratio_is_reported(self):
        records = self._records(5, 300, 280) + self._records(85, 150, 140)
        result = turn_profile.summarise(records)
        head_on, _, side_on = result["bins"]
        self.assertLess(side_on["garment_width_px"], head_on["garment_width_px"])
        self.assertAlmostEqual(head_on["ratio"], 300 / 280)

    def test_a_bin_with_few_frames_is_reported_without_a_ratio(self):
        result = turn_profile.summarise(self._records(5, 300, 280, count=5))
        self.assertNotIn("ratio", result["bins"][0])

    def test_undressed_frames_are_ignored(self):
        records = [{"dressed": False, "profile_angle": 5, "garment_width": 300, "person_width": 280}] * 40
        self.assertEqual(turn_profile.summarise(records)["usable_frames"], 0)


if __name__ == "__main__":
    unittest.main()
