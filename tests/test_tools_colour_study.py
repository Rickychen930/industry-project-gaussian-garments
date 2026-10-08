"""The pure helpers of tools/evaluate/colour_study.py, on synthetic inputs with known answers."""

import unittest

import numpy as np

from tools.evaluate import colour_study as cs


class GainTest(unittest.TestCase):
    def test_gain_offset_recovers_a_linear_change(self):
        rng = np.random.default_rng(0)
        ours = rng.uniform(20, 200, (500, 3))
        gain, offset = cs.channel_gain_offset(ours, ours * [1.1, 0.8, 1.3] + [5, -10, 0])
        np.testing.assert_allclose(gain, [1.1, 0.8, 1.3], atol=1e-6)
        np.testing.assert_allclose(offset, [5, -10, 0], atol=1e-4)

    def test_mean_gain_is_the_ratio_of_means(self):
        ours = np.full((10, 3), [100.0, 50.0, 200.0])
        np.testing.assert_allclose(cs.mean_gain(ours, ours * [0.5, 2.0, 1.0]), [0.5, 2.0, 1.0])

    def test_constant_channel_keeps_unit_gain(self):
        gain, offset = cs.channel_gain_offset(np.full((10, 3), 90.0), np.full((10, 3), 100.0))
        np.testing.assert_allclose(gain, 1.0)
        np.testing.assert_allclose(offset, 10.0)


class RegionTest(unittest.TestCase):
    def test_region_is_overlap_eroded_and_opaque(self):
        rgba = np.zeros((60, 60, 4), np.uint8)
        rgba[10:50, 10:50] = 255
        shirt = np.zeros((60, 60), bool)
        shirt[20:60, 20:60] = True
        region = cs.overlap_region(rgba, shirt, erode=3)
        self.assertTrue(region[30, 30])
        self.assertFalse(region[21, 30])  # inside the overlap but within 3 px of its edge
        self.assertFalse(region[15, 15])  # not on the shirt

    def test_straight_colours_undo_premultiplication(self):
        rgba = np.zeros((2, 2, 4), np.uint8)
        rgba[..., :3], rgba[..., 3] = 50, 100
        colours, alpha = cs.straight_colours(rgba)
        self.assertAlmostEqual(float(colours[0, 0, 0]), 50 / (100 / 255), places=3)
        self.assertAlmostEqual(float(alpha[0, 0]), 100 / 255, places=5)


class SceneTest(unittest.TestCase):
    def test_scene_pixels_exclude_the_garment_and_other_classes(self):
        frame = np.zeros((40, 40, 3), np.uint8)
        frame[:] = 100
        labels = np.zeros((40, 40), np.uint8)
        labels[:, 20:] = 4  # the wearer's own clothing: never scene
        alpha = np.zeros((40, 40), np.uint8)
        alpha[:, :5] = 255  # the garment
        pixels = cs.scene_pixels(frame, labels, alpha, cs.SCENE_BACKGROUND, margin=3)
        self.assertEqual(len(pixels), 40 * 14)  # columns 6..19: the garment's 5 columns and 1 of margin dropped

    def test_scene_gain_is_damped_and_clipped(self):
        self.assertAlmostEqual(cs.scene_gain({"luma": 100.0}, 100.0), 1.0)
        self.assertAlmostEqual(cs.scene_gain({"luma": 144.0}, 100.0, damping=0.5, limits=(0.5, 2.0)), 1.2)
        self.assertEqual(cs.scene_gain({"luma": 1000.0}, 100.0), cs.GAIN_LIMITS[1])

    def test_smoothing_starts_at_the_first_value_and_follows(self):
        out = cs.smoothed([10.0, 20.0, 20.0], rate=0.5)
        self.assertEqual(float(out[0]), 10.0)
        self.assertEqual(float(out[1]), 15.0)
        self.assertEqual(float(out[2]), 17.5)

    def test_reinhard_with_zero_weight_changes_nothing(self):
        colours = np.random.default_rng(1).uniform(30, 220, (12, 12, 3)).astype(np.float32)
        region = np.ones((12, 12), bool)
        scene = {"lab_mean": np.array([60.0, 0.0, 0.0]), "lab_std": np.array([10.0, 5.0, 5.0])}
        np.testing.assert_allclose(cs.reinhard(colours, region, scene, weight=0.0), colours, atol=1.5)


class MetricTest(unittest.TestCase):
    def test_identical_pixels_have_zero_distance(self):
        pixels = np.random.default_rng(2).uniform(0, 255, (50, 3))
        self.assertAlmostEqual(cs.metrics(pixels, pixels)["de76"], 0.0, places=1)

    def test_luminance_of_white_is_255(self):
        self.assertAlmostEqual(float(cs.luminance(np.array([255.0, 255.0, 255.0]))), 255.0, places=3)

    def test_binning_reports_each_range(self):
        values, key = np.array([1.0, 3.0, 10.0]), np.array([0.0, 5.0, 50.0])
        rows = cs._binned(values, key, (0, 10, 100))
        self.assertAlmostEqual(rows[0]["mean_luma_diff"], 2.0)
        self.assertAlmostEqual(rows[1]["share"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
