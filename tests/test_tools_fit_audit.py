"""The pure metric helpers of tools/evaluate/fit_audit.py, on synthetic inputs with known answers."""

import unittest

import numpy as np

from tools.evaluate import fit_audit as fa


def square(size=60, lo=15, hi=45):
    mask = np.zeros((size, size), bool)
    mask[lo:hi, lo:hi] = True
    return mask


class ColourTest(unittest.TestCase):
    def test_identical_colours_are_zero_apart(self):
        image = np.random.default_rng(0).integers(0, 255, (5, 5, 3), dtype=np.uint8)
        self.assertEqual(float(fa.delta_e76(image, image).max()), 0.0)

    def test_black_to_white_is_one_hundred_l_units(self):
        black, white = np.zeros((1, 3), np.uint8), np.full((1, 3), 255, np.uint8)
        self.assertAlmostEqual(float(fa.delta_e76(black, white)[0]), 100.0, delta=0.5)

    def test_gain_offset_recovers_a_global_light_change(self):
        rng = np.random.default_rng(1)
        ours = rng.uniform(40, 200, (500, 3))
        real = 0.8 * ours + 12.0
        gain, offset = fa.fit_gain_offset(ours, real)
        np.testing.assert_allclose(gain, 0.8, atol=1e-6)
        np.testing.assert_allclose(offset, 12.0, atol=1e-4)

    def test_psnr_of_a_known_error(self):
        a = np.zeros((4, 4, 3))
        self.assertAlmostEqual(fa.psnr(a, a + 10.0), 20 * np.log10(255 / 10), places=6)
        self.assertEqual(fa.psnr(a, a), float("inf"))

    def test_laplacian_variance_ranks_sharpness(self):
        rng = np.random.default_rng(2)
        sharp = rng.uniform(0, 255, (40, 40)).astype(np.float32)
        blurred = np.asarray(fa.cv2.GaussianBlur(sharp, (0, 0), 2.0))
        region = np.ones((40, 40), bool)
        self.assertGreater(fa.laplacian_variance(sharp, region), 10 * fa.laplacian_variance(blurred, region))


class ContourTest(unittest.TestCase):
    def test_identical_masks_score_one_and_zero_distance(self):
        mask = square()
        self.assertEqual(fa.boundary_fscore(mask, mask, 2), 1.0)
        self.assertEqual(fa.hausdorff95(mask, mask), 0.0)

    def test_a_shift_inside_the_tolerance_passes_and_outside_fails(self):
        a, b = square(), np.roll(square(), 3, axis=1)
        self.assertGreater(fa.boundary_fscore(a, b, 5), 0.95)
        self.assertLess(fa.boundary_fscore(a, b, 2), 0.6)

    def test_hausdorff_grows_with_the_shift(self):
        a = square()
        near, far = fa.hausdorff95(a, np.roll(a, 2, axis=1)), fa.hausdorff95(a, np.roll(a, 6, axis=1))
        self.assertLess(near, far)
        self.assertAlmostEqual(far, 6.0, delta=1.0)

    def test_empty_masks(self):
        empty = np.zeros((20, 20), bool)
        self.assertEqual(fa.boundary_fscore(empty, empty, 2), 1.0)
        self.assertEqual(fa.boundary_fscore(empty, square(20, 5, 15), 2), 0.0)
        self.assertTrue(np.isnan(fa.hausdorff95(empty, square(20, 5, 15))))


class MotionTest(unittest.TestCase):
    def test_constant_velocity_has_no_second_difference(self):
        track = np.arange(10)[:, None] * np.array([[1.0, 2.0, 3.0]])
        self.assertAlmostEqual(fa.second_difference_mean(track), 0.0, places=12)

    def test_alternating_jitter_is_four_amplitudes(self):
        # x_t = (-1)^t * a  ->  x[t+1] - 2 x[t] + x[t-1] = -4 x[t]
        jitter = np.array([(-1.0) ** t * 0.5 for t in range(8)])
        self.assertAlmostEqual(fa.second_difference_mean(jitter), 2.0, places=12)
        vectors = np.zeros((8, 3, 3))
        vectors[:, :, 0] = jitter[:, None]
        self.assertAlmostEqual(fa.second_difference_mean(vectors), 2.0, places=12)

    def test_lag_of_a_delayed_copy(self):
        rng = np.random.default_rng(4)
        real = np.cumsum(rng.normal(size=(40, 2)), axis=0)
        ours = np.roll(real, 2, axis=0)
        lag, _ = fa.best_lag(ours[2:], real[2:], max_lag=4)
        self.assertAlmostEqual(lag, 2.0, delta=0.3)
        lag_early, _ = fa.best_lag(real[2:], ours[2:], max_lag=4)
        self.assertAlmostEqual(lag_early, -2.0, delta=0.3)

    def test_centroid_and_area(self):
        centroids, areas = fa.centroid_and_area([square(60, 10, 20), np.zeros((60, 60), bool)])
        np.testing.assert_allclose(centroids[0], [14.5, 14.5])
        self.assertTrue(np.isnan(centroids[1]).all())
        self.assertEqual(list(areas), [100, 0])

    def test_torso_frame_is_orthonormal(self):
        origin, axes = fa.torso_frame([0, 0, 0], [0, 1, 0.2], [-1, 1, 0], [1, 1.1, 0])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(origin, 0)


class FitTest(unittest.TestCase):
    def test_penetration_share_and_depth(self):
        signed = np.array([-0.02, -0.04, 0.01, 0.03, 0.0, 0.05])
        stats = fa.penetration_stats(signed)
        self.assertAlmostEqual(stats["inside_share"], 2 / 6)
        self.assertAlmostEqual(stats["depth_mean"], 0.03)
        self.assertAlmostEqual(stats["depth_p95"], 0.039, places=3)
        self.assertEqual(fa.penetration_stats(np.array([0.1, 0.2]))["inside_share"], 0.0)

    def test_gap_ignores_vertices_inside_and_outside_the_region(self):
        signed = np.array([-0.05, 0.01, 0.03, 0.2])
        region = np.array([True, True, True, False])
        self.assertAlmostEqual(fa.gap_stats(signed, region)["mean"], 0.02)
        self.assertTrue(np.isnan(fa.gap_stats(signed, np.array([True, False, False, False]))["mean"]))

    def test_regions_partition_the_garment(self):
        rest = np.zeros((10, 3))
        rest[:, 1] = np.linspace(0, 1, 10)
        sleeve = np.zeros(10, bool)
        sleeve[7:] = True
        s, t, h = fa.region_masks(rest, sleeve)
        self.assertTrue((s ^ t ^ h).all() and not (s & t).any() and not (t & h).any() and not (s & h).any())
        self.assertTrue(h[0] and h[1] and not h[2])

    def test_procrustes_ignores_a_rigid_motion_and_scale(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(12, 3))
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        self.assertAlmostEqual(fa.procrustes_mpjpe(1.3 * truth @ rotation.T + 5.0, truth), 0.0, places=9)
        self.assertGreater(fa.procrustes_mpjpe(truth + rng.normal(scale=0.1, size=truth.shape), truth), 0.01)


if __name__ == "__main__":
    unittest.main()
