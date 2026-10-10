"""`splat_gl.far_to_near`: the radix order of the depths is the float sort's."""

import unittest

import numpy as np

from tryon.rendering.splat_gl import far_to_near


class FarToNearTest(unittest.TestCase):
    def test_it_orders_farthest_first_like_a_float_sort(self):
        rng = np.random.default_rng(0)
        depth = 50.0 + rng.normal(0.0, 0.1, 121_000)
        order = far_to_near(depth)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(depth)))  # a permutation
        # farthest first, to the key's resolution (the span / 2^32): a pair closer than that may swap
        resolution = (depth.max() - depth.min()) / 2**32
        self.assertTrue(np.all(np.diff(depth[order]) <= resolution * 1.01))
        reference = np.argsort(-depth)
        differ = np.flatnonzero(order != reference)
        self.assertLess(len(differ), 20)  # a handful of near-ties among 121,000
        np.testing.assert_allclose(depth[order], depth[reference], atol=resolution * 1.01)

    def test_equal_depths_keep_every_index(self):
        depth = np.full(5000, 42.0)
        np.testing.assert_array_equal(np.sort(far_to_near(depth)), np.arange(5000))

    def test_a_small_input_and_a_single_point(self):
        depth = np.array([3.0, 1.0, 2.0])
        np.testing.assert_array_equal(far_to_near(depth), [0, 2, 1])
        np.testing.assert_array_equal(far_to_near(np.array([7.0])), [0])


if __name__ == "__main__":
    unittest.main()
