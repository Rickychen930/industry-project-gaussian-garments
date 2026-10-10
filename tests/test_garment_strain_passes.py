"""The strain limiter (`_strain_passes`, Gauss-Seidel): over-long edges shorten, short ones are left alone."""

import unittest

import numpy as np

from tryon.garment.renderer import _strain_passes


def _chain(n, spacing=0.01):
    x = np.zeros((n, 3))
    x[:, 0] = spacing * np.arange(n)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1).astype(np.int64)
    return x, edges


class StrainPassesTest(unittest.TestCase):
    def test_edges_within_their_limit_are_not_moved(self):
        x, edges = _chain(20)
        limit = np.full(len(edges), 0.0105)
        moved = x.copy()
        _strain_passes(moved, edges, limit, 10)
        np.testing.assert_array_equal(moved, x)

    def test_a_stretched_chain_is_pulled_back_toward_its_limit(self):
        x, edges = _chain(20, spacing=0.02)  # twice the rest length
        limit = np.full(len(edges), 0.0105)
        moved = x.copy()
        _strain_passes(moved, edges, limit, 10)
        before = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
        after = np.linalg.norm(moved[edges[:, 1]] - moved[edges[:, 0]], axis=1)
        # a local relaxation on a chain stretched everywhere spreads slowly: the mean excess falls, no edge gets worse
        self.assertLess(after.mean(), before.mean())
        self.assertLessEqual(after.max(), before.max() + 1e-12)
        np.testing.assert_allclose(moved.mean(axis=0), x.mean(axis=0), atol=1e-12)  # moves are pairwise: the centre stays

    def test_a_single_over_long_edge_is_corrected_exactly_in_one_pass(self):
        x = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
        edges = np.array([[0, 1]], np.int64)
        _strain_passes(x, edges, np.array([0.03]), 1)
        np.testing.assert_allclose(x[1, 0] - x[0, 0], 0.03, atol=1e-12)

    def test_zero_iterations_change_nothing(self):
        x, edges = _chain(5, spacing=0.05)
        moved = x.copy()
        _strain_passes(moved, edges, np.full(len(edges), 0.01), 0)
        np.testing.assert_array_equal(moved, x)


if __name__ == "__main__":
    unittest.main()
