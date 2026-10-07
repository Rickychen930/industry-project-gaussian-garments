"""tryon/garment/raycast.py and the shared edge list of tryon/garment/scan_pose.py."""

from __future__ import annotations

import unittest

import numpy as np


class UniqueEdgesTest(unittest.TestCase):
    def test_each_edge_once_sorted(self):
        from tryon.garment.scan_pose import unique_edges

        faces = np.array([[0, 1, 2], [2, 1, 3]])
        edges = unique_edges(faces)
        np.testing.assert_array_equal(edges, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])


class OccludedFromTest(unittest.TestCase):
    def setUp(self):
        try:
            import open3d  # noqa: F401
        except ImportError:
            self.skipTest("no open3d")

    def test_a_point_behind_a_wall_is_hidden_and_one_in_front_is_not(self):
        from tryon.garment.raycast import occluded_from

        wall = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        camera = np.array([0.0, 0.0, 10.0], np.float32)
        points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [5.0, 5.0, -1.0]])
        hidden = occluded_from(points, camera, wall, faces, margin=0.03)
        np.testing.assert_array_equal(hidden, [True, False, False])

    def test_the_margin_keeps_a_point_on_the_surface_visible(self):
        from tryon.garment.raycast import occluded_from

        wall = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        camera = np.array([0.0, 0.0, 10.0], np.float32)
        on_it = np.array([[0.0, 0.0, -0.01]])
        self.assertFalse(occluded_from(on_it, camera, wall, faces, margin=0.03)[0])
        self.assertTrue(occluded_from(on_it, camera, wall, faces, margin=0.0)[0])


if __name__ == "__main__":
    unittest.main()
