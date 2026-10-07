"""tryon/app/layer_warp.py: the torso similarity that carries a finished garment layer to a later frame."""

import unittest

import numpy as np

from tryon.app.layer_warp import MAX_ROTATION, SCALE_RANGE, torso_layer, torso_transform


def _keypoints(seed=0):
    rng = np.random.default_rng(seed)
    kp = np.zeros((33, 3))
    kp[:, :2] = rng.uniform(100, 400, (33, 2))
    kp[:, 2] = 1.0
    kp[11, :2], kp[12, :2], kp[23, :2], kp[24, :2] = (300, 150), (200, 150), (285, 330), (215, 330)
    return kp


class TorsoWarpTest(unittest.TestCase):
    def test_a_moved_torso_moves_the_layer_exactly(self):
        old = _keypoints()
        new = old.copy()
        new[:, :2] += (25.0, -10.0)
        m = torso_transform(old, new)
        np.testing.assert_allclose(m, [[1, 0, 25], [0, 1, -10]], atol=1e-6)
        layer = np.zeros((480, 640), np.uint8)
        layer[200:260, 220:280] = 255
        moved = torso_layer(layer, old, new)
        self.assertEqual(moved[190:250, 245:305].min(), 255)

    def test_a_runaway_fit_is_clamped(self):
        old = _keypoints()
        new = old.copy()
        centre = new[[11, 12, 23, 24], :2].mean(0)
        new[:, :2] = centre + 3.0 * (new[:, :2] - centre)  # three times the size in one step
        m = torso_transform(old, new)
        scale = np.sqrt(abs(np.linalg.det(m[:, :2])))
        self.assertAlmostEqual(scale, SCALE_RANGE[1], places=5)
        self.assertLessEqual(abs(np.arctan2(m[1, 0], m[0, 0])), MAX_ROTATION + 1e-9)

    def test_no_torso_no_warp(self):
        old, new = _keypoints(), _keypoints(1)
        new[23, 2] = 0.1  # a hip out of view
        self.assertIsNone(torso_transform(old, new))
        layer = np.full((10, 10), 7, np.uint8)
        self.assertIs(torso_layer(layer, old, new), layer)


if __name__ == "__main__":
    unittest.main()
