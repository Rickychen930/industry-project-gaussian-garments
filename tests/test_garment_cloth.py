"""tryon/garment/cloth.py: springs toward the skinned garment, a max-distance limit, and body collision."""

import unittest

import numpy as np
import trimesh

from tryon.garment.cloth import MAX_DT, LightCloth


def _body():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
    return np.asarray(sphere.vertices), np.asarray(sphere.faces)


class LightClothTest(unittest.TestCase):
    def setUp(self):
        self.body, self.faces = _body()
        rng = np.random.default_rng(0)
        direction = rng.normal(size=(200, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.target = direction * 0.26  # a shell 6 cm off the sphere

    def test_a_still_target_is_reached_and_held(self):
        cloth = LightCloth()
        cloth.step(self.target, 0.0, self.body, self.faces)
        for _ in range(30):
            x = cloth.step(self.target, 1 / 25, self.body, self.faces)
        self.assertLess(np.abs(x - self.target).max(), 1e-6)

    def test_a_moving_target_is_trailed_within_the_radius(self):
        cloth = LightCloth()
        cloth.step(self.target, 0.0, self.body, self.faces)
        moved = self.target + np.array([0.1, 0.0, 0.0])
        x = cloth.step(moved, 1 / 25, self.body + [0.1, 0, 0], self.faces)
        lag = np.linalg.norm(x - moved, axis=1)
        self.assertGreater(lag.max(), 1e-3)  # it trails: secondary motion
        self.assertLessEqual(lag.max(), LightCloth.LOOSE_RADIUS + 1e-9)

    def test_no_vertex_is_left_inside_the_body(self):
        cloth = LightCloth()
        inside = self.target * (0.15 / 0.26)  # every vertex 5 cm inside the sphere
        x = cloth.step(inside, 0.0, self.body, self.faces)
        radius = np.linalg.norm(x, axis=1)
        self.assertGreater(radius.min(), 0.2)

    def test_a_long_gap_restarts_from_the_skin(self):
        cloth = LightCloth()
        cloth.step(self.target, 0.0, self.body, self.faces)
        far = self.target + 1.0
        x = cloth.step(far, MAX_DT * 2, self.body + 1.0, self.faces)
        np.testing.assert_allclose(x, far, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
