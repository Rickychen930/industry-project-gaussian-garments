"""tryon/garment/cloth.py: springs toward the skinned garment, a max-distance limit, and body collision."""

import unittest

import numpy as np
import trimesh

from tryon.garment.cloth import MAX_DT, LightCloth, SmoothCloth, _springs_serial


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

    def test_the_compiled_springs_equal_the_numpy_original_bit_for_bit(self):
        rng = np.random.default_rng(2)
        for cls in (LightCloth, SmoothCloth):
            cloth = cls()
            cloth.step(self.target, 0.0, self.body, self.faces)
            for dt in (1 / 25, 0.05, 0.1, 0.37, 0.004):  # 1 to 45 substeps; big jumps hit the radius clamp
                moved = self.target + rng.normal(0, 0.03, self.target.shape)
                v0 = cloth.v.copy()
                expected_x, expected_v = cloth._springs_numpy(moved, dt, v0)
                saved = (cloth.x.copy(), cloth.target.copy())
                n = max(1, int(np.ceil(dt / (1.0 / 120.0))))
                x, v = _springs_serial(
                    cloth.x, v0, cloth.target, moved, cloth.looseness, cls.STIFF_OMEGA, cls.LOOSE_OMEGA,
                    cls.STIFF_RADIUS, cls.LOOSE_RADIUS, cls.DAMPING_RATIO, n, dt / n,
                )
                np.testing.assert_array_equal(x, expected_x)
                np.testing.assert_array_equal(v, expected_v)
                np.testing.assert_array_equal(cloth.x, saved[0])  # inputs untouched
                cloth.step(moved, dt, self.body, self.faces)


class SmoothClothTest(unittest.TestCase):
    def test_trails_a_moving_target_within_its_radius_and_stays_outside_the_body(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
        body, faces = np.asarray(sphere.vertices), np.asarray(sphere.faces)
        target = np.asarray(trimesh.creation.icosphere(subdivisions=3, radius=0.26).vertices)
        cloth = SmoothCloth()
        cloth.step(target, 0.0, body, faces)
        moved = target + [0.1, 0.0, 0.0]
        x = cloth.step(moved, 0.2, body + [0.1, 0, 0], faces)
        self.assertTrue(np.isfinite(x).all())
        self.assertLessEqual(np.linalg.norm(x - moved, axis=1).max(), SmoothCloth.LOOSE_RADIUS + 1e-9)
        self.assertGreater(np.linalg.norm(x - (body + [0.1, 0, 0]).mean(0), axis=1).min(), 0.2)


if __name__ == "__main__":
    unittest.main()
