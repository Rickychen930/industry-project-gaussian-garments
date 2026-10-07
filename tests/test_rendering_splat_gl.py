"""The OpenGL compositor (tryon/rendering/splat_gl.py) draws what the numba one draws."""

import importlib.util
import unittest

import numpy as np

from tryon.rendering.splat_cpu import look_at, render_splats

HAVE_GL = importlib.util.find_spec("moderngl") is not None


def _splats(n=400, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return {
        "xyz": rng.uniform(-0.4, 0.4, (n, 3)),
        "rotation": q / np.linalg.norm(q, axis=1, keepdims=True),
        "scaling": np.log(rng.uniform(0.01, 0.05, (n, 3))),
        "opacity": rng.normal(0.0, 2.0, (n, 1)),
        "feature_dc": rng.normal(0.0, 1.0, (n, 1, 3)),
        "feature_rest": rng.normal(0.0, 0.1, (n, 15, 3)),
    }


@unittest.skipUnless(HAVE_GL, "moderngl not installed")
class SplatGLTest(unittest.TestCase):
    def test_same_image_as_numba(self):
        from tryon.rendering.splat_gl import render_splats_gl

        splats = _splats()
        view = look_at(np.array([0.0, 0.0, 2.5]), np.zeros(3))
        intrinsics = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])
        image, alpha = render_splats(splats, view, intrinsics, (96, 128))
        image_gl, alpha_gl = render_splats_gl(splats, view, intrinsics, (96, 128))
        self.assertGreater(alpha.sum(), 10)  # something is drawn
        self.assertLess(np.abs(image - image_gl).max(), 1 / 255)
        self.assertLess(np.abs(alpha - alpha_gl).max(), 1 / 255)

    def test_empty(self):
        from tryon.rendering.splat_gl import render_splats_gl

        empty = {k: v[:0] for k, v in _splats().items()}
        image, alpha = render_splats_gl(empty, np.eye(4), np.eye(3), (8, 8))
        self.assertEqual(image.shape, (8, 8, 3))
        self.assertFalse(alpha.any())


if __name__ == "__main__":
    unittest.main()


class Bgra8Test(unittest.TestCase):
    """`bgra8=True` is the float result rounded to 8 bits on the GPU."""

    def test_matches_the_float_path_within_one_level(self):
        from tryon.rendering.splat_gl import render_splats_gl

        rng = np.random.default_rng(3)
        n = 400
        q = rng.normal(size=(n, 4))
        splats = {
            "xyz": np.c_[rng.uniform(-0.3, 0.3, (n, 2)), rng.uniform(-0.1, 0.1, n)],
            "rotation": q / np.linalg.norm(q, axis=1, keepdims=True),
            "scaling": np.log(rng.uniform(0.005, 0.03, (n, 3))),
            "opacity": rng.normal(1.0, 1.0, (n, 1)),
            "feature_dc": rng.normal(0, 1, (n, 1, 3)),
            "feature_rest": rng.normal(0, 0.1, (n, 15, 3)),
        }
        view = np.eye(4)
        view[2, 3] = 2.0
        intrinsics = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])
        image, alpha = render_splats_gl(splats, view, intrinsics, (96, 128))
        packed = render_splats_gl(splats, view, intrinsics, (96, 128), bgra8=True)
        truncated = np.dstack([(np.clip(image, 0, 1) * 255).astype(np.uint8), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
        self.assertEqual(packed.shape, (96, 128, 4))
        self.assertGreater((packed[..., 3] > 0).mean(), 0.05)
        self.assertLessEqual(np.abs(packed.astype(int) - truncated.astype(int)).max(), 1)


class OccluderTest(unittest.TestCase):
    """`occluder=` hides exactly the splats a nearer surface covers."""

    def test_a_wall_over_the_left_half_hides_it(self):
        from tryon.rendering.splat_gl import render_splats_gl

        rng = np.random.default_rng(4)
        n = 600
        splats = {
            "xyz": np.c_[rng.uniform(-0.3, 0.3, (n, 2)), np.zeros(n)],
            "rotation": np.tile([1.0, 0, 0, 0], (n, 1)),
            "scaling": np.full((n, 3), np.log(0.01)),
            "opacity": np.full((n, 1), 3.0),
            "feature_dc": np.full((n, 1, 3), 1.0),
            "feature_rest": np.zeros((n, 15, 3)),
        }
        view = np.eye(4)
        view[2, 3] = 2.0  # camera 2 m in front of the z = 0 plane
        intrinsics = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])
        # a wall 0.5 m nearer the camera over x < 0 (world), and one 1 cm
        # behind the splats over x >= 0 (behind: must hide nothing)
        # (the camera sits at world z = -2 looking along +z)
        wall = np.array([[-1, -1, -0.5], [0, -1, -0.5], [0, 1, -0.5], [-1, 1, -0.5],
                         [0, -1, 0.01], [1, -1, 0.01], [1, 1, 0.01], [0, 1, 0.01]], np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        plain = render_splats_gl(splats, view, intrinsics, (96, 128), bgra8=True)[..., 3]
        hidden = render_splats_gl(splats, view, intrinsics, (96, 128), bgra8=True, occluder=(wall, faces))[..., 3]
        # the camera looks down +z of camera space: world x maps to columns
        left, right = slice(0, 55), slice(73, 128)
        self.assertGreater(plain[:, left].mean(), 20)
        self.assertLess(hidden[:, left].max(), 1)
        np.testing.assert_array_equal(hidden[:, right], plain[:, right])


class FillTrianglesTest(unittest.TestCase):
    def test_the_union_of_overlapping_triangles_has_no_holes(self):
        import cv2

        from tryon.rendering.splat_gl import fill_triangles

        pixels = np.array([[10, 10], [60, 10], [10, 60], [60, 60], [35, 5]], np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1], [0, 1, 3]])  # overlapping
        mask = fill_triangles(pixels, faces, (80, 90))
        union = np.zeros((80, 90), np.uint8)
        for f in faces:
            cv2.fillConvexPoly(union, np.round(pixels[f]).astype(np.int32), 1)
        self.assertEqual(mask.shape, (80, 90))
        inner = cv2.erode(union, np.ones((3, 3), np.uint8)).astype(bool)
        self.assertTrue(mask[inner].all())  # every interior pixel, overlaps included
        self.assertFalse(mask[~cv2.dilate(union, np.ones((3, 3), np.uint8)).astype(bool)].any())
