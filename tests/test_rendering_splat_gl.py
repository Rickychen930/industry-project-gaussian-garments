"""The OpenGL compositor (tryon/rendering/splat_gl.py) draws what the numba one draws."""

import importlib.util
import unittest

import numpy as np

from tryon.rendering.splat_cpu import look_at, render_splats

def _gl_context_available():
    """moderngl installed and an OpenGL context can be created (a headless CI runner has none)."""
    if importlib.util.find_spec("moderngl") is None:
        return False
    try:
        import moderngl

        moderngl.create_standalone_context().release()
    except Exception:
        return False
    return True


HAVE_GL = _gl_context_available()


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


@unittest.skipUnless(HAVE_GL, "no OpenGL context")
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


@unittest.skipUnless(HAVE_GL, "no OpenGL context")
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


@unittest.skipUnless(HAVE_GL, "no OpenGL context")
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
        # A wall 0.5 m nearer the camera over x < 0, and one 1 cm behind
        # the splats over x >= 0 (must hide nothing); camera at z = -2 looking along +z.
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


class InstanceRowsTest(unittest.TestCase):
    """`_instances` fills a reused buffer with what the numpy gathers gave."""

    def test_rows_and_distance_equal_the_numpy_form_bit_for_bit(self):
        from tryon.rendering.splat_cpu import _prepare
        from tryon.rendering.splat_gl import _instance_rows, _instances

        splats = _splats(500, seed=6)
        view = look_at(np.array([0.2, 0.1, 2.5]), np.zeros(3))
        intrinsics = np.array([[300.0, 0, 64], [0, 300.0, 48], [0, 0, 1]])
        pixels, depth, valid, cov2d, determinant, colours, opacity, radius = _prepare(splats, view, intrinsics, 3.0)
        order = np.flatnonzero(valid)[::-1].copy()
        x0, y0, x1, y1 = (pixels[:, 0] - radius), (pixels[:, 1] - radius), (pixels[:, 0] + radius), (pixels[:, 1] + radius)
        opacity = np.asarray(opacity, np.float64).ravel()
        for count in (len(order), len(order) // 2):  # the second frame reuses the first's (larger) buffer
            rows = _instance_rows(count)
            _instances(rows, order[:count], x0, y0, x1, y1, pixels, cov2d, determinant, colours, opacity,
                       splats["xyz"], view[:3, :3].copy(), view[:3, 3].copy(), True)
            k = order[:count]
            expected_distance = np.linalg.norm(splats["xyz"][k] @ view[:3, :3].T + view[:3, 3], axis=1)
            np.testing.assert_array_equal(rows[:, 14], expected_distance.astype(np.float32))
            np.testing.assert_array_equal(rows[:, 0], x0[k].astype(np.float32))
            np.testing.assert_array_equal(rows[:, 10], colours[k, 2].astype(np.float32))
            np.testing.assert_array_equal(rows[:, 15], 0.0)
            _instances(rows, order[:count], x0, y0, x1, y1, pixels, cov2d, determinant, colours, opacity,
                       splats["xyz"], view[:3, :3].copy(), view[:3, 3].copy(), False)
            np.testing.assert_array_equal(rows[:, 14], 0.0)


@unittest.skipUnless(HAVE_GL, "no OpenGL context")
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
