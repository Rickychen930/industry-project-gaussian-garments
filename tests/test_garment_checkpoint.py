"""tryon/garment/checkpoint.py: reading a garment checkpoint's template and
binding its Gaussians to the template's faces through the UV atlas."""

import unittest

import numpy as np


class UvBindingTests(unittest.TestCase):
    """Recovering the Gaussian-to-face binding requires reproducing the
    texel count the model produced. Demanding an exact match rejected two
    real checkpoints that were off by a single pixel - `00190_upper` gives
    125759 against a stored 125758, `10004_outer` 99956 against 99955 - a
    rounding difference on one boundary texel, while the next resolution up
    is out by hundreds of thousands. A small tolerance accepts the former
    without coming close to admitting the latter.
    """

    def _square_mesh(self):
        """One UV-mapped quad, as two triangles."""
        import trimesh

        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        )
        return mesh

    def test_an_exact_count_binds(self):
        from tryon.garment.checkpoint import uv_binding

        mesh = self._square_mesh()
        binding, resolution = uv_binding(mesh, None)
        self.assertEqual(resolution, 512)
        self.assertGreater(len(binding), 0)
        # Every entry names one of the two faces.
        self.assertTrue(set(np.unique(binding)) <= {0, 1})

    def test_a_count_off_by_one_still_binds(self):
        from tryon.garment.checkpoint import uv_binding

        mesh = self._square_mesh()
        exact, _ = uv_binding(mesh, None)
        binding, _ = uv_binding(mesh, len(exact) - 1)
        self.assertEqual(len(binding), len(exact))

    def test_a_wildly_different_count_is_rejected(self):
        from tryon.garment.checkpoint import uv_binding

        with self.assertRaises(SystemExit):
            uv_binding(self._square_mesh(), 7)


class TemplateLoadTest(unittest.TestCase):
    """`load_template` splits vertices by (position, UV), never by normal."""

    def test_per_face_normals_do_not_turn_the_mesh_into_a_triangle_soup(self):
        import tempfile
        from pathlib import Path

        from tryon.garment.checkpoint import load_template

        obj = "\n".join([
            "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 1 1 0",
            "vt 0 0", "vt 1 0", "vt 0 1", "vt 1 1",
            "vn 0 0 1", "vn 0 0 -1",
            "f 1/1/1 2/2/1 3/3/1",   # one normal per face, as 00122_outer stores them
            "f 2/2/2 4/4/2 3/3/2",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.obj"
            path.write_text(obj)
            mesh = load_template(path)
        self.assertEqual(len(mesh.vertices), 4, "the two faces must share their edge's vertices")
        self.assertEqual(len(mesh.faces), 2)
        self.assertEqual(mesh.visual.uv.shape, (4, 2))


class UvBindingCompiledTest(unittest.TestCase):
    """`uv_binding` (numba) returns the Python loop's binding exactly."""

    def test_matches_the_python_loop(self):
        from types import SimpleNamespace

        from tryon.garment.checkpoint import _uv_binding_python, uv_binding

        rng = np.random.default_rng(6)
        mesh = SimpleNamespace(visual=SimpleNamespace(uv=rng.uniform(0.0, 1.0, (200, 2))), faces=rng.integers(0, 200, (150, 3)))
        fast, fast_res = uv_binding(mesh, None)
        slow, slow_res = _uv_binding_python(mesh, None)
        self.assertEqual(fast_res, slow_res)
        np.testing.assert_array_equal(fast, slow)


if __name__ == "__main__":
    unittest.main()
