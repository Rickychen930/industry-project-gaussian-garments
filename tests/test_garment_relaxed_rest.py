"""The relaxed rest shape: `SkinnedGarment.relaxed` and the optional offset file."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tryon.garment.checkpoint import load_relaxed_rest, relaxed_rest_path
from tryon.garment.skinning import SkinnedGarment


class _Fitter:
    """Two joints; joint 1 is rotated 90 degrees about z and moved."""

    def joint_transforms(self):
        t = np.stack([np.eye(4), np.eye(4)])
        t[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t[1, :3, 3] = [0.1, 0.2, 0.3]
        return torch.tensor(t)


def _garment():
    rng = np.random.default_rng(0)
    rest = rng.normal(size=(20, 3))
    w = rng.uniform(size=(20, 2))
    garment = SkinnedGarment(rest, w / w.sum(axis=1, keepdims=True))
    return garment


class RelaxedTests(unittest.TestCase):
    def test_a_zero_offset_is_the_identity(self):
        g = _garment()
        r = g.relaxed(np.zeros_like(g.rest))
        np.testing.assert_allclose(r.apply(_Fitter()), g.apply(_Fitter()))
        np.testing.assert_allclose(r.physical_rest, g.physical_rest)

    def test_the_offset_is_carried_in_each_vertexs_own_frame(self):
        g = _garment()
        offset = np.tile([0.0, -0.05, 0.0], (20, 1))
        moved = g.relaxed(offset).apply(_Fitter()) - g.apply(_Fitter())
        fitter = _Fitter()
        blended = np.einsum("nj,jab->nab", g.weights, fitter.joint_transforms().numpy())[:, :3, :3]
        np.testing.assert_allclose(moved, np.einsum("nij,nj->ni", blended, offset), atol=1e-12)

    def test_the_result_keeps_the_shapes_and_leaves_the_original_alone(self):
        g = _garment()
        rest = g.rest.copy()
        r = g.relaxed(np.full_like(g.rest, 0.01))
        self.assertEqual(r.rest.shape, rest.shape)
        np.testing.assert_array_equal(g.rest, rest)
        r.scale(2.0)
        np.testing.assert_allclose(r.rest, (rest + 0.01) * 2.0)

    def test_a_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            _garment().relaxed(np.zeros((3, 3)))


class OffsetFileTests(unittest.TestCase):
    def test_a_missing_file_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_relaxed_rest(d, 5))

    def test_a_stored_offset_loads_and_a_mismatched_one_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stage1").mkdir()
            np.savez(relaxed_rest_path(d), offset=np.ones((5, 3), np.float32))
            self.assertEqual(load_relaxed_rest(d, 5).shape, (5, 3))
            self.assertIsNone(load_relaxed_rest(d, 6))

    def test_a_corrupt_file_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stage1").mkdir()
            relaxed_rest_path(d).write_bytes(b"not an npz")
            self.assertIsNone(load_relaxed_rest(d, 5))


if __name__ == "__main__":
    unittest.main()
