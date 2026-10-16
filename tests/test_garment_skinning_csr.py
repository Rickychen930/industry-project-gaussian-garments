"""`SkinnedGarment.apply` skins over the non-zero weights only; its result is the dense einsum's."""

import unittest

import numpy as np

from tryon.garment.skinning import SkinnedGarment


class _Fitter:
    def __init__(self, transforms):
        import torch

        self._t = torch.tensor(transforms)

    def joint_transforms(self):
        return self._t


class SkinningCsrTest(unittest.TestCase):
    def test_the_sparse_kernel_matches_the_dense_einsum(self):
        rng = np.random.default_rng(0)
        n, joints = 400, 55
        rest = rng.normal(size=(n, 3))
        weights = np.zeros((n, joints))
        for i in range(n):  # a few joints a vertex, as a transferred skinning has
            chosen = rng.choice(joints, int(rng.integers(1, 15)), replace=False)
            weights[i, chosen] = rng.random(len(chosen))
        weights /= weights.sum(axis=1, keepdims=True)
        transforms = np.tile(np.eye(4), (joints, 1, 1))
        transforms[:, :3, :3] = np.linalg.qr(rng.normal(size=(joints, 3, 3)))[0]
        transforms[:, :3, 3] = rng.normal(size=(joints, 3))
        garment = SkinnedGarment(rest, weights)
        got = garment.apply(_Fitter(transforms))
        per_vertex = np.einsum("nj,jab->nab", weights, transforms)
        want = np.einsum("nab,nb->na", per_vertex, np.concatenate([rest, np.ones((n, 1))], axis=1))[:, :3]
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)

    def test_a_vertex_without_weights_stays_at_the_origin_and_scale_is_picked_up(self):
        rest = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
        weights = np.array([[0.0, 0.0], [1.0, 0.0]])
        transforms = np.tile(np.eye(4), (2, 1, 1))
        garment = SkinnedGarment(rest, weights)
        np.testing.assert_allclose(garment.apply(_Fitter(transforms))[0], 0.0)
        garment.scale(2.0)  # the homogeneous rest changes, the weights do not
        np.testing.assert_allclose(garment.apply(_Fitter(transforms))[1], [2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
