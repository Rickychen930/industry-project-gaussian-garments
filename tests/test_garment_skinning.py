"""Bind a garment mesh to the SMPL-X body so it deforms with the pose.

Each garment vertex borrows the skinning weights of its nearest body
vertices in the rest pose, then moves with them.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.garment.skinning import GarmentBinding, align_to_body
from tryon.paths import GARMENTS


def _body():
    """A stand-in torso: a cylinder with real 3D extent.

    A coplanar sheet has rank-deficient covariance, which would test the
    degeneracy guard rather than the skinning.
    """
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    rings = []
    for height in np.linspace(-1, 1, 10):
        rings.append(
            np.stack(
                [0.2 * np.cos(angles), np.full(12, height), 0.15 * np.sin(angles)],
                axis=1,
            )
        )
    return np.vstack(rings)


def _garment(offset=0.0):
    """A tube around the body's upper half.

    A flat sheet off to one side has an ambiguous binding and shears under
    rotation, which is the fixture's geometry and not the code's fault.
    """
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    rings = []
    for height in np.linspace(0.0, 0.8, 6):
        rings.append(
            np.stack(
                [
                    0.26 * np.cos(angles),
                    np.full(10, height + offset),
                    0.20 * np.sin(angles),
                ],
                axis=1,
            )
        )
    return np.vstack(rings)


class AlignTest(unittest.TestCase):
    def test_it_removes_a_vertical_offset(self):
        body, garment = _body(), _garment(offset=2.0)
        aligned = align_to_body(garment, body)
        self.assertLess(abs(aligned[:, 1].max() - garment[:, 1].max() + 2.0), 0.35)

    def test_an_already_aligned_garment_barely_moves(self):
        body, garment = _body(), _garment()
        moved = np.abs(align_to_body(garment, body) - garment).max()
        self.assertLess(moved, 0.35)


class GarmentBindingTest(unittest.TestCase):
    def test_a_bound_garment_follows_the_body(self):
        """The property the whole rebuild exists for."""
        body = _body()
        garment = _garment()
        binding = GarmentBinding.build(garment, body)
        moved_body = body + np.array([0.5, 0.0, 0.0])
        moved_garment = binding.apply(moved_body)
        shift = moved_garment - garment
        self.assertTrue(
            np.allclose(shift, np.array([0.5, 0.0, 0.0]), atol=1e-6),
            f"garment did not follow the body: mean shift {shift.mean(axis=0)}",
        )

    def test_it_follows_a_rotation_too(self):
        """360 degrees: the garment turns because the body turns."""
        body = _body()
        garment = _garment()
        binding = GarmentBinding.build(garment, body)
        angle = np.radians(90)
        rotation = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        turned = binding.apply(body @ rotation.T)
        # A rigid turn of the body must move the garment rigidly with it.
        expected_spread = np.linalg.norm(garment - garment.mean(axis=0), axis=1)
        actual_spread = np.linalg.norm(turned - turned.mean(axis=0), axis=1)
        self.assertLess(
            float(np.abs(expected_spread - actual_spread).max()),
            0.05,
            "the garment distorted instead of turning with the body",
        )

    def test_neighbour_count_is_respected(self):
        body, garment = _body(), _garment()
        binding = GarmentBinding.build(garment, body, neighbours=3)
        self.assertEqual(binding.indices.shape[1], 3)
        self.assertTrue(np.allclose(binding.weights.sum(axis=1), 1.0))

    def test_a_binding_survives_a_round_trip(self):
        import tempfile
        from pathlib import Path

        body, garment = _body(), _garment()
        binding = GarmentBinding.build(garment, body)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bind.npz"
            binding.save(path)
            reloaded = GarmentBinding.load(path)
        self.assertTrue(np.allclose(binding.apply(body), reloaded.apply(body)))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    __import__("tryon.body.fitter", fromlist=["SMPLX_ROOT"]).SMPLX_ROOT.is_dir(),
    "no SMPL-X body models",
)
class SkinnedBindingTest(unittest.TestCase):
    """Skin the garment with the body's own LBS weights, not by copying positions.

    Nearest-vertex binding tore the garment at the joints, where neighbours
    sat on different bones. Blending the neighbours' LBS weights and skinning
    with the body's joint transforms (as SMPL does) keeps edge stretch low.
    """

    def _fixture(self):

        import torch
        import trimesh

        from tryon.garment.skinning import SkinnedGarment, align_to_body
        from tryon.body.fitter import BodyFitter

        garment = GARMENTS / "00152_outer" / "stage1" / "template_uv.obj"
        if not garment.exists():
            self.skipTest("no garment mesh")
        fitter = BodyFitter()
        with torch.no_grad():
            rest = fitter._forward().vertices[0].numpy().astype(np.float64)
        mesh = trimesh.load(str(garment), process=False)
        verts = align_to_body(np.asarray(mesh.vertices, np.float64), rest)
        return fitter, verts, np.asarray(mesh.faces), SkinnedGarment.build(verts, fitter)

    def test_it_does_not_tear_when_the_arms_rise(self):
        import torch

        fitter, verts, faces, skin = self._fixture()
        rest_edges = np.linalg.norm(verts[faces][:, 1] - verts[faces][:, 0], axis=1)
        with torch.no_grad():
            fitter.body_pose[0, 15 * 3 + 2] = -1.2
            fitter.body_pose[0, 16 * 3 + 2] = 1.2
        posed = skin.apply(fitter)
        stretch = np.linalg.norm(posed[faces][:, 1] - posed[faces][:, 0], axis=1) / np.maximum(rest_edges, 1e-9)
        self.assertLess(float(np.percentile(stretch, 99)), 2.5, f"p99 stretch {np.percentile(stretch, 99):.2f}")
        self.assertLess(int((stretch > 3).sum()), 20, f"{int((stretch > 3).sum())} faces torn")

    def test_rest_pose_reproduces_the_garment(self):
        fitter, verts, _faces, skin = self._fixture()
        self.assertLess(float(np.abs(skin.apply(fitter) - verts).max()), 1e-6)

    def test_weights_are_a_partition_of_unity(self):
        _fitter, _verts, _faces, skin = self._fixture()
        self.assertTrue(np.allclose(skin.weights.sum(axis=1), 1.0))
        self.assertGreaterEqual(float(skin.weights.min()), 0.0)

    def test_it_still_turns_with_the_body(self):
        import torch

        fitter, verts, _faces, skin = self._fixture()
        with torch.no_grad():
            fitter.global_orient[0, 1] = np.pi / 2
        turned = skin.apply(fitter)
        self.assertLess(np.ptp(turned[:, 0]), np.ptp(verts[:, 0]) * 0.8)

    def test_leg_scale_moves_the_leg_weight_to_the_other_joints(self):
        from tryon.garment.skinning import SkinnedGarment

        fitter, verts, _faces, skin = self._fixture()
        legs = list(SkinnedGarment.LEG_JOINTS)
        self.assertGreater(float(skin.weights[:, legs].sum()), 0.0)  # the coat's hem does carry leg weight
        SkinnedGarment.LEG_SCALE = 0.0
        try:
            hemmed = SkinnedGarment.build(verts, fitter)
        finally:
            SkinnedGarment.LEG_SCALE = 1.0
        self.assertLess(float(hemmed.weights[:, legs].sum()), 0.02 * float(skin.weights[:, legs].sum()))
        self.assertTrue(np.allclose(hemmed.weights.sum(axis=1), 1.0))
        self.assertGreaterEqual(float(hemmed.weights.min()), 0.0)

    def test_leg_scale_one_is_the_original_weights(self):
        from tryon.garment.skinning import SkinnedGarment

        fitter, verts, _faces, skin = self._fixture()
        self.assertEqual(SkinnedGarment.LEG_SCALE, 1.0)
        np.testing.assert_array_equal(SkinnedGarment.build(verts, fitter).weights, skin.weights)


class TransferSparseTest(unittest.TestCase):
    """`SkinnedGarment._transfer` (sparse product) equals the dense gather."""

    def test_matches_the_dense_gather(self):
        from tryon.garment.skinning import SkinnedGarment

        rng = np.random.default_rng(7)
        body = rng.normal(size=(300, 3))
        weights = rng.dirichlet(np.ones(12), size=300)
        garment = rng.normal(size=(120, 3))
        for k in (1, 7, 50):
            np.testing.assert_allclose(SkinnedGarment._transfer(garment, body, weights, k),
                                       SkinnedGarment._transfer_dense(garment, body, weights, k), atol=1e-13)
