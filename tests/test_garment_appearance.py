"""Per-frame appearance network of the checkpoint: StyleUNet port against
the reference's CPU code (as oracles), texel binding, and forward shape.
Checkpoint-dependent tests skip without it."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tryon.garment import appearance, styleunet, texels
from tryon.paths import garment

GARMENT = garment("00152_outer")
HAS_CHECKPOINT = (GARMENT / "stage3" / "ckpt" / "net.pt").exists()


def reference_upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """`scene/styleunet/upfirdn2d.py::upfirdn2d_native`, verbatim: the oracle."""
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    return out.view(-1, channel, out_h, out_w)


class UpFirDnTests(unittest.TestCase):
    """The FIR resampler must match the reference for every (up, down, pad) used."""

    def test_matches_reference_for_every_configuration_the_net_uses(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 16, 16)
        kernel = torch.randn(4, 4)
        cases = [
            (1, 1, (1, 1)),        # Blur, no resampling
            (1, 1, (2, 1)),        # Blur before a stride-2 conv
            (2, 1, (2, 1)),        # Upsample
            (1, 2, (1, 1)),        # Downsample
            (2, 1, (1, 0, 1, 0)),  # InverseHaarTransform
            (1, 2, (0, 0)),        # HaarTransform
        ]
        for up, down, pad in cases:
            with self.subTest(up=up, down=down, pad=pad):
                full = pad if len(pad) == 4 else (pad[0], pad[1], pad[0], pad[1])
                ours = styleunet.upfirdn2d(x, kernel, up, down, pad)
                theirs = reference_upfirdn2d(x, kernel, up, up, down, down, *full)
                self.assertEqual(ours.shape, theirs.shape)
                torch.testing.assert_close(ours, theirs, rtol=0, atol=1e-6)

    def test_haar_round_trip_is_the_identity(self):
        # The 2x2 Haar basis is orthonormal, so dwt then iwt must reconstruct.
        x = torch.randn(1, 3, 8, 8)
        back = styleunet.InverseHaarTransform(3)(styleunet.HaarTransform(3)(x))
        torch.testing.assert_close(back, x, rtol=0, atol=1e-6)


class FusedLeakyReLUTests(unittest.TestCase):
    def test_bias_then_leaky_relu_then_sqrt2(self):
        x = torch.tensor([[[[-1.0]], [[2.0]]]])
        bias = torch.tensor([0.5, -1.0])
        out = styleunet.fused_leaky_relu(x, bias)
        expected = torch.tensor([[[[-0.5 * 0.2]], [[1.0]]]]) * 2 ** 0.5
        torch.testing.assert_close(out, expected)


class MergeTests(unittest.TestCase):
    def _splats(self, n):
        return {
            "xyz": np.zeros((n, 3)), "rotation": np.ones((n, 4)), "scaling": np.zeros((n, 3)),
            "feature_dc": np.zeros((n, 1, 3)), "feature_rest": np.zeros((n, 15, 3)), "opacity": np.zeros((n, 1)),
        }

    def _result(self, n):
        return {"xyz": np.arange(n * 3.0).reshape(n, 3), "feature_dc": np.ones((n, 1, 3)),
                "feature_rest": np.ones((n, 15, 3))}

    def test_replaces_only_colour_and_position(self):
        merged = appearance.merge(self._splats(4), self._result(4))
        np.testing.assert_array_equal(merged["xyz"], np.arange(12.0).reshape(4, 3))
        self.assertTrue((merged["feature_dc"] == 1).all())
        self.assertTrue((merged["rotation"] == 1).all())
        self.assertTrue((merged["opacity"] == 0).all())

    def test_subset_needs_indices(self):
        with self.assertRaises(ValueError):
            appearance.merge(self._splats(2), self._result(4))
        merged = appearance.merge(self._splats(2), self._result(4), indices=np.array([1, 3]))
        np.testing.assert_array_equal(merged["xyz"], [[3, 4, 5], [9, 10, 11]])


class WorldToViewTests(unittest.TestCase):
    def test_is_the_transposed_rigid_transform(self):
        # The reference stores the rigid transform transposed (row-vector
        # convention), so rotating a normal applies the camera-to-world rotation.
        theta = 0.3
        w2c = np.eye(4)
        w2c[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        w2c[:3, 3] = [1, 2, 3]
        stored = appearance._world_to_view_transposed(w2c)
        np.testing.assert_allclose(stored, w2c.T, atol=1e-6)
        n = np.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose((stored @ n)[:3], w2c[:3, :3].T @ n[:3], atol=1e-6)


@unittest.skipUnless(HAS_CHECKPOINT, f"no garment checkpoint at {GARMENT}")
class TexelBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = texels.read_template(GARMENT / "stage1" / "template_uv.obj")
        cls.binding = texels.TexelBinding(cls.template)

    def test_one_gaussian_per_covered_texel_matches_the_checkpoint(self):
        state = torch.load(GARMENT / "stage3" / "ckpt" / "net.pt", map_location="cpu")["avatar_net"]
        self.assertEqual(self.binding.num_gs, state["xyz"].shape[0])
        self.assertEqual(self.binding.num_gs, 93491)

    def test_agrees_with_the_renderers_binding(self):
        # The covered texel set must match the separate rasteriser's; an
        # edge texel may fall to either neighbour.
        from tryon.garment.checkpoint import load_checkpoint, uv_binding

        _, mesh = load_checkpoint(GARMENT)
        ours, resolution = uv_binding(mesh, self.binding.num_gs)
        self.assertEqual(resolution, self.binding.res)
        self.assertEqual(len(ours), self.binding.num_gs)
        self.assertLessEqual(int((ours != self.binding.binding).sum()), 4)

    def test_barycentric_weights_are_valid(self):
        # Atlas UVs are float32, so a texel centre can sit ~1e-6 outside the float64 triangle.
        bc = self.binding.bc
        np.testing.assert_allclose(bc.sum(1), 1.0, atol=1e-9)
        self.assertGreaterEqual(bc.min(), -1e-4)
        self.assertLessEqual(bc.max(), 1 + 1e-4)

    def test_anchor_lies_on_its_face(self):
        v, f = self.template["vertices"].astype(np.float64), self.template["faces"]
        anchor = self.binding.barycentric_3d(v[f])
        tri = v[f][self.binding.binding]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        distance = np.abs(((anchor - tri[:, 0]) * normal).sum(1))
        self.assertLess(distance.max(), 1e-9)


@unittest.skipUnless(HAS_CHECKPOINT, f"no garment checkpoint at {GARMENT}")
class AppearanceNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        cls.net = appearance.AppearanceNet(GARMENT)
        cls.vertices = cls.net.template["vertices"].astype(np.float64)
        cams = json.load(open(GARMENT / "stage1" / "cameras.json"))["0004"]
        cls.view = np.eye(4)
        cls.view[:3] = np.asarray(cams["extrinsics"])
        cls.ambient, cls.normal = cls.net.bake(cls.vertices)
        cls.result = cls.net.forward(cls.vertices, cls.view, ambient=cls.ambient, normal=cls.normal)

    def test_output_shapes_match_the_splat_layout(self):
        n = self.net.num_gs
        r = self.result
        self.assertEqual(r["feature_dc"].shape, (n, 1, 3))
        self.assertEqual(r["feature_rest"].shape, (n, 15, 3))
        self.assertEqual(r["xyz"].shape, (n, 3))
        self.assertEqual(r["offset"].shape, (n, 3))
        self.assertEqual(r["shadow_shs"].shape, (n, 16, 3))
        self.assertEqual(r["ambient"].shape, (512, 512))
        self.assertEqual(r["normal"].shape, (512, 512, 3))

    def test_shadow_output_is_not_trivial(self):
        self.assertGreater(np.abs(self.result["shadow_shs"]).mean(), 1e-3)
        self.assertGreater(np.abs(self.result["offset"]).mean(), 1e-3)
        self.assertTrue(np.isfinite(self.result["shadow_shs"]).all())
        # Static + shadow, not shadow alone.
        static = self.net.static_features()
        np.testing.assert_allclose(
            np.concatenate([self.result["feature_dc"], self.result["feature_rest"]], 1) - self.result["shadow_shs"],
            static, atol=1e-6)

    def test_bake_maps_are_in_range_and_dilated(self):
        covered = self.net.binding.gaussian_mask
        self.assertGreaterEqual(self.ambient.min(), 0.0)
        self.assertLessEqual(self.ambient.max(), 1.0)
        self.assertGreater(self.ambient[covered].mean(), 0.5)  # a shirt is mostly open to the sky
        self.assertTrue((self.normal[covered].sum(-1) > 0).all())
        filled = self.normal.sum(-1) > 0
        self.assertGreater(filled.sum(), covered.sum())  # the margin
        self.assertLess(filled.sum(), 3 * covered.sum())

    def test_view_direction_branch_does_not_reach_the_output(self):
        with_view = self.net.forward(self.vertices, self.view, ambient=self.ambient, normal=self.normal, use_viewdir=True)
        self.assertEqual(float(np.abs(with_view["shadow_shs"] - self.result["shadow_shs"]).max()), 0.0)
        self.assertEqual(float(np.abs(with_view["offset"] - self.result["offset"]).max()), 0.0)

    def test_renderer_mesh_order_is_accepted(self):
        from tryon.garment.checkpoint import load_checkpoint

        _, mesh = load_checkpoint(GARMENT)
        net = appearance.AppearanceNet.__new__(appearance.AppearanceNet)
        net.__dict__.update(self.net.__dict__)
        net.faces_by_vertex_count = dict(self.net.faces_by_vertex_count)
        net.faces_by_vertex_count[len(mesh.vertices)] = np.asarray(mesh.faces)
        anchored = net.static_xyz(np.asarray(mesh.vertices, np.float64))
        np.testing.assert_allclose(anchored, self.net.static_xyz(self.vertices), atol=1e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "no MPS device")
    def test_mps_and_cpu_agree(self):
        self.assertEqual(self.net.device.type, "mps")
        cpu = appearance.AppearanceNet(GARMENT, device="cpu")
        on_cpu = cpu.forward(self.vertices, self.view, ambient=self.ambient, normal=self.normal)
        np.testing.assert_allclose(on_cpu["shadow_shs"], self.result["shadow_shs"], atol=1e-4)
        np.testing.assert_allclose(on_cpu["offset"], self.result["offset"], atol=1e-4)
        np.testing.assert_allclose(on_cpu["xyz"], self.result["xyz"], atol=1e-5)


if __name__ == "__main__":
    unittest.main()


class DilationCacheTest(unittest.TestCase):
    def test_cached_fill_equals_a_fresh_one(self):
        from scipy import ndimage

        from tryon.garment.texels import _dilate_map

        rng = np.random.default_rng(1)
        covered = rng.random((64, 64)) > 0.6
        values = rng.random((64, 64, 3))
        distance, (rows, cols) = ndimage.distance_transform_edt(~covered, return_indices=True)
        fill = (~covered) & (distance <= 5)
        expected = values.copy()
        expected[fill] = values[rows[fill], cols[fill]]
        for _ in range(2):  # the second call is served from the cache
            np.testing.assert_array_equal(_dilate_map(values, covered, 5), expected)


class TestWorldXyzCompiled(unittest.TestCase):
    """`world_xyz` (numba) against `_world_xyz_numpy` on a random mesh."""

    def test_matches_numpy(self):
        from types import SimpleNamespace

        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(60, 3))
        faces = rng.integers(0, 60, size=(80, 3))
        binding = rng.integers(0, 80, size=500)
        bc = rng.dirichlet(np.ones(3), size=500)
        bind = SimpleNamespace(binding=binding, bc=bc,
                               barycentric_3d=lambda tri: np.einsum("nk,nkd->nd", bc, np.asarray(tri, np.float64)[binding]))
        net = SimpleNamespace(faces_for=lambda v: faces, binding=bind)
        local = rng.normal(size=(500, 3))
        fast = appearance.AppearanceNet.world_xyz(net, vertices, local)
        slow = appearance.AppearanceNet._world_xyz_numpy(net, vertices, local)
        np.testing.assert_allclose(fast, slow, atol=1e-12)


class TextureBindingCompiledTest(unittest.TestCase):
    """`texture_binding` (numba) is texel-for-texel the Python port."""

    def test_matches_the_python_scanline(self):
        rng = np.random.default_rng(5)
        uvs = rng.uniform(0.0, 1.0, (300, 2)).astype(np.float32)
        faces = rng.integers(0, 300, (400, 3))
        # small, thin and axis-aligned triangles too (the edge cases of the scanline)
        uvs[:3] = [[0.1, 0.1], [0.1, 0.5], [0.4, 0.1]]
        faces[:1] = [0, 1, 2]
        fast = texels.texture_binding(uvs, faces, res=128)
        slow = texels._texture_binding_python(uvs, faces, res=128)
        np.testing.assert_array_equal(fast, slow)
