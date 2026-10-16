"""tryon/garment/shading_student.py: the stored per-Gaussian mean."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tryon.paths import garment

GARMENT = garment("00170_upper")


@unittest.skipUnless((GARMENT / "stage1").is_dir(), f"no garment checkpoint at {GARMENT}")
class StoredMeanTest(unittest.TestCase):
    def test_a_zero_residual_returns_the_stored_mean(self):
        from tryon.garment.shading_student import ShadingStudent, StudentUNet

        model = StudentUNet()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            from tryon.garment.appearance import AppearanceNet

            count = len(AppearanceNet(GARMENT, device="cpu", network=False).xyz)
            mean = np.random.default_rng(0).normal(0, 0.01, (count, 51)).astype(np.float16)
            path = Path(tmp) / "student.pt"
            torch.save({"model": model.state_dict(), "mean": mean}, path)
            student = ShadingStudent(GARMENT, device="cpu", weights=path)
        vertices = np.asarray(student.static.template["vertices"], np.float64)
        body = vertices + 0.05  # any body: the residual is zero whatever the inputs
        result = student.forward(vertices, np.eye(4), body)
        np.testing.assert_allclose(result["local_xyz"], student.static.xyz + mean[:, :3].astype(np.float64), atol=1e-6)
        sh = student.static.static_features() + mean[:, 3:].astype(np.float64).reshape(-1, 16, 3)
        np.testing.assert_allclose(result["feature_dc"], sh[:, :1], atol=1e-6)

    def test_shadow_strength_scales_only_the_shading(self):
        from tryon.garment.appearance import AppearanceNet
        from tryon.garment.shading_student import ShadingStudent, StudentUNet

        model = StudentUNet()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        count = len(AppearanceNet(GARMENT, device="cpu", network=False).xyz)
        mean = np.random.default_rng(1).normal(0, 0.05, (count, 51)).astype(np.float16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.pt"
            torch.save({"model": model.state_dict(), "mean": mean}, path)
            student = ShadingStudent(GARMENT, device="cpu", weights=path)
        vertices = np.asarray(student.static.template["vertices"], np.float64)
        student.shadow_strength = 0.5
        result = student.forward(vertices, np.eye(4), vertices + 0.05)
        shadow = mean[:, 3:].astype(np.float64).reshape(-1, 16, 3)
        np.testing.assert_allclose(result["feature_dc"], student.static.static_features()[:, :1] + 0.5 * shadow[:, :1], atol=1e-6)


class WidthsTest(unittest.TestCase):
    def test_widths_change_the_capacity_not_the_interface(self):
        import torch

        from tryon.garment.shading_student import IN_CHANNELS, OUT_CHANNELS, StudentUNet

        small, wide = StudentUNet(), StudentUNet((48, 96, 192, 288))
        def count(m):
            return sum(p.numel() for p in m.parameters())

        self.assertGreater(count(wide), 2 * count(small))
        x = torch.zeros(1, IN_CHANNELS, 32, 32)
        self.assertEqual(tuple(small(x).shape), tuple(wide(x).shape))
        self.assertEqual(small(x).shape[1], OUT_CHANNELS)


if __name__ == "__main__":
    unittest.main()
