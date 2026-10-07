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


if __name__ == "__main__":
    unittest.main()
