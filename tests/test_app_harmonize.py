"""tryon/app/harmonize.py: the garment recoloured to the scene, nothing else."""

import unittest

import numpy as np

from tryon.app.harmonize import HARMONIZER_ROOT


@unittest.skipUnless((HARMONIZER_ROOT / "pretrained" / "harmonizer.pth").exists(), "needs the Harmonizer clone and its weights")
class LiveHarmonizerTest(unittest.TestCase):
    def test_only_the_foreground_changes_and_the_shape_is_kept(self):
        from tryon.app.harmonize import LiveHarmonizer

        rng = np.random.default_rng(0)
        frame = (rng.random((240, 180, 3)) * 80 + 60).astype(np.uint8)
        frame[60:180, 50:130] = (200, 90, 40)  # a garment far from the scene's colours
        mask = np.zeros((240, 180), bool)
        mask[60:180, 50:130] = True
        out = LiveHarmonizer(predict_device="cpu")(frame, mask)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        outside = np.abs(out[~mask].astype(int) - frame[~mask].astype(int)).max()
        self.assertLessEqual(outside, 1, "pixels outside the garment must stay as they were")

    def test_an_empty_mask_is_a_no_op(self):
        from tryon.app.harmonize import LiveHarmonizer

        frame = np.full((64, 48, 3), 100, np.uint8)
        out = LiveHarmonizer(predict_device="cpu")(frame, np.zeros((64, 48), bool))
        self.assertIs(out, frame)
