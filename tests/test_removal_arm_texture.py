import unittest

import numpy as np

from tryon.removal.remover import ClothingRemover
from tools.evaluate.arm_visibility import visibility


class SkinTileTest(unittest.TestCase):
    def setUp(self):
        self.remover = object.__new__(ClothingRemover)

    def test_tile_has_gray_shape_and_zero_mean_patch(self):
        rng = np.random.default_rng(0)
        gray = rng.normal(120, 8, (100, 90)).astype(np.float32)
        skin = np.zeros(gray.shape, bool)
        skin[10:70, 10:70] = True
        tiled = self.remover._skin_tile(gray, skin)
        self.assertEqual(tiled.shape, gray.shape)
        self.assertGreater(float(tiled.std()), 0.5)

    def test_none_without_enough_skin(self):
        gray = np.zeros((100, 90), np.float32)
        skin = np.zeros(gray.shape, bool)
        skin[:20, :20] = True
        self.assertIsNone(self.remover._skin_tile(gray, skin))

    def test_grain_uses_skin_tile_only_when_switched_on(self):
        rng = np.random.default_rng(1)
        crop = rng.integers(90, 160, (100, 100, 3)).astype(np.uint8)
        skin = np.zeros((100, 100), bool)
        skin[:60, :60] = True
        off = ClothingRemover.__new__(ClothingRemover)
        on = ClothingRemover.__new__(ClothingRemover)
        on.ARM_SKIN_TEXTURE = True
        self.assertFalse(ClothingRemover.ARM_SKIN_TEXTURE)
        self.assertEqual(off._grain(crop, skin).shape, (100, 100, 1))
        self.assertEqual(on._grain(crop, skin).shape, (100, 100, 1))

    def test_visibility_counts_uncovered_tube(self):
        shape = (200, 200)
        pts = np.zeros((33, 3))
        pts[11] = (80, 50, 1)
        pts[12] = (120, 50, 1)
        pts[13] = (60, 100, 1)
        pts[15] = (60, 150, 1)
        mask = np.ones(shape, bool)
        frame = np.zeros(shape + (3,), np.uint8)
        sample = {"points": pts, "mask": mask, "keep": np.zeros(shape, bool), "frame": frame,
                  "erased": frame + 50, "alpha": np.zeros(shape, np.uint8)}
        total, visible = visibility(sample, 0.22)
        self.assertGreater(total, 0)
        self.assertEqual(total, visible)
        sample["alpha"][:] = 255
        self.assertEqual(visibility(sample, 0.22)[1], 0)


if __name__ == "__main__":
    unittest.main()
