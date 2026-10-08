"""The pretrained SCHP parser wrapper (tryon/removal/parsing_schp.py)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATR = ROOT / "models" / "pretrained" / "parsing_atr.onnx"
REAL_FRAME = ROOT / "data" / "annotated" / "Take1" / "0004" / "rgb_images" / "00001.png"


class LetterboxTest(unittest.TestCase):
    def test_the_warp_is_aspect_preserving_and_centred(self):
        from tryon.removal.parsing_schp import SCHPParser

        parser = SCHPParser.__new__(SCHPParser)
        parser.size = 512
        matrix = parser._letterbox(300, 600)  # wide image
        self.assertAlmostEqual(matrix[0, 0], matrix[1, 1])
        self.assertAlmostEqual(matrix[0, 0] * 600, 512, places=3)
        self.assertAlmostEqual(matrix[0, 2], 0.0)
        self.assertAlmostEqual(matrix[1, 2], (512 - 300 * matrix[0, 0]) / 2, places=3)


@unittest.skipUnless(ATR.exists(), "run tools/data/download_pretrained.py schp")
class ParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tryon.removal.parsing_schp import SCHPParser

        cls.parser = SCHPParser()

    def test_the_live_fast_path_matches_the_full_warp(self):
        # softmax / argmax on the square, two maps warped (fast_clothing)
        # against every class warped then softmaxed (probabilities)
        import cv2

        rng = np.random.default_rng(1)
        frame = cv2.GaussianBlur(rng.integers(0, 255, (200, 140, 3), np.uint8), (9, 9), 3)
        full = self.parser.probabilities(frame, flip=False)
        probability, labels = self.parser.fast_clothing(frame, (4, 7))
        self.assertEqual(probability.shape, frame.shape[:2])
        self.assertLess(np.abs(probability - (full[4] + full[7])).mean(), 0.02)
        self.assertGreater((labels == full.argmax(axis=0)).mean(), 0.97)
        np.testing.assert_array_equal(self.parser.fast_labels(frame), labels)

    def test_the_windowed_fast_path_matches_the_reference_labels_exactly(self):
        # wide, tall and square frames (the letterbox leaves padding on two of
        # them): same probability array, same labels, same thresholded mask
        import cv2

        rng = np.random.default_rng(3)
        for shape in ((140, 300), (300, 120), (200, 200)):
            frame = cv2.GaussianBlur(rng.integers(0, 255, (*shape, 3), np.uint8), (9, 9), 3)
            p0, l0 = self.parser.fast_clothing_reference(frame, (4, 7))
            p1, l1 = self.parser.fast_clothing(frame, (4, 7))
            np.testing.assert_allclose(p0, p1, atol=1e-6)  # cv2.exp: one float32 ulp from numpy's
            np.testing.assert_array_equal(l0, l1)
            np.testing.assert_array_equal(p0 > 0.5, p1 > 0.5)
            np.testing.assert_array_equal(self.parser.fast_labels_reference(frame), self.parser.fast_labels(frame))

    def test_probabilities_are_a_distribution_at_frame_resolution(self):
        frame = np.random.default_rng(0).integers(0, 255, (96, 80, 3), np.uint8)
        p = self.parser.probabilities(frame, flip=False)
        self.assertEqual(p.shape, (18, 96, 80))
        np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-4)

    def test_flip_tta_keeps_the_distribution(self):
        frame = np.random.default_rng(1).integers(0, 255, (64, 64, 3), np.uint8)
        p = self.parser.probabilities(frame, flip=True)
        np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-4)

    @unittest.skipUnless(REAL_FRAME.exists(), "no annotated frame")
    def test_finds_upper_clothing_on_the_real_frame(self):
        import cv2

        frame = cv2.imread(str(REAL_FRAME))
        labels = self.parser.labels(frame)
        upper = np.isin(labels, (4, 7))
        self.assertGreater(upper.mean(), 0.01)
        rows = np.nonzero(upper)[0]
        self.assertLess(np.median(rows), frame.shape[0] * 0.6, "the upper clothing sits on the upper body")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless((Path(__file__).resolve().parent.parent / "models" / "pretrained" / "fashn_human_parser.onnx").exists(), "no FASHN ONNX export")
class FashnParserTest(unittest.TestCase):
    def test_probabilities_are_a_distribution_at_frame_size_and_map_to_atr(self):
        from tryon.removal.parsing_fashn import TO_ATR, FashnParser

        frame = np.full((200, 150, 3), 128, np.uint8)
        p = FashnParser().probabilities(frame)
        self.assertEqual(p.shape, (18, 200, 150))
        np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-4)
        self.assertEqual(len(TO_ATR), 18)
        self.assertEqual(int(TO_ATR[3]), 4, "top -> ATR upper-clothes")
        self.assertEqual(int(TO_ATR[16]), 14, "bare torso -> kept skin")
