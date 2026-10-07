"""tools/evaluate/_common.py: the mask IoU, the panel crop, the JSON the evaluators write."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tools.evaluate._common import iou, panel_frames, write_json


class IouTest(unittest.TestCase):
    def test_partial_overlap(self):
        a = np.array([1, 1, 0, 0], bool)
        b = np.array([0, 1, 1, 0], bool)
        self.assertAlmostEqual(iou(a, b), 1 / 3)

    def test_empty_union_defaults_to_agreement(self):
        empty = np.zeros(4, bool)
        self.assertEqual(iou(empty, empty), 1.0)

    def test_empty_union_can_be_zero(self):
        empty = np.zeros(4, bool)
        self.assertEqual(iou(empty, empty, empty=0.0), 0.0)

    def test_empty_keyword_does_not_touch_a_real_union(self):
        a = np.array([1, 1], bool)
        self.assertEqual(iou(a, a, empty=0.0), 1.0)
        self.assertEqual(iou(a, ~a, empty=1.0), 0.0)


class WriteJsonTest(unittest.TestCase):
    def test_bytes_match_the_old_pattern(self):
        data = {"a": [1, 2.5, None], "b": {"c": "é"}, "d": float("nan")}
        with tempfile.TemporaryDirectory() as tmp:
            new, old, old_dump = Path(tmp) / "new.json", Path(tmp) / "old.json", Path(tmp) / "dump.json"
            write_json(new, data)
            old.write_text(json.dumps(data, indent=1))
            with open(old_dump, "w") as handle:
                json.dump(data, handle, indent=1)
            self.assertEqual(new.read_bytes(), old.read_bytes())
            self.assertEqual(new.read_bytes(), old_dump.read_bytes())
            self.assertFalse(new.read_bytes().endswith(b"\n"))

    def test_accepts_str_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.json")
            write_json(path, [1])
            self.assertEqual(json.loads(Path(path).read_text()), [1])


class PanelFramesTest(unittest.TestCase):
    """A release video is a white page with the photo panel on its left half."""

    FRAMES, PANEL = 6, (slice(40, 360), slice(20, 300))  # rows, columns of the photo

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (960, 400))
        for t in range(self.FRAMES):
            frame = np.full((400, 960, 3), 255, np.uint8)
            frame[self.PANEL] = (60 + 20 * t, 90, 120)
            writer.write(frame)
        writer.release()
        if not path.exists() or path.stat().st_size == 0:
            self.skipTest("this OpenCV build cannot write mp4v")

    def panels(self, **kwargs):
        with mock.patch("tryon.paths.RELEASE_VIDEOS", Path(self.tmp.name)):
            return panel_frames("clip", **kwargs)

    def test_crops_and_scales_every_frame(self):
        out = self.panels(height=160, frames=self.FRAMES)
        self.assertEqual(len(out), self.FRAMES)
        for panel in out:
            self.assertEqual(panel.shape[0], 160)
            self.assertAlmostEqual(panel.shape[1], 140, delta=4)

    def test_frames_limit(self):
        self.assertEqual(len(self.panels(height=160, frames=3)), 3)

    def test_more_frames_than_the_video_has(self):
        self.assertEqual(len(self.panels(height=160, frames=50)), self.FRAMES)

    def test_step_keeps_every_nth_frame(self):
        self.assertEqual(len(self.panels(height=160, frames=self.FRAMES, step=2)), 3)


if __name__ == "__main__":
    unittest.main()
