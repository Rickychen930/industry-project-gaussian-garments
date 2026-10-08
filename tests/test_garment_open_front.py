"""tools/evaluate/open_front.py: the pieces of the open-front study that do not need a recorded clip."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate import open_front
from tryon.garment.cloth import LightCloth


class PieceLabelsTest(unittest.TestCase):
    def test_disconnected_triangles_are_separate_pieces_and_shared_vertices_join_them(self):
        faces = np.array([[0, 1, 2], [3, 4, 5], [2, 6, 7]])
        labels = open_front.piece_labels(faces, 8)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[0], labels[6])
        self.assertNotEqual(labels[0], labels[3])
        self.assertEqual(labels[3], labels[5])


class SetAttributesTest(unittest.TestCase):
    def test_assignments_are_applied_and_undone(self):
        original = LightCloth.WRAP
        undo = open_front.set_attributes(["tryon.garment.cloth.LightCloth.WRAP=(0.02, 0.8)"])
        try:
            self.assertEqual(LightCloth.WRAP, (0.02, 0.8))
        finally:
            for owner, attribute, old in reversed(undo):
                setattr(owner, attribute, old)
        self.assertEqual(LightCloth.WRAP, original)

    def test_none_and_strings_are_parsed(self):
        undo = open_front.set_attributes(["tryon.garment.scan_pose.ScanPoseBinder.REFINE_ARMS=None"])
        try:
            from tryon.garment.scan_pose import ScanPoseBinder

            self.assertIsNone(ScanPoseBinder.REFINE_ARMS)
        finally:
            for owner, attribute, old in reversed(undo):
                setattr(owner, attribute, old)


class CompareTest(unittest.TestCase):
    def test_the_table_has_a_row_per_video_and_the_open_front_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            for video, value in (("00122_outer", 0.5), ("00152_outer", 0.7), ("00170_upper", 0.8)):
                Path(folder, f"rep_x_{video}.json").write_text(json.dumps({"summary": {"iou": value, "inside_share": 0.01}}))
            table = open_front.compare(["x"], folder)
        self.assertIn("x | 00122_outer | 0.5000", table)
        self.assertIn("x | open-front mean (2) | 0.6000", table)
        self.assertIn("x | closed mean (1) | 0.8000", table)


if __name__ == "__main__":
    unittest.main()
