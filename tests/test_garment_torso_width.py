"""The lower-front opening of open-front garments and the width profile study."""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_garment_hem_closure import _open_jacket
from tools.evaluate.torso_width import band_widths, pose_class, ratio_table
from tryon.garment.skinning import front_opening, piece_labels


class FrontOpeningTest(unittest.TestCase):
    def setUp(self):
        self.rest, self.faces = _open_jacket()
        self.sleeve = np.zeros(len(self.rest), bool)
        self.front = np.arange(len(self.rest)) >= 24 * 6
        self.right = self.front & (self.rest[:, 0] > 0)

    def test_the_hem_edge_opens_to_the_fraction_of_the_seam_and_the_seam_stays(self):
        moved = self.rest + front_opening(self.rest, self.faces, self.sleeve, 0.3, start=0.0, span=0.5)
        hem = self.right & np.isclose(self.rest[:, 1], -0.25)
        self.assertAlmostEqual(float(moved[hem, 0].min()), 0.3 * 0.7, places=6)
        self.assertAlmostEqual(float(moved[hem, 0].max()), 0.7, places=6)
        np.testing.assert_array_equal(moved[:, 1:], self.rest[:, 1:])

    def test_the_top_row_and_the_back_do_not_move(self):
        shift = front_opening(self.rest, self.faces, self.sleeve, 0.3, start=0.35, span=0.4)
        np.testing.assert_array_equal(shift[np.isclose(self.rest[:, 1], 0.0)], 0.0)
        back = np.arange(len(self.rest)) < 24 * 6
        np.testing.assert_array_equal(shift[back], 0.0)

    def test_zero_fraction_is_the_identity_and_a_wider_gap_is_kept(self):
        np.testing.assert_array_equal(front_opening(self.rest, self.faces, self.sleeve, 0.0), 0.0)
        # the panels' edges at 0.05 of 0.7 are already past a fraction of 0.05
        np.testing.assert_allclose(front_opening(self.rest, self.faces, self.sleeve, 0.05), 0.0, atol=1e-12)

    def test_the_opening_grows_with_depth_below_the_start(self):
        shift = np.abs(front_opening(self.rest, self.faces, self.sleeve, 0.3, start=0.2, span=0.6)[:, 0])
        edge = self.right & np.isclose(self.rest[:, 0], 0.05)
        order = np.argsort(-self.rest[edge, 1])
        self.assertTrue(np.all(np.diff(shift[edge][order]) >= -1e-12))
        self.assertGreater(shift[edge][order][-1], 0.0)

    def test_sleeve_vertices_stay_and_a_closed_garment_is_the_identity(self):
        sleeve = self.right.copy()
        np.testing.assert_array_equal(front_opening(self.rest, self.faces, sleeve, 0.3, start=0.0, span=0.5)[sleeve], 0.0)
        keep = piece_labels(self.faces, len(self.rest)) != piece_labels(self.faces, len(self.rest))[-1]
        index = np.cumsum(keep) - 1
        faces = index[self.faces[keep[self.faces].all(axis=1)]]
        np.testing.assert_array_equal(front_opening(self.rest[keep], faces, self.sleeve[keep], 0.3), 0.0)


class WidthProfileTest(unittest.TestCase):
    def test_band_extent_ignores_a_hole_and_the_count_does_not(self):
        truth = np.zeros((40, 100), bool)
        truth[:, 20:80] = True
        truth[:, 45:55] = False  # an open front
        extent, count = band_widths([truth, truth], truth, (0, 100), bands=4)
        np.testing.assert_allclose(extent[0], 60.0)
        np.testing.assert_allclose(count[0], 50.0)

    def test_the_window_clips_the_sleeves(self):
        mask = np.zeros((20, 100), bool)
        mask[:, 5:95] = True
        extent, _ = band_widths([mask], mask, (30, 70), bands=2)
        np.testing.assert_allclose(extent[0], 40.0)

    def test_pose_classes(self):
        raw = np.zeros((33, 4))
        raw[[11, 12], 1], raw[[23, 24], 1] = 300.0, 500.0
        raw[[15, 16], 1] = 600.0
        self.assertEqual(pose_class(raw), "down")
        raw[15, 1] = 400.0
        self.assertEqual(pose_class(raw), "mid")
        raw[15, 1] = 200.0
        self.assertEqual(pose_class(raw), "up")
        self.assertEqual(pose_class(np.zeros((33, 4))), "none")

    def test_ratio_table_pools_the_frames_of_a_pose(self):
        rows = [{"pose": "up", "ours": np.array([2.0, 4.0]), "truth": np.array([1.0, 2.0])},
                {"pose": "mid", "ours": np.array([1.0, 1.0]), "truth": np.array([1.0, 1.0])}]
        np.testing.assert_allclose(ratio_table(rows, pose="up"), [2.0, 2.0])
        np.testing.assert_allclose(ratio_table(rows), [1.5, 5.0 / 3.0])


if __name__ == "__main__":
    unittest.main()
