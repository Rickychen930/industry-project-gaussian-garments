"""The live window's on-screen controls (tryon/app/ui.py): layout, clicks, keys, drawing."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tryon.app import ui

NAMES = ["00190_upper", "00176_upper", "00136_outer", "00176_lower", "10006_outer", "custom_garment"]
FRAME = np.full((540, 960, 3), 120, np.uint8)


def _centre(rect):
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)


class LayoutTest(unittest.TestCase):
    def test_every_control_fits_inside_the_frame(self):
        rects, per_page = ui.layout(960, 540, len(NAMES), 0, hidden=False)
        self.assertGreaterEqual(per_page, 1)
        for name, (x0, y0, x1, y1) in rects.items():
            self.assertTrue(0 <= x0 < x1 <= 960 and 0 <= y0 < y1 <= 540, name)

    def test_chips_do_not_overlap_and_stay_between_the_arrows(self):
        rects, _ = ui.layout(960, 540, len(NAMES), 0, hidden=False)
        chips = sorted(r for name, r in rects.items() if name.startswith("chip:"))
        self.assertGreater(len(chips), 1)
        for left, right in zip(chips, chips[1:]):
            self.assertLessEqual(left[2], right[0])
        self.assertGreaterEqual(chips[0][0], rects["prev"][2])
        self.assertLessEqual(chips[-1][2], rects["next"][0])

    def test_a_page_shows_the_chips_it_is_asked_for(self):
        _, per_page = ui.layout(960, 540, len(NAMES), 0, hidden=False)
        rects, _ = ui.layout(960, 540, len(NAMES), 1, hidden=False)
        self.assertIn(f"chip:{per_page}", rects)
        self.assertNotIn("chip:0", rects)

    def test_hidden_leaves_only_the_button_to_bring_the_controls_back(self):
        rects, _ = ui.layout(960, 540, len(NAMES), 0, hidden=True)
        self.assertEqual(list(rects), ["show"])

    def test_no_garments_means_no_bar(self):
        rects, _ = ui.layout(960, 540, 0, 0, hidden=False)
        self.assertEqual(set(rects), {"exit", "snapshot", "hide"})

    def test_a_narrow_frame_still_offers_a_chip_inside_the_frame(self):
        rects, per_page = ui.layout(320, 240, len(NAMES), 0, hidden=False)
        self.assertGreaterEqual(per_page, 1)
        self.assertIn("chip:0", rects)
        self.assertLessEqual(rects["chip:0"][2], 320)


class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.overlay = ui.Overlay(NAMES, current=0, snapshot_dir=self.dir.name)
        patch = mock.patch("cv2.getWindowProperty", return_value=1.0)
        patch.start()
        self.addCleanup(patch.stop)
        self.overlay.render(FRAME)  # the first frame shown: the layout is known from here

    def click(self, name):
        rects, _ = ui.layout(960, 540, len(NAMES), self.overlay.page, self.overlay.hidden)
        self.overlay._mouse(cv2.EVENT_LBUTTONUP, *_centre(rects[name]), 0, None)

    def test_render_leaves_the_input_frame_alone(self):
        before = FRAME.copy()
        out = self.overlay.render(FRAME)
        np.testing.assert_array_equal(FRAME, before)
        self.assertFalse(np.array_equal(out, FRAME))

    def test_the_bar_opens_on_the_page_of_the_garment_worn(self):
        overlay = ui.Overlay(NAMES, current=len(NAMES) - 1, snapshot_dir=self.dir.name)
        overlay.render(FRAME)
        rects, _ = ui.layout(960, 540, len(NAMES), overlay.page, False)
        self.assertIn(f"chip:{len(NAMES) - 1}", rects)

    def test_selecting_a_garment_on_another_page_brings_its_page_up(self):
        self.overlay.poll(ord("6"))
        self.overlay.render(FRAME)
        rects, _ = ui.layout(960, 540, len(NAMES), self.overlay.page, False)
        self.assertIn("chip:5", rects)

    def test_clicking_a_chip_selects_that_garment(self):
        self.click("chip:1")
        self.assertEqual(self.overlay.poll(), ("select", 1))
        self.assertEqual(self.overlay.current, 1)

    def test_clicking_the_worn_garment_does_nothing(self):
        self.click("chip:0")
        self.assertIsNone(self.overlay.poll())

    def test_a_click_is_used_once(self):
        self.click("chip:1")
        self.overlay.poll()
        self.assertIsNone(self.overlay.poll())

    def test_exit_button_and_keys_exit(self):
        self.click("exit")
        self.assertEqual(self.overlay.poll(), ("exit",))
        for key in (ord("q"), 27):
            self.assertEqual(self.overlay.poll(key), ("exit",))

    def test_number_keys_select_and_n_p_step_around_the_wardrobe(self):
        self.assertEqual(self.overlay.poll(ord("3")), ("select", 2))
        self.assertEqual(self.overlay.poll(ord("n")), ("select", 3))
        self.assertEqual(self.overlay.poll(ord("p")), ("select", 2))
        self.overlay.current = 0
        self.assertEqual(self.overlay.poll(ord("p")), ("select", len(NAMES) - 1))

    def test_a_number_beyond_the_wardrobe_is_ignored(self):
        self.assertIsNone(self.overlay.poll(ord("9")))

    def test_arrows_turn_the_page_and_wrap(self):
        pages = -(-len(NAMES) // self.overlay._per_page)
        self.assertGreater(pages, 1)
        self.click("next")
        self.assertIsNone(self.overlay.poll())
        self.assertEqual(self.overlay.page, 1)
        self.click("prev")
        self.overlay.poll()
        self.click("prev")
        self.overlay.poll()
        self.assertEqual(self.overlay.page, pages - 1)

    def test_hide_and_show_toggle_the_controls(self):
        self.click("hide")
        self.overlay.poll()
        self.assertTrue(self.overlay.hidden)
        self.overlay.render(FRAME)
        self.click("show")
        self.overlay.poll()
        self.assertFalse(self.overlay.hidden)

    def test_snapshot_saves_the_clean_frame_without_controls(self):
        self.click("snapshot")
        self.assertIsNone(self.overlay.poll())
        saved = list(Path(self.dir.name).glob("tryon-*.jpg"))
        self.assertEqual(len(saved), 1)
        picture = cv2.imread(str(saved[0]))
        self.assertEqual(picture.shape, FRAME.shape)
        self.assertLess(np.abs(picture.astype(int) - FRAME).max(), 8)  # JPEG noise only: no buttons drawn in

    def test_the_windows_own_close_button_exits(self):
        self.assertIsNone(self.overlay.poll())  # seen open once
        with mock.patch("cv2.getWindowProperty", return_value=0.0):
            self.assertEqual(self.overlay.poll(), ("exit",))
        self.assertTrue(self.overlay.closed)

    def test_a_backend_that_never_reports_visible_is_not_taken_for_closed(self):
        overlay = ui.Overlay(NAMES, snapshot_dir=self.dir.name)
        overlay.render(FRAME)
        with mock.patch("cv2.getWindowProperty", return_value=-1.0):
            self.assertIsNone(overlay.poll())

    def test_nothing_happens_before_the_first_frame_is_drawn(self):
        overlay = ui.Overlay(NAMES, snapshot_dir=self.dir.name)
        self.assertIsNone(overlay.poll(ord("q")))

    def test_without_garments_only_exit_snapshot_and_hide_work(self):
        overlay = ui.Overlay(snapshot_dir=self.dir.name)
        overlay.render(FRAME)
        self.assertIsNone(overlay.poll(ord("n")))
        self.assertIsNone(overlay.poll(ord("1")))
        self.assertEqual(overlay.poll(ord("q")), ("exit",))

    def test_unknown_checkpoints_are_shown_by_their_folder_name(self):
        self.assertEqual(ui.describe("custom_garment"), ("custom_garment", ""))
        self.assertEqual(ui.describe("00190_upper")[0], "Striped sweater")

    def test_banner_draws_text_over_the_controls(self):
        self.assertFalse(np.array_equal(self.overlay.banner(FRAME, "Loading ..."), self.overlay.render(FRAME)))


if __name__ == "__main__":
    unittest.main()
