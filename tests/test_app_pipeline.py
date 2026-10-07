"""The live loop's torso gate (tryon/app/pipeline.py::torso_visible)."""

from __future__ import annotations

import unittest

import numpy as np

from tryon.app.person import TORSO, TRACKING_VISIBILITY_FLOOR, torso_visible
from tryon.body.landmarks import VISIBILITY_FLOOR


class TorsoGateTest(unittest.TestCase):
    def _points(self, torso_visibility):
        points = np.zeros((33, 4), np.float32)
        points[:, 2] = 1.0
        for index in TORSO:
            points[index, 2] = torso_visibility
        return points

    def test_a_cold_start_needs_the_full_floor(self):
        self.assertFalse(torso_visible(self._points(VISIBILITY_FLOOR - 0.05)))
        self.assertTrue(torso_visible(self._points(VISIBILITY_FLOOR + 0.05)))

    def test_a_tracked_body_keeps_tracking_at_profile_visibility(self):
        # MediaPipe reports ~0.6-0.9 for the torso of a tracked wearer in profile
        self.assertTrue(torso_visible(self._points(0.3), tracking=True))
        self.assertFalse(torso_visible(self._points(TRACKING_VISIBILITY_FLOOR - 0.05), tracking=True))

    def test_the_torso_is_shoulders_and_hips_and_a_hidden_elbow_does_not_lose_the_wearer(self):
        self.assertEqual(TORSO, (11, 12, 23, 24))
        points = self._points(1.0)
        points[13, 2] = 0.14  # far elbow behind the body in profile (00176_outer)
        self.assertTrue(torso_visible(points))
        self.assertTrue(torso_visible(points, tracking=True))

    def test_no_landmarks_is_never_visible(self):
        self.assertFalse(torso_visible(None))
        self.assertFalse(torso_visible(None, tracking=True))


class KeepInFrontTest(unittest.TestCase):
    def test_face_and_hair_win_but_a_face_label_far_from_the_head_does_not(self):
        from tryon.app.compositing import keep_in_front

        original = np.full((100, 100, 3), 50, np.uint8)
        frame = np.full((100, 100, 3), 200, np.uint8)  # the garment drawn everywhere
        labels = np.zeros((100, 100), np.uint8)
        labels[10:20, 40:60] = 11  # the face
        labels[5:10, 40:60] = 2  # hair
        labels[70:80, 40:60] = 11  # a fur collar the parser called "face"
        points = np.zeros((33, 4), np.float32)
        points[[0, 2, 5, 7, 8], :2] = (50, 15)
        points[[0, 2, 5, 7, 8], 2] = 1.0
        points[11, :3] = (30, 30, 1.0)
        points[12, :3] = (70, 30, 1.0)
        keep_in_front(frame, original, labels, points)
        self.assertTrue((frame[10:20, 40:60] == 50).all(), "the face is in front")
        self.assertTrue((frame[5:10, 40:60] == 50).all(), "hair is in front")
        self.assertTrue((frame[70:80, 40:60] == 200).all(), "a far 'face' label stays under the garment")

    def test_a_hand_in_its_wrist_disc_stays_in_front(self):
        from tryon.app.compositing import keep_in_front

        original = np.full((200, 200, 3), 50, np.uint8)
        frame = np.full((200, 200, 3), 200, np.uint8)
        labels = np.zeros((200, 200), np.uint8)
        points = np.zeros((33, 4), np.float32)
        points[11, :3] = (80, 40, 1.0)  # shoulders 40 px apart
        points[12, :3] = (120, 40, 1.0)
        points[13, :3] = (60, 80, 1.0)  # left elbow
        points[15, :3] = (60, 120, 1.0)  # left wrist; extended wrist at (60, 132)
        labels[128:136, 56:64] = 15  # the hand
        labels[60:70, 56:64] = 15  # forearm skin far from the wrist: not a hand
        keep_in_front(frame, original, labels, points)
        self.assertTrue((frame[128:136, 56:64] == 50).all(), "the hand is in front")
        self.assertTrue((frame[60:70, 56:64] == 200).all(), "the forearm is not")

    def test_an_arm_across_the_head_is_not_restored_as_hair(self):
        from tryon.app.compositing import keep_in_front

        original = np.full((200, 200, 3), 50, np.uint8)
        frame = np.full((200, 200, 3), 200, np.uint8)
        labels = np.zeros((200, 200), np.uint8)
        points = np.zeros((33, 4), np.float32)
        points[[0, 2, 5, 7, 8], :3] = (100, 30, 1.0)
        points[11, :3] = (80, 60, 1.0)
        points[12, :3] = (120, 60, 1.0)
        points[13, :3] = (70, 30, 1.0)   # left elbow up by the head
        points[15, :3] = (110, 10, 1.0)  # left wrist over it
        labels[15:30, 85:100] = 2        # "hair" where the forearm crosses
        labels[40:50, 140:150] = 2       # real hair away from the arm
        keep_in_front(frame, original, labels, points)
        self.assertTrue((frame[20:25, 88:95] == 200).all(), "the sleeve over the head stays covered")
        self.assertTrue((frame[40:50, 140:150] == 50).all(), "hair elsewhere is restored")


class DressableTest(unittest.TestCase):
    """`dressable`: with the image regressor the shoulders are enough."""

    def _desk(self):
        points = np.zeros((33, 4), np.float32)
        points[:, 2] = 1.0
        points[[23, 24], 2] = 0.02  # hips below the frame, a wearer at a desk
        return points

    def test_the_hips_below_the_frame_need_the_regressor(self):
        from tryon.app.person import dressable

        self.assertFalse(dressable(self._desk(), tracking=True, regression=False))
        self.assertTrue(dressable(self._desk(), tracking=True, regression=True))

    def test_the_shoulders_are_still_required(self):
        from tryon.app.person import dressable

        points = self._desk()
        points[11, 2] = 0.1
        self.assertFalse(dressable(points, tracking=False, regression=True))
        self.assertFalse(dressable(None, tracking=False, regression=True))


class FramingHintTest(unittest.TestCase):
    def test_a_person_cut_off_at_the_shoulders_is_told_to_step_back(self):
        from tryon.app.person import framing_hint

        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        self.assertIsNone(framing_hint(None))
        self.assertIn("hips", framing_hint(points))
        points[[11, 12], 2] = 0.0  # shoulders out of view
        self.assertIn("shoulders", framing_hint(points))
