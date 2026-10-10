"""The live try-on opens the laptop's camera by name: a phone offered as a camera never takes its place."""

import unittest
from unittest import mock

from tryon.app import camera

LISTING = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] Ricky's iPhone Camera
[AVFoundation indev @ 0x1] [1] FaceTime HD Camera
[AVFoundation indev @ 0x1] [2] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
"""


class CameraChoiceTest(unittest.TestCase):
    def test_the_listing_gives_cameras_without_screens_and_microphones(self):
        self.assertEqual(camera.parse_avfoundation_listing(LISTING), [(0, "Ricky's iPhone Camera"), (1, "FaceTime HD Camera")])

    def test_the_laptop_camera_wins_over_a_phone_at_index_zero(self):
        cameras = camera.parse_avfoundation_listing(LISTING)
        self.assertEqual(camera.choose_builtin(cameras), 1)

    def test_the_only_camera_is_taken_and_none_means_zero(self):
        self.assertEqual(camera.choose_builtin([(0, "FaceTime HD Camera")]), 0)
        self.assertEqual(camera.choose_builtin([]), 0)

    def test_a_non_phone_camera_is_the_fallback_when_no_name_says_built_in(self):
        self.assertEqual(camera.choose_builtin([(0, "iPhone Camera"), (1, "Logitech C920")]), 1)

    def test_only_the_phone_listed_still_gives_an_index(self):
        self.assertEqual(camera.choose_builtin([(0, "iPhone Camera")]), 0)

    def test_resolve_source_turns_the_alias_into_an_index_and_leaves_the_rest(self):
        with mock.patch.object(camera, "list_cameras", return_value=[(0, "iPhone Camera"), (1, "FaceTime HD Camera")]):
            self.assertEqual(camera.resolve_source("builtin"), "1")
            self.assertEqual(camera.resolve_source("LAPTOP"), "1")
        self.assertEqual(camera.resolve_source("0"), "0")
        self.assertEqual(camera.resolve_source("/tmp/clip.mp4"), "/tmp/clip.mp4")


if __name__ == "__main__":
    unittest.main()
