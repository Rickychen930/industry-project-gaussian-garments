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


class ShoulderWidthTest(unittest.TestCase):
    """A regressed body whose shoulders are far wider or narrower than MediaPipe's is rescaled."""

    class Fitter:
        """The projection of 22 joints: pixels = scale * (x, y) + translation (the shoulders are joints 16 and 17)."""

        def __init__(self, half_width, scale=100.0, translation=(300.0, 200.0)):
            import torch

            joints = torch.zeros(1, 22, 3)
            joints[0, 16, 0], joints[0, 17, 0] = -half_width, half_width
            self._joints = joints
            self.scale = torch.tensor([scale])
            self.translation = torch.tensor([list(translation)])

        def _forward(self):
            import types

            return types.SimpleNamespace(joints=self._joints)

        def _project(self, joints):
            return joints[:, :2] * self.scale + self.translation

    @staticmethod
    def _points(width, facing=True):
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        points[11, :2], points[12, :2] = (300 - width / 2, 200), (300 + width / 2, 200)
        points[11, 3:5] = (0.0, 0.2) if facing else (0.2, 0.0)  # world z, x of the left shoulder
        points[12, 3:5] = (0.0, -0.2) if facing else (-0.2, 0.0)
        return points

    def _width(self, fitter):
        pixels = fitter._project(fitter._forward().joints[0, 16:18]).numpy()
        return float(np.linalg.norm(pixels[0] - pixels[1]))

    def test_a_far_too_wide_fit_is_brought_to_mediapipes_width_about_the_shoulder_midpoint(self):
        from tryon.app.person import match_shoulder_width

        fitter = self.Fitter(half_width=1.0)  # 200 px against MediaPipe's 80
        ratio = match_shoulder_width(fitter, self._points(80))
        self.assertAlmostEqual(ratio, 2.5, places=3)
        self.assertAlmostEqual(self._width(fitter), 80.0, places=2)
        midpoint = fitter._project(fitter._forward().joints[0, 16:18]).numpy().mean(axis=0)
        self.assertTrue(np.allclose(midpoint, (300.0, 200.0), atol=1e-3))

    def test_a_fit_within_the_band_is_left_alone(self):
        from tryon.app.person import match_shoulder_width

        fitter = self.Fitter(half_width=1.0)  # 200 px against 180: ratio 1.11
        self.assertIsNone(match_shoulder_width(fitter, self._points(180)))
        self.assertAlmostEqual(self._width(fitter), 200.0, places=3)

    def test_a_profile_is_left_alone(self):
        from tryon.app.person import match_shoulder_width

        fitter = self.Fitter(half_width=1.0)
        self.assertIsNone(match_shoulder_width(fitter, self._points(80, facing=False)))
        self.assertAlmostEqual(self._width(fitter), 200.0, places=3)

    def test_a_too_narrow_fit_is_widened(self):
        from tryon.app.person import match_shoulder_width

        fitter = self.Fitter(half_width=0.5)  # 100 px against MediaPipe's 200
        self.assertAlmostEqual(match_shoulder_width(fitter, self._points(200)), 0.5, places=3)
        self.assertAlmostEqual(self._width(fitter), 200.0, places=2)

    def test_hidden_shoulders_change_nothing(self):
        from tryon.app.person import match_shoulder_width

        fitter = self.Fitter(half_width=1.0)
        points = self._points(80)
        points[11, 2] = 0.1
        self.assertIsNone(match_shoulder_width(fitter, points))
        self.assertIsNone(match_shoulder_width(fitter, None))


class RefindLimitTest(unittest.TestCase):
    """MediaPipe blind: the regressor may dress the wearer for REFIND_LIMIT
    frames in a row (an arm across the torso), not for as long as a camera
    stays pointed at part of a body."""

    def _session(self):
        import collections
        from unittest import mock

        from tryon.app.pipeline import LiveSession

        class Fitter:
            def fit_from_regression(self, points, prediction):
                pass

        class Regression:
            def predict(self, frame, box=None, keypoints=None):
                return {"joints_2d": np.zeros((4, 2))}

        session = LiveSession.__new__(LiveSession)
        session.remover, session.regression, session.regressor_name = EarlyMaskTest.Remover(), Regression(), "nlf"
        session.renderer, session.fitter, session.plate = object(), Fitter(), None
        session._early, session._last_job, session.last_parse = collections.OrderedDict(), None, None
        session.last_box, session.held, session.refound = (0, 0, 8, 8), 0, 0
        patches = (mock.patch("tryon.app.pipeline.box_from_joints", return_value=(0, 0, 8, 8)),
                   mock.patch("tryon.app.pipeline.person_in_box", return_value=True))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return session

    def test_the_regressor_alone_dresses_a_blind_frame_only_a_few_times_in_a_row(self):
        from tryon.app.pipeline import REFIND_LIMIT

        session = self._session()
        frame = np.full((16, 16, 3), 1, np.uint8)
        statuses = [session._handle_loss(frame, None, "no person")[0] for _ in range(REFIND_LIMIT + 3)]
        self.assertEqual(statuses[:REFIND_LIMIT], ["re-found"] * REFIND_LIMIT)
        self.assertNotIn("re-found", statuses[REFIND_LIMIT:])

    def test_seeing_the_wearer_again_gives_the_allowance_back(self):
        from tryon.app.pipeline import REFIND_LIMIT

        session = self._session()
        frame = np.full((16, 16, 3), 1, np.uint8)
        for _ in range(REFIND_LIMIT):
            session._handle_loss(frame, None, "no person")
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        session._handle_loss(frame, points, "regressed")
        self.assertEqual(session.refound, 0)
        self.assertEqual(session._handle_loss(frame, None, "no person")[0], "re-found")


class EarlyMaskTest(unittest.TestCase):
    """The clothing parse of frame t+1 starts in `prefetch`, while frame t is
    still being erased and shown; each frame must get its own parse."""

    SIZE = 16

    class Remover:
        """Its parse records which frame it was given and carries that frame's tag."""

        ARM_TUBE_WIDTH = 0.22

        def __init__(self):
            self.parsed, self.removed, self.running, self.overlap = [], [], 0, []

        def parse_clothing(self, frame, keypoints=None):
            import time

            from tryon.removal.remover import ClothingParse

            self.running += 1
            self.overlap.append(self.running)
            time.sleep(0.01)
            tag = int(frame[0, 0, 0])
            self.parsed.append(tag)
            self.running -= 1
            labels = np.full(frame.shape[:2], tag, np.uint8)
            return ClothingParse(labels == tag, labels, labels == tag)

        def remove(self, frame, mask, arms=None, keep=None, plate=None, body=None, parse=None):
            self.removed.append((int(frame[0, 0, 0]), int(parse.labels[0, 0])))
            return frame.copy()

    class Regression:
        def __init__(self):
            self.prefetched = []

        def prefetch(self, frame, keypoints=None):
            self.prefetched.append(int(frame[0, 0, 0]))

    class Plate:
        def __init__(self):
            self.updates = []

        def update(self, frame, person):
            self.updates.append((int(frame[0, 0, 0]), bool(person.all())))

    def _session(self):
        from tryon.app.pipeline import LiveSession

        session = LiveSession.__new__(LiveSession)
        session.remover, session.regression, session.regressor_name = self.Remover(), self.Regression(), "nlf"
        session.renderer, session.fitter, session.plate = object(), None, None
        session._early, session._last_job, session.last_parse = __import__("collections").OrderedDict(), None, None
        return session

    def _frame(self, tag):
        return np.full((self.SIZE, self.SIZE, 3), tag, np.uint8)

    @staticmethod
    def _points():
        points = np.zeros((33, 4), np.float32)
        points[:, 2] = 1.0
        return points

    def test_each_frame_takes_the_parse_of_exactly_that_frame(self):
        session = self._session()
        f1, f2, f3 = self._frame(1), self._frame(2), self._frame(3)
        for frame in (f1, f2):  # the loop prefetches t+1 before it processes t
            session.prefetch(frame, self._points())
        job1 = session._take_early_mask(f1)
        self.assertIs(job1.frame, f1)
        session.prefetch(f3, self._points())
        job2 = session._take_early_mask(f2)
        job3 = session._take_early_mask(f3)
        self.assertEqual(int(job1.result().labels[0, 0]), 1)
        self.assertEqual(int(job2.result().labels[0, 0]), 2)
        self.assertEqual(int(job3.result().labels[0, 0]), 3)
        self.assertEqual(session.remover.parsed, [1, 2, 3], "one parse per frame, in frame order")
        self.assertEqual(max(session.remover.overlap), 1, "never two parses at once")
        self.assertEqual(session.regression.prefetched, [1, 2, 3], "the regressor is still prefetched")

    def test_the_parse_reaches_remove_and_the_plate_from_the_job_not_from_shared_state(self):
        import tryon.app.pipeline as pipeline

        session = self._session()
        session.plate = self.Plate()
        f1, f2 = self._frame(1), self._frame(2)
        session.prefetch(f1, self._points())
        session.prefetch(f2, self._points())  # frame 2's parse is in flight while frame 1 is erased
        old = pipeline.ERASE_NEAR_BODY
        pipeline.ERASE_NEAR_BODY = 0
        try:
            parse1, erased1, original1 = session._start_erase(f1, None, session._take_early_mask(f1)).result()
            parse2, erased2, original2 = session._start_erase(f2, None, session._take_early_mask(f2)).result()
        finally:
            pipeline.ERASE_NEAR_BODY = old
        self.assertEqual(session.remover.removed, [(1, 1), (2, 2)], "remove saw its own frame's labels")
        self.assertEqual(session.plate.updates, [(1, True), (2, True)])  # labels != 0 everywhere: all person
        self.assertEqual((int(parse1.labels[0, 0]), int(parse2.labels[0, 0])), (1, 2))
        self.assertEqual((int(original1[0, 0, 0]), int(original2[0, 0, 0])), (1, 2), "the frame as read comes back for the face restore")
        self.assertEqual((int(erased1[0, 0, 0]), int(erased2[0, 0, 0])), (1, 2))

    def test_the_erase_runs_beside_the_caller_and_reraises_its_errors(self):
        import threading

        from tryon.app.pipeline import Call

        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            release.wait(2)
            return 7

        call = Call(work)
        call.start()
        self.assertTrue(started.wait(2), "the function runs while the caller carries on")
        release.set()
        self.assertEqual(call.result(), 7)

        def broken():
            raise ValueError("boom")

        failing = Call(broken)
        failing.start()
        with self.assertRaises(ValueError):
            failing.result()

    def test_a_frame_that_is_not_dressable_starts_no_parse(self):
        session = self._session()
        frame = self._frame(7)
        session.prefetch(frame, None)  # no landmarks: no person
        self.assertIsNone(session._take_early_mask(frame))
        self.assertEqual(session.remover.parsed, [])

    def test_a_frame_without_a_prefetch_has_no_early_job(self):
        session = self._session()
        session.prefetch(self._frame(1), self._points())
        self.assertIsNone(session._take_early_mask(self._frame(1)), "a different frame object is not the prefetched one")

    def test_frames_prefetched_but_never_processed_are_dropped_and_finish_waits(self):
        session = self._session()
        frames = [self._frame(i) for i in (1, 2, 3)]
        for frame in frames:
            session.prefetch(frame, self._points())
        self.assertIsNotNone(session._take_early_mask(frames[2]))  # 1 and 2 were skipped (a cut, a lost person)
        self.assertEqual(len(session._early), 0)
        session.finish()  # the loop ended
        self.assertEqual(session.remover.parsed, [1, 2, 3])
        self.assertEqual(session.remover.running, 0, "no parse still running after finish")

    def test_a_late_start_in_dress_still_waits_for_the_early_jobs(self):
        session = self._session()
        f1, f2 = self._frame(1), self._frame(2)
        session.prefetch(f1, self._points())
        late = session._start_mask(f2, self._points())  # what `_dress` does when `prefetch` did not
        self.assertEqual(int(late.result().labels[0, 0]), 2)
        self.assertEqual(session.remover.parsed, [1, 2])
