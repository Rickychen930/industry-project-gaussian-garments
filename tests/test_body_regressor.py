"""`tryon.body.regressor.MultiHMR` - skipped unless its code and weights
are installed (models/pretrained/multi-hmr, not part of the repository)."""

from __future__ import annotations

import unittest

import numpy as np

from tryon.body.regressor import available


@unittest.skipUnless(available(), "Multi-HMR is not installed")
class MultiHMRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tryon.body.regressor import MultiHMR

        cls.model = MultiHMR()

    def _person(self):
        """A synthetic grey body silhouette is not a person to a detector;
        use an annotated 4D-DRESS view when present."""

        import cv2
        import tools.evaluate.real_fit as real

        path = real.DATA / "Take1" / real.CAMERAS[0] / "rgb_images" / "00001.png"
        if not path.exists():
            self.skipTest("no annotated 4D-DRESS photo")
        return cv2.imread(str(path))

    def test_a_person_facing_the_camera(self):
        frame = self._person()
        answer = self.model.predict(frame)
        self.assertIsNotNone(answer)
        self.assertEqual(answer["body_pose"].shape, (63,))
        self.assertEqual(answer["betas"].shape, (10,))
        # camera 0004 faces the wearer: yaw within 30 degrees of 0
        self.assertLess(abs(np.degrees(answer["global_orient"][1])), 30.0)
        # its projected joints land on the person, inside the frame
        joints = answer["joints_2d"]
        self.assertEqual(joints.shape, (22, 2))
        height, width = frame.shape[:2]
        self.assertTrue(((joints >= 0) & (joints <= [width, height])).all())

    def test_an_empty_frame_has_no_one(self):
        self.assertIsNone(self.model.predict(np.full((480, 640, 3), 128, np.uint8)))


class BoxFromLandmarksTest(unittest.TestCase):
    def test_the_box_grows_around_the_visible_landmarks_inside_the_frame(self):
        from tryon.body.regressor import box_from_landmarks

        points = np.zeros((33, 3), np.float32)
        points[:, :2] = [100.0, 100.0]
        points[:4] = [[100, 100, 1], [200, 100, 1], [100, 300, 1], [200, 300, 1]]
        points[4:, 2] = 0.0  # the rest not seen
        x, y, w, h = box_from_landmarks(points, (400, 600), margin=0.25)
        self.assertAlmostEqual(x, 50.0)
        self.assertAlmostEqual(y, 50.0)
        self.assertAlmostEqual(x + w, 250.0)
        self.assertAlmostEqual(y + h, 350.0)
        # clipped to the frame
        x, y, w, h = box_from_landmarks(points, (320, 600), margin=0.25)
        self.assertAlmostEqual(y + h, 319.0)

    def test_too_few_landmarks_is_no_box(self):
        from tryon.body.regressor import box_from_landmarks

        points = np.zeros((33, 3), np.float32)
        points[:3, 2] = 1.0
        self.assertIsNone(box_from_landmarks(points, (400, 600)))
        self.assertIsNone(box_from_landmarks(None, (400, 600)))


class PersonBoxTest(unittest.TestCase):
    def test_the_box_from_joints_grows_past_the_neck_and_ankles(self):
        from tryon.body.regressor import box_from_joints

        joints = np.array([[100.0, 100.0], [200.0, 100.0], [150.0, 300.0], [150.0, 400.0]])
        x, y, w, h = box_from_joints(joints, (600, 800), margin=0.3)
        self.assertAlmostEqual(y, 10.0)  # 300 px tall, grown 90 px each way
        self.assertAlmostEqual(y + h, 490.0)
        self.assertLess(x, 100.0)

    def test_a_person_in_the_box(self):
        from tryon.body.regressor import person_in_box

        mask = np.zeros((100, 100), bool)
        mask[20:80, 40:60] = True  # 1,200 px inside a 40 x 80 box
        self.assertTrue(person_in_box(mask, [30, 10, 40, 80], fraction=0.15))
        self.assertFalse(person_in_box(np.zeros_like(mask), [30, 10, 40, 80], fraction=0.005))
        self.assertFalse(person_in_box(mask, None))
        self.assertFalse(person_in_box(None, [30, 10, 40, 80]))


def _nlf_available():
    from tryon.body.regressor import nlf_available

    return nlf_available()


@unittest.skipUnless(_nlf_available(), "NLF is not installed")
class NLFTest(unittest.TestCase):
    """The NLF worker process, on an annotated view (camera 0004 faces the
    wearer)."""

    def test_a_person_facing_the_camera(self):

        import cv2

        from tryon.body.regressor import NLF
        import tools.evaluate.real_fit as real

        path = real.DATA / "Take1" / real.CAMERAS[0] / "rgb_images" / "00001.png"
        if not path.exists():
            self.skipTest("no annotated 4D-DRESS photo")
        frame = cv2.imread(str(path))
        model = NLF()
        try:
            answer = model.predict(frame)  # its own detector
        finally:
            model.close()
        self.assertIsNotNone(answer)
        self.assertEqual(answer["body_pose"].shape, (63,))
        self.assertLess(abs(np.degrees(answer["global_orient"][1])), 30.0)
        height, width = frame.shape[:2]
        self.assertTrue(((answer["joints_2d"] >= 0) & (answer["joints_2d"] <= [width, height])).all())


class NLFPrefetchTest(unittest.TestCase):
    """The worker answers requests in order; `predict` must take the answer
    prefetched for ITS frame and drop one nobody asked for. No worker: the
    pipe is a list."""

    def _client(self):
        import collections

        from tryon.body.regressor import NLF

        client = NLF.__new__(NLF)
        client.fov, client._ready, client._pending = 60.0, True, collections.deque()
        client.sent, client.queue = [], collections.deque()

        def send(request):
            client.sent.append(request)
            # the answer says which frame it was for, in every joint rotation
            client.queue.append({"rotvec": np.full((55, 3), float(request["frame"][0, 0, 0]) / 100),
                                 "betas": np.zeros(10), "joints3d": np.tile([0.0, 0.0, 3.0], (22, 1))})

        client._send, client._receive = send, client.queue.popleft
        return client

    def _frame(self, value):
        return np.full((40, 30, 3), value, np.uint8)

    BOX = np.array([5, 5, 20, 30], np.float32)

    def test_the_prefetched_answer_goes_to_its_frame(self):
        client, first, second = self._client(), self._frame(10), self._frame(20)
        client.prefetch(first, box=self.BOX)
        client.prefetch(second, box=self.BOX)
        self.assertAlmostEqual(client.predict(first, box=self.BOX)["body_pose"][0], 0.10)
        self.assertAlmostEqual(client.predict(second, box=self.BOX)["body_pose"][0], 0.20)
        self.assertEqual(len(client.sent), 2, "a prefetched frame was sent twice")

    def test_an_unused_prefetch_is_dropped(self):
        client, skipped, wanted = self._client(), self._frame(10), self._frame(30)
        client.prefetch(skipped, box=self.BOX)
        self.assertAlmostEqual(client.predict(wanted, box=self.BOX)["body_pose"][0], 0.30)
        self.assertFalse(client._pending)

    def test_another_box_for_the_same_frame_is_asked_again(self):
        client, frame = self._client(), self._frame(10)
        client.prefetch(frame, box=self.BOX)
        client.predict(frame, box=np.array([0, 0, 30, 40], np.float32))
        self.assertEqual(len(client.sent), 2)
