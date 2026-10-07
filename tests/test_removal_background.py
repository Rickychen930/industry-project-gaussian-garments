"""BackgroundPlate: the room behind a removed garment, learned from the feed."""

import unittest

import numpy as np

from tryon.removal.background import BackgroundPlate


def _scene(person_x):
    """A striped room with a grey 'person' block at column `person_x`."""
    room = np.zeros((120, 160, 3), np.uint8)
    room[:, :, 1] = (np.arange(160) * 3 % 200 + 40).astype(np.uint8)[None, :]
    frame = room.copy()
    person = np.zeros((120, 160), bool)
    person[20:110, person_x : person_x + 30] = True
    frame[person] = 128
    return room, frame, person


class BackgroundPlateTest(unittest.TestCase):
    def test_a_pixel_uncovered_earlier_is_filled_with_the_room(self):
        plate = BackgroundPlate(margin=2)
        for x in (10, 20, 30, 40, 50, 60):  # the person walks right, uncovering the left
            room, frame, person = _scene(x)
            plate.update(frame, person)
        room, frame, person = _scene(100)
        hole = np.zeros_like(person)
        hole[40:80, 15:25] = True  # where the person used to stand
        filled, rest = plate.fill(frame, hole)
        self.assertFalse(rest[40:80, 15:25].any())
        np.testing.assert_allclose(filled[hole].astype(float), room[hole].astype(float), atol=2)

    def test_the_body_and_unseen_pixels_are_left_to_the_inpainter(self):
        plate = BackgroundPlate(margin=2)
        for _ in range(4):
            room, frame, person = _scene(60)  # never moves: its pixels are never seen
            plate.update(frame, person)
        hole = np.zeros_like(person)
        hole[40:80, 50:100] = True
        body = np.zeros_like(person)
        body[:, 90:100] = True
        filled, rest = plate.fill(frame, hole, body=body)
        self.assertTrue(rest[40:80, 62:88].all(), "pixels under the person were never seen")
        self.assertTrue(rest[40:80, 90:100].all(), "the body is never filled with the room")
        self.assertFalse(rest[40:80, 50:55].any(), "seen room pixels are filled")

    def test_a_scene_cut_clears_the_plate(self):
        plate = BackgroundPlate(margin=2)
        for x in (10, 20, 30, 40):
            _room, frame, person = _scene(x)
            plate.update(frame, person)
        other = np.full((120, 160, 3), 250, np.uint8)
        plate.update(other, np.zeros((120, 160), bool))
        self.assertFalse(plate.known().any(), "a new scene must be relearned")


class OneEuroFilterTest(unittest.TestCase):
    def _points(self, y_shift, rng):
        points = np.zeros((33, 4), np.float32)
        points[:, 0] = np.linspace(100, 200, 33)
        points[:, 1] = np.linspace(50, 450, 33) + y_shift
        points[:, :2] += rng.normal(0, 2.0, (33, 2))
        points[:, 2] = 1.0
        return points

    def test_still_jitter_is_reduced_and_visibility_kept(self):
        from tryon.body.smoothing import OneEuroFilter

        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        raw, out = [], []
        for t in range(60):
            p = self._points(0.0, rng)
            raw.append(p[:, :2])
            q = f(p, t / 25.0)
            out.append(q[:, :2])
            self.assertTrue(np.array_equal(q[:, 2], p[:, 2]))
        def jitter(s):
            return np.abs(np.diff(np.stack(s[10:]), axis=0)).mean()
        self.assertLess(jitter(out), 0.5 * jitter(raw))

    def test_a_fast_move_is_followed(self):
        from tryon.body.smoothing import OneEuroFilter

        rng = np.random.default_rng(1)
        f = OneEuroFilter(min_cutoff=1.0, beta=5.0)
        for t in range(30):
            q = f(self._points(t * 10.0, rng), t / 25.0)
        lag = abs(float(q[:, 1].mean()) - float(self._points(290.0, np.random.default_rng(9))[:, 1].mean()))
        self.assertLess(lag, 40.0, "the filtered body trails a fast move by more than 10% of its height")

    def test_relative_mode_never_delays_the_position(self):
        from tryon.body.smoothing import OneEuroFilter

        rng = np.random.default_rng(2)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0, relative=True)
        for t in range(20):
            p = self._points(t * 15.0, rng)  # the whole body sliding down fast
            q = f(p, t / 25.0)
        hips_in, hips_out = p[[23, 24], :2].mean(axis=0), q[[23, 24], :2].mean(axis=0)
        np.testing.assert_allclose(hips_out, hips_in, atol=1e-3)


class GrowTest(unittest.TestCase):
    """The silhouette's dilation runs in the mask's box; it must equal the whole frame's."""

    def test_roi_dilation_equals_the_full_frame_dilation(self):
        import cv2

        from tryon.removal.background import body_silhouette

        rng = np.random.default_rng(0)
        for name, mask in (
            ("blob", cv2.circle(np.zeros((200, 300), np.uint8), (150, 90), 40, 1, -1)),
            ("corner", cv2.circle(np.zeros((200, 300), np.uint8), (5, 195), 30, 1, -1)),
            ("noise", (rng.random((200, 300)) > 0.999).astype(np.uint8)),
            ("empty", np.zeros((200, 300), np.uint8)),
            ("full", np.ones((200, 300), np.uint8)),
        ):
            for grow in (0.0, 0.02, 0.08, 0.5):
                with self.subTest(mask=name, grow=grow):
                    radius = max(int(round(grow * 200)), 1)
                    want = cv2.dilate(mask, np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool)
                    self.assertTrue(np.array_equal(body_silhouette(None, mask.shape, grow=grow, fill=mask), want))

    def test_a_scene_cut_still_restarts_the_plate(self):
        plate = BackgroundPlate(margin=2)
        room, frame, person = _scene(20)
        for _ in range(5):
            plate.update(frame, person)
        flipped = 255 - frame
        plate.update(flipped, person)
        self.assertEqual(int(plate.count.max()), 1)
        np.testing.assert_array_equal(plate.plate, flipped.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
