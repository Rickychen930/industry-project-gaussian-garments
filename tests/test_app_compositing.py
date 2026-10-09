"""keep_in_front: the face / hair / hands restore, same pixels as the whole-frame formulation."""

import unittest

import numpy as np

from tryon.app.compositing import IN_FRONT_CLASSES, keep_in_front
from tryon.removal.remover import arm_strokes, hand_discs


def _reference(frame, original, labels, points=None):
    if labels is None or labels.shape != frame.shape[:2]:
        return
    front = np.isin(labels, IN_FRONT_CLASSES)
    if points is not None:
        head = [i for i in (0, 2, 5, 7, 8) if points[i, 2] >= 0.3]
        if head and points[11, 2] >= 0.3 and points[12, 2] >= 0.3:
            radius = 0.6 * float(np.linalg.norm(points[11, :2] - points[12, :2]))
            centre = points[head, :2].mean(axis=0)
            rows, cols = np.ogrid[: frame.shape[0], : frame.shape[1]]
            near = (rows - centre[1]) ** 2 + (cols - centre[0]) ** 2 <= radius**2
            front &= near | (labels != 11)
        front &= ~arm_strokes(points, frame.shape)
        front |= hand_discs(points, frame.shape, hand_fraction=0.25) & np.isin(labels, (14, 15))
    frame[front] = original[front]


def _points(head=(250, 60)):
    points = np.zeros((33, 5), np.float32)
    points[:, 2] = 1.0
    for index, xy in {0: head, 2: (head[0] - 8, head[1] - 5), 5: (head[0] + 8, head[1] - 5), 11: (300, 150), 12: (200, 150),
                      13: (330, 230), 14: (170, 230), 15: (350, 300), 16: (150, 300)}.items():
        points[index, :2] = xy
    return points


class KeepInFrontTest(unittest.TestCase):
    def _case(self, labels_dtype=np.uint8, points=None, seed=0):
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 255, (400, 500, 3), dtype=np.uint8)
        original = rng.integers(0, 255, (400, 500, 3), dtype=np.uint8)
        labels = rng.choice([0, 0, 0, 1, 2, 3, 5, 11, 14, 15], (400, 500)).astype(labels_dtype)
        return frame, original, labels, points

    def _same(self, **kwargs):
        frame, original, labels, points = self._case(**kwargs)
        want = frame.copy()
        _reference(want, original, labels, points)
        keep_in_front(frame, original, labels, points)
        self.assertTrue(np.array_equal(frame, want))

    def test_with_landmarks(self):
        self._same(points=_points())

    def test_head_disc_at_the_frame_edges(self):
        for head in ((5, 5), (495, 395), (250, -30), (700, 60)):
            with self.subTest(head=head):
                self._same(points=_points(head))

    def test_without_landmarks(self):
        self._same()

    def test_other_label_dtypes(self):
        self._same(labels_dtype=np.int64, points=_points())

    def test_nothing_to_restore_changes_nothing(self):
        frame, original, labels, _ = self._case()
        labels[:] = 0
        before = frame.copy()
        keep_in_front(frame, original, labels, _points())
        self.assertTrue(np.array_equal(frame, before))


class RepairCollarTest(unittest.TestCase):
    """The collar the parser called neck is blended into the neck."""

    SKIN = (90, 120, 170)  # BGR
    COLLAR = (20, 20, 20)

    def _scene(self):
        frame = np.zeros((200, 200, 3), np.uint8)
        frame[:] = self.SKIN
        labels = np.zeros((200, 200), np.uint8)
        labels[20:130, 60:140] = 11  # face and neck
        frame[110:116, 70:130] = self.COLLAR  # a dark stripe under the chin
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        points[0, :2] = (100, 50)  # nose
        points[9, :2] = (95, 80)  # mouth (each test sets its row, which sets the chin line)
        points[10, :2] = (105, 80)
        return frame, labels, points

    def test_the_stripe_below_the_chin_is_blended_toward_the_skin(self):
        from tryon.app.compositing import repair_collar

        frame, labels, points = self._scene()
        points[9, 1] = points[10, 1] = 70  # chin line at 70 + 1.3 * 20 = 96: the stripe at 110 is below it
        repair_collar(frame, np.zeros((200, 200), np.uint8), labels, points)
        self.assertLess(abs(int(frame[112, 100, 0]) - self.SKIN[0]), 25)

    def test_a_stripe_above_the_chin_line_is_left_alone(self):
        from tryon.app.compositing import repair_collar

        frame, labels, points = self._scene()
        points[9, 1] = points[10, 1] = 100  # chin line at 100 + 1.3 * 50 = 165: the stripe is on the face
        before = frame.copy()
        repair_collar(frame, np.zeros((200, 200), np.uint8), labels, points)
        self.assertTrue(np.array_equal(frame, before))

    def test_pixels_under_the_garment_are_not_touched(self):
        from tryon.app.compositing import repair_collar

        frame, labels, points = self._scene()
        points[9, 1] = points[10, 1] = 70
        alpha = np.full((200, 200), 255, np.uint8)
        before = frame.copy()
        repair_collar(frame, alpha, labels, points)
        self.assertTrue(np.array_equal(frame, before))

    def test_no_landmarks_changes_nothing(self):
        from tryon.app.compositing import repair_collar

        frame, labels, _ = self._scene()
        before = frame.copy()
        repair_collar(frame, np.zeros((200, 200), np.uint8), labels, None)
        self.assertTrue(np.array_equal(frame, before))


class CollarPixelsTest(unittest.TestCase):
    """`keep_in_front` leaves the collar the parser called face on the garment."""

    SKIN = (90, 120, 170)  # BGR
    STRIPE = (230, 230, 230)

    def _scene(self, mouth_row):
        original = np.zeros((200, 200, 3), np.uint8)
        original[:] = self.SKIN
        labels = np.zeros((200, 200), np.uint8)
        labels[20:130, 60:140] = 11  # face and neck
        original[110:116, 70:130] = self.STRIPE  # a pale stripe under the chin
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        points[[0, 2, 5, 7, 8], :2] = (100, 40)  # the head landmarks
        points[13:17, 2] = 0.0  # no arms in this scene
        points[9, :2] = (95, mouth_row)
        points[10, :2] = (105, mouth_row)
        return original, labels, points

    def test_a_stripe_below_the_chin_is_found_and_the_skin_is_not(self):
        from tryon.app.compositing import collar_pixels

        original, labels, points = self._scene(mouth_row=60)  # chin at 60 + 1.3 * 20 = 86
        collar = collar_pixels(original, labels, points)
        self.assertTrue(collar[112, 100])
        self.assertFalse(collar[100, 100])  # skin below the chin
        self.assertFalse(collar[50, 100])  # the face

    def test_nothing_above_the_chin_line_is_found(self):
        from tryon.app.compositing import collar_pixels

        original, labels, points = self._scene(mouth_row=100)  # chin at 100 + 1.3 * 60 = 178
        self.assertIsNone(collar_pixels(original, labels, points))

    def test_keep_in_front_does_not_paste_the_stripe_over_the_garment(self):
        original, labels, points = self._scene(mouth_row=60)
        frame = np.full_like(original, 7)
        points[11, :2], points[12, :2] = (20, 150), (180, 150)  # wide shoulders: the head disc (radius 96) reaches the stripe
        keep_in_front(frame, original, labels, points)
        self.assertTrue((frame[112, 100] == 7).all())  # the garment stays
        self.assertTrue((frame[50, 100] == original[50, 100]).all())  # the face comes back


if __name__ == "__main__":
    unittest.main()
