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


if __name__ == "__main__":
    unittest.main()
