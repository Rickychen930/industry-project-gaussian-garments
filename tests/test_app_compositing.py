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


class RestoreBelowHemTest(unittest.TestCase):
    """The wearer's own shirt stays under a shorter garment's hem."""

    def _scene(self):
        original = np.full((100, 100, 3), 200, np.uint8)
        erased = original.copy()
        erased[60:, :] = 30  # everything under row 60 was erased and filled dark
        frame = erased.copy()
        alpha = np.zeros((100, 100), np.uint8)
        alpha[20:60, 30:70] = 255  # the garment: columns 30-69, hem at row 59
        frame[20:60, 30:70] = (0, 0, 255)
        return frame, original, alpha

    def test_the_erased_pixels_under_the_hem_come_back_in_the_garments_columns(self):
        from tryon.app.compositing import restore_below_hem

        frame, original, alpha = self._scene()
        restore_below_hem(frame, original, alpha)
        self.assertTrue((frame[80, 50] == 200).all())  # under the hem, in the garment's columns
        self.assertTrue((frame[80, 10] == 30).all())  # beside the garment: still the fill
        self.assertTrue((frame[40, 50] == (0, 0, 255)).all())  # the garment itself

    def test_nothing_changes_without_a_garment(self):
        from tryon.app.compositing import restore_below_hem

        frame, original, _ = self._scene()
        before = frame.copy()
        restore_below_hem(frame, original, np.zeros((100, 100), np.uint8))
        self.assertTrue(np.array_equal(frame, before))

    def test_pixels_the_erase_did_not_touch_are_left_alone(self):
        from tryon.app.compositing import restore_below_hem

        frame, original, alpha = self._scene()
        frame[80, 50] = (1, 2, 3)  # a pixel something else painted
        original[80, 50] = (1, 2, 3)  # ... equal to the original: not changed by the erase
        restore_below_hem(frame, original, alpha)
        self.assertTrue((frame[80, 50] == (1, 2, 3)).all())


class RestoreCollarLayerTest(unittest.TestCase):
    """The wearer's own collar stays visible in the neckline."""

    SKIN = (90, 120, 170)
    COLLAR = (20, 20, 20)

    def _scene(self):
        original = np.zeros((200, 200, 3), np.uint8)
        original[:] = self.SKIN
        original[100:115, 70:130] = self.COLLAR  # the real collar, below the chin
        labels = np.zeros((200, 200), np.uint8)
        labels[20:130, 60:140] = 11
        frame = original.copy()
        frame[100:115, 70:130] = (200, 200, 200)  # the erase filled it pale
        alpha = np.zeros((200, 200), np.uint8)
        alpha[120:, 40:160] = 255  # the garment starts below the collar
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        points[[0, 2, 5, 7, 8], :2] = (100, 40)
        points[9, :2] = (95, 60)
        points[10, :2] = (105, 60)  # chin line at 60 + 1.3 * 20 = 86
        points[11, :2], points[12, :2] = (40, 150), (160, 150)
        return frame, original, alpha, labels, points

    def test_the_erased_collar_comes_back_in_the_neck_columns(self):
        from tryon.app.compositing import restore_collar_layer

        frame, original, alpha, labels, points = self._scene()
        restore_collar_layer(frame, original, alpha, labels, points)
        self.assertTrue((frame[108, 100] == self.COLLAR).all())

    def test_pixels_under_the_garment_and_above_the_chin_are_left_alone(self):
        from tryon.app.compositing import restore_collar_layer

        frame, original, alpha, labels, points = self._scene()
        frame[50, 100] = (1, 1, 1)  # above the chin line, the erase changed it (it must not come back)
        frame[130, 100] = (9, 9, 9)
        restore_collar_layer(frame, original, alpha, labels, points)
        self.assertTrue((frame[50, 100] == (1, 1, 1)).all())
        self.assertTrue((frame[130, 100] == (9, 9, 9)).all())

    def test_hair_is_not_restored(self):
        from tryon.app.compositing import restore_collar_layer

        frame, original, alpha, labels, points = self._scene()
        labels[100:115, 70:130] = 2
        restore_collar_layer(frame, original, alpha, labels, points)
        self.assertTrue((frame[108, 100] == (200, 200, 200)).all())

    def test_the_collar_fades_out_at_the_ends_of_its_columns(self):
        from tryon.app.compositing import LAYER_FEATHER, restore_collar_layer

        frame, original, alpha, labels, points = self._scene()
        frame[:] = (200, 200, 200)  # everything changed
        restore_collar_layer(frame, original, alpha, labels, points)
        middle = int(frame[108, 100, 0])
        edge = [int(frame[108, x, 0]) for x in range(40, 60)]
        self.assertTrue(LAYER_FEATHER > 1)
        self.assertLess(min(edge), 200)  # something comes back at the edge
        self.assertTrue(max(edge) >= min(edge))
        self.assertLessEqual(middle, 200)

    def test_a_collapsed_or_implausible_head_geometry_changes_nothing(self):
        from tryon.app.compositing import collar_geometry_ok, restore_collar_layer

        frame, original, alpha, labels, points = self._scene()
        self.assertTrue(collar_geometry_ok(points))
        before = frame.copy()
        collapsed = points.copy()
        collapsed[11, :2], collapsed[12, :2] = (90, 150), (120, 150)  # shoulders 30 px apart
        self.assertFalse(collar_geometry_ok(collapsed))
        restore_collar_layer(frame, original, alpha, labels, collapsed)
        self.assertTrue(np.array_equal(frame, before))
        bowed = points.copy()
        bowed[9, 1] = bowed[10, 1] = 200  # the mouth below the shoulders
        self.assertFalse(collar_geometry_ok(bowed))

    def test_nothing_without_landmarks(self):
        from tryon.app.compositing import restore_collar_layer

        frame, original, alpha, labels, _ = self._scene()
        before = frame.copy()
        restore_collar_layer(frame, original, alpha, labels, None)
        self.assertTrue(np.array_equal(frame, before))


class RestoreInsideGarmentTest(unittest.TestCase):
    """The wearer's own pixels show in the gaps of the garment, not the filled background."""

    def _scene(self):
        original = np.full((120, 120, 3), 200, np.uint8)
        frame = original.copy()
        frame[:] = 30  # the erase filled everything dark
        alpha = np.zeros((120, 120), np.uint8)
        alpha[20:100, 20:100] = 255  # the garment
        alpha[50:70, 50:70] = 0  # a gap inside (lace): 400 of 6,400 px, 6 % of the garment
        return frame, original, alpha

    def test_a_gap_inside_the_outline_gets_the_original_pixels_back(self):
        from tryon.app.compositing import restore_inside_garment

        frame, original, alpha = self._scene()
        restore_inside_garment(frame, original, alpha)
        self.assertTrue((frame[60, 60] == 200).all())

    def test_a_speck_is_painted_over_from_the_garment_not_restored(self):
        from tryon.app.compositing import restore_inside_garment

        frame, original, alpha = self._scene()
        frame[:] = 30
        frame[20:100, 20:100] = 120  # the garment's colour
        alpha[:] = 0
        alpha[20:100, 20:100] = 255
        alpha[58:62, 58:62] = 0  # 16 px, 0.25 % of the garment
        frame[58:62, 58:62] = 5  # what the fill left there (dark)
        restore_inside_garment(frame, original, alpha)
        self.assertTrue((np.abs(frame[59, 59].astype(int) - 120) < 20).all())

    def test_the_outside_is_left_alone(self):
        from tryon.app.compositing import restore_inside_garment

        frame, original, alpha = self._scene()
        restore_inside_garment(frame, original, alpha)
        self.assertTrue((frame[5, 5] == 30).all())
        self.assertTrue((frame[110, 60] == 30).all())

    def test_nothing_changes_without_a_garment(self):
        from tryon.app.compositing import restore_inside_garment

        frame, original, _ = self._scene()
        before = frame.copy()
        restore_inside_garment(frame, original, np.zeros((120, 120), np.uint8))
        self.assertTrue(np.array_equal(frame, before))

    def test_a_loop_bigger_than_the_share_limit_is_not_a_gap(self):
        from tryon.app.compositing import restore_inside_garment

        frame, original, alpha = self._scene()
        alpha[:] = 0
        alpha[20:100, 20:100] = 255
        alpha[35:85, 35:85] = 0  # a hole of 2,500 of 6,400 px (39 %): an arm loop, not lace
        before = frame.copy()
        restore_inside_garment(frame, original, alpha)
        self.assertTrue(np.array_equal(frame, before))


if __name__ == "__main__":
    unittest.main()
