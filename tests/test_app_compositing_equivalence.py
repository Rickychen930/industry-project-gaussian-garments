"""repair_collar and restore_below_hem work in a small window; their output is the whole-frame version's, bit for bit.

The whole-frame versions (the first implementations, 9-10 Oct 2026) are kept here as the oracle.
"""

import unittest

import cv2
import numpy as np

from tryon.app.compositing import COLLAR_CHIN, COLLAR_DISTANCE, COLLAR_GROW, COLLAR_RADIUS, composite_garment, repair_collar, restore_below_hem


def _reference_restore_below_hem(frame, original, garment_alpha, margin=0):
    """Put the wearer's own pixels back below the garment's lower edge, in place.
    A wearer whose shirt is longer than the garment (a T-shirt under a cropped
    sweater) had everything under the hem erased and filled from the room's
    learned background, which came out as grey, green and purple blocks (a
    recorded session, arms raised, 9 Oct); the real shirt continuing under the
    hem reads as a layer. Only the columns the garment covers, only below its
    lowest covered row in that column, only the pixels the erase changed."""
    covered = garment_alpha >= 25
    if not covered.any():
        return
    rows = covered.shape[0]
    columns = covered.any(axis=0)
    bottom = np.where(columns, rows - 1 - np.argmax(covered[::-1], axis=0), rows)  # last covered row per column
    below = np.arange(rows)[:, None] > (bottom[None, :] + margin)
    changed = (frame != original).any(axis=2)
    restore = below & changed & columns[None, :] & ~covered
    if restore.any():
        frame[restore] = original[restore]


def _reference_repair_collar(frame, garment_alpha, labels, points):
    """Blend the wearer's own collar into the neck, in place, after the
    garment is composited. SCHP labels the dark stripe or band of the shirt
    under the chin "face", so it is not erased, and `keep_in_front` restores
    it; it then showed above the new neckline (a recorded session, 9 Oct).
    The face-labelled pixels below the chin that no garment covers and whose
    Lab colour is far from the face's skin are inpainted from the skin above
    and the garment below (filling them with a flat colour, or erasing them
    with the general fill, left blocks that read worse than the line)."""
    if labels is None or points is None or labels.shape != frame.shape[:2]:
        return
    nose, mouth = points[0], 0.5 * (points[9] + points[10])
    if min(points[0, 2], points[9, 2], points[10, 2]) < 0.3 or not np.isfinite(mouth[:2]).all():
        return
    gap = float(mouth[1] - nose[1])
    chin, mouth_row, nose_row = int(mouth[1] + COLLAR_CHIN * gap), int(mouth[1]), int(nose[1])
    if gap < 2 or chin >= frame.shape[0] - 1 or mouth_row <= nose_row:
        return
    below = labels[chin + 1 :] == 11
    rows = np.flatnonzero(below.any(axis=1))
    if rows.size == 0:
        return
    cols = np.flatnonzero(below.any(axis=0))
    grow = COLLAR_GROW + COLLAR_RADIUS
    r0, r1 = chin + 1 + int(rows[0]), chin + 1 + int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    window = (slice(max(r0 - grow, 0), min(r1 + grow, frame.shape[0])), slice(max(c0 - grow, 0), min(c1 + grow, frame.shape[1])))
    face_rows = labels[nose_row:mouth_row] == 11
    if face_rows.sum() < 50:
        return
    skin_bgr = frame[nose_row:mouth_row][face_rows].reshape(-1, 1, 3)
    skin = np.median(cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)
    neck = np.zeros(frame.shape[:2], bool)
    neck[chin + 1 :] = below
    lab = cv2.cvtColor(frame[window], cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = np.array([0.5, 1.0, 1.0], np.float32)  # shade on the neck is lightness: weigh it less
    far = np.sqrt((((lab - skin) * weights) ** 2).sum(axis=2)) > COLLAR_DISTANCE
    marked = (neck[window] & far & (garment_alpha[window] < 25)).astype(np.uint8)
    if not marked.any():
        return
    marked = cv2.dilate(marked, np.ones((2 * COLLAR_GROW + 1,) * 2, np.uint8)) & (garment_alpha[window] < 25) & (labels[window] == 11)
    frame[window] = cv2.inpaint(np.ascontiguousarray(frame[window]), marked * 255, COLLAR_RADIUS, cv2.INPAINT_TELEA)


def _scene(seed):
    rng = np.random.default_rng(seed)
    height, width = 180, 240
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3)
    labels = np.zeros((height, width), np.uint8)
    cx = int(rng.integers(90, 150))
    labels[10 : int(rng.integers(100, 130)), cx - 30 : cx + 30] = 11  # face and neck
    labels[int(rng.integers(60, 90)) : 130, cx - 12 : cx + 12] = 11
    alpha = np.zeros((height, width), np.uint8)
    top = int(rng.integers(95, 125))
    alpha[top : int(rng.integers(140, 175)), int(rng.integers(20, 70)) : int(rng.integers(170, 220))] = 255
    alpha[top : top + 12, cx - 15 : cx + 15] = 0  # a neckline
    alpha[int(rng.integers(150, 170)) : , int(rng.integers(60, 100)) : int(rng.integers(110, 140))] = 0  # a notch in the hem
    points = np.zeros((33, 5), np.float32)
    points[:, 2] = 1.0
    points[0, :2] = (cx, 30)
    points[9, :2] = (cx - 5, 55)
    points[10, :2] = (cx + 5, 55)
    return frame, labels, alpha, points


def _reference_composite_garment(frame, bgra):
    """The numpy form before the numba kernel (10 Oct 2026)."""
    x, y, w, h = cv2.boundingRect(bgra[..., 3])
    if w == 0 or h == 0:
        return
    region = (slice(y, y + h), slice(x, x + w))
    alpha = bgra[region + (slice(3, 4),)].astype(np.float32) / 255.0
    garment_bgr = bgra[region + (slice(0, 3),)].astype(np.float32)
    frame[region] = (garment_bgr * alpha + frame[region].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


class CompositeEquivalenceTest(unittest.TestCase):
    def test_same_bytes_as_the_numpy_form(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
            bgra = rng.integers(0, 256, (120, 160, 4), dtype=np.uint8)
            alpha = bgra[..., 3]
            alpha[rng.random(alpha.shape) < 0.4] = 0  # transparent pixels
            alpha[rng.random(alpha.shape) < 0.2] = 255  # opaque
            alpha[:10] = 0  # a box that does not start at the origin
            a, b = frame.copy(), frame.copy()
            _reference_composite_garment(a, bgra)
            composite_garment(b, bgra)
            self.assertTrue(np.array_equal(a, b), f"seed {seed}")
            self.assertFalse(np.array_equal(a, frame))

    def test_a_fully_transparent_garment_changes_nothing(self):
        frame = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        before = frame.copy()
        composite_garment(frame, np.zeros((40, 40, 4), np.uint8))
        self.assertTrue(np.array_equal(frame, before))


class RepairCollarEquivalenceTest(unittest.TestCase):
    def test_same_pixels_as_the_whole_frame_version(self):
        changed = 0
        for seed in range(25):
            frame, labels, alpha, points = _scene(seed)
            a, b = frame.copy(), frame.copy()
            _reference_repair_collar(a, alpha, labels, points)
            repair_collar(b, alpha, labels, points)
            self.assertTrue(np.array_equal(a, b), f"seed {seed}")
            changed += int(not np.array_equal(a, frame))
        self.assertGreater(changed, 5)  # the scenes do exercise the repair


class RestoreBelowHemEquivalenceTest(unittest.TestCase):
    def test_same_pixels_as_the_whole_frame_version(self):
        changed = 0
        for seed in range(25):
            frame, labels, alpha, points = _scene(seed)
            original = np.random.default_rng(seed + 100).integers(0, 255, frame.shape, dtype=np.uint8)
            erased = np.where((np.random.default_rng(seed + 200).random(frame.shape[:2]) > 0.3)[..., None], frame, original)
            for margin in (0, 3):
                a, b = erased.copy(), erased.copy()
                _reference_restore_below_hem(a, original, alpha, margin)
                restore_below_hem(b, original, alpha, margin)
                self.assertTrue(np.array_equal(a, b), f"seed {seed} margin {margin}")
                changed += int(not np.array_equal(a, erased))
        self.assertGreater(changed, 10)


if __name__ == "__main__":
    unittest.main()
