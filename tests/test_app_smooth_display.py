"""smooth_display._compose_newest works inside the layer's box: same pixels as the whole frame."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from tryon.app import smooth_display
from tryon.app.layer_warp import torso_transform


def _reference(frame, result, points, layer):
    """The full-frame composite, as it was before the bounding box."""
    from tryon.removal.remover import hand_discs

    height, width = frame.shape[:2]
    image, changed, garment, layer_points, hud = layer[:5]
    m = torso_transform(layer_points, points)
    if m is not None:
        image = cv2.warpAffine(image, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
        changed = cv2.warpAffine(changed, m, (width, height))
        garment = cv2.warpAffine(garment, m, (width, height))
    person = smooth_display._person_mask(result)
    if person is not None:
        changed = changed * person
    alpha = np.maximum(garment, changed)
    if points is not None:
        keep = hand_discs(points, frame.shape).astype(np.uint8)
        nose, left, right = points[0], points[11], points[12]
        if min(nose[2], left[2], right[2]) > 0.5:
            radius = int(0.45 * np.hypot(*(left[:2] - right[:2])))
            cv2.circle(keep, (int(nose[0]), int(nose[1])), max(radius, 1), 1, -1)
        alpha = alpha * (1.0 - cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0))
    out = (frame * (1.0 - alpha[..., None]) + image * alpha[..., None]).astype(np.uint8)
    return out, garment * (alpha > 0), hud


def _points(shift=(0.0, 0.0), scale=1.0, angle=0.0):
    points = np.zeros((33, 5), np.float32)
    points[:, 2] = 1.0
    base = {0: (250, 60), 11: (300, 150), 12: (200, 150), 13: (330, 230), 14: (170, 230), 15: (350, 300),
            16: (150, 300), 23: (285, 330), 24: (215, 330)}
    centre = np.array([250.0, 240.0])
    rotation = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for index, xy in base.items():
        points[index, :2] = (np.array(xy) - centre) @ rotation.T + centre + shift
    return points


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    height, width = 400, 500
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    body = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(body, np.array([[200, 150], [300, 150], [285, 330], [215, 330]], np.int32), 1)
    garment = cv2.GaussianBlur(body.astype(np.float32), (5, 5), 0)
    changed = cv2.GaussianBlur(cv2.dilate(body, np.ones((15, 15), np.uint8)).astype(np.float32), (5, 5), 0)
    layer = (image, changed, garment, _points(), "hud", None, 0)
    mask = np.zeros((height, width), np.float32)
    mask[40:360, 150:350] = rng.uniform(0.3, 1.0, (320, 200))
    return frame, layer, SimpleNamespace(segmentation_mask=mask)


class ComposeNewestTest(unittest.TestCase):
    def test_every_torso_move_gives_the_full_frame_pixels(self):
        frame, layer, result = _scene()
        moves = {
            "still": _points(),
            "shift": _points(shift=(40, -25)),
            "shift to the edge": _points(shift=(-230, 120)),
            "off the frame": _points(shift=(900, 0)),
            "rotated": _points(angle=0.3),
            "scaled up": _points(scale=1.6),
            "scaled down": _points(scale=0.5),
            "all three": _points(shift=(15, 10), scale=1.1, angle=-0.12),
            "no torso": None,
        }
        for name, points in moves.items():
            for seg in (result, SimpleNamespace()):
                with self.subTest(move=name, segmentation=hasattr(seg, "segmentation_mask")):
                    got = smooth_display._compose_newest(frame, seg, points, layer)
                    want = _reference(frame, seg, points, layer)
                    self.assertTrue(np.array_equal(got[0], want[0]))
                    self.assertTrue(np.array_equal(got[1], want[1]))
                    self.assertEqual(got[2], want[2])

    def test_a_new_layer_is_not_confused_with_the_cached_box(self):
        frame, layer, result = _scene()
        smooth_display._compose_newest(frame, result, _points(), layer)
        moved = (layer[0], np.roll(layer[1], 120, axis=1), np.roll(layer[2], 120, axis=1)) + layer[3:]
        got = smooth_display._compose_newest(frame, result, _points(), moved)
        want = _reference(frame, result, _points(), moved)
        self.assertTrue(np.array_equal(got[0], want[0]))

    def test_an_empty_layer_returns_the_frame(self):
        frame, layer, result = _scene()
        empty = (layer[0], np.zeros_like(layer[1]), np.zeros_like(layer[2])) + layer[3:]
        out, drawn, _ = smooth_display._compose_newest(frame, result, _points(), empty)
        self.assertTrue(np.array_equal(out, frame))
        self.assertFalse(drawn.any())

    def test_the_person_mask_window_equals_the_full_mask_cropped(self):
        _, _, result = _scene()
        full = smooth_display._person_mask(result)
        for box in ((0, 0, 500, 400), (100, 50, 260, 200), (480, 380, 500, 400), (3, 3, 20, 20)):
            x0, y0, x1, y1 = box
            self.assertTrue(np.array_equal(smooth_display._person_mask(result, box), full[y0:y1, x0:x1]))


if __name__ == "__main__":
    unittest.main()
