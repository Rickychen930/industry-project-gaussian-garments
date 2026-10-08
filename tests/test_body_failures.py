import unittest

import numpy as np

from tools.evaluate.body_failures import mask_box, outside_box, pair_errors, torso_length, union_box


class BodyFailureHelpers(unittest.TestCase):
    def test_mask_box_wraps_the_mask_with_margin(self):
        mask = np.zeros((100, 100), bool)
        mask[20:60, 30:50] = True
        x, y, w, h = mask_box(mask, margin=0.1)
        self.assertLess(x, 30)
        self.assertLess(y, 20)
        self.assertGreater(x + w, 49)
        self.assertGreater(y + h, 59)

    def test_empty_mask_has_no_box(self):
        self.assertIsNone(mask_box(np.zeros((10, 10), bool)))

    def test_union_box_covers_both(self):
        self.assertEqual(union_box([0, 0, 10, 10], [5, 5, 20, 20]), [0, 0, 25, 25])
        self.assertEqual(union_box(None, [1, 2, 3, 4]), [1, 2, 3, 4])

    def test_outside_box_share(self):
        mask = np.zeros((10, 10), bool)
        mask[:, :4] = True
        self.assertAlmostEqual(outside_box(mask, [0, 0, 2, 10]), 0.5)

    def test_pair_errors_skip_invisible_landmarks(self):
        joints = np.zeros((22, 2))
        landmarks = np.zeros((33, 5))
        landmarks[11] = [3, 4, 1.0, 0, 0]
        errors = pair_errors(joints, landmarks, ((16, 11), (17, 12)))
        self.assertAlmostEqual(errors[0], 5.0)
        self.assertTrue(np.isnan(errors[1]))

    def test_torso_length_needs_the_four_corners(self):
        landmarks = np.zeros((33, 5))
        self.assertIsNone(torso_length(landmarks))
        landmarks[[11, 12, 23, 24], 2] = 1.0
        landmarks[[23, 24], 1] = 100.0
        self.assertAlmostEqual(torso_length(landmarks), 100.0)


if __name__ == "__main__":
    unittest.main()
