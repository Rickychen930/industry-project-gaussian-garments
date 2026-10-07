"""tools/data/refine_scan_pose.py: the scan body fitted to the checkpoint template."""

import unittest
from pathlib import Path

import numpy as np
from tryon.paths import garment

ROOT = Path(__file__).resolve().parent.parent
GARMENT = garment("00152_outer")


@unittest.skipUnless((GARMENT / "stage1" / "scan_pose.npz").exists(), "needs the 00152_outer checkpoint and its scan pose")
class RefineScanPoseTest(unittest.TestCase):
    def test_a_matching_template_is_left_where_the_trajectory_put_it(self):
        from tools.data.refine_scan_pose import refine

        result, before, after = refine(GARMENT, iterations=120, coarse=40, verbose=False)
        for key in ("betas", "body_pose", "global_orient", "offset", "refined"):
            self.assertIn(key, result)
        self.assertLess(abs(np.median(after) - np.median(before)), 0.005, "a correct pose must not wander")
        self.assertLessEqual(np.quantile(after, 0.9), np.quantile(before, 0.9) + 0.002)
