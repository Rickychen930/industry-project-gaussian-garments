"""Body shape is estimated from many frames, then held.

A single-frame fit chases pixel error with beta, and the same person fitted
from four cameras gets four different shapes. So shape is the running mean
of per-frame estimates until enough frames are in, then frozen.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.test_body_fitter import _landmarks
from tryon.body.fitter import SMPLX_ROOT, BodyFitter


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class ShapeEstimateTest(unittest.TestCase):
    def test_shape_settles_after_the_warmup_window(self):
        fitter = BodyFitter(shape_frames=6)
        fitter.fit(_landmarks())
        for frame in range(6):
            fitter.track(_landmarks(shift_x=frame * 3.0), iterations=3)
        self.assertTrue(fitter.shape_locked)
        held = fitter.betas.detach().clone()
        for frame in range(6, 14):
            fitter.track(_landmarks(shift_x=frame * 3.0), iterations=3)
        self.assertTrue(torch.allclose(fitter.betas.detach(), held), "shape moved after locking")

    def test_shape_is_still_free_during_warmup(self):
        fitter = BodyFitter(shape_frames=10)
        fitter.fit(_landmarks())
        for frame in range(3):
            fitter.track(_landmarks(shift_x=frame * 3.0), iterations=3)
        self.assertFalse(fitter.shape_locked)

    def test_the_held_shape_is_an_average_not_the_last_guess(self):
        """Evidence accumulates; one noisy frame must not set the body."""
        fitter = BodyFitter(shape_frames=8)
        fitter.fit(_landmarks())
        seen = []
        for frame in range(8):
            fitter.track(_landmarks(shift_x=frame * 3.0), iterations=3)
            seen.append(fitter._shape_history[-1].copy())
        held = fitter.betas.detach().numpy().ravel()
        self.assertTrue(np.allclose(held, np.mean(seen, axis=0), atol=1e-5))

    def test_a_new_person_starts_shape_fresh(self):
        fitter = BodyFitter(shape_frames=4)
        fitter.fit(_landmarks())
        for frame in range(4):
            fitter.track(_landmarks(shift_x=frame * 3.0), iterations=3)
        self.assertTrue(fitter.shape_locked)
        fitter.fit(_landmarks(scale=0.8))  # cold fit = new sighting
        self.assertFalse(fitter.shape_locked)

    def test_a_consistent_shape_is_cheap_in_reprojection(self):
        """The measurement behind the design: holding shape costs ~1 px."""
        fitter = BodyFitter(shape_frames=5)
        fitter.fit(_landmarks())
        for frame in range(5):
            fitter.track(_landmarks(shift_x=frame * 2.0), iterations=3)
        target = _landmarks(shift_x=12.0)
        fitter.track(target, iterations=3)
        self.assertLess(fitter.reprojection_error(target), 25.0)


if __name__ == "__main__":
    unittest.main()
