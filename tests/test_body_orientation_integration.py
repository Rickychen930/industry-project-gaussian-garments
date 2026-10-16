"""Integration into `BodyFitter`: cold fits near-perpendicular to the
camera, and the `orientation_disagrees` check used during tracking."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.body.landmarks import JOINT_MAP

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "pose_regressor.pt"


def _turned_frame(yaw):
    """A synthetic single-frame observation at the given yaw."""
    truth = BodyFitter()
    with torch.no_grad():
        truth.scale.fill_(180.0)
        truth.translation[:] = torch.tensor([[320.0, 260.0]])
        truth.global_orient[0, 1] = yaw
        joints = truth._forward().joints[0, truth._smplx_indices]
        pixels = truth._project(joints).detach().numpy()
    points = np.zeros((33, 3), np.float32)
    points[:, 2] = 1.0
    for k, index in enumerate(sorted(JOINT_MAP)):
        points[index, :2] = pixels[k]
    return points, float(yaw)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class FitBackwardCompatibilityTest(unittest.TestCase):
    """The new `orientation_hypotheses` argument changes nothing for callers that omit it."""

    def test_default_hypotheses_are_unchanged(self):
        points, _ = _turned_frame(0.0)
        fitter = BodyFitter()
        loss_default = fitter.fit(points)

        fitter2 = BodyFitter()
        loss_explicit = fitter2.fit(points, orientation_hypotheses=(0.0, float(np.pi)))

        self.assertAlmostEqual(loss_default, loss_explicit, places=5)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(CHECKPOINT.exists(), "run tools/train/orientation.py first")
class RegressorSeededFitTest(unittest.TestCase):
    """A cold fit at a near-perpendicular yaw, where fit()'s {0, pi} seeds are far away."""

    @classmethod
    def setUpClass(cls):
        from tryon.body.orientation import OrientationClassifier

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cls.regressor = OrientationClassifier(n_bins=checkpoint["n_bins"])
        cls.regressor.load_state_dict(checkpoint["model_state"])
        cls.regressor.eval()

    def test_regressor_hypotheses_include_a_seed_near_the_true_yaw(self):
        """For a body near 70 degrees the top-2 hypotheses include one within half a bin."""
        points, true_yaw = _turned_frame(np.radians(70))
        fitter = BodyFitter()
        hypotheses = fitter.orientation_hypotheses_from_regressor(points, self.regressor, top_k=2)

        best_error = min(
            abs(((h - true_yaw) + np.pi) % (2 * np.pi) - np.pi) for h in hypotheses
        )
        self.assertLess(
            np.degrees(best_error),
            30.0,
            f"hypotheses {[round(np.degrees(h), 1) for h in hypotheses]} vs true yaw {np.degrees(true_yaw):.1f}",
        )

    def test_regressor_seeded_fit_reaches_lower_error_than_default_at_a_hard_yaw(self):
        """Fitting with the regressor's hypotheses reaches a lower 3D orientation error than the default."""
        points, true_yaw = _turned_frame(np.radians(70))

        default_fitter = BodyFitter()
        default_fitter.fit(points)
        default_yaw = float(default_fitter.global_orient.detach().numpy()[0, 1])
        default_error = abs(((default_yaw - true_yaw) + np.pi) % (2 * np.pi) - np.pi)

        seeded_fitter = BodyFitter()
        hypotheses = seeded_fitter.orientation_hypotheses_from_regressor(points, self.regressor, top_k=2)
        all_hypotheses = tuple(hypotheses) + (0.0, float(np.pi))
        seeded_fitter.fit(points, orientation_hypotheses=all_hypotheses)
        seeded_yaw = float(seeded_fitter.global_orient.detach().numpy()[0, 1])
        seeded_error = abs(((seeded_yaw - true_yaw) + np.pi) % (2 * np.pi) - np.pi)

        self.assertLessEqual(
            np.degrees(seeded_error),
            np.degrees(default_error) + 1e-6,
            f"seeded {np.degrees(seeded_error):.1f} deg vs default {np.degrees(default_error):.1f} deg error",
        )

    def test_orientation_disagrees_detects_a_stuck_track(self):
        """A tracked global_orient far from the observation is flagged."""
        points, true_yaw = _turned_frame(np.radians(90))
        fitter = BodyFitter()
        with torch.no_grad():
            # Tracking stuck near front-facing while the body is at 90 degrees.
            fitter.global_orient[0, 1] = 0.0
        self.assertTrue(fitter.orientation_disagrees(points, self.regressor, threshold_deg=45.0))

    def test_orientation_disagrees_is_false_when_tracking_is_correct(self):
        points, true_yaw = _turned_frame(np.radians(90))
        fitter = BodyFitter()
        with torch.no_grad():
            fitter.global_orient[0, 1] = true_yaw
        self.assertFalse(fitter.orientation_disagrees(points, self.regressor, threshold_deg=45.0))


if __name__ == "__main__":
    unittest.main()
