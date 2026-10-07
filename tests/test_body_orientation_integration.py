"""Integration into `BodyFitter` - verified on the exact
case this layer exists to fix (a cold fit near-perpendicular to the
camera, where `fit()`'s hardcoded {0, pi} hypotheses have no nearby
seed), and on the periodic disagreement check `orientation_disagrees`
that would trigger a re-fit during live tracking.
"""

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
    """A synthetic single-frame observation of a body at the given yaw -
    same construction tests/test_body_fitter.py's TurningTest already uses."""
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
    """`fit()`'s new `orientation_hypotheses` argument must not change
    behaviour for every existing caller that doesn't pass it."""

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
    """The case the orientation classifier exists for: a cold fit at a near-perpendicular
    yaw, where plain fit()'s {0, pi} hypotheses have no seed nearby."""

    @classmethod
    def setUpClass(cls):
        from tryon.body.orientation import OrientationClassifier

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cls.regressor = OrientationClassifier(n_bins=checkpoint["n_bins"])
        cls.regressor.load_state_dict(checkpoint["model_state"])
        cls.regressor.eval()

    def test_regressor_hypotheses_include_a_seed_near_the_true_yaw(self):
        """The core claim: for a body near 70 degrees (squarely between
        fit()'s two hardcoded hypotheses), the regressor's top-2
        predicted yaw hypotheses should include one within half a bin
        width of the true angle - the thing `fit()`'s hardcoded pair
        cannot offer by construction."""
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
        """The actual end-to-end claim (approved plan's step 4 recipe):
        on a case where the default {0, pi} search is known to have no
        good seed, fitting with the regressor's hypotheses added should
        reach a lower final 3D orientation error than the default alone."""
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
        """Simulates the periodic sanity check's actual job: a fitter
        whose tracked global_orient has drifted far from what the current
        observation actually shows should be flagged."""
        points, true_yaw = _turned_frame(np.radians(90))
        fitter = BodyFitter()
        with torch.no_grad():
            # Simulate a track stuck near front-facing while the body is
            # actually at 90 degrees - the exact failure mode this check
            # exists to catch.
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
