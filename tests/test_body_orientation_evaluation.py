"""Evaluate the trained checkpoint on a held-out
synthetic test set (a fresh seed the training run never saw), checking
both overall accuracy and the specific hard-case metric the approved
design plan names: accuracy on near-perpendicular (near-bin-boundary)
poses, where the disambiguation problem this layer exists to solve is
actually hard.
"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT
from tryon.body.orientation import OrientationClassifier
from tryon.body.orientation_data import N_BINS, make_dataset

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "pose_regressor.pt"


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(CHECKPOINT.exists(), "run tools/train/orientation.py first")
class TrainedCheckpointEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cls.model = OrientationClassifier(n_bins=checkpoint["n_bins"])
        cls.model.load_state_dict(checkpoint["model_state"])
        cls.model.eval()

        # A seed disjoint from every seed tools/train/orientation.py used
        # (0, 1, 2 by default) - a genuinely held-out set.
        cls.features, cls.yaws, cls.labels = make_dataset(3000, seed=999, noise_px=2.0)
        with torch.no_grad():
            logits = cls.model(torch.as_tensor(cls.features))
        cls.predictions = logits.argmax(dim=1).numpy()

    def test_beats_majority_class_baseline(self):
        overall_accuracy = (self.predictions == self.labels).mean()
        majority_class = Counter(self.labels).most_common(1)[0][0]
        baseline_accuracy = (self.labels == majority_class).mean()
        self.assertGreater(
            overall_accuracy,
            baseline_accuracy + 0.3,
            f"overall {overall_accuracy:.3f} vs baseline {baseline_accuracy:.3f} - "
            "not clearly doing real work over the trivial baseline",
        )

    def test_overall_accuracy_is_reasonable(self):
        overall_accuracy = (self.predictions == self.labels).mean()
        # Measured on this exact checkpoint/seed: 0.845. A generous floor
        # below that, not a re-assertion of the exact number, so a
        # retrained checkpoint with minor variation does not spuriously
        # fail this test.
        self.assertGreater(overall_accuracy, 0.70)

    def test_far_from_boundary_accuracy_is_high(self):
        """The easy case (yaw solidly in the middle of an octant) should
        be classified reliably - if this is also weak, something more
        fundamental than "boundaries are hard" is wrong."""
        mask = self._distance_to_boundary_deg() >= 15.0
        accuracy = (self.predictions[mask] == self.labels[mask]).mean()
        self.assertGreater(accuracy, 0.85)

    def test_near_boundary_is_the_hardest_but_still_above_baseline(self):
        """The metric named directly in the approved design plan: how
        well does the model do on the genuinely ambiguous cases this
        layer exists to help with. It should be measurably worse than
        the far-from-boundary case (confirms the model is behaving
        sensibly, not just memorising screen-position artifacts) but
        still clearly above the random/majority baseline (confirms it is
        doing SOME real disambiguation work even in the hard region,
        which is the actual point of this layer)."""
        near = self._distance_to_boundary_deg() < 10.0
        far = self._distance_to_boundary_deg() >= 15.0
        near_accuracy = (self.predictions[near] == self.labels[near]).mean()
        far_accuracy = (self.predictions[far] == self.labels[far]).mean()

        self.assertLess(near_accuracy, far_accuracy, "near-boundary cases should be harder than far-from-boundary ones")
        majority_class = Counter(self.labels).most_common(1)[0][0]
        baseline_accuracy = (self.labels == majority_class).mean()
        self.assertGreater(near_accuracy, baseline_accuracy + 0.15)

    def _distance_to_boundary_deg(self) -> np.ndarray:
        bin_width_deg = 360.0 / N_BINS
        yaws_deg = np.degrees(self.yaws)
        distance_to_center = np.abs(((yaws_deg + bin_width_deg / 2) % bin_width_deg) - bin_width_deg / 2)
        return bin_width_deg / 2 - distance_to_center


if __name__ == "__main__":
    unittest.main()
