"""Evaluate the trained checkpoint on a held-out synthetic set: overall
accuracy and accuracy on near-bin-boundary (hard) poses."""

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

        # A seed disjoint from those used in training.
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
        # Generous floor below the measured 0.845, so retraining noise does not fail it.
        self.assertGreater(overall_accuracy, 0.70)

    def test_far_from_boundary_accuracy_is_high(self):
        """Poses solidly inside an octant are classified reliably."""
        mask = self._distance_to_boundary_deg() >= 15.0
        accuracy = (self.predictions[mask] == self.labels[mask]).mean()
        self.assertGreater(accuracy, 0.85)

    def test_near_boundary_is_the_hardest_but_still_above_baseline(self):
        """Near-boundary poses are worse than easy ones but above the majority baseline."""
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
