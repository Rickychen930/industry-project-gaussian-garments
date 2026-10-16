"""The OrientationClassifier model + training pipeline,
sanity-checked before trusting it on a real (noisy) dataset.
"""

from __future__ import annotations

import unittest

import torch

from tryon.body.fitter import SMPLX_ROOT
from tryon.body.orientation import INPUT_SIZE, OrientationClassifier
from tryon.body.orientation_data import N_BINS


class OrientationClassifierShapeTest(unittest.TestCase):
    def test_output_shape(self):
        model = OrientationClassifier()
        x = torch.randn(5, INPUT_SIZE)
        logits = model(x)
        self.assertEqual(logits.shape, (5, N_BINS))

    def test_custom_bin_count(self):
        model = OrientationClassifier(n_bins=4)
        x = torch.randn(3, INPUT_SIZE)
        logits = model(x)
        self.assertEqual(logits.shape, (3, 4))


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class TrainingPipelineSanityTest(unittest.TestCase):
    """Checks the training pipeline is wired correctly, not real-data accuracy."""

    def test_overfits_a_tiny_noiseless_dataset(self):
        from tools.train.orientation import run_epoch, _to_loader
        from tryon.body.orientation_data import make_dataset

        # A small clean dataset should be memorised; the best epoch's
        # accuracy is checked because the final one can hit a dip.
        features, _, labels = make_dataset(200, seed=0, noise_px=0.0)
        loader = _to_loader(features, labels, batch_size=32, shuffle=True)

        # Seeded: init and loader shuffling use torch's global RNG, and an
        # unseeded run fell under the threshold once.
        torch.manual_seed(0)
        model = OrientationClassifier()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

        best_accuracy = 0.0
        for _ in range(150):
            _, accuracy = run_epoch(model, loader, optimiser, torch.device("cpu"), train=True)
            best_accuracy = max(best_accuracy, accuracy)

        self.assertGreaterEqual(
            best_accuracy, 0.90, f"best train accuracy only {best_accuracy:.3f} - pipeline likely broken"
        )


if __name__ == "__main__":
    unittest.main()
