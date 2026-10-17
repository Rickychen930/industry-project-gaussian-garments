"""An orientation classifier for 360-degree tracking.

Two failure modes are already solved elsewhere in this project and are
NOT what this file targets:

- The discrete front/back (yaw=0 vs yaw=pi) ambiguity at cold start:
  `BodyFitter.fit()` already tries both hypotheses and keeps whichever
  fits better (tryon/body/fitter.py, `fit`'s docstring).
- A turn that starts front-facing and proceeds at a normal pace: warm
  tracking with `TRACK_ITERATIONS=15` already follows it
  (tests/test_body_fitter.py's TurningTest).

What is NOT solved: a person who first appears (or a track that gets
stuck) near-perpendicular to the camera - roughly the 45-135 degree
range, where `fit()`'s two hardcoded hypotheses (0, pi) have no seed
nearby, so gradient descent from either can converge to a
locally-plausible but wrong yaw. This is a genuinely discrete
disambiguation problem (which coarse octant is the body facing), not an
imprecision problem - SMPLify's own continuous optimisation already
refines a pose well once seeded in roughly the right basin.

Scope, deliberately narrow: classify which 45-degree yaw octant a body
is facing, from the same 12 MediaPipe-derived keypoints
`BodyFitter`/`landmarks_to_targets` already track - not a full
image-based pose network. A plain small MLP over 36 normalised scalars
(12 joints x (x, y, visibility)) is the right scale for an 8-way
classification task; a CNN over raw pixels would be a far larger
architecture solving a much bigger problem than the one that actually
needs solving here, and would add a second image model to the loop (MediaPipe's keypoints are
the input, as everywhere else in this project).

Trained entirely on synthetic data (`tryon/body/orientation_data.py`),
generated the same way `tests/synthetic_sequence.py` already does for
tracking-accuracy tests: pose a `BodyFitter`, read exact ground truth,
project to 2D, add realistic pixel noise and occlusion dropout. No real
photos needed or used - this project's own annotated data (48 images
total) is far too little to train a generalising classifier on directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tryon.body.orientation_data import N_BINS, _JOINT_ORDER

INPUT_SIZE = 3 * len(_JOINT_ORDER)  # (x, y, visibility) per tracked joint


class OrientationClassifier(nn.Module):
    """Input: (x, y, visibility) per tracked joint, body-centred and
    scale-normalised (see `pose_regressor_data._normalize_features`).
    Output: unnormalised logits over `N_BINS` yaw octants.

    ~6.9k parameters (36x64 + 64x64 + 64x8, plus biases) - small enough
    to train in seconds to low minutes on CPU and run every frame at
    negligible cost if ever needed, though the actual usage (see
    `BodyFitter`) only calls it at cold start and on periodic sanity
    checks during tracking, not every frame.
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 64, n_bins: int = N_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
