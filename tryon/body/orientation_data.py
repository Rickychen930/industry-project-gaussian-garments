"""Synthetic training data for the orientation classifier: no real dataset needed for this narrow a problem.

Reuses the exact ground-truth-generation pattern `tests/synthetic_sequence.py`
already established for tracking-accuracy tests: pose a `BodyFitter` with
`torch.no_grad()`, read `_forward().joints`, project with `_project()`,
add Gaussian pixel noise to simulate MediaPipe's real detection noise.
Here the thing being predicted is not joint position but which yaw
OCTANT a body is facing - see `tryon/body/orientation.py`'s docstring for
why classification into coarse octants, not continuous angle regression,
is the right scope for what this layer actually needs to fix (BodyFitter.fit's
existing yaw in {0, pi} search already resolves the front/back case; this
targets the ~45-135 degree range that search has no seed near).

Sampling ranges (each chosen against a specific number already measured
elsewhere in this project, not guessed):
- betas: N(0,1) per component, clipped to [-2,2] - covers the observed
  real-data beta spread (0.75-3.4, measured on the annotated fits) without
  extrapolating far past it.
- body_pose: rest pose plus small random perturbation to shoulders,
  elbows, hips, knees only (SMPL-X body_pose joint indices 15,16,17,18,
  1,2,4,5) - covers natural stance variation during a turn without
  turning this into a full pose-estimation training set, which is out
  of scope (SMPLify's own continuous optimisation already handles pose
  refinement once seeded in the right orientation).
- global_orient yaw: uniform over the full [-pi, pi) turn; pitch/roll
  near zero with small jitter, matching a standing person's upright
  prior (the same prior `BodyFitter._step` already enforces during
  fitting).
- camera scale/translation: varied so the classifier cannot cheat by
  learning screen position instead of body orientation.
- 2D pixel noise: Gaussian, same `noise_px` default as
  `synthetic_sequence.py` uses.
- missing joints: each joint independently dropped (visibility zeroed)
  with its own probability - higher for wrists/ankles, which
  `JOINT_WEIGHTS` already down-weights for being more often occluded or
  out of frame.
"""

from __future__ import annotations

import numpy as np
import torch

from tryon.body.fitter import BodyFitter
from tryon.body.landmarks import JOINT_MAP

N_BINS = 8  # 45-degree octants, matching the design plan's default
_JOINT_ORDER = sorted(JOINT_MAP)

# SMPL-X body_pose axis indices (each joint has 3, x/y/z) to perturb -
# shoulders, elbows, hips, knees. Pelvis/spine/neck/wrists/ankles/feet
# left at rest: perturbing everything would make this a full pose
# variation dataset, which is out of scope (see module docstring).
_PERTURBED_JOINTS = (1, 2, 4, 5, 15, 16, 17, 18)

# Per-tracked-landmark dropout probability - keyed by MediaPipe landmark
# index (matches JOINT_MAP's keys), mirroring JOINT_WEIGHTS' own view of
# which joints are least reliable.
_DROPOUT_PROBABILITY = {
    11: 0.05,  # LEFT_SHOULDER
    12: 0.05,  # RIGHT_SHOULDER
    13: 0.08,  # LEFT_ELBOW
    14: 0.08,  # RIGHT_ELBOW
    15: 0.12,  # LEFT_WRIST
    16: 0.12,  # RIGHT_WRIST
    23: 0.05,  # LEFT_HIP
    24: 0.05,  # RIGHT_HIP
    25: 0.08,  # LEFT_KNEE
    26: 0.08,  # RIGHT_KNEE
    27: 0.15,  # LEFT_ANKLE
    28: 0.15,  # RIGHT_ANKLE
}


def yaw_to_bin(yaw: float, n_bins: int = N_BINS) -> int:
    """Which octant (or n_bins-way division) of [-pi, pi) a yaw falls in.

    Bin 0 is centred on yaw=0 (facing the camera), matching the natural
    reference direction `BodyFitter.fit`'s own yaw=0 hypothesis uses.
    """
    bin_width = 2.0 * np.pi / n_bins
    shifted = (yaw + np.pi / n_bins) % (2.0 * np.pi)
    return int(shifted // bin_width) % n_bins


def bin_center(bin_index: int, n_bins: int = N_BINS) -> float:
    """Inverse of `yaw_to_bin`: the representative yaw for a bin, in
    (-pi, pi]. Used as the seed angle when a hypothesis is generated from
    a predicted bin."""
    bin_width = 2.0 * np.pi / n_bins
    yaw = bin_index * bin_width
    return float(((yaw + np.pi) % (2.0 * np.pi)) - np.pi)


def sample_pose_and_keypoints(rng: np.random.Generator, noise_px: float = 2.0, fitter: BodyFitter | None = None):
    """One synthetic (features, yaw, bin) training example.

    Returns (features, yaw, bin_label) where `features` is the flattened,
    body-centred/scale-normalised (x, y, visibility) for each of the 12
    tracked landmarks, in `_JOINT_ORDER` - exactly the layout
    `tryon/body/orientation.py`'s model consumes.

    `fitter`: reuse an existing `BodyFitter` instead of constructing a new
    one (measured at ~0.67s each - the dominant cost of generating many
    samples, since it loads the SMPL-X model from disk every time). Every
    parameter this function touches is overwritten before use, so reusing
    one fitter across samples is safe - nothing carries over between
    calls. `make_dataset` always passes one; a fresh one is constructed
    here only for a single ad-hoc call.
    """
    if fitter is None:
        fitter = BodyFitter()
    with torch.no_grad():
        # Drawn from `rng` (numpy), not `torch.randn` - the latter uses
        # torch's own unseeded global RNG, which made `make_dataset` with
        # a fixed seed non-deterministic (caught by
        # test_shapes_and_determinism before this was trusted).
        betas = np.clip(rng.normal(0.0, 1.0, 10), -2.0, 2.0)
        fitter.betas[0] = torch.as_tensor(betas, dtype=torch.float32)
        fitter.body_pose.zero_()
        for joint_index in _PERTURBED_JOINTS:
            for axis in range(3):
                fitter.body_pose[0, joint_index * 3 + axis] = float(rng.normal(0.0, 0.25))

        yaw = float(rng.uniform(-np.pi, np.pi))
        pitch = float(rng.normal(0.0, 0.05))
        roll = float(rng.normal(0.0, 0.05))
        fitter.global_orient[0, 0] = roll
        fitter.global_orient[0, 1] = yaw
        fitter.global_orient[0, 2] = pitch

        scale = float(rng.uniform(150.0, 300.0))
        translation = rng.uniform([200.0, 150.0], [450.0, 350.0])
        fitter.scale.fill_(scale)
        fitter.translation[:] = torch.as_tensor(translation, dtype=torch.float32).unsqueeze(0)

        joints = fitter._forward().joints[0, fitter._smplx_indices]
        pixels = fitter._project(joints).numpy()

    xy = pixels + rng.normal(0.0, noise_px, pixels.shape)
    visibility = np.ones(len(_JOINT_ORDER), dtype=np.float64)
    for row, landmark_index in enumerate(_JOINT_ORDER):
        drop_probability = _DROPOUT_PROBABILITY.get(landmark_index, 0.05)
        if rng.uniform() < drop_probability:
            visibility[row] = 0.0

    # Normalise using a centre/scale derived from the OBSERVED keypoints
    # themselves (visible-landmark mean and bounding-box spread), NOT the
    # ground-truth camera scale/translation used to render them. The
    # regressor is meant to run at inference time BEFORE any camera fit
    # exists (that is the whole point - it seeds `fit()`), so it must be
    # trained on the same normalisation it will actually receive at
    # inference (`BodyFitter.orientation_hypotheses_from_regressor` uses
    # this identical scheme) - using the ground-truth camera params here
    # was a real train/inference mismatch bug, caught by
    # tests/test_body_orientation_integration.py producing nonsense
    # hypotheses (off by 100+ degrees) before this fix.
    visible_xy = xy[visibility > 0]
    if len(visible_xy) > 0:
        centre = visible_xy.mean(axis=0)
        spread = float(np.linalg.norm(visible_xy.max(axis=0) - visible_xy.min(axis=0)))
    else:
        centre = xy.mean(axis=0)
        spread = 1.0
    observed_scale = max(spread, 1.0)

    features = _normalize_features(xy, visibility, observed_scale, centre)
    bin_label = yaw_to_bin(yaw)
    return features, yaw, bin_label


def _normalize_features(xy: np.ndarray, visibility: np.ndarray, scale: float, translation) -> np.ndarray:
    """Body-centred, scale-normalised (x, y, visibility) per joint,
    flattened - invariant to the person's position/size in frame, so the
    classifier learns orientation from joint CONFIGURATION, not screen
    placement (translation/scale are sampled independently of yaw in
    `sample_pose_and_keypoints` specifically so a model that ignored this
    normalisation could not cheat by learning them anyway).
    """
    centered = (xy - np.asarray(translation)) / max(scale, 1e-6)
    return np.concatenate([centered.reshape(-1), visibility]).astype(np.float32)


def make_dataset(n_samples: int, seed: int = 0, noise_px: float = 2.0):
    """(features, yaws, bin_labels) arrays for `n_samples` synthetic examples.

    Constructs a single `BodyFitter` and reuses it for every sample (see
    `sample_pose_and_keypoints`'s docstring for why - ~0.67s/construction
    otherwise dominates generation time at the 10k-50k sample counts this
    is meant to produce).
    """
    rng = np.random.default_rng(seed)
    fitter = BodyFitter()
    features = np.zeros((n_samples, 3 * len(_JOINT_ORDER)), dtype=np.float32)
    yaws = np.zeros(n_samples, dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        features[i], yaws[i], labels[i] = sample_pose_and_keypoints(rng, noise_px, fitter=fitter)
    return features, yaws, labels
