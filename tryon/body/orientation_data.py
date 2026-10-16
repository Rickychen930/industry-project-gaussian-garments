"""Synthetic training data for the orientation classifier.

Poses a `BodyFitter` with `torch.no_grad()`, reads `_forward().joints`, projects with
`_project()` and adds Gaussian pixel noise, the way `tests/synthetic_sequence.py` does.
The target is which yaw OCTANT the body faces (see `tryon/body/orientation.py`): the
{0, pi} search in `BodyFitter.fit` already covers front/back, this covers ~45-135 degrees.

Sampling:
- betas: N(0,1) clipped to [-2,2], covering the observed real-data spread.
- body_pose: rest plus small perturbation of shoulders, elbows, hips, knees only.
- global_orient yaw uniform over the full turn; pitch/roll near zero (upright prior).
- camera scale/translation varied so position cannot stand in for orientation.
- 2D pixel noise: Gaussian, `noise_px` as in `synthetic_sequence.py`.
- missing joints: dropped independently, more often for wrists and ankles.
"""

from __future__ import annotations

import numpy as np
import torch

from tryon.body.fitter import BodyFitter
from tryon.body.landmarks import JOINT_MAP

N_BINS = 8  # 45-degree octants
_JOINT_ORDER = sorted(JOINT_MAP)

# SMPL-X body_pose axis indices (3 per joint) to perturb: shoulders, elbows, hips,
# knees. The rest stay at rest (a full pose dataset is out of scope).
_PERTURBED_JOINTS = (1, 2, 4, 5, 15, 16, 17, 18)

# Per-landmark dropout probability, keyed by MediaPipe index; mirrors JOINT_WEIGHTS.
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
    """Which octant (or n_bins-way division) of [-pi, pi) a yaw falls in. Bin 0 is centred
    on yaw=0 (facing the camera)."""
    bin_width = 2.0 * np.pi / n_bins
    shifted = (yaw + np.pi / n_bins) % (2.0 * np.pi)
    return int(shifted // bin_width) % n_bins


def bin_center(bin_index: int, n_bins: int = N_BINS) -> float:
    """Inverse of `yaw_to_bin`: the representative yaw for a bin, in (-pi, pi], used as a
    seed angle."""
    bin_width = 2.0 * np.pi / n_bins
    yaw = bin_index * bin_width
    return float(((yaw + np.pi) % (2.0 * np.pi)) - np.pi)


def sample_pose_and_keypoints(rng: np.random.Generator, noise_px: float = 2.0, fitter: BodyFitter | None = None):
    """One synthetic (features, yaw, bin_label) training example.

    `features` is the flattened, body-centred, scale-normalised (x, y, visibility) of the
    12 tracked landmarks in `_JOINT_ORDER`, the layout `tryon/body/orientation.py` consumes.

    `fitter`: reuse an existing `BodyFitter` (constructing one loads SMPL-X, ~0.67 s);
    every parameter touched is overwritten, so reuse is safe. `make_dataset` passes one.
    """
    if fitter is None:
        fitter = BodyFitter()
    with torch.no_grad():
        # Drawn from `rng` (numpy), not torch's unseeded global RNG, so a fixed seed is deterministic.
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

    # Normalise using a centre/scale from the OBSERVED keypoints, NOT the ground-truth
    # camera: inference runs before any camera fit exists and uses the same scheme
    # (`BodyFitter.orientation_hypotheses_from_regressor`).
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
    """Body-centred, scale-normalised (x, y, visibility) per joint, flattened: invariant to
    position and size, so the classifier learns joint CONFIGURATION, not screen placement.
    """
    centered = (xy - np.asarray(translation)) / max(scale, 1e-6)
    return np.concatenate([centered.reshape(-1), visibility]).astype(np.float32)


def make_dataset(n_samples: int, seed: int = 0, noise_px: float = 2.0):
    """(features, yaws, bin_labels) arrays for `n_samples` synthetic examples, reusing one
    `BodyFitter` for every sample."""
    rng = np.random.default_rng(seed)
    fitter = BodyFitter()
    features = np.zeros((n_samples, 3 * len(_JOINT_ORDER)), dtype=np.float32)
    yaws = np.zeros(n_samples, dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        features[i], yaws[i], labels[i] = sample_pose_and_keypoints(rng, noise_px, fitter=fitter)
    return features, yaws, labels
