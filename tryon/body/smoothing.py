"""Measuring a body track through video.

Following VIBE (Kocabas et al., CVPR 2020): per-frame fitting is jittery in a way
position metrics (MPJPE) cannot see, so the number used is **acceleration error**, in
mm/s^2. Acceleration is the second finite difference of position, so it needs at least
three frames; a constant offset scores zero (that is a position error).
"""

from __future__ import annotations

import numpy as np


def _acceleration(joints, fps):
    """Second finite difference, in metres per second squared."""
    joints = np.asarray(joints, np.float64)
    return (joints[2:] - 2 * joints[1:-1] + joints[:-2]) * (float(fps) ** 2)


def acceleration_error(predicted, truth, fps):
    """Mean |accel(predicted) - accel(truth)| over joints and frames, mm/s^2.

    `predicted` and `truth` are (frames, joints, 3) in metres.
    """
    predicted = np.asarray(predicted, np.float64)
    truth = np.asarray(truth, np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("expected (frames, joints, 3)")
    if predicted.shape[0] < 3:
        raise ValueError("need at least 3 frames to measure acceleration")
    difference = _acceleration(predicted, fps) - _acceleration(truth, fps)
    return float(np.linalg.norm(difference, axis=-1).mean()) * 1000.0


def joint_acceleration(joints, fps):
    """Mean |accel| of a track on its own, mm/s^2, for when there is no ground truth (live
    webcam). Lower is steadier; it cannot tell steadiness from lag, so always read it with
    reprojection error."""
    joints = np.asarray(joints, np.float64)
    if joints.shape[0] < 3:
        raise ValueError("need at least 3 frames to measure acceleration")
    return float(np.linalg.norm(_acceleration(joints, fps), axis=-1).mean()) * 1000.0


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for landmarks.

    A low-pass filter whose cutoff rises with speed: heavy smoothing at rest, little lag
    when fast. Applied to MediaPipe's (33, 4) landmark array: x and y in pixels (normalised
    by the person's height), world z as is; visibility passes through.

    `min_cutoff` Hz: cutoff at rest. `beta`: how fast the cutoff grows with speed (per
    person-height per second). `d_cutoff` Hz: cutoff of the speed estimate.
    """

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, relative=False):
        """`relative`: filter the pose about the hip midpoint (MediaPipe 23,
        24) and add the RAW hip midpoint back, so limbs are smoothed but
        the body's position is never delayed."""
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.relative = bool(relative)
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None
        self._height = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, points, t):
        points = np.asarray(points, np.float64)
        columns = [0, 1] + [c for c in (3, 4) if points.shape[1] > c]
        body = points[:, :2]
        height = float(np.ptp(body[np.isfinite(body[:, 1]), 1])) if np.isfinite(body).any() else 1.0
        height = height if height > 1.0 else 1.0
        # the normaliser itself is measured on noisy landmarks: smooth it,
        # or dividing and re-multiplying by it puts the jitter back
        self._height = height if self._height is None else self._height + 0.05 * (height - self._height)
        height = self._height
        x = points[:, columns].copy()
        anchor = points[[23, 24], :2].mean(axis=0) if self.relative else np.zeros(2)
        x[:, :2] -= anchor
        x[:, :2] /= height
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return points.astype(np.float32)
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        out = points.copy()
        filtered = x_hat.copy()
        filtered[:, :2] *= height
        filtered[:, :2] += anchor
        out[:, columns] = filtered
        return out.astype(np.float32)
