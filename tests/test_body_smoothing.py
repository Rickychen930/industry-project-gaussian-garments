"""Tracking a body through video: the metric, then the behaviour.

Following VIBE (Kocabas et al., 2020), jitter is measured as acceleration
error of 3D joints, and shape is held fixed over a sequence. The metric is
tested first on synthetic sequences with exact ground truth.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.body.smoothing import acceleration_error


def _walk(frames=30, joints=13, speed=0.02, seed=0):
    """Joints moving at constant velocity: zero true acceleration."""
    rng = np.random.default_rng(seed)
    start = rng.normal(0, 0.3, (joints, 3))
    velocity = rng.normal(0, speed, (joints, 3))
    return np.stack([start + velocity * t for t in range(frames)])


class AccelerationErrorTest(unittest.TestCase):
    def test_a_perfect_prediction_scores_zero(self):
        truth = _walk()
        self.assertAlmostEqual(acceleration_error(truth, truth, fps=30.0), 0.0)

    def test_jitter_is_penalised_even_when_positions_are_close(self):
        """The whole point of the metric: MPJPE cannot see this."""
        truth = _walk()
        rng = np.random.default_rng(1)
        jittery = truth + rng.normal(0, 0.002, truth.shape)  # 2 mm noise
        position_error = np.linalg.norm(jittery - truth, axis=-1).mean() * 1000
        self.assertLess(position_error, 5.0, "noise too big for the test to mean anything")
        self.assertGreater(
            acceleration_error(jittery, truth, fps=30.0),
            50.0,
            "2 mm of per-frame jitter must produce a large acceleration error",
        )

    def test_a_constant_offset_is_not_acceleration(self):
        """Being consistently 5 cm off is a position error, not jitter."""
        truth = _walk()
        shifted = truth + np.array([0.05, 0.0, 0.0])
        self.assertAlmostEqual(acceleration_error(shifted, truth, fps=30.0), 0.0, places=6)

    def test_lag_is_penalised_at_a_turn(self):
        """Over-smoothing that misses a direction change must score badly."""
        frames = 40
        truth = np.zeros((frames, 1, 3))
        truth[:20, 0, 0] = np.linspace(0, 0.4, 20)
        truth[20:, 0, 0] = np.linspace(0.4, 0.0, 20)  # sharp reversal
        lagged = np.roll(truth, 3, axis=0)
        lagged[:3] = truth[:3]
        self.assertGreater(acceleration_error(lagged, truth, fps=30.0), 1.0)

    def test_units_are_millimetres_per_second_squared(self):
        """Scale with fps^2 and with metres-to-millimetres, as the paper's."""
        truth = _walk()
        jittery = truth + np.random.default_rng(2).normal(0, 0.002, truth.shape)
        at_30 = acceleration_error(jittery, truth, fps=30.0)
        at_60 = acceleration_error(jittery, truth, fps=60.0)
        self.assertAlmostEqual(at_60 / at_30, 4.0, places=3)

    def test_it_refuses_sequences_too_short_to_differentiate_twice(self):
        with self.assertRaises(ValueError):
            acceleration_error(_walk(frames=2), _walk(frames=2), fps=30.0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    __import__("tryon.body.fitter", fromlist=["SMPLX_ROOT"]).SMPLX_ROOT.is_dir(),
    "no SMPL-X body models",
)
class TrackerJitterTest(unittest.TestCase):
    """Freezing shape during tracking does not reduce jitter (measured).

    Pins the finding so the optimisation is not re-added from the paper alone.
    """

    def test_freezing_shape_is_not_an_improvement(self):
        from tests.synthetic_sequence import make_sequence, run

        ratios = []
        for seed in range(3):
            truth, observed = make_sequence(frames=24, seed=seed)
            _free_mpjpe, free_accel = run(observed, truth, track_shape=True)
            _frozen_mpjpe, frozen_accel = run(observed, truth, track_shape=False)
            ratios.append(frozen_accel / free_accel)
        mean_ratio = float(np.mean(ratios))
        self.assertGreater(
            mean_ratio,
            0.85,
            f"freezing shape now reduces jitter (ratio {mean_ratio:.2f}) - if that "
            "is real, re-measure over more seeds and change the tracker",
        )

    def test_tracking_a_known_motion_is_smoother_than_refitting_each_frame(self):
        """Warm-started tracking stays about as smooth as cold per-frame fitting.

        Tracking's value is yaw continuity and lower per-frame cost; the
        bound pins the measured ratio with room for noise."""
        import torch

        from tests.synthetic_sequence import make_sequence
        from tryon.body.fitter import BodyFitter

        truth, observed = make_sequence(frames=20, seed=0)
        cold = []
        for points in observed:
            fitter = BodyFitter()
            fitter.fit(points, iterations=60, camera_iterations=40)
            with torch.no_grad():
                cold.append(fitter._forward().joints[0, fitter._smplx_indices].numpy().copy())
        tracked = []
        fitter = BodyFitter()
        fitter.fit(observed[0])
        for points in observed:
            fitter.track(points, iterations=3)
            with torch.no_grad():
                tracked.append(fitter._forward().joints[0, fitter._smplx_indices].numpy().copy())
        cold_accel = acceleration_error(np.stack(cold), truth, fps=30.0)
        tracked_accel = acceleration_error(np.stack(tracked), truth, fps=30.0)
        self.assertLess(
            tracked_accel,
            1.6 * cold_accel,
            f"tracking ({tracked_accel:.0f}) is much jerkier than per-frame fitting ({cold_accel:.0f})",
        )
