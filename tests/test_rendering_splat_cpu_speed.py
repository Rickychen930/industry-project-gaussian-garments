"""The rasteriser must stay exact while it gets fast.

Compositing is order-dependent, so only the work within a tile is
vectorised (8x8 tiles; see `TILE` in `tryon/rendering/splat_cpu.py`). The old
loop is kept as `render_splats_reference` and is the oracle: the fast path
must match it to a quarter of a grey level.
"""

from __future__ import annotations

import statistics
import time
import unittest

import numpy as np

from tryon.rendering.splat_cpu import (
    _rotation_to_quaternion_reference,
    look_at,
    quaternion_to_rotation,
    render_splats,
    render_splats_reference,
    rotation_to_quaternion,
)


def _splats(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {
        "xyz": rng.normal(0, 0.3, (n, 3)),
        "rotation": q,
        "scaling": np.log(rng.uniform(0.005, 0.03, (n, 3))),
        "opacity": rng.normal(1.0, 1.0, (n, 1)),
        "feature_dc": rng.uniform(-1, 1, (n, 1, 3)),
        "feature_rest": rng.normal(0, 0.1, (n, 15, 3)),
    }


def _camera(size=(120, 160)):
    h, w = size
    view = look_at((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
    k = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1.0]])
    return view, k, size


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


class RasteriserExactnessTest(unittest.TestCase):
    def test_fast_matches_reference_to_a_quarter_level(self):
        splats, (view, k, size) = _splats(), _camera()
        ref_img, ref_a = render_splats_reference(splats, view, k, size)
        img, a = render_splats(splats, view, k, size)
        self.assertLess(float(np.abs(img - ref_img).max()) * 255, 0.25)
        self.assertLess(float(np.abs(a - ref_a).max()) * 255, 0.25)

    def test_fast_matches_reference_with_large_splats(self):
        """Splats spanning many tiles - the case a tiling bug would break."""
        splats = _splats(300)
        splats["scaling"] = np.log(np.full((300, 3), 0.15))
        view, k, size = _camera()
        ref_img, ref_a = render_splats_reference(splats, view, k, size)
        img, a = render_splats(splats, view, k, size)
        self.assertLess(float(np.abs(img - ref_img).max()) * 255, 0.25)
        self.assertLess(float(np.abs(a - ref_a).max()) * 255, 0.25)

    def test_empty_and_offscreen_inputs(self):
        view, k, size = _camera()
        _img, a = render_splats(_splats(0), view, k, size)
        self.assertEqual(float(a.max()), 0.0)
        far = _splats(50)
        far["xyz"] += np.array([100.0, 0, 0])
        _img, a = render_splats(far, view, k, size)
        self.assertEqual(float(a.max()), 0.0)

    def test_fast_is_substantially_faster(self):
        """A conservative floor under noisy timings on both workloads.

        Measured speedups ranged 2.7x to 3.9x, so 2x is asserted rather than
        the largest number seen once.
        """
        splats, (view, k, size) = _splats(20000), _camera((240, 320))
        render_splats(splats, view, k, size)
        render_splats_reference(splats, view, k, size)
        fast = statistics.median(
            _timed(lambda: render_splats(splats, view, k, size)) for _ in range(5)
        )
        slow = statistics.median(
            _timed(lambda: render_splats_reference(splats, view, k, size)) for _ in range(3)
        )
        self.assertLess(fast * 2, slow, f"fast {fast*1000:.0f} ms vs reference {slow*1000:.0f} ms")


class QuaternionTest(unittest.TestCase):
    def test_vectorised_matches_reference(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(5000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        m = quaternion_to_rotation(q)
        a, b = rotation_to_quaternion(m), _rotation_to_quaternion_reference(m)
        # q and -q are the same rotation; compare up to sign.
        sign = np.sign((a * b).sum(axis=1, keepdims=True))
        self.assertLess(float(np.abs(a * sign - b).max()), 1e-9)

    def test_round_trip_through_rotation(self):
        rng = np.random.default_rng(2)
        q = rng.normal(size=(2000, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        back = quaternion_to_rotation(rotation_to_quaternion(quaternion_to_rotation(q)))
        self.assertLess(float(np.abs(back - quaternion_to_rotation(q)).max()), 1e-9)

    def test_all_four_shepperd_branches(self):
        """Rotations whose largest diagonal is each of the three axes, plus trace>0."""
        angles = [np.pi - 0.01] * 3 + [0.1]
        axes = np.eye(3).tolist() + [[0, 0, 1]]
        ms = []
        for ang, ax in zip(angles, axes):
            ax = np.asarray(ax, float)
            kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            ms.append(np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx)
        ms = np.stack(ms)
        self.assertLess(float(np.abs(quaternion_to_rotation(rotation_to_quaternion(ms)) - ms).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
