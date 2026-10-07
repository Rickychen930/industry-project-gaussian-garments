"""A joints-only SMPL-X forward for tracking: the same math, without the mesh.

Profiling a live frame put `track` at 29 ms of a 51 ms frame - the
difference between choppy and smooth. Each of its three iterations ran the
library's full forward, skinning 10,475 vertices to read 12 joints, because
SMPL-X regresses its joints from the posed template inside `lbs`.

The joints do not need the skinned mesh. `lbs` computes them as
`rigid_transform(rodrigues(pose), regressor @ (template + shape_blend))`,
which is a kinematic chain over 55 joints - a few hundred multiply-adds.
Measured: 2.2 ms against 9.7 ms per forward+backward, and 0.0000000 m
difference from the library once the inputs match.

Matching the inputs is the whole difficulty, and two of them are not what
the model's public attributes suggest. Both are pinned here so they cannot
silently drift:

* the shape blend uses the 20-wide `shapedirs` (betas AND expression),
  not the 10-wide tensor the attribute of that name returns
* the hand pose has `left_hand_mean` / `right_hand_mean` ADDED to it.
  Zeros are not the rest hand; the mean is. Omitting it bent the wrists
  by 4 cm - exactly the joints a sleeve hangs from.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT, BodyFitter


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class FastJointsTest(unittest.TestCase):
    def _posed_fitter(self):
        fitter = BodyFitter()
        with torch.no_grad():
            fitter.body_pose[0, 15 * 3 + 2] = -1.0
            fitter.body_pose[0, 16 * 3 + 2] = 1.0
            fitter.global_orient[0, 1] = 0.7
            fitter.betas[0, 0] = 1.5
        return fitter

    def test_it_matches_the_library_forward_exactly(self):
        fitter = self._posed_fitter()
        library = fitter._forward().joints[0, fitter._smplx_indices].detach()
        fast = fitter.joints().detach()
        self.assertLess(
            float((library - fast).abs().max()),
            1e-5,
            "joints-only forward disagrees with the library",
        )

    def test_it_matches_at_rest_too(self):
        fitter = BodyFitter()
        library = fitter._forward().joints[0, fitter._smplx_indices].detach()
        self.assertLess(float((library - fitter.joints().detach()).abs().max()), 1e-5)

    def test_gradients_flow_to_pose_shape_and_camera(self):
        fitter = self._posed_fitter()
        loss = (fitter._project(fitter.joints()) ** 2).sum()
        loss.backward()
        for name in ("body_pose", "global_orient", "betas", "scale", "translation"):
            grad = getattr(fitter, name).grad
            self.assertIsNotNone(grad, f"no gradient reached {name}")
            self.assertTrue(torch.isfinite(grad).all(), f"non-finite gradient on {name}")

    def test_it_is_substantially_faster(self):
        import time

        fitter = self._posed_fitter()

        def bench(fn, n=20):
            for _ in range(3):
                fn()
            start = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - start) / n

        def library():
            fitter._forward().joints[0, fitter._smplx_indices].sum().backward()

        def fast():
            fitter.joints().sum().backward()

        self.assertLess(bench(fast), bench(library) * 0.6)

    def test_tracking_still_converges_on_the_fast_path(self):
        from tests.test_body_fitter import _landmarks

        fitter = BodyFitter()
        fitter.fit(_landmarks(), iterations=40)
        errors = [fitter.reprojection_error(_landmarks())]
        for frame in range(1, 6):
            moved = _landmarks(shift_x=frame * 4.0)
            fitter.track(moved, iterations=3)
            errors.append(fitter.reprojection_error(moved))
        self.assertLess(max(errors), errors[0] * 2.0, f"diverged: {np.round(errors, 1)}")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), f"no SMPL-X body models at {SMPLX_ROOT}")
class ChainCacheTest(unittest.TestCase):
    """The kinematic chain is memoised by the parameters' version counters."""

    def _fitter(self):
        fitter = BodyFitter()
        with torch.no_grad():
            fitter.body_pose.normal_(0, 0.1)
            fitter.betas[0, 0] = 0.5
        return fitter

    def test_a_cached_call_equals_a_fresh_one_and_is_not_recomputed(self):
        fitter = self._fitter()
        with torch.no_grad():
            first, transforms = fitter.joints(), fitter.joint_transforms()
            calls = []
            original = fitter._chain_uncached
            fitter._chain_uncached = lambda *a: calls.append(1) or original(*a)
            again, transforms_again = fitter.joints(), fitter.joint_transforms()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.equal(transforms, transforms_again))

    def test_an_in_place_parameter_change_invalidates_it(self):
        fitter = self._fitter()
        with torch.no_grad():
            before = fitter.joints().clone()
            fitter.body_pose[0, 16 * 3 + 2] += 0.5
            fitter.betas[0, 1] = 1.0
            after = fitter.joints()
            fresh = BodyFitter()
            fresh.body_pose.copy_(fitter.body_pose)
            fresh.betas.copy_(fitter.betas)
            expected = fresh.joints()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.equal(after, expected))

    def test_callers_cannot_edit_the_cache(self):
        fitter = self._fitter()
        with torch.no_grad():
            joints, transforms = fitter.joints(), fitter.joint_transforms()
            reference_joints, reference_transforms = joints.clone(), transforms.clone()
            joints.zero_()
            transforms.zero_()
            self.assertTrue(torch.equal(fitter.joints(), reference_joints))
            self.assertTrue(torch.equal(fitter.joint_transforms(), reference_transforms))
            fitter.mesh_offset = torch.tensor([[0.1, -0.2, 0.3]])
            offset_transforms = fitter.joint_transforms()
            self.assertTrue(torch.equal(fitter.joint_transforms(), offset_transforms))
            self.assertTrue(torch.equal(reference_transforms[:, :3, :3], offset_transforms[:, :3, :3]))

    def test_gradients_still_flow_through_the_uncached_path(self):
        fitter = self._fitter()
        fitter.joints().sum().backward()
        self.assertGreater(float(fitter.body_pose.grad.abs().sum()), 0.0)

    def test_the_working_factor_leaves_the_fit_alone_and_is_stable(self):
        from tryon.body.landmarks import JOINT_MAP

        fitter = self._fitter()
        points = np.zeros((33, 5), np.float32)
        points[:, 2] = 1.0
        rng = np.random.default_rng(0)
        for index in JOINT_MAP:
            points[index, :2] = rng.uniform(100, 500, 2)
        pose, betas = fitter.body_pose.detach().clone(), fitter.betas.detach().clone()
        versions = (fitter.body_pose._version, fitter.betas._version)
        first = fitter._working_factor(points)
        self.assertEqual(first, fitter._working_factor(points))
        self.assertTrue(torch.equal(fitter.body_pose, pose) and torch.equal(fitter.betas, betas))
        self.assertEqual(versions, (fitter.body_pose._version, fitter.betas._version))


if __name__ == "__main__":
    unittest.main()
