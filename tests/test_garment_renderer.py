"""Render a garment onto the tracked body.

Splats are stored in face-local frames, so deforming the mesh moves them
(Gaussian Garments, Rong et al.). The tests check the pieces compose: a
garment that renders at rest must also render posed, turned, and on a real
fitted body.
"""

from __future__ import annotations

import unittest

import numpy as np

from tryon.body.fitter import SMPLX_ROOT
from tryon.garment.renderer import GarmentRenderer
from tryon.paths import garment

GARMENT = garment("00152_outer")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class GarmentRendererTest(unittest.TestCase):
    """`dynamics=False`: cloth springs would integrate wall-clock time between
    calls, which rigid-follow assertions and frame timing do not want."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = GarmentRenderer(GARMENT, dynamics=False)

    def test_it_renders_something_at_rest(self):
        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        rgba = self.renderer.render(fitter, (240, 320))
        self.assertEqual(rgba.shape, (240, 320, 4))
        coverage = float((rgba[..., 3] > 25).mean())
        self.assertGreater(coverage, 0.02, f"garment covered {coverage:.1%} of the frame")

    def test_the_garment_follows_a_turn(self):
        import torch

        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        front = self.renderer.render(fitter, (240, 320))
        with torch.no_grad():
            fitter.global_orient[0, 1] = np.pi / 2
        side = self.renderer.render(fitter, (240, 320))
        def width(rgba):
            return np.ptp(np.flatnonzero((rgba[..., 3] > 25).any(axis=0)))
        self.assertLess(width(side), width(front) * 0.8, "profile is not narrower")

    def test_the_garment_follows_the_arms(self):
        import torch

        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        before = self.renderer.posed_mesh(fitter).copy()
        down = self.renderer.render(fitter, (240, 320))
        with torch.no_grad():
            wrist_before = fitter._forward().joints[0, 20].numpy().copy()
            fitter.body_pose[0, 15 * 3 + 2] = -1.2
            fitter.body_pose[0, 16 * 3 + 2] = 1.2
            wrist_after = fitter._forward().joints[0, 20].numpy().copy()
        after = self.renderer.posed_mesh(fitter)
        up = self.renderer.render(fitter, (240, 320))
        # Vertices skinned to the left arm must travel with it. Sleeves are
        # only ~0.4% of this small frame, so the vertex displacement is
        # checked rather than alpha change.
        weights = self.renderer._binding.weights
        sleeve = weights[:, [16, 18, 20]].sum(axis=1) > 0.5
        self.assertGreater(int(sleeve.sum()), 200, "no vertices bound to the left arm")
        moved = np.median(np.linalg.norm(after[sleeve] - before[sleeve], axis=1))
        swing = float(np.linalg.norm(wrist_after - wrist_before))
        self.assertGreater(moved, 0.4 * swing, f"sleeve moved {moved:.2f} m for a {swing:.2f} m wrist swing")
        changed = float((np.abs(down[..., 3].astype(int) - up[..., 3].astype(int)) > 25).mean())
        self.assertGreater(changed, 0.002, "lowering the arms changed nothing in the render")

    def test_the_garment_lands_on_the_fitted_body(self):
        """The whole point: it must sit where the person is in the frame."""
        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        fitter.fit(_landmarks_centred_at(200.0, 150.0))
        rgba = self.renderer.render(fitter, (300, 400))
        ys, xs = np.nonzero(rgba[..., 3] > 25)
        self.assertGreater(len(xs), 100)
        self.assertLess(abs(xs.mean() - 200.0), 60.0, f"garment centred at x={xs.mean():.0f}, body at 200")
        self.assertLess(abs(ys.mean() - 150.0), 80.0, f"garment centred at y={ys.mean():.0f}, torso at ~150")

    def test_the_garment_scales_with_an_unusually_sized_body(self):
        """A garment must cover the same fraction of a taller body, not the same absolute size.

        `align_to_body` only translates, so this is reproduced synthetically
        rather than depending on an external checkpoint.
        """
        import torch

        from tryon.body.fitter import BodyFitter

        neutral = BodyFitter()
        with torch.no_grad():
            neutral_body = neutral._forward().vertices[0].numpy()
        neutral_height = neutral_body[:, 1].max() - neutral_body[:, 1].min()
        neutral_mesh = self.renderer.posed_mesh(neutral)
        neutral_coverage = (neutral_mesh[:, 1].max() - neutral_mesh[:, 1].min()) / neutral_height

        tall = BodyFitter()
        with torch.no_grad():
            tall.betas[0, 0] = 3.0  # a real, large but not implausible shape component
            tall_body = tall._forward().vertices[0].numpy()
        tall_height = tall_body[:, 1].max() - tall_body[:, 1].min()
        self.assertGreater(tall_height, neutral_height * 1.05, "fixture assumption: betas should change height")
        tall_mesh = self.renderer.posed_mesh(tall)
        tall_coverage = (tall_mesh[:, 1].max() - tall_mesh[:, 1].min()) / tall_height

        self.assertLess(
            abs(tall_coverage - neutral_coverage),
            0.05,
            f"neutral body coverage {neutral_coverage:.2f} vs taller body {tall_coverage:.2f} "
            "- the garment should cover the same FRACTION of both",
        )

    def test_it_stays_within_a_measured_frame_budget(self):
        """A floor under measured numbers, not a guess.

        900 ms is about 3x the noisiest quiet-machine median of the compiled
        tile loop and still catches a regression to the uncompiled path.
        """
        import statistics
        import time

        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        self.renderer.render(fitter, (240, 320))
        times = []
        for _ in range(5):
            start = time.perf_counter()
            self.renderer.render(fitter, (240, 320))
            times.append(time.perf_counter() - start)
        per_frame = statistics.median(times)
        self.assertLess(per_frame, 0.9, f"{per_frame*1000:.0f} ms a frame at 320x240 (median of 5)")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class GarmentRendererDynamicsTest(unittest.TestCase):
    """The cloth in the renderer: `dynamics=True` on a real checkpoint and body."""

    @classmethod
    def setUpClass(cls):
        cls.renderer = GarmentRenderer(GARMENT, dynamics=True)

    def test_settles_without_nan_and_differs_from_rigid_target(self):
        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        rigid_renderer = GarmentRenderer(GARMENT, dynamics=False)
        rigid_target = rigid_renderer.posed_mesh(fitter)

        mesh = None
        for _ in range(30):
            mesh = self.renderer.posed_mesh(fitter)
        self.assertFalse(np.isnan(mesh).any())

        # The cloth step must do something, not fall back to rigid skinning.
        self.assertGreater(float(np.mean(np.linalg.norm(mesh - rigid_target, axis=1))), 1e-4)

        # And must not fly off: bounded relative to the garment's size.
        rigid_diagonal = np.linalg.norm(rigid_target.max(axis=0) - rigid_target.min(axis=0))
        mesh_diagonal = np.linalg.norm(mesh.max(axis=0) - mesh.min(axis=0))
        self.assertLess(mesh_diagonal, rigid_diagonal * 3.0)

    def test_reacts_smoothly_to_a_sudden_pose_change(self):
        """A sudden pose change across calls gives a bounded, non-exploding response."""
        import time

        import torch

        from tryon.body.fitter import BodyFitter

        renderer = GarmentRenderer(GARMENT, dynamics=True)
        fitter = BodyFitter()
        renderer.posed_mesh(fitter)  # settle at rest first

        with torch.no_grad():
            fitter.body_pose[0, 15 * 3 + 2] = -1.2
            fitter.body_pose[0, 16 * 3 + 2] = 1.2

        previous = renderer.posed_mesh(fitter)
        max_step_displacement = 0.0
        for gap in (0.01, 0.02, 0.05, 0.1, 0.2):
            time.sleep(gap)
            current = renderer.posed_mesh(fitter)
            self.assertFalse(np.isnan(current).any())
            step_displacement = float(np.max(np.linalg.norm(current - previous, axis=1)))
            max_step_displacement = max(max_step_displacement, step_displacement)
            previous = current

        # A generous bound that only rules out an unbounded spring blow-up.
        garment_scale = np.linalg.norm(previous.max(axis=0) - previous.min(axis=0))
        self.assertLess(max_step_displacement, garment_scale * 2.0)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class ScanPoseBindingTest(unittest.TestCase):
    """The scanned coat has hanging arms; bound on a body posed like the scan
    (two-pass, see `scan_pose.ScanPoseBinder.bind`) the sleeves follow the arms."""

    @classmethod
    def setUpClass(cls):
        import torch

        from tryon.body.fitter import BodyFitter

        cls.fitter = BodyFitter()  # zero pose = T-pose: arms straight out
        cls.renderer = GarmentRenderer(GARMENT, dynamics=False)
        cls.mesh = cls.renderer.posed_mesh(cls.fitter)
        with torch.no_grad():
            cls.joints = cls.fitter._forward().joints[0].numpy()

    def _sleeve_angle(self, side):
        shoulder, wrist = {"L": (16, 20), "R": (17, 21)}[side]
        weights = self.renderer._binding.weights
        arm_ids = {"L": [16, 18, 20], "R": [17, 19, 21]}[side]
        sleeve = weights[:, arm_ids].sum(axis=1) > 0.5
        points = self.mesh[sleeve]
        axis = np.linalg.svd(points - points.mean(axis=0))[2][0]
        arm = self.joints[wrist] - self.joints[shoulder]
        arm /= np.linalg.norm(arm)
        return float(np.degrees(np.arccos(abs(axis @ arm)))), int(sleeve.sum())

    def test_sleeves_follow_the_arms_on_a_t_pose(self):
        for side in ("L", "R"):
            angle, count = self._sleeve_angle(side)
            self.assertGreater(count, 300, f"{side}: only {count} sleeve vertices bound to the arm")
            self.assertLess(angle, 35.0, f"{side} sleeve axis {angle:.1f} degrees off the arm")

    def test_the_body_panels_are_not_torn_by_the_arms(self):
        torn = self.renderer.torn_faces(self.mesh)
        self.assertLess(int(torn.sum()), 600, f"{torn.sum()} torn faces on a T-pose")

    def test_the_binding_reproduces_the_scan_pose(self):
        """Zero-frame rest re-posed to the bind pose returns the aligned scan.

        Checked before the renderer scales the binding to the wearer."""
        from tryon.garment.skinning import SkinnedGarment

        bind_fitter, _theta, aligned, sleeve = self.renderer.scan_binder.bind(self.fitter)
        binding = SkinnedGarment.build_in_pose(aligned, bind_fitter, sleeve=sleeve)
        back = binding.apply(bind_fitter)
        self.assertLess(float(np.abs(back - aligned).max()), 1e-6)


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class TornFaceCullingTest(unittest.TestCase):
    """A face torn open by the solver would paint blobs the size of the tear; those splats are skipped."""

    def test_the_default_threshold_keeps_a_stretch_of_150_percent_and_drops_one_of_250_percent(self):
        """Uniform stretch: +150% is kept, +250% is not."""
        renderer = GarmentRenderer(GARMENT, dynamics=False)
        self.assertEqual(GarmentRenderer.TORN_FACE_STRAIN, 2.0)
        rest = renderer.rest_vertices
        centre = rest.mean(axis=0)
        self.assertFalse(renderer.torn_faces(centre + (rest - centre) * 2.5).any())
        self.assertTrue(renderer.torn_faces(centre + (rest - centre) * 3.5).all())

    def test_the_compiled_placement_matches_the_numpy_oracle(self):
        """`splats` (numba kernel) matches `_splats_numpy`."""
        renderer = GarmentRenderer(GARMENT, dynamics=False)
        vertices = renderer.rest_vertices.copy()
        vertices[100] += np.array([1.0, 0.0, 0.0])  # include a torn face in the comparison
        fast = renderer.splats(vertices)
        slow = renderer._splats_numpy(vertices)
        for key in fast:
            np.testing.assert_allclose(fast[key], slow[key], rtol=0, atol=1e-12, err_msg=key)

    def test_an_intact_mesh_loses_nothing(self):
        renderer = GarmentRenderer(GARMENT, dynamics=False)
        splats = renderer.splats(renderer.rest_vertices)
        self.assertEqual(len(splats["xyz"]), len(renderer.face_of))
        self.assertFalse(renderer.torn_faces(renderer.rest_vertices).any())

    def test_a_pulled_vertex_removes_exactly_its_faces_splats(self):
        renderer = GarmentRenderer(GARMENT, dynamics=False)
        vertices = renderer.rest_vertices.copy()
        victim = 100
        vertices[victim] += np.array([1.0, 0.0, 0.0])  # a metre sideways: every edge to it is torn
        torn = renderer.torn_faces(vertices)
        touching = (renderer.faces == victim).any(axis=1)
        np.testing.assert_array_equal(torn, touching)
        splats = renderer.splats(vertices)
        expected = int((~touching[renderer.face_of]).sum())
        self.assertEqual(len(splats["xyz"]), expected)
        self.assertLess(expected, len(renderer.face_of))


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class SubsampleTest(unittest.TestCase):
    """Thinning the checkpoint for speed; see GarmentRenderer's docstring."""

    def test_it_keeps_the_requested_fraction(self):
        full = GarmentRenderer(GARMENT)
        half = GarmentRenderer(GARMENT, subsample=0.5)
        self.assertEqual(len(half.local_xyz), int(len(full.local_xyz) * 0.5))

    def test_the_same_seed_is_reproducible(self):
        a = GarmentRenderer(GARMENT, subsample=0.3, seed=7)
        b = GarmentRenderer(GARMENT, subsample=0.3, seed=7)
        np.testing.assert_array_equal(a.local_xyz, b.local_xyz)

    def test_it_still_renders_something(self):
        from tryon.body.fitter import BodyFitter

        renderer = GarmentRenderer(GARMENT, subsample=0.2)
        fitter = BodyFitter()
        rgba = renderer.render(fitter, (240, 320))
        coverage = float((rgba[..., 3] > 25).mean())
        self.assertGreater(coverage, 0.02, f"garment covered {coverage:.1%} of the frame")

    def test_a_fraction_outside_zero_one_is_refused(self):
        with self.assertRaises(ValueError):
            GarmentRenderer(GARMENT, subsample=1.0)
        with self.assertRaises(ValueError):
            GarmentRenderer(GARMENT, subsample=0.0)


def _landmarks_centred_at(x, y, scale=1.0):
    from tests.test_body_fitter import _landmarks

    points = _landmarks(scale=scale)
    from tryon.body.landmarks import JOINT_MAP

    visible = [index for index in JOINT_MAP if points[index, 2] > 0]
    centre = points[visible, :2].mean(axis=0)
    points[:, 0] += x - centre[0]
    points[:, 1] += y - centre[1] + 100  # torso sits above the landmark centroid
    return points


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(GARMENT.is_dir() and (GARMENT / "stage3" / "ckpt" / "net.pt").exists(), "no garment checkpoint")
class AppearanceRenderTest(unittest.TestCase):
    """`GarmentRenderer(appearance=True)` renders a frame that differs from static shading but agrees in coverage."""

    def test_shaded_render_differs_in_colour_not_in_coverage(self):
        import torch

        from tryon.body.fitter import BodyFitter

        fitter = BodyFitter()
        with torch.no_grad():
            fitter.scale.fill_(180.0)
            fitter.translation[:] = torch.tensor([[160.0, 130.0]])
        static = GarmentRenderer(GARMENT, subsample=0.35, dynamics=False).render(fitter, (240, 320))
        shaded = GarmentRenderer(GARMENT, subsample=0.35, dynamics=False, appearance=True).render(fitter, (240, 320))
        covered = (static[..., 3] > 25) & (shaded[..., 3] > 25)
        self.assertGreater(covered.mean(), 0.02)
        overlap = ((static[..., 3] > 25) == (shaded[..., 3] > 25)).mean()
        # The network also moves the splats, so covered pixels differ by a few percent.
        self.assertGreater(overlap, 0.90, f"coverage agreement {overlap:.3f}")
        colour_change = np.abs(static[covered][:, :3].astype(float) - shaded[covered][:, :3].astype(float)).mean()
        self.assertGreater(colour_change, 2.0, "the network should change the colours")
        self.assertLess(colour_change, 80.0, "but not beyond recognition")



@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class FullSplatsTest(unittest.TestCase):
    """`splats(full=True)` (every Gaussian, torn ones masked by "keep") draws
    the same image in the GL rasteriser as the gathered subset."""

    def test_same_image_as_the_gathered_subset(self):
        import torch

        from tryon.rendering.splat_gl import render_splats_gl
        from tryon.body.fitter import BodyFitter
        from tryon.garment.renderer import GarmentRenderer

        renderer = GarmentRenderer(GARMENT, dynamics=False)
        fitter = BodyFitter()
        with torch.no_grad():
            fitter.scale.fill_(300.0)
            fitter.translation.copy_(torch.tensor([320.0, 450.0]).reshape(fitter.translation.shape))
            fitter.body_pose[0, 47] = -1.2  # a raised arm tears some faces
        mesh = renderer.posed_mesh(fitter)
        view, intrinsics, _ = renderer.camera(fitter, (870, 640))
        subset = render_splats_gl(renderer.splats(mesh), view, intrinsics, (870, 640), bgra8=True)
        full = renderer.splats(mesh, full=True)
        self.assertEqual(len(full["xyz"]), len(renderer.face_of))
        np.testing.assert_array_equal(render_splats_gl(full, view, intrinsics, (870, 640), bgra8=True), subset)
