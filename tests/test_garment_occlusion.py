"""The wearer's body must hide the parts of the garment behind it.

Found on a real frame: the hooded coat's back panel rendered across the
wearer's face, because the rasteriser draws every splat and the real
body (a photo, not geometry) is not in the scene to occlude anything.
`GarmentRenderer.occluded_by_body` culls splats behind the body surface.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tryon.body.fitter import SMPLX_ROOT
from tryon.paths import garment

GARMENT = garment("00152_outer")


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
@unittest.skipUnless(GARMENT.exists(), f"no garment checkpoint at {GARMENT}")
class BodyOcclusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tryon.body.fitter import BodyFitter
        from tryon.garment.renderer import GarmentRenderer

        cls.renderer = GarmentRenderer(GARMENT, subsample=0.35, dynamics=False)
        # These pin the ray cast itself, with the WHOLE body as occluder;
        # the live default occludes only with the parts the garment leaves
        # bare (`CoveredOcclusionTest` below).
        cls.renderer.OCCLUDE_UNCOVERED_ONLY = False
        cls.fitter = BodyFitter()
        with torch.no_grad():
            cls.fitter.scale.fill_(180.0)
            cls.fitter.translation[:] = torch.tensor([[320.0, 260.0]])
        cls.size = (520, 640)
        cls.with_occlusion = cls.renderer.render(cls.fitter, cls.size, occlude_body=True)
        cls.without_occlusion = cls.renderer.render(cls.fitter, cls.size, occlude_body=False)

    def test_it_culls_a_meaningful_share_of_splats(self):
        splats = self.renderer.splats(self.renderer.posed_mesh(self.fitter))
        culled = self.renderer.occluded_by_body(splats, self.fitter).mean()
        # The back half of a closed garment is behind the body from any
        # single viewpoint, so roughly a third to a half of its splats
        # should be hidden - not none, not nearly all.
        self.assertGreater(float(culled), 0.2, f"only {culled:.0%} of splats culled")
        self.assertLess(float(culled), 0.8, f"{culled:.0%} of splats culled - the body cannot hide that much")

    def _behind_the_body(self, splats, pixel_radius=4.0):
        """Per splat: deeper than the front-most body vertex within
        `pixel_radius` px of THAT splat by more than the occlusion margin
        (the depth-buffer rule's own definition), whether it projects
        inside the body silhouette, and whether it projects DEEP inside
        it (every pixel within 2 * `pixel_radius` covered by the body) -
        where the depth rule and a ray cast cannot disagree."""
        import cv2
        from scipy.spatial import cKDTree

        with torch.no_grad():
            out = self.fitter._forward()
            body = out.vertices[0].numpy()
            body_px = self.fitter._project(out.vertices[0]).numpy()
            splat_px = self.fitter._project(torch.as_tensor(splats["xyz"], dtype=torch.float32)).numpy()
        distance, nearest = cKDTree(body_px).query(splat_px, k=16, distance_upper_bound=pixel_radius)
        padded_z = np.concatenate([body[:, 2], [-np.inf]])
        front_z = padded_z[np.minimum(nearest, len(body))].max(axis=1)
        inside = np.isfinite(front_z)
        behind = (splats["xyz"][:, 2] < front_z - 0.03) & inside
        height, width = self.size
        cover = np.zeros((height, width), np.uint8)
        px = np.round(body_px).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
        cover[px[ok, 1], px[ok, 0]] = 1
        cover = cv2.dilate(cover, np.ones((5, 5), np.uint8))  # body vertices sit 1-2 px apart
        interior = cv2.erode(cover, np.ones((int(4 * pixel_radius) + 1,) * 2, np.uint8)).astype(bool)
        sp = np.round(splat_px).astype(int)
        ok = (sp[:, 0] >= 0) & (sp[:, 0] < width) & (sp[:, 1] >= 0) & (sp[:, 1] < height)
        deep = np.zeros(len(splat_px), bool)
        deep[ok] = interior[sp[ok, 1], sp[ok, 0]]
        return behind, inside, deep

    def test_nothing_behind_the_body_survives(self):
        """The geometric guarantee: after culling, no splat projecting
        well inside the body's silhouette sits behind the body's front
        surface. Until 25 Sep 2026 this was stated on a face window, on
        the belief that 00152_outer was a hooded coat whose hood-back
        showed through the face; the checkpoint's release video shows an
        open plaid shirt, and with the scan-frame binding nothing
        projects onto the face at all. The property that matters is the
        same for the torso: the shirt's back panel must not paint through
        the chest. Since 26 Sep 2026 the rule is a ray cast and the
        depth-buffer definition here is the independent oracle; the two
        legitimately differ within a few pixels of the silhouette (the
        buffer's radius reads body depth where the body is not - see
        `occluded_by_body`), so the check is over the silhouette's
        interior."""
        splats = self.renderer.splats(self.renderer.posed_mesh(self.fitter))
        keep = ~self.renderer.occluded_by_body(splats, self.fitter)
        behind, inside, deep = self._behind_the_body(splats)
        survivors = int((keep & behind & deep).sum())
        self.assertGreater(int((behind & deep).sum()), 1000, "fixture assumption: the back panel projects well inside the torso")
        self.assertLessEqual(survivors, 0.005 * (inside & deep).sum(), f"{survivors} splats behind the body survived")
        # And the culling did real work: the back panel of an open shirt
        # projects onto the torso from behind.
        share = float((inside & ~keep).sum() / max(inside.sum(), 1))
        self.assertGreater(share, 0.2, f"only {share:.0%} of the splats over the body were culled")

    def test_a_splat_just_outside_the_surface_facing_the_camera_is_kept(self):
        """The case the depth buffer got wrong on the profile cameras: a
        splat a centimetre outside the body, on a surface that faces the
        camera, is visible whatever body pixels lie within 4 px of it.
        Measured on the annotated profiles at frame 3: the buffer culled
        9,279 / 7,656 visible splats (10 / 8 % of all) on 0028 / 0076 -
        41 % of the near sleeve, 2,053 of those 5,085 wrongly."""
        import trimesh

        with torch.no_grad():
            body = self.fitter._forward().vertices[0].numpy().astype(np.float64)
        mesh = trimesh.Trimesh(body, np.asarray(self.fitter.model.faces), process=False)
        normals = np.asarray(mesh.vertex_normals)
        outside = {"xyz": body + 0.01 * normals}
        from tryon.garment.renderer import CAMERA_DISTANCE

        to_camera = np.array([0.0, 0.0, CAMERA_DISTANCE]) - body
        facing = (normals * to_camera).sum(axis=1) > 0.3 * np.linalg.norm(to_camera, axis=1)
        ray_kept = ~self.renderer.occluded_by_body(outside, self.fitter)
        # Not 100 %: a camera-facing surface can still sit behind another
        # body part (fingers behind fingers, the chin's underside); measured 97.1 %.
        self.assertGreater(float(ray_kept[facing].mean()), 0.95, f"ray cast kept {ray_kept[facing].mean():.1%} of the camera-facing splats")
        # ...and the ones on the far side are still hidden where the body is
        # thicker than the 3 cm margin (measured 62.5 %: fingers, ears and
        # the edges of the limbs are thinner than that and survive by design).
        away = (normals * to_camera).sum(axis=1) < -0.3 * np.linalg.norm(to_camera, axis=1)
        self.assertGreater(float((~ray_kept)[away].mean()), 0.5, f"only {(~ray_kept)[away].mean():.1%} of the far-side splats culled")

    def test_the_torso_alpha_drops_but_the_front_stays(self):
        """Over the torso the back panel is culled and the front panels
        are drawn, so alpha drops without vanishing: the front of an open
        shirt is physically in front of the body."""
        with torch.no_grad():
            out = self.fitter._forward()
            chest = out.joints[0, 9]  # spine3
            cx, cy = self.fitter._project(chest[None]).numpy()[0]
        rows = slice(max(int(cy) - 15, 0), int(cy) + 15)
        cols = slice(max(int(cx) - 40, 0), int(cx) + 40)
        before = self.without_occlusion[rows, cols, 3].astype(float).mean()
        after = self.with_occlusion[rows, cols, 3].astype(float).mean()
        self.assertGreater(before, 40.0, "fixture assumption: the shirt covers the chest without occlusion")
        self.assertLess(after, before, f"torso alpha did not drop {before:.0f} -> {after:.0f}")
        self.assertGreater(after, 10.0, "the front panels should still be drawn")

    def test_the_ray_cast_is_cheap(self):
        """~10 ms at 93k splats measured; the budget is a render (~360 ms)."""
        import time

        splats = self.renderer.splats(self.renderer.posed_mesh(self.fitter))
        self.renderer.occluded_by_body(splats, self.fitter)  # warm
        t0 = time.perf_counter()
        self.renderer.occluded_by_body(splats, self.fitter)
        self.assertLess(time.perf_counter() - t0, 0.15)

    def test_the_front_of_the_garment_is_untouched(self):
        # Total garment coverage must remain substantial - occlusion trims
        # the back, it must not erase the coat.
        coverage = (self.with_occlusion[..., 3] > 25).mean()
        self.assertGreater(float(coverage), 0.02)


@unittest.skipUnless(SMPLX_ROOT.is_dir() and GARMENT.is_dir(), "needs SMPL-X and the 00152_outer checkpoint")
class CoveredOcclusionTest(unittest.TestCase):
    """Only the body the garment leaves bare occludes it."""

    def test_the_head_still_occludes_and_the_sleeved_arms_do_not(self):
        import torch

        from tryon.body.fitter import BodyFitter
        from tryon.garment.renderer import GarmentRenderer

        renderer = GarmentRenderer(GARMENT, subsample=0.35, dynamics=False)
        fitter = BodyFitter()
        with torch.no_grad():
            fitter.scale.fill_(180.0)
            fitter.translation[:] = torch.tensor([[320.0, 260.0]])
        renderer.posed_mesh(fitter)
        covered = renderer._covered_body
        self.assertIsNotNone(covered)
        body = np.asarray(fitter.vertices())
        top = body[:, 1] > np.quantile(body[:, 1], 0.93)  # the head
        self.assertLess(covered[top].mean(), 0.2, "the head is bare, so it must still occlude")
        self.assertGreater(covered.mean(), 0.2, "the torso and arms inside the shirt are covered")


if __name__ == "__main__":
    unittest.main()
