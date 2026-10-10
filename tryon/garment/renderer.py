"""Render a garment onto the tracked body.

    garment mesh --bind--> body --pose--> posed mesh --face frames--> splats
                                                    --camera--> pixels

The checkpoint stores its 93,491 Gaussians in FACE-LOCAL frames - position,
scale and rotation all relative to the mesh face each was bound to. Move
the face and the splat moves with it. That is how Gaussian Garments (Rong
et al. 2409.08189) animates a garment, and it means the mesh is the only
thing that has to be deformed: `bind.py` carries the mesh with the body,
`splat.face_frames` re-derives every face's frame from the deformed mesh,
and the splats fall out.

One thing the existing rasteriser did not need and this does: composing
the splat ROTATIONS with the face frame. For the rigid turntable catalog
the face frames never changed, so a face-local rotation and a world one
were the same thing. On a posed mesh they are not - a sleeve rotated 72
degrees carries its splats' covariance with it, or the shading on the arm
stays lit from the wrong side.

Camera: the fitter is weak-perspective (`pixel = scale * xy + translation`),
the rasteriser is perspective. They agree when the perspective camera is
placed far enough that depth variation across a body is negligible: at
distance `d` with focal `scale * d`, the projection of a point at depth
`d + dz` is off by a factor `d / (d + dz)`, under 2% for a 1 m body at
`d = 50 m`. So the garment lands exactly where the fit put the joints.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from numba import njit, prange

from tryon.garment.checkpoint import load_checkpoint, load_relaxed_rest, uv_binding
from tryon.garment.cloth import SmoothCloth, geodesic_from
from tryon.garment.raycast import build_scene, hit_distance, occluded_from
from tryon.garment.scan_pose import ScanPoseBinder
from tryon.garment.skinning import SkinnedGarment, closed_front_flag, drape_narrowing, front_closure, front_opening
from tryon.rendering.splat_cpu import face_frame, face_frames, quaternion_to_rotation, render_splats, rotation_to_quaternion

# Far enough that weak perspective and perspective agree to <2% over a
# body, near enough that float precision in the rasteriser is untroubled.
CAMERA_DISTANCE = 50.0
# The stored scales leave gaps between neighbours on a garment this dense;
# measured on the catalog builds, 1.3 closes them for the least blur.
SPLAT_DILATION = 1.3

@njit(cache=True, parallel=True)
def _place_splats_numba(vertices, faces, face_of, bary, local_xyz, local_rotation, log_scaling, log_dilation, keep_faces):
    """`tryon.rendering.splat_cpu.face_frames` and the splat placement of
    `GarmentRenderer._splats_numpy`, one face / one splat per iteration.
    Same maths, same order of operations. Splats on faces not in
    `keep_faces` are skipped (their rows are left uninitialised; the
    caller filters them out, or masks them with "keep"). Also returns the
    face frames and scales, for whoever places more splats on the same mesh
    (`AppearanceNet.world_xyz`)."""
    n_faces = len(faces)
    frames = np.empty((n_faces, 3, 3))
    scale = np.empty(n_faces)
    for f in prange(n_faces):
        a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z, scale[f] = face_frame(vertices, faces, f)
        frames[f, 0, 0], frames[f, 1, 0], frames[f, 2, 0] = a0x, a0y, a0z
        frames[f, 0, 1], frames[f, 1, 1], frames[f, 2, 1] = a1x, a1y, a1z
        frames[f, 0, 2], frames[f, 1, 2], frames[f, 2, 2] = a2x, a2y, a2z
    n = len(face_of)
    xyz = np.empty((n, 3))
    rotation = np.empty((n, 3, 3))
    scaling = np.empty((n, 3))
    for s in prange(n):
        f = face_of[s]
        if not keep_faces[f]:
            # a torn face's splat: a harmless placeholder (the caller drops
            # it, or hands it on masked out by "keep")
            for r in range(3):
                xyz[s, r] = 0.0
                scaling[s, r] = -30.0
                for c in range(3):
                    rotation[s, r, c] = 1.0 if r == c else 0.0
            continue
        sc = scale[f]
        log_sc = np.log(max(sc, 1e-20)) + log_dilation
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        b0, b1, b2 = bary[s, 0], bary[s, 1], bary[s, 2]
        for r in range(3):
            anchor = b0 * vertices[i0, r] + b1 * vertices[i1, r] + b2 * vertices[i2, r]
            xyz[s, r] = (
                frames[f, r, 0] * local_xyz[s, 0] + frames[f, r, 1] * local_xyz[s, 1] + frames[f, r, 2] * local_xyz[s, 2]
            ) * sc + anchor
            for c in range(3):
                rotation[s, r, c] = (
                    frames[f, r, 0] * local_rotation[s, 0, c]
                    + frames[f, r, 1] * local_rotation[s, 1, c]
                    + frames[f, r, 2] * local_rotation[s, 2, c]
                )
            scaling[s, r] = log_scaling[s, r] + log_sc
    return xyz, rotation, scaling, frames, scale


@njit(cache=True)
def _strain_passes(x, edges, limit, count, iterations):
    """Jacobi strain limiting in place (`GarmentRenderer._limit_strain`):
    each pass moves both ends of every over-long edge half its excess
    toward each other, averaged over each vertex's edge count; stops when
    no edge is over. Numba: 42 -> ~3 ms for 8,208 vertices, 30 passes."""
    delta = np.zeros_like(x)
    for _ in range(iterations):
        delta[:] = 0.0
        any_over = False
        for k in range(edges.shape[0]):
            a, b = edges[k, 0], edges[k, 1]
            d0, d1, d2 = x[b, 0] - x[a, 0], x[b, 1] - x[a, 1], x[b, 2] - x[a, 2]
            length = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            if length > limit[k]:
                any_over = True
                f = 0.5 * (length - limit[k]) / length
                delta[a, 0] += f * d0
                delta[a, 1] += f * d1
                delta[a, 2] += f * d2
                delta[b, 0] -= f * d0
                delta[b, 1] -= f * d1
                delta[b, 2] -= f * d2
        if not any_over:
            break
        for i in range(x.shape[0]):
            for c in range(3):
                x[i, c] += delta[i, c] / count[i]


class GarmentRenderer:
    """A garment checkpoint, bound to a body once, rendered per pose.

    `subsample`: keep only this fraction of the checkpoint's Gaussians,
    chosen once at load with a fixed seed (so a run is reproducible), not
    re-chosen per frame. Rasterising bins Gaussians into tiles and
    composites each tile's stack (see `tryon/rendering/splat_cpu.py`'s `render_splats`,
    a numba-compiled kernel throughout - both the per-tile binning and
    the compositing loop itself), so its cost tracks splat count, not
    output resolution: measured on this checkpoint (93,491 splats) at
    480x640, thinning the splats cuts render time roughly linearly since
    it drops the same fraction of tile work everywhere: 100%=358-375 ms
    (2.7 fps), 50%=218 ms, 35%=176 ms, 20%=54 ms (18.6 fps), 15%=43 ms
    (23.0 fps). This is genuinely a render-quality-for-speed trade, not a
    free win: at 25% the plaid pattern visibly thins to speckle (see the
    comparison this was measured with); at 50% it is close to
    indistinguishable at normal viewing size. Default is `None` - full
    quality, unchanged from before this existed - because that trade is
    the caller's to make, not a default to silently spend for them.
    """

    # Anchor splats at their texel's barycentric point (the reference) rather
    # than the face centre; see __init__.
    BARYCENTRIC_ANCHOR = True

    def __init__(
        self,
        checkpoint_dir,
        dilation=SPLAT_DILATION,
        subsample=None,
        seed=0,
        dynamics=None,
        render_scale=1.0,
        appearance=False,
    ):
        """`appearance`: run the checkpoint's own per-frame appearance
        network (tryon/garment/appearance.py: the shadow_net that gives the
        garment its studio shading, ~0.75 s a frame on MPS in fp16)
        instead of drawing the static spherical harmonics only."""
        checkpoint, mesh = load_checkpoint(Path(checkpoint_dir))
        self.appearance_enabled = bool(appearance)
        self._appearance_net = None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.faces = np.asarray(mesh.faces)
        local_xyz = checkpoint["xyz"].numpy().astype(np.float64)
        binding, _resolution = uv_binding(mesh, len(local_xyz))
        usable = min(len(binding), len(local_xyz))
        face_of = binding[:usable]
        local_xyz = local_xyz[:usable]
        local_rotation = quaternion_to_rotation(
            checkpoint["rotation"].numpy().astype(np.float64)[:usable]
        )
        log_scaling = checkpoint["scaling"].numpy().astype(np.float64)[:usable]
        appearance = {
            key: checkpoint[key].numpy().astype(np.float64)[:usable]
            for key in ("feature_dc", "feature_rest", "opacity")
        }
        # Where on its face each splat is anchored. The reference's stage-3
        # model (AvatarGaussianModel.get_xyz) puts a Gaussian at its UV
        # texel's BARYCENTRIC point on the face; until 25 Sep 2026 this
        # renderer used the face centre (the stage-2 convention), so the
        # ~6 splats of a face piled up at its centre - the "gaps that
        # needed 1.3x dilation" of the first renderer - and stretched front panels
        # rendered as scattered dots. The stored xyz does not compensate
        # (measured: correlation with the centre-to-texel offset ~0).
        bary = np.full((usable, 3), 1.0 / 3.0)
        if self.BARYCENTRIC_ANCHOR:
            try:
                from tryon.garment.texels import TexelBinding, read_template

                texels = TexelBinding(read_template(self.checkpoint_dir / "stage1" / "template_uv.obj"))
                same = texels.binding[:usable] == face_of
                if same.mean() > 0.99:
                    bary = np.asarray(texels.bc[:usable], np.float64)
                    bary[~same] = 1.0 / 3.0
            except (OSError, ValueError, KeyError):
                pass
        # which of the checkpoint's Gaussians this renderer holds (None: all):
        # the shading network answers for every one of them
        self._subset = None
        if subsample is not None:
            fraction = float(subsample)
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"subsample must be in (0, 1), got {fraction}")
            count = int(usable * fraction)
            if self.SUBSAMPLE_BY_IMPORTANCE:
                # the splats that cover the most opaque area first: opacity
                # times the footprint (the two largest scales' product)
                opacity = 1.0 / (1.0 + np.exp(-appearance["opacity"].ravel()))
                widest = np.sort(np.exp(log_scaling), axis=1)[:, -2:]
                keep = np.argsort(-(opacity * widest[:, 0] * widest[:, 1]), kind="stable")[:count]
            else:
                keep = np.random.default_rng(seed).choice(usable, count, replace=False)
            keep.sort()
            self._subset = keep
            face_of = face_of[keep]
            bary = bary[keep]
            local_xyz = local_xyz[keep]
            local_rotation = local_rotation[keep]
            log_scaling = log_scaling[keep]
            appearance = {key: value[keep] for key, value in appearance.items()}
        # the opacity the rasteriser blends with (`_prepare` takes it from
        # here instead of a sigmoid of the 93k logits every frame)
        appearance["opacity_alpha"] = 1.0 / (1.0 + np.exp(-appearance["opacity"].ravel()))
        self.face_of = face_of
        self.bary = bary
        self.local_xyz = local_xyz
        self.local_rotation = local_rotation
        self.log_scaling = log_scaling
        self.appearance = appearance
        # Contiguous copies for the compiled placement kernel.
        self._faces_c = np.ascontiguousarray(self.faces, dtype=np.int64)
        self._local_xyz_c = np.ascontiguousarray(local_xyz, dtype=np.float64)
        self._bary_c = np.ascontiguousarray(bary, dtype=np.float64)
        self._local_rotation_c = np.ascontiguousarray(local_rotation, dtype=np.float64)
        self._log_scaling_c = np.ascontiguousarray(log_scaling, dtype=np.float64)
        self.dilation = float(dilation)
        self.rest_vertices = np.asarray(mesh.vertices, np.float64)
        self.scan_binder = ScanPoseBinder(self.rest_vertices, self.faces, self.checkpoint_dir)
        self._binding = None
        self._bound_to = None
        # Garment motion: `LightCloth` springs toward the skinned
        # pose plus body collision. False: pure skinning, for A/B
        # comparison. None: the class default `DYNAMICS` (the CLI's --rigid).
        self.dynamics_enabled = self.DYNAMICS if dynamics is None else bool(dynamics)
        # Fraction of the output size the splats are rasterised at; see `render`.
        self.render_scale = float(render_scale)
        self._light = None
        self._last_tick = None
        self._frames = None  # (vertices, frames, scale) of the last `splats`
        self._strain = None  # (key, edges, limit, count) of `_limit_strain`, per binding
        self._torn_limit = None  # (key, per-face-edge tear length) of `torn_faces`, per binding

    # `subsample` keeps the splats with the largest opacity * footprint
    # (True) or a uniform random subset (False).
    SUBSAMPLE_BY_IMPORTANCE = False
    # the cloth model class: SmoothCloth (parameters searched against the authors' ContourCraft
    # simulation: same distance and IoU as LightCloth, about a third of the jitter), or LightCloth
    CLOTH = SmoothCloth

    # Cloth motion on by default (see `__init__`'s `dynamics`).
    DYNAMICS = True
    # Never draw the faces straddling the sleeve label (see `bind`).
    CUT_SLEEVE_SEAM = True
    # Off since the fit runs in a normalised frame (`bodyfit.WORK_SCALE`):
    # the wearer's betas now settle near zero, so the chest ratio compares
    # an average body with the scan subject rather than measuring the
    # wearer. Reference frame 00152_outer: garment IoU 0.73 with, 0.75
    # without (erased-photo PSNR 14.05 vs 14.00 dB).
    WIDEN_TO_WEARER = False
    # How the wearer's size ratios scale the garment: `ScanPoseBinder.SIZE_MODES`.
    SIZE_MODE = "current"
    # Bind to the garment's RELAXED rest shape instead of the raw template:
    # None (the template) or "mean" (template + the garment's stored offset,
    # `stage1/relaxed_rest.npz`, silently the template when the file is
    # absent). `relaxed_offset` set on an instance overrides the file.
    RELAXED_REST = None
    RELAXED_SCALE = 1.0  # fraction of the stored offset applied
    relaxed_offset = None

    # Bring the two front panels of an open-front garment together at the body's midline
    # (`skinning.front_closure`, once at bind, in the zero-pose frame: skinning carries it). None = off (the
    # garment as scanned); "auto": only for a garment whose wearer wears it closed, which one frame cannot tell, so
    # it is a per-garment flag file, `stage1/closed_front.json` (`skinning.closed_front_flag`; absent = off);
    # "always": every garment (for evaluation).
    CLOSE_FRONT = "auto"
    CLOSE_SEAM = 0.01  # m: gap left between the two inner edges
    CLOSE_BLEND = 0.2  # m: lateral distance over which the pull fades out

    # Narrow the loose lower torso toward the body's centre line by this fraction of its distance from it, scaled
    # by how far it hangs off the body (`skinning.drape_narrowing`). None = off. 0.15 hurt 00152_outer (-0.005), 0.08 with the long-range attachment is the default.
    DRAPE_NARROW = 0.08
    DRAPE_START = 0.3  # of the way from the top to the hem where the narrowing starts, and the span it ramps up over
    DRAPE_SPAN = 0.4

    # Open the lower front of an open-front garment (`skinning.front_opening`): the inner edge of each front panel
    # is moved out to this fraction of the half-width at its height, from `FRONT_OPEN_START` of the way from the
    # top to the hem over `FRONT_OPEN_SPAN`. Measured: the real shirt's front gap is 0.3 of its width below the
    # chest, ours 0.1-0.25 (docs/figures/torso-width.jpg). None = off; not applied to a garment worn closed.
    FRONT_OPEN = None
    FRONT_OPEN_START = 0.35
    FRONT_OPEN_SPAN = 0.4

    def closes_front(self):
        if self.CLOSE_FRONT == "always":
            return True
        return self.CLOSE_FRONT == "auto" and closed_front_flag(self.checkpoint_dir)

    def bind(self, fitter):
        """Skin the garment to this fitter's body at rest. Once per person.

        `SkinnedGarment`, not nearest-vertex binding: the latter tore 332
        faces up to 58x when the arm rose and the splats on them ballooned
        into a smear across the chest. See `bind.SkinnedGarment`.

        The garment is scaled by this body's height relative to a neutral
        (beta=0) SMPL-X body before alignment. `align_to_body` only
        translates - it matches the garment's top to the body's top but
        never touches the garment's own absolute size - so on a body
        whose fitted shape is taller or shorter than average, a garment
        sized for the average body covers a smaller or larger FRACTION of
        it. Measured on a real annotated frame (camera 0028, a profile
        view where beta drifts furthest from zero - see docs/PLAN.md):
        fitted body height 2.49 vs the neutral 1.72, garment coverage
        59.5% of body height on a neutral body but only 38.2% on this
        one - visibly a coat floating near the shoulders rather than
        reaching the hip. Scaling the garment by height ratio (1.45x
        here) restores coverage to 59.5%, matching the neutral case
        exactly, because it is the same proportional garment on a
        proportionally different body - not a special case for a
        mis-fit, but the same scaling any correctly-sized garment needs
        for a genuinely taller or shorter wearer.
        """
        # Bind on a body posed like the garment's SCAN, not like this frame.
        # The scanned coat has its arms hanging; bound to the T-posed body
        # of a first frame, its sleeves sat beside the hips and took torso
        # and leg weights (measured: sleeve axes 57 / 67 degrees off the
        # arms, sleeves hanging to the hip - see SkinnedGarment.build_in_pose).
        # The scan's shoulder angle is not stored with the checkpoint; the
        # angles used, and how they were chosen, are on `SCAN_BIND_ANGLE`.
        # A grid search on median garment-to-body distance was tried first
        # and picked 80 degrees (4.3 cm vs 4.9 at 0-45) - the pose where
        # the sleeves fit the arms best is also the one where the hanging
        # arms contaminate the body panels most, which is why the binding
        # is two-pass rather than one angle.
        bind_fitter, _, aligned, sleeve = self.scan_binder.bind(fitter)
        self.sleeve_vertices = sleeve
        self._binding = SkinnedGarment.build_in_pose(aligned, bind_fitter, sleeve=sleeve)
        if self.RELAXED_REST:
            offset = self.relaxed_offset
            if offset is None:
                offset = load_relaxed_rest(self.checkpoint_dir, len(self.rest_vertices))
            if offset is not None:
                self._binding = self._binding.relaxed(offset * self.RELAXED_SCALE)
        if self.closes_front():
            self._binding = self._binding.moved(front_closure(self._binding.rest, self.faces, self.CLOSE_SEAM, self.CLOSE_BLEND))
        if self.FRONT_OPEN and not self.closes_front():
            self._binding = self._binding.moved(front_opening(self._binding.rest, self.faces, sleeve, self.FRONT_OPEN, self.FRONT_OPEN_START, self.FRONT_OPEN_SPAN))
        if self.DRAPE_NARROW:
            from scipy.spatial import cKDTree

            with torch.no_grad():
                bind_body = bind_fitter._forward().vertices[0].numpy().astype(np.float64)
            distance = cKDTree(bind_body).query(aligned)[0]
            offset = drape_narrowing(self._binding.rest, distance, sleeve, self.DRAPE_NARROW, start=self.DRAPE_START, span=self.DRAPE_SPAN)
            self._binding = self._binding.moved(offset)
        # Bound in the template frame, the garment is the scan subject's
        # size; scale its zero-pose rest by the wearer's height over the
        # scan body's so it fits the person in front of the camera (the
        # zero-pose body of either is, to first order, the other scaled).
        mode = self.SIZE_MODE
        height = self.scan_binder.height_ratio(fitter, bind_fitter) if self.scan_pose_used else 1.0
        # Width: measured on the authors' resize simulations (20 bodies,
        # tools/evaluate/resize_trajectories.py) a height-only resize
        # leaves 34-40% of the garment inside a short-and-wide body (the
        # camera-0004 shape) while scaling x/z by the chest ratio brings
        # wide bodies to 3-11% - and breaks narrow ones (sleeves pulled
        # into the arms, 21-43%). So the width factor is applied only
        # when the wearer is WIDER than the scan subject, never below 1.
        needs_chest = mode == "widen" or (mode == "current" and self.WIDEN_TO_WEARER)
        chest = self.scan_binder.chest_ratio(fitter, bind_fitter) if self.scan_pose_used and needs_chest else 1.0
        self._size_history = [height]
        self._size_binder = (fitter, bind_fitter)
        factors = self.scan_binder.size_factors(mode, height, chest, self.WIDEN_TO_WEARER) if self.scan_pose_used else np.ones(3)
        self._size_ratio = float(factors[1])
        self._width_ratio = float(factors[0] / max(factors[1], 1e-9))
        self._size_applied = factors
        self._binding.scale(factors)
        self._bound_to = id(fitter)
        self._covered_body = self._body_covered_by_garment(fitter) if self.OCCLUDE_UNCOVERED_ONLY else None

        # The scan fused the hanging sleeves to the body where they touched;
        # the faces that straddle the sleeve label are that false contact and
        # tear whatever is done (328 torn edges even rigid, all on the seam),
        # so they are never drawn (`torn_faces`).
        self._seam_faces = np.zeros(len(self.faces), dtype=bool)
        if self.CUT_SLEEVE_SEAM:
            on_sleeve = sleeve[self.faces]
            self._seam_faces = on_sleeve.any(axis=1) & ~on_sleeve.all(axis=1)
        self._light = None
        self._last_tick = None
        self._strain = self._torn_limit = None

    @property
    def scan_pose_used(self):
        """Whether the binding used the scan-frame pose cached with the checkpoint."""
        return self.scan_binder.scan_pose_used

    def reset_motion(self, fitter):
        """Restart the cloth at rest on the fitter's current pose: after a
        refit the pose jumps, and the springs would otherwise swing the
        cloth from the old pose with the old velocities."""
        self._light = None
        self._last_tick = None

    # Strain limiting on the posed fine mesh (Provot 1995): edges longer
    # than (1 + STRAIN_LIMIT) x rest are shortened, STRAIN_ITERATIONS
    # Jacobi passes. None = off. See `_limit_strain`.
    # 5 % / 30 passes: on 00190_upper the posed surface over rest fell from
    # 1.10-1.31 to 1.05-1.13 (cloth solver on); against the registration
    # IoU 0.652 -> 0.659 and 0.587 -> 0.594 (00190_upper, 00152_outer).
    # ~60 ms a frame.
    STRAIN_LIMIT = 0.05
    STRAIN_ITERATIONS = 30

    def _limit_strain(self, vertices):
        """Pull stretched edges back toward the garment's physical rest.

        Skinning from the scan's pose stretched the fabric: the posed
        garment's 3D surface ran 7-24 % over its rest area on 00190_upper
        as the arms moved (the cloth solver 9-31 %), which drew the garment
        1.2-1.3x the real one in the image against the registration's
        0.9-1.0x, and tore the armpit seam. Real fabric barely stretches."""
        if self.STRAIN_LIMIT is None:
            return vertices
        key = (id(self._binding), self.STRAIN_LIMIT)
        if self._strain is None or self._strain[0] != key:
            edges = self.scan_binder.edges
            rest = self._binding.physical_rest
            length = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
            limit = length * (1.0 + self.STRAIN_LIMIT)
            count = np.maximum(np.bincount(edges.ravel(), minlength=len(rest)).astype(np.float64), 1.0)
            self._strain = (key, edges, limit, count)
        _, edges, limit, count = self._strain
        x = np.array(vertices, np.float64, copy=True)
        _strain_passes(x, edges, limit, count, int(self.STRAIN_ITERATIONS))
        return x

    def posed_mesh(self, fitter, dt=None):
        """The garment on the fitter's current pose (strain-limited when
        STRAIN_LIMIT is set). `dt`: seconds to advance the cloth solver by
        (default: the wall-clock time since the previous call - the live
        path); an offline caller replaying a sequence passes the sequence's
        own frame interval."""
        return self._limit_strain(self._posed_mesh(fitter, dt))

    # A shirt rides up when the arms rise (00152_outer: the hem is 10-12 % of
    # the shirt's height too low on the frames with the hands above the
    # shoulders, correlation 0.85 between the two over the clip; leg weights
    # of the hem were not it). Lift the garment's body vertices along the
    # spine by RIDE_UP x the garment's height x how far the arms are raised x
    # how far down the garment the vertex hangs (0 at the collar, 1 at the
    # hem). None = off. The arms' raise is the mean elevation angle of the
    # shoulder-to-wrist directions, a smoothstep between RIDE_FROM and RIDE_TO
    # degrees (-90 = hanging, 0 = horizontal, +90 = straight up).
    RIDE_UP = None
    RIDE_FROM, RIDE_TO = -20.0, 70.0
    _RIDE_JOINTS = (0, 12, 16, 17, 20, 21)  # pelvis, neck, shoulders, wrists

    def arm_raise(self, fitter):
        """(raise in [0, 1], unit spine direction) of the fitter's current pose."""
        joints = fitter.joints(indices=list(self._RIDE_JOINTS)).detach().numpy().astype(np.float64)
        spine = joints[1] - joints[0]
        up = spine / max(float(np.linalg.norm(spine)), 1e-9)
        elevations = []
        for shoulder, wrist in ((2, 4), (3, 5)):
            arm = joints[wrist] - joints[shoulder]
            elevations.append(np.degrees(np.arcsin(np.clip(float(arm @ up) / max(float(np.linalg.norm(arm)), 1e-9), -1.0, 1.0))))
        t = np.clip((float(np.mean(elevations)) - self.RIDE_FROM) / (self.RIDE_TO - self.RIDE_FROM), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t)), up

    def _ride_weights(self):
        """Per vertex: 0 at the garment's top, 1 at its hem, 0 on the sleeves (cached per binding)."""
        key = id(self._binding)
        if getattr(self, "_ride", None) is None or self._ride[0] != key:
            rest = self._binding.physical_rest
            body = ~np.asarray(self.sleeve_vertices, bool)
            top, bottom = float(np.percentile(rest[body, 1], 98)), float(np.percentile(rest[body, 1], 2))
            weight = np.clip((top - rest[:, 1]) / max(top - bottom, 1e-9), 0.0, 1.0) * body
            self._ride = (key, weight, top - bottom)
        return self._ride[1], self._ride[2]

    def _ride_up(self, vertices, fitter):
        raised, up = self.arm_raise(fitter)
        if raised <= 0.0:
            return vertices
        weight, height = self._ride_weights()
        return vertices + (self.RIDE_UP * height * raised * weight)[:, None] * up

    # What a long-range attachment (`LightCloth.LRA`) hangs the garment from: "top" the highest
    # TOP_RING_FRACTION of the body (non-sleeve) vertices at rest (measured: a hem never gets farther from them
    # than its rest path, 1.00-1.03, so it does nothing); "sleeve" the sleeve's border with the body panels
    # (the armhole: the arm carries it up, the panel below it may not be farther than its rest path from it,
    # so a raised arm lifts the hem). Only body (non-sleeve) vertices are limited.
    LRA_ANCHORS = "sleeve"
    TOP_RING_FRACTION = 0.03

    def _long_range_anchors(self):
        """(anchor_of, rest geodesic) per vertex from the anchors, over the garment's physical rest; once per binding."""
        key = (id(self._binding), self.LRA_ANCHORS)
        if getattr(self, "_anchors", None) is None or self._anchors[0] != key:
            rest = self._binding.physical_rest
            sleeve = np.asarray(self.sleeve_vertices, bool)
            edges = self.scan_binder.edges
            if self.LRA_ANCHORS == "sleeve":
                crossing = sleeve[edges[:, 0]] != sleeve[edges[:, 1]]
                ring = np.unique(np.where(sleeve[edges[crossing, 0]], edges[crossing, 0], edges[crossing, 1]))
            else:
                ids = np.flatnonzero(~sleeve)
                ring = ids[rest[ids, 1] >= np.quantile(rest[ids, 1], 1.0 - self.TOP_RING_FRACTION)]
            distance, anchor_of = geodesic_from(len(rest), edges, rest, ring)
            anchor_of = np.where(sleeve, -1, anchor_of)  # a sleeve follows its arm
            self._anchors = (key, anchor_of, distance)
        return self._anchors[1:]

    def _freeze_size(self, fitter):
        """SIZE_MODE "frozen": the garment's scale is the median height ratio
        of the first `FREEZE_FRAMES` frames of this person, then held (one
        frame's regressed shape is noisy; the median is not)."""
        if self.SIZE_MODE != "frozen" or not self.scan_pose_used or len(self._size_history) >= self.scan_binder.FREEZE_FRAMES:
            return
        self._size_history.append(self.scan_binder.height_ratio(*self._size_binder))
        target = float(np.median(self._size_history))
        change = target / max(float(self._size_applied[1]), 1e-9)
        self._binding.scale(change)
        self._size_applied = self._size_applied * change
        self._size_ratio = float(self._size_applied[1])

    def _posed_mesh(self, fitter, dt=None):
        if self._binding is None or self._bound_to != id(fitter):
            self.bind(fitter)
        self._freeze_size(fitter)
        skin_target = self._binding.apply(fitter)
        if self.RIDE_UP:
            skin_target = self._ride_up(skin_target, fitter)
        if not self.dynamics_enabled:
            return skin_target
        now = time.perf_counter()
        if dt is None:
            dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        if self._light is None:
            self._light = self.CLOTH(self.scan_binder.edges)
            if self.CLOTH.LRA is not None:
                self._light.set_long_range(*self._long_range_anchors())
        return self._light.step(skin_target, dt, fitter.vertices(), fitter.faces)

    # A face whose edge has stretched past this factor of its rest length
    # is broken geometry (the solver tears a few dozen edges per frame at
    # the seams of the real checkpoint); a splat scales with its face, so
    # drawing it paints a smear the size of the tear. Skip those splats.
    # 2.0 since 10 Oct (1.0 before): a cropped sweater under raised arms tore at its hem (faces stretched past
    # 100 % were skipped and the wearer's dark shirt showed through the notches: ragged, black-edged hem on
    # three recorded sessions); garment IoU against the real shirt on four release videos, 1.0 -> 2.0:
    # 00152_outer 0.6333 -> 0.6365, 00190_upper 0.6578 -> 0.6688, 10006_outer and 00170_upper unchanged.
    TORN_FACE_STRAIN = 2.0

    def torn_faces(self, vertices):
        """Faces with an edge stretched beyond `TORN_FACE_STRAIN` of rest."""
        # The garment's PHYSICAL rest (scan pose), not the zero-pose frame
        # the skinning stores, whose edges are distorted at the armpits.
        key = (id(self._binding), self.TORN_FACE_STRAIN)
        if self._torn_limit is None or self._torn_limit[0] != key:
            rest = self.rest_vertices if self._binding is None else self._binding.physical_rest
            limits = [
                np.linalg.norm(rest[self.faces[:, b]] - rest[self.faces[:, a]], axis=1) * (1.0 + self.TORN_FACE_STRAIN)
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
            self._torn_limit = (key, limits)
        vertices = np.asarray(vertices, np.float64)
        torn = np.zeros(len(self.faces), dtype=bool)
        for (a, b), limit in zip(((0, 1), (1, 2), (2, 0)), self._torn_limit[1]):
            torn |= np.linalg.norm(vertices[self.faces[:, b]] - vertices[self.faces[:, a]], axis=1) > limit
        if getattr(self, "_seam_faces", None) is not None:
            torn |= self._seam_faces
        return torn

    # The per-Gaussian colour arrays: the biggest thing `splats` gathers
    # (2 x 90k x 24-45 floats), and the one a shading result replaces.
    COLOUR_KEYS = ("feature_dc", "feature_rest")

    def splats(self, vertices, full=False, colours=True):
        """Face-local Gaussians placed on a (posed) mesh, in world space.

        Splats on torn faces (see `torn_faces`) are left out. The face
        frames and the per-splat placement run in a numba kernel
        (`_place_splats_numba`); `_splats_numpy` is the original and the
        oracle it is tested against (max diff 1e-16 on the checkpoint,
        110 -> 24 ms for 93k splats).

        `colours=False`: leave out the checkpoint's static colours
        (`COLOUR_KEYS`) - `render` does when a shading result is about to
        replace them, and adds them back (`_static_colours`) if none has
        arrived yet."""
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        keep_faces = ~self.torn_faces(vertices)
        keep = keep_faces[self.face_of]
        xyz, rotation, scaling, frames, scale = _place_splats_numba(
            vertices,
            self._faces_c,
            self.face_of,
            self._bary_c,
            self._local_xyz_c,
            self._local_rotation_c,
            self._log_scaling_c,
            float(np.log(self.dilation)),
            keep_faces,
        )
        self._frames = (vertices, frames, scale)  # for `_shaded`: the shading result's xyz are placed on the same mesh
        appearance = self.appearance if colours else {k: v for k, v in self.appearance.items() if k not in self.COLOUR_KEYS}
        if full:
            # every Gaussian, torn ones flagged by "keep" rather than cut
            # out: nothing per-Gaussian is copied (the appearance, 121k x 48
            # floats, was gathered anew every frame - 11 ms, and again for
            # the shading result - 16 ms; the torn set moves by ~150 splats a
            # frame, so a cache of the gather never hit). Only the GL
            # rasteriser takes this (`render_splats_gl` honours "keep").
            self._kept_indices = None
            return {"xyz": xyz, "rotation": rotation_to_quaternion(rotation), "scaling": scaling, "keep": keep, **appearance}
        self._kept_indices = np.nonzero(keep)[0]  # which checkpoint Gaussians the result holds
        return {
            "xyz": xyz[keep],
            "rotation": rotation_to_quaternion(rotation[keep]),
            "scaling": scaling[keep],
            **{key: value[keep] for key, value in appearance.items()},
        }

    def _checkpoint_index(self, index):
        """`index` (into the splats this renderer holds; None: all of them) as
        an index into the checkpoint's Gaussians, which is what the shading
        network's result is indexed by (they differ with `subsample`)."""
        if self._subset is None:
            return index
        return self._subset if index is None else self._subset[index]

    def _static_colours(self, splats):
        """`splats` with the checkpoint's own colours, if `splats(colours=False)` left them out."""
        if all(key in splats for key in self.COLOUR_KEYS):
            return splats
        index = self._kept_indices
        return {**splats, **{key: self.appearance[key] if index is None else self.appearance[key][index] for key in self.COLOUR_KEYS}}

    def _frames_of(self, mesh):
        """(frames, scale) `splats` computed for `mesh`, or None if it was another mesh."""
        cached = getattr(self, "_frames", None)
        if cached is not None and cached[0] is np.ascontiguousarray(mesh, dtype=np.float64):
            return cached[1:]
        return None

    def _shading_result(self, mesh, view, fitter):
        """This frame's colours and positions from the checkpoint's
        appearance network (see `appearance`), the wearer's body as an
        occluder for its ambient-occlusion bake; None while an asynchronous
        worker has not finished its first pass."""
        from tryon.garment.appearance import AppearanceNet, AsyncAppearance

        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        if self.STUDENT_SHADING:
            from tryon.garment.shading_student import AsyncStudent, ShadingStudent

            if self._appearance_net is None and ShadingStudent.exists(self.checkpoint_dir):
                student = ShadingStudent(self.checkpoint_dir, faces=self.faces)
                student.per_vertex_distance = True  # same shading (31.2 dB either way), fewer queries
                self._appearance_net = AsyncStudent(student) if self.APPEARANCE_ASYNC else student
            if isinstance(self._appearance_net, AsyncStudent):
                self._appearance_net.submit(mesh, view, body)
                return self._appearance_net.latest(mesh, self._frames_of(mesh))
            if isinstance(self._appearance_net, ShadingStudent):
                # this frame's shading, from the distilled network
                return self._appearance_net.forward(mesh, view, body)
        occluders = [(body, np.asarray(fitter.model.faces))]
        if self.APPEARANCE_ASYNC:
            if self._appearance_net is None:
                self._appearance_net = AsyncAppearance(self.checkpoint_dir, faces=self.faces)
            self._appearance_net.submit(mesh, view, occluders)
            return self._appearance_net.latest(mesh, self._frames_of(mesh))
        if self._appearance_net is None:
            self._appearance_net = AppearanceNet(self.checkpoint_dir, faces=self.faces, half=True)
        return self._appearance_net.forward(mesh, view, occluders=occluders)

    def _shaded(self, splats, mesh, view, fitter):
        """The splats with this frame's colours (`_shading_result`), or the
        checkpoint's own before the first one."""
        from tryon.garment.appearance import merge

        result = self._shading_result(mesh, view, fitter)
        return self._static_colours(splats) if result is None else merge(splats, result, self._checkpoint_index(self._kept_indices))

    def _splats_numpy(self, vertices):
        """The original numpy placement; the oracle for `splats`."""
        _, orientation, scale = face_frames(vertices, self.faces)
        keep = ~self.torn_faces(vertices)[self.face_of]
        face_of = self.face_of[keep]
        frames = orientation[face_of]
        anchors = np.einsum("nk,nkd->nd", self.bary[keep], vertices[self.faces[face_of]])
        xyz = np.einsum("nij,nj->ni", frames, self.local_xyz[keep]) * scale[face_of, None] + anchors
        # Compose the face frame with the splat's own rotation, so the
        # covariance turns with the sleeve. R_world = R_face @ R_local.
        world_rotation = np.einsum("nij,njk->nik", frames, self.local_rotation[keep])
        scaling = (
            self.log_scaling[keep]
            + np.log(np.maximum(scale[face_of], 1e-20))[:, None]
            + np.log(self.dilation)
        )
        return {
            "xyz": xyz,
            "rotation": rotation_to_quaternion(world_rotation),
            "scaling": scaling,
            **{key: value[keep] for key, value in self.appearance.items()},
        }

    @staticmethod
    def camera(fitter, size):
        """Perspective camera reproducing the fitter's weak-perspective one.

        The fitter maps model (x, y) to pixels as `scale * (x, -y) +
        translation` - y flipped because image rows grow downward. A camera
        on the +z axis at `CAMERA_DISTANCE`, looking at the origin, with
        focal `scale * CAMERA_DISTANCE` and principal point at
        `translation`, projects `(x, y, 0)` to the same pixel; depth away
        from that plane costs under 2% (see module docstring).
        """
        s = float(fitter.scale.detach())
        tx, ty = (float(v) for v in fitter.translation.detach().ravel())
        d = CAMERA_DISTANCE
        # World -> camera: camera at (0, 0, d) looking down -z at the origin
        # in a frame where model +y is up; the rasteriser's camera has +y
        # DOWN and +z forward, so flip y and z.
        view = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        focal = s * d
        intrinsics = np.array([[focal, 0.0, tx], [0.0, focal, ty], [0.0, 0.0, 1.0]])
        return view, intrinsics, size

    # Only body parts the garment does not cover occlude it (see
    # `_body_covered_by_garment`); COVER_DISTANCE: how far along its outward
    # normal a body vertex may be from the garment and still be inside it.
    OCCLUDE_UNCOVERED_ONLY = True
    # Composite the splats with OpenGL (tryon/rendering/splat_gl.py) instead of the
    # numba tile loop: the same image to float32 rounding (PSNR 110 dB on
    # a 68k-splat frame), 143 -> 37 ms at 640 x 870. Off by default
    # (live.py --gpu-render) - it needs moderngl and a GL 3.3 context.
    GPU_RASTER = False
    # With GPU_RASTER: the body occlusion as a depth pass in the rasteriser
    # (`render_splats_gl(occluder=...)`) instead of `occluded_by_body`'s ray
    # cast on the CPU - the same test, sampled at the splat's centre pixel.
    GL_OCCLUSION = True
    # With `appearance`: shade on a worker thread and draw each frame with
    # the newest finished shading (tryon/garment/appearance.py AsyncAppearance)
    # instead of waiting ~0.6 s for this frame's. The first frames, before
    # the first result, are drawn with the static colours.
    APPEARANCE_ASYNC = False
    # With `appearance`: shade every frame with the garment's distilled
    # student (tryon/garment/shading_student.py, stage3/shading_student.pt) when
    # it has one; falls back to the appearance network otherwise.
    STUDENT_SHADING = False
    COVER_DISTANCE = 0.10

    def _body_covered_by_garment(self, fitter):
        """Body vertices inside the garment at bind time (sleeves, panels).

        A body part inside the garment never hides it: its own sleeve or
        panel wraps it. When the far arm of a dancer bent behind her in
        profile (00176_outer frame 20), 807 faces of that sleeve tore past
        `TORN_FACE_STRAIN` and were skipped, the bare fitted arm then
        counted as in front of the back panel, and the cull punched a
        hole through the shirt to the background. What still occludes is
        what the garment leaves bare: the head (a collar or hood behind
        it), the hands, and the chest under an open front (the back panel
        behind it)."""
        from tryon.garment.skinning import _vertex_normals

        garment = self._binding.apply(fitter).astype(np.float32)
        body = np.asarray(fitter.vertices(), np.float64)
        # Enclosed, not merely near: a ray along the body's outward normal
        # must hit the garment within COVER_DISTANCE. Nearness alone
        # counted the chest under the open front as covered (the gap is
        # narrower than 2 x 8 cm), so it stopped hiding the back panel and
        # its inside showed through the opening as a flat red patch.
        normals = _vertex_normals(body, np.asarray(fitter.faces)).astype(np.float32)
        scene = build_scene([(garment, self.faces)])
        origins = (body - 0.005 * normals).astype(np.float32)
        hit = hit_distance(scene, origins, normals)
        return hit < self.COVER_DISTANCE

    def occluded_by_body(self, splats, fitter, margin=0.03):
        """Which splats the wearer's own body hides from the camera.

        The garment is drawn over a real photo, so the rasteriser never
        sees the body: a splat on the BACK of the garment, behind the
        torso or inside a hood, has nothing in front of it and is drawn
        straight over the face and chest. Measured on a real frame with
        the hooded coat checkpoint (00152_outer): the hood's back panel
        rendered across the wearer's face, which was first misread as a
        vertical misalignment (see `bind.align_to_body`).

        A ray from the camera (`camera`: on the +z axis at
        `CAMERA_DISTANCE`, the perspective camera the splats are actually
        rasterised with) to each splat, cast against the fitted body mesh
        (open3d, the same caster the appearance bake uses): the splat is
        hidden when the body is hit more than `margin` metres before it.
        This replaced a screen-space depth buffer (removed 1 Oct 2026)
        on 26 Sep 2026, measured on the annotated profile
        cameras at frame 3 with the ray cast as the ground truth: the
        buffer culled 53.6 / 36.5 % of the splats on 0028 / 0076, of
        which 9,279 / 7,656 (10 / 8 % of all splats) were NOT behind the
        body - the buffer grows every body pixel by `pixel_radius`, so a
        splat beside a silhouette edge (the near sleeve wrapping a thin
        arm, the front panel at the torso's edge in profile: 41 % of the
        near sleeve culled, 2,053 of those 5,085 wrongly) reads the
        body's front depth from a pixel the body does not cover and is
        culled by a surface its own ray never crosses. The buffer kept
        only 17-50 splats the ray cast hides. ~10 ms at 93k splats.
        """
        body, faces = self.body_occluder(fitter)
        if len(faces) == 0:
            return np.zeros(len(splats["xyz"]), dtype=bool)
        return occluded_from(splats["xyz"], np.array([0.0, 0.0, CAMERA_DISTANCE], np.float32), body, faces, margin)

    def body_occluder(self, fitter):
        """(vertices float32, faces int32) of the body that hides the
        garment: the fitted body without the faces the garment covers (see
        `_body_covered_by_garment`)."""
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float32)
        faces = np.asarray(fitter.model.faces, np.int32)
        covered = getattr(self, "_covered_body", None)
        if covered is not None and len(covered) == len(body):
            faces = faces[~covered[faces].all(axis=1)]
        return body, faces


    def render(self, fitter, size, occlude_body=True, render_scale=None, dt=None):
        """BGRA uint8 of the garment on the fitter's current pose (BGR,
        the rasteriser's own order, plus alpha).

        `occlude_body`: drop splats the wearer's body hides (see
        `occluded_by_body`). On by default - a garment's back panel showing
        through the face is never wanted; the switch exists for measuring
        the effect.

        `render_scale`: rasterise at this fraction of `size` and upscale
        the result (default: the instance's `render_scale`). The
        rasteriser's cost is per splat-PIXEL, so half scale is ~4x
        cheaper while every splat is still drawn - unlike `subsample`,
        which drops splats and leaves holes in the garment.
        """
        height, width = size
        scale = float(self.render_scale if render_scale is None else render_scale)
        mesh = self.posed_mesh(fitter, dt)
        # the GL path takes every Gaussian with a "keep" mask (no gathers)
        full = bool(self.GPU_RASTER and self.GL_OCCLUSION)
        shaded = self.appearance_enabled
        # a shading result replaces the colours (and positions) wholesale, so
        # the static colours are not gathered; the colours are put on last,
        # gathered once for the splats that survive the occlusion cull
        splats = self.splats(mesh, full=full, colours=not shaded)
        view, intrinsics, _ = self.camera(fitter, size)
        index = self._kept_indices  # which checkpoint Gaussians `splats` holds (None: all)
        result = None
        if shaded:
            from tryon.garment.appearance import merge

            result = self._shading_result(mesh, view, fitter)
            if result is not None:
                splats = merge(splats, result, self._checkpoint_index(index), keys=("xyz",))
        occluder = None
        if occlude_body and self.GPU_RASTER and self.GL_OCCLUSION:
            occluder = self.body_occluder(fitter)
        elif occlude_body:
            keep = ~self.occluded_by_body(splats, fitter)
            splats = {key: value[keep] for key, value in splats.items()}
            index = np.flatnonzero(keep) if index is None else index[keep]
        if shaded:
            source, where = (self.appearance, index) if result is None else (result, self._checkpoint_index(index))
            splats = {**splats, **{key: source[key] if where is None else source[key][where] for key in self.COLOUR_KEYS}}
        small = (max(int(round(height * scale)), 1), max(int(round(width * scale)), 1))
        if small != (height, width):
            # Same camera, fewer pixels: focal length and principal point
            # scale with the image.
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= small[1] / width
            intrinsics[1, 1] *= small[0] / height
            intrinsics[0, 2] *= small[1] / width
            intrinsics[1, 2] *= small[0] / height
        if self.GPU_RASTER:
            from tryon.rendering.splat_gl import render_splats_gl

            # 8-bit straight from the GPU (rounded there; the float path
            # below truncates - at most one level apart)
            rgba = render_splats_gl(splats, view, intrinsics, small, bgra8=True, occluder=occluder)
        else:
            image, alpha = render_splats(splats, view, intrinsics, small)
            rgba = np.dstack(
                [
                    (np.clip(image, 0, 1) * 255).astype(np.uint8),
                    (np.clip(alpha, 0, 1) * 255).astype(np.uint8),
                ]
            )
        if small != (height, width):
            import cv2

            rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_LINEAR)
        return rgba
