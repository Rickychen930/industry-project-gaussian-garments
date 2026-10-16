"""Render a garment onto the tracked body.

    garment mesh --bind--> body --pose--> posed mesh --face frames--> splats
                                                    --camera--> pixels

The checkpoint stores its Gaussians in face-local frames (position, scale,
rotation relative to the bound mesh face), so only the mesh has to be
deformed; `face_frames` re-derives every face's frame and the splats follow
(Gaussian Garments, Rong et al. 2409.08189). Splat rotations are composed
with the face frame, so shading on a posed sleeve stays lit correctly.

Camera: the fitter is weak-perspective, the rasteriser perspective. Placing
the perspective camera at `CAMERA_DISTANCE` makes them agree to under 2%
for a body, so the garment lands where the fit put the joints.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from numba import njit, prange

from tryon.garment.checkpoint import load_checkpoint, uv_binding
from tryon.garment.cloth import SmoothCloth, geodesic_from
from tryon.garment.raycast import build_scene, hit_distance, occluded_from
from tryon.garment.scan_pose import ScanPoseBinder
from tryon.garment.skinning import SkinnedGarment, closed_front_flag, drape_narrowing, front_closure
from tryon.rendering.splat_cpu import face_frame, face_frames, quaternion_to_rotation, render_splats, rotation_to_quaternion

# far enough that weak perspective and perspective agree, near enough for float precision
CAMERA_DISTANCE = 50.0
# stored scales leave gaps between neighbours; 1.3 closes them for the least blur
SPLAT_DILATION = 1.3

@njit(cache=True, parallel=True)
def _place_splats_numba(vertices, faces, face_of, bary, local_xyz, local_rotation, log_scaling, log_dilation, keep_faces):
    """`splat_cpu.face_frames` plus the splat placement of
    `GarmentRenderer._splats_numpy`, with the same maths and order.

    Splats on faces not in `keep_faces` get placeholder rows the caller
    drops or masks. Also returns frames and scales for
    `AppearanceNet.world_xyz`."""
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
            # torn face: harmless placeholder
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
def _torn_faces(vertices, faces, l0, l1, l2, seam):
    """`GarmentRenderer.torn_faces`: a face is torn when an edge exceeds its
    limit or it lies on the sleeve seam. Edge lengths use `np.linalg.norm`'s
    summation order so results match the numpy form."""
    out = np.zeros(faces.shape[0], np.bool_)
    for f in range(faces.shape[0]):
        if seam[f]:
            out[f] = True
            continue
        i, j, k = faces[f, 0], faces[f, 1], faces[f, 2]
        d0, d1, d2 = vertices[j, 0] - vertices[i, 0], vertices[j, 1] - vertices[i, 1], vertices[j, 2] - vertices[i, 2]
        if np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) > l0[f]:
            out[f] = True
            continue
        d0, d1, d2 = vertices[k, 0] - vertices[j, 0], vertices[k, 1] - vertices[j, 1], vertices[k, 2] - vertices[j, 2]
        if np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) > l1[f]:
            out[f] = True
            continue
        d0, d1, d2 = vertices[i, 0] - vertices[k, 0], vertices[i, 1] - vertices[k, 1], vertices[i, 2] - vertices[k, 2]
        if np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) > l2[f]:
            out[f] = True
    return out


@njit(cache=True)
def _strain_passes(x, edges, limit, iterations):
    """Strain limiting in place (`GarmentRenderer._limit_strain`), Gauss-Seidel:
    each over-long edge in order moves both ends half its excess toward each
    other, so later edges see the moved ends. Stops when a pass finds none.
    Converges where the Jacobi form did not, in a third of the work."""
    for _ in range(iterations):
        any_over = False
        for k in range(edges.shape[0]):
            a, b = edges[k, 0], edges[k, 1]
            d0, d1, d2 = x[b, 0] - x[a, 0], x[b, 1] - x[a, 1], x[b, 2] - x[a, 2]
            length = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            if length > limit[k]:
                any_over = True
                f = 0.5 * (length - limit[k]) / length
                x[a, 0] += f * d0
                x[a, 1] += f * d1
                x[a, 2] += f * d2
                x[b, 0] -= f * d0
                x[b, 1] -= f * d1
                x[b, 2] -= f * d2
        if not any_over:
            break


class GarmentRenderer:
    """A garment checkpoint, bound to a body once, rendered per pose.

    `subsample`: keep only this fraction of the Gaussians, chosen once at
    load with a fixed seed. Render cost tracks splat count roughly linearly,
    so this trades quality for speed (visible speckle below ~25%). Default
    `None` keeps all.
    """

    # anchor splats at their texel's barycentric point (as the reference) rather than the face centre
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
        """`appearance`: run the checkpoint's own appearance network
        (`appearance.py`, ~0.75 s a frame on MPS) instead of static SH only."""
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
        # The stage-3 model anchors each Gaussian at its UV texel's barycentric
        # point; the face centre piles ~6 splats per face together. The stored
        # xyz does not compensate.
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
        # which of the checkpoint's Gaussians are held (None: all)
        self._subset = None
        if subsample is not None:
            fraction = float(subsample)
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"subsample must be in (0, 1), got {fraction}")
            count = int(usable * fraction)
            keep = np.random.default_rng(seed).choice(usable, count, replace=False)
            keep.sort()
            self._subset = keep
            face_of = face_of[keep]
            bary = bary[keep]
            local_xyz = local_xyz[keep]
            local_rotation = local_rotation[keep]
            log_scaling = log_scaling[keep]
            appearance = {key: value[keep] for key, value in appearance.items()}
        # precomputed so `_prepare` skips a per-frame sigmoid
        appearance["opacity_alpha"] = 1.0 / (1.0 + np.exp(-appearance["opacity"].ravel()))
        self.face_of = face_of
        self.bary = bary
        self.local_xyz = local_xyz
        self.local_rotation = local_rotation
        self.log_scaling = log_scaling
        self.appearance = appearance
        # contiguous copies for the placement kernel
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
        # cloth springs plus collision; False = pure skinning; None = class `DYNAMICS`
        self.dynamics_enabled = self.DYNAMICS if dynamics is None else bool(dynamics)
        # fraction of output size the splats are rasterised at (see `render`)
        self.render_scale = float(render_scale)
        self._light = None
        self._last_tick = None
        self._frames = None  # (vertices, frames, scale) of the last `splats`
        self._strain = None  # (key, edges, limit) of `_limit_strain`, per binding
        self._torn_limit = None  # (key, per-face-edge tear length) of `torn_faces`, per binding

    # cloth model: SmoothCloth (about a third of LightCloth's jitter) or LightCloth
    CLOTH = SmoothCloth

    # cloth motion on by default
    DYNAMICS = True
    # never draw faces straddling the sleeve label (see `bind`)
    CUT_SLEEVE_SEAM = True
    # Off: the fit runs in a normalised frame, so betas settle near zero and
    # the chest ratio no longer measures the wearer (IoU 0.73 with, 0.75 without).
    WIDEN_TO_WEARER = False
    # How the wearer's size ratios scale the garment: `ScanPoseBinder.SIZE_MODES`.
    SIZE_MODE = "current"
    # Bring the front panels of an open-front garment together at the midline
    # (`skinning.front_closure`, once at bind). None = off; "auto": only when
    # the per-garment flag file says it is worn closed (`closed_front_flag`);
    # "always": every garment (evaluation).
    CLOSE_FRONT = "auto"
    CLOSE_SEAM = 0.01  # m: gap left between the two inner edges
    CLOSE_BLEND = 0.2  # m: lateral distance over which the pull fades out

    # Narrow the loose lower torso toward the centre line by this fraction of
    # its distance from it, scaled by looseness (`skinning.drape_narrowing`).
    # None = off.
    DRAPE_NARROW = 0.08
    DRAPE_START = 0.3  # fraction of the way from top to hem where narrowing starts, and the ramp span
    DRAPE_SPAN = 0.4

    def closes_front(self):
        if self.CLOSE_FRONT == "always":
            return True
        return self.CLOSE_FRONT == "auto" and closed_front_flag(self.checkpoint_dir)

    def bind(self, fitter):
        """Skin the garment to this fitter's body at rest. Once per person.

        Uses `SkinnedGarment` (nearest-vertex binding tears at the joints).
        The binding body is posed like the garment's scan, not like this
        frame (two-pass; see `ScanPoseBinder`), and the garment is then scaled
        by the wearer's height ratio so it covers the same fraction of the
        body.
        """
        bind_fitter, _, aligned, sleeve = self.scan_binder.bind(fitter)
        self.sleeve_vertices = sleeve
        self._binding = SkinnedGarment.build_in_pose(aligned, bind_fitter, sleeve=sleeve)
        if self.closes_front():
            self._binding = self._binding.moved(front_closure(self._binding.rest, self.faces, self.CLOSE_SEAM, self.CLOSE_BLEND))
        if self.DRAPE_NARROW:
            from scipy.spatial import cKDTree

            with torch.no_grad():
                bind_body = bind_fitter._forward().vertices[0].numpy().astype(np.float64)
            distance = cKDTree(bind_body).query(aligned)[0]
            offset = drape_narrowing(self._binding.rest, distance, sleeve, self.DRAPE_NARROW, start=self.DRAPE_START, span=self.DRAPE_SPAN)
            self._binding = self._binding.moved(offset)
        # bound in the template frame the garment is the scan subject's size;
        # scale it by the wearer's height over the scan body's
        mode = self.SIZE_MODE
        height = self.scan_binder.height_ratio(fitter, bind_fitter) if self.scan_pose_used else 1.0
        # Width: scaling x/z by the chest ratio fixes wide bodies but breaks
        # narrow ones, so it only applies when the wearer is wider (never below 1).
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

        # The scan fused hanging sleeves to the body; faces straddling the
        # sleeve label are that false contact and always tear, so never draw them.
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
        """Restart the cloth at rest on the current pose (after a refit jump)."""
        self._light = None
        self._last_tick = None

    # Strain limiting (Provot 1995): edges longer than (1 + STRAIN_LIMIT) x
    # rest are shortened. None = off. See `_limit_strain`.
    STRAIN_LIMIT = 0.05
    STRAIN_ITERATIONS = 10  # Gauss-Seidel passes (`_strain_passes`)

    def _limit_strain(self, vertices):
        """Pull stretched edges back toward the garment's physical rest
        (skinning from the scan pose stretches fabric; real fabric barely does)."""
        if self.STRAIN_LIMIT is None:
            return vertices
        key = (id(self._binding), self.STRAIN_LIMIT)
        if self._strain is None or self._strain[0] != key:
            edges = self.scan_binder.edges
            rest = self._binding.physical_rest
            length = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
            limit = length * (1.0 + self.STRAIN_LIMIT)
            self._strain = (key, edges, limit)
        _, edges, limit = self._strain
        x = np.array(vertices, np.float64, copy=True)
        _strain_passes(x, edges, limit, int(self.STRAIN_ITERATIONS))
        return x

    def posed_mesh(self, fitter, dt=None):
        """The garment on the fitter's current pose (strain-limited when
        STRAIN_LIMIT is set). `dt`: seconds to advance the cloth solver
        (default: wall-clock time since the previous call; offline callers
        pass the frame interval)."""
        return self._limit_strain(self._posed_mesh(fitter, dt))

    # What the long-range attachment (`LightCloth.LRA`) hangs from: "top" =
    # the highest TOP_RING_FRACTION of body vertices (does nothing in
    # practice); "sleeve" = the sleeve/body-panel border (the armhole), so a
    # raised arm lifts the hem. Only non-sleeve vertices are limited.
    LRA_ANCHORS = "sleeve"
    TOP_RING_FRACTION = 0.03

    def _long_range_anchors(self):
        """(anchor_of, rest geodesic) per vertex over the physical rest; once per binding."""
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
            anchor_of = np.where(sleeve, -1, anchor_of)  # sleeves follow the arm
            self._anchors = (key, anchor_of, distance)
        return self._anchors[1:]

    def _freeze_size(self, fitter):
        """SIZE_MODE "frozen": scale by the median height ratio of the first
        `FREEZE_FRAMES` frames, then hold (single-frame shape is noisy)."""
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
        if not self.dynamics_enabled:
            return skin_target
        now = time.perf_counter()
        if dt is None:
            dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        if self._light is None:
            self._light = self.CLOTH()
            if self.CLOTH.LRA is not None:
                self._light.set_long_range(*self._long_range_anchors())
        return self._light.step(skin_target, dt, fitter.vertices(), fitter.faces)

    # A face with an edge stretched past this factor of rest is broken
    # geometry and its splats would smear; skip them. 1.0 tore cropped
    # sweaters at the hem under raised arms.
    TORN_FACE_STRAIN = 2.0

    def torn_faces(self, vertices):
        """Faces with an edge stretched beyond `TORN_FACE_STRAIN` of rest."""
        # physical rest (scan pose), not the zero-pose frame, whose edges distort at the armpits
        key = (id(self._binding), self.TORN_FACE_STRAIN)
        if self._torn_limit is None or self._torn_limit[0] != key:
            rest = self.rest_vertices if self._binding is None else self._binding.physical_rest
            limits = [
                np.linalg.norm(rest[self.faces[:, b]] - rest[self.faces[:, a]], axis=1) * (1.0 + self.TORN_FACE_STRAIN)
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
            self._torn_limit = (key, limits)
        seam = getattr(self, "_seam_faces", None)
        l0, l1, l2 = self._torn_limit[1]
        return _torn_faces(np.ascontiguousarray(vertices, np.float64), np.ascontiguousarray(self.faces), l0, l1, l2,
                           np.zeros(len(self.faces), np.bool_) if seam is None else np.asarray(seam, np.bool_))

    # per-Gaussian colour arrays: the largest gather in `splats`, replaced by a shading result
    COLOUR_KEYS = ("feature_dc", "feature_rest")

    def splats(self, vertices, full=False, colours=True):
        """Face-local Gaussians placed on a (posed) mesh, in world space.

        Splats on torn faces (see `torn_faces`) are left out. Placement runs
        in `_place_splats_numba`; `_splats_numpy` is the oracle.

        `colours=False`: omit the static colours (`COLOUR_KEYS`) when a
        shading result will replace them; `_static_colours` restores them if
        none has arrived."""
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
        self._frames = (vertices, frames, scale)  # for `_shaded`
        appearance = self.appearance if colours else {k: v for k, v in self.appearance.items() if k not in self.COLOUR_KEYS}
        if full:
            # every Gaussian, torn ones flagged by "keep" instead of cut, so
            # nothing per-Gaussian is copied; only the GL rasteriser honours "keep"
            self._kept_indices = None
            return {"xyz": xyz, "rotation": rotation_to_quaternion(rotation), "scaling": scaling, "keep": keep, **appearance}
        self._kept_indices = np.nonzero(keep)[0]  # checkpoint Gaussians the result holds
        return {
            "xyz": xyz[keep],
            "rotation": rotation_to_quaternion(rotation[keep]),
            "scaling": scaling[keep],
            **{key: value[keep] for key, value in appearance.items()},
        }

    def _checkpoint_index(self, index):
        """`index` into this renderer's splats (None: all) as an index into the
        checkpoint's Gaussians, which the shading result is indexed by."""
        if self._subset is None:
            return index
        return self._subset if index is None else self._subset[index]

    def _static_colours(self, splats):
        """`splats` with the checkpoint's colours if `splats(colours=False)` left them out."""
        if all(key in splats for key in self.COLOUR_KEYS):
            return splats
        index = self._kept_indices
        return {**splats, **{key: self.appearance[key] if index is None else self.appearance[key][index] for key in self.COLOUR_KEYS}}

    def _frames_of(self, mesh):
        """(frames, scale) `splats` computed for `mesh`, or None for another mesh."""
        cached = getattr(self, "_frames", None)
        if cached is not None and cached[0] is np.ascontiguousarray(mesh, dtype=np.float64):
            return cached[1:]
        return None

    def _shading_result(self, mesh, view, fitter):
        """This frame's colours and positions from the appearance network (body
        as AO occluder); None until an async worker finishes its first pass."""
        from tryon.garment.appearance import AppearanceNet, AsyncAppearance

        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        if self.STUDENT_SHADING:
            from tryon.garment.shading_student import AsyncStudent, ShadingStudent

            if self._appearance_net is None and ShadingStudent.exists(self.checkpoint_dir):
                student = ShadingStudent(self.checkpoint_dir, faces=self.faces)
                student.per_vertex_distance = True  # same shading, fewer queries
                self._appearance_net = AsyncStudent(student) if self.APPEARANCE_ASYNC else student
            if isinstance(self._appearance_net, AsyncStudent):
                self._appearance_net.submit(mesh, view, body)
                return self._appearance_net.latest(mesh, self._frames_of(mesh))
            if isinstance(self._appearance_net, ShadingStudent):
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

    def _splats_numpy(self, vertices):
        """The original numpy placement; the oracle for `splats`."""
        _, orientation, scale = face_frames(vertices, self.faces)
        keep = ~self.torn_faces(vertices)[self.face_of]
        face_of = self.face_of[keep]
        frames = orientation[face_of]
        anchors = np.einsum("nk,nkd->nd", self.bary[keep], vertices[self.faces[face_of]])
        xyz = np.einsum("nij,nj->ni", frames, self.local_xyz[keep]) * scale[face_of, None] + anchors
        # R_world = R_face @ R_local, so the covariance turns with the sleeve
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

        On the +z axis at `CAMERA_DISTANCE` with focal `scale * CAMERA_DISTANCE`
        and principal point at `translation`; y is flipped (image rows grow
        downward). Depth away from the z=0 plane costs under 2%.
        """
        s = float(fitter.scale.detach())
        tx, ty = (float(v) for v in fitter.translation.detach().ravel())
        d = CAMERA_DISTANCE
        # the rasteriser's camera has +y down and +z forward, so flip y and z
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

    # Only body parts the garment does not cover occlude it
    # (`_body_covered_by_garment`).
    OCCLUDE_UNCOVERED_ONLY = True
    # Composite with OpenGL (`splat_gl.py`) instead of the numba tile loop;
    # same image, ~4x faster. Needs moderngl and a GL 3.3 context.
    GPU_RASTER = False
    # With GPU_RASTER: body occlusion as a depth pass in the rasteriser
    # instead of `occluded_by_body`'s CPU ray cast.
    GL_OCCLUSION = True
    # With `appearance`: shade on a worker thread and draw the newest finished
    # result (static colours until the first arrives).
    APPEARANCE_ASYNC = False
    # With `appearance`: use the garment's distilled student
    # (`shading_student.py`) when it has one, else the full network.
    STUDENT_SHADING = False
    # m along a body vertex's outward normal within which it counts as inside the garment
    COVER_DISTANCE = 0.10

    def _body_covered_by_garment(self, fitter):
        """Body vertices inside the garment at bind time (sleeves, panels).

        A body part inside the garment never hides it; if its own sleeve
        tears, the bare arm would otherwise punch a hole through the shirt.
        What still occludes is what the garment leaves bare: head, hands,
        and the chest under an open front."""
        from tryon.garment.skinning import _vertex_normals

        garment = self._binding.apply(fitter).astype(np.float32)
        body = np.asarray(fitter.vertices(), np.float64)
        # Enclosed, not merely near: nearness alone counted the chest under an
        # open front as covered, so the back panel showed through the opening.
        normals = _vertex_normals(body, np.asarray(fitter.faces)).astype(np.float32)
        scene = build_scene([(garment, self.faces)])
        origins = (body - 0.005 * normals).astype(np.float32)
        hit = hit_distance(scene, origins, normals)
        return hit < self.COVER_DISTANCE

    def occluded_by_body(self, splats, fitter, margin=0.03):
        """Which splats the wearer's own body hides from the camera.

        The garment is drawn over a real photo, so the rasteriser never sees
        the body: a splat on the back of the garment would show over the face
        and chest. A ray from the camera (`CAMERA_DISTANCE` on +z) to each
        splat is cast against the fitted body (open3d); the splat is hidden
        when the body is hit more than `margin` metres before it. A
        screen-space depth buffer culled splats beside silhouette edges
        wrongly, so rays are used.
        """
        body, faces = self.body_occluder(fitter)
        if len(faces) == 0:
            return np.zeros(len(splats["xyz"]), dtype=bool)
        return occluded_from(splats["xyz"], np.array([0.0, 0.0, CAMERA_DISTANCE], np.float32), body, faces, margin)

    def body_occluder(self, fitter):
        """(vertices float32, faces int32) of the fitted body without the faces the garment covers."""
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float32)
        faces = np.asarray(fitter.model.faces, np.int32)
        covered = getattr(self, "_covered_body", None)
        if covered is not None and len(covered) == len(body):
            faces = faces[~covered[faces].all(axis=1)]
        return body, faces

    def render(self, fitter, size, occlude_body=True, render_scale=None, dt=None):
        """BGRA uint8 of the garment on the fitter's current pose.

        `occlude_body`: drop splats the wearer's body hides (see
        `occluded_by_body`); the switch exists for measuring.

        `render_scale`: rasterise at this fraction of `size` and upscale
        (default: the instance's). Cost is per splat-pixel, so this is cheaper
        than `subsample` and leaves no holes.
        """
        height, width = size
        scale = float(self.render_scale if render_scale is None else render_scale)
        mesh = self.posed_mesh(fitter, dt)
        # the GL path takes every Gaussian with a "keep" mask (no gathers)
        full = bool(self.GPU_RASTER and self.GL_OCCLUSION)
        shaded = self.appearance_enabled
        # a shading result replaces colours wholesale: skip the static gather
        # and put colours on last, once, for the splats that survive the cull
        splats = self.splats(mesh, full=full, colours=not shaded)
        view, intrinsics, _ = self.camera(fitter, size)
        index = self._kept_indices  # checkpoint Gaussians `splats` holds (None: all)
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
            # same camera, fewer pixels: focal and principal point scale with the image
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= small[1] / width
            intrinsics[1, 1] *= small[0] / height
            intrinsics[0, 2] *= small[1] / width
            intrinsics[1, 2] *= small[0] / height
        if self.GPU_RASTER:
            from tryon.rendering.splat_gl import render_splats_gl

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
