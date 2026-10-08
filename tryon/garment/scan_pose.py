"""Binding a garment to the body in the pose it was scanned in.

The garment was captured on a person standing in their own pose, arms
hanging; skin weights taken on a T-posed body put the sleeves beside the
hips. `ScanPoseBinder.bind` poses a body like the scan (the scan-frame
SMPL-X fit cached next to the checkpoint by tools/data/fit_scan_pose.py,
or a default arm angle), labels the sleeve vertices, and returns the body
and garment to bind on (`GarmentRenderer.bind`). The size ratios between
the wearer and the scan body are here too.
"""

from __future__ import annotations

import numpy as np
import torch

from tryon.garment.skinning import SkinnedGarment, align_to_body


_neutral_body_height = None


def _neutral_height():
    """The height (rest pose, beta=0) of the SMPL-X body this project uses.

    Measured, not hardcoded, so it tracks whatever model file
    `BodyFitter` actually loads rather than silently drifting from it if
    the model is ever swapped. Computed once and cached: building a
    `BodyFitter` loads the SMPL-X model from disk, which is not free to
    repeat on every `bind()` call.
    """
    global _neutral_body_height
    if _neutral_body_height is None:
        from tryon.body.fitter import BodyFitter

        with torch.no_grad():
            vertices = BodyFitter()._forward().vertices[0].numpy()
        _neutral_body_height = float(vertices[:, 1].max() - vertices[:, 1].min())
    return _neutral_body_height


def unique_edges(faces):
    """(E, 2) the mesh's undirected edges, each once, as sorted vertex pairs
    in lexicographic order. Depends only on the faces, so a garment builds it
    once (`ScanPoseBinder.edges`) and the strain limiter shares it."""
    faces = np.asarray(faces)
    both = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(both, axis=1), axis=0)


class ScanPoseBinder:
    """The scan-pose binding of one garment: its rest mesh, faces and
    checkpoint directory."""

    def __init__(self, rest_vertices, faces, checkpoint_dir):
        self.rest_vertices = rest_vertices
        self.faces = faces
        self.checkpoint_dir = checkpoint_dir
        self.scan_pose_used = False
        self._edges = None

    @property
    def edges(self):
        """The garment's unique edges (`unique_edges`), built on first use."""
        if self._edges is None:
            self._edges = unique_edges(self.faces)
        return self._edges

    # Shoulder angle (degrees below horizontal) of the body the sleeves are
    # LABELLED on, and of the body the garment is BOUND on. Measured on
    # 00152_outer re-posed to a T-pose frame (torn edges / sleeve axis off
    # the arm, left / right): one pass at 30: 1 / 42, 33 deg (sleeves not
    # bound to the arms); one pass at 80: 2,903 / 4.7, 26 (sleeves right,
    # body panels torn by arm weights from the arm hanging beside them);
    # two-pass, label at 30 then bind at 60 with candidates restricted by
    # the label: 278 / 27, 16; at 80: 304 / 35, 27. Smoothing the weights
    # along the garment mesh made every case worse.
    SCAN_BIND_ANGLE = 60.0

    @staticmethod
    def chest_ratio(fitter, bind_fitter):
        """Zero-pose chest width (x extent of the non-arm vertices within
        2 cm of the spine3 joint's height) of `fitter`'s body over
        `bind_fitter`'s - the girth cue keypoints never give."""
        from tryon.body.fitter import BodyFitter

        def width(source):
            probe = BodyFitter()
            with torch.no_grad():
                probe.betas[:] = source.betas.detach()
                out = probe._forward()
                v = out.vertices[0].numpy()
                y = float(out.joints[0, 9, 1])
            w = probe.model.lbs_weights.detach().numpy()
            arm = w[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
            band = (~arm) & (np.abs(v[:, 1] - y) < 0.02)
            return float(np.ptp(v[band, 0])) if band.any() else 1.0

        return width(fitter) / max(width(bind_fitter), 1e-6)

    # How the wearer/scan size ratios become the garment's (x, y, z) scale
    # (`size_factors`); "current" is the live behaviour. Measured by
    # tools/evaluate/vs_registration --size-mode (docs/baseline/size-ablation.json).
    #   current   height ratio on all axes (+ chest ratio >= 1 when the
    #             renderer's WIDEN_TO_WEARER is on)
    #   no_chest  the same with the chest ratio never used
    #   widen     height ratio, and the chest ratio >= 1 on x / z
    #   none      no resizing
    #   shrink    ratios capped at 1 (a larger wearer does not grow the garment)
    #   damped    ratios ** 0.5
    #   frozen    height ratio = median over the first FREEZE_FRAMES frames
    #             (renderer), then held
    #   height_y  height ratio on y only, x / z untouched
    SIZE_MODES = ("current", "no_chest", "widen", "none", "shrink", "damped", "frozen", "height_y")
    FREEZE_FRAMES = 30

    @staticmethod
    def size_factors(mode, height, chest, widen=False):
        """(3,) x, y, z scale of the garment for `mode` from the wearer's
        `height` and `chest` ratios over the scan body's."""
        if mode not in ScanPoseBinder.SIZE_MODES:
            raise ValueError(f"unknown size mode {mode!r}: {ScanPoseBinder.SIZE_MODES}")
        if mode == "none":
            return np.ones(3)
        width = max(1.0, chest) if (mode == "widen" or (mode == "current" and widen)) else 1.0
        if mode == "shrink":
            height, width = min(height, 1.0), 1.0
        if mode == "damped":
            height = height**0.5
        if mode == "height_y":
            return np.array([1.0, height, 1.0])
        return np.array([width * height, height, width * height])

    @staticmethod
    def height_ratio(fitter, bind_fitter):
        """Zero-pose height of `fitter`'s body over `bind_fitter`'s."""
        from tryon.body.fitter import BodyFitter

        def height(source):
            probe = BodyFitter()
            with torch.no_grad():
                probe.betas[:] = source.betas.detach()
                v = probe._forward().vertices[0].numpy()
            return float(v[:, 1].max() - v[:, 1].min())

        return height(fitter) / max(height(bind_fitter), 1e-6)

    def _scan_pose(self):
        """The scan-frame SMPL-X pose cached next to the checkpoint by
        tools/data/fit_scan_pose.py (from the authors' trajectory files:
        `obstacle[0]` is the body of the frame the template was scanned
        in; fitted to 1.0 cm mean vertex residual on 00152_outer, arms 19 /
        31 degrees from vertical with the right upper arm tilted 18
        degrees back). None when there is no cache."""
        path = self.checkpoint_dir / "stage1" / "scan_pose.npz"
        if not path.exists():
            return None
        data = np.load(path)
        pose = {key: data[key] for key in ("body_pose", "global_orient", "betas", "offset")}
        # a pose a one-off tool re-fitted because the trajectory file's first
        # frame was not this template (`REFINE_ARMS`)
        pose["refined"] = bool("refined" in data.files and int(data["refined"]))
        return pose

    def _posed_like_scan(self, fitter, theta, arm_axes=None, scan_pose=None):
        """A fitter with this body's shape and camera, arms `theta` degrees
        below horizontal, and the garment aligned to it.

        `arm_axes`: optional {"L": unit vector, "R": unit vector}, the
        shoulder-to-wrist direction each arm should take in the body's
        frame (measured from the scan's own sleeves, see
        `bind`); when given it replaces the in-plane `theta`
        for that arm, so an arm the wearer held forward or back during
        the scan is bound forward or back, not in the body plane."""
        from tryon.body.fitter import BodyFitter

        bind_fitter = BodyFitter()
        with torch.no_grad():
            bind_fitter.betas[:] = fitter.betas.detach()
            bind_fitter.scale.fill_(float(fitter.scale.detach()))
            bind_fitter.translation[:] = fitter.translation.detach()
            bind_fitter.global_orient.zero_()
            bind_fitter.body_pose.zero_()
            # Shoulders (joints 16, 17) rotate about z: the +x arm goes down.
            bind_fitter.body_pose[0, (16 - 1) * 3 + 2] = -np.radians(theta)
            bind_fitter.body_pose[0, (17 - 1) * 3 + 2] = +np.radians(theta)
            if scan_pose is not None:
                # The true scan body (see _scan_pose): shape, every joint,
                # orientation and world offset, so the body reproduces the
                # trajectory file's obstacle[0] (0.7 cm) IN THE TEMPLATE'S
                # OWN FRAME. The garment then needs no alignment, no
                # vertical settle and no height scaling: it is bound where
                # it was scanned, which is what Gaussian-Garments does
                # (lbs.py: nearest body vertex in the template frame).
                bind_fitter.betas[:] = torch.tensor(np.asarray(scan_pose["betas"], np.float32))[None]
                bind_fitter.body_pose[:] = torch.tensor(np.asarray(scan_pose["body_pose"], np.float32))[None]
                bind_fitter.global_orient[:] = torch.tensor(np.asarray(scan_pose["global_orient"], np.float32))[None]
                # The body stays at the model's origin (its skinning
                # transforms know nothing of a world offset); the garment
                # is moved by the negative offset instead.
            if arm_axes:
                # The shoulder joint's parent frame is the (zero-posed)
                # torso, i.e. the body frame: the rotation taking the
                # rest arm direction (+x left, -x right) onto the target
                # is the joint's axis-angle directly.
                for side, joint, rest_dir in (("L", 16, np.array([1.0, 0.0, 0.0])), ("R", 17, np.array([-1.0, 0.0, 0.0]))):
                    target = arm_axes.get(side)
                    if target is None:
                        continue
                    target = np.asarray(target, np.float64) / max(np.linalg.norm(target), 1e-9)
                    axis = np.cross(rest_dir, target)
                    angle = float(np.arctan2(np.linalg.norm(axis), float(rest_dir @ target)))
                    if np.linalg.norm(axis) > 1e-9:
                        rotvec = axis / np.linalg.norm(axis) * angle
                        bind_fitter.body_pose[0, (joint - 1) * 3 : (joint - 1) * 3 + 3] = torch.tensor(rotvec, dtype=torch.float32)
            body = bind_fitter._forward().vertices[0].numpy().astype(np.float64)
        if scan_pose is not None:
            return bind_fitter, np.asarray(self.rest_vertices, np.float64) - np.asarray(scan_pose["offset"], np.float64)
        height = float(body[:, 1].max() - body[:, 1].min())
        scale = height / _neutral_height() if height > 0 else 1.0
        aligned = align_to_body(self.rest_vertices * scale, body)
        return bind_fitter, self._settle_vertically(aligned, body)

    # Vertical shifts tried below the head-top alignment, in metres.
    VERTICAL_SEARCH = np.arange(-0.40, 0.051, 0.025)

    def _settle_vertically(self, aligned, body):
        """Slide the garment vertically to where it sits closest to the body.

        `align_to_body` puts the garment's top at the top of the head. That
        is right for the hooded coat (00152_outer) and measured wrong for
        every other checkpoint: rendered on the same frame, 00122_outer,
        00134_upper and 00176_outer put their collars over the FACE
        (face-window alpha 243-254 of 255) and bound no sleeve vertex to
        the arms; 10004/10006 half-covered it. The vertical offset a
        garment needs is a property of the garment (hood or not, collar
        height), not of the body, so it is found the same way as the bind
        angle: the shift with the smallest median garment-to-body
        distance, searched over `VERTICAL_SEARCH`.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(body)
        best = None
        for shift in self.VERTICAL_SEARCH:
            candidate = aligned + np.array([0.0, shift, 0.0])
            median = float(np.median(tree.query(candidate)[0]))
            if best is None or median < best[0]:
                best = (median, shift)
        return aligned + np.array([0.0, best[1], 0.0])

    # Majority-vote passes over the garment mesh that clean the sleeve label.
    SLEEVE_LABEL_VOTES = 10

    def bind(self, fitter):
        """Two-pass binding. Returns (bind_fitter, angle, aligned garment,
        sleeve mask).

        The sleeve LABEL: at the bind pose, a vertex is a sleeve vertex if
        its nearest ARM body vertex is closer than its nearest non-arm one,
        cleaned by `SLEEVE_LABEL_VOTES` majority votes over the garment's
        own mesh (a binary label can be smoothed along the garment; the
        weights themselves could not - measured worse). An earlier label
        took the arm-dominant vertices of a bind on a body with the arms
        only 30 degrees down; on garments whose sleeves hang close to the
        body (00134_upper, 00122_outer) that arm never reached the sleeve
        and NO vertex was labelled. Measured across five checkpoints
        (sleeve axis off the arm L / R, torn edges): the nearest-arm label
        finds 1,276 sleeve vertices on 00134_upper (21 / 39 deg) where the
        old one found 0, gives 0 tears on 00134 / 00176 / 10006 where the
        old one gave 0 / 156 / 500, and costs a few degrees on the two
        loose coats (00152: 17.6 / 18.3 -> 26.7 / 19.4). The union of the
        two labels adds nothing over it; the raw union without votes tears.
        The garment is then bound with each vertex restricted to its own
        kind of body vertex (`SkinnedGarment.build`'s `sleeve`).
        """
        from scipy.spatial import cKDTree

        edges = self.edges
        degree = np.bincount(edges.ravel(), minlength=len(self.rest_vertices)).astype(np.float64)

        def label(bind_fitter, aligned):
            with torch.no_grad():
                body = bind_fitter._forward().vertices[0].numpy().astype(np.float64)
            body_weights = bind_fitter.model.lbs_weights.detach().numpy().astype(np.float64)
            body_is_arm = body_weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
            to_arm = cKDTree(body[body_is_arm]).query(aligned)[0]
            to_rest = cKDTree(body[~body_is_arm]).query(aligned)[0]
            sleeve = to_arm < to_rest
            for _ in range(self.SLEEVE_LABEL_VOTES):
                votes = np.bincount(edges[:, 0], weights=sleeve[edges[:, 1]], minlength=len(aligned))
                votes += np.bincount(edges[:, 1], weights=sleeve[edges[:, 0]], minlength=len(aligned))
                sleeve = votes > 0.5 * np.maximum(degree, 1.0)
            return sleeve

        scan_pose = self._scan_pose()
        bind_fitter, aligned = self._posed_like_scan(fitter, self.SCAN_BIND_ANGLE, scan_pose=scan_pose)
        sleeve = label(bind_fitter, aligned)
        if scan_pose is not None and self.refine_arms(bind_fitter, aligned, sleeve, scan_pose["refined"]):
            sleeve = label(bind_fitter, aligned)
        arm_axes = None
        self.scan_pose_used = scan_pose is not None
        if scan_pose is None and self.SCAN_AXIS_REFINE and sleeve.any():
            # Second pass: the scan's own sleeves say where its arms were.
            # Measured on 00152_outer with the in-plane 60-degree bind
            # alone: the left sleeve's axis was 29 degrees off the bound
            # arm and its vertices a median 18 cm from the arm axis (right:
            # 13 degrees / 11 cm) - the scanned wearer held the arms
            # forward of the body plane (sleeve axes tilted 23 / 18 degrees
            # in z), which no in-plane angle can reproduce. Each sleeve's
            # principal axis, pointed away from its shoulder, becomes that
            # arm's direction for a re-bind and re-label.
            arm_axes = self._sleeve_axes(bind_fitter, aligned, sleeve)
            if arm_axes:
                bind_fitter, aligned = self._posed_like_scan(fitter, self.SCAN_BIND_ANGLE, arm_axes)
                sleeve = label(bind_fitter, aligned)
        return bind_fitter, self.SCAN_BIND_ANGLE, aligned, sleeve

    SCAN_AXIS_REFINE = True

    # Refit the arms of a cached scan pose to the garment's sleeves. The cache
    # is a fit to the authors' trajectory body `obstacle[0]`, which is the body
    # of the template only when the trajectory's `pred[0]` IS the template:
    # for 00122_outer, 00176_outer and 10004_outer it is not (8-9 cm median
    # vertex distance between the two, 0.6 mm for the other garments), so the
    # bound arms stood 5-14 cm from their sleeves and the sleeves, skinned from
    # those arms, stuck out from the wearer's (00122_outer: median sleeve to
    # arm 55 mm, p90 143 mm). The cache flags such a pose `refined` (a one-off
    # tool re-fitted it, 00148_upper too). None = off; "refined": only the
    # flagged poses; "poor": when the median sleeve-to-arm distance exceeds
    # REFINE_POOR; "always".
    # Measured against the real shirt (tools/evaluate/open_front.py, NLF body,
    # 30-38 frames each): garment IoU 00122_outer 0.526 -> 0.609 (sleeve to arm
    # 38 -> 14 mm, garment vertices inside the body 2.1 -> 0.9 %), 10004_outer
    # 0.319 -> 0.336, 00176_outer 0.494 -> 0.492, 00148_upper 0.802 -> 0.806;
    # the unflagged garments are unchanged to the bit.
    REFINE_ARMS = "refined"
    REFINE_POOR = 0.04  # m
    REFINE_STEPS = 50
    REFINE_LEARNING_RATE = 0.03
    REFINE_JOINTS = (16, 17, 18, 19, 20, 21)  # shoulders, elbows, wrists
    REFINE_CLAMP = 0.10  # m: a sleeve vertex further than this from every arm vertex does not pull
    REFINE_PRIOR = 1e-3  # on the squared rotation change, rad^2

    @staticmethod
    def sleeve_arm_distance(body, body_is_arm, aligned, sleeve):
        """Median distance (m) from the sleeve vertices to the nearest arm vertex of `body`."""
        from scipy.spatial import cKDTree

        if not sleeve.any():
            return 0.0
        return float(np.median(cKDTree(body[body_is_arm]).query(aligned[sleeve])[0]))

    def refine_arms(self, bind_fitter, aligned, sleeve, flagged=False):
        """Rotate the bind body's shoulders, elbows and wrists (in place) so its
        arms sit inside the garment's sleeves; True when the pose changed.

        Adam on the joints' axis-angle change, minimising the clamped nearest
        distance from each sleeve vertex to the posed ARM vertices (a sleeve is
        a tube around the arm, so this centres the arm in it) plus a small
        prior on the change. Accepted only if the median sleeve-to-arm distance
        falls."""
        if not self.REFINE_ARMS or not sleeve.any() or (self.REFINE_ARMS == "refined" and not flagged):
            return False
        weights = bind_fitter.model.lbs_weights.detach().numpy().astype(np.float64)
        body_is_arm = weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
        with torch.no_grad():
            before_body = bind_fitter._forward().vertices[0].numpy().astype(np.float64)
        before = self.sleeve_arm_distance(before_body, body_is_arm, aligned, sleeve)
        if self.REFINE_ARMS == "poor" and before <= self.REFINE_POOR:
            return False
        slices = [slice((j - 1) * 3, (j - 1) * 3 + 3) for j in self.REFINE_JOINTS]
        base = bind_fitter.body_pose.detach().clone()
        delta = torch.zeros(len(slices), 3, requires_grad=True)
        points = torch.as_tensor(aligned[sleeve][::3], dtype=torch.float32)
        arm_ids = torch.as_tensor(np.flatnonzero(body_is_arm)[::3])

        def posed(change):
            pose = base.clone()
            for k, part in enumerate(slices):
                pose[0, part] = base[0, part] + change[k]
            return bind_fitter.model(betas=bind_fitter.betas.detach(), body_pose=pose, global_orient=bind_fitter.global_orient.detach())

        optimiser = torch.optim.Adam([delta], self.REFINE_LEARNING_RATE)
        for _ in range(int(self.REFINE_STEPS)):
            optimiser.zero_grad()
            arm = posed(delta).vertices[0][arm_ids]
            nearest = torch.cdist(points, arm).min(dim=1).values.clamp(max=self.REFINE_CLAMP)
            loss = nearest.mean() + self.REFINE_PRIOR * (delta**2).sum()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            pose = posed(delta.detach())
            after_body = pose.vertices[0].numpy().astype(np.float64)
        after = self.sleeve_arm_distance(after_body, body_is_arm, aligned, sleeve)
        if after >= before:
            return False
        with torch.no_grad():
            for k, part in enumerate(slices):
                bind_fitter.body_pose[0, part] = base[0, part] + delta.detach()[k]
        return True

    @staticmethod
    def _sleeve_axes(bind_fitter, aligned, sleeve):
        """{"L": dir, "R": dir}: each sleeve's principal axis (PCA over its
        labelled vertices, split by the side of the body's midline),
        oriented away from that side's shoulder joint. A side with fewer
        than 30 labelled vertices is left out."""
        with torch.no_grad():
            joints = bind_fitter._forward().joints[0].numpy().astype(np.float64)
        mid_x = joints[0, 0]
        axes = {}
        for side, shoulder, on_side in (("L", 16, aligned[:, 0] > mid_x), ("R", 17, aligned[:, 0] < mid_x)):
            members = sleeve & on_side
            if members.sum() < 30:
                continue
            v = aligned[members]
            centred = v - v.mean(axis=0)
            axis = np.linalg.svd(centred, full_matrices=False)[2][0]
            if axis @ (v.mean(axis=0) - joints[shoulder]) < 0:
                axis = -axis
            axes[side] = axis
        return axes
