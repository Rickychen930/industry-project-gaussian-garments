"""Bind a garment to the body in the pose it was scanned in.

Skin weights taken on a T-posed body put the sleeves beside the hips, so
`ScanPoseBinder.bind` poses a body like the scan (the cached scan-frame
SMPL-X fit from tools/data/fit_scan_pose.py, or a default arm angle),
labels sleeve vertices, and returns the body and garment to bind on
(`GarmentRenderer.bind`). Wearer/scan size ratios live here too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tryon import paths
from tryon.garment.skinning import SkinnedGarment, align_to_body


_neutral_body_height = None


def _neutral_height():
    """Height (rest pose, beta=0) of the SMPL-X body in use; measured and cached (BodyFitter loads from disk)."""
    global _neutral_body_height
    if _neutral_body_height is None:
        from tryon.body.fitter import BodyFitter

        with torch.no_grad():
            vertices = BodyFitter()._forward().vertices[0].numpy()
        _neutral_body_height = float(vertices[:, 1].max() - vertices[:, 1].min())
    return _neutral_body_height


def scan_pose_file(checkpoint_dir):
    """(path, source) of the scan pose a garment binds on.

    "asset" = assets/scan_poses/<garment>.npz (authoritative), "cached" =
    the checkpoint's stage1/scan_pose.npz, else (None, "missing")."""
    if checkpoint_dir is None:
        return None, "missing"
    asset = paths.SCAN_POSES / f"{Path(checkpoint_dir).name}.npz"
    if asset.exists():
        return asset, "asset"
    cached = Path(checkpoint_dir) / "stage1" / "scan_pose.npz"
    if cached.exists():
        return cached, "cached"
    return None, "missing"


def unique_edges(faces):
    """(E, 2) undirected edges, each once, as sorted vertex pairs in lexicographic order."""
    faces = np.asarray(faces)
    both = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(both, axis=1), axis=0)


class ScanPoseBinder:
    """The scan-pose binding of one garment (rest mesh, faces, checkpoint directory)."""

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

    # Shoulder angle (degrees below horizontal) of the body the garment is
    # bound on. 30 leaves sleeves unbound to the arms; 80 tears body panels
    # with arm weights; label at 30 then bind at 60 was the best compromise.
    SCAN_BIND_ANGLE = 60.0

    @staticmethod
    def chest_ratio(fitter, bind_fitter):
        """Zero-pose chest width (x extent of non-arm vertices within 2 cm of
        spine3 height) of `fitter`'s body over `bind_fitter`'s."""
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

    # How wearer/scan size ratios become the garment's (x, y, z) scale
    # (`size_factors`); "current" is the live behaviour.
    #   current   height ratio on all axes (+ chest ratio >= 1 if WIDEN_TO_WEARER)
    #   no_chest  as current, chest ratio never used
    #   widen     height ratio, and chest ratio >= 1 on x / z
    #   none      no resizing
    #   shrink    ratios capped at 1
    #   damped    ratios ** 0.5
    #   frozen    height ratio = median of the first FREEZE_FRAMES frames, then held
    #   height_y  height ratio on y only
    SIZE_MODES = ("current", "no_chest", "widen", "none", "shrink", "damped", "frozen", "height_y")
    FREEZE_FRAMES = 30

    @staticmethod
    def size_factors(mode, height, chest, widen=False):
        """(3,) x, y, z garment scale for `mode` from the wearer/scan `height` and `chest` ratios."""
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
        """The cached scan-frame SMPL-X pose (tools/data/fit_scan_pose.py, fitted
        to the authors' `obstacle[0]` body); the repository's copy wins
        (`scan_pose_file`). None when there is neither."""
        path, _source = scan_pose_file(self.checkpoint_dir)
        if path is None:
            return None
        data = np.load(path)
        pose = {key: data[key] for key in ("body_pose", "global_orient", "betas", "offset")}
        # re-fitted because the trajectory's first frame was not this template (`REFINE_ARMS`)
        pose["refined"] = bool("refined" in data.files and int(data["refined"]))
        return pose

    def _posed_like_scan(self, fitter, theta, arm_axes=None, scan_pose=None):
        """A fitter with this body's shape and camera, arms `theta` degrees
        below horizontal, and the garment aligned to it.

        `arm_axes`: optional {"L": unit vector, "R": unit vector}, the
        shoulder-to-wrist direction per arm in the body frame (from the scan's
        sleeves); replaces the in-plane `theta` for that arm."""
        from tryon.body.fitter import BodyFitter

        bind_fitter = BodyFitter()
        with torch.no_grad():
            bind_fitter.betas[:] = fitter.betas.detach()
            bind_fitter.scale.fill_(float(fitter.scale.detach()))
            bind_fitter.translation[:] = fitter.translation.detach()
            bind_fitter.global_orient.zero_()
            bind_fitter.body_pose.zero_()
            # shoulders (joints 16, 17) rotate about z: the +x arm goes down
            bind_fitter.body_pose[0, (16 - 1) * 3 + 2] = -np.radians(theta)
            bind_fitter.body_pose[0, (17 - 1) * 3 + 2] = +np.radians(theta)
            if scan_pose is not None:
                # The true scan body: reproduces obstacle[0] in the template's
                # own frame, so the garment needs no alignment, settle or
                # scaling (as in Gaussian-Garments' lbs.py).
                bind_fitter.betas[:] = torch.tensor(np.asarray(scan_pose["betas"], np.float32))[None]
                bind_fitter.body_pose[:] = torch.tensor(np.asarray(scan_pose["body_pose"], np.float32))[None]
                bind_fitter.global_orient[:] = torch.tensor(np.asarray(scan_pose["global_orient"], np.float32))[None]
                # The body stays at the origin; the garment moves by the negative offset.
            if arm_axes:
                # the shoulder's parent frame is the zero-posed torso, so the
                # rotation from the rest arm direction to the target is the axis-angle
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

    # vertical shifts tried below the head-top alignment, in metres
    VERTICAL_SEARCH = np.arange(-0.40, 0.051, 0.025)

    def _settle_vertically(self, aligned, body):
        """Slide the garment vertically to where it sits closest to the body.

        `align_to_body` puts the garment top at the head top, which is only
        right for hooded coats; other collars end up over the face. The needed
        offset is a property of the garment, so take the shift in
        `VERTICAL_SEARCH` with the smallest median garment-to-body distance.
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

    # majority-vote passes over the garment mesh that clean the sleeve label
    SLEEVE_LABEL_VOTES = 10

    def bind(self, fitter):
        """Two-pass binding. Returns (bind_fitter, angle, aligned garment, sleeve mask).

        A vertex is labelled sleeve when its nearest ARM body vertex is closer
        than its nearest non-arm one, cleaned by `SLEEVE_LABEL_VOTES` majority
        votes over the garment mesh (smoothing the labels works; smoothing
        the skin weights did not). The garment is then bound with each vertex
        restricted to its own kind of body vertex (`SkinnedGarment.build`).
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
            # Second pass: the scan's sleeves show arms held forward of the
            # body plane, which no in-plane angle reproduces. Each sleeve's
            # principal axis becomes that arm's direction for a re-bind.
            arm_axes = self._sleeve_axes(bind_fitter, aligned, sleeve)
            if arm_axes:
                bind_fitter, aligned = self._posed_like_scan(fitter, self.SCAN_BIND_ANGLE, arm_axes)
                sleeve = label(bind_fitter, aligned)
        return bind_fitter, self.SCAN_BIND_ANGLE, aligned, sleeve

    SCAN_AXIS_REFINE = True

    # Refit the arms of a cached scan pose to the garment's sleeves. For some
    # garments the cached body is not the template's body, so bound arms stand
    # 5-14 cm from their sleeves; the cache flags those poses `refined`.
    # None = off; "refined": flagged poses only; "poor": when the median
    # sleeve-to-arm distance exceeds REFINE_POOR; "always".
    REFINE_ARMS = "refined"
    REFINE_POOR = 0.04  # m
    REFINE_STEPS = 50
    REFINE_LEARNING_RATE = 0.03
    REFINE_JOINTS = (16, 17, 18, 19, 20, 21)  # shoulders, elbows, wrists
    REFINE_CLAMP = 0.10  # m: farther sleeve vertices do not pull
    REFINE_PRIOR = 1e-3  # on the squared rotation change, rad^2

    @staticmethod
    def sleeve_arm_distance(body, body_is_arm, aligned, sleeve):
        """Median distance (m) from the sleeve vertices to the nearest arm vertex of `body`."""
        from scipy.spatial import cKDTree

        if not sleeve.any():
            return 0.0
        return float(np.median(cKDTree(body[body_is_arm]).query(aligned[sleeve])[0]))

    def refine_arms(self, bind_fitter, aligned, sleeve, flagged=False):
        """Rotate the bind body's shoulders, elbows and wrists (in place) so the
        arms sit inside the sleeves; True when the pose changed.

        Adam on the axis-angle change, minimising the clamped nearest distance
        from sleeve vertices to arm vertices plus a small prior; accepted only
        if the median sleeve-to-arm distance falls."""
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
        """{"L": dir, "R": dir}: each sleeve's principal axis (PCA, split at the
        body midline), pointing away from the shoulder. Sides with fewer than
        30 labelled vertices are omitted."""
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
