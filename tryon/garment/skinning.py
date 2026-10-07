"""Bind a garment mesh to a body mesh so it deforms with the pose.

This is what makes 360 degrees and motion-following structural. The 2D
system faked rotation with a catalog of 24 pre-rendered views and faked
motion by warping anchors; both are approximations of something a skinned
mesh does exactly. A garment bound here turns when the body turns and bends
when the body bends, because its vertices are carried by the body's.

Each garment vertex is expressed as a weighted combination of its nearest
body vertices in the rest pose, and thereafter moves with them. This is the
standard way to dress a parametric body (SMPL, SIGGRAPH Asia 2015), and the
same nearest-body-vertex initialisation DeePSD (arXiv:2009.02715) uses
before learning a correction on top.

What it does not do: simulate cloth. A bound garment follows the body
rigidly-per-neighbourhood, so it will not swing or wrinkle on its own. That
is a deliberate scope line - the goal is a garment that sticks to the body
accurately through a turn, and dynamics can be added later on top of a
correct binding. It could not be added on top of the 2D warp at all.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_NEIGHBOURS = 6
# Distance weighting falls off smoothly; the epsilon keeps a garment vertex
# that lands exactly on a body vertex from producing an infinite weight.
_EPSILON = 1e-6


def align_to_body(garment, body, top=None):
    """Put a garment into the body's coordinate frame.

    4D-DRESS garments are floor-origin (measured: y=1.36 for a top) while
    SMPL-X is pelvis-origin at y=0, so the two are a translation apart
    rather than a different shape. Aligning by the vertical extent of the
    region the garment actually covers - not the whole body, most of which
    is legs a top never reaches - brought garment vertices to a median
    7.7 cm from the body surface.

    `top`: the body height (y) the garment's topmost vertices should sit
    at; default (None) is the top of the head. A lower anchor (the SMPL-X
    neck joint, 31 cm below the head top) was tried after an end-to-end
    render on a real annotated frame showed the garment apparently
    covering the wearer's face - and measured to be WRONG, in two ways:
    (1) sleeve-to-arm skinning collapsed from 22.4% of garment vertices
    bound mainly to arm joints (head-top) to 1.0% (neck), monotonically
    with every centimetre lowered, so the sleeves stopped following the
    arms (`tests/test_garment_renderer.py::test_the_garment_follows_the_arms`
    caught it); (2) the garment checkpoint used (00152_outer) is 1.02 m
    tall and only 15 cm wide at its top - a knee-length coat scanned with
    its HOOD UP, whose top genuinely sits at the top of the head. The face
    was not covered by a misplaced collar but by the hood's BACK panel
    showing through where the head should occlude it, which is a body
    occlusion problem for the renderer, not an alignment one. The median
    garment-to-body distance that originally justified this alignment is
    ~5.5 cm for every shift tried, so it cannot discriminate placement
    either way; arm-skinning fraction can, and it favours the head top.
    `top` is kept as an option for garments whose top is known to sit
    elsewhere (a vest, a skirt); nothing in the live pipeline passes it.
    """
    garment = np.asarray(garment, np.float64)
    body = np.asarray(body, np.float64)
    if garment.size == 0 or body.size == 0:
        return garment
    height = garment[:, 1].max() - garment[:, 1].min()
    if height <= 0:
        return garment - garment.mean(axis=0) + body.mean(axis=0)
    shift = np.zeros(3)
    shift[1] = (body[:, 1].max() if top is None else float(top)) - garment[:, 1].max()
    shift[0] = body[:, 0].mean() - garment[:, 0].mean()
    shift[2] = body[:, 2].mean() - garment[:, 2].mean()
    return garment + shift


class GarmentBinding:
    """Garment vertices carried by the nearest body vertices."""

    _rest = None

    def __init__(self, indices, weights, offsets):
        self.indices = np.asarray(indices, np.int64)
        self.weights = np.asarray(weights, np.float64)
        # The rest-pose displacement from the weighted body position, so a
        # garment keeps its own thickness and drape rather than collapsing
        # onto the skin.
        self.offsets = np.asarray(offsets, np.float64)

    @classmethod
    def build(cls, garment, body, neighbours=DEFAULT_NEIGHBOURS):
        garment = np.asarray(garment, np.float64)
        body = np.asarray(body, np.float64)
        neighbours = int(min(max(neighbours, 1), len(body)))
        distance, indices = cKDTree(body).query(garment, k=neighbours)
        distance = np.atleast_2d(distance.T).T
        indices = np.atleast_2d(indices.T).T
        weights = 1.0 / (distance + _EPSILON)
        weights /= weights.sum(axis=1, keepdims=True)
        anchored = (body[indices] * weights[..., None]).sum(axis=1)
        binding = cls(indices, weights, garment - anchored)
        # The rest pose is what `_local_rotations` measures movement against.
        binding._rest = body
        return binding

    def apply(self, body):
        """Where the garment sits when the body is in this pose.

        The offset is carried in the body's LOCAL frame, so a garment keeps
        its thickness through a rotation instead of being sheared flat. The
        frame is estimated per vertex from how its neighbourhood moved.
        """
        body = np.asarray(body, np.float64)
        anchored = (body[self.indices] * self.weights[..., None]).sum(axis=1)
        rotations = self._local_rotations(body)
        return anchored + np.einsum("nij,nj->ni", rotations, self.offsets)

    def _local_rotations(self, body):
        """Per-vertex rotation, from the spread of each vertex's neighbours.

        Without this the offset stays in world space and a garment turned
        90 degrees keeps pointing the way it did at rest - the garment would
        sink into the body on one side and float off it on the other.
        """
        if self._rest is None:
            return np.broadcast_to(np.eye(3), (len(self.indices), 3, 3))
        rest = self._rest[self.indices]
        posed = body[self.indices]
        rest_centred = rest - rest.mean(axis=1, keepdims=True)
        posed_centred = posed - posed.mean(axis=1, keepdims=True)
        # Kabsch per neighbourhood: the rotation best explaining the move.
        # A neighbourhood that is collinear or coplanar does not determine a
        # rotation: the SVD returns an arbitrary spin about the degenerate
        # axis, which shows up as a garment drifting sideways under a pure
        # translation. Judge that from the REST neighbourhood's own spread -
        # the cross-covariance below is rank-deficient for the same reason
        # but its singular values also carry the motion, so it cannot tell
        # a flat neighbourhood from a stationary one.
        spread = np.linalg.svd(
            np.einsum("nki,nkj->nij", rest_centred, rest_centred),
            compute_uv=False,
        )
        degenerate = spread[:, -1] <= 1e-8 * np.maximum(spread[:, 0], 1e-12)
        covariance = np.einsum("nki,nkj->nij", rest_centred, posed_centred)
        u, _singular, vt = np.linalg.svd(covariance)
        determinant = np.linalg.det(np.einsum("nij,njk->nik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
        correction = np.broadcast_to(np.eye(3), (len(rest), 3, 3)).copy()
        correction[:, 2, 2] = np.sign(determinant)
        rotations = np.einsum(
            "nij,njk,nkl->nil", vt.transpose(0, 2, 1), correction, u.transpose(0, 2, 1)
        )
        rotations[degenerate] = np.eye(3)
        return rotations

    def save(self, path):
        np.savez_compressed(
            path,
            indices=self.indices,
            weights=self.weights,
            offsets=self.offsets,
            rest=self._rest if self._rest is not None else np.zeros((0, 3)),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        binding = cls(data["indices"], data["weights"], data["offsets"])
        rest = data["rest"]
        binding._rest = rest if len(rest) else None
        return binding




def _vertex_normals(vertices, faces):
    face_normals = _face_normals(vertices, faces)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _face_normals(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


class SkinnedGarment:
    """A garment skinned by the body's own joint transforms.

    `GarmentBinding` copies positions from the nearest body vertices, and it
    tears at the joints: garment vertices near an armpit bind to body
    vertices on different bones, which move apart when the arm rises. 332
    faces stretched up to 58x, and their splats ballooned into a smear.
    Smoothing over more neighbours barely helped (p99 7.2 -> 5.5) - distance
    is the wrong axis.

    The bone is the right axis. Every SMPL-X body vertex carries a skinning
    weight over the 55 joints (`lbs_weights`); a garment vertex takes the
    distance-weighted blend of its neighbours' weights and is skinned by the
    body's joint transforms directly. That is how SMPL skins its own mesh
    (Loper et al. 2015, Eq. 1), applied to a mesh the body model did not
    ship with. Arms raised 1.2 rad, edge-stretch p99 7.24 -> 1.89 and torn
    faces 331 -> 2 at 200 neighbours; the wide neighbourhood is what makes
    the weights vary smoothly across a joint instead of jumping bone to bone.
    """

    def __init__(self, rest_vertices, weights, physical_rest=None):
        # `rest` is in the body model's ZERO-pose frame (what the joint
        # transforms are relative to); `physical_rest` is the garment as it
        # actually is - the scan pose - for anything that needs true edge
        # lengths (strain limiting, tearing).
        self.rest = np.asarray(rest_vertices, np.float64)
        self.weights = np.asarray(weights, np.float64)
        self.physical_rest = self.rest if physical_rest is None else np.asarray(physical_rest, np.float64)
        self._rest_h = np.concatenate([self.rest, np.ones((len(self.rest), 1))], axis=1)

    def scale(self, ratio):
        """Scale the garment about the zero-pose origin (the pelvis): a
        garment bound at the scan subject's size, resized to the wearer.
        `ratio`: one factor, or three (x, y, z) for an anisotropic resize.
        Rest, physical rest and the homogeneous cache all move together."""
        factor = np.asarray(ratio, np.float64).reshape(-1)
        factor = factor if factor.size == 3 else np.repeat(factor[:1], 3)
        self.rest = self.rest * factor
        self.physical_rest = self.physical_rest * factor
        self._rest_h = np.concatenate([self.rest, np.ones((len(self.rest), 1))], axis=1)
        return self

    # SMPL-X joints of the arms: shoulders, elbows, wrists and the fingers.
    ARM_JOINTS = (16, 17, 18, 19, 20, 21) + tuple(range(25, 55))

    @classmethod
    def build(cls, garment_vertices, fitter, neighbours=200, sleeve=None):
        """`sleeve`: optional boolean mask over garment vertices. Sleeve
        vertices then take their weights only from ARM body vertices and
        every other vertex only from non-arm ones (see `build_in_pose`)."""
        import torch

        garment = np.asarray(garment_vertices, np.float64)
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        body_weights = fitter.model.lbs_weights.detach().numpy().astype(np.float64)
        if sleeve is None:
            return cls(garment, cls._transfer(garment, body, body_weights, neighbours))
        sleeve = np.asarray(sleeve, bool)
        body_is_arm = body_weights[:, list(cls.ARM_JOINTS)].sum(axis=1) > 0.5
        weights = np.zeros((len(garment), body_weights.shape[1]))
        for garment_mask, body_mask in ((sleeve, body_is_arm), (~sleeve, ~body_is_arm)):
            if garment_mask.any():
                ids = np.flatnonzero(body_mask)
                weights[garment_mask] = cls._transfer(garment[garment_mask], body[ids], body_weights[ids], neighbours)
        return cls(garment, weights)

    @staticmethod
    def _transfer(garment, body, body_weights, neighbours):
        neighbours = int(min(max(neighbours, 1), len(body)))
        distance, indices = cKDTree(body).query(garment, k=neighbours)
        distance = np.atleast_2d(distance.T).T
        indices = np.atleast_2d(indices.T).T
        blend = 1.0 / (distance + _EPSILON)
        blend /= blend.sum(axis=1, keepdims=True)
        # sum_k blend[n, k] * body_weights[indices[n, k]] as one sparse
        # product: the gather it replaced built an (N, 200, 55) float64
        # array (~700 MB for a garment) - ~1 s of each of the four calls
        # in a garment binding. Equal to 1e-15.
        from scipy.sparse import csr_matrix

        rows = np.repeat(np.arange(len(garment)), indices.shape[1])
        mixer = csr_matrix((blend.ravel(), (rows, indices.ravel())), shape=(len(garment), len(body)))
        return np.asarray(mixer @ body_weights)

    @staticmethod
    def _transfer_dense(garment, body, body_weights, neighbours):
        """The original gather-and-sum; the oracle for `_transfer`."""
        neighbours = int(min(max(neighbours, 1), len(body)))
        distance, indices = cKDTree(body).query(garment, k=neighbours)
        distance = np.atleast_2d(distance.T).T
        indices = np.atleast_2d(indices.T).T
        blend = 1.0 / (distance + _EPSILON)
        blend /= blend.sum(axis=1, keepdims=True)
        return (body_weights[indices] * blend[..., None]).sum(axis=1)

    @classmethod
    def build_in_pose(cls, garment_vertices, posed_fitter, neighbours=200, sleeve=None):
        """Skin a garment given IN THE POSE `posed_fitter` currently holds.

        `build` assumes the garment is laid out in the body's zero pose
        (SMPL-X's T-pose) because `apply` uses joint transforms relative to
        that pose. A scanned garment is not: the real coat (00152_outer)
        was captured with its arms hanging, and binding it to a T-posed
        body put its sleeves beside the hips, where they took torso and
        leg weights - measured: sleeve axes 57 and 67 degrees off the arms
        on a T-pose frame, sleeves hanging to the hip. Binding on a body
        posed like the scan instead (shoulders rotated down; see
        `render.GarmentRenderer.bind` for how that pose is found), then
        carrying every vertex into the zero-pose frame by the inverse of
        its own blended transform, gives 4.7 and 26 degrees on the same
        frame with the sleeves along the arms. The round trip (zero frame
        -> bind pose) reproduces the input to 0.000 mm.

        `sleeve` (see `build`): with the arms hanging, the body panels sit
        right next to the arms and nearest-neighbour transfer gives them
        arm weights too - measured at 80 degrees: 2,903 torn edges when
        re-posed to a T-pose, 1,120 of them between torso-weighted
        vertices. Restricting sleeve vertices to arm body vertices and the
        rest to non-arm ones (two-pass binding, see
        `render.scan_pose.ScanPoseBinder.bind`) brought that to 278 torn
        edges with the sleeves 27 / 16 degrees off the arms.
        """
        garment = np.asarray(garment_vertices, np.float64)
        skinned = cls.build(garment, posed_fitter, neighbours=neighbours, sleeve=sleeve)
        transforms = posed_fitter.joint_transforms().detach().numpy().astype(np.float64)
        per_vertex = np.einsum("nj,jab->nab", skinned.weights, transforms)
        homogeneous = np.concatenate([garment, np.ones((len(garment), 1))], axis=1)
        zero_frame = np.einsum("nab,nb->na", np.linalg.inv(per_vertex), homogeneous)[:, :3]
        return cls(zero_frame, skinned.weights, physical_rest=garment)

    def apply(self, fitter):
        """Where the garment sits in the fitter's current pose."""
        transforms = fitter.joint_transforms().detach().numpy().astype(np.float64)
        per_vertex = np.einsum("nj,jab->nab", self.weights, transforms)
        return np.einsum("nab,nb->na", per_vertex, self._rest_h)[:, :3]

    def save(self, path):
        np.savez_compressed(path, rest=self.rest, weights=self.weights, physical_rest=self.physical_rest)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["rest"], data["weights"], physical_rest=data.get("physical_rest"))
