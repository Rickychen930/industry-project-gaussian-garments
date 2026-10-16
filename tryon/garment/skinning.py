"""Bind a garment mesh to a body mesh so it deforms with the pose.

Each garment vertex is a weighted combination of its nearest body vertices
in the rest pose (the standard way to dress a parametric body; SMPL 2015,
DeePSD arXiv:2009.02715) and moves with them. This is skinning only, not
cloth simulation: a bound garment follows the body rigidly per
neighbourhood.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

DEFAULT_NEIGHBOURS = 6
# keeps a vertex lying exactly on a body vertex from getting an infinite weight
_EPSILON = 1e-6


def align_to_body(garment, body, top=None):
    """Put a garment into the body's coordinate frame.

    4D-DRESS garments are floor-origin while SMPL-X is pelvis-origin, so they
    differ by a translation. Aligns by the garment's own vertical extent.

    `top`: body height (y) the garment's top should sit at; default is the
    top of the head. Lower anchors (e.g. the neck) were measured worse:
    sleeves stopped following the arms. Nothing in the live pipeline passes it.
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
        # rest-pose displacement from the weighted body position, so the garment keeps its thickness
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
        # rest pose for `_local_rotations`
        binding._rest = body
        return binding

    def apply(self, body):
        """Where the garment sits when the body is in this pose.

        The offset is carried in a per-vertex local frame estimated from how
        the neighbourhood moved, so thickness survives rotation.
        """
        body = np.asarray(body, np.float64)
        anchored = (body[self.indices] * self.weights[..., None]).sum(axis=1)
        rotations = self._local_rotations(body)
        return anchored + np.einsum("nij,nj->ni", rotations, self.offsets)

    def _local_rotations(self, body):
        """Per-vertex rotation from the spread of each vertex's neighbours."""
        if self._rest is None:
            return np.broadcast_to(np.eye(3), (len(self.indices), 3, 3))
        rest = self._rest[self.indices]
        posed = body[self.indices]
        rest_centred = rest - rest.mean(axis=1, keepdims=True)
        posed_centred = posed - posed.mean(axis=1, keepdims=True)
        # Kabsch per neighbourhood. A collinear or coplanar neighbourhood has
        # no defined rotation (arbitrary spin drifts the garment sideways), so
        # detect it from the REST spread; the cross-covariance also carries
        # motion and cannot tell flat from stationary.
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


def piece_labels(faces, count):
    """Connected component (0..k-1) of every vertex over the mesh's faces."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces)
    both = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = sp.coo_matrix((np.ones(len(both)), (both[:, 0], both[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def _edge_profile(y, x, rows, bin_height):
    """Inner edge of a panel (smallest x per height), sampled at heights `rows`.

    Binned by `bin_height`, gaps interpolated, min over neighbouring bins."""
    bins = np.floor((y - y.min()) / bin_height).astype(int)
    lowest = np.full(bins.max() + 1, np.inf)
    np.minimum.at(lowest, bins, x)
    filled = np.isfinite(lowest)
    centres = (np.arange(len(lowest)) + 0.5) * bin_height + y.min()
    lowest = np.interp(centres, centres[filled], lowest[filled])
    lowest = np.minimum(lowest, np.minimum(np.r_[lowest[1:], lowest[-1]], np.r_[lowest[0], lowest[:-1]]))
    return np.interp(rows, centres, lowest)


def front_closure(rest, faces, seam=0.01, blend=0.2, bin_height=0.01, max_gap=0.15):
    """Displacement (N, 3) that brings the two front panels of an open-front garment together.

    `rest` is in the body's zero-pose frame (midline x = 0). Front panels are
    the connected components other than the largest (back, sleeves), one per
    side, whose inner edge is within `max_gap` of the midline. Each inner edge
    moves to the midline (`seam` apart), fading out by a smoothstep over
    `blend` metres so side seams and sleeves stay put. Zero when there are not
    two such panels."""
    rest = np.asarray(rest, np.float64)
    out = np.zeros_like(rest)
    labels = piece_labels(faces, len(rest))
    sizes = np.bincount(labels)
    if len(sizes) < 3:
        return out
    pieces = [p for p in np.argsort(-sizes)[1:] if sizes[p] >= 0.05 * len(rest)]
    side = {sign: [p for p in pieces if np.sign(rest[labels == p, 0].mean()) == sign] for sign in (1.0, -1.0)}
    if not side[1.0] or not side[-1.0]:
        return out
    for sign, candidates in side.items():
        piece = candidates[0]
        members = np.flatnonzero(labels == piece)
        lateral = sign * rest[members, 0]  # distance from the midline on the panel's own side
        edge = _edge_profile(rest[members, 1], lateral, rest[members, 1], bin_height)
        if lateral.min() > max_gap:
            continue
        shift = np.maximum(edge - 0.5 * seam, 0.0)
        distance = np.maximum(lateral - edge, 0.0)
        t = np.clip(1.0 - distance / max(blend, 1e-9), 0.0, 1.0)
        out[members, 0] = -sign * shift * t * t * (3.0 - 2.0 * t)
    return out


def drape_narrowing(rest, distance, sleeve, strength, loose_from=0.01, loose_span=0.06, start=0.3, span=0.4):
    """Displacement (N, 3), along x only, that narrows the loose lower torso toward the body's centre line.

    A loose shirt hangs in folds, narrower than the flat scanned panel.
    `rest` is in the zero-pose frame (centre line x = 0), `distance` the
    vertex-to-body distance at bind pose, `sleeve` the sleeve mask. A vertex
    moves by `strength` x looseness (0 at `loose_from` m off the body, 1 at
    `loose_from + loose_span`) x a height weight (0 until `start` of the way
    down, smoothstep to 1 over `span`) x its distance from the centre line.
    Sleeves do not move."""
    rest = np.asarray(rest, np.float64)
    out = np.zeros_like(rest)
    body = ~np.asarray(sleeve, bool)
    if not body.any():
        return out
    top, bottom = float(np.percentile(rest[body, 1], 98)), float(np.percentile(rest[body, 1], 2))
    t = np.clip((top - rest[:, 1]) / max(top - bottom, 1e-9), 0.0, 1.0)
    h = np.clip((t - start) / max(span, 1e-9), 0.0, 1.0)
    h = h * h * (3.0 - 2.0 * h)
    loose = np.clip((np.asarray(distance, np.float64) - loose_from) / loose_span, 0.0, 1.0)
    out[:, 0] = -strength * loose * h * rest[:, 0] * body
    return out


def closed_front_flag(checkpoint_dir):
    """Whether the garment is worn closed.

    Read from `assets/closed_front/<garment>.json` (wins) or
    `stage1/closed_front.json`, holding `true` or `{"closed": true}`. No file
    is False; malformed content raises ValueError so a typo cannot silently
    turn the closure off."""
    import json
    from pathlib import Path

    from tryon.paths import CLOSED_FRONT

    path = CLOSED_FRONT / f"{Path(checkpoint_dir).name}.json"
    if not path.exists():
        path = Path(checkpoint_dir) / "stage1" / "closed_front.json"
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if isinstance(value, dict):
        value = value.get("closed")
    if not isinstance(value, bool):
        raise ValueError(f'{path} must hold true/false or {{"closed": true/false}}')
    return value


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


@njit(cache=True)
def _skin(ptr, joints, values, transforms, rest_h):
    """Linear blend skinning over non-zero weights: `sum_j w_j (T_j @ rest_h)`.
    Serial so it never overlaps another thread's numba parallel region."""
    out = np.zeros((ptr.shape[0] - 1, 3))
    for n in range(ptr.shape[0] - 1):
        x, y, z, one = rest_h[n, 0], rest_h[n, 1], rest_h[n, 2], rest_h[n, 3]
        for k in range(ptr[n], ptr[n + 1]):
            t = transforms[joints[k]]
            w = values[k]
            out[n, 0] += w * (t[0, 0] * x + t[0, 1] * y + t[0, 2] * z + t[0, 3] * one)
            out[n, 1] += w * (t[1, 0] * x + t[1, 1] * y + t[1, 2] * z + t[1, 3] * one)
            out[n, 2] += w * (t[2, 0] * x + t[2, 1] * y + t[2, 2] * z + t[2, 3] * one)
    return out


class SkinnedGarment:
    """A garment skinned by the body's own joint transforms.

    `GarmentBinding` tears at the joints: vertices near an armpit bind to
    body vertices on different bones. Instead each garment vertex takes the
    distance-weighted blend of its neighbours' skinning weights
    (`lbs_weights`, 55 joints) and is skinned by the body's joint transforms
    (SMPL, Loper et al. 2015, Eq. 1). A wide neighbourhood (200) makes the
    weights vary smoothly across a joint.
    """

    # (N, 3, 3) linear part of each vertex's blended bind transform (zero-pose
    # frame -> scan pose); set by `build_in_pose`, None = identity.
    bind_linear = None
    # (weights, ptr, joints, values) of `apply`: non-zero weights as CSR, built lazily
    _csr = None

    def __init__(self, rest_vertices, weights, physical_rest=None):
        # `rest` is in the body model's zero-pose frame; `physical_rest` is the
        # garment as scanned, for true edge lengths (strain limiting, tearing).
        self.rest = np.asarray(rest_vertices, np.float64)
        self.weights = np.asarray(weights, np.float64)
        self.physical_rest = self.rest if physical_rest is None else np.asarray(physical_rest, np.float64)
        self._rest_h = np.concatenate([self.rest, np.ones((len(self.rest), 1))], axis=1)

    def scale(self, ratio):
        """Resize the garment about the zero-pose origin (the pelvis) to the wearer.
        `ratio`: one factor, or three (x, y, z)."""
        factor = np.asarray(ratio, np.float64).reshape(-1)
        factor = factor if factor.size == 3 else np.repeat(factor[:1], 3)
        self.rest = self.rest * factor
        self.physical_rest = self.physical_rest * factor
        self._rest_h = np.concatenate([self.rest, np.ones((len(self.rest), 1))], axis=1)
        return self

    # SMPL-X arm joints: shoulders, elbows, wrists, fingers
    ARM_JOINTS = (16, 17, 18, 19, 20, 21) + tuple(range(25, 55))
    # SMPL-X leg joints: hips, knees, ankles, feet
    LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
    # Factor on leg-joint skinning weights (rows renormalised): 1 = as
    # transferred; 0 = a hem that hangs from the pelvis (`GarmentRenderer.bind`).
    LEG_SCALE = 1.0

    @classmethod
    def build(cls, garment_vertices, fitter, neighbours=200, sleeve=None):
        """`sleeve`: optional boolean mask; sleeve vertices take weights only
        from arm body vertices, all others only from non-arm ones."""
        import torch

        garment = np.asarray(garment_vertices, np.float64)
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        body_weights = fitter.model.lbs_weights.detach().numpy().astype(np.float64)
        if cls.LEG_SCALE != 1.0:
            scaled = body_weights.copy()
            scaled[:, list(cls.LEG_JOINTS)] *= cls.LEG_SCALE
            total = scaled.sum(axis=1, keepdims=True)
            # a body vertex carried by the legs alone (a foot) keeps its weights
            body_weights = np.where(total > 1e-6, scaled / np.maximum(total, 1e-12), body_weights)
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
        # one sparse product; the dense gather needs an (N, 200, 55) array (~700 MB)
        from scipy.sparse import csr_matrix

        rows = np.repeat(np.arange(len(garment)), indices.shape[1])
        mixer = csr_matrix((blend.ravel(), (rows, indices.ravel())), shape=(len(garment), len(body)))
        return np.asarray(mixer @ body_weights)

    @staticmethod
    def _transfer_dense(garment, body, body_weights, neighbours):
        """Dense gather-and-sum; oracle for `_transfer`."""
        neighbours = int(min(max(neighbours, 1), len(body)))
        distance, indices = cKDTree(body).query(garment, k=neighbours)
        distance = np.atleast_2d(distance.T).T
        indices = np.atleast_2d(indices.T).T
        blend = 1.0 / (distance + _EPSILON)
        blend /= blend.sum(axis=1, keepdims=True)
        return (body_weights[indices] * blend[..., None]).sum(axis=1)

    @classmethod
    def build_in_pose(cls, garment_vertices, posed_fitter, neighbours=200, sleeve=None):
        """Skin a garment laid out in the pose `posed_fitter` currently holds.

        `build` assumes the zero pose (T-pose), but scans have arms hanging;
        binding to a T-posed body gives sleeves torso and leg weights. Bind on
        a body posed like the scan (`GarmentRenderer.bind`), then carry each
        vertex into the zero-pose frame by the inverse of its blended
        transform (exact round trip).

        `sleeve` (see `build`) stops body panels next to hanging arms from
        taking arm weights (`scan_pose.ScanPoseBinder.bind`).
        """
        garment = np.asarray(garment_vertices, np.float64)
        skinned = cls.build(garment, posed_fitter, neighbours=neighbours, sleeve=sleeve)
        transforms = posed_fitter.joint_transforms().detach().numpy().astype(np.float64)
        per_vertex = np.einsum("nj,jab->nab", skinned.weights, transforms)
        homogeneous = np.concatenate([garment, np.ones((len(garment), 1))], axis=1)
        zero_frame = np.einsum("nab,nb->na", np.linalg.inv(per_vertex), homogeneous)[:, :3]
        built = cls(zero_frame, skinned.weights, physical_rest=garment)
        built.bind_linear = per_vertex[:, :3, :3]
        return built

    def moved(self, offset):
        """A copy with rest displaced by `offset` ((N, 3), zero-pose frame) and
        physical rest by the same displacement carried to the scan pose, so
        strain limiting and torn-face tests use the moved edge lengths
        (`front_closure`)."""
        offset = np.asarray(offset, np.float64)
        if offset.shape != self.rest.shape:
            raise ValueError(f"offset {offset.shape} does not match the garment's {self.rest.shape}")
        carried = offset if self.bind_linear is None else np.einsum("nij,nj->ni", self.bind_linear, offset)
        moved = SkinnedGarment(self.rest + offset, self.weights, physical_rest=self.physical_rest + carried)
        moved.bind_linear = self.bind_linear
        return moved

    def apply(self, fitter):
        """Where the garment sits in the fitter's current pose (sparse linear blend skinning)."""
        transforms = np.ascontiguousarray(fitter.joint_transforms().detach().numpy().astype(np.float64))
        if self._csr is None or self._csr[0] is not self.weights:
            counts = (self.weights != 0.0).sum(axis=1)
            ptr = np.zeros(len(self.weights) + 1, np.int64)
            np.cumsum(counts, out=ptr[1:])
            rows, joints = np.nonzero(self.weights)  # row-major: grouped by vertex, ascending joint
            self._csr = (self.weights, ptr, joints.astype(np.int64), np.ascontiguousarray(self.weights[rows, joints]))
        _, ptr, joints, values = self._csr
        return _skin(ptr, joints, values, transforms, np.ascontiguousarray(self._rest_h))

    def save(self, path):
        np.savez_compressed(path, rest=self.rest, weights=self.weights, physical_rest=self.physical_rest)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["rest"], data["weights"], physical_rest=data.get("physical_rest"))
