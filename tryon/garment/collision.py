"""Keep the garment outside the wearer's body.

Each garment vertex is tested against the tangent plane of its nearest body
vertex and pushed out along the normal if closer than `margin` (Muller et
al. 2007, section 3.3). A KD-tree keeps this cheap enough for every frame.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def scatter_rows(index, values, count):
    """`out[index[i]] += values[i]` for (N, 3) `values` into (count, 3); bincount per column, faster than `np.add.at`."""
    index = np.asarray(index)
    return np.stack([np.bincount(index, weights=values[:, c], minlength=count) for c in range(values.shape[1])], axis=1)


@njit(cache=True)
def _vertex_normals_csr(vertices, faces, offset, entry):
    """Per-vertex sums of face normals (2 x area) around each vertex.

    `entry[offset[v]:offset[v + 1]]` are the corners (`corner * n_faces + face`)
    at `v`, ascending, so sums match `scatter_rows` bit for bit. Serial on
    purpose: two parallel numba regions at once abort the workqueue."""
    n_faces = len(faces)
    face_normals = np.empty((n_faces, 3))
    for f in range(n_faces):
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        e1x, e1y, e1z = vertices[i1, 0] - vertices[i0, 0], vertices[i1, 1] - vertices[i0, 1], vertices[i1, 2] - vertices[i0, 2]
        e2x, e2y, e2z = vertices[i2, 0] - vertices[i0, 0], vertices[i2, 1] - vertices[i0, 1], vertices[i2, 2] - vertices[i0, 2]
        face_normals[f, 0] = e1y * e2z - e1z * e2y
        face_normals[f, 1] = e1z * e2x - e1x * e2z
        face_normals[f, 2] = e1x * e2y - e1y * e2x
    out = np.zeros((len(vertices), 3))
    for v in range(len(vertices)):
        x = y = z = 0.0
        for j in range(offset[v], offset[v + 1]):
            f = entry[j] % n_faces
            x += face_normals[f, 0]
            y += face_normals[f, 1]
            z += face_normals[f, 2]
        out[v, 0], out[v, 1], out[v, 2] = x, y, z
    return out


_CORNERS = {}  # (faces, n_vertices, offset, entry) of the last topology


def _corner_table(faces, n_vertices):
    """Vertex -> corners CSR for `_vertex_normals_csr`; cached per topology."""
    cached = _CORNERS.get("last")
    if cached is not None and cached[1] == n_vertices and (cached[0] is faces or np.array_equal(cached[0], faces)):
        return cached[2], cached[3]
    corner_vertex = faces.T.ravel()  # corner c * n_faces + f -> its vertex
    entry = np.argsort(corner_vertex, kind="stable").astype(np.int64)  # ascending corner within a vertex
    offset = np.zeros(n_vertices + 1, np.int64)
    np.cumsum(np.bincount(corner_vertex, minlength=n_vertices), out=offset[1:])
    _CORNERS["last"] = (faces.copy(), n_vertices, offset, entry)
    return offset, entry


def body_vertex_normals(body_vertices, body_faces):
    """Area-weighted outward per-vertex normals of a closed body mesh."""
    body_vertices = np.ascontiguousarray(body_vertices, np.float64)
    body_faces = np.ascontiguousarray(body_faces, np.int64)
    offset, entry = _corner_table(body_faces, len(body_vertices))
    normals = _vertex_normals_csr(body_vertices, body_faces, offset, entry)
    return normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)


def _body_vertex_normals_numpy(body_vertices, body_faces):
    """Reference numpy form; oracle for `body_vertex_normals`."""
    body_vertices = np.asarray(body_vertices, np.float64)
    body_faces = np.asarray(body_faces, np.int64)
    v0, v1, v2 = (body_vertices[body_faces[:, k]] for k in range(3))
    face_normals = np.cross(v1 - v0, v2 - v0)  # length = 2 x area: the averaging weight
    normals = scatter_rows(body_faces.T.ravel(), np.tile(face_normals, (3, 1)), len(body_vertices))
    return normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)


class BodySurface:
    """One frame's body for collision queries: vertices, normals, KD-tree."""

    def __init__(self, body_vertices, body_faces):
        self.vertices = np.asarray(body_vertices, np.float64)
        self.normals = body_vertex_normals(self.vertices, body_faces)
        # non-compact 32-point leaves: builds ~2.5x faster, same nearest vertex
        self.tree = cKDTree(self.vertices, leafsize=32, balanced_tree=False, compact_nodes=False)

    def nearest(self, points):
        """(nearest body vertex, its normal) for every row of `points`."""
        _, index = self.tree.query(points, k=1, workers=4)
        return self.vertices[index], self.normals[index]


@njit(cache=True)
def _signed_distances(points, index, vertices, normals):
    """Distance of every point above the tangent plane of body vertex `index[i]`."""
    out = np.empty(len(points))
    for i in range(len(points)):
        k = index[i]
        out[i] = (
            (points[i, 0] - vertices[k, 0]) * normals[k, 0]
            + (points[i, 1] - vertices[k, 1]) * normals[k, 1]
            + (points[i, 2] - vertices[k, 2]) * normals[k, 2]
        )
    return out


@njit(cache=True)
def _push_violations(points, index, signed, normals, margin, moved, moved_points):
    """Push points closer than `margin` out along their vertex normal, in place.

    `moved[:n]` are the pushed indices, `moved_points[:n]` their new positions
    (for the next KD query). Returns n."""
    n = 0
    for i in range(len(points)):
        violation = margin - signed[i]
        if violation > 0.0:
            k = index[i]
            points[i, 0] += violation * normals[k, 0]
            points[i, 1] += violation * normals[k, 1]
            points[i, 2] += violation * normals[k, 2]
            moved[n] = i
            moved_points[n, 0], moved_points[n, 1], moved_points[n, 2] = points[i, 0], points[i, 1], points[i, 2]
            n += 1
    return n


@njit(cache=True)
def _renew_nearest(points, moved, count, new_index, index, signed, vertices, normals):
    for j in range(count):
        i = moved[j]
        k = new_index[j]
        index[i] = k
        signed[i] = (
            (points[i, 0] - vertices[k, 0]) * normals[k, 0]
            + (points[i, 1] - vertices[k, 1]) * normals[k, 1]
            + (points[i, 2] - vertices[k, 2]) * normals[k, 2]
        )


def push_out_of_body(points, body, margin, passes=3):
    """Move `points` (N, 3) to at least `margin` outside the tangent plane of
    their nearest body vertex, in place, and return them.

    Repeats up to `passes` times: a push can land a point inside another body
    vertex's region (armpit, elbow). Serial numba kernels, bit-identical to
    `_push_out_of_body_numpy`."""
    if points.dtype != np.float64 or not points.flags.c_contiguous or len(points) == 0:
        return _push_out_of_body_numpy(points, body, margin, passes)
    vertices, normals = body.vertices, body.normals
    _, index = body.tree.query(points, k=1, workers=4)
    signed = _signed_distances(points, index, vertices, normals)
    moved = np.empty(len(points), np.int64)
    moved_points = np.empty_like(points)
    for step in range(passes):
        count = _push_violations(points, index, signed, normals, margin, moved, moved_points)
        if count == 0 or step == passes - 1:
            break
        _, renewed = body.tree.query(moved_points[:count], k=1, workers=4)
        _renew_nearest(points, moved, count, renewed, index, signed, vertices, normals)
    return points


def _push_out_of_body_numpy(points, body, margin, passes=3):
    """Numpy form of `push_out_of_body`: its oracle and the fallback for other dtypes."""
    nearest_vertex, nearest_normal = body.nearest(points)
    signed_distance = np.sum((points - nearest_vertex) * nearest_normal, axis=1)
    for _ in range(passes):
        violation = margin - signed_distance
        hit = violation > 0.0
        if not hit.any():
            break
        points[hit] += violation[hit, None] * nearest_normal[hit]
        # only moved points can have a new nearest vertex
        moved = np.flatnonzero(hit)
        vertex, normal = body.nearest(points[moved])
        nearest_vertex[moved], nearest_normal[moved] = vertex, normal
        signed_distance[moved] = np.sum((points[moved] - vertex) * normal, axis=1)
    return points
