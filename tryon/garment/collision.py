"""Keep the garment outside the wearer's body.

The body is the fitted SMPL-X mesh. For every garment vertex the nearest
body vertex and its outward normal give a local tangent plane; a vertex
closer to that plane than `margin` (or behind it) is pushed out along the
normal (Muller et al. 2007, position-based dynamics, section 3.3). A KD-tree
over the body's vertices makes this O(n log m) per frame, cheap enough to
run every frame on the ~8,000-vertex garments.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def scatter_rows(index, values, count):
    """`out[index[i]] += values[i]` for (N, 3) `values` into (count, 3):
    `np.add.at`'s sum, by one `np.bincount` per column (~8x faster)."""
    index = np.asarray(index)
    return np.stack([np.bincount(index, weights=values[:, c], minlength=count) for c in range(values.shape[1])], axis=1)


@njit(cache=True)
def _vertex_normals_csr(vertices, faces, offset, entry):
    """Per-vertex sums of the face normals (2 x area, `np.cross`'s arithmetic)
    around each vertex. `entry[offset[v]:offset[v + 1]]` are the corners
    (`corner * n_faces + face`) at vertex `v`, in ascending order - the order
    `np.bincount` added them in, so the sums are bit-identical to
    `scatter_rows`. Serial: the shading workers run compiled code on their
    own threads, and two parallel regions at once abort numba's workqueue."""
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


_CORNERS = {}  # one entry: (faces, n_vertices, offset, entry) of the last mesh topology


def _corner_table(faces, n_vertices):
    """The vertex -> corners CSR of `_vertex_normals_csr`; depends only on the
    topology, so it is built once per mesh (the body's faces never change)."""
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
    """Outward per-vertex normals of a closed body mesh: the area-weighted
    average of the normals of the faces around each vertex."""
    body_vertices = np.ascontiguousarray(body_vertices, np.float64)
    body_faces = np.ascontiguousarray(body_faces, np.int64)
    offset, entry = _corner_table(body_faces, len(body_vertices))
    normals = _vertex_normals_csr(body_vertices, body_faces, offset, entry)
    return normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)


def _body_vertex_normals_numpy(body_vertices, body_faces):
    """The original (cross, tile, three bincounts): the oracle for `body_vertex_normals`."""
    body_vertices = np.asarray(body_vertices, np.float64)
    body_faces = np.asarray(body_faces, np.int64)
    v0, v1, v2 = (body_vertices[body_faces[:, k]] for k in range(3))
    face_normals = np.cross(v1 - v0, v2 - v0)  # length = 2 x area: the averaging weight
    normals = scatter_rows(body_faces.T.ravel(), np.tile(face_normals, (3, 1)), len(body_vertices))
    return normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)


class BodySurface:
    """One frame's body for collision queries: vertices, outward normals
    and a KD-tree over the vertices."""

    def __init__(self, body_vertices, body_faces):
        self.vertices = np.asarray(body_vertices, np.float64)
        self.normals = body_vertex_normals(self.vertices, body_faces)
        self.tree = cKDTree(self.vertices)

    def nearest(self, points):
        """(nearest body vertex, its normal) for every row of `points`.
        Four query threads: 0.72 -> 0.41 ms for 1,000 points against the
        10,475-vertex body (more threads cost more than they save)."""
        _, index = self.tree.query(points, k=1, workers=4)
        return self.vertices[index], self.normals[index]


def push_out_of_body(points, body, margin, passes=3):
    """`points` (N, 3) with every point moved to at least `margin` outside
    the tangent plane of its nearest body vertex, in place, and returned.

    Repeated up to `passes` times because a push can carry a point into a
    different body vertex's neighbourhood (an armpit, the inside of an
    elbow) where it is still inside; stops early once nothing is."""
    nearest_vertex, nearest_normal = body.nearest(points)
    signed_distance = np.sum((points - nearest_vertex) * nearest_normal, axis=1)
    for _ in range(passes):
        violation = margin - signed_distance
        hit = violation > 0.0
        if not hit.any():
            break
        points[hit] += violation[hit, None] * nearest_normal[hit]
        # only the points just moved can have a new nearest body vertex
        moved = np.flatnonzero(hit)
        vertex, normal = body.nearest(points[moved])
        nearest_vertex[moved], nearest_normal[moved] = vertex, normal
        signed_distance[moved] = np.sum((points[moved] - vertex) * normal, axis=1)
    return points
