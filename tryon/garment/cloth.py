"""Cheap garment secondary motion for the live loop.

A full position-based solver was no more accurate than plain skinning and
cost 140-190 ms a frame; only its collision mattered. Instead every vertex
follows its skinned position through a damped spring (loose where the
garment hangs off the body, stiff where it lies on it), is clamped to a
per-vertex radius, then pushed out of the body (`collision.push_out_of_body`).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from tryon.garment.collision import BodySurface, push_out_of_body

SUBSTEP = 1.0 / 120.0  # s: semi-implicit Euler is stable here for omega * SUBSTEP < ~1
MAX_DT = 0.5  # s: a longer gap restarts from the skin; 0.25 reset every frame on a loaded machine


@njit(cache=True)
def _springs_serial(x, v, start, target, looseness, stiff_omega, loose_omega, stiff_radius, loose_radius, damping, n, h):
    """Serial, bit-identical form of `LightCloth._springs_numpy`.

    `n` semi-implicit Euler substeps of `h` toward a goal moving linearly from
    `start` to `target`, then the radius clamp. Returns new (x, v)."""
    count = len(x)
    x_new = np.empty((count, 3))
    v_new = np.empty((count, 3))
    over = np.zeros(count, np.bool_)
    any_over = False
    clamp = np.ones(count)
    for p in range(count):
        w = looseness[p]
        omega = stiff_omega + (loose_omega - stiff_omega) * w
        radius = stiff_radius + (loose_radius - stiff_radius) * w
        stiffness = omega * omega
        friction = 2.0 * damping * omega
        for c in range(3):
            xc = x[p, c]
            vc = v[p, c]
            for i in range(1, n + 1):
                goal = start[p, c] + (target[p, c] - start[p, c]) * (i / n)
                a = stiffness * (goal - xc) - friction * vc
                vc = vc + h * a
                xc = xc + h * vc
            x_new[p, c] = xc
            v_new[p, c] = vc
        o0 = x_new[p, 0] - target[p, 0]
        o1 = x_new[p, 1] - target[p, 1]
        o2 = x_new[p, 2] - target[p, 2]
        length = np.sqrt(o0 * o0 + o1 * o1 + o2 * o2)
        if length > radius:
            over[p] = True
            any_over = True
            clamp[p] = radius / max(length, 1e-12)
    if any_over:
        for p in range(count):
            for c in range(3):
                x_new[p, c] = target[p, c] + (x_new[p, c] - target[p, c]) * clamp[p]
                if over[p]:
                    v_new[p, c] = v_new[p, c] * 0.5
    return x_new, v_new


def geodesic_from(count, edges, rest, anchors):
    """Shortest path along the mesh edges from each vertex to the nearest of `anchors`.

    Returns (distance, anchor_of): path length over the edges of `rest` (inf
    when unreachable) and the anchor it is measured from (-1 when unreachable).
    `edges` is (E, 2) vertex pairs."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    e = np.asarray(edges)
    rest = np.asarray(rest, np.float64)
    length = np.maximum(np.linalg.norm(rest[e[:, 1]] - rest[e[:, 0]], axis=1), 1e-9)
    graph = sp.coo_matrix((length, (e[:, 0], e[:, 1])), shape=(count, count)).tocsr()
    distance, _, source = dijkstra(graph, directed=False, indices=np.asarray(anchors), min_only=True, return_predecessors=True)
    anchor_of = np.where(np.isfinite(distance), source, -1).astype(np.int64)
    return distance, anchor_of


@njit(cache=True)
def _long_range(x, anchor_of, limit):
    """Pull each vertex toward its anchor to at most `limit` (one-sided). Returns a new array."""
    out = x.copy()
    for p in range(len(x)):
        a = anchor_of[p]
        if a < 0 or a == p:
            continue
        d0 = x[p, 0] - x[a, 0]
        d1 = x[p, 1] - x[a, 1]
        d2 = x[p, 2] - x[a, 2]
        length = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if length > limit[p]:
            k = limit[p] / length
            out[p, 0] = x[a, 0] + d0 * k
            out[p, 1] = x[a, 1] + d1 * k
            out[p, 2] = x[a, 2] + d2 * k
    return out


class LightCloth:
    """Damped springs toward the skinned garment, a per-vertex looseness map, and body collision."""

    STIFF_OMEGA, LOOSE_OMEGA = 60.0, 14.0  # rad/s: natural frequency on / off the body
    DAMPING_RATIO = 0.55  # < 1: a little overshoot, i.e. swing
    STIFF_RADIUS, LOOSE_RADIUS = 0.005, 0.05  # m: how far a vertex may trail its skinned place
    LOOSE_FROM, LOOSE_SPAN = 0.01, 0.06  # m off the body at rest: where looseness starts / saturates
    MARGIN = 0.01  # m: collision clearance; the fitted body is ~1.2 cm off the true one
    PASSES = 3
    # Long-range attachment (`set_long_range`): after the springs, a vertex is
    # no farther from its anchor than its rest path times (1 + LRA), so a hem
    # follows rising shoulders. None = off.
    LRA = 0.03
    LRA_MARGIN = 0.06  # m: slack added to the limit; without it vertices next to an anchor are pinned and the panel narrows

    def __init__(self):
        self._long_range_state = None
        self.x = self.v = self.target = None
        self.looseness = None

    def set_long_range(self, anchor_of, geodesic):
        """Install the attachment (`geodesic_from` of the rest mesh); `LRA` says whether it is applied."""
        limit = np.where(np.isfinite(geodesic), geodesic, 0.0)
        self._long_range_state = (np.ascontiguousarray(anchor_of, np.int64), np.ascontiguousarray(limit, np.float64))

    def _attach(self, x):
        if self.LRA is None or self._long_range_state is None or len(self._long_range_state[0]) != len(x):
            return x
        anchor_of, limit = self._long_range_state
        return _long_range(np.ascontiguousarray(x, np.float64), anchor_of, limit * (1.0 + self.LRA) + self.LRA_MARGIN)

    def _looseness(self, target, body_vertices):
        distance, _ = cKDTree(body_vertices).query(target, k=1)
        return np.clip((distance - self.LOOSE_FROM) / self.LOOSE_SPAN, 0.0, 1.0)

    def reset(self, target, body_vertices=None):
        self.x = np.array(target, np.float64)
        self.v = np.zeros_like(self.x)
        self.target = self.x.copy()
        # looseness belongs to the garment; kept across resets (rebuilding costs ~10 ms)
        if body_vertices is not None and (self.looseness is None or len(self.looseness) != len(self.x)):
            self.looseness = self._looseness(self.x, body_vertices)

    def _springs_numpy(self, target, dt, v):
        """Numpy spring step; oracle for `_springs_serial`. Returns (x, v)."""
        w = self.looseness[:, None]
        omega = self.STIFF_OMEGA + (self.LOOSE_OMEGA - self.STIFF_OMEGA) * w
        radius = self.STIFF_RADIUS + (self.LOOSE_RADIUS - self.STIFF_RADIUS) * w
        n = max(1, int(np.ceil(dt / SUBSTEP)))
        h = dt / n
        x = self.x
        for i in range(1, n + 1):
            # the target moves linearly across the frame
            goal = self.target + (target - self.target) * (i / n)
            a = omega * omega * (goal - x) - 2.0 * self.DAMPING_RATIO * omega * v
            v = v + h * a
            x = x + h * v
        offset = x - target
        length = np.linalg.norm(offset, axis=1, keepdims=True)
        over = length > radius
        if over.any():
            clamp = np.where(over, radius / np.maximum(length, 1e-12), 1.0)
            x = target + offset * clamp
            v = v * np.where(over, 0.5, 1.0)  # lose speed at the limit
        return x, v

    def step(self, target, dt, body_vertices, body_faces):
        target = np.asarray(target, np.float64)
        if self.x is None or len(self.x) != len(target) or dt <= 0.0 or dt > MAX_DT:
            self.reset(target, body_vertices)
            x = target.copy()
        else:
            if self.looseness is None:
                self.looseness = self._looseness(target, body_vertices)
            n = max(1, int(np.ceil(dt / SUBSTEP)))
            h = dt / n
            v = self.v
            x, v = _springs_serial(
                self.x, v, self.target, target, np.ascontiguousarray(self.looseness, np.float64),
                self.STIFF_OMEGA, self.LOOSE_OMEGA, self.STIFF_RADIUS, self.LOOSE_RADIUS, self.DAMPING_RATIO, n, h,
            )
            self.v = v
        x = self._attach(x)
        x = push_out_of_body(x, BodySurface(body_vertices, body_faces), self.MARGIN, self.PASSES)
        self.x, self.target = x, target
        return x

class SmoothCloth(LightCloth):
    """`LightCloth` with parameters searched against ContourCraft trajectories.

    Stiffer and less swinging than the default, with much lower jitter on
    held-out garments (tools/evaluate/cloth_smoothness.py). Opt in with
    `GarmentRenderer.CLOTH = SmoothCloth`."""

    STIFF_OMEGA, LOOSE_OMEGA = 41.4, 32.5
    DAMPING_RATIO = 0.243
    STIFF_RADIUS, LOOSE_RADIUS = 0.0095, 0.102
    LOOSE_FROM, LOOSE_SPAN = 0.0249, 0.0208
    MARGIN = 0.0073
