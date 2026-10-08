"""Cheap garment secondary motion for the live loop.

An earlier position-based solver (GAPS energy terms; removed
on 1 Oct 2026, see the archive/pre-cleanup-2026-10-01 tag) cost 140-190 ms
a frame and, measured against the authors' own ContourCraft simulation of
00152_outer, landed where plain skinning does (7.53 vs 7.57 cm mean vertex
distance); what it bought was collision: skinning leaves 5.6 % of the
vertices inside the body (sleeves 12 %). This keeps the collision and replaces the rest
with what real-time cloth in games and AR try-on ships (Unity / Unreal /
NvCloth style): every vertex follows its skinned position through a damped
spring - loose where the garment hangs away from the body (a hem, a coat's
front), stiff where it lies on it - never further from it than a
per-vertex radius (a "max distance" / long-range attachment), then is
pushed out of the body (`collision.push_out_of_body`).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from tryon.garment.collision import BodySurface, push_out_of_body

SUBSTEP = 1.0 / 120.0  # s: semi-implicit Euler is stable here for omega * SUBSTEP < ~1
MAX_DT = 0.5  # s: a longer gap (a hitch, a lost person) restarts from the skin; the springs are stable and measured fine at the 0.4 s of a slow frame (tools/evaluate/cloth_smoothness.py), while 0.25 restarted them on every frame of a loaded machine, i.e. no cloth at all


@njit(cache=True)
def _springs_serial(x, v, start, target, looseness, stiff_omega, loose_omega, stiff_radius, loose_radius, damping, n, h):
    """`LightCloth._springs_numpy` in one serial pass per vertex (the same
    scalar operations in the same order, so the result is bit-identical):
    `n` semi-implicit Euler substeps of `h` toward the goal that moves
    linearly from `start` to `target`, then the radius clamp. Returns the new
    (x, v); `x`, `v` are not touched."""
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
        # as the numpy form: every vertex is rebuilt as target + offset * clamp (1.0 where not over)
        for p in range(count):
            for c in range(3):
                x_new[p, c] = target[p, c] + (x_new[p, c] - target[p, c]) * clamp[p]
                if over[p]:
                    v_new[p, c] = v_new[p, c] * 0.5
    return x_new, v_new


def geodesic_from(count, edges, rest, anchors):
    """Shortest path along the mesh edges from each vertex to the nearest of `anchors`.

    Returns (distance, anchor_of): `distance[i]` is the path length over the edges of `rest` (inf when no anchor
    can be reached, e.g. a piece of cloth with no anchor on it), `anchor_of[i]` the anchor it is measured from
    (-1 when unreachable). One Dijkstra over the whole mesh, `edges` as (E, 2) vertex pairs."""
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
    """Pull every vertex back toward its anchor to at most `limit` (a long-range attachment, one-sided: a vertex
    already inside its limit does not move). Returns a new array."""
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


@njit(cache=True)
def _long_range_vertical(x, anchor_of, limit):
    """`_long_range`, but a vertex only moves along y (up, toward its anchor), by as much as restores the
    distance to the anchor: a hem hangs by gravity, it is lifted and never dragged sideways. A vertex already
    inside its limit, or not below its anchor, stays."""
    out = x.copy()
    for p in range(len(x)):
        a = anchor_of[p]
        if a < 0 or a == p:
            continue
        d0 = x[a, 0] - x[p, 0]
        d1 = x[a, 1] - x[p, 1]
        d2 = x[a, 2] - x[p, 2]
        lateral2 = d0 * d0 + d2 * d2
        if lateral2 + d1 * d1 <= limit[p] * limit[p] or d1 <= 0.0:
            continue
        room = limit[p] * limit[p] - lateral2
        out[p, 1] = x[p, 1] + d1 - (np.sqrt(room) if room > 0.0 else 0.0)
    return out


class LightCloth:
    """Per-vertex damped springs toward the skinned garment, a looseness
    map from how far each vertex hangs off the body at rest, and a hard
    collision projection."""

    STIFF_OMEGA, LOOSE_OMEGA = 60.0, 14.0  # rad/s: natural frequency on / off the body
    DAMPING_RATIO = 0.55  # < 1: a little overshoot, i.e. swing
    STIFF_RADIUS, LOOSE_RADIUS = 0.005, 0.05  # m: how far a vertex may trail its skinned place
    LOOSE_FROM, LOOSE_SPAN = 0.01, 0.06  # m off the body at rest: where looseness starts / saturates
    MARGIN = 0.01  # m: collision clearance (the fitted body is ~1.2 cm off the true one; 2 / 1 / 0.5 / 0 cm gave garment IoU 0.647 / 0.659 / 0.664 / 0.666 with sleeves inside the body 0.1 / 2.6 / 5.6 / 8.1 %)
    PASSES = 3
    # Neighbour coupling of the displacement d = x - skinned target after the
    # springs (the springs are per vertex, so d is spatially noisy). None = off
    # (the original). "laplacian": COUPLING_ITERATIONS Jacobi steps of
    # d += COUPLING_RATE * (1 - stiffness weight) * (mean of neighbours' d - d).
    COUPLING = None
    COUPLING_RATE, COUPLING_ITERATIONS = 0.5, 3

    # One-sided horizontal wrap: a vertex further than distance d from the
    # tangent plane of its nearest body vertex is pulled toward the body by
    # `rate` of the excess, along the plane normal's HORIZONTAL part (a coat
    # hangs by gravity along the torso: the pull narrows it and never lifts
    # the hem). None = off (the original); else (d m, rate).
    WRAP = None

    # One Euro filter (Casiez et al. 2012) on the skinned target, per vertex,
    # before the springs: the body regressor's per-frame pose noise passes
    # straight into the cloth otherwise. None = off (the original); else
    # (min_cutoff Hz, beta s/m): the cutoff is min_cutoff + beta * speed.
    TARGET_FILTER = None
    # Fraction of the previous frame's skinned velocity the springs start the
    # frame with (feed-forward: a swinging sleeve keeps swinging). 0 = off.
    VELOCITY_INHERIT = 0.0

    # Long-range attachment from the garment's top ring (`set_long_range`): after the springs, a vertex is no
    # farther from its anchor than its rest path along the mesh times (1 + LRA); a hem is pulled up when the
    # shoulders rise and nothing happens when they do not. None = off. Default on since 9 Oct (anchors: the sleeve's border, `GarmentRenderer.LRA_ANCHORS`): loose-garment IoU +0.015 with `DRAPE_NARROW`, see docs/LOG.md.
    LRA = 0.03
    LRA_VERTICAL = False  # True: the attachment only lifts (along y), never drags a vertex sideways
    LRA_MARGIN = 0.06  # m: added to the limit (without it the vertices next to an anchor are tied to it and the panel narrows)

    def __init__(self, edges=None):
        self._long_range_state = None
        self.x = self.v = self.target = None
        self.looseness = None
        self.edges = edges  # (E, 2) mesh edges, needed by COUPLING
        self._mean_neighbours = None
        self._filtered = self._filtered_rate = None

    def _filter_target(self, target, dt):
        """The One Euro filter of `TARGET_FILTER`, state kept in `_filtered`."""
        if self.TARGET_FILTER is None:
            return target
        if self._filtered is None or len(self._filtered) != len(target):
            self._filtered, self._filtered_rate = target.copy(), np.zeros_like(target)
            return target
        min_cutoff, beta = self.TARGET_FILTER

        def alpha(cutoff):
            return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

        rate = (target - self._filtered) / dt
        a_rate = alpha(1.0)
        self._filtered_rate = self._filtered_rate + a_rate * (rate - self._filtered_rate)
        speed = np.linalg.norm(self._filtered_rate, axis=1, keepdims=True)
        a = alpha(min_cutoff + beta * speed)
        self._filtered = self._filtered + a * (target - self._filtered)
        return self._filtered.copy()

    def _neighbour_mean(self, n):
        if self._mean_neighbours is None or self._mean_neighbours.shape[0] != n:
            import scipy.sparse as sp

            e = np.asarray(self.edges)
            a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
            a = ((a + a.T) > 0).astype(np.float64)
            degree = np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1.0)
            self._mean_neighbours = (sp.diags(1.0 / degree) @ a).tocsr()
        return self._mean_neighbours

    def _couple(self, x, target):
        """Smooth the displacement over the mesh, loose vertices most."""
        if self.COUPLING is None or self.edges is None:
            return x
        mean = self._neighbour_mean(len(x))
        rate = (self.COUPLING_RATE * self.looseness)[:, None]
        d = x - target
        for _ in range(int(self.COUPLING_ITERATIONS)):
            d = d + rate * (mean @ d - d)
        return target + d

    def set_long_range(self, anchor_of, geodesic):
        """Install the attachment (`geodesic_from` of the rest mesh); `LRA` says whether it is applied."""
        limit = np.where(np.isfinite(geodesic), geodesic, 0.0)
        self._long_range_state = (np.ascontiguousarray(anchor_of, np.int64), np.ascontiguousarray(limit, np.float64))

    def _attach(self, x):
        if self.LRA is None or self._long_range_state is None or len(self._long_range_state[0]) != len(x):
            return x
        anchor_of, limit = self._long_range_state
        kernel = _long_range_vertical if self.LRA_VERTICAL else _long_range
        return kernel(np.ascontiguousarray(x, np.float64), anchor_of, limit * (1.0 + self.LRA) + self.LRA_MARGIN)

    def _looseness(self, target, body_vertices):
        distance, _ = cKDTree(body_vertices).query(target, k=1)
        return np.clip((distance - self.LOOSE_FROM) / self.LOOSE_SPAN, 0.0, 1.0)

    def reset(self, target, body_vertices=None):
        self.x = np.array(target, np.float64)
        self.v = np.zeros_like(self.x)
        self.target = self.x.copy()
        self._filtered, self._filtered_rate = None, None
        self._skin_velocity = np.zeros_like(self.x)
        # the looseness map is the garment's (how far each vertex hangs off
        # the body), kept across resets: a frame longer than MAX_DT resets, and
        # rebuilding this KD-tree query each time cost ~10 ms for nothing
        if body_vertices is not None and (self.looseness is None or len(self.looseness) != len(self.x)):
            self.looseness = self._looseness(self.x, body_vertices)

    def _springs_numpy(self, target, dt, v):
        """The original numpy form of the spring step: the oracle for
        `_springs_serial`. Returns (x, v) from `self.x`, `self.target`."""
        w = self.looseness[:, None]
        omega = self.STIFF_OMEGA + (self.LOOSE_OMEGA - self.STIFF_OMEGA) * w
        radius = self.STIFF_RADIUS + (self.LOOSE_RADIUS - self.STIFF_RADIUS) * w
        n = max(1, int(np.ceil(dt / SUBSTEP)))
        h = dt / n
        x = self.x
        for i in range(1, n + 1):
            # the skinned target moves linearly across the frame
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
            v = v * np.where(over, 0.5, 1.0)  # lose some speed on hitting the limit
        return x, v

    def step(self, target, dt, body_vertices, body_faces):
        target = np.asarray(target, np.float64)
        if self.x is None or len(self.x) != len(target) or dt <= 0.0 or dt > MAX_DT:
            self.reset(target, body_vertices)
            x = target.copy()
        else:
            if self.looseness is None:
                self.looseness = self._looseness(target, body_vertices)
            target = self._filter_target(target, dt)
            n = max(1, int(np.ceil(dt / SUBSTEP)))
            h = dt / n
            v = self.v
            if self.VELOCITY_INHERIT:
                v = v + self.VELOCITY_INHERIT * (self._skin_velocity - v)
                self._skin_velocity = (target - self.target) / dt
            x, v = _springs_serial(
                self.x, v, self.target, target, np.ascontiguousarray(self.looseness, np.float64),
                self.STIFF_OMEGA, self.LOOSE_OMEGA, self.STIFF_RADIUS, self.LOOSE_RADIUS, self.DAMPING_RATIO, n, h,
            )
            self.v = v
        x = self._couple(x, target)
        x = self._attach(x)
        surface = BodySurface(body_vertices, body_faces)
        x = self._wrap(x, surface)
        x = push_out_of_body(x, surface, self.MARGIN, self.PASSES)
        self.x, self.target = x, target
        return x

    def _wrap(self, x, surface):
        """Pull vertices that hang far off the body toward it (`WRAP`)."""
        if self.WRAP is None or len(x) == 0:
            return x
        distance, rate = self.WRAP
        vertex, normal = surface.nearest(x)
        excess = np.sum((x - vertex) * normal, axis=1) - distance
        pull = np.where(excess > 0.0, rate * excess, 0.0)
        horizontal = normal * np.array([1.0, 0.0, 1.0])
        return x - pull[:, None] * horizontal


class SmoothCloth(LightCloth):
    """`LightCloth` with the parameters of a 30-sample random search against
    the authors' ContourCraft trajectories (train: 00152_outer, 00170_upper,
    00134_upper, 00148_outer, 00176_lower; stride 5). Held out (00122_outer,
    00136_outer, 00170_lower, 00176_upper, 10004_outer), dt 0.2 s: distance
    9.361 vs 9.398 cm, displacement jitter 0.26 vs 0.74 cm (tools/evaluate/
    cloth_smoothness.py). Stiffer and less swinging than the default; opt in
    with `GarmentRenderer.CLOTH = SmoothCloth`."""

    STIFF_OMEGA, LOOSE_OMEGA = 41.4, 32.5
    DAMPING_RATIO = 0.243
    STIFF_RADIUS, LOOSE_RADIUS = 0.0095, 0.102
    LOOSE_FROM, LOOSE_SPAN = 0.0249, 0.0208
    MARGIN = 0.0073
