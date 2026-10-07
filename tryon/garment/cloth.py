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
from scipy.spatial import cKDTree

from tryon.garment.collision import BodySurface, push_out_of_body

SUBSTEP = 1.0 / 120.0  # s: semi-implicit Euler is stable here for omega * SUBSTEP < ~1
MAX_DT = 0.25  # s: a longer gap (a hitch, a lost person) restarts from the skin


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

    def __init__(self):
        self.x = self.v = self.target = None
        self.looseness = None

    def _looseness(self, target, body_vertices):
        distance, _ = cKDTree(body_vertices).query(target, k=1)
        return np.clip((distance - self.LOOSE_FROM) / self.LOOSE_SPAN, 0.0, 1.0)

    def reset(self, target, body_vertices=None):
        self.x = np.array(target, np.float64)
        self.v = np.zeros_like(self.x)
        self.target = self.x.copy()
        # the looseness map is the garment's (how far each vertex hangs off
        # the body), kept across resets: the synchronous display's frame
        # (~0.36 s) is longer than MAX_DT, so it reset every frame and
        # rebuilt this KD-tree query each time (~10 ms) for nothing
        if body_vertices is not None and (self.looseness is None or len(self.looseness) != len(self.x)):
            self.looseness = self._looseness(self.x, body_vertices)

    def step(self, target, dt, body_vertices, body_faces):
        target = np.asarray(target, np.float64)
        if self.x is None or len(self.x) != len(target) or dt <= 0.0 or dt > MAX_DT:
            self.reset(target, body_vertices)
            x = target.copy()
        else:
            if self.looseness is None:
                self.looseness = self._looseness(target, body_vertices)
            w = self.looseness[:, None]
            omega = self.STIFF_OMEGA + (self.LOOSE_OMEGA - self.STIFF_OMEGA) * w
            radius = self.STIFF_RADIUS + (self.LOOSE_RADIUS - self.STIFF_RADIUS) * w
            n = max(1, int(np.ceil(dt / SUBSTEP)))
            h = dt / n
            x, v = self.x, self.v
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
            self.v = v
        x = push_out_of_body(x, BodySurface(body_vertices, body_faces), self.MARGIN, self.PASSES)
        self.x, self.target = x, target
        return x
