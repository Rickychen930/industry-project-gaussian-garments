"""Carry a garment layer from the frame it was rendered for to a later frame.

The full pipeline (body regression, cloth, splats, removal) takes ~0.65 s a
frame on the M1, so the camera would show ~1.5 fps. Instead the display can
run at the camera's rate: every new frame gets MediaPipe's landmarks
(~20 ms), and the newest finished layer - rendered for an EARLIER frame, on
that frame's own body - is warped by where the body's joints went since.
The 3D pose is never applied to another frame (a pose 0.6 s late scored
below the keypoint fit, docs/LOG.md); only the finished 2D layer moves.

What moves the layer is the TORSO: one similarity (shift, rotation,
scale) from the shoulders and hips of the old frame to the new, applied to
the whole layer (`torso_layer`). Measured on 00152_outer (40 frames,
keyframe every 8, IoU of the REAL shirt warped from frame k against the
real shirt at t): no warp 0.379, torso shift 0.510, torso similarity
0.502, a thin-plate spline through shoulders / elbows / wrists / hips
0.331, 2D linear blend skinning with arm bones 0.281. The two
articulated warps lose because a coat's side panels hang next to the
forearms and get carried off with them when an arm swings (a wrist moved
166 px while its elbow moved 25); the spline additionally smears its far
field across the background. Both were removed on 1 Oct 2026 (in the
`archive/pre-cleanup-2026-10-01` tag).
"""

from __future__ import annotations

import cv2
import numpy as np

from tryon.app.person import TORSO  # MediaPipe pose: shoulders and hips

VISIBLE = 0.5
SCALE_RANGE = (0.85, 1.18)  # a torso does not grow or shrink faster than this in ~1 s
MAX_ROTATION = np.radians(20.0)


def torso_transform(old_keypoints, new_keypoints):
    """2x3 similarity taking the old frame's shoulders and hips onto the
    new frame's, or None without both torsos visible."""
    if old_keypoints is None or new_keypoints is None:
        return None
    old, new = np.asarray(old_keypoints), np.asarray(new_keypoints)
    if min(old[i, 2] for i in TORSO) <= VISIBLE or min(new[i, 2] for i in TORSO) <= VISIBLE:
        return None
    src, dst = old[list(TORSO), :2], new[list(TORSO), :2]
    m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32))
    if m is None:
        return None
    # A body turned side-on puts shoulders and hips nearly in a line and
    # the fit's scale / rotation run away (the layer blew up to several
    # times its size on 00152_outer); clamp them, and keep the torso's
    # centre where the fit put it.
    scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
    angle = float(np.arctan2(m[1, 0], m[0, 0]))
    scale = float(np.clip(scale, *SCALE_RANGE))
    angle = float(np.clip(angle, -MAX_ROTATION, MAX_ROTATION))
    r = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre_old, centre_new = src.mean(0), dst.mean(0)
    return np.hstack([r, (centre_new - r @ centre_old)[:, None]])


def torso_layer(layer, old_keypoints, new_keypoints):
    """`layer` rendered for the frame with `old_keypoints`, moved with the
    torso to the frame with `new_keypoints` (unchanged without a torso)."""
    m = torso_transform(old_keypoints, new_keypoints)
    if m is None:
        return layer
    return cv2.warpAffine(layer, m, (layer.shape[1], layer.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
