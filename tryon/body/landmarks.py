"""MediaPipe Pose landmarks as SMPL-X fitting targets.

Maps MediaPipe's 33 landmarks (pixels, visibility, world z / x as packed by
`app.person.landmarks_from`) onto the SMPL-X joints both models agree on (`JOINT_MAP`),
weights them (`JOINT_WEIGHTS`), and builds the targets `body.fitter.BodyFitter` fits to:
2D joints, bone depths, and the torso's facing (`landmark_yaw`).
"""

from __future__ import annotations

import mediapipe as _mp
import numpy as np


POSE = _mp.solutions.pose


# MediaPipe landmark -> SMPL-X joint. Only joints both models agree on; hips differ
# (pelvis surface vs joint centre), so weighting matters (see `JOINT_WEIGHTS`).
_IDS = POSE.PoseLandmark


# Indices verified against a rest-pose forward pass (an earlier version had every
# left/right pair swapped).
JOINT_MAP = {
    # NOSE is deliberately absent: SMPL-X's nearest joint is the head centre, well above
    # a nose, and including it pulled the fit by 50 px.
    _IDS.LEFT_SHOULDER.value: 16,
    _IDS.RIGHT_SHOULDER.value: 17,
    _IDS.LEFT_ELBOW.value: 18,
    _IDS.RIGHT_ELBOW.value: 19,
    _IDS.LEFT_WRIST.value: 20,
    _IDS.RIGHT_WRIST.value: 21,
    _IDS.LEFT_HIP.value: 1,
    _IDS.RIGHT_HIP.value: 2,
    _IDS.LEFT_KNEE.value: 4,
    _IDS.RIGHT_KNEE.value: 5,
    _IDS.LEFT_ANKLE.value: 7,
    _IDS.RIGHT_ANKLE.value: 8,
}


# The torso is what a garment hangs from, so it is trusted most. Ankles are often out
# of frame and must not drag the body when guessed.
JOINT_WEIGHTS = {
    _IDS.LEFT_SHOULDER.value: 2.0,
    _IDS.RIGHT_SHOULDER.value: 2.0,
    _IDS.LEFT_HIP.value: 2.0,
    _IDS.RIGHT_HIP.value: 2.0,
    _IDS.LEFT_ANKLE.value: 0.3,
    _IDS.RIGHT_ANKLE.value: 0.3,
}


VISIBILITY_FLOOR = 0.4


# The rigid part of the body: what the camera is solved from in `fit`.
_TORSO = {
    _IDS.LEFT_SHOULDER.value,
    _IDS.RIGHT_SHOULDER.value,
    _IDS.LEFT_HIP.value,
    _IDS.RIGHT_HIP.value,
}


# Bones whose DEPTH is fitted from MediaPipe's 3D estimate (see `landmarks_to_depths`):
# the arms. A 2D fit alone cannot tell an arm reaching sideways from one angled into
# depth and can put a forearm behind the body's front surface, where the occlusion
# cull removes its sleeve.
DEPTH_BONES = (
    (_IDS.LEFT_SHOULDER.value, _IDS.LEFT_ELBOW.value),
    (_IDS.LEFT_ELBOW.value, _IDS.LEFT_WRIST.value),
    (_IDS.RIGHT_SHOULDER.value, _IDS.RIGHT_ELBOW.value),
    (_IDS.RIGHT_ELBOW.value, _IDS.RIGHT_WRIST.value),
)


# MediaPipe's world z is noisy, so the depth targets are shrunk toward the image
# plane by this factor (1 = trust fully, 0 = a pure in-plane prior).
DEPTH_SHRINK = 0.5


# The torso's facing, also from MediaPipe's world z: shoulder-to-shoulder and
# hip-to-hip. In profile the 2D loss cannot tell "turned 85 degrees" from "turned 40
# and the spine twisted"; the depth across the shoulders is the 3D quantity MediaPipe
# estimates well, so it is not shrunk like the arms.
TORSO_DEPTH_BONES = (
    (_IDS.LEFT_SHOULDER.value, _IDS.RIGHT_SHOULDER.value),
    (_IDS.LEFT_HIP.value, _IDS.RIGHT_HIP.value),
)


TORSO_DEPTH_SHRINK = 1.0


# Weight of a torso depth bone relative to an arm one inside the depth term. Tuned on
# warm profile frames; more weight does not close the remaining 2D / 3D disagreement.
TORSO_DEPTH_WEIGHT = 1.0


def _all_depth_bones():
    return DEPTH_BONES + TORSO_DEPTH_BONES


def landmarks_to_depths(points):
    """(dz targets, weights) per DEPTH_BONES then TORSO_DEPTH_BONES, in the fitter's frame.

    `points` may carry a 4th column: MediaPipe's world-landmark z in metres, where SMALLER z
    is CLOSER to the camera; the fitter's camera sits on +z, so its dz is the negative.
    Without that column, or where either end is not visible, the weight is 0 and the term
    is inert.
    """
    points = np.asarray(points, np.float64)
    bones = _all_depth_bones()
    shrink = [DEPTH_SHRINK] * len(DEPTH_BONES) + [TORSO_DEPTH_SHRINK] * len(TORSO_DEPTH_BONES)
    targets = np.zeros(len(bones), np.float64)
    weights = np.zeros(len(bones), np.float64)
    if points.shape[1] < 4:
        return targets, weights
    for k, (a, b) in enumerate(bones):
        za, zb = points[a, 3], points[b, 3]
        if np.isfinite(za) and np.isfinite(zb) and min(points[a, 2], points[b, 2]) >= VISIBILITY_FLOOR:
            targets[k] = -(zb - za) * shrink[k]
            weights[k] = 1.0 if k < len(DEPTH_BONES) else TORSO_DEPTH_WEIGHT
    return targets, weights


def _scaled_points(points, factor):
    """Landmarks with their pixel columns multiplied by `factor`."""
    if factor == 1.0:
        return points
    points = np.array(points, dtype=np.float32, copy=True)
    points[:, :2] *= factor
    return points


def landmark_yaw(points):
    """The torso's yaw in radians from MediaPipe's world landmarks, in the fitter's own
    convention (`global_orient[0, 1]`), or None.

    Reprojection cannot tell a profile's facing, but the world landmarks can. Mean of
    the shoulder and hip directions (left minus right, x and z columns)."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 5:
        return None
    angles = []
    for left, right in ((11, 12), (23, 24)):
        dx = float(points[left, 4] - points[right, 4])
        dz = float(points[left, 3] - points[right, 3])
        if np.isfinite(dx) and np.isfinite(dz) and (dx * dx + dz * dz) > 1e-6:
            angles.append(np.arctan2(dz, dx))
    if not angles:
        return None
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def yaw_difference(a, b):
    """Smallest absolute angle between two yaws, radians."""
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def landmarks_to_targets(points):
    """(pixels, weights) for the landmarks worth fitting to.

    An unseen landmark gets zero weight rather than being dropped, so the arrays keep a
    fixed shape and the optimiser graph does not change between frames (needed for warm
    starting).
    """
    points = np.asarray(points, np.float64)
    order = sorted(JOINT_MAP)
    targets = np.stack([points[index, :2] for index in order])
    weights = np.array(
        [
            JOINT_WEIGHTS.get(index, 1.0)
            if points[index, 2] >= VISIBILITY_FLOOR
            else 0.0
            for index in order
        ],
        np.float64,
    )
    return targets, weights
