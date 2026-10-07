"""MediaPipe Pose landmarks as SMPL-X fitting targets.

MediaPipe gives 33 landmarks per frame (pixels, visibility, a relative
depth, and world coordinates in the extra columns `app.person.landmarks_from`
packs). This module maps them onto the SMPL-X joints both models agree on
(`JOINT_MAP`), weights them (`JOINT_WEIGHTS`), and turns them into the
targets `body.fitter.BodyFitter` fits to: 2D joint targets, bone depths
from MediaPipe's own depth, and the torso's facing (`landmark_yaw`).
"""

from __future__ import annotations

import mediapipe as _mp
import numpy as np


POSE = _mp.solutions.pose


# MediaPipe landmark -> SMPL-X joint. Only the joints both models agree on:
# MediaPipe's hips sit at the pelvis surface and SMPL-X's at the joint
# centre, but the correspondence is close enough to drive a fit and wrong
# enough that weighting matters (see `JOINT_WEIGHTS`).
_IDS = POSE.PoseLandmark


# Indices verified against a rest-pose forward pass rather than taken from
# memory: an earlier version had every pair swapped (LEFT_SHOULDER to 17,
# which is SMPL-X's RIGHT shoulder), and the fit stalled at 34-48 px on the
# torso while the limbs reached 1-5 px - the optimiser was being asked to
# put the left shoulder where the right one is.
JOINT_MAP = {
    # NOSE is deliberately absent: SMPL-X's nearest body joint is the head
    # centre, which sits well above a nose, and including it pulled the fit
    # by 50 px while contributing nothing a garment depends on.
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


# The torso is what a garment hangs from, so it is trusted most. Ankles are
# often out of frame at a desk and must not drag the whole body when they
# are guessed.
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


# Bones whose DEPTH is fitted from MediaPipe's own 3D estimate (see
# `landmarks_to_depths`): the arms. A 2D fit alone cannot tell an arm
# reaching sideways from one angled into depth (the same weak-perspective
# ambiguity as the upright prior), and on camera 0004 it chose depth: the
# right forearm was fitted 45 degrees into the screen (dz -20 cm over a
# 28 cm bone) while MediaPipe's world landmarks put it within 4-13 cm of
# the image plane. The sleeve followed the fitted arm behind the body's
# own front surface and the body-occlusion cull removed 79% of the
# forearm splats - "the sleeves end at the elbow".
DEPTH_BONES = (
    (_IDS.LEFT_SHOULDER.value, _IDS.LEFT_ELBOW.value),
    (_IDS.LEFT_ELBOW.value, _IDS.LEFT_WRIST.value),
    (_IDS.RIGHT_SHOULDER.value, _IDS.RIGHT_ELBOW.value),
    (_IDS.RIGHT_ELBOW.value, _IDS.RIGHT_WRIST.value),
)


# MediaPipe's world z is noisy: on camera 0004 it put the left forearm
# 11 cm toward the camera for an arm held sideways, and following it
# tilted that forearm 23 degrees into depth while it straightened the
# right one (45 -> 9). The targets are shrunk toward the image plane by
# this factor (1 = trust MediaPipe fully, 0 = a pure in-plane prior).
DEPTH_SHRINK = 0.5


# The torso's facing, also from MediaPipe's world z: shoulder-to-shoulder
# and hip-to-hip. In profile the 2D loss cannot tell "turned 85 degrees"
# from "turned 40 and the spine twisted": measured on the annotated
# profile clips with warm landmarks (probe, 25/26 Sep), camera 0028's
# shoulders were fitted at 33-45 degrees while MediaPipe's own world
# landmarks put them at 73-89, and on 0076 tracking drifted from -103 to
# -140 against a steady -100 to -104. The depth across the shoulders is
# the one 3D quantity MediaPipe estimates well (it is the whole facing
# signal), so it is not shrunk like the arms.
TORSO_DEPTH_BONES = (
    (_IDS.LEFT_SHOULDER.value, _IDS.RIGHT_SHOULDER.value),
    (_IDS.LEFT_HIP.value, _IDS.RIGHT_HIP.value),
)


TORSO_DEPTH_SHRINK = 1.0


# Weight of a torso depth bone relative to an arm one inside the depth
# term. Measured over the 20 warm profile frames (mean shoulder yaw 0028
# / 0076, MediaPipe's own 87 / -101; mean reprojection): off 44 / -128,
# 6.5 / 10.5 px; 1 -> 70 / -106, 7.6 / 11.4 px; 3 -> 71 / -105, 7.2 /
# 11.3 px. The front camera (0004) stays at 9.8 px and the back (0052) at
# 7.5 with either. What 0028 still lacks is not weight: its 2D shoulder
# separation (43 px at 230 px/m) bounds the model's 35 cm shoulders to
# ~60 degrees on frame 0, so MediaPipe's 2D and 3D disagree there.
TORSO_DEPTH_WEIGHT = 1.0


def _all_depth_bones():
    return DEPTH_BONES + TORSO_DEPTH_BONES


def landmarks_to_depths(points):
    """(dz targets, weights) per DEPTH_BONES then TORSO_DEPTH_BONES, in the
    fitter's frame.

    `points` may carry a 4th column: MediaPipe's world-landmark z in
    metres, where SMALLER z is CLOSER to the camera (MediaPipe's
    convention); the fitter's camera sits on +z, so its dz is the
    negative. Without that column, or where either end is not visible,
    the weight is 0 and the term is inert - a 3-column caller fits in 2D
    exactly as before.
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
    """The torso's yaw in radians from MediaPipe's world landmarks, in the
    fitter's own convention (`global_orient[0, 1]`), or None.

    Reprojection cannot tell a profile's facing: on the four-camera turn
    clip the tracker sat at 22 degrees on a wearer turned ~69 while
    reprojecting at 8 px, and drew the open shirt front-on over a profile
    (the front panel hung off as a strip). MediaPipe's world landmarks knew
    at once - shoulders 75-88, hips 47-67 degrees on those frames, 173 / 171
    on the back camera where the refit reached 172, -102 / -91 on the far
    profile where it reached -100 - in the same sign convention. Mean of
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

    A landmark the detector cannot see is given zero weight rather than
    dropped, so the arrays keep a fixed shape and the optimiser graph does
    not change between frames - which is what makes warm starting possible.
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
