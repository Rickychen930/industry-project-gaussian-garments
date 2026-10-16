"""The person in the frame: MediaPipe keypoints, the visibility gates
that decide whether they can be dressed, and one tracking step of the
fitted body.

    landmarks_from   MediaPipe's result as a (33, 5) array
    dressable        can this frame's person be dressed?
    track_step       warm-track the body; refit it on a reprojection spike
    match_shoulder_width  the regressed body's shoulders as wide as MediaPipe's, front and back views
"""

from __future__ import annotations

import numpy as np

from tryon.body.landmarks import VISIBILITY_FLOOR

# Shoulders and hips (MediaPipe 11, 12, 23, 24). Not shoulders and elbows: a far elbow
# hidden in profile would mark the wearer lost.
TORSO = (11, 12, 23, 24)

# Per-frame tracking iterations. A turning body needs more than `track`'s default (3)
# or the warm-started chain stalls; 15 follows a 3 s turn to within a few degrees
# for ~36 ms a frame.
TRACK_ITERATIONS = 15
# A tracked frame whose reprojection exceeds REFIT_RATIO x the median of the last ten,
# and REFIT_FLOOR x the frame height, triggers `BodyFitter.refit`. Camera cuts spike
# 2.4-4x; steady tracking stays within 1.5x.
REFIT_RATIO = 2.0
# One Euro filter on the landmarks the fitter sees (tryon/body/smoothing.py), about the
# hip midpoint so position is never delayed. None = off.
LANDMARK_FILTER = {"min_cutoff": 4.0, "beta": 5.0, "relative": True}
REFIT_FLOOR = 0.014
# A tracked body facing more than this far from MediaPipe's own torso yaw
# (body.landmarks.landmark_yaw) is refit toward it; after a refit that was not
# adopted, the check rests YAW_COOLDOWN frames.
YAW_DISAGREE_DEG = 45.0
YAW_COOLDOWN = 10


def track_step(fitter, points, errors, height, regressor=None, raw_points=None):
    """One live tracking step: warm track, then refit on a reprojection
    spike (see REFIT_RATIO). `errors` is the caller's recent-error deque,
    updated here. Returns "tracking" or "refit"."""
    from tryon.body.landmarks import landmark_yaw, yaw_difference

    fitter.track(points, iterations=TRACK_ITERATIONS)
    status = "tracking"
    error = fitter.reprojection_error(points)
    spike = len(errors) >= 3 and error > max(REFIT_RATIO * float(np.median(errors)), REFIT_FLOOR * height)
    # The refit and the facing check read the RAW landmarks: filtered ones still blend
    # in the old pose right after a jump.
    raw = points if raw_points is None else raw_points
    target_yaw = landmark_yaw(raw)
    cooldown = getattr(fitter, "_yaw_cooldown", 0)
    fitter._yaw_cooldown = max(cooldown - 1, 0)
    disagrees = (
        target_yaw is not None
        and cooldown == 0
        and yaw_difference(float(fitter.global_orient.detach()[0, 1]), target_yaw) > np.radians(YAW_DISAGREE_DEG)
    )
    if spike or disagrees:
        extra = tuple(fitter.orientation_hypotheses_from_regressor(points, regressor, top_k=2)) if regressor is not None else ()
        if target_yaw is not None:
            extra = extra + (target_yaw,)
        if fitter.refit(raw, extra, prefer_yaw=target_yaw if disagrees else None):
            status = "refit"
            errors.clear()
            error = fitter.reprojection_error(points)
        elif disagrees:
            fitter._yaw_cooldown = YAW_COOLDOWN
    errors.append(error)
    return status


# A regressed fit's shoulders, projected, against MediaPipe's: the regressor has no
# metric anchor for a distant or cut-off wearer and can draw a garment far too wide.
# Corrected only front or back on; a profile's shoulders overlap and say nothing.
SHOULDER_RATIO_HIGH = 1.35
SHOULDER_RATIO_LOW = 0.74
SHOULDER_FRONT_BACK_DEG = 35.0
SHOULDER_VISIBILITY = 0.5


def rescaled_camera(scale, translation, anchor, ratio):
    """(scale, translation) of the weak-perspective camera with every pixel's distance
    from `anchor` divided by `ratio`, `anchor` itself staying where it is."""
    new_scale = scale / ratio
    new_translation = np.asarray(anchor, np.float64) - (new_scale / scale) * (np.asarray(anchor, np.float64) - np.asarray(translation, np.float64))
    return new_scale, new_translation


def match_shoulder_width(fitter, points):
    """Rescale `fitter`'s camera so its shoulders are as wide, in pixels, as MediaPipe's, about
    their midpoint, when they differ by more than SHOULDER_RATIO_HIGH / _LOW and the wearer
    faces the camera or away from it. Returns the ratio that was undone, or None."""
    import torch

    from tryon.body.landmarks import landmark_yaw

    if points is None or min(points[11, 2], points[12, 2]) < SHOULDER_VISIBILITY:
        return None
    yaw = landmark_yaw(points)
    if yaw is None:
        return None
    away = abs((np.degrees(yaw) + 180.0) % 360.0 - 180.0)  # 0 facing the camera, 180 away from it
    if min(away, 180.0 - away) > SHOULDER_FRONT_BACK_DEG:
        return None
    wanted = float(np.linalg.norm(points[11, :2] - points[12, :2]))
    if wanted < 4.0:
        return None
    with torch.no_grad():
        pixels = fitter._project(fitter._forward().joints[0, 16:18]).numpy()
    ratio = float(np.linalg.norm(pixels[0] - pixels[1])) / wanted
    if SHOULDER_RATIO_LOW <= ratio <= SHOULDER_RATIO_HIGH:
        return None
    scale, translation = rescaled_camera(float(fitter.scale), fitter.translation.detach().numpy().reshape(2), 0.5 * (pixels[0] + pixels[1]), ratio)
    with torch.no_grad():
        fitter.scale.copy_(torch.tensor([scale], dtype=torch.float32))
        fitter.translation.copy_(torch.tensor(translation, dtype=torch.float32).reshape(fitter.translation.shape))
    return ratio


def landmarks_from(result, width, height):
    """(33, 5): pixel x, pixel y, visibility, and MediaPipe's world-landmark
    z and x in metres (NaN when absent). Column 3 feeds the fitter's
    bone-depth term (body.landmarks.landmarks_to_depths); columns 3 and 4 give the
    torso's facing (body.landmarks.landmark_yaw)."""
    if result.pose_landmarks is None:
        return None
    world = getattr(result, "pose_world_landmarks", None)
    count = len(result.pose_landmarks.landmark)
    depth = [lm.z for lm in world.landmark] if world is not None else [np.nan] * count
    across = [lm.x for lm in world.landmark] if world is not None else [np.nan] * count
    return np.array(
        [[lm.x * width, lm.y * height, lm.visibility, z, x] for lm, z, x in zip(result.pose_landmarks.landmark, depth, across)],
        np.float32,
    )


# Torso visibility a TRACKED body needs. MediaPipe reports low confidence for a
# profile when cold but high once tracking, so the cold floor would drop a wearer
# the moment they turn sideways.
TRACKING_VISIBILITY_FLOOR = 0.2


def torso_visible(points, tracking=False):
    floor = TRACKING_VISIBILITY_FLOOR if tracking else VISIBILITY_FLOOR
    return points is not None and all(points[index, 2] >= floor for index in TORSO)


def shoulders_visible(points, floor=VISIBILITY_FLOOR):
    return points is not None and points[11, 2] >= floor and points[12, 2] >= floor


def dressable(points, tracking, regression):
    """Whether this frame's person can be dressed: the torso visible, or - with the
    image regressor, which recovers a truncated body - the shoulders alone."""
    return torso_visible(points, tracking=tracking) or (regression and shoulders_visible(points))


def framing_hint(points):
    """Hint for a person MediaPipe sees but who cannot be dressed, or None."""
    if points is None:
        return None
    if not shoulders_visible(points):
        return "step back or face the camera: shoulders not in view"
    return "step back: hips not in view"
