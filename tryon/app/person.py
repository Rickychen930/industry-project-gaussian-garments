"""The person in the frame: MediaPipe keypoints, the visibility gates
that decide whether they can be dressed, and one tracking step of the
fitted body.

    landmarks_from   MediaPipe's result as a (33, 5) array
    dressable        can this frame's person be dressed?
    track_step       warm-track the body; refit it on a reprojection spike
"""

from __future__ import annotations

import numpy as np

from tryon.body.landmarks import VISIBILITY_FLOOR

# Shoulders and hips (MediaPipe 11, 12, 23, 24). This used to be
# `sorted(JOINT_MAP)[:4]`, which is shoulders and ELBOWS: on 00176_outer
# the far elbow hidden behind the body in profile (visibility 0.14) made
# the wearer "lost" for 19 of 120 frames - no garment, and the removal
# still erased her jacket, leaving a headless ghost.
TORSO = (11, 12, 23, 24)

# `track`'s own default (3) is tuned for the case its docstring measures:
# a body moving sideways, where 3 iterations/frame holds up (see
# test_tracking_keeps_up_with_a_moving_body). A body TURNING is a
# different regime: a synthetic sequence turning 68 degrees over 30
# frames (1s at 30fps - a fast turn) with 3 iterations/frame got stuck
# near +10 degrees and barely moved; 15 iterations/frame reached -38.7
# degrees (still short, but moving the right way); over a more realistic
# 90-frame turn (3s), 15 iterations/frame reached -65.2 degrees against
# a true -68. The cost is small next to a garment render (~360ms): 15
# iterations measured at ~36ms/frame, negligible in that budget. This is
# what makes tracking recover the yaw a single cold fit provably cannot
# (see docs/PLAN.md and tests/test_body_fit_real_photos.py) - not a fix
# to the per-frame optimisation itself, more iterations to let the
# warm-started chain actually follow the turn instead of getting stuck
# a few frames in.
TRACK_ITERATIONS = 15
# A tracked frame whose reprojection exceeds REFIT_RATIO x the median of
# the last ten, and REFIT_FLOOR x the frame height, triggers
# `BodyFitter.refit` (a cold fit that may only improve). On the
# four-camera turn clip every cut spiked 2.4-4x (8 -> 19-27 px at 870 px
# tall) while steady tracking stayed within 1.5x; ratio 2.5 missed the
# 0 -> 69 degree cut. A refit costs ~4 s, only on a spike.
REFIT_RATIO = 2.0
# One Euro filter on the landmarks the fitter sees (tryon/body/smoothing.py),
# about the hip midpoint so position is never delayed. On the 15 release
# clips (tools/evaluate/tracking_jitter.py) with hip recentring: joint
# shake 24.39 -> 21.92, 3D limb acceleration 34.5 -> 24.3 mm/frame^2,
# reprojection 2.23 -> 2.49% of height; the rendered garment's on-screen
# acceleration 15.2 -> 12.9 (3 clips, cloth solver on). None = off.
LANDMARK_FILTER = {"min_cutoff": 4.0, "beta": 5.0, "relative": True}
REFIT_FLOOR = 0.014
# A tracked body facing more than this far from MediaPipe's own torso yaw
# (body.landmarks.landmark_yaw) is refit toward it; after a refit that was not
# adopted, the check rests YAW_COOLDOWN frames.
YAW_DISAGREE_DEG = 45.0
YAW_COOLDOWN = 10
# Re-measured 2026-09-25 when 15 iterations cost ~80-100 ms of a ~560 ms
# frame: on the 90-frame synthetic -68 degree turn, 15 iterations end 2.8
# degrees off (79 ms/frame), 10 end 6.7 off (44 ms), 8 end 25 off (35
# ms); on the fast 30-frame turn 15 / 10 / 8 end 29 / 42 / 61 degrees
# off. On a straight-on real clip 10 iterations cost only +0.33 px of
# reprojection - which is why the turn, not the clip, decides: 15 stays.


def track_step(fitter, points, errors, height, regressor=None, raw_points=None):
    """One live tracking step: warm track, then refit on a reprojection
    spike (see REFIT_RATIO). `errors` is the caller's recent-error deque,
    updated here. Returns "tracking" or "refit"."""
    from tryon.body.landmarks import landmark_yaw, yaw_difference

    fitter.track(points, iterations=TRACK_ITERATIONS)
    status = "tracking"
    error = fitter.reprojection_error(points)
    spike = len(errors) >= 3 and error > max(REFIT_RATIO * float(np.median(errors)), REFIT_FLOOR * height)
    # The refit and the facing check read the RAW landmarks: right after a
    # jump the filtered ones still blend the old pose in (the back view
    # refit landed at 116 instead of ~170 degrees on filtered input).
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


# Torso visibility a TRACKED body needs to keep tracking. MediaPipe reports
# 0.02-0.06 for the shoulders and hips of a wearer seen in profile when it
# starts cold, but 0.6-0.9 once it has been tracking (measured on the
# annotated profile cameras 0028 / 0076, warm_probe.json): the landmarks
# are on the body, only the confidence is low. Gating every frame at the
# cold-fit floor (0.4) dropped the wearer the moment they turned sideways.
TRACKING_VISIBILITY_FLOOR = 0.2


def torso_visible(points, tracking=False):
    floor = TRACKING_VISIBILITY_FLOOR if tracking else VISIBILITY_FLOOR
    return points is not None and all(points[index, 2] >= floor for index in TORSO)


def shoulders_visible(points, floor=VISIBILITY_FLOOR):
    return points is not None and points[11, 2] >= floor and points[12, 2] >= floor


def dressable(points, tracking, regression):
    """Whether this frame's person can be dressed: the torso (shoulders and
    hips) as before, or - with the image regressor, which recovers the
    whole body from a truncated view - the shoulders alone, so a wearer at
    a desk with the hips below the frame is still dressed."""
    return torso_visible(points, tracking=tracking) or (regression and shoulders_visible(points))


def framing_hint(points):
    """What to tell a person MediaPipe sees but who cannot be dressed, or
    None. The garment needs the shoulders (and, without the image
    regressor, the hips) in view; a person cut off at the shoulders, or
    too close, just saw the garment vanish."""
    if points is None:
        return None
    if not shoulders_visible(points):
        return "step back or face the camera: shoulders not in view"
    return "step back: hips not in view"
