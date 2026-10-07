"""Tracking jitter on real video: the authors' release clips.

The synthetic sequences in `tests/synthetic_sequence.py` add iid pixel
noise to exact projections; MediaPipe's video mode does not produce iid
noise (it smooths across frames), so a tracker tuned on them may be tuned
to the wrong problem. This runs the LIVE per-frame body logic
(`tryon.app.track_step`: warm track, refit on a spike) over the
"Ground Truth" panel of the Gaussian-Garments release videos - real people
moving smoothly at 25 fps - and reports, per clip:

  screen_accel   mean |second difference| of the fitted body's projected
                 joints, in thousandths of the person's height per frame^2.
                 What the viewer sees shake. A real body moving smoothly
                 has little of it, so lower is better as long as
  reprojection   (mean px from the fitted joints to MediaPipe's, as % of
                 the person's height) does not rise - a tracker that lags
                 or ignores the input also has low acceleration.
  input_accel    the same acceleration of MediaPipe's own landmarks: the
                 detector's jitter plus true motion, for scale.
  accel_3d       root-relative 3D joint acceleration, mm/frame^2.

There is no ground-truth body for these frames (the trajectories do not
align with the video past frame 0 - see docs/LOG.md), so this is a
proxy: it ranks tracker variants, it does not score them absolutely.

    .venv/bin/python -m tools.evaluate.tracking_jitter \\
        [--videos 00152_outer,00134_upper] [--height 870] [--frames 120] \\
        [--out docs/baseline/tracking-jitter.json]
"""

from __future__ import annotations

import argparse
import collections
import time

import cv2
import numpy as np

from tools.evaluate._common import panel_frames, write_json
from tryon.paths import SCRATCH

DEFAULT_VIDEOS = tuple("00122_outer,00134_upper,00136_outer,00148_outer,00148_upper,00152_lower,00152_outer,00170_lower,00170_upper,00176_lower,00176_outer,00176_upper,00190_upper,10004_outer,10006_outer".split(","))
CACHE = SCRATCH / "landmarks"


def landmarks_for(video, height, frames):
    """MediaPipe landmarks of the GT panel, video mode (as live), cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{video}_{height}_{frames}.npz"
    if cache.exists():
        data = np.load(cache)
        return list(data["points"]), int(data["height"])
    import mediapipe as mp

    from tryon.app.person import landmarks_from

    points = []
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        for panel in panel_frames(video, height, frames):
            found = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
            points.append(np.full((33, 4), np.nan, np.float32) if found is None else np.asarray(found, np.float32))
    np.savez(cache, points=np.stack(points), height=height)
    return points, height


def _second_difference(series):
    series = np.asarray(series, np.float64)
    return np.linalg.norm(series[2:] - 2 * series[1:-1] + series[:-2], axis=-1)


def run_clip(points, height, make_fitter=None, step=None, smoother=None, fps=25.0):
    """Replays the live body logic over one clip. `make_fitter`/`step` let
    a sweep swap in a variant; the defaults are the live ones."""
    import torch

    from tryon.body.fitter import BodyFitter
    from tryon.body.landmarks import JOINT_MAP, landmarks_to_targets
    from tryon.app.person import torso_visible, track_step

    make_fitter = make_fitter or BodyFitter
    step = step or track_step
    if smoother is None:
        from tryon.app.person import LANDMARK_FILTER
        from tryon.body.smoothing import OneEuroFilter

        if LANDMARK_FILTER is not None:
            def smoother():
                return OneEuroFilter(**LANDMARK_FILTER)  # the live default
    elif smoother == "none":
        smoother = None
    fitter, errors = None, collections.deque(maxlen=10)
    order = sorted(JOINT_MAP)
    screen, inputs, world, reprojection, statuses = [], [], [], [], collections.Counter()
    started = time.perf_counter()
    filt = smoother() if smoother is not None else None
    for index, raw_points in enumerate(points):
        frame_points = raw_points
        if np.isnan(frame_points).any() or not torso_visible(frame_points, tracking=fitter is not None):
            if filt is not None:
                filt.reset()
            fitter = None
            screen.append(None)
            inputs.append(None)
            world.append(None)
            statuses["lost"] += 1
            continue
        if filt is not None:
            frame_points = filt(raw_points, index / fps)
        if fitter is None:
            fitter = make_fitter()
            fitter.fit(frame_points)
            errors.clear()
            statuses["fitted"] += 1
        else:
            status = step(fitter, frame_points, errors, height, raw_points=raw_points)
            statuses[status] += 1
            if status == "refit" and filt is not None:
                filt.reset()
        # Scored against the RAW landmarks: a smoother that lags pays for it here.
        targets, weights = landmarks_to_targets(raw_points)
        with torch.no_grad():
            joints = fitter.joints()
            projected = fitter._project(joints).numpy()
            joints = joints.numpy()
        visible = weights > 0
        person = float(np.ptp(raw_points[order][:, 1])) or 1.0
        screen.append(projected / person * 1000.0)
        inputs.append(targets / person * 1000.0)
        world.append((joints - joints[:1]) * 1000.0)
        reprojection.append(float(np.linalg.norm(projected - targets, axis=1)[visible].mean()) / person * 100.0)
    seconds = time.perf_counter() - started

    def accel(series):
        values = []
        for t in range(1, len(series) - 1):
            window = series[t - 1 : t + 2]
            if any(w is None for w in window):
                continue
            values.append(_second_difference(np.stack(window)).mean())
        return float(np.mean(values)) if values else float("nan")

    return {
        "frames": len(points),
        "screen_accel": accel(screen),
        "input_accel": accel(inputs),
        "accel_3d_mm": accel(world),
        "reprojection_pct": float(np.mean(reprojection)) if reprojection else float("nan"),
        "statuses": dict(statuses),
        "seconds_per_frame": seconds / max(len(points), 1),
    }


def evaluate(videos, height, frames, **variant):
    results = {}
    for video in videos:
        points, h = landmarks_for(video, height, frames)
        results[video] = run_clip(points, h, **variant)
    keys = ("screen_accel", "input_accel", "accel_3d_mm", "reprojection_pct")
    results["mean"] = {k: float(np.nanmean([results[v][k] for v in videos])) for k in keys}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--height", type=int, default=870)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    videos = args.videos.split(",")
    results = evaluate(videos, args.height, args.frames)
    for video in videos + ["mean"]:
        r = results[video]
        print(f"{video:12s} screen accel {r['screen_accel']:6.2f}  input accel {r['input_accel']:6.2f}  "
              f"3D accel {r['accel_3d_mm']:6.1f} mm  reprojection {r['reprojection_pct']:.2f}%  {r.get('statuses', '')}")
    if args.out:
        write_json(args.out, results)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
