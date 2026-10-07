"""How often the live garment disappears while a person is in view.

Every frame of these clips shows the wearer, so every frame should be
dressed. This runs the live loop's decision per frame - MediaPipe (video
mode, as live), `person.dressable`, the body regressor, the fallback cold
fit, `HOLD_FRAMES` - without rendering, and reports the fraction of frames
that would carry a garment, and the longest run without one.

Clips: the release videos' left panels (people turn, lean, kick; every
`--step`-th frame) and webcam-like variants made from two of them
(`--scenarios`): a waist-high crop (wearer at a desk), a head-and-chest
close-up, walking half out of the side, far away (small), dim light with
noise, motion blur, and a bar passing in front of the torso.

    .venv/bin/python -m tools.evaluate.dropouts [--frames 150] [--step 2] [--out j.json]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from tools.evaluate._common import panel_frames, write_json



def padded_frames(name, frames, step, width=960):
    """The panel at 540 px tall, every `step`-th frame, on a 16:9 canvas like a webcam, the sides filled from the edges."""
    out = []
    for panel in panel_frames(name, 540, frames, step):
        pad = max(width - panel.shape[1], 0)
        out.append(cv2.copyMakeBorder(panel, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_REPLICATE))
    return out


def variants(frames):
    """name -> frames, each a webcam situation made from `frames`."""
    rng = np.random.default_rng(0)
    h, w = frames[0].shape[:2]
    n = len(frames)
    out = {}

    def crop(f, y1):
        c = f[: int(h * y1)]
        s = h / c.shape[0]
        c = cv2.resize(c, None, fx=s, fy=s)
        x = (c.shape[1] - w) // 2
        return c[:, x : x + w] if c.shape[1] >= w else cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_REPLICATE)

    out["desk_waist"] = [crop(f, 0.55) for f in frames]
    out["closeup_chest"] = [crop(f, 0.40) for f in frames]
    out["half_out_side"] = [np.roll(f, int(w * 0.45 * min(1.0, 2.0 * i / n)), axis=1) for i, f in enumerate(frames)]
    out["far_small"] = [cv2.copyMakeBorder(cv2.resize(f, None, fx=0.4, fy=0.4), 0, h - int(h * 0.4), (w - int(w * 0.4)) // 2, w - int(w * 0.4) - (w - int(w * 0.4)) // 2, cv2.BORDER_REPLICATE) for f in frames]
    out["dim_noisy"] = [np.clip(f * 0.3 + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8) for f in frames]
    kernel = np.zeros((1, 21), np.float32)
    kernel[0] = 1 / 21
    out["motion_blur"] = [cv2.filter2D(f, -1, kernel) for f in frames]

    def bar(f, i):
        g = f.copy()
        x = int((i / max(n - 1, 1)) * (w + 200)) - 200
        g[:, max(x, 0) : max(x + 160, 0)] = (70, 70, 70)
        return g

    out["bar_in_front"] = [bar(f, i) for i, f in enumerate(frames)]
    return out


def dressed_flags(frames, regression, pose, parser=None):
    from tryon.app import person, pipeline
    from tryon.body.regressor import box_from_joints, person_in_box

    flags, fitter_alive, held, last_box = [], False, 0, None
    for frame in frames:
        height, width = frame.shape[:2]
        points = person.landmarks_from(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), width, height)
        dressed = False
        if person.dressable(points, fitter_alive, regression is not None):
            prediction = regression.predict(frame, keypoints=points) if regression is not None else None
            if prediction is not None or fitter_alive or person.torso_visible(points):
                dressed, fitter_alive = True, True
            if prediction is not None and prediction.get("joints_2d") is not None:
                last_box = box_from_joints(prediction["joints_2d"], frame.shape)
        glimpsed = False
        if not dressed and fitter_alive and parser is not None and last_box is not None and points is None:
            # the pipeline's re-find: the parser still sees a person in the last box
            person = parser.labels(frame) != 0
            if person_in_box(person, last_box):
                prediction = regression.predict(frame, box=last_box)
                if prediction is not None:
                    dressed = True
                    last_box = box_from_joints(prediction["joints_2d"], frame.shape)
            glimpsed = person_in_box(person, last_box, fraction=pipeline.REFIND_GLIMPSE)
        if dressed:
            held = 0
        elif fitter_alive and (points is not None or glimpsed) and held < pipeline.HOLD_FRAMES:
            held, dressed = held + 1, True
        else:
            fitter_alive = False
        flags.append(dressed)
    return flags


def summary(flags):
    flags = np.asarray(flags, bool)
    longest = run = 0
    for f in flags:
        run = 0 if f else run + 1
        longest = max(longest, run)
    return {"frames": int(len(flags)), "dressed": float(flags.mean()) if len(flags) else 0.0, "longest_gap": int(longest)}


def main():
    import mediapipe as mp

    from tools.evaluate.tracking_jitter import DEFAULT_VIDEOS as VIDEOS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--regressor", default="nlf", help="nlf, a Multi-HMR name, or none")
    parser.add_argument("--videos", default=None, help="comma list (default: all release videos)")
    parser.add_argument("--no-scenarios", action="store_true")
    parser.add_argument("--no-refind", action="store_true", help="without the pipeline's re-find through the parser")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from tryon.body import regressor as body_regressor

    regression = None
    if args.regressor == "nlf":
        regression = body_regressor.NLF()
    elif args.regressor != "none":
        model = body_regressor.MultiHMR(args.regressor)
        regression = type("R", (), {"predict": lambda self, f, keypoints=None: model.predict(f)})()

    schp = None
    if not args.no_refind and args.regressor == "nlf":
        from tryon.removal.parsing_schp import ATR_CHECKPOINT, SCHPParser

        schp = SCHPParser(ATR_CHECKPOINT) if ATR_CHECKPOINT.exists() else None

    clips = {}
    names = args.videos.split(",") if args.videos else list(VIDEOS)
    for name in names:
        clips[name] = padded_frames(name, args.frames, args.step)
    if not args.no_scenarios:
        for base in ("00152_outer", "00170_upper"):
            source = clips.get(base) or padded_frames(base, args.frames, args.step)
            for kind, frames in variants(source).items():
                clips[f"{kind}:{base}"] = frames
    report = {}
    for name, frames in clips.items():
        # a fresh tracker per clip, as a new session would start
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            report[name] = summary(dressed_flags(frames, regression, pose, schp))
        print(f"{name:28s} dressed {report[name]['dressed']:.3f}  longest gap {report[name]['longest_gap']:3d} of {report[name]['frames']}", flush=True)
    release = [v["dressed"] for k, v in report.items() if ":" not in k]
    scenario = [v["dressed"] for k, v in report.items() if ":" in k]
    print(f"MEAN release {np.mean(release) if release else float('nan'):.3f}  scenarios {np.mean(scenario) if scenario else float('nan'):.3f}")
    if args.out:
        write_json(args.out, report)
    if hasattr(regression, "close"):
        regression.close()


if __name__ == "__main__":
    main()
