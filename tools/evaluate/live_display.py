"""End-to-end accuracy of the live display, in real time, against the real shirt.

Runs the actual `tryon.app --smooth-display` on a release video at its own frame
rate (frames the pipeline cannot keep up with are skipped), records the garment
alpha the viewer sees (`--dump-alpha`) and scores it against the truth
`async_display --save` captured (SCHP upper clothing, same crop as
`vs_registration`). Reports per video and mode the mean IoU after the first
garment appeared and how many frames start-up took.

    .venv/bin/python -m tools.evaluate.live_display 00152_outer,00170_upper,00134_upper [--modes warp,reskin]

Needs `~/tryon-data/captures/<video>.npz` (`async_display --save`).
"""

from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tools.evaluate.vs_registration import HEIGHT
from tryon.paths import CAPTURES, GARMENTS, RELEASE_VIDEOS, REPO_ROOT, SCRATCH

CAPTURE = CAPTURES
WORK = SCRATCH / "live_display_eval"
MODES = {
    "warp": ["--smooth-display"],
    "reskin": ["--smooth-display", "--reskin"],
    "reskin-half": ["--smooth-display", "--reskin", "--reskin-subsample", "0.5"],
    "delayed": ["--smooth-display", "--delayed-sync"],
}


def panel_clip(name, frames):
    """The release video's photo panel, cropped and scaled exactly as the
    reference evaluator does, as an mp4 at the video's frame rate."""
    from tools.evaluate.reference_frame import find_panel

    out = WORK / f"{name}.mp4"
    if out.exists():
        return out
    WORK.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer, box = None, None
    for _ in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        box = box or find_panel(frame, "left")
        y0, y1, x0, x1 = box
        scale = HEIGHT / (y1 - y0)
        panel = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    writer.release()
    return out


def run_mode(name, clip, mode, frames, timeout):
    alpha = WORK / f"{name}-{mode}.npz"
    cmd = [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "tryon.app", "--source", str(clip),
           "--garment", str(GARMENTS / name), "--remove-clothing",
           "--orientation-regressor", "models/pose_regressor.pt", "--gpu-render", "--light-cloth",
           "--headless", "--max-width", "0", "--max-frames", str(frames), "--dump-alpha", str(alpha)] + MODES[mode]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"}, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-f", "nlf_worker"], check=False)
        return None
    return alpha if alpha.exists() else None


def score(alpha_path, truth):
    data = np.load(alpha_path)
    shape = tuple(data["shape"])
    shown = [np.unpackbits(p)[: shape[0] * shape[1]].reshape(shape).astype(bool) for p in data["packed"]]
    ious, started = [], None
    for t, drawn in enumerate(shown[: len(truth)]):
        if started is None and drawn.any():
            started = t
        if started is None:
            continue
        shirt = truth[t]
        union = (drawn | shirt).sum()
        ious.append(float((drawn & shirt).sum() / union) if union else 1.0)
    return {"iou": float(np.mean(ious)) if ious else 0.0, "scored_frames": len(ious), "startup_frames": started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", default="00152_outer")
    parser.add_argument("--modes", default="warp,reskin")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = {}
    for name in args.videos.split(","):
        truth = np.load(CAPTURE / f"{name}.npz")["shirt"]
        clip = panel_clip(name, args.frames)
        report[name] = {}
        for mode in args.modes.split(","):
            alpha = run_mode(name, clip, mode, args.frames, args.timeout)
            result = score(alpha, truth) if alpha is not None else {"error": "timed out or no output"}
            report[name][mode] = result
            print(f"{name} {mode}: {result}", flush=True)
    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
