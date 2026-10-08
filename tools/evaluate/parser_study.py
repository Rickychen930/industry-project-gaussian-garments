"""SCHP parser speed / accuracy study (docs/LOG.md, 9 Oct): compute units,
input size, post-processing, LIP. One process, run under `nice -n 10`.

    nice -n 10 .venv/bin/python -m tools.evaluate.parser_study frames  OUT.npz
    nice -n 10 .venv/bin/python -m tools.evaluate.parser_study units   FRAMES.npz
    nice -n 10 .venv/bin/python -m tools.evaluate.parser_study size    FRAMES.npz
    nice -n 10 .venv/bin/python -m tools.evaluate.parser_study post    FRAMES.npz
    nice -n 10 .venv/bin/python -m tools.evaluate.parser_study lip     FRAMES.npz
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np

CLIPS = ("00170_upper", "00152_outer", "00122_outer", "00176_upper")
PANEL_HEIGHT = 870


def load_average():
    return os.getloadavg()[0]


def top_process():
    out = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True).stdout.splitlines()
    return out[1].strip()[:60] if len(out) > 1 else "?"


def percentile_row(times):
    t = np.asarray(times) * 1000.0
    return f"med {np.median(t):6.1f}  p95 {np.percentile(t, 95):6.1f}  min {t.min():6.1f} ms (n={len(t)})"


def extract_frames(path, per_clip=15):
    from tools.evaluate.reference_frame import find_panel
    from tryon.paths import RELEASE_VIDEOS

    frames, clips = [], []
    for name in CLIPS:
        capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        box = None
        for index in np.linspace(5, total - 5, per_clip).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            box = box or find_panel(frame, "left")
            y0, y1, x0, x1 = box
            scale = PANEL_HEIGHT / (y1 - y0)
            frames.append(cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
            clips.append(name)
    shape = max(f.shape[1] for f in frames)
    padded = np.zeros((len(frames), PANEL_HEIGHT, shape, 3), np.uint8)
    widths = np.array([f.shape[1] for f in frames])
    for i, f in enumerate(frames):
        padded[i, : f.shape[0], : f.shape[1]] = f[:PANEL_HEIGHT]
    np.savez_compressed(path, frames=padded, widths=widths, clips=np.array(clips))


def load_frames(path):
    data = np.load(path)
    return [f[:, :w].copy() for f, w in zip(data["frames"], data["widths"])], list(data["clips"])


def main():
    command, path = sys.argv[1], sys.argv[2]
    if command == "frames":
        extract_frames(path)
        print("saved", path)
        return
    frames, clips = load_frames(path)
    import tools.evaluate.parser_study_runs as runs

    getattr(runs, command)(frames, clips, sys.argv[3:])


if __name__ == "__main__":
    main()
