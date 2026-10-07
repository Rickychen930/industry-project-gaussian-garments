"""What the evaluators share: the mask IoU, the release videos' photo panel, the JSON they write."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


def iou(a, b, empty=1.0):
    """Intersection over union of two boolean masks; `empty` when both are empty
    (1.0: two empty masks agree; reference_frame passes 0.0: nothing was drawn)."""
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else empty


def panel_frames(video, height, frames, step=1):
    """The left (photo) panel of a release video's first `frames` frames, every
    `step`-th, each scaled to `height` pixels tall (the crop is found on frame 0)."""
    from tools.evaluate.reference_frame import find_panel
    from tryon.paths import RELEASE_VIDEOS

    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{video}.mp4"))
    out, box = [], None
    for t in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        if t % step:
            continue
        if box is None:
            box = find_panel(frame, "left")
        y0, y1, x0, x1 = box
        panel = frame[y0:y1, x0:x1]
        scale = height / panel.shape[0]
        out.append(cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    return out


def write_json(path, data, indent=1):
    """`data` as JSON at `path` (no trailing newline: the bytes the baselines in docs/baseline/ have)."""
    Path(path).write_text(json.dumps(data, indent=indent))
