"""Record a guided camera session: the frames as read, with a cue for each pose.

    ./live-tryon record [seconds-per-pose=5]

Run it in your own terminal (the camera permission is the terminal's). The
cues are printed and spoken (`say`); the file goes to the data root's
captures/ folder and replays with `./live-tryon run 00190_upper --source FILE`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime

import cv2

from tryon.paths import CAPTURES

MAX_WIDTH = 960
FPS = 15.0
GET_READY = 5.0

# what to do, in order; each lasts `seconds-per-pose`
POSES = (
    "Face the camera, arms down, upper body in view",
    "Arms out to the sides, like a T",
    "Wave one hand",
    "Both hands above your head",
    "Reach one hand toward the camera, close to it",
    "Turn to show your left side",
    "Turn your back to the camera",
    "Turn to show your right side, then face the camera again",
    "Lean in: only your chest and arms in view, close to the camera",
    "Lean forward and back, then bend sideways",
    "Step out of view, then come back in",
    "Face the camera, arms down, still",
)


def say(text: str) -> None:
    subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: list[str]) -> int:
    seconds = float(argv[0]) if argv else 5.0
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open the camera (System Settings > Privacy > Camera: allow your terminal).", file=sys.stderr)
        return 1
    CAPTURES.mkdir(parents=True, exist_ok=True)
    path = CAPTURES / f"session-{datetime.now():%Y%m%d-%H%M%S}-input.mp4"
    total = GET_READY + seconds * len(POSES)
    print(f"Recording {total:.0f} s ({len(POSES)} poses of {seconds:.0f} s) to {path}")
    say("Get ready")
    writer, start, last, written = None, time.perf_counter(), -1.0, 0
    announced = -1
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= total:
            break
        ok, frame = capture.read()
        if not ok:
            continue
        index = int((elapsed - GET_READY) // seconds) if elapsed >= GET_READY else -1
        if index != announced and index >= 0:
            announced = index
            print(f"  {index + 1:2d}/{len(POSES)}  {POSES[index]}", flush=True)
            say(POSES[index])
        if frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (MAX_WIDTH, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if elapsed - last < 1.0 / FPS:
            continue
        last = elapsed
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        written += 1
    capture.release()
    if writer is not None:
        writer.release()
    say("Done")
    print(f"{written} frames saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
