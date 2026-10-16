"""The fluid display modes: every camera frame is shown, not only the ones the pipeline finishes.

The pipeline (`pipeline.run`) runs in its own process at its own rate. The display reads
the camera, finds landmarks (MediaPipe lite) and draws the newest finished garment layer
moved with the torso to this frame (--smooth), or shows the video about one pipeline
period late with the nearest layer (--delayed).
"""

from __future__ import annotations

import collections
import os
import sys
import time

import cv2
import numpy as np

from tryon.app.person import landmarks_from
from tryon.app.ui import WINDOW, Overlay


# MediaPipe model for the display's per-frame landmarks: it shares the CPU with the
# pipeline process, so every millisecond counts.
DISPLAY_POSE_COMPLEXITY = 0
# --delayed-sync: above this many waiting frames the display shows two a tick
DELAY_CATCH_UP = 45
# A file is played once the pipeline has finished this many frames: its first few are
# slow (compilation, GPU contexts) and starved --delayed-sync of layers.
WARMUP_LAYERS = 3


class Link:
    """One end of a `multiprocessing.Pipe` whose incoming messages a thread drains into a
    queue. Plain pipes deadlocked when both processes blocked in `send` on full pipes."""

    def __init__(self, conn):
        import queue
        import threading

        self._conn = conn
        self._inbox = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        try:
            while True:
                self._inbox.put(self._conn.recv())
        except (EOFError, OSError):
            self._inbox.put(("closed", None))

    def send(self, message):
        try:
            self._conn.send(message)
        except (BrokenPipeError, OSError):
            pass

    def poll(self):
        return not self._inbox.empty()

    def recv(self, timeout=None):
        return self._inbox.get(timeout=timeout)


class Channel:
    """The pipeline process's end of `smooth_display`'s pipe: a cv2.VideoCapture look-alike
    whose `read` returns the display's newest camera frame (skipping those the pipeline was
    too slow for), plus `publish` / `take_key` for finished frames and garment keys."""

    is_camera = True

    def __init__(self, conn, fps):
        self._conn = Link(conn)
        self._fps = fps
        self._keys = []
        self.stopped = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def _receive(self):
        kind, payload = self._conn.recv()
        if kind == "key":
            self._keys.append(payload)
        elif kind in ("stop", "closed"):
            self.stopped = True
        return kind, payload

    def read(self):
        """The display's newest frame, maybe as (frame id, frame); the id travels back with
        the finished layer (`publish`)."""
        if os.environ.get("TRYON_DEBUG"):
            print(f"[pipeline] need at {time.perf_counter():.2f}", file=sys.stderr, flush=True)
        self._conn.send(("need", None))
        while not self.stopped:
            kind, payload = self._receive()
            if kind == "frame":
                if isinstance(payload, tuple):
                    self.frame_id, payload = payload
                else:
                    self.frame_id = None
                return True, payload
        return False, None

    def release(self):
        pass

    def publish(self, source_frame, frame, points, hud, garment_alpha=None):
        # what the pipeline changed (erased clothing and garment), and of that, the garment
        changed = (cv2.absdiff(source_frame, frame).max(axis=2) > 6).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        alpha = cv2.GaussianBlur(changed.astype(np.float32), (5, 5), 0)
        garment = np.zeros_like(alpha) if garment_alpha is None else garment_alpha.astype(np.float32) / 255.0
        self._conn.send(("layer", (frame, alpha, garment, None if points is None else np.array(points), hud, None,
                                   getattr(self, "frame_id", None))))
        if os.environ.get("TRYON_DEBUG"):
            print(f"[pipeline] layer at {time.perf_counter():.2f}: {hud}", file=sys.stderr, flush=True)

    def take_key(self):
        while self._conn.poll():
            self._receive()
        return self._keys.pop(0) if self._keys else None


def _pipeline_process(conn, fps, renderer_flags, pipeline):
    """`pipeline.run` in its own process (no shared interpreter lock with the display)."""
    from tryon.app.pipeline import run
    from tryon.garment.renderer import GarmentRenderer

    for name, value in renderer_flags.items():
        setattr(GarmentRenderer, name, value)
    channel = Channel(conn, fps)
    try:
        run(channel, headless=True, max_width=0, channel=channel, **pipeline)
    finally:
        conn.send(("done", None))


def _face_and_hands(points, shape):
    """Soft mask (0-1) of this frame's face and hands, which stay in front of a carried layer."""
    from tryon.removal.remover import hand_discs

    keep = hand_discs(points, shape).astype(np.uint8)
    nose, left, right = points[0], points[11], points[12]
    if min(nose[2], left[2], right[2]) > 0.5:
        radius = int(0.45 * np.hypot(*(left[:2] - right[:2])))
        cv2.circle(keep, (int(nose[0]), int(nose[1])), max(radius, 1), 1, -1)
    return cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0)


def _compose_between(frame, points, person, keyframes, frame_id):
    """The frame with the garment from the pipeline layer nearest in time, moved to this
    frame's torso. Returns the composite and the garment alpha shown."""
    from tryon.app.layer_warp import torso_transform

    height, width = frame.shape[:2]
    ids = sorted(keyframes)
    before = [k for k in ids if k <= frame_id]
    after = [k for k in ids if k >= frame_id]
    # The NEAREST layer, not a blend: cross-fading two poses showed doubled sleeves and hems.
    candidates = ([before[-1]] if before else []) + ([after[0]] if after else [])
    nearest = min(candidates, key=lambda k: (abs(frame_id - k), -k))
    parts = [(keyframes[nearest], 1.0)]
    colour = np.zeros(frame.shape, np.float32)
    coverage = np.zeros(frame.shape[:2], np.float32)
    garment_shown = np.zeros(frame.shape[:2], np.float32)
    for layer, weight in parts:
        image, changed, garment, layer_points = layer[0], layer[1], layer[2], layer[3]
        if image.shape != frame.shape or weight <= 0.0:
            continue
        m = torso_transform(layer_points, points)
        if m is not None:
            image = cv2.warpAffine(image, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
            changed = cv2.warpAffine(changed, m, (width, height))
            garment = cv2.warpAffine(garment, m, (width, height))
        if person is not None:
            changed = changed * person  # the erased region only on this frame's person
        alpha = np.maximum(garment, changed) * weight
        colour += image.astype(np.float32) * alpha[..., None]
        coverage += alpha
        garment_shown += garment * weight
    if points is not None:
        keep = 1.0 - _face_and_hands(points, frame.shape)
        coverage, colour = coverage * keep, colour * keep[..., None]
    image = colour / np.maximum(coverage, 1e-6)[..., None]
    out = (frame * (1.0 - coverage[..., None]) + image * coverage[..., None]).astype(np.uint8)
    return out, garment_shown * (coverage > 0)


def _fit_width(frame, max_width):
    """`frame` at most `max_width` wide (aspect kept)."""
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return frame


def _person_mask(result, box=None):
    """MediaPipe's person segmentation, grown by 7 px, or None. `box`
    (x0, y0, x1, y1): only that window of it (same values)."""
    person = getattr(result, "segmentation_mask", None)
    if person is None:
        return None
    if box is None:
        return cv2.dilate((person > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
    x0, y0, x1, y1 = box
    height, width = person.shape[:2]
    px0, py0, px1, py1 = max(x0 - 7, 0), max(y0 - 7, 0), min(x1 + 7, width), min(y1 + 7, height)
    grown = cv2.dilate((person[py0:py1, px0:px1] > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8))
    return grown[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0].astype(np.float32)


class DisplayOutput:
    """Where the display's frames go: the window (unless `headless`; n / p go to the pipeline,
    q / Esc stop), a video file, and the shown garment's alpha per frame (`--dump-alpha`).

    `ui`: the window's `ui.Overlay`, drawn on the shown copy only; its garment choices go
    to the pipeline as ("goto", i). None: the bare window with the n / p / q keys."""

    def __init__(self, link, headless, output_video, fps, dump_alpha, ui=None):
        self.link, self.headless, self.output_video, self.fps = link, headless, output_video, fps
        self.dump_alpha, self.alphas, self.writer, self.shown = dump_alpha, [], None, 0
        self.ui = ui
        if ui is not None and not headless:
            ui.attach()

    def show(self, out, drawn):
        """Show one finished frame; True when the viewer asked to stop."""
        self.shown += 1
        if self.dump_alpha is not None:
            self.alphas.append(np.packbits(drawn > 0.1))
        if self.output_video is not None:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.output_video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (out.shape[1], out.shape[0]))
            self.writer.write(out)
        if not self.headless:
            cv2.imshow(WINDOW, out if self.ui is None else self.ui.render(out))
            key = cv2.waitKey(1) & 0xFF
            if self.ui is not None:
                action = self.ui.poll(key)
                if action is not None and action[0] == "exit":
                    return True
                if action is not None:
                    self.link.send(("key", ("goto", action[1])))
                return False
            if key in (27, ord("q")):
                return True
            if key in (ord("n"), ord("p")):
                self.link.send(("key", key))
        return False

    def close(self, frame_shape):
        if self.writer is not None:
            self.writer.release()
        if not self.headless:
            cv2.destroyAllWindows()
        if self.dump_alpha is not None and self.alphas:
            np.savez_compressed(self.dump_alpha, packed=np.stack(self.alphas), shape=np.array(frame_shape))


def _delayed_loop(capture, here, pose, is_camera, fps_in, max_width, display, max_frames, first_layer, wants_frame):
    """`smooth_display --delayed-sync`: show every camera frame about one pipeline period
    late, with the garment interpolated between the layers before and after it. Returns
    (pipeline layers, the shape of the frames shown)."""
    pending = collections.deque()  # (frame id, frame, landmarks, person mask) waiting for a later layer
    keyframes = collections.OrderedDict()
    if first_layer is not None:
        keyframes[first_layer[6] if first_layer[6] is not None else 0] = first_layer
    captured, layers = 0, len(keyframes)
    frame_shape = None
    hud = "starting ..."
    start = time.perf_counter()
    finished = False
    while not finished:
        ok, frame = capture.read()
        if not ok or frame is None:
            if is_camera:
                continue
            finished = True  # a file ended: show what is left with the layers there are
        else:
            if not is_camera:
                wait = captured / fps_in - (time.perf_counter() - start)
                if wait > 0:
                    time.sleep(wait)
            frame = _fit_width(frame, max_width)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            points = landmarks_from(result, frame.shape[1], frame.shape[0])
            pending.append((captured, frame, points, _person_mask(result)))
            captured += 1
        while here.poll():
            kind, payload = here.recv()
            if kind == "need":
                wants_frame = True
            elif kind == "layer":
                keyframes[payload[6] if payload[6] is not None else captured - 1] = payload
                layers += 1
                hud = payload[4]
                while len(keyframes) > 8:
                    keyframes.popitem(last=False)
        if wants_frame and not finished:
            here.send(("frame", (captured - 1, frame)))
            wants_frame = False
        # one frame out per frame in (two while catching up): the delay settles at one pipeline period
        budget = len(pending) if finished else (2 if len(pending) > DELAY_CATCH_UP else 1)
        while pending and budget > 0 and keyframes:
            frame_id, target, target_points, target_person = pending[0]
            if not finished and max(keyframes) < frame_id:
                break  # no layer after this frame yet
            pending.popleft()
            budget -= 1
            out, drawn = _compose_between(target, target_points, target_person, keyframes, frame_id)
            frame_shape = target.shape[:2]
            delay = len(pending) / fps_in
            cv2.putText(out, f"{hud}  delayed sync {delay:3.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if display.show(out, drawn) or (max_frames and display.shown >= max_frames):
                finished, pending = True, collections.deque()
                break
    return layers, frame_shape


_PAD = 4  # a 9x9 blur reaches 4 px beyond its output pixel
_WARP_MARGIN = 3  # bilinear taps plus fixed-point rounding of the warp


def _layer_box(layer):
    """(x0, y0, x1, y1) bounding the nonzero pixels of the layer's changed / garment alpha
    (half-open), or None if empty. Cached for the newest layer."""
    cached = getattr(_layer_box, "cache", None)
    if cached is not None and cached[0] is layer:
        return cached[1]
    support = cv2.max(layer[1], layer[2]) > 0
    box = None
    if support.any():
        ys, xs = np.flatnonzero(support.any(axis=1)), np.flatnonzero(support.any(axis=0))
        box = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
    _layer_box.cache = (layer, box)
    return box


def _moved_box(box, m, shape):
    """Where `box` lands after the warp `m` (None = identity), grown by interpolation smear,
    clipped to the frame."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    if m is None:
        return x0, y0, x1, y1
    corners = np.array([[x0 - 1, y0 - 1, 1], [x1 + 1, y0 - 1, 1], [x0 - 1, y1 + 1, 1], [x1 + 1, y1 + 1, 1]], np.float64)
    moved = corners @ np.asarray(m, np.float64).T
    lo = np.floor(moved.min(axis=0)).astype(int) - _WARP_MARGIN
    hi = np.ceil(moved.max(axis=0)).astype(int) + _WARP_MARGIN
    return max(lo[0], 0), max(lo[1], 0), min(hi[0], width), min(hi[1], height)


def _compose_newest(frame, result, points, layer):
    """`frame` with the newest pipeline `layer` moved onto it with the torso. Returns
    (frame shown, the garment's alpha as shown, HUD text). Works on the layer's moved
    bounding box only: alpha is zero outside it."""
    from tryon.app.layer_warp import torso_transform
    from tryon.removal.remover import hand_discs

    height, width = frame.shape[:2]
    out = frame.copy()
    drawn = np.zeros(frame.shape[:2], np.float32)
    hud = "starting ..."
    if layer is None or layer[0].shape != frame.shape:
        return out, drawn, hud
    image, changed, garment, layer_points, hud = layer[:5]
    m = torso_transform(layer_points, points)
    if m is not None:
        image = cv2.warpAffine(image, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
        changed = cv2.warpAffine(changed, m, (width, height))
        garment = cv2.warpAffine(garment, m, (width, height))
    box = _layer_box(layer)
    if box is None:
        return out, drawn, hud
    x0, y0, x1, y1 = _moved_box(box, m, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return out, drawn, hud
    window = (slice(y0, y1), slice(x0, x1))
    image, changed, garment = image[window], changed[window], garment[window]
    # The erased-and-filled region belongs to the OLD frame's body: carried along it paints
    # stale patches. Keep it only on this frame's person; the garment is always drawn.
    person = _person_mask(result, (x0, y0, x1, y1))
    if person is not None:
        changed = changed * person
    alpha = np.maximum(garment, changed)
    if points is not None:
        # this frame's face and hands stay in front of the old layer
        keep = hand_discs(points, frame.shape).astype(np.uint8)
        nose, left, right = points[0], points[11], points[12]
        if min(nose[2], left[2], right[2]) > 0.5:
            radius = int(0.45 * np.hypot(*(left[:2] - right[:2])))
            cv2.circle(keep, (int(nose[0]), int(nose[1])), max(radius, 1), 1, -1)
        px0, py0, px1, py1 = max(x0 - _PAD, 0), max(y0 - _PAD, 0), min(x1 + _PAD, width), min(y1 + _PAD, height)
        blurred = cv2.GaussianBlur(keep[py0:py1, px0:px1].astype(np.float32), (9, 9), 0)
        alpha = alpha * (1.0 - blurred[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0])
    base = frame[window]
    out[window] = (base * (1.0 - alpha[..., None]) + image * alpha[..., None]).astype(np.uint8)
    drawn[window] = garment * (alpha > 0)
    return out, drawn, hud


def _live_loop(capture, here, pose, is_camera, fps_in, max_width, display, max_frames, layer, wants_frame, layers):
    """`smooth_display`: every camera frame shown as it arrives, with the newest pipeline
    layer carried onto it. Returns (pipeline layers, the shape of the frames shown)."""
    times = collections.deque(maxlen=30)
    start = time.perf_counter()
    frame_shape = None
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            if is_camera:
                continue
            break
        if not is_camera:  # a file plays at its own rate, as a camera would deliver it
            wait = display.shown / fps_in - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
        tick = time.perf_counter()
        frame = _fit_width(frame, max_width)
        while here.poll():
            kind, payload = here.recv()
            if kind == "need":
                wants_frame = True
            elif kind == "layer":
                layer, layers = payload, layers + 1
        if wants_frame:
            here.send(("frame", frame))
            wants_frame = False
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        points = landmarks_from(result, frame.shape[1], frame.shape[0])
        out, drawn, hud = _compose_newest(frame, result, points, layer)
        frame_shape = frame.shape[:2]
        times.append(time.perf_counter() - tick)
        display_fps = len(times) / max(sum(times), 1e-6)
        cv2.putText(out, f"{hud}  display {min(display_fps, fps_in):4.1f}fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, "smooth mode: garment follows ~1 s behind on fast moves (--sync: exact, choppy)", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if display.show(out, drawn) or (max_frames and display.shown >= max_frames):
            break
    return layers, frame_shape


def _warm_up(capture, here, max_width):
    """Hold a video file at frame 0 while the pipeline loads, feeding it frame 0 until
    WARMUP_LAYERS layers are back. Returns (newest layer, layers received, whether the
    pipeline wants a frame)."""
    layer, layers, wants_frame, warmed = None, 0, False, 0
    ok, first = capture.read()
    if ok:
        first = _fit_width(first, max_width)
    while ok:
        try:
            kind, payload = here.recv(timeout=300)
        except Exception:  # queue.Empty: the pipeline never came up
            print("pipeline did not produce a first frame within 5 min", flush=True)
            break
        if kind == "need" and warmed < WARMUP_LAYERS:
            here.send(("frame", (0, first)))
        elif kind == "layer":
            layer, layers, warmed = payload, 1, warmed + 1
        elif kind == "need":
            wants_frame = True
            break
        elif kind in ("done", "closed"):
            break
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return layer, layers, wants_frame


def smooth_display(source, max_width=960, headless=False, output_video=None, max_frames=0, dump_alpha=None,
                   delayed=False, ui=True, **pipeline):
    """Show every camera frame, not only the ones the pipeline finishes.

    The pipeline (`pipeline.run`) runs in its own process; this process reads the camera,
    finds landmarks and draws the newest finished layer moved with the torso
    (`layer_warp.torso_transform`), keeping this frame's face and hands in front
    (`_live_loop`); or, with `delayed`, shows the video one pipeline period late with the
    nearest layer (`_delayed_loop`). The 3D pose is never moved to another frame; only the
    2D layer is."""
    import multiprocessing

    import mediapipe as mp

    from tryon.garment.renderer import GarmentRenderer

    is_camera = str(source).isdigit()
    capture = cv2.VideoCapture(int(source) if is_camera else source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {source}")
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 25.0
    here, there = multiprocessing.Pipe()
    flags = {name: getattr(GarmentRenderer, name) for name in ("GPU_RASTER", "APPEARANCE_ASYNC", "STUDENT_SHADING")}
    worker = multiprocessing.get_context("spawn").Process(target=_pipeline_process, args=(there, fps_in, flags, pipeline), daemon=True)
    worker.start()
    here = Link(here)
    layer, layers, wants_frame = _warm_up(capture, here, max_width) if not is_camera else (None, 0, False)
    overlay = Overlay.for_garment(pipeline.get("garment")) if ui and not headless else None
    display = DisplayOutput(here, headless, output_video, fps_in, dump_alpha, overlay)
    loop = _delayed_loop if delayed else _live_loop
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=DISPLAY_POSE_COMPLEXITY, enable_segmentation=True) as pose:
        if delayed:
            layers, frame_shape = loop(capture, here, pose, is_camera, fps_in, max_width, display, max_frames, layer, wants_frame)
        else:
            layers, frame_shape = loop(capture, here, pose, is_camera, fps_in, max_width, display, max_frames, layer, wants_frame, layers)
    here.send(("stop", None))
    worker.join(timeout=15)
    if worker.is_alive():
        worker.terminate()
    capture.release()
    display.close(frame_shape)
    print(f"{display.shown} frames shown, {layers} pipeline layers")
    return 0
