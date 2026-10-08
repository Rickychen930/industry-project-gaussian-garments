"""The live try-on, one frame at a time (`run`).

    camera frame
      -> MediaPipe keypoints                      (app/person.py)
      -> 3D body: NLF regressor, or keypoint fit   (body/)
      -> clothing parse, on a worker thread        (removal/)
      -> garment posed, shaded and rendered        (garment/, rendering/)
      -> own clothing erased and filled            (removal/remover.py)
      -> garment composited, face and hands kept   (app/compositing.py)

Two numbers judge the body every frame and are shown in the HUD:

    reproj   mean pixel distance from the model's joints to the detected
             ones - accuracy
    accel    mean joint acceleration in mm/s^2 over the last 30 frames -
             smoothness (VIBE's metric). It has no ground truth live, so it
             cannot tell steady from lagging; that is why reproj sits beside
             it, always.

A person is "lost" when MediaPipe stops seeing a torso; the next sighting
triggers a fresh cold fit rather than tracking from a stale pose. A new
fitter means the garment's skinning is stale too - `render()` re-binds
automatically the first time it sees a different fitter object (see
`GarmentRenderer.posed_mesh`). With no garment the fitted skeleton is drawn.
"""

from __future__ import annotations

import collections
import os
import time
from pathlib import Path

import cv2
import numpy as np

from tryon.app.compositing import composite_garment, draw_body, keep_in_front
from tryon.app.person import (
    LANDMARK_FILTER,
    dressable,
    framing_hint,
    landmarks_from,
    torso_visible,
    track_step,
)
from tryon.body.fitter import BodyFitter
from tryon.body.regressor import box_from_joints, person_in_box
from tryon.body.smoothing import OneEuroFilter, joint_acceleration

# Multi-HMR below its native size resizes its position embeddings with a
# bicubic kernel MPS lacks; that one op falls back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Frames the garment stays on the last body when the person is lost.
HOLD_FRAMES = 8
# With MediaPipe blind, the share of the last box the parser must still see
# as person to keep holding (an empty room measured 0 %; a torso behind a
# bar 1-3 %).
REFIND_GLIMPSE = 0.005
# Input size of the --body-regressor on the live path (see run()).
LIVE_REGRESSOR_SIZE = 518
# Read the next frame and send it to NLF before this one is rendered, so
# the worker regresses frame t+1 while the loop renders, erases and shows
# frame t. Every shown frame still carries the garment fitted to it; it is
# shown one frame later. See `LiveSession.prefetch`.
PREFETCH = True
# MediaPipe Pose model for the landmarks (0 lite, 1 full). The landmarks
# give NLF its person box and the fit its keypoints; garment IoU against the
# real shirt (00170_upper / 00190_upper / 00152_outer) 0.789 / 0.731 / 0.658
# with lite against 0.788 / 0.731 / 0.657 with full, MediaPipe 18 -> 11-13 ms.
POSE_COMPLEXITY = 0


def learn_background(plate, labels, frame):
    """Feeds the frame's non-person pixels (the parser's `labels` of THIS
    frame) to the background plate, before removal edits it."""
    if plate is not None and labels is not None and labels.shape == frame.shape[:2]:
        plate.update(frame, labels != 0)


class MaskJob:
    """`remover.parse_clothing(frame)` on a worker thread; `result()` joins
    and returns that frame's `ClothingParse` (mask, labels, raw clothing):
    nothing is left on the remover, so the parses of two frames in flight
    cannot be mixed. `after`: a job to wait for before starting, so the jobs
    of consecutive frames run one at a time, in frame order (one SCHP call at
    a time on the GPU, and the remover's LIP skip counter stays in order).
    An exception in the worker is re-raised by `result()`, not lost."""

    def __init__(self, remover, frame, keypoints=None, after=None):
        import threading

        self.frame = frame
        self._remover = remover
        self._keypoints = keypoints
        self._after = after
        self._parse = None
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        if self._after is not None:
            self._after.wait()
            self._after = None
        try:
            self._parse = self._remover.parse_clothing(self.frame, keypoints=self._keypoints)
        except Exception as error:  # surfaced in result()
            self._error = error

    def start(self):
        self._thread.start()

    def wait(self):
        """Joins without raising."""
        self._thread.join()

    def result(self):
        self.wait()
        if self._error is not None:
            raise self._error
        return self._parse


class Call:
    """`function()` on a worker thread; `result()` joins and returns its value,
    or re-raises what it raised."""

    def __init__(self, function):
        import threading

        self._function = function
        self._value = self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self._value = self._function()
        except Exception as error:  # surfaced in result()
            self._error = error

    def start(self):
        self._thread.start()

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value


class StageTimer:
    """Per-stage wall time of one frame into `times` (name -> list), for
    `--profile`: each call records the time since the previous one."""

    def __init__(self, times):
        self._times = times
        self._start = self._last = time.perf_counter()

    def __call__(self, name, since_start=False):
        now = time.perf_counter()
        self._times.setdefault(name, []).append(now - (self._start if since_start else self._last))
        self._last = now

    def restart(self):
        """Time the stages from now on (after the next frame was read and
        sent ahead): `total` is then this frame's own work."""
        self._start = self._last = time.perf_counter()


# Start a frame's clothing parse as soon as its landmarks are found
# (`LiveSession.prefetch`: while the frame before it is erased, composited and
# shown) when True, or only once its fit is done, beside the render, when False.
EARLY_MASK = True

# Erase the wearer's clothing on a worker thread while the garment renders
# (True), or only after the render (False).
ERASE_BESIDE_RENDER = True

# Clothing is erased only within this fraction of the frame height around
# the fitted body (and on the arms); 0 turns the limit off.
ERASE_NEAR_BODY = 0.08


# ---------------------------------------------------------------------------
# Loading the models
# ---------------------------------------------------------------------------


def load_body_regressor(name, garment):
    """(regressor name or None, regressor or None). The default whenever a
    garment is worn and the model is installed, like the SCHP parser. Garment
    IoU against the real shirt on the five release videos: keypoint fit 0.653,
    Multi-HMR 672_S 0.716, NLF-S 0.730 (and 0.1-0.2 s a frame against 0.28); a
    wearer at a desk (hips out of frame) dressed on 75 of 75 frames against 41.
    NLF starts loading in its own process (~5 s) and is waited for on its
    first prediction, so it loads while everything else does."""
    if name is None and garment is not None:
        from tryon.body.regressor import available, nlf_available

        name = "nlf" if nlf_available() else "multiHMR_672_S" if available("multiHMR_672_S") else None
    if name in (None, "none"):
        return None, None
    import torch

    from tryon.body.regressor import NLF, MultiHMR

    if name == "nlf":
        return name, NLF()
    # MPS at 518 px: 194 ms a frame against 547 ms at the native 672 on the
    # CPU (same answers to 0.001 deg), for 0.710 vs 0.716 garment IoU on the
    # five release videos (docs/LOG.md, Multi-HMR).
    return name, MultiHMR(name, device="mps" if torch.backends.mps.is_available() else "cpu", input_size=LIVE_REGRESSOR_SIZE)


def wardrobe_of(garment):
    """Every garment checkpoint beside `garment`, and its index among them
    (for switching with n / p)."""
    here = Path(garment).expanduser().resolve()
    wardrobe = sorted(p for p in here.parent.iterdir() if (p / "stage1").is_dir())
    return wardrobe, next((i for i, p in enumerate(wardrobe) if p == here), 0)


def load_clothing_remover(parser, fill, background_plate):
    """(ClothingRemover, BackgroundPlate or None) for `--remove-clothing`."""
    from tryon.removal.background import BackgroundPlate
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover

    # SCHP is the default: it is the most accurate parser that is cheap
    # enough for the live path (~0.1 s on CoreML, on the worker thread it
    # costs nothing measurable) and the only one that removes fur sleeves;
    # FASHN is more accurate but ~0.8 s a frame.
    if parser is None:
        parser = "schp"
    schp = fashn = None
    if parser == "schp":
        schp = ATR_CHECKPOINT
    elif parser == "fashn":
        from tryon.removal.parsing_fashn import FASHN_CHECKPOINT

        fashn, schp = FASHN_CHECKPOINT, ATR_CHECKPOINT
    remover = ClothingRemover(schp_checkpoint=schp, fashn_checkpoint=fashn)
    # The room behind the wearer, learned from the feed (see
    # tryon/removal/background.py); needs the SCHP parser's person labels.
    plate = BackgroundPlate() if schp is not None and background_plate else None
    if fill is not None:
        remover.fill = fill
    print(f"clothing removal: {'FASHN' if fashn is not None else 'SCHP'} parser, fill {remover.fill}")
    return remover, plate


def load_orientation_classifier(path):
    """The facing classifier (tryon/body/orientation.py), or None."""
    if path is None:
        return None
    import torch

    from tryon.body.orientation import OrientationClassifier

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classifier = OrientationClassifier(n_bins=checkpoint["n_bins"])
    classifier.load_state_dict(checkpoint["model_state"])
    return classifier.eval()


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------


class LiveSession:
    """Everything that lives from one frame to the next: the models, the
    tracked body, its filters and histories. `process` dresses one frame."""

    # Orientation check: every ORIENTATION_CHECK_INTERVAL frames of tracking, check
    # whether the classifier's independent orientation estimate disagrees
    # with the tracked pose (see BodyFitter.orientation_disagrees) - a
    # stuck warm-started track otherwise has no way to notice it is
    # wrong. Matches the history's 30-frame window as a cadence that is
    # cheap (one small-MLP forward pass) without checking every frame.
    ORIENTATION_CHECK_INTERVAL = 30

    def __init__(self, garment=None, garment_subsample=None, orientation_regressor=None, remove_clothing=False, parser=None,
                 fill=None, appearance=False, background_plate=True, harmonize=False, body_regressor=None, headless=True):
        if garment is not None:
            # open3d (the garment binding's body-cover ray cast) takes 3-7 s
            # to import; the first frame with a person otherwise waited for
            # it. Imported on a thread while the models below load.
            import threading

            threading.Thread(target=lambda: __import__("open3d"), daemon=True).start()
        # First the body regressor: NLF loads in its own process while the
        # renderer and the clothing parsers below load here.
        self.regressor_name, self.regression = load_body_regressor(body_regressor, garment)
        self.garment_subsample, self.appearance = garment_subsample, appearance
        self.renderer, self.wardrobe, self.worn = None, [], 0
        if garment is not None:
            from tryon.garment.renderer import GarmentRenderer

            self.renderer = GarmentRenderer(garment, subsample=garment_subsample, appearance=appearance)
            self.wardrobe, self.worn = wardrobe_of(garment)
            if not headless and len(self.wardrobe) > 1:
                print(f"garment {self.wardrobe[self.worn].name} ({self.worn + 1}/{len(self.wardrobe)}): "
                      "press n / p for the next / previous one, q to quit")
        self.harmonizer = None
        if harmonize:
            from tryon.app.harmonize import LiveHarmonizer

            self.harmonizer = LiveHarmonizer()
        self.remover = self.plate = None
        if remove_clothing:
            self.remover, self.plate = load_clothing_remover(parser, fill, background_plate)
        self.orientation = load_orientation_classifier(orientation_regressor)

        self.fitter = None
        self.landmark_filter = None
        self.history = collections.deque(maxlen=30)  # joint positions, for the acceleration metric
        self.errors = collections.deque(maxlen=10)  # recent reprojection errors, for refit spikes
        self.frames_since_orientation_check = 0
        self.held = 0  # frames the garment has been held on a lost person
        self.last_box = None  # the wearer's box from the regressor, to re-find them
        self._early = collections.OrderedDict()  # id(frame) -> its MaskJob, started by `prefetch`
        self._last_job = None  # the newest mask job: the next one waits for it
        self.last_parse = None  # the newest ClothingParse used (the tracker's silhouette)

    # -- the steps of one frame ----------------------------------------------

    def prefetch(self, frame, points):
        """Start the body regressor and the clothing parse on `frame` now,
        before the frame before it is finished; `process(frame, ...)` then
        reads the answers."""
        self._start_mask_early(frame, points)
        if (PREFETCH and self.regressor_name == "nlf" and self.regression is not None
                and dressable(points, self.fitter is not None, True)):
            self.regression.prefetch(frame, keypoints=points)

    def process(self, frame, points, stamp, stage, untouched=None):
        """Dress `frame` in place, given this frame's MediaPipe `points` and
        the time `stamp` for the landmark filter. Returns (status, reprojection
        error, the garment's alpha or None). `untouched`: a copy of `frame`
        as read, if the caller already made one (it is only read)."""
        early_mask = self._take_early_mask(frame)
        # The fitter sees the One Euro filtered landmarks; the mask, arm
        # strokes and HUD keep the raw ones, which match the image.
        fit_points = self._filtered(points, stamp)
        status = self._update_body(frame, points, fit_points, stage)
        status, visible = self._handle_loss(frame, points, status)
        garment_alpha = None
        if not visible and status != "holding":
            # Nothing to dress: the frame is left as it is. Erasing the
            # wearer's clothing with no garment to put over it showed a
            # grey, half-undressed figure (desk-height clip, the hips out of
            # frame for 18 of 75 frames) - worse than the original clothes.
            self.fitter = None  # next sighting starts from a cold fit
            if self.landmark_filter is not None:
                self.landmark_filter.reset()
            reprojection = float("nan")
        elif self.fitter is not None:
            garment_alpha = self._dress(frame, points, early_mask, stage, untouched)
            reprojection = self.fitter.reprojection_error(points) if points is not None else float("nan")
        else:
            # A visible torso, but the fitter was just dropped by the
            # orientation check - this frame has no trustworthy pose yet;
            # the next frame cold-fits and resumes.
            reprojection = float("nan")
        if status == "no person" and points is not None:
            status = framing_hint(points)  # someone is there: say what to change
        return status, reprojection, garment_alpha

    def _start_mask(self, frame, points):
        """The clothing parse of `frame` on its own thread, queued behind the
        one before it."""
        job = MaskJob(self.remover, frame, keypoints=points, after=self._last_job)
        self._last_job = job
        job.start()
        return job

    def _start_mask_early(self, frame, points):
        """With the body regressor the clothing parse does not wait for the
        fit, nor for the frame before it to be finished: it starts on its
        thread as soon as the landmarks are found (called from `prefetch`),
        and `process(frame)` picks it up. Whether the frame will be dressed
        is judged on the fitter as it stands now, one frame early: a wrong
        guess costs a parse nobody reads, or `_dress` starting it late."""
        if (EARLY_MASK and self.regression is not None and self.remover is not None and self.renderer is not None
                and dressable(points, self.fitter is not None, True)):
            self._early[id(frame)] = self._start_mask(frame, points)

    def _take_early_mask(self, frame):
        """The job `prefetch` started for exactly this `frame`, or None. Jobs
        of frames older than it that were never processed are dropped (they
        finish on their own, in order)."""
        job = self._early.get(id(frame))
        if job is None or job.frame is not frame:
            return None
        for key in list(self._early):
            del self._early[key]
            if key == id(frame):
                break
        return job

    def finish(self):
        """Waits for every mask job still running (the frames prefetched but
        never processed: the loop ended), so no thread outlives the session."""
        self._early.clear()
        if self._last_job is not None:
            self._last_job.wait()

    def _filtered(self, points, stamp):
        if points is None or LANDMARK_FILTER is None:
            return points
        if self.landmark_filter is None:
            self.landmark_filter = OneEuroFilter(**LANDMARK_FILTER)
        return self.landmark_filter(points, stamp)

    def _update_body(self, frame, points, fit_points, stage):
        """Regress, cold-fit or track the body for this frame. Returns the
        status shown in the HUD."""
        if not dressable(points, self.fitter is not None, self.regression is not None):
            return "no person"
        # On THIS frame, every frame: answers ~0.6 s late (a worker thread,
        # or every 15th frame) scored below the keypoint tracker on the
        # release videos (docs/LOG.md, Multi-HMR).
        prediction = None
        if self.regression is not None:
            # NLF takes its person box from MediaPipe (its own detector is 1.5 s)
            prediction = (self.regression.predict(frame, keypoints=points) if self.regressor_name == "nlf"
                          else self.regression.predict(frame))
        stage("regressor")
        if prediction is not None:
            fresh = self.fitter is None
            self.fitter = self.fitter or BodyFitter()
            self.fitter.fit_from_regression(fit_points, prediction)
            if prediction.get("joints_2d") is not None:
                self.last_box = box_from_joints(prediction["joints_2d"], frame.shape)
            if fresh:
                self.history.clear()
                self.errors.clear()
            return "regressed"
        if self.fitter is None and torso_visible(points):
            self._cold_fit(points, fit_points)
            return "fitted"
        return self._track(frame, points, fit_points)

    def _cold_fit(self, points, fit_points):
        self.fitter = BodyFitter()
        if self.orientation is not None:
            # Seed the cold fit with the classifier's predicted yaw
            # octants IN ADDITION TO the {0, pi} pair, rather than instead of
            # it - covers the near-perpendicular case (fit()'s own pair has
            # no seed there) without giving up the pair's front/back
            # disambiguation.
            hypotheses = self.fitter.orientation_hypotheses_from_regressor(points, self.orientation, top_k=2)
            self.fitter.fit(fit_points, orientation_hypotheses=tuple(hypotheses) + (0.0, float(np.pi)))
        else:
            self.fitter.fit(fit_points)
        self.history.clear()
        self.errors.clear()
        self.frames_since_orientation_check = 0

    def _track(self, frame, points, fit_points):
        height, width = frame.shape[:2]
        # the previous frame's parse (the mask runs after tracking, on its
        # own thread) - the silhouette term in the fitter
        labels = self.last_parse.labels if self.last_parse is not None else None
        self.fitter.set_silhouette(labels != 0 if labels is not None and labels.shape == (height, width) else None)
        status = track_step(self.fitter, fit_points, self.errors, height, self.orientation, raw_points=points)
        if status == "refit":
            self.history.clear()
            if self.landmark_filter is not None:
                # a refit marks a discontinuity; the filter would otherwise
                # blend the old pose in for frames
                self.landmark_filter.reset()
            if self.renderer is not None:
                self.renderer.reset_motion(self.fitter)
        if self.orientation is not None:
            self.frames_since_orientation_check += 1
            if self.frames_since_orientation_check >= self.ORIENTATION_CHECK_INTERVAL:
                self.frames_since_orientation_check = 0
                if self.fitter.orientation_disagrees(points, self.orientation):
                    # The "lost -> cold fit next frame" path: this frame has
                    # no trustworthy pose, the next one cold-fits.
                    self.fitter = None
                    status = "reorienting"
        return status

    def _handle_loss(self, frame, points, status):
        """Re-find a wearer MediaPipe lost, or hold the garment on them for a
        few frames. Returns (status, whether the person counts as visible)."""
        rescued, glimpsed = False, False
        if (status == "no person" and self.fitter is not None and self.regressor_name == "nlf"
                and self.remover is not None and self.last_box is not None):
            # MediaPipe lost the wearer outright (a bar or an arm across the
            # torso: no landmarks, no mask) while they are still there. The
            # clothing parser often still sees them in the last box: then
            # NLF, given that box, finds the body. When it sees only a sliver
            # (1-3 % of the box, the rest hidden) the garment holds; an empty
            # room is 0 %.
            self.last_parse = self._start_mask(frame, None).result()
            labels = self.last_parse.labels
            if labels is not None and labels.shape == frame.shape[:2]:
                if person_in_box(labels != 0, self.last_box):
                    prediction = self.regression.predict(frame, box=self.last_box)
                    if prediction is not None:
                        self.fitter.fit_from_regression(None, prediction)
                        self.last_box = box_from_joints(prediction["joints_2d"], frame.shape)
                        status, rescued = "re-found", True
                glimpsed = person_in_box(labels != 0, self.last_box, fraction=REFIND_GLIMPSE)
        visible = rescued or dressable(points, self.fitter is not None, self.regression is not None)
        if visible:
            self.held = 0
        elif self.fitter is not None and (points is not None or glimpsed) and self.held < HOLD_FRAMES:
            # A short, partial loss (a lean, a hand across the hips): someone
            # is still there, and the garment stays on the last body rather
            # than vanishing for a frame or two. Nobody in the frame at all
            # ends it at once - a garment left floating over an empty room
            # for 8 frames.
            self.held += 1
            status = "holding"
        return status, visible

    def _dress(self, frame, points, early_mask, stage, untouched=None):
        """Render the garment on the fitted body, erase the wearer's own
        clothing under it and composite. `early_mask`: the frame's clothing
        parse if `prefetch` started it. Returns the garment's alpha."""
        import torch

        with torch.no_grad():
            self.history.append(self.fitter._forward().joints[0, self.fitter._smplx_indices].numpy().copy())
        if self.renderer is None:
            draw_body(frame, self.fitter)
            return None
        # The clothing parse only READS the frame and the render never
        # touches it, so the two overlap on a worker thread (the parser's
        # ONNX / CoreML call and numba both release the interpreter lock).
        mask_job = early_mask
        if self.remover is not None and mask_job is None:
            # The arms are added from the keypoints (see ClothingRemover.parse_clothing).
            mask_job = self._start_mask(frame, points)
        stage("fit")
        erase = None
        if mask_job is not None and ERASE_BESIDE_RENDER:
            # The erase (plate update, mask limits, the fill) reads the frame
            # and writes into a copy, and the render never touches the frame,
            # so the two run side by side; the body fill is the one step that
            # needs the GL context, so `_start_erase` does it here, first.
            erase = self._start_erase(frame, points, mask_job, untouched)
            stage("erase start")
        height, width = frame.shape[:2]
        rgba = self.renderer.render(self.fitter, (height, width))
        stage("render")
        if mask_job is not None and erase is None:
            erase = self._start_erase(frame, points, mask_job, untouched)
        if erase is not None:
            parse, erased, original = erase.result()
            self.last_parse = parse
            stage("erase wait")
            frame[:] = erased
            composite_garment(frame, rgba)
            keep_in_front(frame, original, parse.labels, points)
            stage("composite")
        else:
            composite_garment(frame, rgba)
        if self.harmonizer is not None:
            # the garment recoloured to the scene's light
            frame[:] = self.harmonizer(frame, rgba[..., 3] > 25)
        return rgba[..., 3]

    def _start_erase(self, frame, points, mask_job, untouched=None):
        """Erase the wearer's own clothing on a worker thread, so the rendered
        garment lands on a clean base instead of over the real one. The
        thread's result is (the frame's parse, the erased copy, the frame as
        read, to restore the face and hands from: the caller's copy, the one
        the display diffs, when it made one)."""
        from tryon.removal.background import body_fill
        from tryon.removal.remover import arm_strokes, hand_discs

        fill = body_fill(self.fitter, frame.shape) if points is not None and (ERASE_NEAR_BODY or self.plate is not None) else None
        arms = arm_strokes(points, frame.shape, width_fraction=self.remover.ARM_TUBE_WIDTH) if points is not None else None
        hands = hand_discs(points, frame.shape) if points is not None else None
        original = frame.copy() if untouched is None else untouched
        call = Call(lambda: self._erase(frame, points, mask_job, fill, arms, hands, original))
        call.start()
        return call

    def _erase(self, frame, points, mask_job, fill, arms, hands, original):
        """The worker's part of `_start_erase`: no GL, no numba, no writes to
        `frame`, no state but the background plate (only touched here)."""
        from tryon.removal.background import body_silhouette
        from tryon.removal.remover import arm_strokes

        parse = mask_job.result()
        mask = parse.mask
        if points is not None and ERASE_NEAR_BODY:
            # only the clothing the wearer has on: SCHP also labels shirts
            # hanging in a wardrobe behind them, which were erased into flat
            # patches on a webcam
            mask = mask & (body_silhouette(self.fitter, frame.shape, grow=ERASE_NEAR_BODY, fill=fill) | arm_strokes(points, frame.shape))
        learn_background(self.plate, parse.labels, frame)
        body = None
        if self.plate is not None and points is not None:
            body = body_silhouette(self.fitter, frame.shape, fill=fill) | arm_strokes(points, frame.shape)
        erased = self.remover.remove(frame, mask, arms=arms, keep=hands, plate=self.plate, body=body, parse=parse)
        return parse, erased, original

    def switch_garment(self, step):
        """Wear the next (+1) or previous (-1) garment of the wardrobe."""
        if len(self.wardrobe) < 2:
            return
        from tryon.garment.renderer import GarmentRenderer

        self.worn = (self.worn + step) % len(self.wardrobe)
        print(f"garment {self.wardrobe[self.worn].name} ({self.worn + 1}/{len(self.wardrobe)})", flush=True)
        shading = getattr(self.renderer, "_appearance_net", None)
        if hasattr(shading, "close"):
            shading.close()
        self.renderer = GarmentRenderer(self.wardrobe[self.worn], subsample=self.garment_subsample, appearance=self.appearance)

    def hud(self, status, reprojection, fps):
        """The status line drawn on (or sent with) every frame."""
        accel = joint_acceleration(np.stack(self.history), fps) if len(self.history) >= 3 else float("nan")
        line = f"{status}  reproj {reprojection:5.1f}px  accel {accel:8.0f}mm/s2  {fps:4.1f}fps"
        return f"{self.wardrobe[self.worn].name}  " + line if self.wardrobe else line


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class FrameSource:
    """Frames for the loop: a camera, a video file, or the smooth display's
    channel (`smooth_display.Channel`), each frame at most `max_width` wide.
    A camera that skips a frame is tolerated (`read` retries for ~1.2 s);
    a file that ends is the end."""

    def __init__(self, source, max_width):
        self.is_camera = getattr(source, "is_camera", str(source).isdigit())
        self._own = not hasattr(source, "read")
        self.capture = cv2.VideoCapture(int(source) if self.is_camera else source) if self._own else source
        if not self.capture.isOpened():
            raise SystemExit(f"Cannot open {source}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.max_width = max_width

    def read(self):
        """The next frame (BGR), or None at the end."""
        empty_reads = 0
        while True:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                break
            empty_reads += 1
            if not self.is_camera or empty_reads > 60:
                return None
            time.sleep(0.02)
        if self.max_width and frame.shape[1] > self.max_width:
            # a 1920 x 1080 webcam ran every stage at full size: render
            # 577 ms, removal 373 ms, 0.7 fps (measured, --profile)
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return frame

    def release(self):
        if self._own:
            self.capture.release()


class FrameOutput:
    """Where finished frames go: the window (unless `headless`), a video
    file, and with the smooth display its channel instead of the window.
    `show` returns the key pressed (or None) and whether to stop."""

    WINDOW = "Gaussian Garments try-on"

    def __init__(self, headless, output_video, channel):
        self.headless, self.output_video, self.channel = headless, output_video, channel
        self.writer, self.last_frame = None, None

    def show(self, frame, source_frame, points, hud, garment_alpha):
        key = None
        if self.channel is not None:
            self.channel.publish(source_frame, frame, points, hud, garment_alpha)
            key = self.channel.take_key()
            if self.channel.stopped:
                return key, True
        else:
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self.last_frame = frame
        if self.output_video is not None:
            if self.writer is None:
                height, width = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.output_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
            self.writer.write(frame)
        if not self.headless:
            cv2.imshow(self.WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return key, True
        return key, False

    def close(self, still=None):
        """Release the video and window; `still`: write the last frame there."""
        if self.writer is not None:
            self.writer.release()
        if not self.headless:
            cv2.destroyAllWindows()
        if still is not None and self.last_frame is not None:
            cv2.imwrite(str(still), self.last_frame)


def run(
    source,
    max_frames=0,
    headless=False,
    output=None,
    garment=None,
    garment_subsample=None,
    orientation_regressor=None,
    remove_clothing=False,
    output_video=None,
    parser=None,
    fill=None,
    appearance=False,
    background_plate=True,
    harmonize=False,
    body_regressor=None,
    profile=False,
    max_width=960,
    channel=None,
):
    """The synchronous live loop: every shown frame carries the garment drawn
    for it. `channel`: a `smooth_display.Channel` when the loop runs behind
    the smooth display: frames come from it (the display's newest camera
    frame) and every finished frame is published to it instead of shown.

    With PREFETCH the next frame is read, its landmarks found and its body
    regression sent before this frame is drawn (`LiveSession.prefetch`)."""
    import mediapipe as mp

    session = LiveSession(garment, garment_subsample, orientation_regressor, remove_clothing, parser, fill, appearance,
                          background_plate, harmonize, body_regressor, headless)
    stage_times = collections.OrderedDict()
    frame_times = collections.deque(maxlen=30)
    out = FrameOutput(headless, output_video, channel)
    frames, read, hud = 0, 0, "no frames"
    # Build the pose model before touching the camera. MediaPipe brings up
    # its own GL context on start, and on this Mac a camera opened before
    # that returned nothing on its first read - which ended the session.
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=POSE_COMPLEXITY) as pose:
        frames_in = FrameSource(source, max_width)

        def next_frame():
            """(frame, points, stamp, stage, source_frame) of the next frame,
            its landmarks found and its regression sent; None at the end."""
            nonlocal read
            frame = frames_in.read()
            if frame is None:
                return None
            source_frame = frame.copy() if channel is not None else None
            stage = StageTimer(stage_times)
            height, width = frame.shape[:2]
            points = landmarks_from(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), width, height)
            stage("mediapipe")
            stamp = time.perf_counter() if frames_in.is_camera else read / frames_in.fps
            read += 1
            session.prefetch(frame, points)
            return frame, points, stamp, stage, source_frame

        def more():
            return not (max_frames and frames + 1 >= max_frames)

        finished = time.perf_counter()
        upcoming = next_frame()
        while upcoming is not None:
            frame, points, stamp, stage, source_frame = upcoming
            upcoming = None
            if PREFETCH and more():
                upcoming = next_frame()  # its regression runs while this frame is drawn
                stage.restart()
            status, reprojection, garment_alpha = session.process(frame, points, stamp, stage, source_frame)
            if not PREFETCH and more():
                upcoming = next_frame()
            now = time.perf_counter()
            frame_times.append(now - finished)  # frame to frame: the rate frames are shown at
            finished = now
            stage("total", since_start=True)
            hud = session.hud(status, reprojection, len(frame_times) / max(sum(frame_times), 1e-6))
            key, stop = out.show(frame, source_frame, points, hud, garment_alpha)
            if stop and channel is not None and channel.stopped:
                break  # the display closed: this frame was not shown
            frames += 1
            if stop:
                break
            if key in (ord("n"), ord("p")):
                session.switch_garment(1 if key == ord("n") else -1)
            if max_frames and frames >= max_frames:
                break
        frames_in.release()
    session.finish()
    out.close(still=output)
    print(f"{frames} frames  last: {hud}")
    if profile and stage_times:
        print("median ms a frame: " + "  ".join(f"{k} {1000 * np.median(v):.0f}" for k, v in stage_times.items()))
    return 0
