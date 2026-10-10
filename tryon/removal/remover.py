"""Remove the wearer's real clothing before drawing the new one.

The renderer puts a garment ON TOP of the frame. Wherever the real
garment is bulkier than the rendered one (a fur jacket under a slim coat,
sleeves the render's silhouette does not reach), the real clothing shows
around the edges and the result reads as "a coat drawn over a jacket",
not "a person wearing the coat". This module erases the real clothing
first.

    frame --SCHP / FASHN parser--> per-pixel clothing probability
          --threshold + dilate--> mask of the wearer's real clothing
          --fill--> frame with that region filled from its surroundings
                    (skin, background plate, or LaMa)
          --alpha composite--> the rendered garment drawn over the clean base

The default fill is Telea's fast-marching inpainting (`cv2.inpaint` with
`INPAINT_TELEA`), a classical, deterministic fill from the hole's boundary
inward, blended toward the wearer's skin colour on the arms; LaMa is the
optional learned fill. Everything the rendered garment covers is
overwritten anyway, so the fill only has to be plausible in the fringe the
render leaves exposed. The WHOLE detected-clothing region is filled (not
just the exposed fringe) so the fill draws from skin and background rather
than from neighbouring real-garment pixels that would smear its colour
into the fringe.

The parser's mask is a few pixels off the true boundary at frame
resolution, which is why the mask is dilated before the fill - better to
erase a little skin at the boundary than leave a rim of the real garment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from tryon.paths import MODELS


# MediaPipe pose landmark indices (shoulder, elbow, wrist) per arm.
_ARMS = ((11, 13, 15), (12, 14, 16))
_VISIBLE = 0.5


def arm_strokes(keypoints, shape, width_fraction=0.45, wrist_extension=1.3, hand_fraction=0.12):
    """Boolean mask covering each arm as a thick polyline, from keypoints.

    Stroke width is a fraction of the shoulder width (so it scales with
    the person), the wrist is extended past itself by `wrist_extension`
    (IDM-VTON's rule, so the cuff is covered), and a disc around the
    extended wrist is cut back out so the hand stays. Arms whose three
    landmarks are not all visible are left alone.

    Width measured on the camera-0004 frame against the fur sleeves the
    segmenter misses (35,587 colourful pixels along the arms): 0.32 left
    87% of them and erased 356 non-fur pixels, 0.45 left 78% / 697,
    0.6 left 66% / 1,147. 0.45 is the middle; a fur sleeve twice as
    thick as the arm is beyond what a stroke should cover and is left
    to the parser.
    """
    keypoints = np.asarray(keypoints, np.float64)
    height, width = shape[:2]
    stroke = np.zeros((height, width), np.uint8)
    hands = np.zeros((height, width), np.uint8)
    left, right = keypoints[11], keypoints[12]
    if left[2] < _VISIBLE or right[2] < _VISIBLE:
        return stroke.astype(bool)
    shoulder_width = float(np.linalg.norm(left[:2] - right[:2]))
    thickness = max(int(round(shoulder_width * width_fraction)), 3)
    hand_radius = max(int(round(shoulder_width * hand_fraction)), 2)
    for shoulder, elbow, wrist in _ARMS:
        s, e, w = keypoints[shoulder], keypoints[elbow], keypoints[wrist]
        if min(s[2], e[2], w[2]) < _VISIBLE:
            continue
        w_ext = e[:2] + wrist_extension * (w[:2] - e[:2])
        points = np.array([s[:2], e[:2], w_ext], np.int32).reshape(-1, 1, 2)
        cv2.polylines(stroke, [points], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_AA)
        for p in (s[:2], e[:2], w_ext):
            cv2.circle(stroke, (int(p[0]), int(p[1])), thickness // 2, 1, -1)
        cv2.circle(hands, (int(w_ext[0]), int(w_ext[1])), hand_radius, 1, -1)
    stroke[hands > 0] = 0
    return stroke.astype(bool)


def hand_discs(keypoints, shape, wrist_extension=1.3, hand_fraction=0.12):
    """Boolean mask of the discs `arm_strokes` keeps for the hands, so a
    fill can leave them alone (the LaMa margin otherwise eats them)."""
    keypoints = np.asarray(keypoints, np.float64)
    hands = np.zeros(shape[:2], np.uint8)
    left, right = keypoints[11], keypoints[12]
    if left[2] < _VISIBLE or right[2] < _VISIBLE:
        return hands.astype(bool)
    radius = max(int(round(float(np.linalg.norm(left[:2] - right[:2])) * hand_fraction)), 2)
    for shoulder, elbow, wrist in _ARMS:
        e, w = keypoints[elbow], keypoints[wrist]
        if min(e[2], w[2]) < _VISIBLE:
            continue
        w_ext = e[:2] + wrist_extension * (w[:2] - e[:2])
        cv2.circle(hands, (int(w_ext[0]), int(w_ext[1])), radius, 1, -1)
    return hands.astype(bool)


# A bare arm narrows from the upper arm to the wrist: width at the elbow and at
# the wrist as a share of the width at the shoulder, and how far past the wrist
# keypoint (share of the forearm) the tube runs before the hand takes over.
TUBE_ELBOW = 0.9
TUBE_WRIST = 0.7
TUBE_PAST_WRIST = 0.12
_TUBE_STEP = 3.0  # px between the discs that make the tube


def arm_tube(keypoints, shape, width_fraction=0.22):
    """(mask, half-width) of the bare arms as TAPERED tubes along each shoulder-elbow-wrist
    polyline: full width (`width_fraction` of the shoulder width) at the shoulder, `TUBE_ELBOW` at
    the elbow, `TUBE_WRIST` at the wrist. The half-width map (px, 0 outside) lets a fill shade each
    point against the tube's own radius. Arms whose landmarks are not all visible are skipped."""
    keypoints = np.asarray(keypoints, np.float64)
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    radius = np.zeros((height, width), np.float32)
    left, right = keypoints[11], keypoints[12]
    if left[2] < _VISIBLE or right[2] < _VISIBLE:
        return mask.astype(bool), radius
    shoulder_width = float(np.linalg.norm(left[:2] - right[:2]))
    full = max(shoulder_width * width_fraction / 2.0, 1.5)
    for shoulder, elbow, wrist in _ARMS:
        s, e, w = keypoints[shoulder], keypoints[elbow], keypoints[wrist]
        if min(s[2], e[2], w[2]) < _VISIBLE or not np.isfinite(keypoints[[shoulder, elbow, wrist], :2]).all():
            continue
        end = w[:2] + TUBE_PAST_WRIST * (w[:2] - e[:2])
        path = ((s[:2], e[:2], full, full * TUBE_ELBOW), (e[:2], end, full * TUBE_ELBOW, full * TUBE_WRIST))
        for a, b, ra, rb in path:
            steps = max(int(np.hypot(*(b - a)) / _TUBE_STEP), 1)
            for k in range(steps + 1):
                t = k / steps
                centre = (int(round((a + t * (b - a))[0])), int(round((a + t * (b - a))[1])))
                r = ra + t * (rb - ra)
                disc = max(int(round(r)), 1)
                cv2.circle(mask, centre, disc, 1, -1)
                cv2.circle(radius, centre, disc, float(r), -1)
    return mask.astype(bool), radius


# ATR labels of the wearer's LOWER body: skirt, pants, left / right leg.
LOWER_BODY_CLASSES = (5, 6, 12, 13)
# ATR labels of bare skin a fill must keep: left / right arm (hands included).
SKIN_CLASSES = (14, 15)
# Rows of lower-body pixels whose 5th percentile is the parser's own hip
# line, when no keypoints give one.
_HIP_ROW_PERCENTILE = 5.0


# A long shirt hangs from the torso: the dress-labelled pixels below the hip line that
# are one region with the upper clothing above it, still at least this share of the torso's
# width a few rows below the hip line, are the shirt's hem, not a leak. Rows of the
# region above the hip line the torso width is read from, and the row below it where
# the width is read.
HANG_WIDTH = 0.5
HANG_ROWS = 12
HANG_BELOW = 4


def hanging_hem(labels, hip_y):
    """Boolean mask of the upper-clothing (ATR 4 / 7) pixels below `hip_y` that belong to a
    garment hanging from the torso: the 8-connected region reaches above the hip line and is
    still `HANG_WIDTH` of the torso's width (median row width over the `HANG_ROWS` rows above the
    hip line) `HANG_BELOW` rows below it. A blob the parser leaves under a short jacket's hem
    (camera 0076) is narrower than the torso there, or not joined to it."""
    labels = np.asarray(labels)
    height = labels.shape[0]
    hip_row = int(round(hip_y))
    out = np.zeros(labels.shape, bool)
    if hip_row < HANG_ROWS or hip_row + HANG_BELOW >= height:
        return out
    upper = np.isin(labels, (4, 7)).astype(np.uint8)
    count, region = cv2.connectedComponents(upper, connectivity=8)
    above = region[hip_row - HANG_ROWS : hip_row + 1]
    for index in np.unique(above[above > 0]):
        member = region == index
        widths = member[hip_row - HANG_ROWS : hip_row + 1].sum(axis=1)
        widths = widths[widths > 0]
        if widths.size < HANG_ROWS // 2:
            continue
        if member[hip_row + HANG_BELOW].sum() >= HANG_WIDTH * np.median(widths):
            out |= member
    return out


def hip_line(labels, keypoints=None):
    """Row of the hip line: the two hip landmarks' mean when visible, else the 5th percentile
    row of the parser's lower-body region; None when there is neither."""
    if keypoints is not None:
        keypoints = np.asarray(keypoints, np.float64)
        if min(keypoints[23, 2], keypoints[24, 2]) >= _VISIBLE:
            return float((keypoints[23, 1] + keypoints[24, 1]) / 2)
    rows = np.nonzero(np.isin(labels, LOWER_BODY_CLASSES).sum(axis=1) >= 3)[0]
    return float(np.percentile(rows, _HIP_ROW_PERCENTILE)) if len(rows) else None


# The long plaid hem of 00152_outer is labelled LEG (12 / 13) by the parser where it
# overlaps the thighs, 4-9k px a frame (measured; docs/LOG.md, 9 Oct), so no label
# rule can tell it from a leg. Its colour can: a candidate below the hip line is given back
# to the erase when it is within GARMENT_MATCH CIELAB units of a colour the garment itself
# shows above the hip line (colours holding at least GARMENT_MIN_SHARE of its pixels, on a
# GARMENT_BIN grid) and is one region with the erase mask.
GARMENT_BIN = 8.0
GARMENT_MATCH = 12.0
GARMENT_MIN_SHARE = 0.01
GARMENT_MIN_PIXELS = 400


def _lab(bgr):
    return cv2.cvtColor((np.asarray(bgr, np.float32) / 255.0).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)


def hem_rescue(frame_bgr, mask, leak, clothing, hip_y):
    """Boolean mask of the `hem_leak` pixels that are the garment's own hem: in `mask` and in `leak`,
    colour-matched to the garment's pixels above `hip_y` (the parser's `clothing` inside `mask`,
    eroded 5 px), and connected to the part of `mask` the leak leaves."""
    candidates = leak & mask
    if hip_y is None or int(candidates.sum()) == 0:
        return candidates
    above = np.arange(mask.shape[0])[:, None] < hip_y
    reference = cv2.erode((clothing & mask & above).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if int(reference.sum()) < GARMENT_MIN_PIXELS:
        return np.zeros_like(candidates)
    lab = _lab(frame_bgr[reference][:: max(int(reference.sum()) // 3000, 1)])
    cells, inverse, counts = np.unique(np.floor(lab / GARMENT_BIN).astype(np.int32), axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    keep = counts >= GARMENT_MIN_SHARE * len(lab)
    if not keep.any():
        return np.zeros_like(candidates)
    centres = np.stack([lab[inverse == k].mean(axis=0) for k in np.flatnonzero(keep)])
    rows, cols = np.flatnonzero(candidates.any(axis=1)), np.flatnonzero(candidates.any(axis=0))
    window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    pixels = _lab(frame_bgr[window][candidates[window]])
    nearest = np.sqrt(((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    matched = np.zeros_like(candidates)
    matched[window][candidates[window]] = nearest <= GARMENT_MATCH
    # one region with the mask that stays: a match is the hem only when it hangs from the garment
    stays = mask & ~leak
    wide = (slice(max(rows[0] - 1, 0), rows[-1] + 2), slice(max(cols[0] - 1, 0), cols[-1] + 2))
    joined = (matched | stays)[wide].astype(np.uint8)
    count, region = cv2.connectedComponents(joined, connectivity=8)
    touching = np.unique(region[stays[wide] & (region > 0)])
    full = np.zeros(mask.shape, np.int32)
    full[wide] = region
    return matched & np.isin(full, touching[touching > 0])


def hem_leak(labels, keypoints=None, hanging=True):
    """Pixels below the hip line that the parser itself assigns to the
    lower body, or calls "dress" on a wearer who is wearing trousers.

    Measured on the annotated profile camera 0076 (SCHP ATR, warm
    landmarks): the parser's upper-clothing mask ran 4,865 px below the
    hips onto the trousers as a round blob under the hem - 1,450 of them
    argmax'd "pants" (the 6 px dilation's rim along the hem plus part of
    the blob) and 2,120 "dress" (2,151 dress pixels in the whole frame,
    all of them in the blob, on a wearer the same parser gives 10,680
    pixels of pants). A jacket over trousers does not continue below the
    hip as a dress, so where the parser sees more trousers/skirt than
    dress, its dress label below the hip line is the leak. The front
    camera's mask has 10-15 px below its hip line and is untouched.

    The hip line is the mean of the two hip landmarks when `keypoints`
    are given (visibility >= `_VISIBLE`), else the top of the parser's
    own lower-body region (5th percentile row of rows with at least a
    few lower-body pixels: 405 vs the landmarks' 400 on camera 0076).

    A long shirt over shorts is the opposite case (00152_outer, a plaid shirt to
    the thigh over denim shorts): its hem is labelled dress below the hip line, and
    with the rule above the real plaid hem was dropped from the erase and stayed
    visible round the new garment. Dress pixels that are one region with the torso's
    upper clothing and as wide as the torso below the hip line (`hanging_hem`) stay in
    the mask; `hanging=False` is the rule without that exception.
    """
    labels = np.asarray(labels)
    lower = np.isin(labels, LOWER_BODY_CLASSES)
    hip_y = hip_line(labels, keypoints)
    if hip_y is None:
        return np.zeros(labels.shape, bool)
    below = np.arange(labels.shape[0])[:, None] > hip_y
    leak = lower
    if np.isin(labels, (5, 6)).sum() > (labels == 7).sum():
        dress = labels == 7
        if hanging and (dress & below).any():
            dress &= ~hanging_hem(labels, hip_y)
        leak = leak | dress
    return leak & below


# Pixels with less probability than this do not seed the guided filter's crop.
_CROP_SEED = 0.05
# The crop must cover the filter's reach (two box filters of `radius`: 2 * radius)
# plus one pixel of rounding; a crop wider than this share of the frame is not worth it.
_CROP_MAX_SHARE = 0.7


def _guided_refine_full(frame_bgr, probability, radius=8, eps=1e-3):
    """The guided filter on the whole frame (`_guided_refine`'s oracle)."""
    try:
        return cv2.ximgproc.guidedFilter(frame_bgr, probability.astype(np.float32), radius, eps)
    except AttributeError:
        return cv2.bilateralFilter(probability.astype(np.float32), 9, 0.1, 9)


def _guided_refine(frame_bgr, probability, radius=8, eps=1e-3):
    """Guided filter (He et al. 2010) of the probability map under the
    image, so its edges follow the image's; joint bilateral fallback when
    cv2.ximgproc is not built in.

    Run on the bounding box of the probability above `_CROP_SEED` plus
    2 * radius + 1 px (the filter's reach), the rest of the map as it came:
    the filter changes a pixel only through the window around it, so the
    result inside the box equals the full-frame one, and outside it the map
    is below `_CROP_SEED` either way (the 0.5 threshold never sees it)."""
    probability = np.asarray(probability, np.float32)
    seed = probability > _CROP_SEED
    rows, cols = np.flatnonzero(seed.any(axis=1)), np.flatnonzero(seed.any(axis=0))
    if rows.size == 0:
        return probability.copy()
    height, width = probability.shape
    margin = 2 * radius + 1
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + 1 + margin, height)
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + 1 + margin, width)
    if (y1 - y0) * (x1 - x0) > _CROP_MAX_SHARE * height * width:
        return _guided_refine_full(frame_bgr, probability, radius, eps)
    out = probability.copy()
    out[y0:y1, x0:x1] = _guided_refine_full(np.ascontiguousarray(frame_bgr[y0:y1, x0:x1]), np.ascontiguousarray(probability[y0:y1, x0:x1]), radius, eps)
    return out


@dataclass(frozen=True)
class ClothingParse:
    """What one frame's clothing parse produced, returned by
    `ClothingRemover.parse_clothing` so that two frames' parses can never be
    mixed through shared state: `mask` the pixels to erase, `labels` the
    parser's per-pixel ATR labels, `clothing` the raw (undilated, stroke-free)
    clothing the parser itself saw. Treat the arrays as read-only."""

    mask: np.ndarray
    labels: np.ndarray | None
    clothing: np.ndarray
    points: np.ndarray | None = None  # the (33, 3) keypoints the parse was made with, for the arm tubes


class ClothingRemover:
    """Erase the wearer's detected clothing so a rendered garment can be
    composited onto a clean base.

    `schp_checkpoint`: the pretrained SCHP parser (ONNX, the parser
    IDM-VTON runs; `tryon/removal/parsing_schp.py`), whose upper-clothes +
    dress probability is the mask. `fashn_checkpoint`: the FASHN SegFormer
    parser instead (`tryon/removal/parsing_fashn.py`); SCHP is still
    loaded beside it when its weights exist, for the face colour and the
    inner-layer rule.
    `threshold`: clothing probability above which a pixel is erased.
    `dilate_px`: how far to grow the mask at frame resolution, to cover
    the parser's boundary blur (see module docstring).
    `inpaint_radius`: `cv2.inpaint`'s neighbourhood radius in pixels.
    """

    def __init__(self, schp_checkpoint=None, fashn_checkpoint=None, threshold=0.5, dilate_px=6, inpaint_radius=5):
        # SCHP upper-clothing IoU / BF@2px on 50 test images a source:
        # lip_style 0.889 / 0.817, plus_neck 0.834 / 0.650, vitonhd 0.922 /
        # 0.771. It also segments fur sleeves (camera 0004: 29,124 masked
        # pixels).
        self.schp = None
        # FASHN is better than SCHP on all four test sources (upper-clothing
        # IoU / BF@2px, 100 images each, lip_style 0.929 / 0.850 vs SCHP
        # 0.884 / 0.802, plus_neck 0.881 / 0.672 vs 0.844 / 0.664,
        # fashionpedia 0.680 / 0.630 vs 0.590 / 0.513, vitonhd 0.951 / 0.809
        # vs 0.927 / 0.769) but ~800 vs ~100 ms a frame on the CPU, so it is
        # the quality path's parser, not the live default.
        self.fashn = None
        if fashn_checkpoint is not None:
            from tryon.removal.parsing_fashn import FashnParser

            self.fashn = FashnParser(fashn_checkpoint)
            if schp_checkpoint is None:
                from tryon.removal.parsing_schp import ATR_CHECKPOINT

                schp_checkpoint = ATR_CHECKPOINT if ATR_CHECKPOINT.exists() else None
        if schp_checkpoint is not None and not Path(schp_checkpoint).exists():
            schp_checkpoint = None
        if schp_checkpoint is not None:
            from tryon.removal.parsing_schp import SCHPParser

            self.schp = SCHPParser(schp_checkpoint)
        if self.schp is None and self.fashn is None:
            raise RuntimeError("ClothingRemover needs a clothing parser: run ./setup.sh (tools.data.download_pretrained schp) for SCHP")
        # The LIP-trained SCHP separates "Coat" (7) from "Upper-clothes"
        # (5), which ATR merges; see `KEEP_INNER_LAYER`.
        self.lip = None
        if self.schp is not None and self.KEEP_INNER_LAYER:
            from tryon.removal.parsing_schp import LIP_CHECKPOINT, SCHPParser

            if LIP_CHECKPOINT.exists():
                self.lip = SCHPParser(LIP_CHECKPOINT, size=473)
        self.threshold = float(threshold)
        self.dilate_px = int(dilate_px)
        self.inpaint_radius = int(inpaint_radius)

    # Flip test-time augmentation (SCHP only, see `SCHP_FLIP`) and image-guided
    # edge refinement of the probability map.
    tta = True
    refine = "guided"

    def clothing_probability(self, frame_bgr):
        """Per-pixel probability, at frame resolution, that a pixel is the
        wearer's own clothing (upper-clothes + dress). Probabilities are
        upsampled, not labels, so the edge keeps a soft ramp. (Also leaves
        the labels in `_last_labels`: the single-threaded legacy interface.)"""
        probability, self._last_labels = self._probability_and_labels(frame_bgr)
        return probability

    def _probability_and_labels(self, frame_bgr):
        """(clothing probability, the parser's ATR labels), no shared state."""
        if self.fashn is not None:
            from tryon.removal.parsing_fashn import TO_ATR, UPPER_CLASSES

            parsed = self.fashn.probabilities(frame_bgr)
            labels = TO_ATR[parsed.argmax(axis=0)]
            clothing = parsed[list(UPPER_CLASSES)].sum(axis=0)
            return (_guided_refine(frame_bgr, clothing) if self.refine == "guided" else clothing), labels
        if not (self.tta and self.SCHP_FLIP):
            # softmax and argmax on the network's square, only the two maps
            # warped to the frame (`fast_clothing`): 131 -> 109 ms, mask IoU
            # 0.996-0.999 and 99.7 % of labels as before (9 frames, 3 clips)
            clothing, labels = self.schp.fast_clothing(frame_bgr, (4, 7))
            return (_guided_refine(frame_bgr, clothing) if self.refine == "guided" else clothing), labels
        parsed = self.schp.probabilities(frame_bgr, flip=self.tta and self.SCHP_FLIP)
        labels = parsed.argmax(axis=0)
        clothing = parsed[4] + parsed[7]
        return (_guided_refine(frame_bgr, clothing) if self.refine == "guided" else clothing), labels

    # Keep an undershirt: when the wearer has a coat (LIP "Coat" covers at
    # least COAT_FRACTION of the LIP upper-body clothing), the pixels LIP
    # calls "Upper-clothes" are the inner layer - a T-shirt or henley under
    # an open jacket - and stay; only the coat is erased. ATR calls both
    # "upper clothes", so the removal erased the undershirt that should
    # show through the new garment's opening (00122_outer's white henley;
    # the white T-shirts under the fur and white jackets). LIP separated
    # the layers on all four clips checked (coat 30-43k px, inner 0.5-17k).
    KEEP_INNER_LAYER = True
    # Give back to the erase the lower-body-labelled pixels below the hip line that are the garment's
    # own colour and hang from it (a long shirt over shorts), see `hem_rescue`.
    HEM_COLOUR = True
    COAT_FRACTION = 0.5
    # While LIP finds no coat, it runs on only every LIP_RECHECK-th frame
    # (the other frames: no inner layer, as it found): most wearers have no
    # open coat, and the pass is ~87 ms of the GPU the body regressor and
    # the render share. A coat is looked for again within LIP_RECHECK
    # frames (~3 s at 2 fps); once seen, LIP runs on every frame.
    LIP_RECHECK = 6

    def _inner_layer(self, frame_bgr):
        """Boolean mask of the inner upper layer to keep, or None."""
        if self.lip is None:
            return None
        skip = getattr(self, "_lip_skip", 0)
        if skip > 0:
            self._lip_skip = skip - 1
            return None
        # one pass, no flip (`fast_labels`): 213 -> 87 ms; 95-98 % of the
        # labels as the flipped pair, the coat / no-coat decision the same
        # on all 9 frames checked (3 clips), the kept inner layer 0-20 %
        # smaller where there is one
        labels = self.lip.fast_labels(frame_bgr)
        # "Jumpsuit" (10) too: LIP read the turn clip's white T-shirt and
        # beige trousers as one piece (8,808 px jumpsuit, 561 upper); the
        # trousers are never in the removal mask, so it only keeps the top.
        coat, inner = labels == 7, np.isin(labels, (5, 10))
        if coat.sum() < self.COAT_FRACTION * max(int((coat | inner).sum()), 1):
            self._lip_skip = self.LIP_RECHECK - 1
            return None
        # eroded, so the coat's own edge pixels that bleed into the class
        # boundary are still erased
        inner = cv2.erode(inner.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        return inner

    def clothing_mask(self, frame_bgr, keypoints=None):
        """`parse_clothing(...).mask`, also leaving the parse in `_last_labels`
        / `_last_clothing` for `remove` and `skin_colour` (the single-threaded
        legacy interface; the live loop passes the `ClothingParse` instead)."""
        parse = self.parse_clothing(frame_bgr, keypoints)
        self._last_labels, self._last_clothing, self._last_points = parse.labels, parse.clothing, parse.points
        return parse.mask

    def parse_clothing(self, frame_bgr, keypoints=None):
        """The `ClothingParse` of a frame; the mask is a boolean mask of pixels to erase: thresholded probability, grown
        by `dilate_px` (see the module docstring for why it is grown).

        `keypoints`: MediaPipe's (33, 3) landmarks in pixels; when given,
        the arms are added as strokes (see `arm_strokes`), which is how
        the try-on literature builds its "clothing-agnostic" region
        (IDM-VTON / OOTDiffusion / DCI-VTON: parsing for the torso, thick
        shoulder-elbow-wrist strokes for the sleeves, hands kept). The
        parser alone left the fur sleeves on the camera-0004 frame
        (measured probability 0.05-0.11 on them); the strokes cover
        whatever the wearer's arm is wearing without needing to recognise
        it, and the rendered garment's sleeves land on top.
        """
        # The LIP pass (the inner-layer rule) is independent of the clothing
        # parse and as long: run it beside it. Identical outputs; mask job
        # 406 -> 314 ms on the dancer clip under load (ONNX releases the GIL).
        inner_job = None
        if self.lip is not None and self.KEEP_INNER_LAYER:
            if getattr(self, "_lip_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor

                self._lip_pool = ThreadPoolExecutor(1)
            inner_job = self._lip_pool.submit(self._inner_layer, frame_bgr)
        probability, labels = self._probability_and_labels(frame_bgr)
        mask = (probability > self.threshold).astype(np.uint8)
        # the raw (undilated, stroke-free) clothing, so `remove(keep=...)`
        # can refuse to keep pixels the parser itself calls clothing
        clothing = mask.astype(bool)
        if self.dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilate_px + 1,) * 2)
            mask = cv2.dilate(mask, kernel)
        inner = inner_job.result() if inner_job is not None else self._inner_layer(frame_bgr)
        if inner is not None and keypoints is not None:
            # never on the arms: LIP called 00122_outer's raised plaid
            # sleeve "Upper-clothes" and it was kept; an undershirt that
            # matters shows on the torso, through the new garment's opening
            inner = inner & ~arm_strokes(keypoints, mask.shape)
        if inner is not None:
            mask &= ~inner
            clothing &= ~inner
        if labels is not None:
            leak = hem_leak(labels, keypoints)
            if self.HEM_COLOUR and leak.any():
                leak &= ~hem_rescue(frame_bgr, mask.astype(bool), leak, clothing, hip_line(labels, keypoints))
            mask &= ~leak
        if keypoints is not None:
            mask |= arm_strokes(keypoints, mask.shape).astype(np.uint8)
        return ClothingParse(mask.astype(bool), labels, clothing, None if keypoints is None else np.asarray(keypoints, np.float64))

    # How the removed region is filled: "skin" (Telea blended toward the
    # wearer's skin colour) or "lama" (LaMa, Suvorov et al.
    # 2022, a pretrained Fourier-convolution inpainter; TorchScript weights at
    # models/pretrained/big-lama.pt, the file simple-lama-inpainting ships -
    # that package itself is not installed because it pins an old Pillow
    # and a different torch).
    fill = "skin"
    LAMA_CHECKPOINT = MODELS / "pretrained" / "big-lama.pt"
    LAMA_MARGIN = 15  # px of extra hole around the mask
    LAMA_CONTEXT = 64  # px of image kept around the hole's bounding box
    LAMA_MAX_SIDE = 384  # longer side of the crop LaMa runs on (512 and 384 clean, 256 blotchy)
    _lama_model = None

    def _lama(self, frame_bgr, mask, keep=None):
        """Input: RGB in [0, 1] and a 0/1 mask, sides padded to a multiple
        of 8; output RGB in [0, 1]. Runs on the hole's bounding box only
        (plus LAMA_CONTEXT px of context) with the longer side capped at
        LAMA_MAX_SIDE: the cost is proportional to the area (full 870x640
        frame 24 s on this CPU, 4 s at half size) and MPS cannot run the
        Fourier convolutions."""
        if ClothingRemover._lama_model is None:
            ClothingRemover._lama_model = torch.jit.load(str(self.LAMA_CHECKPOINT), map_location="cpu").eval()
        # LaMa continues whatever texture touches the hole, so the margin
        # must swallow the fur hairs / halo the parser leaves at the garment
        # boundary: on camera 0004 a 5 px margin painted fur strands into
        # the hole, 15 px gave a clean fill (31 px no better).
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((self.LAMA_MARGIN, self.LAMA_MARGIN), np.uint8)).astype(bool)
        if keep is not None:
            grown &= ~np.asarray(keep, bool)
        rows, cols = np.nonzero(grown)
        if rows.size == 0:
            return frame_bgr.copy()
        height, width = frame_bgr.shape[:2]
        y0, y1 = max(rows.min() - self.LAMA_CONTEXT, 0), min(rows.max() + 1 + self.LAMA_CONTEXT, height)
        x0, x1 = max(cols.min() - self.LAMA_CONTEXT, 0), min(cols.max() + 1 + self.LAMA_CONTEXT, width)
        crop, hole = frame_bgr[y0:y1, x0:x1], grown[y0:y1, x0:x1]
        scale = min(1.0, self.LAMA_MAX_SIDE / max(crop.shape[:2]))
        run_h, run_w = max(8, round(crop.shape[0] * scale / 8) * 8), max(8, round(crop.shape[1] * scale / 8) * 8)
        rgb = cv2.cvtColor(cv2.resize(crop, (run_w, run_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        small_hole = cv2.resize(hole.astype(np.float32), (run_w, run_h), interpolation=cv2.INTER_LINEAR) > 0
        with torch.no_grad():
            out = ClothingRemover._lama_model(torch.from_numpy(rgb).permute(2, 0, 1)[None], torch.from_numpy(small_hole.astype(np.float32))[None, None])
        out = np.clip(out[0].permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)
        result = frame_bgr.copy()
        # keep every untouched pixel exactly as it was
        result[y0:y1, x0:x1][hole] = out[hole]
        return result

    FACE_CLASS = 11  # LIP/ATR id of the face, the skin sample
    SKIN_BLEND = 0.7  # share of the estimated skin colour in the fill; the rest is the inpaint

    def _parse_state(self, parse):
        """(labels, raw clothing) of `parse`, or of the last `clothing_mask`
        call when no parse is given."""
        if parse is not None:
            return parse.labels, parse.clothing
        return getattr(self, "_last_labels", None), getattr(self, "_last_clothing", None)

    def skin_colour(self, frame_bgr, parse=None):
        """Median BGR of the wearer's face pixels (ATR class 11), or None
        when no face is confidently seen. What the removed region is filled
        with: a torso under an open coat, or a forearm past a sleeve, IS
        skin, and the inpainter cannot know that - it fills from the
        boundary, which on a fur coat is fur. `parse`: this frame's
        `ClothingParse` (default: the last `clothing_mask` call's)."""
        labels, _ = self._parse_state(parse)
        if labels is not None and labels.shape == frame_bgr.shape[:2]:
            # the parse `clothing_mask` just ran on this frame: a second
            # SCHP pass for the face alone cost ~150 ms a frame
            face = labels == self.FACE_CLASS
        elif self.schp is not None:
            # same ATR ids, so the face is class 11 there too
            face = self.schp.probabilities(frame_bgr, flip=False)[self.FACE_CLASS] > 0.5
        else:
            return None
        if face.sum() < 50:
            return None
        return np.median(frame_bgr[face].reshape(-1, 3), axis=0)

    # Width of the arm tubes drawn over a LaMa fill, as a fraction of the
    # shoulder width (the mask strokes are 0.45 wide to swallow sleeves;
    # a bare arm is nearer 0.22 - 0.30 looked like sausages on camera 0004).
    ARM_TUBE_WIDTH = 0.22
    # Flip-averaged SCHP (two network passes) or one. 300 test images each
    # of lip_style / vitonhd (measured with a one-pass flag on the parser benchmark): IoU 0.875
    # / 0.933 -> 0.874 / 0.932, BF@2px 0.785 / 0.770 -> 0.780 / 0.768, at
    # 256 -> 145 ms an image. One pass.
    SCHP_FLIP = False
    # Keep the parser's own arm-skin pixels through the fill (see `remove`).
    KEEP_PARSED_SKIN = True
    # The "skin" fill tints only the arm strokes (see `remove`).
    SKIN_ON_ARMS_ONLY = True

    # The Telea fill of the whole hole runs at this fraction of the frame
    # and is scaled back up: its cost grows with the hole's area, and the
    # fill is low-frequency (then mostly under the garment). Frames saved
    # from the live loop (640 wide): 57-132 -> ~10 ms, 26-32 dB from the
    # full-size fill inside the hole, no visible difference. The seam ring
    # is still inpainted at full size.
    INPAINT_SCALE = 0.5

    def _telea(self, frame_bgr, mask):
        mask = np.asarray(mask, bool)
        scale = self.INPAINT_SCALE
        if scale >= 1.0:
            return cv2.inpaint(frame_bgr, mask.astype(np.uint8), self.inpaint_radius, cv2.INPAINT_TELEA)
        height, width = mask.shape
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hole = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        # grown by a pixel so no clothing pixel at the hole's edge leaks
        # into the fill through the downscale
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        radius = max(1, int(round(self.inpaint_radius * scale)))
        filled = cv2.resize(cv2.inpaint(small, hole, radius, cv2.INPAINT_TELEA), (width, height), interpolation=cv2.INTER_LINEAR)
        out = frame_bgr.copy()
        out[mask] = filled[mask]
        return out

    def remove(self, frame_bgr, mask=None, skin_bgr=None, arms=None, keep=None, plate=None, body=None, parse=None):
        """A copy of `frame_bgr` with the wearer's clothing removed.

        `mask` (optional) skips re-running the parser when the caller
        already has this frame's `clothing_mask`. `arms` (optional): the
        keypoint arm strokes; `keep` (optional): pixels the fill must not
        touch (the hands). `plate` (a `tryon.removal.background.BackgroundPlate`)
        and `body` (where the wearer's own body is, e.g.
        `background.body_silhouette`): hole pixels outside the body that
        the plate has seen are filled with the observed background first
        (`_prefill_from_plate`); the rest by the fill, `skin` (`_fill_skin`)
        or `lama` (`_fill_lama`). `parse`: this frame's `ClothingParse` (its
        labels and raw clothing are what the fills read; default: the last
        `clothing_mask` call's).
        """
        if mask is None:
            parse = self.parse_clothing(frame_bgr)
            mask = parse.mask
        labels, clothing = self._parse_state(parse)
        if not mask.any():
            return frame_bgr.copy()
        if self.fill == "lama":
            keep = self._lama_keep(mask, keep, labels, clothing)
        frame_bgr, mask, keep = self._prefill_from_plate(frame_bgr, mask, keep, plate, body)
        if not mask.any():
            return frame_bgr.copy()
        if self.fill == "lama":
            return self._fill_lama(frame_bgr, mask, keep, arms, skin_bgr, parse)
        if self.LABEL_FILL:
            labels, clothing = self._parse_state(parse)
            if labels is not None and labels.shape == mask.shape:
                points = parse.points if parse is not None else getattr(self, "_last_points", None)
                return self._fill_labels(frame_bgr, mask, arms, skin_bgr, labels, clothing, body, points)
        return self._fill_skin(frame_bgr, mask, arms, skin_bgr, parse)

    def _lama_keep(self, mask, keep, labels, clothing):
        """The pixels LaMa must leave alone: the parser's bare arm / hand
        skin when there is a parse, else the given hand discs, minus any
        clothing the parser saw."""
        if self.KEEP_PARSED_SKIN and labels is not None and labels.shape == mask.shape:
            # Keep what the parser calls bare arm / hand skin, not a
            # keypoint disc: on the turn clip the disc sat on the fur
            # cuff, so the fill kept a pink fur "ghost hand" and ate
            # the fingers past it. Eroded so the parser's soft edge
            # (fur hairs over the skin) is still filled.
            skin = np.isin(labels, SKIN_CLASSES).astype(np.uint8)
            keep = cv2.erode(skin, np.ones((5, 5), np.uint8)).astype(bool)
        if keep is not None and clothing is not None and clothing.shape == mask.shape:
            # a hand disc also covers the fur cuff: keep the hand, not the cuff
            keep = np.asarray(keep, bool) & ~clothing
        return keep

    @staticmethod
    def _prefill_from_plate(frame_bgr, mask, keep, plate, body):
        """(frame, hole, keep) with the hole pixels the background plate
        knows (outside `body`) filled from it: they leave the hole, and are
        kept out of the fill's reach."""
        if plate is None:
            return frame_bgr, mask, keep
        prefilled, rest = plate.fill(frame_bgr, mask, body)
        from_plate = mask & ~rest
        if not from_plate.any():
            return frame_bgr, mask, keep
        keep = from_plate if keep is None else (np.asarray(keep, bool) | from_plate)
        return prefilled, rest, keep

    def _fill_lama(self, frame_bgr, mask, keep, arms, skin_bgr, parse=None):
        out = self._lama(frame_bgr, mask, keep=keep)
        # LaMa fills the hole from the background, so the wearer's
        # arms vanish with their sleeves (camera 0004: the person read
        # as armless). Where the caller knows the arms (`arms`, the
        # keypoint strokes), draw them back as skin-coloured tubes.
        if arms is not None and skin_bgr is None:
            skin_bgr = self.skin_colour(frame_bgr, parse)
        if arms is None or skin_bgr is None:
            return out
        tube = np.asarray(arms, bool) & mask
        if keep is not None:
            # real skin stays real; the tube only replaces a sleeve
            tube &= ~np.asarray(keep, bool)
        if not tube.any():
            return out
        # cylinder shading: full skin colour on the axis, darker toward
        # the edge, from the distance to the tube's border
        depth = cv2.distanceTransform(tube.astype(np.uint8), cv2.DIST_L2, 5)
        radius = max(float(depth.max()), 1.0)
        shade = 0.6 + 0.4 * np.sqrt(np.clip(depth / radius, 0.0, 1.0))
        skin = np.asarray(skin_bgr, np.float32)
        out[tube] = np.clip(skin[None, :] * shade[tube][:, None], 0, 255).astype(np.uint8)
        ring = tube & ~cv2.erode(tube.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        return cv2.inpaint(out, ring.astype(np.uint8), 2, cv2.INPAINT_TELEA)

    def _fill_skin(self, frame_bgr, mask, arms, skin_bgr, parse=None):
        """Telea inpaint, blended toward the wearer's skin colour on the arms."""
        inpainted = self._telea(frame_bgr, mask)
        if skin_bgr is None:
            skin_bgr = self.skin_colour(frame_bgr, parse)
        if skin_bgr is None:
            return inpainted
        out = inpainted.copy()
        skin = np.asarray(skin_bgr, np.float32)
        # Skin colour only where skin is likely: the arms, when the caller
        # gives their strokes. Over the torso it painted a bare chest where
        # the wearer had an undershirt the parser also calls upper clothing
        # (00122_outer's white henley under an open shirt); the plain
        # inpaint there takes its colour from what surrounds the hole.
        tinted = mask & np.asarray(arms, bool) if arms is not None and self.SKIN_ON_ARMS_ONLY else mask
        out[tinted] = np.clip(self.SKIN_BLEND * skin + (1.0 - self.SKIN_BLEND) * inpainted[tinted].astype(np.float32), 0, 255).astype(np.uint8)
        # Soften the seam: re-inpaint a thin ring just inside the boundary.
        ring = tinted & ~cv2.erode(tinted.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        return self._inpaint_ring(out, ring)

    # ------------------------------------------------------------------
    # The label-aware fill (`LABEL_FILL`)
    # ------------------------------------------------------------------
    # Telea fills a hole from ALL the boundary at once, so a torso hole beside
    # a lamp, jeans and bare arm took a bit of each: radial grey and white
    # wedges, and a flat pink body. This fill reads the parser's labels: every
    # hole pixel is skin (the arm tube; or inside the body with a skin pixel
    # nearest) or "other" (nearest non-skin source: room, trousers, undershirt).
    # Skin is the colour of the visible skin of the same limb (a smooth field
    # of the nearby skin pixels, so it varies along the arm), shaded darker
    # toward the tube's edge and given the frame's own grain; "other" is the
    # nearest source colour, softened. No blending across the two.
    LABEL_FILL = True
    # skin from the visible skin of the same limb and shaded like a cylinder (True), or the one face-median colour (False)
    SKIN_MODEL = True
    LABEL_SCALE = 0.5  # the fill runs at this fraction of the frame
    LABEL_PAD = 32  # px of context around the hole's bounding box
    SKIN_LABELS = (11, 14, 15)  # ATR: face, left arm, right arm
    SKIN_FIELD_SIGMA = 10.0  # px (at LABEL_SCALE) over which the skin colour varies
    OTHER_SIGMA = 3.0  # px (at LABEL_SCALE) the nearest-source fill is softened by
    TUBE_SHADE = 0.15  # the tube's edge is this much darker than its axis
    LABEL_INNER_REACH = 40.0  # px (at LABEL_SCALE): how far an undershirt is continued into the hole
    LABEL_MIN_INNER = 150  # px (at LABEL_SCALE) of kept inner layer that count as an undershirt
    GRAIN = 1.0  # share of the neighbouring pixels' grain (std of gray - blur) the fill gets
    _rng = None

    @staticmethod
    def _nearest_source(sources):
        """(row, col) of the nearest True pixel of `sources` for every pixel."""
        distance, nearest = cv2.distanceTransformWithLabels((~sources).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(sources)
        index = np.clip(nearest - 1, 0, ys.size - 1)
        return ys[index], xs[index]

    @staticmethod
    def _smooth(image, weight, sigma):
        """Normalised convolution: `image` averaged over `weight` with a Gaussian."""
        numerator = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma)
        denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
        return numerator / np.maximum(denominator, 1e-4)[..., None], denominator

    # The parser's own bare arm / hand pixels (not clothing, 2 px inside their edge) are
    # never refilled: the 0.45-wide strokes and the 6 px dilation put them in the hole, and
    # a fill that replaced them with the nearest non-skin colour drew a pale halo on the
    # real arm (00122_outer: white undershirt).
    KEEP_REAL_SKIN = True
    # The bare-arm tube narrows along the real limb (`arm_tube`, needs the keypoints in the
    # parse); False: the constant-width stroke passed as `arms`.
    TAPER_TUBE = True
    SKIN_REACH = 3.0  # px at LABEL_SCALE, see `_fill_labels`

    # torso / neck skin from the face, not the arm (True), or the field of all visible skin (False)
    TORSO_FROM_FACE = True

    # The parser calls a white cuff or sleeve "arm" now and then (00170_upper: 2,592 arm
    # pixels, median colour white): an arm-labelled pixel counts as skin only when its
    # CIELAB chroma (a, b) is within this of the same frame's face (white, grey, plaid:
    # 25-40 away; skin in shade or sun: under 12).
    SKIN_CHROMA = 16.0

    def _real_skin_labels(self, frame_bgr, labels, mask):
        """`labels` with the arm-labelled pixels around the hole that are not skin-coloured
        (against the face) set to 255, so neither the keep nor the fill's skin sources see them."""
        face = labels == self.FACE_CLASS
        if int(face.sum()) < 50:
            return labels
        pixels = np.ascontiguousarray(frame_bgr[face][::5]).reshape(-1, 1, 3)
        reference = np.median(cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)[:, 1:].astype(np.float32), axis=0)
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        margin = self.LABEL_PAD + 2
        window = (slice(max(rows[0] - margin, 0), rows[-1] + 1 + margin), slice(max(cols[0] - margin, 0), cols[-1] + 1 + margin))
        arm = np.isin(labels[window], SKIN_CLASSES)
        if not arm.any():
            return labels
        lab = cv2.cvtColor(np.ascontiguousarray(frame_bgr[window]), cv2.COLOR_BGR2LAB).astype(np.float32)
        away = np.hypot(lab[..., 1] - reference[0], lab[..., 2] - reference[1]) >= self.SKIN_CHROMA
        out = labels.copy()
        out[window][arm & away] = 255
        return out

    def _spare_real_skin(self, mask, labels, clothing):
        """`mask` without the parser's real arm / hand skin."""
        skin = np.isin(labels, SKIN_CLASSES)
        if clothing is not None:
            skin &= ~clothing
        skin &= mask
        if not skin.any():
            return mask
        rows, cols = np.flatnonzero(skin.any(axis=1)), np.flatnonzero(skin.any(axis=0))
        window = (slice(max(rows[0] - 2, 0), rows[-1] + 3), slice(max(cols[0] - 2, 0), cols[-1] + 3))
        spare = cv2.erode(np.ascontiguousarray(skin[window]).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        out = mask.copy()
        out[window] &= ~spare
        return out

    def _fill_labels(self, frame_bgr, mask, arms, skin_bgr, labels, clothing, body, points=None):
        if self.KEEP_REAL_SKIN:
            labels = self._real_skin_labels(frame_bgr, labels, mask)
            mask = self._spare_real_skin(mask, labels, clothing)
            if not mask.any():
                return frame_bgr.copy()
        height, width = mask.shape
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        pad = self.LABEL_PAD
        y0, y1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, height)
        x0, x1 = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, width)
        window = (slice(y0, y1), slice(x0, x1))
        crop = np.ascontiguousarray(frame_bgr[window])
        hole_full = mask[window]
        scale = self.LABEL_SCALE
        size = (max(int(round(crop.shape[1] * scale)), 2), max(int(round(crop.shape[0] * scale)), 2))
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA).astype(np.float32)

        def shrink(array):
            return cv2.resize(np.ascontiguousarray(array).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

        hole = cv2.dilate(shrink(hole_full).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        labels_small = cv2.resize(np.ascontiguousarray(labels[window]), size, interpolation=cv2.INTER_NEAREST)
        covered = hole | (shrink(clothing[window]) if clothing is not None else False)
        sources = ~covered
        if not sources.any():
            return self._telea(frame_bgr, mask)
        skin_pixel = np.isin(labels_small, self.SKIN_LABELS)
        skin_source = cv2.erode((sources & skin_pixel).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        tube = shrink(arms[window]) & hole if arms is not None else np.zeros_like(hole)
        tube_radius = None
        if arms is not None and points is not None and self.TAPER_TUBE:
            tapered, half_width = arm_tube(points, mask.shape, self.ARM_TUBE_WIDTH)
            tube = shrink(tapered[window]) & hole
            tube_radius = cv2.resize(np.ascontiguousarray(half_width[window]), size, interpolation=cv2.INTER_NEAREST) * (size[0] / crop.shape[1])
        inside = shrink(body[window]) | tube if body is not None else np.ones_like(hole)

        ny, nx = self._nearest_source(sources)
        nearest_skin = skin_pixel[ny, nx]
        # an undershirt the parser kept (the inner layer: its label is clothing, but it is not in the
        # mask) is what lies under the removed coat, not skin: it is drawn from it wherever the
        # body is, and the bare skin of the neck above it stays out of it
        inner = sources & np.isin(labels_small, (4, 7))
        has_inner = int(inner.sum()) > self.LABEL_MIN_INNER
        # skin continues from the visible skin only SKIN_REACH px (half scale) into a hole pixel the
        # parser does not call clothing: the strokes and the dilation reach past a thin arm, and
        # skin there drew a skin-coloured halo round it. Under the garment (clothing) it runs on.
        if self.KEEP_REAL_SKIN:
            yy, xx = np.mgrid[: hole.shape[0], : hole.shape[1]]
            reach = (np.hypot(ny - yy, nx - xx) <= self.SKIN_REACH) | (shrink(clothing[window]) if clothing is not None else True)
        else:
            reach = True
        is_skin = hole & (tube | (inside & nearest_skin & skin_source.any() & ~has_inner & reach))
        # "other" everywhere in the hole: the nearest NON-skin source, softened
        other_sources = sources & ~skin_pixel
        if other_sources.any():
            oy, ox = self._nearest_source(other_sources)
            nearest = small.copy()
            nearest[hole] = small[oy, ox][hole]
            weight = (other_sources | hole).astype(np.float32)
            out, _ = self._smooth(nearest, weight, self.OTHER_SIGMA)
        else:
            out = small[ny, nx]
        if has_inner:
            iy, ix = self._nearest_source(inner)
            under = small.copy()
            under[hole] = small[iy, ix][hole]
            soft, _ = self._smooth(under, (inner | hole).astype(np.float32), self.OTHER_SIGMA)
            # not where the trousers or legs are the nearest thing (the thigh under a long hem)
            near_inner = cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 3) < self.LABEL_INNER_REACH
            lower = np.isin(labels_small[ny, nx], LOWER_BODY_CLASSES)
            out = np.where((hole & inside & ~tube & near_inner & ~lower)[..., None], soft, out)
        if is_skin.any():
            arm_source = skin_source & np.isin(labels_small, SKIN_CLASSES)
            face_source = skin_source & (labels_small == self.FACE_CLASS)
            # the torso and neck are the FACE's colour, not the arm's (an arm in the sun, or
            # a tan line, made 00170_upper's chest brown beside a pale face); a limb is its own visible skin
            if self.TORSO_FROM_FACE and face_source.sum() >= 20:
                body_skin = np.median(small[face_source], axis=0)
            elif self.TORSO_FROM_FACE and arm_source.any():
                body_skin = small[arm_source].mean(axis=0)
            else:
                body_skin = np.asarray(skin_bgr, np.float32) if skin_bgr is not None else small[skin_source].mean(axis=0)
            field, density = self._smooth(small, (arm_source if self.SKIN_MODEL else skin_source).astype(np.float32), self.SKIN_FIELD_SIGMA)
            confidence = np.clip(density * 4.0, 0.0, 1.0)[..., None] * float(self.SKIN_MODEL)
            limb = confidence * field + (1.0 - confidence) * body_skin
            depth = cv2.distanceTransform(tube.astype(np.uint8), cv2.DIST_L2, 3)
            local = np.maximum(tube_radius, 1.0) if tube_radius is not None else max(float(depth.max()), 1.0)
            shade = (1.0 - self.TUBE_SHADE * float(self.SKIN_MODEL) * (1.0 - np.sqrt(np.clip(depth / local, 0.0, 1.0))))[..., None]
            colour = np.where(tube[..., None], limb * shade, body_skin[None, None, :] if self.SKIN_MODEL else limb)
            # a soft edge between the two kinds (the known skin counts as skin, so the
            # fill meets it at full strength)
            alpha = cv2.GaussianBlur((is_skin | (sources & skin_pixel)).astype(np.float32), (0, 0), 1.5)[..., None]
            out = np.where(hole[..., None], alpha * colour + (1.0 - alpha) * out, small)
        out = np.where(hole[..., None], out, small)
        filled = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.GRAIN > 0:
            filled = filled + self._grain(crop, cv2.resize(skin_source.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)) * self.GRAIN
        result = frame_bgr.copy()
        result[window][hole_full] = np.clip(filled[hole_full], 0, 255).astype(np.uint8)
        # seam: a thin ring just inside the hole's edge is re-inpainted from both sides
        ring = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        return self._inpaint_ring(result, ring)

    def _grain(self, crop, skin):
        """Luminance noise at the level of the visible skin's own grain (std of gray - its blur over the
        `skin` pixels), the same value on all channels; none when no skin is visible."""
        if skin.sum() < 200:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
        level = min(float(detail[skin].std()), 6.0)
        if ClothingRemover._rng is None:
            ClothingRemover._rng = np.random.default_rng(0)
        noise = ClothingRemover._rng.standard_normal(gray.shape).astype(np.float32)
        return (cv2.GaussianBlur(noise, (0, 0), 0.7) * (level * 1.6))[..., None]

    SEAM_RADIUS = 3
    # Telea reads `SEAM_RADIUS` px around a pixel: a crop this far past the
    # ring holds everything the full-frame fill would read.
    SEAM_PAD = 8

    def _inpaint_ring(self, image, ring):
        """`cv2.inpaint` of the seam ring, run on the ring's bounding box
        plus `SEAM_PAD` px (identical output, the fill is local; a thin ring
        is a fraction of the frame)."""
        rows, cols = np.flatnonzero(ring.any(axis=1)), np.flatnonzero(ring.any(axis=0))
        if rows.size == 0:
            return image
        height, width = ring.shape
        y0, y1 = max(rows[0] - self.SEAM_PAD, 0), min(rows[-1] + 1 + self.SEAM_PAD, height)
        x0, x1 = max(cols[0] - self.SEAM_PAD, 0), min(cols[-1] + 1 + self.SEAM_PAD, width)
        out = image.copy()
        out[y0:y1, x0:x1] = cv2.inpaint(np.ascontiguousarray(image[y0:y1, x0:x1]), np.ascontiguousarray(ring[y0:y1, x0:x1]).astype(np.uint8), self.SEAM_RADIUS, cv2.INPAINT_TELEA)
        return out
