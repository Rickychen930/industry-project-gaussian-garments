"""Put the rendered garment over the camera frame.

    composite_garment   alpha-blend the garment (BGRA) over the frame
    keep_in_front       restore the wearer's face, hair and hands on top
    restore_inside_garment the wearer's own pixels back in the garment's gaps (lace, open knit)
    restore_collar_layer the wearer's own collar left above the neckline
    restore_below_hem   the wearer's own shirt back under a shorter garment's hem
    repair_collar       blend the wearer's own collar left below the chin into the neck
    draw_body           draw the fitted skeleton (no garment: debugging)
"""

from __future__ import annotations

import cv2
import numpy as np
from numba import njit

from tryon.body.landmarks import JOINT_MAP


def draw_body(frame, fitter, colour=(60, 220, 60)):
    """The fitted body's joints and torso, projected into the frame."""
    import torch

    with torch.no_grad():
        joints = fitter._forward().joints[0, fitter._smplx_indices]
        pixels = fitter._project(joints).numpy()
    for x, y in pixels:
        cv2.circle(frame, (int(x), int(y)), 4, colour, -1)
    order = sorted(JOINT_MAP)
    by_index = {index: pixels[k] for k, index in enumerate(order)}
    bones = (
        (11, 12), (11, 23), (12, 24), (23, 24),  # torso box
        (11, 13), (13, 15), (12, 14), (14, 16),  # arms
        (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    )
    for a, b in bones:
        if a in by_index and b in by_index:
            cv2.line(frame, tuple(by_index[a].astype(int)), tuple(by_index[b].astype(int)), colour, 2)


# ATR labels that always stay in front of the garment: hat, hair, sunglasses, face.
# The fitted body has no face landmarks, so its head is not a trustworthy occluder.
IN_FRONT_CLASSES = (1, 2, 3, 11)

COLLAR_CHIN = 1.3  # the chin is this many nose-to-mouth distances below the mouth
# The chin line comes from the nose and mouth landmarks and is nonsense for a bowed or
# turned head or collapsed shoulders (normal: mouth 0.2-0.55 shoulder widths above them).
COLLAR_MOUTH_ABOVE = (0.12, 1.0)
COLLAR_NOSE_MOUTH = (0.04, 0.2)
COLLAR_MIN_SHOULDERS = 40.0  # px


def collar_geometry_ok(points):
    """Whether the face / shoulder landmarks of `points` are a head above shoulders that the collar functions can use."""
    shoulders = float(np.linalg.norm(points[11, :2] - points[12, :2]))
    if shoulders <= 1.0:
        return True  # no shoulders given (a caller that only has the face): nothing to judge by
    if shoulders < COLLAR_MIN_SHOULDERS:
        return False
    mouth_y = 0.5 * float(points[9, 1] + points[10, 1])
    above = (0.5 * float(points[11, 1] + points[12, 1]) - mouth_y) / shoulders
    gap = (mouth_y - float(points[0, 1])) / shoulders
    return COLLAR_MOUTH_ABOVE[0] <= above <= COLLAR_MOUTH_ABOVE[1] and COLLAR_NOSE_MOUTH[0] <= gap <= COLLAR_NOSE_MOUTH[1]
COLLAR_DISTANCE = 12.0  # Lab distance from the face's skin that marks a pixel as collar


def _in_classes(labels, classes):
    """`np.isin(labels, classes)` via a 256-entry lookup for 8-bit labels (faster, same mask)."""
    if labels.dtype != np.uint8:
        return np.isin(labels, classes)
    table = np.zeros(256, bool)
    table[[c for c in classes if 0 <= c < 256]] = True
    return table[labels]


def collar_pixels(original, labels, points):
    """The face-labelled pixels below the chin whose colour is far from the face's skin: the
    wearer's own collar stripes, which SCHP calls "face" and `keep_in_front` must not
    restore over the new garment. A boolean mask of the frame, or None."""
    nose, mouth = points[0], 0.5 * (points[9] + points[10])
    if min(points[0, 2], points[9, 2], points[10, 2]) < 0.3 or not np.isfinite(mouth[:2]).all():
        return None
    gap = float(mouth[1] - nose[1])
    nose_row, mouth_row = int(nose[1]), int(mouth[1])
    chin = int(mouth[1] + COLLAR_CHIN * gap)
    if gap < 2 or mouth_row <= nose_row or chin >= original.shape[0] - 1 or nose_row < 0:
        return None
    below = labels[chin + 1 :] == 11
    rows = np.flatnonzero(below.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(below.any(axis=0))
    window = (slice(chin + 1 + int(rows[0]), chin + 1 + int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
    face = labels[nose_row:mouth_row] == 11
    if face.sum() < 50:
        return None
    skin = np.median(cv2.cvtColor(original[nose_row:mouth_row][face].reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)
    lab = cv2.cvtColor(np.ascontiguousarray(original[window]), cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = np.array([0.5, 1.0, 1.0], np.float32)
    far = np.sqrt((((lab - skin) * weights) ** 2).sum(axis=2)) > COLLAR_DISTANCE
    collar = np.zeros(labels.shape, bool)
    below_window = np.zeros(labels.shape, bool)
    below_window[chin + 1 :] = labels[chin + 1 :] == 11
    collar[window] = below_window[window] & far
    return collar if collar.any() else None


def keep_in_front(frame, original, labels, points=None):
    """Restores `original` where the parser saw the wearer's face or hair.

    "Face" only within a disc around MediaPipe's head landmarks (radius 0.6 shoulder widths)
    when given: the parser has called a fur collar "face"."""
    if labels is None or labels.shape != frame.shape[:2]:
        return
    front = _in_classes(labels, IN_FRONT_CLASSES)
    if points is not None:
        head = [i for i in (0, 2, 5, 7, 8) if points[i, 2] >= 0.3]
        if head and points[11, 2] >= 0.3 and points[12, 2] >= 0.3:
            radius = 0.6 * float(np.linalg.norm(points[11, :2] - points[12, :2]))
            centre = points[head, :2].mean(axis=0)
            # front &= near | (labels != 11), the disc worked out only in its own bounding box
            height, width = frame.shape[:2]
            if np.isfinite(radius) and np.isfinite(centre).all():
                r0, r1 = max(int(centre[1] - radius) - 1, 0), min(int(centre[1] + radius) + 3, height)
                c0, c1 = max(int(centre[0] - radius) - 1, 0), min(int(centre[0] + radius) + 3, width)
            else:
                r0, r1, c0, c1 = 0, height, 0, width
            face_outside = labels == 11
            if r1 > r0 and c1 > c0:
                rows, cols = np.ogrid[r0:r1, c0:c1]
                near = (rows - centre[1]) ** 2 + (cols - centre[0]) ** 2 <= radius**2
                face_outside[r0:r1, c0:c1] &= ~near
            front &= ~face_outside
        # Hands: the parser's arm skin (ATR 14 / 15) within a disc of a quarter shoulder
        # width at each extended wrist; a hand in the photo is in front of the clothing.
        from tryon.removal.remover import arm_strokes, hand_discs

        # An arm raised across the head is in front of it: nothing inside the arm strokes
        # is restored as face or hair.
        front &= ~arm_strokes(points, frame.shape)
        collar = collar_pixels(original, labels, points)
        if collar is not None:
            front &= ~collar
        hands = hand_discs(points, frame.shape, hand_fraction=0.25) & _in_classes(labels, (14, 15))
        front |= hands
    # only inside the restored pixels' bounding box
    rows = np.flatnonzero(front.any(axis=1))
    if rows.size == 0:
        return
    cols = np.flatnonzero(front.any(axis=0))
    window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    np.copyto(frame[window], original[window], where=front[window][..., None])


COLLAR_GROW = 4  # px the marked pixels grow by before they are repainted
COLLAR_RADIUS = 6  # inpainting radius


def repair_collar(frame, garment_alpha, labels, points):
    """Blend the wearer's own collar into the neck, in place, after compositing. SCHP labels
    the band of the shirt under the chin "face", so it is not erased and `keep_in_front`
    restores it above the new neckline. The face-labelled pixels below the chin that no
    garment covers and whose Lab colour is far from the skin are inpainted from the skin
    above and the garment below."""
    if labels is None or points is None or labels.shape != frame.shape[:2]:
        return
    nose, mouth = points[0], 0.5 * (points[9] + points[10])
    if min(points[0, 2], points[9, 2], points[10, 2]) < 0.3 or not np.isfinite(mouth[:2]).all() or not collar_geometry_ok(points):
        return
    gap = float(mouth[1] - nose[1])
    chin, mouth_row, nose_row = int(mouth[1] + COLLAR_CHIN * gap), int(mouth[1]), int(nose[1])
    if gap < 2 or chin >= frame.shape[0] - 1 or mouth_row <= nose_row:
        return
    below = labels[chin + 1 :] == 11
    rows = np.flatnonzero(below.any(axis=1))
    if rows.size == 0:
        return
    cols = np.flatnonzero(below.any(axis=0))
    grow = COLLAR_GROW + COLLAR_RADIUS
    r0, r1 = chin + 1 + int(rows[0]), chin + 1 + int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    window = (slice(max(r0 - grow, 0), min(r1 + grow, frame.shape[0])), slice(max(c0 - grow, 0), min(c1 + grow, frame.shape[1])))
    face_rows = labels[nose_row:mouth_row] == 11
    if face_rows.sum() < 50:
        return
    skin_bgr = frame[nose_row:mouth_row][face_rows].reshape(-1, 1, 3)
    skin = np.median(cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)
    window_labels = labels[window]
    neck = window_labels == 11
    neck[: max(chin + 1 - window[0].start, 0)] = False  # nothing above the chin line
    lab = cv2.cvtColor(frame[window], cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = np.array([0.5, 1.0, 1.0], np.float32)  # shade on the neck is lightness: weigh it less
    far = np.sqrt((((lab - skin) * weights) ** 2).sum(axis=2)) > COLLAR_DISTANCE
    open_air = garment_alpha[window] < 25
    marked = (neck & far & open_air).astype(np.uint8)
    if not marked.any():
        return
    # grow and inpainting reach only COLLAR_GROW / COLLAR_RADIUS px from a marked pixel: work in that box
    x, y, w, h = cv2.boundingRect(marked)
    reach = COLLAR_GROW + COLLAR_RADIUS + 1
    y0, y1 = max(y - reach, 0), min(y + h + reach, marked.shape[0])
    x0, x1 = max(x - reach, 0), min(x + w + reach, marked.shape[1])
    box = (slice(y0, y1), slice(x0, x1))
    grown = cv2.dilate(marked[box], np.ones((2 * COLLAR_GROW + 1,) * 2, np.uint8)) & open_air[box] & (window_labels[box] == 11)
    patch = frame[window][box]
    patch[:] = cv2.inpaint(np.ascontiguousarray(patch), grown * 255, COLLAR_RADIUS, cv2.INPAINT_TELEA)


INSIDE_CLOSE = 21  # px: gaps up to this wide between garment pixels count as inside the garment
INSIDE_MAX_SHARE = 0.25  # an enclosed hole bigger than this share of the garment is not a lace gap (a loop made by an arm)
INSIDE_HEAL_SHARE = 0.03  # gaps up to this share of the garment are painted over from the garment around them (0: never): black specks in a lace sleeve


def restore_inside_garment(frame, original, garment_alpha):
    """The wearer's own pixels back wherever the garment has a gap inside its outline, in place
    (lace, open knit, a collar opening), instead of the room's learned background. Gaps are
    the pixels with no garment inside the outline's closing (`INSIDE_CLOSE` px) whose hole
    does not reach the image border, each at most INSIDE_MAX_SHARE of the garment's area."""
    import cv2

    covered = (garment_alpha >= 25).astype(np.uint8)
    area = int(covered.sum())
    if area == 0:
        return
    x, y, w, h = cv2.boundingRect(covered)
    pad = INSIDE_CLOSE + 2
    y0, y1 = max(y - pad, 0), min(y + h + pad, covered.shape[0])
    x0, x1 = max(x - pad, 0), min(x + w + pad, covered.shape[1])
    window = (slice(y0, y1), slice(x0, x1))
    solid = covered[window]
    closed = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((INSIDE_CLOSE, INSIDE_CLOSE), np.uint8))
    # holes of `closed` = what flood filling the outside leaves unreached
    flood = np.pad((closed == 0).astype(np.uint8), 1, constant_values=1)
    mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    inside = (flood[1:-1, 1:-1] != 2) | (closed > 0)
    gaps = inside & (solid == 0)
    if not gaps.any():
        return
    # a closed region of gaps this big is something else (two arms making a loop)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gaps.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= INSIDE_MAX_SHARE * area
    gaps = keep[labels]
    part = frame[window]
    if INSIDE_HEAL_SHARE > 0:
        small = np.zeros(count, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] <= INSIDE_HEAL_SHARE * area
        heal = small[labels] & gaps
        if heal.any():
            mask = cv2.dilate(heal.astype(np.uint8), np.ones((3, 3), np.uint8))
            part[:] = cv2.inpaint(np.ascontiguousarray(part), mask * 255, 3, cv2.INPAINT_TELEA)
            gaps = gaps & ~heal
    changed = (part != original[window]).any(axis=2)
    restore = gaps & changed
    if restore.any():
        part[restore] = original[window][restore]


LAYER_SPAN = 0.15  # shoulder widths either side of the neck the collar layer reaches
LAYER_DEPTH = 0.45  # shoulder widths below the chin line it reaches
LAYER_FEATHER = 14  # px over which the collar fades out at either end of its columns


def restore_collar_layer(frame, original, garment_alpha, labels, points):
    """The wearer's own collar left visible in the new neckline, in place: the pixels of the
    neck window the erase changed, that no garment covers and that are not hair, go back as
    read (a crew neck under a sweater reads as a layer). The window: below the chin line
    (mouth + `COLLAR_CHIN` nose-to-mouth distances) down `LAYER_DEPTH` shoulder widths, in the
    neck columns (face-labelled pixels below the chin, `LAYER_SPAN` shoulder widths either
    side), fading over `LAYER_FEATHER` px at both ends. Not tied to the garment's top edge: a
    thick collar close to the camera encloses the opening from three sides."""
    if labels is None or points is None or labels.shape != frame.shape[:2]:
        return
    nose, mouth = points[0], 0.5 * (points[9] + points[10])
    if min(points[0, 2], points[9, 2], points[10, 2], points[11, 2], points[12, 2]) < 0.3 or not np.isfinite(mouth[:2]).all() or not collar_geometry_ok(points):
        return
    gap = float(mouth[1] - nose[1])
    chin = int(mouth[1] + COLLAR_CHIN * gap)
    shoulder_width = float(np.linalg.norm(points[11, :2] - points[12, :2]))
    if gap < 2 or chin >= frame.shape[0] - 1 or shoulder_width < 8:
        return
    r0, r1 = chin + 1, min(chin + 1 + int(LAYER_DEPTH * shoulder_width), frame.shape[0])
    below = labels[r0:] == 11
    cols = np.flatnonzero(below.any(axis=0))
    if cols.size == 0:
        return
    reach = int(LAYER_SPAN * shoulder_width)
    c0, c1 = max(int(cols[0]) - reach, 0), min(int(cols[-1]) + 1 + reach, frame.shape[1])
    window = (slice(r0, r1), slice(c0, c1))
    band = (garment_alpha[window] < 25) & (labels[window] != 2)  # hair stays
    band &= (frame[window] != original[window]).any(axis=2)
    if not band.any():
        return
    across = np.arange(c1 - c0)
    ramp = np.clip(np.minimum(across + 1, c1 - c0 - across) / float(LAYER_FEATHER), 0.0, 1.0)
    weight = band * ramp[None, :]
    part = frame[window]
    blended = part.astype(np.float32) * (1.0 - weight[..., None]) + original[window].astype(np.float32) * weight[..., None]
    part[band] = (blended[band] + 0.5).astype(np.uint8)


@njit(cache=True)
def _restore_below(frame, original, alpha, x0, x1, margin):
    """`restore_below_hem`'s loop: per column, from the row under the garment's last covered
    pixel down, copy `original` over the pixels the erase changed. One pass, no temporaries."""
    rows = alpha.shape[0]
    for x in range(x0, x1):
        bottom = -1
        for r in range(rows - 1, -1, -1):
            if alpha[r, x] >= 25:
                bottom = r
                break
        if bottom < 0:
            continue
        for r in range(bottom + margin + 1, rows):
            if frame[r, x, 0] != original[r, x, 0] or frame[r, x, 1] != original[r, x, 1] or frame[r, x, 2] != original[r, x, 2]:
                frame[r, x, 0] = original[r, x, 0]
                frame[r, x, 1] = original[r, x, 1]
                frame[r, x, 2] = original[r, x, 2]


def restore_below_hem(frame, original, garment_alpha, margin=0):
    """Put the wearer's own pixels back below the garment's lower edge, in place: a shirt
    longer than the garment reads as a layer instead of a filled background. Only the
    columns the garment covers, below its lowest covered row, where the erase changed pixels."""
    columns = np.flatnonzero((garment_alpha >= 25).any(axis=0))
    if columns.size == 0:
        return
    _restore_below(frame, original, garment_alpha, int(columns[0]), int(columns[-1]) + 1, int(margin))


@njit(cache=True)
def _blend(frame, bgra, x0, y0, x1, y1):
    """Alpha-blend `bgra` over `frame` inside the box, float32 as the numpy form (a / 255 first,
    then g a + f (1 - a), truncated to 8 bits): same bytes, without the float32 copies."""
    one, full = np.float32(1.0), np.float32(255.0)
    for y in range(y0, y1):
        for x in range(x0, x1):
            a = bgra[y, x, 3]
            if a == 0:
                continue
            alpha = np.float32(a) / full
            for c in range(3):
                frame[y, x, c] = np.uint8(np.float32(bgra[y, x, c]) * alpha + np.float32(frame[y, x, c]) * (one - alpha))


def composite_garment(frame, bgra):
    """Alpha-blend a rendered BGRA garment over a BGR frame, in place.

    `GarmentRenderer.render` returns BGR colours (tryon/rendering/splat_cpu.py) plus alpha;
    they must not be treated as RGB."""
    # only inside the garment's bounding box: outside it alpha is 0
    x, y, w, h = cv2.boundingRect(bgra[..., 3])
    if w == 0 or h == 0:
        return
    _blend(frame, bgra, x, y, x + w, y + h)
