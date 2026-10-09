"""Put the rendered garment over the camera frame.

    composite_garment   alpha-blend the garment (BGRA) over the frame
    keep_in_front       restore the wearer's face, hair and hands on top
    repair_collar       blend the wearer's own collar left below the chin into the neck
    draw_body           draw the fitted skeleton (no garment: debugging)
"""

from __future__ import annotations

import cv2
import numpy as np

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


# ATR labels of the wearer that always stay in front of the rendered
# garment: hat, hair, sunglasses, face. The fitted body has no face
# landmarks, so its head is not a trustworthy occluder: on 00176_outer a
# dancer's bowed head put the collar across her chin; and long hair over
# a shirt collar is the physical order anyway.
IN_FRONT_CLASSES = (1, 2, 3, 11)

COLLAR_CHIN = 1.3  # the chin is this many nose-to-mouth distances below the mouth
COLLAR_DISTANCE = 12.0  # Lab distance from the face's skin that marks a pixel as collar


def _in_classes(labels, classes):
    """`np.isin(labels, classes)`; a 256-entry lookup for 8-bit labels
    (1.4 ms against 11.7 at 1280 x 938, same mask)."""
    if labels.dtype != np.uint8:
        return np.isin(labels, classes)
    table = np.zeros(256, bool)
    table[[c for c in classes if 0 <= c < 256]] = True
    return table[labels]


def collar_pixels(original, labels, points):
    """The face-labelled pixels below the chin whose colour is far from the
    face's skin: the wearer's own collar stripes, which SCHP calls "face".
    `keep_in_front` must not restore them over the new garment: the disc it
    restores a face in reaches below the chin for a close camera, and they
    came back as hard-edged pale shards in the neckline (a recorded session,
    9 Oct). A boolean mask of the frame, or None (no face found, or none)."""
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

    "Face" only within a disc around MediaPipe's head landmarks (nose,
    eyes, ears; radius 0.6 shoulder widths) when they are given: the
    parser has called a fur collar "face" (docs/LOG.md, clothing removal), and
    that must not be painted back over the new garment."""
    if labels is None or labels.shape != frame.shape[:2]:
        return
    front = _in_classes(labels, IN_FRONT_CLASSES)
    if points is not None:
        head = [i for i in (0, 2, 5, 7, 8) if points[i, 2] >= 0.3]
        if head and points[11, 2] >= 0.3 and points[12, 2] >= 0.3:
            radius = 0.6 * float(np.linalg.norm(points[11, :2] - points[12, :2]))
            centre = points[head, :2].mean(axis=0)
            # front &= near | (labels != 11), with the disc worked out only
            # in its own bounding box (outside it `near` is false)
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
        # Hands: the parser's arm skin (ATR 14 / 15) within a disc of a
        # quarter shoulder width at each extended wrist. A sleeve longer than the wearer's arm
        # otherwise ran over the hand (00122_outer, 10004_outer); a hand
        # seen in the photo is in front of the clothing, and a real sleeve
        # would bunch at the wrist, not cover it.
        from tryon.removal.remover import arm_strokes, hand_discs

        # An arm raised across the head is in front of it: the parser
        # called 00122_outer's plaid sleeve over the face "hair", and it
        # was painted back over the new garment. Nothing inside the arm
        # strokes is restored as face or hair.
        front &= ~arm_strokes(points, frame.shape)
        collar = collar_pixels(original, labels, points)
        if collar is not None:
            front &= ~collar
        hands = hand_discs(points, frame.shape, hand_fraction=0.25) & _in_classes(labels, (14, 15))
        front |= hands
    # only inside the restored pixels' bounding box (they are a face and
    # two hands, a small part of the frame)
    rows = np.flatnonzero(front.any(axis=1))
    if rows.size == 0:
        return
    cols = np.flatnonzero(front.any(axis=0))
    window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    np.copyto(frame[window], original[window], where=front[window][..., None])


COLLAR_GROW = 4  # px the marked pixels grow by before they are repainted
COLLAR_RADIUS = 6  # inpainting radius


def repair_collar(frame, garment_alpha, labels, points):
    """Blend the wearer's own collar into the neck, in place, after the
    garment is composited. SCHP labels the dark stripe or band of the shirt
    under the chin "face", so it is not erased, and `keep_in_front` restores
    it; it then showed above the new neckline (a recorded session, 9 Oct).
    The face-labelled pixels below the chin that no garment covers and whose
    Lab colour is far from the face's skin are inpainted from the skin above
    and the garment below (filling them with a flat colour, or erasing them
    with the general fill, left blocks that read worse than the line)."""
    if labels is None or points is None or labels.shape != frame.shape[:2]:
        return
    nose, mouth = points[0], 0.5 * (points[9] + points[10])
    if min(points[0, 2], points[9, 2], points[10, 2]) < 0.3 or not np.isfinite(mouth[:2]).all():
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
    neck = np.zeros(frame.shape[:2], bool)
    neck[chin + 1 :] = below
    lab = cv2.cvtColor(frame[window], cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = np.array([0.5, 1.0, 1.0], np.float32)  # shade on the neck is lightness: weigh it less
    far = np.sqrt((((lab - skin) * weights) ** 2).sum(axis=2)) > COLLAR_DISTANCE
    marked = (neck[window] & far & (garment_alpha[window] < 25)).astype(np.uint8)
    if not marked.any():
        return
    marked = cv2.dilate(marked, np.ones((2 * COLLAR_GROW + 1,) * 2, np.uint8)) & (garment_alpha[window] < 25) & (labels[window] == 11)
    frame[window] = cv2.inpaint(np.ascontiguousarray(frame[window]), marked * 255, COLLAR_RADIUS, cv2.INPAINT_TELEA)


def composite_garment(frame, bgra):
    """Alpha-blend a rendered BGRA garment over a BGR frame, in place.

    `GarmentRenderer.render` returns the rasteriser's BGR (tryon/rendering/splat_cpu.py
    stores colours BGR "as everywhere here") plus alpha. Until 25 Sep 2026
    this reversed the channels as if they were RGB, which painted the red
    plaid 00152_outer BLUE in every composite - found by comparing with
    the checkpoint's own ground-truth video, not by reading the code."""
    # only inside the garment's bounding box: outside it alpha is 0 and the
    # blend returns the frame unchanged (8.5 -> ~2 ms a frame)
    x, y, w, h = cv2.boundingRect(bgra[..., 3])
    if w == 0 or h == 0:
        return
    region = (slice(y, y + h), slice(x, x + w))
    alpha = bgra[region + (slice(3, 4),)].astype(np.float32) / 255.0
    garment_bgr = bgra[region + (slice(0, 3),)].astype(np.float32)
    frame[region] = (garment_bgr * alpha + frame[region].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
