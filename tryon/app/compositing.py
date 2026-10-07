"""Put the rendered garment over the camera frame.

    composite_garment   alpha-blend the garment (BGRA) over the frame
    keep_in_front       restore the wearer's face, hair and hands on top
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


def _in_classes(labels, classes):
    """`np.isin(labels, classes)`; a 256-entry lookup for 8-bit labels
    (1.4 ms against 11.7 at 1280 x 938, same mask)."""
    if labels.dtype != np.uint8:
        return np.isin(labels, classes)
    table = np.zeros(256, bool)
    table[[c for c in classes if 0 <= c < 256]] = True
    return table[labels]


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
