"""A per-pixel background plate learned from the live feed.

Clothing removal has to invent what is behind the wearer's garment. Inside
the body that is skin or an undershirt, which only an inpainter can guess.
Outside the body - the part of a loose jacket that sticks out past the
torso and arms - it is the room, and a fixed camera SEES the room there
whenever the wearer moves. LaMa guessed that area from the garment's own
fringe on the four-camera turn clip and painted a fur-coloured blob beside
the hip; the plate fills it with the pixels actually observed.

The plate is a running mean over frames where the pixel is not the person
(the parser's person mask, grown by `margin` px so the hair and fur halo
never enter it), with a per-pixel count of how often it was seen. A scene
cut - most of the known background changing at once - clears it. Pixels
seen fewer than `min_count` times are unknown and left to the inpainter.
"""

from __future__ import annotations

import cv2
import numpy as np


class BackgroundPlate:
    def __init__(self, rate=0.1, margin=15, min_count=3, cut_fraction=0.5, cut_threshold=40.0):
        self.rate = float(rate)
        self.margin = int(margin)
        self.min_count = int(min_count)
        self.cut_fraction = float(cut_fraction)
        self.cut_threshold = float(cut_threshold)
        self.reset()

    def reset(self):
        self.plate = None
        self.count = None

    def update(self, frame_bgr, person_mask):
        """Learns from the pixels outside the (grown) person mask."""
        frame = frame_bgr.astype(np.float32)
        if self.plate is None or self.plate.shape != frame.shape:
            self.plate = frame.copy()
            self.count = np.zeros(frame.shape[:2], np.int32)
        grown = cv2.dilate(np.asarray(person_mask, np.uint8), np.ones((2 * self.margin + 1,) * 2, np.uint8)).astype(bool)
        seen = ~grown
        # Whole-array operations, not boolean indexing of the 720p frame:
        # 102 -> ~15 ms a frame, same numbers.
        known = seen & (self.count >= self.min_count)
        known_count = int(np.count_nonzero(known))
        if known_count > 1000:
            # "any channel differs by more than the threshold", by cv2:
            # numpy's max over the last axis of an (H, W, 3) array took
            # 7 ms, and thresholding first then OR-ing three uint8 planes
            # beats splitting three float planes (same mask, 2.5 -> 1.9 ms)
            exceeds = cv2.compare(cv2.absdiff(frame, self.plate), self.cut_threshold, cv2.CMP_GT)
            b, g, r = cv2.split(exceeds)
            changed = cv2.bitwise_or(cv2.bitwise_or(b, g), r) > 0
            if np.count_nonzero(changed & known) > self.cut_fraction * known_count:
                # a different scene (camera moved or cut): start over
                self.plate = frame.copy()
                self.count = np.zeros(frame.shape[:2], np.int32)
        first = seen & (self.count == 0)
        later = (seen & ~first).astype(np.uint8)
        cv2.accumulateWeighted(frame, self.plate, self.rate, mask=later)
        if first.any():  # only while pixels are seen for the first time
            np.copyto(self.plate, frame, where=first[..., None])
        self.count += seen

    def known(self):
        if self.count is None:
            return None
        return self.count >= self.min_count

    def fill(self, frame_bgr, hole, body=None):
        """(filled frame, remaining hole): hole pixels the plate knows and
        that are not the body (`body`, a mask of where the wearer's own
        body is) take the plate's colour; the rest is left for the
        inpainter."""
        hole = np.asarray(hole, bool)
        known = self.known()
        if known is None or known.shape != hole.shape:
            return frame_bgr, hole
        use = hole & known
        if body is not None:
            use &= ~np.asarray(body, bool)
        if not use.any():
            return frame_bgr, hole
        out = frame_bgr.copy()
        out[use] = np.clip(self.plate[use], 0, 255).astype(np.uint8)
        return out, hole & ~use


# Rasterise the body on the GPU (`tryon.rendering.splat_gl.fill_triangles`) when GL
# is available; the CPU fill is the fallback.
USE_GL = True


def body_fill(fitter, shape):
    """The fitted body's projected triangles, filled (uint8), before any
    growing; `body_silhouette` grows it. Pass it to several
    `body_silhouette` calls on one frame: the fill is the cost."""
    pixels = fitter.projected_vertices()
    faces = np.asarray(fitter.faces)
    if USE_GL:
        try:
            from tryon.rendering.splat_gl import fill_triangles

            return fill_triangles(pixels, faces, shape)
        except Exception:  # no moderngl / GL context: the CPU fill
            pass
    mask = np.zeros(shape[:2], np.uint8)
    triangles = np.round(pixels[faces]).astype(np.int32)
    cv2.fillPoly(mask, list(triangles), 1)
    return mask


def body_silhouette(fitter, shape, grow=0.02, fill=None):
    """Filled mask of the fitted body's projection, grown by `grow` of the
    frame height (the fit is ~1% of height off; see docs/LOG.md).
    `fill`: this frame's `body_fill`, to skip filling it again."""
    mask = body_fill(fitter, shape) if fill is None else fill
    radius = max(int(round(grow * shape[0])), 1)
    return _grown(mask, radius)


def _grown(mask, radius):
    """`mask` (uint8) dilated by a (2r+1)-square, as a bool array - worked
    out only in the mask's bounding box plus `radius`: outside it nothing
    is set, so the result is the full-frame dilation's (the 151 px kernel
    of the 0.08 grow is 7 ms on the whole frame, ~2.5 ms around a body)."""
    x, y, w, h = cv2.boundingRect(mask)
    grown = np.zeros(mask.shape[:2], bool)
    if w == 0 or h == 0:
        return grown
    y0, y1 = max(y - radius, 0), min(y + h + radius, mask.shape[0])
    x0, x1 = max(x - radius, 0), min(x + w + radius, mask.shape[1])
    grown[y0:y1, x0:x1] = cv2.dilate(mask[y0:y1, x0:x1], np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool)
    return grown
