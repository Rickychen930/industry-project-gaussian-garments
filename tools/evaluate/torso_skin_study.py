"""Is the erased torso the colour of the wearer's face?

On the real erase masks (the fill_quality cache), the chest pixels the fill painted as bare skin
(the mask inside the fitted body, above the hip line, off the arm strokes, whose filled colour has a
CIELAB chroma within `NEAR` and a lightness within `NEAR_L` of the same frame's face) against the face's median colour: mean dE76 of the
two colours and of the lightness. An undershirt or a trouser continued into the hole is not skin and does not count.

    .venv/bin/python -m tools.evaluate.torso_skin_study 00170_upper 00176_upper
"""

from __future__ import annotations

import sys

import numpy as np

from tools.evaluate import fill_quality as fq
from tools.evaluate.fit_audit import lab_of
from tryon.removal.parsing_schp import ATR_CHECKPOINT
from tryon.removal.remover import ClothingRemover, arm_strokes

NEAR = 25.0
NEAR_L = 45.0  # a black skirt or hair has a skin-like chroma too: lightness within this of the face


def study(video, variants=("prev", "new")):
    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.KEEP_INNER_LAYER = False
    totals = {v: [] for v in variants}
    for sample in fq.load(video):
        p, shape, labels, mask = sample["points"], sample["image"].shape, sample["labels"], sample["mask"]
        face = fq._eroded(labels == 11, 5)
        hip = float(p[[23, 24], 1].mean()) if min(p[23, 2], p[24, 2]) > 0.5 else shape[0] * 0.6
        chest = fq._eroded(mask & sample["silhouette"] & ~arm_strokes(p, shape) & (np.arange(shape[0])[:, None] < hip), 3)
        if face.sum() < fq.MIN_SKIN or chest.sum() < 100:
            continue
        face_lab = lab_of(np.median(sample["image"][face], axis=0)[None].astype(np.uint8))[0]
        for variant in variants:
            out = fq.erased_crop(remover, sample, variant)
            lab = lab_of(out[chest].astype(np.uint8))
            skin = (np.hypot(lab[:, 1] - face_lab[1], lab[:, 2] - face_lab[2]) < NEAR) & (np.abs(lab[:, 0] - face_lab[0]) < NEAR_L)
            if skin.sum() < 50:
                continue
            totals[variant].append((int(skin.sum()), float(np.linalg.norm(lab[skin].mean(axis=0) - face_lab)), float(lab[skin, 0].mean() - face_lab[0]), skin.sum() / chest.sum()))
    return totals


def main():
    for video in sys.argv[1:]:
        for variant, rows in study(video).items():
            if rows:
                w = np.array([r[0] for r in rows], float)
                print(f"{video} {variant:5s} frames {len(rows)} skin px {int(w.sum())} dE76 to face {np.average([r[1] for r in rows], weights=w):.1f} dL {np.average([r[2] for r in rows], weights=w):+.1f} share {np.mean([r[3] for r in rows]):.2f}")
            else:
                print(video, variant, "no skin-filled chest")


if __name__ == "__main__":
    main()
