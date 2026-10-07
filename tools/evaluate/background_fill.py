"""How well the removed region OUTSIDE the body is filled, with and without the background plate.

On the release videos' "Ground Truth" panels (a fixed camera, a person
moving), every pixel the person uncovers at some point has a known true
background: its median over the frames where the parser calls it
background. For evaluation frames spread over the clip, the wearer's
clothing is removed as the live path does, and the error against that
true background is measured on the removed pixels outside the fitted body
(`background.body_silhouette`) - the only pixels where "the room" is the
right answer. Two fills are compared:

  fill    the fill alone (`--fill`: skin, the live default - Telea toward
          the skin colour - or lama) on the whole hole
  plate   `BackgroundPlate` learned CAUSALLY from the frames before, then
          the fill on what the plate does not know

Reported: mean absolute error (0-255, BGR mean) over the evaluable pixels,
and the plate's coverage of them.

    .venv/bin/python -m tools.evaluate.background_fill \\
        [--videos 00170_upper,00148_upper] [--frames 120] [--every 10]
"""

from __future__ import annotations

import argparse
import collections

import cv2
import numpy as np

from tools.evaluate._common import panel_frames, write_json



def evaluate_clip(video, height=870, frames=120, every=10, start=30, verbose=True, fill="skin"):
    from tools.evaluate.tracking_jitter import landmarks_for

    from tryon.removal.background import BackgroundPlate, body_silhouette
    from tryon.body.fitter import BodyFitter
    from tryon.app.person import torso_visible, track_step
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover, arm_strokes, hand_discs

    images = panel_frames(video, height, frames)
    points, _ = landmarks_for(video, height, frames)
    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.fill = fill

    # pass 1: parser labels for every frame, and the true background
    masks, labels = [], []
    for image, p in zip(images, points):
        p = None if np.isnan(p).any() else p
        masks.append(remover.clothing_mask(image, keypoints=p))
        labels.append(remover._last_labels.copy())
    stack = np.stack(images)
    background = np.stack([
        ~cv2.dilate((lab != 0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) for lab in labels
    ])
    seen = background.sum(axis=0)
    truth = np.full(stack.shape[1:], np.nan, np.float32)
    for y0 in range(0, stack.shape[1], 40):
        chunk = stack[:, y0 : y0 + 40].astype(np.float32)
        chunk[~background[:, y0 : y0 + 40]] = np.nan
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                truth[y0 : y0 + 40] = np.nanmedian(chunk, axis=0)
    truth[seen < 3] = np.nan
    known_truth = np.isfinite(truth).all(axis=2)

    # pass 2: causal - plate and tracker see only the past
    plate = BackgroundPlate()
    fitter, errors = None, collections.deque(maxlen=10)
    rows = []
    for t, (image, p) in enumerate(zip(images, points)):
        body = None
        if not np.isnan(p).any() and torso_visible(p, tracking=fitter is not None):
            if fitter is None:
                fitter = BodyFitter()
                fitter.fit(p)
            else:
                track_step(fitter, p, errors, height)
            body = body_silhouette(fitter, image.shape) | arm_strokes(p, image.shape)
        else:
            fitter = None
        if t >= start and (t - start) % every == 0 and body is not None:
            region = masks[t] & ~body & known_truth
            if region.sum() >= 200:
                remover._last_labels = labels[t]
                remover._last_clothing = masks[t]
                arms = arm_strokes(p, image.shape, width_fraction=remover.ARM_TUBE_WIDTH)
                keep = hand_discs(p, image.shape)
                alone = remover.remove(image, masks[t], arms=arms, keep=keep)
                with_plate = remover.remove(image, masks[t], arms=arms, keep=keep, plate=plate, body=body)
                covered = region & plate.known()
                def error(out):
                    return float(np.abs(out[region].astype(np.float32) - truth[region]).mean())
                rows.append({"frame": t, "pixels": int(region.sum()), "plate_coverage": float(covered.sum() / region.sum()),
                             "fill_mae": error(alone), "plate_mae": error(with_plate)})
                if verbose:
                    print(f"{video} frame {t}: {rows[-1]['pixels']} px, plate covers {rows[-1]['plate_coverage']:.0%}, "
                          f"MAE {fill} {rows[-1]['fill_mae']:.1f} plate {rows[-1]['plate_mae']:.1f}", flush=True)
        plate.update(image, labels[t] != 0)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default="00170_upper,00148_upper,00176_outer")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--fill", choices=["skin", "lama"], default="skin")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = {}
    for video in args.videos.split(","):
        rows = evaluate_clip(video, frames=args.frames, every=args.every, fill=args.fill)
        weights = np.array([r["pixels"] for r in rows], np.float64)
        summary = {k: float(np.average([r[k] for r in rows], weights=weights)) for k in ("plate_coverage", "fill_mae", "plate_mae")} if rows else {}
        results[video] = {"frames": rows, "summary": summary}
        print(video, summary, flush=True)
    if args.out:
        write_json(args.out, results)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
