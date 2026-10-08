"""How much real shirt does `hem_leak` take out of the erase mask?

`hem_leak` drops the pixels below the hip line that the parser calls lower body, or dress on a
wearer in trousers, from the erase mask (the 0076 profile frame: the parser's upper-clothing
mask ran onto the trousers as a round blob under the hem). On a long shirt over shorts
(00152_outer: a plaid shirt to the thigh, over denim shorts) the hem is labelled dress, lies
below the hip line, and was dropped: the real plaid hem stayed visible under the new garment.

Ground truth for "this pixel is the real shirt": the plaid itself. 00152_outer's shirt is the
only red thing on the wearer (the top is orange, the shorts blue-grey, the room green), so
red hue + saturation, closed by a 9 px kernel to bridge the plaid's dark stripes. It is a colour
ground truth, not a registration mask, and it says nothing about the other videos; for them the
pixels are counted by label only.

For every `--step`-th frame of the video: the pixels the leak removes from the erase mask (the
raw clothing dilated by the remover's 6 px, intersected with the leak), split by the parser's
label (dress / skirt+pants / legs) and, on a plaid video, how many are red plaid.

    .venv/bin/python -m tools.evaluate.hem_leak_study 00152_outer            # one video per process
    .venv/bin/python -m tools.evaluate.hem_leak_study 00152_outer --crops    # before / after crops
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from tools.evaluate._common import panel_frames, write_json
from tryon.paths import BASELINE, REPO_ROOT

HEIGHT = 870
FRAMES = 150
STEP = 10
PLAID_VIDEOS = ("00152_outer",)


def plaid_red(image_bgr):
    """Red-plaid pixels of a frame: hue near red, saturated, closed over the dark stripes."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue <= 8) | (hue >= 170)) & (sat > 110) & (val > 60)
    return cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


def parse_with(remover, image, p, leak_rule):
    """The live parse with `hem_leak` replaced by `leak_rule(labels, keypoints)` (and the colour rescue as set on the remover)."""
    from tryon.removal import remover as remover_module

    real = remover_module.hem_leak
    remover_module.hem_leak = leak_rule
    try:
        return remover.parse_clothing(image, keypoints=p)
    finally:
        remover_module.hem_leak = real


def study(video, step=STEP, frames=FRAMES, crops=False):
    """Per frame, the pixels of the erase mask that `hem_leak` takes out: `before` (the label rule, no hanging-hem
    exception, no colour rescue) and `after` (the committed rule), against the mask with no leak removal at all."""
    from tools.evaluate.tracking_jitter import landmarks_for
    from tryon.removal import remover as remover_module
    from tryon.removal.parsing_schp import ATR_CHECKPOINT
    from tryon.removal.remover import ClothingRemover

    label_rule = remover_module.hem_leak
    images = panel_frames(video, HEIGHT, frames)
    points, _ = landmarks_for(video, HEIGHT, frames)
    remover = ClothingRemover(schp_checkpoint=ATR_CHECKPOINT)
    remover.KEEP_INNER_LAYER = False
    plaid = video in PLAID_VIDEOS
    rows, tiles = [], []
    for t in range(0, len(images), step):
        image, p = images[t], points[t]
        p = None if np.isnan(p).any() else np.asarray(p[:, :3], np.float64)
        remover.HEM_COLOUR = False
        none = parse_with(remover, image, p, lambda labels, keypoints: np.zeros(labels.shape, bool))
        before = parse_with(remover, image, p, lambda labels, keypoints: label_rule(labels, keypoints, False))
        remover.HEM_COLOUR = True
        after = remover.parse_clothing(image, keypoints=p)
        labels = none.labels
        gone_before, gone_after = none.mask & ~before.mask, none.mask & ~after.mask
        row = {"t": t}
        for name, gone in (("before", gone_before), ("after", gone_after)):
            row[f"{name}_px"] = int(gone.sum())
            row[f"{name}_legs_px"] = int((gone & np.isin(labels, (12, 13))).sum())
            row[f"{name}_skirt_pants_px"] = int((gone & np.isin(labels, (5, 6))).sum())
            row[f"{name}_dress_px"] = int((gone & (labels == 7)).sum())
            if plaid:
                row[f"{name}_plaid_px"] = int((gone & plaid_red(image)).sum())
        rescued = gone_before & ~gone_after
        row["rescued_px"] = int(rescued.sum())
        row["rescued_not_plaid_px"] = int((rescued & ~plaid_red(image)).sum()) if plaid else int(rescued.sum())
        rows.append(row)
        if crops and gone_before.any():
            tiles.append((t, image, none.mask, before.mask, after.mask))
    remover.HEM_COLOUR = True
    return rows, tiles


def crop_sheet(video, tiles, path, count=4):
    """Before / after crops of the hem: the real frame, the old mask, the new mask (erased area tinted)."""
    picks = [tiles[i] for i in np.linspace(0, len(tiles) - 1, min(count, len(tiles))).round().astype(int)]
    sheet = []
    for t, image, none, before, after in picks:
        ys, xs = np.nonzero(none & ~before | none & ~after)
        y0, y1 = max(ys.min() - 140, 0), min(ys.max() + 60, image.shape[0])
        x0, x1 = max(xs.min() - 100, 0), min(xs.max() + 100, image.shape[1])
        tiles_row = []
        for k, mask in enumerate((None, before, after)):
            tile = image.copy()
            if mask is not None:
                tile[mask] = (0.45 * tile[mask] + 0.55 * np.array((255, 0, 255))).astype(np.uint8)
            tile = tile[y0:y1, x0:x1]
            tile = cv2.resize(tile, None, fx=300 / tile.shape[0], fy=300 / tile.shape[0], interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"{video} f{t} " + ("real", "before", "after")[k], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles_row.append(tile)
        sheet.append(np.hstack(tiles_row))
    width = max(r.shape[1] for r in sheet)
    cv2.imwrite(str(path), np.vstack([cv2.copyMakeBorder(r, 0, 3, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--crops", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--match", type=float, default=None, help="GARMENT_MATCH to try")
    args = parser.parse_args()
    if args.match is not None:
        from tryon.removal import remover as remover_module

        remover_module.GARMENT_MATCH = args.match
    rows, tiles = study(args.video, args.step, crops=args.crops)
    keys = [k for k in rows[0] if k != "t"]
    totals = {k: int(sum(r[k] for r in rows)) for k in keys}
    print(args.video, len(rows), "frames", totals)
    write_json(BASELINE / f"hem-leak-{args.video}{args.tag}.json", {"video": args.video, "frames": rows, "totals": totals})
    if args.crops and tiles:
        crop_sheet(args.video, tiles, REPO_ROOT / "docs" / "figures" / f"hem-leak-{args.video}{args.tag}.jpg")


if __name__ == "__main__":
    main()
