"""A contact sheet of every garment checkpoint: what each one looks like,
drawn front-on by the live renderer on one neutral body (arms lowered),
labelled with its number and name - the numbers `./live-tryon run N` takes.

    .venv/bin/python -m tools.data.garment_previews [--models ~/tryon-data/garments] [--out docs/figures/garments.jpg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from tryon.body.fitter import BodyFitter
from tryon.garment.renderer import GarmentRenderer
from tryon.paths import GARMENTS

SIZE = (560, 400)  # height, width of one tile


def wardrobe(models):
    return sorted(p for p in Path(models).expanduser().iterdir() if (p / "stage1").is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=str(GARMENTS))
    parser.add_argument("--out", default="docs/figures/garments.jpg")
    args = parser.parse_args()
    body = BodyFitter()
    with torch.no_grad():
        pose = body.body_pose.view(-1, 3)
        pose[15, 2] = -1.1  # left shoulder (joint 16) down
        pose[16, 2] = 1.1  # right shoulder (joint 17) down
        body.scale.fill_(300.0)
        # model y +0.6 (above the shoulders) .. -1.1 (ankles) onto the tile
        body.translation.copy_(torch.tensor([SIZE[1] / 2, 20.0 + 300.0 * 0.6]).reshape(body.translation.shape))
    tiles = []
    for number, path in enumerate(wardrobe(args.models), 1):
        renderer = GarmentRenderer(path, dynamics=False)
        rgba = renderer.render(body, SIZE).astype(np.float32)
        alpha = rgba[..., 3:] / 255.0
        tile = (rgba[..., :3] * alpha + 235 * (1 - alpha)).astype(np.uint8)
        cv2.putText(tile, f"{number}  {path.name}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        tiles.append(tile)
        print(number, path.name, flush=True)
    columns = 5
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 235))
    sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
