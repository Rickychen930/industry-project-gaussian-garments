"""Colour harmonisation of the rendered garment into the live frame.

The garment was captured under 4D-DRESS studio light; a webcam scene is lit
differently, so the pasted garment reads as a sticker. Harmonizer (Ke et al.,
ECCV 2022, github.com/ZHKKKe/Harmonizer, CC BY-NC-SA 4.0 - loaded from its
own clone, not vendored) predicts white-box filter arguments (brightness,
contrast, saturation, temperature, ...) for the foreground from a low-res
look at the composite and applies them at full resolution.

Measured on the reference frame (00152_outer frame 0, `tools/compare_tryon_
systems.py`, docs/REFERENCE_RUNS.md):
  - same studio light as the capture: slightly worse (PSNR 14.03 -> 13.39,
    DISTS 0.108 -> 0.114) - nothing to correct, so it only adds error;
  - scene recoloured warm and dim (BGR gains 0.72 / 0.88 / 1.05, gamma
    1.15) with the garment left in studio colours, as a webcam would see
    it: PSNR 14.88 -> 15.37 (garment lit by the scene: 15.50), DISTS
    0.122 -> 0.112 (ideal 0.116), LPIPS 0.237 -> 0.239.
Cost at 640 x 870: predicting the arguments 31 ms on MPS (110 ms CPU),
applying them 57 ms on the CPU (507 ms on MPS). So the arguments are
predicted every `every` frames on MPS and smoothed (EMA), and applied every
frame on the CPU. Opt-in (`live.py --harmonize`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tryon.paths import HARMONIZER

HARMONIZER_ROOT = HARMONIZER


class LiveHarmonizer:
    def __init__(self, root=HARMONIZER_ROOT, every=10, ema=0.5, predict_device=None):
        import torch

        root = Path(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src import model  # the clone's own package

        self.torch = torch
        self.model = model.Harmonizer()
        self.model.load_state_dict(torch.load(root / "pretrained" / "harmonizer.pth", map_location="cpu"), strict=True)
        self.model.eval()
        if predict_device is None:
            predict_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.predict_device = predict_device
        self.every = max(int(every), 1)
        self.ema = float(ema)
        self.arguments = None
        self.frames = 0

    def __call__(self, frame_bgr, mask):
        """The frame with the `mask` region (bool, H x W) harmonised to the rest."""
        torch = self.torch
        if not np.asarray(mask).any():
            return frame_bgr
        comp = torch.from_numpy(np.ascontiguousarray(frame_bgr[..., ::-1])).permute(2, 0, 1)[None].float() / 255.0
        fg = torch.from_numpy(np.asarray(mask, np.float32))[None, None]
        with torch.no_grad():
            if self.arguments is None or self.frames % self.every == 0:
                self.model.to(self.predict_device)
                predicted = [a.to("cpu") for a in self.model.predict_arguments(comp.to(self.predict_device), fg.to(self.predict_device))]
                if self.arguments is None:
                    self.arguments = predicted
                else:
                    self.arguments = [self.ema * p + (1.0 - self.ema) * a for p, a in zip(predicted, self.arguments)]
                self.model.to("cpu")
            out = self.model.restore_image(comp, fg, self.arguments)[-1]
        self.frames += 1
        return (out[0].permute(1, 2, 0).numpy()[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
