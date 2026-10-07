"""Pretrained human parsing: SCHP (Li et al. 2020) ATR / LIP models, ONNX.

The parser IDM-VTON and VTON360 use (`preprocess/humanparsing` in
yisol/IDM-VTON, weights `humanparsing/parsing_atr.onnx` and
`parsing_lip.onnx` on HuggingFace). Same conventions as their
`parsing_api.py`, read from the source: the image is letterboxed into a
512x512 (ATR) or 473x473 (LIP) square with an aspect-preserving affine
warp, normalised in BGR with mean (0.406, 0.456, 0.485) and std
(0.225, 0.224, 0.229), the logits are upsampled bilinearly to the square,
warped back per channel and argmax'd LAST - labels are never resampled.

ATR classes (18): 0 Background, 1 Hat, 2 Hair, 3 Sunglasses,
4 Upper-clothes, 5 Skirt, 6 Pants, 7 Dress, 8 Belt, 9 Left-shoe,
10 Right-shoe, 11 Face, 12 Left-leg, 13 Right-leg, 14 Left-arm,
15 Right-arm, 16 Bag, 17 Scarf - the same scheme as this project's own
U-Net, so the two are directly comparable and the merged upper-clothing
region is classes 4 and 7 for both.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tryon.paths import CACHE, REPO_ROOT

ROOT = REPO_ROOT
ATR_CHECKPOINT = ROOT / "models" / "pretrained" / "parsing_atr.onnx"
LIP_CHECKPOINT = ROOT / "models" / "pretrained" / "parsing_lip.onnx"
_MEAN = np.array([0.406, 0.456, 0.485], np.float32)
_STD = np.array([0.225, 0.224, 0.229], np.float32)


def _coreml_cache(checkpoint):
    """Where ONNX Runtime keeps the CoreML compilation of `checkpoint`: it
    compiled the graph on every start (ATR 7.6 s, LIP about as long) and
    reuses it from here (0.5 s, the same output). The directory is named by
    the file's size and time, so a changed model never meets a stale
    compilation (ORT keys its cache by the model's path only)."""
    import os

    stat = os.stat(checkpoint)
    path = CACHE / "coreml" / f"{Path(checkpoint).stem}-{stat.st_size}-{int(stat.st_mtime)}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class SCHPParser:
    def __init__(self, checkpoint=ATR_CHECKPOINT, size=512, providers=None):
        """`providers`: ONNX Runtime execution providers, by preference.
        Default: CoreML (NeuralNetwork format - the MLProgram format fails
        to compile this graph's AveragePool) with the CPU as fallback.
        Measured on the 870x640 camera-0004 frame: CoreML 304 ms vs CPU
        2,125 ms a pass, argmax labels 99.995% identical."""
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                providers.insert(0, ("CoreMLExecutionProvider", {"ModelFormat": "NeuralNetwork", "ModelCacheDirectory": _coreml_cache(checkpoint)}))
            if "CUDAExecutionProvider" in ort.get_available_providers():  # onnxruntime-gpu on NVIDIA
                providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3  # CoreML reports its partition count as a warning
        self.session = ort.InferenceSession(str(checkpoint), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(size)

    def _letterbox(self, height, width):
        """Affine from the image to a `size` square, aspect preserved."""
        scale = self.size / max(height, width)
        matrix = np.array([[scale, 0.0, (self.size - width * scale) / 2.0], [0.0, scale, (self.size - height * scale) / 2.0]], np.float32)
        return matrix

    def _run(self, square):
        """(classes, h, w) logits for one normalised HWC square."""
        tensor = np.ascontiguousarray(square.transpose(2, 0, 1)[None])
        out = self.session.run(None, {self.input_name: tensor})[0]
        return out[0] if out.ndim == 4 else out

    def probabilities(self, frame_bgr, flip=True):
        """(classes, H, W) softmax probabilities at frame resolution."""
        height, width = frame_bgr.shape[:2]
        matrix = self._letterbox(height, width)
        square = cv2.warpAffine(frame_bgr, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR)
        x = (square.astype(np.float32) / 255.0 - _MEAN) / _STD
        # the exported graph has a fixed batch of 1, so the flip is a second run
        logits = [self._run(x)] + ([self._run(x[:, ::-1])] if flip else [])
        if flip:
            flipped = logits[1][:, :, ::-1].copy()
            # swap the sided classes back (ATR: shoes 9/10, legs 12/13, arms 14/15)
            for a, b in ((9, 10), (12, 13), (14, 15)):
                if flipped.shape[0] > b:
                    flipped[[a, b]] = flipped[[b, a]]
            logits = 0.5 * (logits[0] + flipped)
        else:
            logits = logits[0]
        # logits may come at a lower resolution than the square: upsample first
        if logits.shape[1] != self.size:
            logits = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for c in logits])
        inverse = cv2.invertAffineTransform(matrix)
        full = np.stack([cv2.warpAffine(c, inverse, (width, height), flags=cv2.INTER_LINEAR) for c in logits])
        full = full - full.max(axis=0, keepdims=True)
        exp = np.exp(full)
        return exp / exp.sum(axis=0, keepdims=True)

    def labels(self, frame_bgr):
        return self.probabilities(frame_bgr).argmax(axis=0).astype(np.uint8)

    def _square_logits(self, frame_bgr):
        """One pass, no flip: (classes, size, size) logits and the letterbox."""
        height, width = frame_bgr.shape[:2]
        matrix = self._letterbox(height, width)
        square = cv2.warpAffine(frame_bgr, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR)
        logits = self._run((square.astype(np.float32) / 255.0 - _MEAN) / _STD)
        if logits.shape[1] != self.size:
            logits = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for c in logits])
        return logits, cv2.invertAffineTransform(matrix)

    def fast_labels(self, frame_bgr):
        """`labels` for the live loop: one pass, no flip, the argmax taken
        on the network's square and carried to the frame by nearest
        neighbour - instead of every class warped to the frame first."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        return cv2.warpAffine(logits.argmax(axis=0).astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)

    def fast_clothing(self, frame_bgr, classes):
        """(summed probability of `classes`, labels) at frame size, for the
        live loop: `probabilities(flip=False)` with the softmax and argmax
        on the network's square and only those two maps warped."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        shifted = np.exp(logits - logits.max(axis=0, keepdims=True))
        probability = shifted[list(classes)].sum(axis=0) / shifted.sum(axis=0)
        probability = cv2.warpAffine(np.ascontiguousarray(probability, np.float32), inverse, (width, height), flags=cv2.INTER_LINEAR)
        labels = cv2.warpAffine(logits.argmax(axis=0).astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)
        return probability, labels
