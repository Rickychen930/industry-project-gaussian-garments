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
        x = square.astype(np.float32)  # in place, the same arithmetic as (x / 255 - mean) / std
        x /= 255.0
        x -= _MEAN
        x /= _STD
        logits = self._run(x)
        if logits.shape[1] != self.size:
            logits = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for c in logits])
        return logits, cv2.invertAffineTransform(matrix)

    def fast_labels(self, frame_bgr):
        """`labels` for the live loop: one pass, no flip, the argmax taken
        on the network's square and carried to the frame by nearest
        neighbour - instead of every class warped to the frame first."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        window, inverse = self._window(logits, inverse, height, width)
        return cv2.warpAffine(self._first_argmax(window, window.max(axis=0)), inverse, (width, height), flags=cv2.INTER_NEAREST)

    def fast_labels_reference(self, frame_bgr):
        """The original `fast_labels` (oracle)."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        return cv2.warpAffine(logits.argmax(axis=0).astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)

    def _window(self, logits, inverse, height, width):
        """(contiguous window of the square the picture occupies, the
        square -> frame affine from the window's origin). A contiguous copy:
        numpy's reductions over axis 0 are 9x faster on it than on a slice."""
        y0, y1, x0, x1 = self._content_window(height, width, self._letterbox(height, width))
        window = np.ascontiguousarray(logits[:, y0:y1, x0:x1])
        inverse = inverse.copy()
        inverse[:, 2] += inverse[:, :2] @ np.array([x0, y0], np.float32)
        return window, inverse

    @staticmethod
    def _first_argmax(window, peak):
        """`window.argmax(axis=0)` as uint8, 3x faster: numpy returns the
        first maximum, so walking the classes downwards lets the lowest index
        have the last word."""
        best = np.zeros(peak.shape, np.uint8)
        for c in range(window.shape[0] - 1, -1, -1):
            best[window[c] == peak] = c
        return best

    def fast_clothing_reference(self, frame_bgr, classes):
        """The original `fast_clothing` (the oracle `fast_clothing` is tested
        equal to, and the baseline of the parser study of 9 Oct, docs/LOG.md)."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        shifted = np.exp(logits - logits.max(axis=0, keepdims=True))
        probability = shifted[list(classes)].sum(axis=0) / shifted.sum(axis=0)
        probability = cv2.warpAffine(np.ascontiguousarray(probability, np.float32), inverse, (width, height), flags=cv2.INTER_LINEAR)
        labels = cv2.warpAffine(logits.argmax(axis=0).astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST)
        return probability, labels

    def _content_window(self, height, width, matrix):
        """(y0, y1, x0, x1) of the letterboxed picture inside the square, 2 px
        wider than it: a wide or tall frame leaves up to half of the square
        as padding that no frame pixel reads (bilinear reads one pixel past)."""
        scale = float(matrix[0, 0])
        x0 = max(int(np.floor(matrix[0, 2])) - 2, 0)
        y0 = max(int(np.floor(matrix[1, 2])) - 2, 0)
        x1 = min(int(np.ceil(matrix[0, 2] + width * scale)) + 2, self.size)
        y1 = min(int(np.ceil(matrix[1, 2] + height * scale)) + 2, self.size)
        return y0, y1, x0, x1

    def _buffer(self, shape):
        buffer = getattr(self, "_exp_buffer", None)
        if buffer is None or buffer.shape != shape:
            buffer = self._exp_buffer = np.empty(shape, np.float32)
        return buffer

    def fast_clothing(self, frame_bgr, classes):
        """(summed probability of `classes`, labels) at frame size, for the
        live loop: `probabilities(flip=False)` with the softmax and argmax
        on the network's square and only those two maps warped. Labels
        bit-identical to `fast_clothing_reference`, the probability within one
        float32 ulp of the exponential (the thresholded mask equal), in under
        half the time: only the window of the square the picture occupies is
        touched, as a contiguous copy, the exponentials go into one reused
        buffer, and the argmax is a per-class comparison with the running
        maximum instead of numpy's strided argmax."""
        logits, inverse = self._square_logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        window, inverse = self._window(logits, inverse, height, width)
        peak = window.max(axis=0)
        np.subtract(window, peak, out=self._buffer(window.shape))
        shifted = self._exp_buffer
        # cv2.exp: 2.1 ms against numpy's 5.6, within one float32 ulp of it
        cv2.exp(shifted.reshape(-1, shifted.shape[-1]), shifted.reshape(-1, shifted.shape[-1]))
        numerator = shifted[classes[0]].copy()
        for c in classes[1:]:
            numerator += shifted[c]
        probability = numerator / shifted.sum(axis=0)
        best = self._first_argmax(window, peak)
        probability = cv2.warpAffine(probability, inverse, (width, height), flags=cv2.INTER_LINEAR)
        labels = cv2.warpAffine(best, inverse, (width, height), flags=cv2.INTER_NEAREST)
        return probability, labels
