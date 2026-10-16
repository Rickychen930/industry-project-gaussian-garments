"""FASHN human parser (SegFormer-B4, 18 classes), as ONNX.

`fashn-ai/fashn-human-parser` on HuggingFace (NVIDIA SegFormer licence, non-commercial).
Unlike ATR / LIP it labels the bare body under clothing as its own classes - arms (12),
hands (13), torso (16) - which clothing removal needs. Exported to
`models/pretrained/fashn_human_parser.onnx` in an isolated venv (transformers 4.44.2,
torch 2.2.2, opset 17); the export matches the PyTorch model in the argmax.

Preprocessing is the package's: RGB, `cv2.INTER_AREA` resize to 384 x 576 (width x
height), ImageNet mean / std. Logits come out at a quarter of that and are upsampled
bilinearly to the frame, then softmax.

Classes: 0 background, 1 face, 2 hair, 3 top, 4 dress, 5 skirt, 6 pants, 7 belt, 8 bag,
9 hat, 10 scarf, 11 glasses, 12 arms, 13 hands, 14 legs, 15 feet, 16 torso, 17 jewelry.
"""

from __future__ import annotations

import cv2
import numpy as np

from tryon.paths import REPO_ROOT

ROOT = REPO_ROOT
FASHN_CHECKPOINT = ROOT / "models" / "pretrained" / "fashn_human_parser.onnx"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
UPPER_CLASSES = (3, 4)  # top, dress


class FashnParser:
    def __init__(self, checkpoint=FASHN_CHECKPOINT, providers=None):
        """CPU by default: CoreML split the graph into 249 partitions and ran 4x slower."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(checkpoint), options, providers=providers or ["CPUExecutionProvider"])

    def probabilities(self, frame_bgr):
        """(18, H, W) softmax probabilities at frame resolution."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (384, 576), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.session.run(None, {"pixel_values": x})[0][0]
        full = np.stack([cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR) for c in logits])
        full = full - full.max(axis=0, keepdims=True)
        exp = np.exp(full)
        return exp / exp.sum(axis=0, keepdims=True)

    def labels(self, frame_bgr):
        return self.probabilities(frame_bgr).argmax(axis=0).astype(np.uint8)


# FASHN class -> the ATR id the rest of the removal code reads. FASHN has no left /
# right split: arms and hands go to ATR 14 (arm skin, kept by the fill), and so does its
# bare TORSO (ATR has no class for it), so visible skin is never erased.
TO_ATR = np.array([0, 11, 2, 4, 7, 5, 6, 8, 16, 1, 17, 3, 14, 14, 12, 9, 14, 16], np.uint8)
