"""Export the FASHN human parser to ONNX for `tryon/removal/parsing_fashn.py`.

Runs in a SEPARATE venv so this project's pinned torch / torchvision stay
untouched (transformers is not a project dependency):

    python3.10 -m venv /tmp/fashn_venv
    /tmp/fashn_venv/bin/pip install "torch==2.2.2" "transformers==4.44.2" "onnx==1.16.2" onnxscript "numpy==1.26.4"
    /tmp/fashn_venv/bin/.venv/bin/python -m tools.data.export_fashn_parser

Writes models/pretrained/fashn_human_parser.onnx (~257 MB; input
pixel_values (1, 3, 576, 384), output logits (1, 18, 144, 96)). The model
is `fashn-ai/fashn-human-parser` on HuggingFace, NVIDIA SegFormer licence
(non-commercial).
"""

from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

OUT = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "fashn_human_parser.onnx"


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    model = SegformerForSemanticSegmentation.from_pretrained("fashn-ai/fashn-human-parser").eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Logits(model), torch.randn(1, 3, 576, 384), str(OUT), input_names=["pixel_values"], output_names=["logits"], opset_version=17)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
