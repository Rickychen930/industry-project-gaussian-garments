"""Fetch the pretrained weights into models/pretrained/.

    .venv/bin/python -m tools.data.download_pretrained [schp lama nlf]

schp = SCHP parsers (clothing removal), lama = Big-LaMa inpainter (optional),
nlf = NLF-S body regressor (non-commercial; needs its own venv, see setup.sh).
Everything but SCHP is skipped cleanly when absent.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "models" / "pretrained"

WEIGHTS = {
    # SCHP (Li et al. 2020) ATR / LIP parsers as exported by IDM-VTON.
    "schp": [
        ("parsing_atr.onnx", "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx"),
        ("parsing_lip.onnx", "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx"),
    ],
    # Big-LaMa as TorchScript (simple-lama-inpainting's file; torch.jit.load).
    "nlf": [
        ("nlf/nlf_s_multi_0.2.2.torchscript", "https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_s_multi_0.2.2.torchscript"),
    ],
    "lama": [
        ("big-lama.pt", "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"),
    ],
}


def main(names):
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in names or WEIGHTS:
        for filename, url in WEIGHTS[name]:
            path = TARGET / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > 1_000_000:
                print(f"{filename}: present ({path.stat().st_size / 1e6:.0f} MB)")
                continue
            print(f"{filename}: downloading", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"{filename}: {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"components to fetch: {', '.join(WEIGHTS)} (default: all)")
    names = parser.parse_args().names
    unknown = sorted(set(names) - set(WEIGHTS))
    if unknown:
        parser.error(f"unknown component(s): {', '.join(unknown)}")
    main(names)
