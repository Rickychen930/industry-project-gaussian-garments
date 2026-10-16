#!/usr/bin/env python3
"""Check that this machine has what the 3D try-on needs, and nothing else.

Each check names the fix. Exit status is the number of blocking problems,
so `./live-tryon doctor && ./live-tryon test` reads naturally.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from tryon.garment.scan_pose import scan_pose_file
from tryon.paths import DATA, GARMENTS, SCAN_POSES, SMPLX_ROOT

GARMENTS_ROOT = GARMENTS
ANNOTATED = DATA / "annotated"
CORPUS = DATA / "corpus"

# (module, why it is needed)
PACKAGES = (
    ("numpy", "arrays"),
    ("scipy", "cKDTree for garment binding"),
    ("torch", "SMPL-X forward pass and training"),
    ("cv2", "image I/O and the webcam"),
    ("mediapipe", "2D keypoints - the pre-trained input the fit starts from"),
    ("smplx", "the SMPL-X body model loader"),
    ("trimesh", "garment mesh loading"),
)


def check_packages():
    problems = 0
    for module, why in PACKAGES:
        try:
            importlib.import_module(module)
            print(f"  ok       {module:10} {why}")
        except ImportError:
            problems += 1
            print(f"  MISSING  {module:10} {why}  ->  ./setup.sh")
    return problems


def check_body_models():
    neutral = SMPLX_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"
    if neutral.exists():
        print(f"  ok       SMPL-X body models at {SMPLX_ROOT}")
        return 0
    print(
        f"  MISSING  SMPL-X body models expected at {SMPLX_ROOT}/smplx/\n"
        "           Register and download from https://smpl-x.is.tue.mpg.de\n"
        "           (not redistributable; nothing here can fetch them for you)"
    )
    return 1


def check_garments():
    if not GARMENTS_ROOT.is_dir():
        print(
            f"  MISSING  garment checkpoints at {GARMENTS_ROOT} - the 4D-DRESS\n"
            "           subject templates; see docs/DATASETS.md"
        )
        return 1
    meshes = sorted(GARMENTS_ROOT.glob("*/stage1/template_uv.obj"))
    print(f"  ok       {len(meshes)} garment meshes under {GARMENTS_ROOT}")
    return 0 if meshes else 1


def check_scan_poses():
    """The scan pose each garment binds on: where it comes from, and whether a
    garment whose trajectory frame is not its template has the refined one."""
    import numpy as np

    groups = {}
    for template in sorted(GARMENTS_ROOT.glob("*/stage1/template_uv.obj")):
        name = template.parents[1].name
        path, source = scan_pose_file(template.parents[1])
        if path is not None and source == "cached" and not (SCAN_POSES / f"{name}.npz").exists():
            # a pose of our own fitting: suspicious when the fit says it is not this template's body
            data = np.load(path)
            mismatch = float(data["template_mismatch_m"]) if "template_mismatch_m" in data.files else 0.0
            if mismatch > 0.01 and "refined" not in data.files:
                source = "suspicious"
        groups.setdefault(source, []).append(name)
    if not groups:
        print("  info     no garments, no scan poses")
        return 0
    texts = {
        "asset": "from assets/scan_poses/ (the repository's own; same on every machine)",
        "cached": "cached in the checkpoint (fitted here, no repository asset)",
        "suspicious": "UNREFINED: the trajectory frame is not the template -> python -m tools.data.refine_scan_pose <checkpoint>",
        "missing": "none: binds on a default arm angle -> ./setup.sh",
    }
    for source in ("asset", "cached", "suspicious", "missing"):
        if source in groups:
            flag = "ok      " if source in ("asset", "cached") else "WARNING "
            print(f"  {flag} scan pose {texts[source]}: {len(groups[source])} ({' '.join(groups[source])})")
    return 0


def check_data():
    # Neither blocks: the fit and the binding work without them; evaluation and goal tests
    # measure against them.
    for root, what in ((ANNOTATED, "annotated photographs"), (CORPUS, "robustness corpus")):
        count = len(list(root.rglob("*.png"))) if root.is_dir() else 0
        state = "ok      " if count else "absent  "
        print(f"  {state} {count:4d} {what} under {root}")
    return 0


def check_compute():
    import torch

    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    print(f"  info     torch {torch.__version__}, MPS {'available' if mps else 'absent'}, CUDA {'available' if cuda else 'absent'}")
    print("           SMPL-X runs on CPU by design (6 ms; MPS falls back on padding ops)")
    print("           appearance net (--appearance) on " + ("MPS" if mps else "the CPU (slow: ~10x)"))
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        accel = [p for p in providers if p in ("CoreMLExecutionProvider", "CUDAExecutionProvider")]
        print(f"  {'ok      ' if accel else 'info    '} SCHP parser on {accel[0] if accel else 'the CPU only (~2 s a frame)'}")
    except ImportError:
        print("  MISSING  onnxruntime (the SCHP clothing parser)  ->  ./setup.sh")
    try:
        import moderngl

        context = moderngl.create_standalone_context(require=330)
        print(f"  ok       OpenGL for --gpu-render: {context.info['GL_RENDERER']}, GL {context.info['GL_VERSION']}")
        context.release()
    except Exception as error:  # no GL 3.3 context: the CPU splat loop is used
        print(f"  info     no OpenGL 3.3 context for --gpu-render ({error}); drop --gpu-render")
    root = Path(__file__).resolve().parent / "models" / "pretrained" / "nlf"
    ready = (root / "venv" / "bin" / "python").exists() and (root / "nlf_s_multi_0.2.2.torchscript").exists()
    print(f"  {'ok      ' if ready else 'info    '} NLF-S body regressor {'installed' if ready else 'absent (the keypoint fit is used; README: body regressor)'}")
    return 0


def main():
    sections = (
        ("Packages", check_packages),
        ("Body models", check_body_models),
        ("Garments", check_garments),
        ("Scan poses", check_scan_poses),
        ("Data", check_data),
        ("Compute", check_compute),
    )
    problems = 0
    for title, check in sections:
        print(f"\n{title}")
        problems += check()
    print()
    print("Ready." if not problems else f"{problems} blocking problem(s) above.")
    return problems


if __name__ == "__main__":
    sys.exit(main())
