"""Where every file the project reads lives, in one place.

Inside the repository:

    REPO_ROOT       the repository (this package's parent)
    MODELS          models/: trained and pretrained weights (`./setup.sh` fills it)
    DATA            data/: test sets (annotated views, parsing sets; git-ignored)
    BASELINE        docs/baseline/: every measured result, as JSON

Outside it, under one data root - `~/tryon-data` unless the environment
variable TRYON_DATA says otherwise (docs/DATASETS.md says where each was
downloaded from):

    GARMENTS        garments/<name>/                Gaussian Garments checkpoints
    TRAJECTORIES    trajectories/                   the authors' cloth trajectories
    RELEASE_VIDEOS  release_videos/<name>.mp4       the authors' release videos
    SMPLX_ROOT      body_models/smplx/*.npz         SMPL-X body models
    HARMONIZER      references/Harmonizer           optional colour harmoniser
    CAPTURES        captures/                       recorded pipeline output for display tests
    TRAINING_LOGS   training_logs/                  logs of the long training runs

Caches and scratch output:

    CACHE           ~/.cache/tryon                  compiled models (CoreML), ...
    SCRATCH         <temp dir>/tryon                throw-away pictures and caches
"""

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "data"
BASELINE = REPO_ROOT / "docs" / "baseline"

DATA_ROOT = Path(os.environ.get("TRYON_DATA", Path.home() / "tryon-data")).expanduser()
GARMENTS = DATA_ROOT / "garments"
TRAJECTORIES = DATA_ROOT / "trajectories"
RELEASE_VIDEOS = DATA_ROOT / "release_videos"
SMPLX_ROOT = DATA_ROOT / "body_models"
HARMONIZER = DATA_ROOT / "references" / "Harmonizer"
CAPTURES = DATA_ROOT / "captures"
TRAINING_LOGS = DATA_ROOT / "training_logs"

# The layout before 8 Oct 2026 (data loose in ~/Downloads). Used only while
# `tools/data/migrate_layout.sh` has not been run; delete this block after.
_LEGACY = Path.home() / "Downloads"
if "TRYON_DATA" not in os.environ and not DATA_ROOT.exists() and (_LEGACY / "gaugar_models").is_dir():
    DATA_ROOT = _LEGACY
    GARMENTS = _LEGACY / "gaugar_models"
    TRAJECTORIES = _LEGACY / "gaugar_trajectories"
    RELEASE_VIDEOS = _LEGACY / "drive-download-20260829T090506Z-1-001"
    SMPLX_ROOT = _LEGACY / "models"
    HARMONIZER = _LEGACY / "tryon_refs" / "survey" / "Harmonizer"
    CAPTURES = _LEGACY / "async_capture"
    TRAINING_LOGS = _LEGACY / "student_training"

CACHE = Path.home() / ".cache" / "tryon"
SCRATCH = Path(tempfile.gettempdir()) / "tryon"


def garment(name):
    """The checkpoint directory of garment `name` (e.g. "00152_outer")."""
    return GARMENTS / name
