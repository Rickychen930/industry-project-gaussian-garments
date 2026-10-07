#!/usr/bin/env bash
# Set up the live try-on on a fresh machine (macOS on Apple Silicon is the
# tested target; see docs/USAGE.md "GPU setup" for NVIDIA). Safe to re-run:
# every step skips what is already done.
#
#   ./setup.sh             Python env, pretrained weights, NLF body regressor, checks
#   ./setup.sh --no-nlf    skip the NLF body regressor (the keypoint fit is used)
#
# Two things cannot be downloaded by a script (licences); the final check
# says where to put them:
#   - SMPL-X body models  -> ~/tryon-data/body_models/smplx/   (https://smpl-x.is.tue.mpg.de)
#   - garment checkpoints -> ~/tryon-data/garments/<name>/  (docs/DATASETS.md)
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
WITH_NLF=1
for arg in "$@"; do [[ "$arg" == "--no-nlf" ]] && WITH_NLF=0; done

step() { printf '\n[%s] %s\n' "$1" "$2"; }

step 1/5 "Python environment (.venv, Python 3.10, requirements.txt)"
PYTHON="${PYTHON:-python3.10}"
command -v "$PYTHON" >/dev/null || PYTHON=python3
[[ -x .venv/bin/python ]] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip wheel
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "  done: $(.venv/bin/python --version)"

step 2/5 "Pretrained weights (SCHP clothing parsers, NLF-S body regressor)"
if [[ $WITH_NLF == 1 ]]; then .venv/bin/python -m tools.data.download_pretrained schp nlf; else .venv/bin/python -m tools.data.download_pretrained schp; fi

step 3/5 "NLF-S environment (models/pretrained/nlf/venv: torch 2.14, which NLF needs)"
if [[ $WITH_NLF == 0 ]]; then
  echo "  skipped (--no-nlf)"
elif [[ -x models/pretrained/nlf/venv/bin/python ]]; then
  echo "  present"
elif command -v uv >/dev/null; then
  uv venv models/pretrained/nlf/venv --python 3.12 --quiet
  VIRTUAL_ENV=models/pretrained/nlf/venv uv pip install --quiet torch==2.14.0 torchvision==0.29.0 "numpy<2" scipy "opencv-python-headless<4.11"
  echo "  done"
else
  echo "  needs uv (https://docs.astral.sh/uv/: 'brew install uv'), then re-run ./setup.sh"
fi

step 4/5 "Scan poses of the garment checkpoints (once per garment)"
MODELS="$(PYTHONPATH="$ROOT" .venv/bin/python -c 'from tryon.paths import GARMENTS; print(GARMENTS)')"
if [[ -d "$MODELS" ]]; then
  for dir in "$MODELS"/*/; do
    [[ -d "$dir/stage1" && ! -f "$dir/stage1/scan_pose.npz" ]] || continue
    echo "  fitting $(basename "$dir")"
    PYTHONPATH="$ROOT" .venv/bin/python -m tools.data.fit_scan_pose "$dir" >/dev/null || echo "  (no trajectory for $(basename "$dir"); it binds on the default pose)"
  done
  echo "  done"
else
  echo "  no garments at $MODELS yet (see the check below)"
fi

step 5/5 "Check"
.venv/bin/python doctor.py || true
echo
echo "Next: ./live-tryon test   then   ./live-tryon"
