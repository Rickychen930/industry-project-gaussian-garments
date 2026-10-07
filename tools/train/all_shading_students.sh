#!/usr/bin/env bash
# Train a shading student for every garment, one after another, and make
# each the live one when it is done (tools/train/shading_student.py), with
# the multi-garment motions as extra training poses (--with-motions).
#
#   tools/train/all_shading_students.sh [garment ...]     # default: all
#   CYCLES=400 tools/train/all_shading_students.sh 00134_upper
#
# Runs under caffeinate: the training renders with OpenGL, and creating a
# GL context aborted (exit 134) once the display had gone to sleep. Logs go
# to $LOG (default ~/tryon-data/training_logs), one file a garment, and a
# line per garment to progress.log there.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG=${LOG:-$(PYTHONPATH=. .venv/bin/python -c 'from tryon.paths import TRAINING_LOGS; print(TRAINING_LOGS)')}
mkdir -p "$LOG"
GARMENTS=("$@")
if [[ ${#GARMENTS[@]} -eq 0 ]]; then
  GARMENTS=($(.venv/bin/python -c "from tryon.paths import GARMENTS; print(' '.join(sorted(p.name for p in GARMENTS.iterdir() if (p / 'stage1').is_dir())))"))
fi
for g in "${GARMENTS[@]}"; do
  echo "$(date '+%F %T') student $g" >> "$LOG/progress.log"
  caffeinate -dims .venv/bin/python -m tools.train.shading_student "$g" --cycles "${CYCLES:-400}" \
    --with-motions --out-name shading_student_new.pt --promote > "$LOG/$g.log" 2>&1
  echo "$(date '+%F %T') $g (exit $?): $(grep -E '^(done|promoted)' "$LOG/$g.log" | tr '\n' ' ')" >> "$LOG/progress.log"
done
echo "$(date '+%F %T') all students done" >> "$LOG/progress.log"
