#!/usr/bin/env bash
# Train a shading student for every garment in turn and promote each when done
# (tools/train/shading_student.py, with --with-motions).
#
#   tools/train/all_shading_students.sh [garment ...]     # default: all
#   CYCLES=400 tools/train/all_shading_students.sh 00134_upper
#
# Runs under caffeinate: GL context creation aborted (exit 134) once the display
# slept. Logs: $LOG/<garment>.log (default ~/tryon-data/training_logs), plus progress.log.
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
