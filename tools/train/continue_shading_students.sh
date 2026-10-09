#!/usr/bin/env bash
# Continue every garment's shading student for more cycles, and make the result live only when it
# scores higher on the held-out poses than the one it started from (the live one is kept as
# stage3/shading_student_before_more.pt the first time).
#
#   tools/train/continue_shading_students.sh [garment ...]     # default: all but 00190_upper
#   CYCLES=400 tools/train/continue_shading_students.sh 00134_upper
#
# Logs: $LOG/<garment>-more.log, one line a garment in $LOG/progress.log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
read -r LOG GARMENT_DIR < <(PYTHONPATH=. .venv/bin/python -c 'from tryon.paths import TRAINING_LOGS, GARMENTS; print(TRAINING_LOGS, GARMENTS)')
mkdir -p "$LOG"
GARMENTS=("$@")
if [[ ${#GARMENTS[@]} -eq 0 ]]; then
  GARMENTS=($(.venv/bin/python -c "from tryon.paths import GARMENTS; print(' '.join(sorted(p.name for p in GARMENTS.iterdir() if (p / 'stage1').is_dir() and p.name != '00190_upper')))"))
fi
for g in "${GARMENTS[@]}"; do
  stage="$GARMENT_DIR/$g/stage3"
  [[ -f "$stage/shading_student.pt" ]] || { echo "$(date '+%F %T') $g: no student, skipped" >> "$LOG/progress.log"; continue; }
  cp -n "$stage/shading_student.pt" "$stage/shading_student_before_more.pt"
  cp "$stage/shading_student.pt" "$stage/shading_student_more.pt"
  echo "$(date '+%F %T') more $g" >> "$LOG/progress.log"
  caffeinate -dims .venv/bin/python -m tools.train.shading_student "$g" --resume --cycles "${CYCLES:-400}" \
    --with-motions --out-name shading_student_more.pt > "$LOG/$g-more.log" 2>&1
  start=$(grep -oE 're-measured [0-9.]+' "$LOG/$g-more.log" | grep -oE '[0-9.]+$')
  best=$(grep -oE '^done: best held-out [0-9.]+' "$LOG/$g-more.log" | grep -oE '[0-9.]+$')
  if [[ -n "$start" && -n "$best" ]] && awk "BEGIN{exit !($best > $start)}"; then
    cp "$stage/shading_student_more.pt" "$stage/shading_student.pt"
    echo "$(date '+%F %T') $g: $start -> $best dB, promoted" >> "$LOG/progress.log"
  else
    echo "$(date '+%F %T') $g: ${start:-?} -> ${best:-?} dB, kept the old student" >> "$LOG/progress.log"
  fi
done
echo "$(date '+%F %T') continued students done" >> "$LOG/progress.log"
