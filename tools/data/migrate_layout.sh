#!/usr/bin/env bash
# One-time move of the project's data from the old, loose layout in
# ~/Downloads into the data root ~/tryon-data (tryon/paths.py, docs/DATASETS.md).
# Only moves folders (mv, same disk: instant, nothing is copied or deleted);
# each step is skipped when its source is missing or its target exists.
#
#   tools/data/migrate_layout.sh [--dry-run]
#
# After it, delete the "legacy layout" block at the end of tryon/paths.py.
set -Eeuo pipefail
OLD="$HOME/Downloads"
NEW="${TRYON_DATA:-$HOME/tryon-data}"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

move() {  # move <old> <new>
  local from="$OLD/$1" to="$NEW/$2"
  [[ -e "$from" ]] || { echo "  skip   $1 (not there)"; return; }
  [[ ! -e "$to" ]] || { echo "  skip   $2 (already exists)"; return; }
  echo "  move   $1 -> $2"
  [[ $DRY == 1 ]] || { mkdir -p "$(dirname "$to")"; mv "$from" "$to"; }
}

echo "data root: $NEW"
move gaugar_models                                   garments
move gaugar_trajectories                             trajectories
move drive-download-20260829T090506Z-1-001           release_videos
move models/smplx                                    body_models/smplx
move student_training                                training_logs
move async_capture                                   captures
move tryon_refs/survey/Harmonizer                    references/Harmonizer

# the optional Multi-HMR clone reads the SMPL-X models through a symlink
LINK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/models/pretrained/multi-hmr/models/smplx"
if [[ -L "$LINK" && $DRY == 0 && -d "$NEW/body_models/smplx" ]]; then ln -sfn "$NEW/body_models/smplx" "$LINK"; echo "  relink $LINK"; fi
echo "done. Check with: ./live-tryon doctor"
