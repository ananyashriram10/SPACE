#!/bin/bash
# Regenerate only the Van Gogh UCE baseline with official-style UCE settings.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
EXPECTED_COUNT=50

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "This repair script is Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

export ONLY_ARTIST="Van Gogh"
export UCE_GUIDE_CONCEPT="${UCE_GUIDE_CONCEPT:-art}"
export UCE_ERASE_SCALE="${UCE_ERASE_SCALE:-1}"
export UCE_PRESERVE_SCALE="${UCE_PRESERVE_SCALE:-1}"
export UCE_LAMB="${UCE_LAMB:-0.5}"
export UCE_EXPAND_PROMPTS="${UCE_EXPAND_PROMPTS:-false}"

count_pngs() {
  local dir="$1"
  if [ -d "$dir" ]; then
    find "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' '
  else
    echo 0
  fi
}

progress_banner "Repair Van Gogh UCE baseline"
echo "Repo: $ROOT"
echo "Expected images: $EXPECTED_COUNT"
echo "Python: $PYTHON_BIN"
echo "UCE settings:"
echo "  guide=$UCE_GUIDE_CONCEPT"
echo "  erase_scale=$UCE_ERASE_SCALE"
echo "  preserve_scale=$UCE_PRESERVE_SCALE"
echo "  lamb=$UCE_LAMB"
echo "  expand_prompts=$UCE_EXPAND_PROMPTS"

cd "$ROOT"

progress_step 1 8 "setup official repos"
bash "$ROOT/baselines/setup_official_repos.sh"
export BASELINE_REPOS_READY=1

progress_step 2 8 "preflight"
env AUTO_REPAIR_ENV=1 bash "$ROOT/baselines/preflight_env.sh"
export BASELINE_PREFLIGHT_READY=1

progress_step 3 8 "validate inputs"
bash "$ROOT/baselines/validate_baseline_inputs.sh"
export BASELINE_INPUTS_READY=1

progress_step 4 8 "remove broken UCE outputs"
rm -f "$ROOT/baseline-models/uce/uce-Van_Gogh.safetensors"
rm -rf "$ROOT/results/uce/uce-Van_Gogh"
rm -f "$ROOT/results/provenance/uce/train_uce-Van_Gogh.json" "$ROOT/results/provenance/uce/uce-Van_Gogh.json"

progress_step 5 8 "train fixed UCE"
bash "$ROOT/baselines/run_uce_training.sh"

progress_step 6 8 "generate fixed UCE images"
bash "$ROOT/baselines/run_uce_images.sh"

uce_count="$(count_pngs "$ROOT/results/uce/uce-Van_Gogh")"
if [ "$uce_count" != "$EXPECTED_COUNT" ]; then
  echo "Expected $EXPECTED_COUNT fixed UCE images, found $uce_count"
  exit 1
fi

progress_step 7 8 "recompute Van Gogh metrics/grid"
rm -f "$ROOT/results/evaluation/metrics_cache_v4_van_gogh.json"
"$PYTHON_BIN" "$ROOT/evaluate.py" --only-artist "Van Gogh" --method-keys "esd,uce,ca" --strict-validation --strict-artist "Van Gogh"
"$PYTHON_BIN" "$ROOT/make_comparison_grids.py" --only-artist "Van Gogh" --methods "ESD-x,UCE,CA"

progress_step 8 8 "rewrite summary"
"$PYTHON_BIN" "$ROOT/baselines/validate_vangogh_replication.py" --write-summary

cat <<EOF

Fixed Van Gogh UCE rerun complete.

UCE images: results/uce/uce-Van_Gogh ($uce_count PNGs)
Comparison chart: results/comparisons/van_gogh.png
Metrics CSV: results/evaluation/metrics.csv
Summary: results/evaluation/van_gogh_run_summary.md

Review the UCE column visually before committing/pushing over the previous results.
EOF
