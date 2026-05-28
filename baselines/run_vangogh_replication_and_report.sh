#!/bin/bash
# Run the Van Gogh ESD-x/UCE/CA/SPACE benchmark and report.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

LAST_STEP="init"
STEP_INDEX=0
TOTAL_STEPS=14
EXPECTED_COUNT=50
ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
REUSE_BASELINE_OUTPUTS="${REUSE_BASELINE_OUTPUTS:-1}"

on_error() {
  local code="$1"
  echo ""
  echo "Van Gogh replication failed at step: $LAST_STEP"
  echo "Exit code: $code"
  exit "$code"
}
trap 'on_error $?' ERR

run_step() {
  LAST_STEP="$1"
  shift
  STEP_INDEX=$((STEP_INDEX + 1))
  progress_step "$STEP_INDEX" "$TOTAL_STEPS" "$LAST_STEP"
  run_with_progress "$LAST_STEP" "$@"
}

count_pngs() {
  local dir="$1"
  if [ -d "$dir" ]; then
    find "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' '
  else
    echo 0
  fi
}

require_count() {
  local label="$1"
  local dir="$2"
  local count
  count="$(count_pngs "$dir")"
  if [ "$count" != "$EXPECTED_COUNT" ]; then
    echo "Expected $EXPECTED_COUNT $label PNGs, found $count in $dir"
    exit 1
  fi
  echo "$label PNGs: $count / $EXPECTED_COUNT"
}

has_expected_count() {
  local dir="$1"
  [ "$(count_pngs "$dir")" = "$EXPECTED_COUNT" ]
}

skip_step() {
  local index="$1"
  local label="$2"
  STEP_INDEX="$index"
  progress_step "$STEP_INDEX" "$TOTAL_STEPS" "$label skipped; cached artifacts present"
  echo "$label cached; skipping regeneration."
}

commit_and_push_results() {
  local push_status="push not requested"
  local commit_hash=""

  if [ "$PUSH_RESULTS" != "1" ]; then
    echo "PUSH_RESULTS=$PUSH_RESULTS; skipping result commit/push."
    return 0
  fi

  progress_banner "Commit and push Van Gogh result artifacts"
  git -C "$ROOT" add \
    results/uce/uce-Van_Gogh \
    results/space/space-Van_Gogh \
    results/concept_ablation_compvis/concept_ablation-Van_Gogh \
    results/provenance/uce \
    results/provenance/space \
    results/provenance/concept_ablation_compvis \
    results/evaluation/metrics.csv \
    results/evaluation/ablation_table.tex \
    results/evaluation/comparison_bars.png \
    results/evaluation/van_gogh_run_summary.md \
    results/evaluation/van_gogh_run_summary.json \
    results/comparisons/van_gogh.png
  git -C "$ROOT" add -f results/comparisons/van_gogh.pdf

  if git -C "$ROOT" diff --cached --quiet; then
    echo "No result changes to commit."
    commit_hash="$(git -C "$ROOT" rev-parse HEAD)"
    push_status="nothing to push"
  else
    git -C "$ROOT" commit -m "Add Van Gogh SPACE baseline comparison results"
    commit_hash="$(git -C "$ROOT" rev-parse HEAD)"
    push_status="commit created"
  fi

  if [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
    local token
    token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
    local askpass
    askpass="$(mktemp)"
    chmod 700 "$askpass"
    cat > "$askpass" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' "${GITHUB_USERNAME:-Vedang-P}" ;;
  *Password*) printf '%s\n' "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ;;
  *) printf '\n' ;;
esac
EOF
    GIT_ASKPASS="$askpass" GITHUB_USERNAME="${GITHUB_USERNAME:-Vedang-P}" git -C "$ROOT" push myprivate HEAD:main
    rm -f "$askpass"
  else
    git -C "$ROOT" push myprivate HEAD:main
  fi
  push_status="pushed to myprivate/main"

  echo "Result commit: $commit_hash"
  echo "Push status: $push_status"
}

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "This script is intentionally Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

export ONLY_ARTIST="Van Gogh"
export UCE_EXPAND_PROMPTS="${UCE_EXPAND_PROMPTS:-false}"
export UCE_GUIDE_CONCEPT="${UCE_GUIDE_CONCEPT:-art}"
export UCE_ERASE_SCALE="${UCE_ERASE_SCALE:-1}"
export UCE_PRESERVE_SCALE="${UCE_PRESERVE_SCALE:-1}"
export UCE_LAMB="${UCE_LAMB:-0.5}"
export BASELINE_INPUTS_READY=0

progress_banner "Van Gogh SPACE benchmark"
echo "Repo: $ROOT"
echo "Expected Van Gogh prompts/images per method: $EXPECTED_COUNT"
echo "Python: $PYTHON_BIN"
echo "PUSH_RESULTS: $PUSH_RESULTS"
echo "REUSE_BASELINE_OUTPUTS: $REUSE_BASELINE_OUTPUTS"
echo "Progress heartbeat interval: ${PROGRESS_INTERVAL:-30}s"
echo "UCE settings: guide=$UCE_GUIDE_CONCEPT erase_scale=$UCE_ERASE_SCALE preserve_scale=$UCE_PRESERVE_SCALE lamb=$UCE_LAMB expand_prompts=$UCE_EXPAND_PROMPTS"
echo "SPACE checkpoint target: space-models/sd/space-Van_Gogh.safetensors"

cd "$ROOT"

run_step "setup_official_repos" bash "$ROOT/baselines/setup_official_repos.sh"
export BASELINE_REPOS_READY=1

run_step "preflight_env" env AUTO_REPAIR_ENV=1 bash "$ROOT/baselines/preflight_env.sh"
export BASELINE_PREFLIGHT_READY=1

run_step "validate_baseline_inputs" bash "$ROOT/baselines/validate_baseline_inputs.sh"
export BASELINE_INPUTS_READY=1

prompt_count="$("$PYTHON_BIN" - <<'PY'
import pandas as pd
print(len(pd.read_csv("data/vangogh_prompts.csv")))
PY
)"
if [ "$prompt_count" != "$EXPECTED_COUNT" ]; then
  echo "Expected $EXPECTED_COUNT Van Gogh prompts, found $prompt_count"
  exit 1
fi
require_count "Vanilla Van Gogh" "$ROOT/results/baseline/vangogh"
require_count "ESD-x Van Gogh" "$ROOT/results/erased/vangogh"

ca_delta_present=0
if [ -f "$ROOT/baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt" ] || \
   [ -f "$ROOT/baseline-models/concept_ablation_compvis/deltas/concept_ablation-Van_Gogh.ckpt" ]; then
  ca_delta_present=1
fi
if [ "$REUSE_BASELINE_OUTPUTS" = "1" ] && [ "$ca_delta_present" = "1" ] && has_expected_count "$ROOT/results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples"; then
  skip_step 4 "setup_concept_ablation_assets"
else
  run_step "setup_concept_ablation_assets" bash "$ROOT/baselines/setup_concept_ablation_compvis.sh"
fi

if [ "${FORCE_UCE_REBUILD:-0}" = "1" ]; then
  progress_step 5 "$TOTAL_STEPS" "removing existing UCE checkpoint/images"
  rm -f "$ROOT/baseline-models/uce/uce-Van_Gogh.safetensors"
  rm -rf "$ROOT/results/uce/uce-Van_Gogh"
fi

if [ -f "$ROOT/baseline-models/uce/uce-Van_Gogh.safetensors" ]; then
  skip_step 5 "uce_training"
  echo "UCE checkpoint already present: baseline-models/uce/uce-Van_Gogh.safetensors"
else
  run_step "uce_training" bash "$ROOT/baselines/run_uce_training.sh"
fi

if [ "$REUSE_BASELINE_OUTPUTS" = "1" ] && has_expected_count "$ROOT/results/uce/uce-Van_Gogh"; then
  skip_step 6 "uce_images"
else
  run_step "uce_images" bash "$ROOT/baselines/run_uce_images.sh"
fi
require_count "UCE Van Gogh" "$ROOT/results/uce/uce-Van_Gogh"

if [ "${FORCE_SPACE_REBUILD:-0}" = "1" ]; then
  progress_step 8 "$TOTAL_STEPS" "removing existing SPACE checkpoint/images"
  rm -f "$ROOT/space-models/sd/space-Van_Gogh.safetensors"
  rm -rf "$ROOT/results/space/space-Van_Gogh"
fi

if [ -f "$ROOT/space-models/sd/space-Van_Gogh.safetensors" ]; then
  skip_step 8 "space_training"
  echo "SPACE checkpoint already present: space-models/sd/space-Van_Gogh.safetensors"
else
  run_step "space_training" bash "$ROOT/run_space_training.sh"
fi

if [ "${FORCE_SPACE_IMAGES:-0}" = "1" ]; then
  progress_step 9 "$TOTAL_STEPS" "removing existing SPACE images"
  rm -rf "$ROOT/results/space/space-Van_Gogh"
fi
if [ "${FORCE_SPACE_REBUILD:-0}" != "1" ] && [ "${FORCE_SPACE_IMAGES:-0}" != "1" ] && has_expected_count "$ROOT/results/space/space-Van_Gogh"; then
  skip_step 9 "space_images"
else
  run_step "space_images" bash "$ROOT/run_space_images.sh"
fi
require_count "SPACE Van Gogh" "$ROOT/results/space/space-Van_Gogh"

ca_delta_present=0
if [ -f "$ROOT/baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt" ] || \
   [ -f "$ROOT/baseline-models/concept_ablation_compvis/deltas/concept_ablation-Van_Gogh.ckpt" ]; then
  ca_delta_present=1
fi
if [ "$REUSE_BASELINE_OUTPUTS" = "1" ] && [ "$ca_delta_present" = "1" ]; then
  skip_step 10 "concept_ablation_training_or_official_delta"
else
  run_step "concept_ablation_training_or_official_delta" bash "$ROOT/baselines/run_concept_ablation_training.sh"
fi
if [ "$REUSE_BASELINE_OUTPUTS" = "1" ] && has_expected_count "$ROOT/results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples"; then
  skip_step 11 "concept_ablation_images"
else
  run_step "concept_ablation_images" bash "$ROOT/baselines/run_concept_ablation_images.sh"
fi
require_count "Concept Ablation Van Gogh" "$ROOT/results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples"

run_step "clear_vangogh_eval_cache" rm -f "$ROOT/results/evaluation/metrics_cache_v4_van_gogh.json"
run_step "evaluate_vangogh_strict" "$PYTHON_BIN" "$ROOT/evaluate.py" --only-artist "Van Gogh" --method-keys "esd,uce,ca,space" --strict-validation --strict-artist "Van Gogh"
run_step "comparison_grid_vangogh" "$PYTHON_BIN" "$ROOT/make_comparison_grids.py" --only-artist "Van Gogh" --methods "ESD-x,UCE,CA,SPACE"
run_step "write_summary" "$PYTHON_BIN" "$ROOT/baselines/validate_vangogh_replication.py" --write-summary

commit_and_push_results

cat <<EOF

Van Gogh benchmark completed.

Prompt count: $prompt_count
Vanilla outputs: results/baseline/vangogh
ESD-x outputs: results/erased/vangogh
UCE checkpoint: baseline-models/uce/uce-Van_Gogh.safetensors
UCE outputs: results/uce/uce-Van_Gogh
SPACE checkpoint: space-models/sd/space-Van_Gogh.safetensors
SPACE outputs: results/space/space-Van_Gogh
Concept Ablation outputs: results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples
Comparison chart: results/comparisons/van_gogh.png
Comparison PDF: results/comparisons/van_gogh.pdf
Metrics CSV: results/evaluation/metrics.csv
Ablation table: results/evaluation/ablation_table.tex
Summary: results/evaluation/van_gogh_run_summary.md
EOF
