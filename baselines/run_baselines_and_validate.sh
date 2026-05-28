#!/bin/bash
# Orchestrate official baseline runs and validation in strict fail-fast order.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAST_STEP="init"
SMOKE="${SMOKE:-0}"
STRICT_EVAL="${STRICT_EVAL:-1}"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

on_error() {
  local code="$1"
  echo ""
  echo "Baseline pipeline failed at step: $LAST_STEP"
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

STEP_INDEX=0
TOTAL_STEPS=11

if [ "$SMOKE" = "1" ]; then
  export ONLY_ARTIST="Van Gogh"
  echo "Smoke mode enabled: ONLY_ARTIST=$ONLY_ARTIST"
fi

progress_banner "SPACE baseline pipeline"
echo "Repo: $ROOT"
echo "Smoke mode: $SMOKE"
echo "Strict eval: $STRICT_EVAL"
echo "Progress heartbeat interval: ${PROGRESS_INTERVAL:-30}s"

run_step "setup_official_repos" bash "$ROOT/baselines/setup_official_repos.sh"
export BASELINE_REPOS_READY=1
run_step "preflight_env" env AUTO_REPAIR_ENV=1 bash "$ROOT/baselines/preflight_env.sh"
export BASELINE_PREFLIGHT_READY=1
run_step "validate_baseline_inputs" bash "$ROOT/baselines/validate_baseline_inputs.sh"
export BASELINE_INPUTS_READY=1
run_step "setup_concept_ablation_assets" bash "$ROOT/baselines/setup_concept_ablation_compvis.sh"
run_step "uce_training" bash "$ROOT/baselines/run_uce_training.sh"
run_step "uce_images" bash "$ROOT/baselines/run_uce_images.sh"
run_step "concept_ablation_training" bash "$ROOT/baselines/run_concept_ablation_training.sh"
run_step "concept_ablation_images" bash "$ROOT/baselines/run_concept_ablation_images.sh"
run_step "clear_eval_cache" rm -f "$ROOT/results/evaluation/metrics_cache_v4.json"

if [ "$STRICT_EVAL" = "1" ]; then
  eval_args=(--strict-validation)
  if [ -n "${ONLY_ARTIST:-}" ]; then
    eval_args+=(--strict-artist "$ONLY_ARTIST")
  fi
  run_step "evaluate_strict" python "$ROOT/evaluate.py" "${eval_args[@]}"
else
  run_step "evaluate_relaxed" python "$ROOT/evaluate.py"
fi

run_step "comparison_grids" python "$ROOT/make_comparison_grids.py"

echo ""
echo "Baseline pipeline completed successfully."
