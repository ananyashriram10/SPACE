#!/bin/bash
# Static audit for official baseline wrappers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v rg >/dev/null 2>&1; then
  SEARCH_CMD=(rg -n)
else
  SEARCH_CMD=(grep -R -n -E)
fi

echo "Checking official repo pins..."
git -C baselines/external/unified-concept-editing rev-parse HEAD
git -C baselines/external/concept-ablation rev-parse HEAD
git -C baselines/external/unified-concept-editing status --short
git -C baselines/external/concept-ablation status --short

echo ""
echo "Checking official baseline scripts do not call local method generators..."
if "${SEARCH_CMD[@]}" "evalscripts/generate-images.py|generate_concept_ablation_images.py" \
  baselines/run_uce_images.sh \
  baselines/run_concept_ablation_images.sh \
  baselines/run_uce_training.sh \
  baselines/run_concept_ablation_training.sh; then
  echo "Official baseline wrapper is calling a local generator."
  exit 1
fi

echo ""
echo "Checking expected official calls..."
"${SEARCH_CMD[@]}" "trainscripts/uce_sd_erase.py|evalscripts/generate-images-sd.py" baselines/run_uce_*.sh
"${SEARCH_CMD[@]}" "compvis|train.py|sample.py" baselines/run_concept_ablation_training.sh baselines/run_concept_ablation_images.sh

echo ""
echo "Checking prompt/result path references..."
"${SEARCH_CMD[@]}" "uce-|concept_ablation-|results/uce|results/concept_ablation_compvis" baselines README.md evaluate.py make_comparison_grids.py

echo ""
echo "Static baseline audit OK."
