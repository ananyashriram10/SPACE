#!/bin/bash
# Future-facing UCE concurrent six-artist erasure checkpoint.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UCE_DIR="$ROOT/baselines/external/unified-concept-editing"
OUT="$ROOT/baseline-models/uce"
LOGS="$ROOT/logs"
PROVENANCE="$ROOT/results/provenance/uce"
DEVICE="${DEVICE:-cuda:0}"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/six_artist_config.sh"

if [ "${BASELINE_REPOS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/setup_official_repos.sh"
fi
if [ "${BASELINE_PREFLIGHT_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/preflight_env.sh"
fi

if [ ! -f "$UCE_DIR/trainscripts/uce_sd_erase.py" ]; then
  echo "Missing official UCE repo at $UCE_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

mkdir -p "$OUT" "$LOGS" "$PROVENANCE"
bash "$ROOT/baselines/apply_official_patches.sh"

progress_banner "UCE concurrent six-artist training"
echo "Edit concepts: $SPACE_SIX_ARTIST_UCE_CONCEPTS"
echo "Guide concepts: $SPACE_SIX_ARTIST_GUIDE_CONCEPTS"
echo "Preserve concepts: $SPACE_DEFAULT_PRESERVE_CONCEPTS"
echo "Output: $OUT/$SPACE_SIX_ARTIST_EXP_NAME.safetensors"

cmd="python trainscripts/uce_sd_erase.py --model_id CompVis/stable-diffusion-v1-4 --edit_concepts $SPACE_SIX_ARTIST_UCE_CONCEPTS --guide_concepts $SPACE_SIX_ARTIST_GUIDE_CONCEPTS --preserve_concepts $SPACE_DEFAULT_PRESERVE_CONCEPTS --device $DEVICE --concept_type art --expand_prompts true --save_dir $OUT --exp_name $SPACE_SIX_ARTIST_EXP_NAME"
(
  cd "$UCE_DIR"
  run_with_progress "UCE concurrent six-artist official training" python trainscripts/uce_sd_erase.py \
    --model_id "CompVis/stable-diffusion-v1-4" \
    --edit_concepts "$SPACE_SIX_ARTIST_UCE_CONCEPTS" \
    --guide_concepts "$SPACE_SIX_ARTIST_GUIDE_CONCEPTS" \
    --preserve_concepts "$SPACE_DEFAULT_PRESERVE_CONCEPTS" \
    --device "$DEVICE" \
    --concept_type "art" \
    --expand_prompts true \
    --save_dir "$OUT" \
    --exp_name "$SPACE_SIX_ARTIST_EXP_NAME"
) 2>&1 | tee "$LOGS/uce_train_$SPACE_SIX_ARTIST_EXP_NAME.log"

python "$ROOT/baselines/record_provenance.py" \
  --method "UCE" \
  --artist "six artists concurrent" \
  --run_type "official-paper-concurrent" \
  --repo_url "https://github.com/rohitgandikota/unified-concept-editing" \
  --repo_dir "$UCE_DIR" \
  --command "$cmd" \
  --outputs "$OUT/$SPACE_SIX_ARTIST_EXP_NAME.safetensors" \
  --out "$PROVENANCE/train_$SPACE_SIX_ARTIST_EXP_NAME.json"

echo ""
echo "UCE concurrent six-artist checkpoint ready: $OUT/$SPACE_SIX_ARTIST_EXP_NAME.safetensors"
