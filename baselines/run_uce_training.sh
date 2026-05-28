#!/bin/bash
# Train UCE style-erasure baselines with the official UCE repository.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UCE_DIR="$ROOT/baselines/external/unified-concept-editing"
OUT="$ROOT/baseline-models/uce"
LOGS="$ROOT/logs"
PROVENANCE="$ROOT/results/provenance/uce"
DEVICE="${DEVICE:-cuda:0}"
PRESERVE="${PRESERVE_CONCEPTS:-Monet; Rembrandt; Warhol}"
UCE_GUIDE_CONCEPT="${UCE_GUIDE_CONCEPT:-art}"
UCE_ERASE_SCALE="${UCE_ERASE_SCALE:-1}"
UCE_PRESERVE_SCALE="${UCE_PRESERVE_SCALE:-1}"
UCE_LAMB="${UCE_LAMB:-0.5}"
UCE_EXPAND_PROMPTS="${UCE_EXPAND_PROMPTS:-false}"
ONLY_ARTIST="${ONLY_ARTIST:-}"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

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

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

run_uce() {
  local idx="$1"
  local total="$2"
  local concept="$3"
  local exp="$4"
  local cmd

  echo ""
  progress_step "$idx" "$total" "UCE training: $concept"
  cmd="python trainscripts/uce_sd_erase.py --model_id CompVis/stable-diffusion-v1-4 --edit_concepts $concept --guide_concepts $UCE_GUIDE_CONCEPT --preserve_concepts $PRESERVE --device $DEVICE --concept_type art --erase_scale $UCE_ERASE_SCALE --preserve_scale $UCE_PRESERVE_SCALE --lamb $UCE_LAMB --expand_prompts $UCE_EXPAND_PROMPTS --save_dir $OUT --exp_name $exp"
  (
    cd "$UCE_DIR"
    run_with_progress "UCE official training for $concept" python trainscripts/uce_sd_erase.py \
      --model_id "CompVis/stable-diffusion-v1-4" \
      --edit_concepts "$concept" \
      --guide_concepts "$UCE_GUIDE_CONCEPT" \
      --preserve_concepts "$PRESERVE" \
      --device "$DEVICE" \
      --concept_type "art" \
      --erase_scale "$UCE_ERASE_SCALE" \
      --preserve_scale "$UCE_PRESERVE_SCALE" \
      --lamb "$UCE_LAMB" \
      --expand_prompts "$UCE_EXPAND_PROMPTS" \
      --save_dir "$OUT" \
      --exp_name "$exp"
  ) 2>&1 | tee "$LOGS/uce_train_$exp.log"

  python "$ROOT/baselines/record_provenance.py" \
    --method "UCE" \
    --artist "$concept" \
    --run_type "official-paper" \
    --repo_url "https://github.com/rohitgandikota/unified-concept-editing" \
    --repo_dir "$UCE_DIR" \
    --command "$cmd" \
    --outputs "$OUT/$exp.safetensors" \
    --out "$PROVENANCE/train_$exp.json"
}

progress_banner "UCE official baseline training"
echo "Output: $OUT"
echo "Guide concept: $UCE_GUIDE_CONCEPT"
echo "Preserve concepts: $PRESERVE"
echo "erase_scale=$UCE_ERASE_SCALE preserve_scale=$UCE_PRESERVE_SCALE lamb=$UCE_LAMB expand_prompts=$UCE_EXPAND_PROMPTS"

should_run "Kelly McKernan" && run_uce 1 7 "Kelly McKernan" "uce-Kelly_McKernan"
should_run "Van Gogh" && run_uce 2 7 "Van Gogh" "uce-Van_Gogh"
should_run "Tyler Edlin" && run_uce 3 7 "Tyler Edlin" "uce-Tyler_Edlin"
should_run "Thomas Kinkade" && run_uce 4 7 "Thomas Kinkade" "uce-Thomas_Kinkade"
should_run "Kilian Eng" && run_uce 5 7 "Kilian Eng" "uce-Kilian_Eng"
should_run "Ajin: Demi Human" && run_uce 6 7 "Ajin: Demi Human" "uce-Ajin_Demi_Human"
should_run "Andy Warhol" && PRESERVE="Monet; Rembrandt; Picasso" run_uce 7 7 "Andy Warhol" "uce-Andy_Warhol"

echo ""
echo "UCE training complete. Next: bash baselines/run_uce_images.sh"
