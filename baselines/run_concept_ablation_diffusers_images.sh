#!/bin/bash
# Secondary smoke-test path: generate with official Concept Ablation diffusers pipeline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$ROOT/baseline-models/concept_ablation_diffusers"
OUT="$ROOT/results/concept_ablation_diffusers"
LOGS="$ROOT/logs"
REPO="$ROOT/baselines/external/concept-ablation/diffusers"
STEPS="${STEPS:-50}"
CFG="${CFG:-6.0}"
ETA="${ETA:-1.0}"
DEVICE="${DEVICE:-cuda:0}"
CA_SAMPLES="${CA_SAMPLES:-1}"
ONLY_ARTIST="${ONLY_ARTIST:-}"

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

mkdir -p "$OUT" "$LOGS"

find_delta() {
  local exp_dir="$1"
  if [ -f "$exp_dir/delta.bin" ]; then
    echo "$exp_dir/delta.bin"
    return 0
  fi
  local found
  found="$(find "$exp_dir" -name 'delta*.bin' -type f | sort | tail -n 1)"
  if [ -n "$found" ]; then
    echo "$found"
    return 0
  fi
  return 1
}

run_gen() {
  local idx="$1"
  local total="$2"
  local label="$3"
  local exp="$4"
  local csv="$5"
  local filter="${6:-}"
  local log="$7"
  local exp_dir="$WEIGHTS/$exp"
  local delta

  if ! delta="$(find_delta "$exp_dir")"; then
    echo "Could not find delta*.bin inside $exp_dir"
    exit 1
  fi

  echo "[$idx/$total] Concept Ablation diffusers smoke images: $label"
  python "$ROOT/baselines/generate_concept_ablation_images.py" \
    --repo_dir "$REPO" \
    --delta_path "$delta" \
    --prompts_path "$csv" \
    --artist_filter "$filter" \
    --save_path "$OUT" \
    --exp_name "$exp" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$CFG" \
    --eta "$ETA" \
    --num_samples "$CA_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee "$LOGS/concept_ablation_diffusers_images_$log"
}

should_run "Kelly McKernan" && run_gen 1 7 "Kelly McKernan" "concept_ablation-Kelly_McKernan" "$ROOT/data/kelly_prompts.csv" "" "kelly.log"
should_run "Van Gogh" && run_gen 2 7 "Van Gogh" "concept_ablation-Van_Gogh" "$ROOT/data/vangogh_prompts.csv" "" "vangogh.log"
should_run "Tyler Edlin" && run_gen 3 7 "Tyler Edlin" "concept_ablation-Tyler_Edlin" "$ROOT/data/short_niche_art_prompts.csv" "Tyler Edlin" "tyler_edlin.log"
should_run "Thomas Kinkade" && run_gen 4 7 "Thomas Kinkade" "concept_ablation-Thomas_Kinkade" "$ROOT/data/short_niche_art_prompts.csv" "Thomas Kinkade" "thomas_kinkade.log"
should_run "Kilian Eng" && run_gen 5 7 "Kilian Eng" "concept_ablation-Kilian_Eng" "$ROOT/data/short_niche_art_prompts.csv" "Kilian Eng" "kilian_eng.log"
should_run "Ajin: Demi Human" && run_gen 6 7 "Ajin: Demi Human" "concept_ablation-Ajin_Demi_Human" "$ROOT/data/short_niche_art_prompts.csv" "Ajin: Demi Human" "ajin.log"
should_run "Andy Warhol" && run_gen 7 7 "Andy Warhol" "concept_ablation-Andy_Warhol" "$ROOT/data/andy_warhol_prompts.csv" "Andy Warhol" "andy_warhol.log"
