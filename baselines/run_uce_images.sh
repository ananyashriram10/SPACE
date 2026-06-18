#!/bin/bash
# Generate images from official UCE checkpoints with the official UCE generator.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UCE_DIR="$ROOT/baselines/external/unified-concept-editing"
WEIGHTS="$ROOT/baseline-models/uce"
OUT="$ROOT/results/uce"
LOGS="$ROOT/logs"
CACHE="$ROOT/baselines/cache/official_prompts/uce"
PROVENANCE="$ROOT/results/provenance/uce"
STEPS="${STEPS:-50}"
CFG="${CFG:-7.5}"
DEVICE="${DEVICE:-cuda:0}"
UCE_SAMPLES="${UCE_SAMPLES:-1}"
ONLY_ARTIST="${ONLY_ARTIST:-}"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

if [ "${BASELINE_REPOS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/setup_official_repos.sh"
fi
if [ "${BASELINE_PREFLIGHT_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/preflight_env.sh"
fi
if [ "${BASELINE_INPUTS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/validate_baseline_inputs.sh"
fi

mkdir -p "$OUT" "$LOGS" "$CACHE" "$PROVENANCE"
bash "$ROOT/baselines/apply_official_patches.sh"

if [ ! -f "$UCE_DIR/evalscripts/generate-images-sd.py" ]; then
  echo "Missing official UCE repo at $UCE_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

check_weight() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing UCE checkpoint: $path"
    echo "Run: bash baselines/run_uce_training.sh"
    exit 1
  fi
}

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

run_gen() {
  local idx="$1"
  local total="$2"
  local label="$3"
  local ckpt="$4"
  local csv="$5"
  local filter="${6:-}"
  local log="$7"
  local exp
  exp="$(basename "$ckpt" .safetensors)"
  local prompt_csv="$CACHE/$exp.csv"
  local prompt_txt="$CACHE/$exp.txt"
  local manifest="$CACHE/$exp.manifest.json"
  local image_dir="$OUT/$exp"
  local cmd

  check_weight "$ckpt"
  python "$ROOT/baselines/prepare_official_prompts.py" \
    --input_csv "$csv" \
    --artist "$label" \
    --artist_filter "$filter" \
    --out_csv "$prompt_csv" \
    --out_txt "$prompt_txt" \
    --manifest "$manifest"

  rm -rf "$image_dir"

  progress_step "$idx" "$total" "UCE images: $label"
  cmd="python evalscripts/generate-images-sd.py --model_id CompVis/stable-diffusion-v1-4 --uce_model_path $ckpt --prompts_path $prompt_csv --save_path $OUT --exp_name $exp --num_images_per_prompt 1 --num_inference_steps $STEPS --guidance_scale $CFG --device $DEVICE"
  (
    cd "$UCE_DIR"
    run_with_progress "UCE official image generation for $label" python evalscripts/generate-images-sd.py \
      --model_id "CompVis/stable-diffusion-v1-4" \
      --uce_model_path "$ckpt" \
      --prompts_path "$prompt_csv" \
      --save_path "$OUT" \
      --exp_name "$exp" \
      --num_images_per_prompt "$UCE_SAMPLES" \
      --num_inference_steps "$STEPS" \
      --guidance_scale "$CFG" \
      --device "$DEVICE"
  ) 2>&1 | tee "$LOGS/$log"

  python "$ROOT/baselines/check_baseline_integrity.py" \
    --manifest "$manifest" \
    --image_dir "$image_dir" \
    --mode uce

  python "$ROOT/baselines/record_provenance.py" \
    --method "UCE" \
    --artist "$label" \
    --run_type "official-paper" \
    --repo_url "https://github.com/rohitgandikota/unified-concept-editing" \
    --repo_dir "$UCE_DIR" \
    --command "$cmd" \
    --outputs "$ckpt" "$image_dir" \
    --prompt_manifest "$manifest" \
    --out "$PROVENANCE/$exp.json"
}

progress_banner "UCE image generation"
echo "Output: $OUT"

should_run "Kelly McKernan" && run_gen 1 7 "Kelly McKernan" "$WEIGHTS/uce-Kelly_McKernan.safetensors" "$ROOT/data/kelly_prompts.csv" "" "uce_images_kelly.log"
should_run "Van Gogh" && run_gen 2 7 "Van Gogh" "$WEIGHTS/uce-Van_Gogh.safetensors" "$ROOT/data/vangogh_prompts.csv" "" "uce_images_vangogh.log"
should_run "Tyler Edlin" && run_gen 3 7 "Tyler Edlin" "$WEIGHTS/uce-Tyler_Edlin.safetensors" "$ROOT/data/short_niche_art_prompts.csv" "Tyler Edlin" "uce_images_tyler_edlin.log"
should_run "Thomas Kinkade" && run_gen 4 7 "Thomas Kinkade" "$WEIGHTS/uce-Thomas_Kinkade.safetensors" "$ROOT/data/short_niche_art_prompts.csv" "Thomas Kinkade" "uce_images_thomas_kinkade.log"
should_run "Kilian Eng" && run_gen 5 7 "Kilian Eng" "$WEIGHTS/uce-Kilian_Eng.safetensors" "$ROOT/data/short_niche_art_prompts.csv" "Kilian Eng" "uce_images_kilian_eng.log"
should_run "Ajin: Demi Human" && run_gen 6 7 "Ajin: Demi Human" "$WEIGHTS/uce-Ajin_Demi_Human.safetensors" "$ROOT/data/short_niche_art_prompts.csv" "Ajin: Demi Human" "uce_images_ajin.log"
should_run "Andy Warhol" && run_gen 7 7 "Andy Warhol" "$WEIGHTS/uce-Andy_Warhol.safetensors" "$ROOT/data/andy_warhol_prompts.csv" "Andy Warhol" "uce_images_andy_warhol.log"

echo ""
echo "UCE image generation complete."
