#!/bin/bash
# Generate images from official Concept Ablation CompVis checkpoints with official sample.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CA_ROOT="$ROOT/baselines/external/concept-ablation"
CA_DIR="$CA_ROOT/compvis"
WEIGHTS="$ROOT/baseline-models/concept_ablation_compvis"
OUT="$ROOT/results/concept_ablation_compvis"
LOGS="$ROOT/logs"
CACHE="$ROOT/baselines/cache/official_prompts/concept_ablation_compvis"
PROVENANCE="$ROOT/results/provenance/concept_ablation_compvis"
ASSETS="$ROOT/baseline-assets/concept_ablation/pretrained_models"
STEPS="${STEPS:-100}"
CFG="${CFG:-6.0}"
ETA="${ETA:-1.0}"
SD_CKPT="${SD_CKPT:-$ASSETS/sd-v1-4.ckpt}"
ONLY_ARTIST="${ONLY_ARTIST:-}"
source "$ROOT/baselines/progress.sh"

if [ "${BASELINE_REPOS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/setup_official_repos.sh"
fi
if [ "${BASELINE_PREFLIGHT_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/preflight_env.sh"
fi
if [ "${BASELINE_INPUTS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/validate_baseline_inputs.sh"
fi

mkdir -p "$OUT" "$LOGS" "$CACHE" "$PROVENANCE" "$WEIGHTS/deltas"

if [ ! -f "$CA_DIR/sample.py" ]; then
  echo "Missing official Concept Ablation CompVis repo at $CA_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

if [ ! -f "$SD_CKPT" ]; then
  echo "Missing SD v1.4 original checkpoint: $SD_CKPT"
  echo "Run: bash baselines/setup_concept_ablation_compvis.sh"
  exit 1
fi

# OpenAI CLIP is required by ldm/modules/encoders/modules.py.
# Install from source if not present (e.g. on a freshly restored RunPod venv).
if ! python -c "import clip" >/dev/null 2>&1; then
  echo "[$(progress_now)] OpenAI CLIP not found — installing from source..."
  python -m pip install --progress-bar on "git+https://github.com/openai/CLIP.git@main#egg=clip"
fi

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

find_delta() {
  local exp="$1"
  local official="$WEIGHTS/official_weights/$exp.ckpt"
  if [ -f "$official" ]; then
    echo "$official"
    return 0
  fi

  local found=""
  found="$(find "$WEIGHTS/logs" -path "*_$exp/checkpoints/step_*.ckpt" -type f 2>/dev/null | sort | tail -n 1 || true)"
  if [ -n "$found" ]; then
    local materialized="$WEIGHTS/deltas/$exp.ckpt"
    cp "$found" "$materialized"
    echo "$materialized"
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
  local prompt_csv="$CACHE/$exp.csv"
  local prompt_txt="$CACHE/$exp.txt"
  local manifest="$CACHE/$exp.manifest.json"
  local image_dir="$OUT/$exp"
  local cmd

  python "$ROOT/baselines/prepare_official_prompts.py" \
    --input_csv "$csv" \
    --artist "$label" \
    --artist_filter "$filter" \
    --out_csv "$prompt_csv" \
    --out_txt "$prompt_txt" \
    --manifest "$manifest"

  local delta
  if ! delta="$(find_delta "$exp")"; then
    echo "Could not find official/downloaded or trained CompVis delta for $exp"
    echo "Run: bash baselines/setup_concept_ablation_compvis.sh and/or bash baselines/run_concept_ablation_training.sh"
    exit 1
  fi

  rm -rf "$image_dir"
  mkdir -p "$image_dir"

  progress_step "$idx" "$total" "Concept Ablation images: $label"
  cmd="python sample.py --ckpt $SD_CKPT --delta_ckpt $delta --from-file $prompt_txt --ddim_steps $STEPS --scale $CFG --ddim_eta $ETA --outdir $image_dir --n_samples 1 --n_copies 1 --skip_grid --metadata --seed 42"
  (
    cd "$CA_DIR"
    run_with_progress "Concept Ablation official image generation for $label" python sample.py \
      --ckpt "$SD_CKPT" \
      --delta_ckpt "$delta" \
      --from-file "$prompt_txt" \
      --ddim_steps "$STEPS" \
      --scale "$CFG" \
      --ddim_eta "$ETA" \
      --outdir "$image_dir" \
      --n_samples 1 \
      --n_copies 1 \
      --skip_grid \
      --metadata \
      --seed 42
  ) 2>&1 | tee "$LOGS/concept_ablation_compvis_images_$log"

  python "$ROOT/baselines/check_baseline_integrity.py" \
    --manifest "$manifest" \
    --image_dir "$image_dir/samples" \
    --mode concept-ablation-compvis

  python "$ROOT/baselines/record_provenance.py" \
    --method "Concept Ablation" \
    --artist "$label" \
    --run_type "official-paper" \
    --repo_url "https://github.com/nupurkmr9/concept-ablation" \
    --repo_dir "$CA_ROOT" \
    --command "$cmd" \
    --outputs "$delta" "$image_dir" \
    --prompt_manifest "$manifest" \
    --out "$PROVENANCE/images_$exp.json"
}

progress_banner "Concept Ablation official CompVis image generation"
echo "Output: $OUT"

should_run "Kelly McKernan" && run_gen 1 6 "Kelly McKernan" "concept_ablation-Kelly_McKernan" "$ROOT/data/kelly_prompts.csv" "" "kelly.log"
should_run "Van Gogh" && run_gen 2 6 "Van Gogh" "concept_ablation-Van_Gogh" "$ROOT/data/vangogh_prompts.csv" "" "vangogh.log"
should_run "Tyler Edlin" && run_gen 3 6 "Tyler Edlin" "concept_ablation-Tyler_Edlin" "$ROOT/data/short_niche_art_prompts.csv" "Tyler Edlin" "tyler_edlin.log"
should_run "Thomas Kinkade" && run_gen 4 6 "Thomas Kinkade" "concept_ablation-Thomas_Kinkade" "$ROOT/data/short_niche_art_prompts.csv" "Thomas Kinkade" "thomas_kinkade.log"
should_run "Kilian Eng" && run_gen 5 6 "Kilian Eng" "concept_ablation-Kilian_Eng" "$ROOT/data/short_niche_art_prompts.csv" "Kilian Eng" "kilian_eng.log"
should_run "Ajin: Demi Human" && run_gen 6 6 "Ajin: Demi Human" "concept_ablation-Ajin_Demi_Human" "$ROOT/data/short_niche_art_prompts.csv" "Ajin: Demi Human" "ajin.log"

echo ""
echo "Concept Ablation CompVis image generation complete."
