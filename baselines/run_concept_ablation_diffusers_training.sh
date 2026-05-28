#!/bin/bash
# Secondary smoke-test path: official Concept Ablation diffusers implementation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CA_ROOT="$ROOT/baselines/external/concept-ablation"
CA_DIR="$CA_ROOT/diffusers"
OUT_ROOT="$ROOT/baseline-models/concept_ablation_diffusers"
DATA_ROOT="$ROOT/baselines/cache/concept_ablation_diffusers"
LOGS="$ROOT/logs"
PROVENANCE="$ROOT/results/provenance/concept_ablation_diffusers"

MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
STEPS="${STEPS:-200}"
BATCH="${BATCH:-4}"
LR="${LR:-2e-6}"
XFORMERS_FLAG="${XFORMERS_FLAG:---enable_xformers_memory_efficient_attention}"
ONLY_ARTIST="${ONLY_ARTIST:-}"

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

if [ ! -f "$CA_DIR/train.py" ]; then
  echo "Missing official Concept Ablation diffusers repo at $CA_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$DATA_ROOT/samples_painting" "$LOGS" "$PROVENANCE"

run_ca() {
  local idx="$1"
  local total="$2"
  local concept="$3"
  local exp="$4"
  local cmd

  echo ""
  echo "[$idx/$total] Concept Ablation diffusers smoke path: $concept"
  cmd="accelerate launch train.py --pretrained_model_name_or_path=$MODEL_NAME --output_dir=$OUT_ROOT/$exp --class_data_dir=$DATA_ROOT/samples_painting --class_prompt=painting --caption_target $concept --concept_type style --resolution=512 --train_batch_size=$BATCH --learning_rate=$LR --max_train_steps=$STEPS --scale_lr --hflip --noaug --parameter_group cross-attn $XFORMERS_FLAG"
  (
    cd "$CA_DIR"
    accelerate launch train.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --output_dir="$OUT_ROOT/$exp" \
      --class_data_dir="$DATA_ROOT/samples_painting" \
      --class_prompt="painting" \
      --caption_target "$concept" \
      --concept_type style \
      --resolution=512 \
      --train_batch_size="$BATCH" \
      --learning_rate="$LR" \
      --max_train_steps="$STEPS" \
      --scale_lr \
      --hflip \
      --noaug \
      --parameter_group cross-attn \
      $XFORMERS_FLAG
  ) 2>&1 | tee "$LOGS/concept_ablation_diffusers_train_$exp.log"

  python "$ROOT/baselines/record_provenance.py" \
    --method "Concept Ablation" \
    --artist "$concept" \
    --run_type "official-diffusers" \
    --repo_url "https://github.com/nupurkmr9/concept-ablation" \
    --repo_dir "$CA_ROOT" \
    --command "$cmd" \
    --outputs "$OUT_ROOT/$exp" \
    --out "$PROVENANCE/train_$exp.json"
}

should_run "Kelly McKernan" && run_ca 1 7 "Kelly McKernan" "concept_ablation-Kelly_McKernan"
should_run "Van Gogh" && run_ca 2 7 "Van Gogh" "concept_ablation-Van_Gogh"
should_run "Tyler Edlin" && run_ca 3 7 "Tyler Edlin" "concept_ablation-Tyler_Edlin"
should_run "Thomas Kinkade" && run_ca 4 7 "Thomas Kinkade" "concept_ablation-Thomas_Kinkade"
should_run "Kilian Eng" && run_ca 5 7 "Kilian Eng" "concept_ablation-Kilian_Eng"
should_run "Ajin: Demi Human" && run_ca 6 7 "Ajin: Demi Human" "concept_ablation-Ajin_Demi_Human"
should_run "Andy Warhol" && run_ca 7 7 "Andy Warhol" "concept_ablation-Andy_Warhol"
