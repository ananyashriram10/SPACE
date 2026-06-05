#!/bin/bash
# Ablation study: train 6 SPACE variants for Van Gogh, each with one loss term zeroed.
# Full model (Van_Gogh) is assumed already trained.
# Runs all 6 sequentially on a single GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"

DEVICE="${DEVICE:-cuda:0}"
SAVE_DIR="${SAVE_DIR:-space-models/sd}"
mkdir -p "$ROOT/$SAVE_DIR" "$ROOT/logs"

BASE_ARGS=(
  "$PYTHON_BIN" "$ROOT/space_sd.py"
  --erase_concept "Vincent van Gogh"
  --target_prompts_path "data/vangogh_prompts.csv"
  --target_artist_filter "Vincent van Gogh"
  --preserve_prompts_path "data/coco_30k.csv"
  --neutral_art_template "a high quality painting of {content}"
  --neutral_image_template "a high quality image of {content}"
  --style_basis_rank 4
  --saliency_topk_blocks 0.25
  --trajectory_bands "0.2,0.5,0.8"
  --robust_prompt_mode "full"
  --vulnerable_preserve_k 3
  --lora_rank 8
  --stage1_steps 300
  --stage2_steps 100
  --lr 1e-4
  --guidance_scale 3.0
  --num_inference_steps 50
  --alpha_art 0.30
  --erase_scale 1.0
  --preserve_limit 256
  --save_path "$SAVE_DIR"
  --device "$DEVICE"
)

run_variant() {
  local exp_name="$1"
  shift
  echo "============================================================"
  echo "  Ablation: $exp_name"
  echo "============================================================"
  "${BASE_ARGS[@]}" --exp_name "$exp_name" "$@" \
    2>&1 | tee "$ROOT/logs/space_ablation_${exp_name,,}.log"
  echo "Done: $exp_name"
  echo ""
}

# 1. No style loss
run_variant "Van_Gogh_no_style" \
  --lambda_style 0.0 \
  --lambda_content 1.0 \
  --lambda_image 0.5 \
  --lambda_preserve 0.20 \
  --lambda_vulnerable 0.30 \
  --lambda_lora 1e-4

# 2. No content anchor
run_variant "Van_Gogh_no_content" \
  --lambda_style 0.8 \
  --lambda_content 0.0 \
  --lambda_image 0.5 \
  --lambda_preserve 0.20 \
  --lambda_vulnerable 0.30 \
  --lambda_lora 1e-4

# 3. No image anchor
run_variant "Van_Gogh_no_image" \
  --lambda_style 0.8 \
  --lambda_content 1.0 \
  --lambda_image 0.0 \
  --lambda_preserve 0.20 \
  --lambda_vulnerable 0.30 \
  --lambda_lora 1e-4

# 4. No dual anchor (both content and image zeroed)
run_variant "Van_Gogh_no_dual_anchor" \
  --lambda_style 0.8 \
  --lambda_content 0.0 \
  --lambda_image 0.0 \
  --lambda_preserve 0.20 \
  --lambda_vulnerable 0.30 \
  --lambda_lora 1e-4

# 5. No preservation (general + vulnerable both zeroed)
run_variant "Van_Gogh_no_preserve" \
  --lambda_style 0.8 \
  --lambda_content 1.0 \
  --lambda_image 0.5 \
  --lambda_preserve 0.0 \
  --lambda_vulnerable 0.0 \
  --lambda_lora 1e-4

# 6. No LoRA regularization
run_variant "Van_Gogh_no_lora_reg" \
  --lambda_style 0.8 \
  --lambda_content 1.0 \
  --lambda_image 0.5 \
  --lambda_preserve 0.20 \
  --lambda_vulnerable 0.30 \
  --lambda_lora 0.0

echo "All 6 ablation checkpoints saved to $SAVE_DIR/"
echo "Run evaluate.py on each to compare metrics."
