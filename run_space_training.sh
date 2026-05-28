#!/bin/bash
# Train SPACE-v1 for the current artist benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
mkdir -p "$ROOT/space-models/sd" "$ROOT/logs"

ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
DEVICE="${DEVICE:-cuda:0}"
SAVE_DIR="${SAVE_DIR:-space-models/sd}"
ROBUST_PROMPT_MODE="${ROBUST_PROMPT_MODE:-full}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"
ERASE_CONCEPT="${ERASE_CONCEPT:-Vincent van Gogh}"
EXP_NAME="${EXP_NAME:-Van_Gogh}"

echo "============================================================"
echo "  SPACE-v1 Training"
echo "============================================================"
echo "Artist: $ONLY_ARTIST"
echo "Erase concept: $ERASE_CONCEPT"
echo "Target prompts: $TARGET_PROMPTS_PATH"
echo "Target artist filter: $TARGET_ARTIST_FILTER"
echo "Save dir: $SAVE_DIR"
echo "Erase scale: ${ERASE_SCALE:-1.0}"
echo "Debug steps: ${SPACE_DEBUG_STEPS:-0}"
echo ""

cmd=(
  "$PYTHON_BIN" "$ROOT/space_sd.py"
  --erase_concept "$ERASE_CONCEPT" \
  --target_prompts_path "$TARGET_PROMPTS_PATH" \
  --target_artist_filter "$TARGET_ARTIST_FILTER" \
  --preserve_prompts_path data/coco_30k.csv \
  --neutral_art_template "a high quality painting of {content}" \
  --neutral_image_template "a high quality image of {content}" \
  --style_basis_rank "${STYLE_BASIS_RANK:-4}" \
  --saliency_topk_blocks "${SALIENCY_TOPK_BLOCKS:-0.25}" \
  --trajectory_bands "${TRAJECTORY_BANDS:-0.2,0.5,0.8}" \
  --robust_prompt_mode "$ROBUST_PROMPT_MODE" \
  --vulnerable_preserve_k "${VULNERABLE_PRESERVE_K:-3}" \
  --lora_rank "${LORA_RANK:-8}" \
  --stage1_steps "${STAGE1_STEPS:-300}" \
  --stage2_steps "${STAGE2_STEPS:-100}" \
  --lr "${LR:-1e-4}" \
  --guidance_scale "${GUIDANCE_SCALE:-3.0}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --alpha_art "${ALPHA_ART:-0.30}" \
  --erase_scale "${ERASE_SCALE:-1.0}" \
  --lambda_style "${LAMBDA_STYLE:-0.8}" \
  --lambda_content "${LAMBDA_CONTENT:-1.0}" \
  --lambda_image "${LAMBDA_IMAGE:-0.5}" \
  --lambda_preserve "${LAMBDA_PRESERVE:-0.20}" \
  --lambda_vulnerable "${LAMBDA_VULNERABLE:-0.30}" \
  --lambda_lora "${LAMBDA_LORA:-1e-4}" \
  --preserve_limit "${PRESERVE_LIMIT:-256}" \
  --save_path "$SAVE_DIR" \
  --exp_name "$EXP_NAME" \
  --device "$DEVICE"
)

if [ "${GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
  cmd+=(--gradient_checkpointing)
fi
if [ "${ALLOW_TF32:-0}" = "1" ]; then
  cmd+=(--allow_tf32)
fi
if [ "${SPACE_DEBUG_STEPS:-0}" != "0" ]; then
  cmd+=(--debug_steps "$SPACE_DEBUG_STEPS")
fi

"${cmd[@]}" 2>&1 | tee "$ROOT/logs/space_train_${EXP_NAME,,}.log"

echo ""
echo "SPACE training complete."
echo "Checkpoint: $SAVE_DIR/space-$EXP_NAME.safetensors"
