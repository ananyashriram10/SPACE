#!/bin/bash
# Train SPACE-v3 for Van Gogh only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/use_venv.sh"

DEVICE="${DEVICE:-cuda:0}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"
SAVE_DIR="${SAVE_DIR:-space-models/sd-v3}"
EXP_NAME="${EXP_NAME:-Van_Gogh_v3}"
SPACE_DEBUG_STEPS="${SPACE_DEBUG_STEPS:-0}"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/$SAVE_DIR" "$ROOT/results/provenance/space_v3"

debug_arg=""
if [ "$SPACE_DEBUG_STEPS" != "0" ]; then
  debug_arg="--debug_steps $SPACE_DEBUG_STEPS"
fi

echo "============================================================"
echo "  SPACE-v3 Training"
echo "============================================================"
echo "Artist: Van Gogh"
echo "Erase concept: Vincent van Gogh"
echo "Target prompts: $TARGET_PROMPTS_PATH"
echo "Target artist filter: $TARGET_ARTIST_FILTER"
echo "Save dir: $SAVE_DIR"
echo "Objective: preserve better than UCE while matching SPACE-v1/UCE erasure"
echo ""

"$PYTHON_BIN" "$ROOT/space_v3/space_sd_v3.py" \
  --erase_concept "Vincent van Gogh" \
  --target_prompts_path "$TARGET_PROMPTS_PATH" \
  --target_artist_filter "$TARGET_ARTIST_FILTER" \
  --preserve_prompts_path data/coco_30k.csv \
  --save_path "$SAVE_DIR" \
  --exp_name "$EXP_NAME" \
  --erase_scale "${ERASE_SCALE:-1.20}" \
  --residual_mix "${RESIDUAL_MIX:-0.90}" \
  --alpha_art "${ALPHA_ART:-0.40}" \
  --alpha_img "${ALPHA_IMG:-0.25}" \
  --style_gate_mode "${STYLE_GATE_MODE:-residual_norm}" \
  --style_gate_min "${STYLE_GATE_MIN:-0.50}" \
  --style_gate_max "${STYLE_GATE_MAX:-1.10}" \
  --saliency_topk_blocks "${SALIENCY_TOPK_BLOCKS:-0.18}" \
  --lr "${SPACE_LR:-7e-5}" \
  --stage1_steps "${STAGE1_STEPS:-350}" \
  --stage2_steps "${STAGE2_STEPS:-150}" \
  --lambda_content "${LAMBDA_CONTENT:-1.25}" \
  --lambda_image "${LAMBDA_IMAGE:-1.0}" \
  --lambda_preserve "${LAMBDA_PRESERVE:-0.75}" \
  --lambda_fid_preserve "${LAMBDA_FID_PRESERVE:-2.0}" \
  --lambda_vulnerable "${LAMBDA_VULNERABLE:-0.50}" \
  --preserve_null_interval "${PRESERVE_NULL_INTERVAL:-10}" \
  --lora_target_layers attn2 \
  --trajectory_bands "${TRAJECTORY_BANDS:-0.25,0.40,0.55}" \
  --device "$DEVICE" \
  $debug_arg \
  2>&1 | tee "$ROOT/logs/space_v3_train_van_gogh.log"

echo ""
echo "SPACE-v3 training complete."
echo "Checkpoint: $SAVE_DIR/space-$EXP_NAME.safetensors"
