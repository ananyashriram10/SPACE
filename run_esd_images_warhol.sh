#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
CKPT="${CKPT:-$ROOT/esd-models/sd/esd-Andy_Warhol-from-Andy_Warhol-esdx.safetensors}"
DEVICE="${DEVICE:-cuda:0}"
"$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
  --base_model "CompVis/stable-diffusion-v1-4" \
  --esd_path "$CKPT" \
  --prompts_path "$ROOT/data/andy_warhol_prompts.csv" \
  --save_path "$ROOT/results/erased" \
  --device "$DEVICE" \
  --guidance_scale "${EVAL_GUIDANCE_SCALE:-7.5}" \
  --num_inference_steps "${EVAL_STEPS:-20}" \
  --num_samples "${NUM_SAMPLES:-5}" \
  --artist_filter "Andy Warhol" \
  --model_name_override "andy_warhol" \
  2>&1 | tee "$ROOT/logs/esd_images_andy_warhol.log"
