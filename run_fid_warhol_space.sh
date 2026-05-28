#!/bin/bash
# Generate 30k FID samples for the SPACE Warhol checkpoint using COCO prompts.
# FID is computed for SPACE only — ESD-x/UCE/CA values are taken from their papers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
CHECKPOINT="${CHECKPOINT:-$ROOT/space-models/sd/space-Andy_Warhol.safetensors}"
SAVE_PATH="${SAVE_PATH:-$ROOT/results/fid/space-Andy_Warhol}"
FID_N="${FID_N:-30000}"
DEVICE="${DEVICE:-cuda:0}"
"$PYTHON_BIN" "$ROOT/evalscripts/generate_fid_samples.py" \
  --base_model "CompVis/stable-diffusion-v1-4" \
  --esd_path "$CHECKPOINT" \
  --prompts_path "$ROOT/data/coco_30k.csv" \
  --save_path "$SAVE_PATH" \
  --n_images "$FID_N" \
  --device "$DEVICE" \
  --guidance_scale 7.5 \
  --num_inference_steps 20
