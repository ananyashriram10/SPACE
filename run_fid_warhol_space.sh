#!/bin/bash
# Generate 30k FID samples for the SPACE Warhol checkpoint using COCO prompts.
# FID is computed against real COCO val2014 images (30k subset), not a generated baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
CHECKPOINT="${CHECKPOINT:-$ROOT/space-models/sd/space-Andy_Warhol.safetensors}"
SPACE_SAVE_PATH="${SPACE_SAVE_PATH:-$ROOT/results/fid/andy_warhol/space}"
COCO_REF_PATH="${COCO_REF_PATH:-$ROOT/coco_30k_ref}"
FID_N="${FID_N:-30000}"
DEVICE="${DEVICE:-cuda:0}"

echo "Step 1/1: Generating SPACE erased model images..."
"$PYTHON_BIN" "$ROOT/evalscripts/generate_fid_samples.py" \
  --base_model "CompVis/stable-diffusion-v1-4" \
  --esd_path "$CHECKPOINT" \
  --prompts_path "$ROOT/data/coco_30k.csv" \
  --save_path "$SPACE_SAVE_PATH" \
  --n_images "$FID_N" \
  --device "$DEVICE" \
  --guidance_scale 7.5 \
  --num_inference_steps 50

echo "Images ready at $SPACE_SAVE_PATH"
echo "Compute FID against real COCO val2014 images with:"
echo "  python -m cleanfid.fid $SPACE_SAVE_PATH $COCO_REF_PATH --device cuda --num-workers 0"
