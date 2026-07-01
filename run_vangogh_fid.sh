#!/bin/bash
# Train SPACE for Van Gogh and compute FID-COCO-30k.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"

DEVICE="${DEVICE:-cuda:0}"
FID_SAVE_PATH="${FID_SAVE_PATH:-/tmp/fid_space_vangogh}"
CHECKPOINT="$ROOT/space-models/sd/space-Van_Gogh.safetensors"

mkdir -p "$ROOT/logs"

# ── Step 1: Train SPACE for Van Gogh ────────────────────────────
if [ -f "$CHECKPOINT" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
  echo "[1/2] Checkpoint exists, skipping training: $CHECKPOINT"
else
  echo "[1/2] Training SPACE for Van Gogh..."
  bash "$ROOT/run_space_training.sh"
  echo "[1/2] SPACE training DONE"
fi

# ── Step 2: Generate 30k FID images ─────────────────────────────
echo "[2/2] Generating 30k FID samples..."
"$PYTHON_BIN" "$ROOT/evalscripts/generate_fid_samples.py" \
  --base_model "CompVis/stable-diffusion-v1-4" \
  --esd_path "$CHECKPOINT" \
  --prompts_path "$ROOT/data/coco_30k.csv" \
  --save_path "$FID_SAVE_PATH" \
  --n_images 30000 \
  --device "$DEVICE" \
  --guidance_scale 7.5 \
  --num_inference_steps 50

echo "[2/2] FID images DONE"
echo ""
echo "Now compute FID:"
echo "  python3 -c \"from cleanfid import fid; score = fid.compute_fid('$FID_SAVE_PATH/', '/tmp/coco_30k_ref/', device='cuda', num_workers=0); print('FID vs COCO:', score)\""
