#!/bin/bash
# ============================================================
# setup_runpod.sh — One-shot environment setup for RunPod A40
# Run this once after SSHing in.
# ============================================================

set -e

echo "============================================================"
echo "  ESD Replication — RunPod Environment Setup"
echo "============================================================"

# 1. Install Python deps
echo ""
echo "[1/4] Installing Python dependencies..."
pip install --quiet --pre \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128

pip install --quiet \
  diffusers==0.21.4 \
  transformers==4.35.0 \
  accelerate==0.24.1 \
  safetensors \
  pandas \
  pillow \
  tqdm \
  huggingface_hub

echo "[1/4] Python deps installed."

# 2. Install aria2c for fast parallel downloads
echo ""
echo "[2/4] Installing aria2c..."
apt-get install -y -q aria2 2>/dev/null || \
  conda install -y -q aria2 -c conda-forge 2>/dev/null || \
  echo "  Warning: could not install aria2c, will fall back to wget"
echo "[2/4] aria2c ready."

# 3. HuggingFace login
echo ""
echo "[3/4] HuggingFace login..."
echo "  You need a token from https://huggingface.co/settings/tokens"
echo "  AND must accept the SD v1.4 license at:"
echo "  https://huggingface.co/CompVis/stable-diffusion-v1-4"
echo ""
huggingface-cli login

# 4. Pre-download SD v1.4 (so all 6 parallel jobs share the cache)
echo ""
echo "[4/4] Pre-caching SD v1.4 model weights..."
python3 -c "
from diffusers import StableDiffusionPipeline
import torch
print('  Downloading SD v1.4 to HF cache...')
pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    torch_dtype=torch.float16,
    safety_checker=None,
)
print('  SD v1.4 cached.')
del pipe
"
echo "[4/4] SD v1.4 cached."

echo ""
echo "============================================================"
echo "  Setup complete! Next steps:"
echo "    1. bash download_weights.sh    # ~19 GB, runs in parallel"
echo "    2. bash run_baseline.sh        # baseline images"
echo "    3. bash run_all_erased.sh      # all 6 erased models"
echo "============================================================"
