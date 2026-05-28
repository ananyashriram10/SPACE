#!/bin/bash
# Prepare official Concept Ablation CompVis assets on RunPod.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CA_ROOT="$ROOT/baselines/external/concept-ablation"
CA_DIR="$CA_ROOT/compvis"
ASSETS="$ROOT/baseline-assets/concept_ablation/pretrained_models"
WEIGHTS="$ROOT/baseline-models/concept_ablation_compvis/official_weights"

SD_URL="https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
SSCD_URL="https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt"
CA_MODELS_URL="https://www.cs.cmu.edu/~concept-ablation/models"
source "$ROOT/baselines/progress.sh"

if [ "${BASELINE_REPOS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/setup_official_repos.sh"
fi

if [ ! -f "$CA_DIR/train.py" ]; then
  echo "Missing official Concept Ablation repo at $CA_ROOT"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

mkdir -p "$ASSETS" "$WEIGHTS"

download_if_missing() {
  local url="$1"
  local out="$2"
  local label="$3"
  if [ -f "$out" ]; then
    echo "[$(progress_now)] Already present: $out"
    return 0
  fi
  echo "[$(progress_now)] Downloading: $url"
  if command -v aria2c >/dev/null 2>&1; then
    run_with_progress "$label" aria2c -x 8 -s 8 -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  else
    run_with_progress "$label" wget --progress=bar:force:noscroll -O "$out" "$url"
  fi
}

progress_banner "Concept Ablation persistent asset setup"

# OpenAI CLIP is required by ldm/modules/encoders/modules.py but is not
# a standard pip package — it must be installed from source.
if ! python -c "import clip" >/dev/null 2>&1; then
  echo "[$(progress_now)] Installing OpenAI CLIP (required by CompVis concept-ablation)..."
  run_with_progress "install openai/CLIP" python -m pip install --progress-bar on \
    "git+https://github.com/openai/CLIP.git@main#egg=clip"
else
  echo "[$(progress_now)] OpenAI CLIP already installed."
fi
download_if_missing "$SD_URL" "$ASSETS/sd-v1-4.ckpt" "download SD v1.4 checkpoint"
download_if_missing "$SSCD_URL" "$ASSETS/sscd_imagenet_mixup.torchscript.pt" "download SSCD model"

# Official author-provided deltas are available only for a subset of artists.
# Our six-artist benchmark can use this Van Gogh delta directly; the remaining
# artists must be trained with the official CompVis code.
download_if_missing \
  "$CA_MODELS_URL/delta_vangogh_ablated.ckpt" \
  "$WEIGHTS/concept_ablation-Van_Gogh.ckpt" \
  "download official Concept Ablation Van Gogh delta"

cat <<EOF

Concept Ablation CompVis assets are ready.

Primary paper-faithful commands:
  bash baselines/run_concept_ablation_training.sh
  bash baselines/run_concept_ablation_images.sh

Note:
  The official README recommends a separate conda env:
  conda env create -f $CA_DIR/environment.yaml
  conda activate ablate

Persistent assets:
  $ASSETS
EOF
