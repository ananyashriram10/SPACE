#!/bin/bash
# Generate only missing assets required by run_fair_vangogh_eval.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
source "$ROOT/baselines/progress.sh"

DEVICE="${DEVICE:-cuda:0}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-50}"
PROMPT_SAMPLES="${PROMPT_SAMPLES:-5}"
EXPECTED_PAIRS="$((EXPECTED_PROMPTS * PROMPT_SAMPLES))"
FID_IMAGES="${FID_IMAGES:-2000}"
FID_STEPS="${FID_STEPS:-20}"
FID_CFG="${FID_CFG:-7.5}"
PROMPT_STEPS="${PROMPT_STEPS:-20}"
PROMPT_CFG="${PROMPT_CFG:-7.5}"
ESD_PROMPT_STEPS="${ESD_PROMPT_STEPS:-50}"
FORCE_REGEN_CA="${FORCE_REGEN_CA:-0}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-0}"

VANGOGH_CSV="$ROOT/data/vangogh_prompts.csv"
COCO_CSV="$ROOT/data/coco_30k.csv"
ESD_PT="$ROOT/esd-weights/art/diffusers-VanGogh-ESDx1-UNET.pt"
UCE_CKPT="$ROOT/baseline-models/uce/uce-Van_Gogh.safetensors"
SPACE_V3_CKPT="$ROOT/space-models/sd-v3/space-Van_Gogh_v3.safetensors"

CA_ROOT="$ROOT/baselines/external/concept-ablation"
CA_DIR="$CA_ROOT/compvis"
CA_ASSETS="$ROOT/baseline-assets/concept_ablation/pretrained_models"
CA_CKPT="$CA_ASSETS/sd-v1-4.ckpt"
CA_DELTA="$ROOT/baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt"
CA_CACHE="$ROOT/baselines/cache/official_prompts/concept_ablation_compvis"
CA_PROMPT_TXT="$CA_CACHE/concept_ablation-Van_Gogh.txt"
CA_PROMPT_CSV="$CA_CACHE/concept_ablation-Van_Gogh.csv"
CA_MANIFEST="$CA_CACHE/concept_ablation-Van_Gogh.manifest.json"
CA_OUTDIR="$ROOT/results/concept_ablation_compvis/concept_ablation-Van_Gogh"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/results/fid"

count_pngs() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d " "
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "Missing $label: $path"
    exit 1
  fi
}

resolve_space_v1_ckpt() {
  local candidates=(
    "$ROOT/space-models/sd/space-Van_Gogh.safetensors"
    "/workspace/space-claude-implementation/space-models/sd/space-Van_Gogh.safetensors"
    "/workspace/space-checkpoints/van_gogh_space_v1/space-Van_Gogh.safetensors"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "Could not find SPACE-v1 checkpoint. Checked:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

validate_prompt_csv() {
  "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$VANGOGH_CSV")
if "artist" in df.columns:
    df = df[df["artist"].astype(str).str.strip().str.lower() == "vincent van gogh"]
n = len(df)
if n != int("$EXPECTED_PROMPTS"):
    raise SystemExit(f"{'$VANGOGH_CSV'} has {n} Van Gogh prompts; expected {'$EXPECTED_PROMPTS'}")
print(f"OK: {n} Van Gogh prompts")
PY
}

count_case_pairs() {
  local dir="$1"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import pandas as pd
df = pd.read_csv("$VANGOGH_CSV")
if "artist" in df.columns:
    df = df[df["artist"].astype(str).str.strip().str.lower() == "vincent van gogh"]
folder = Path("$dir")
count = 0
for case in df["case_number"].astype(int).tolist():
    for idx in range(int("$PROMPT_SAMPLES")):
        if (folder / f"{case}_{idx}.png").exists():
            count += 1
print(count)
PY
}

generate_case_method_if_needed() {
  local label="$1"
  local dir="$2"
  shift 2
  local count
  count="$(count_case_pairs "$dir")"
  echo "[$(progress_now)] $label prompt pairs: $count/$EXPECTED_PAIRS"
  if [ "$count" = "$EXPECTED_PAIRS" ]; then
    return
  fi
  run_with_progress "$label prompt images" "$@"
  count="$(count_case_pairs "$dir")"
  echo "[$(progress_now)] $label prompt pairs after generation: $count/$EXPECTED_PAIRS"
  if [ "$count" != "$EXPECTED_PAIRS" ]; then
    echo "$label still has $count/$EXPECTED_PAIRS prompt-pair images after generation."
    exit 1
  fi
}

generate_fid_if_needed() {
  local label="$1"
  local dir="$2"
  local ckpt="${3:-}"
  local count
  count="$(count_pngs "$dir")"
  echo "[$(progress_now)] $label FID images: $count/$FID_IMAGES"
  if [ "$count" = "$FID_IMAGES" ]; then
    return
  fi
  mkdir -p "$dir"
  local cmd=(
    "$PYTHON_BIN" "$ROOT/evalscripts/generate_fid_samples.py"
    --base_model "CompVis/stable-diffusion-v1-4"
    --prompts_path "$COCO_CSV"
    --save_path "$dir"
    --n_images "$FID_IMAGES"
    --device "$DEVICE"
    --num_inference_steps "$FID_STEPS"
    --guidance_scale "$FID_CFG"
  )
  if [ -n "$ckpt" ]; then
    cmd+=(--esd_path "$ckpt")
  fi
  run_with_progress "$label COCO-FID images" "${cmd[@]}"
  count="$(count_pngs "$dir")"
  echo "[$(progress_now)] $label FID images after generation: $count/$FID_IMAGES"
  if [ "$count" != "$FID_IMAGES" ]; then
    echo "$label still has $count/$FID_IMAGES FID images after generation."
    exit 1
  fi
}

generate_ca_if_needed() {
  local count
  count="$(count_pngs "$CA_OUTDIR/samples")"
  echo "[$(progress_now)] Concept Ablation prompt images: $count/$EXPECTED_PAIRS"
  if [ "$count" = "$EXPECTED_PAIRS" ] && [ "$FORCE_REGEN_CA" != "1" ]; then
    return
  fi

  bash "$ROOT/baselines/setup_concept_ablation_compvis.sh"
  require_file "$CA_DIR/sample.py" "official Concept Ablation sample.py"
  require_file "$CA_CKPT" "Concept Ablation SD v1.4 checkpoint"
  require_file "$CA_DELTA" "official Concept Ablation Van Gogh delta"

  "$PYTHON_BIN" "$ROOT/baselines/prepare_official_prompts.py" \
    --input_csv "$VANGOGH_CSV" \
    --artist "Van Gogh" \
    --out_csv "$CA_PROMPT_CSV" \
    --out_txt "$CA_PROMPT_TXT" \
    --manifest "$CA_MANIFEST"

  if [ "$FORCE_REGEN_CA" = "1" ]; then
    echo "[$(progress_now)] FORCE_REGEN_CA=1; removing CA samples before regeneration."
    rm -rf "$CA_OUTDIR/samples"
  fi
  mkdir -p "$CA_OUTDIR"

  run_with_progress "Concept Ablation prompt images" bash -c "
    cd '$CA_DIR'
    '$PYTHON_BIN' sample.py \
      --ckpt '$CA_CKPT' \
      --delta_ckpt '$CA_DELTA' \
      --from-file '$CA_PROMPT_TXT' \
      --ddim_steps 100 \
      --scale 6.0 \
      --ddim_eta 1.0 \
      --outdir '$CA_OUTDIR' \
      --n_samples 1 \
      --n_copies '$PROMPT_SAMPLES' \
      --skip_grid \
      --metadata \
      --seed 42
  "

  count="$(count_pngs "$CA_OUTDIR/samples")"
  echo "[$(progress_now)] Concept Ablation prompt images after generation: $count/$EXPECTED_PAIRS"
  if [ "$count" != "$EXPECTED_PAIRS" ]; then
    echo "Concept Ablation still has $count/$EXPECTED_PAIRS prompt images."
    echo "If the folder has a corrupt ordered layout, rerun with FORCE_REGEN_CA=1."
    exit 1
  fi
}

progress_banner "Generate missing fair Van Gogh assets"
validate_prompt_csv

require_file "$ESD_PT" "ESD-x Van Gogh checkpoint"
require_file "$UCE_CKPT" "UCE Van Gogh checkpoint"
require_file "$SPACE_V3_CKPT" "SPACE-v3 Van Gogh checkpoint"
SPACE_V1_CKPT="$(resolve_space_v1_ckpt)"
echo "SPACE-v1 checkpoint: $SPACE_V1_CKPT"

progress_step 1 11 "Vanilla prompt-pair images"
generate_case_method_if_needed \
  "Vanilla" \
  "$ROOT/results/baseline/vangogh" \
  "$PYTHON_BIN" "$ROOT/generate_esd_old.py" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/baseline/vangogh" \
    --num_samples "$PROMPT_SAMPLES" \
    --num_inference_steps "$ESD_PROMPT_STEPS" \
    --guidance_scale "$PROMPT_CFG"

progress_step 2 11 "ESD-x prompt-pair images"
generate_case_method_if_needed \
  "ESD-x" \
  "$ROOT/results/erased/vangogh" \
  "$PYTHON_BIN" "$ROOT/generate_esd_old.py" \
    --esd_path "$ESD_PT" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/erased/vangogh" \
    --num_samples "$PROMPT_SAMPLES" \
    --num_inference_steps "$ESD_PROMPT_STEPS" \
    --guidance_scale "$PROMPT_CFG"

progress_step 3 11 "UCE prompt-pair images"
generate_case_method_if_needed \
  "UCE" \
  "$ROOT/results/uce/uce-Van_Gogh" \
  "$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
    --base_model "CompVis/stable-diffusion-v1-4" \
    --esd_path "$UCE_CKPT" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/uce" \
    --model_name_override "uce-Van_Gogh" \
    --num_samples "$PROMPT_SAMPLES" \
    --num_inference_steps "$PROMPT_STEPS" \
    --guidance_scale "$PROMPT_CFG" \
    --device "$DEVICE"

progress_step 4 11 "SPACE-v1 prompt-pair images"
generate_case_method_if_needed \
  "SPACE-v1" \
  "$ROOT/results/space/space-Van_Gogh" \
  "$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
    --base_model "CompVis/stable-diffusion-v1-4" \
    --esd_path "$SPACE_V1_CKPT" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/space" \
    --model_name_override "space-Van_Gogh" \
    --num_samples "$PROMPT_SAMPLES" \
    --num_inference_steps "$PROMPT_STEPS" \
    --guidance_scale "$PROMPT_CFG" \
    --device "$DEVICE"

progress_step 5 11 "SPACE-v3 prompt-pair images"
generate_case_method_if_needed \
  "SPACE-v3" \
  "$ROOT/results/space_v3/space-Van_Gogh" \
  "$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
    --base_model "CompVis/stable-diffusion-v1-4" \
    --esd_path "$SPACE_V3_CKPT" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/space_v3" \
    --model_name_override "space-Van_Gogh" \
    --num_samples "$PROMPT_SAMPLES" \
    --num_inference_steps "$PROMPT_STEPS" \
    --guidance_scale "$PROMPT_CFG" \
    --device "$DEVICE"

progress_step 6 11 "Concept Ablation prompt-pair images"
generate_ca_if_needed

progress_step 7 11 "Vanilla COCO-FID images"
generate_fid_if_needed "Vanilla" "$ROOT/results/fid/baseline"

progress_step 8 11 "ESD-x COCO-FID images"
generate_fid_if_needed "ESD-x" "$ROOT/results/fid/erased" "$ESD_PT"

progress_step 9 11 "UCE COCO-FID images"
generate_fid_if_needed "UCE" "$ROOT/results/fid/uce" "$UCE_CKPT"

progress_step 10 11 "SPACE-v1 COCO-FID images"
generate_fid_if_needed "SPACE-v1" "$ROOT/results/fid/space" "$SPACE_V1_CKPT"

progress_step 11 11 "SPACE-v3 COCO-FID images"
generate_fid_if_needed "SPACE-v3" "$ROOT/results/fid/space_v3" "$SPACE_V3_CKPT"

echo ""
echo "All required fair Van Gogh assets are present."
echo "Prompt pairs per method: $EXPECTED_PAIRS"
echo "COCO-FID images per scored FID method: $FID_IMAGES"
echo "CA FID/KID remains excluded."

if [ "$RUN_EVAL_AFTER" = "1" ]; then
  echo ""
  echo "RUN_EVAL_AFTER=1; running fair evaluation now."
  bash "$ROOT/run_fair_vangogh_eval.sh"
else
  echo ""
  echo "Next:"
  echo "  bash run_fair_vangogh_eval.sh"
fi
