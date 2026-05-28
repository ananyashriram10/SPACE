#!/bin/bash
# End-to-end fair multi-artist SPACE benchmark.
#
# Default scope: Van Gogh + Tyler Edlin + Thomas Kinkade.
# Compares: ESD-x, UCE, SPACE-v1, SPACE-v2R e175, SPACE-v3.
# Excludes Concept Ablation by design.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
source "$ROOT/baselines/progress.sh"

DEVICE="${DEVICE:-cuda:0}"
ARTISTS="${ARTISTS:-Van Gogh,Tyler Edlin,Thomas Kinkade}"
export ARTISTS
EXPECTED_PAIRS="${EXPECTED_PAIRS:-500}"
FID_IMAGES="${FID_IMAGES:-2000}"
PROMPT_STEPS="${PROMPT_STEPS:-20}"
ESD_PROMPT_STEPS="${ESD_PROMPT_STEPS:-50}"
FID_STEPS="${FID_STEPS:-20}"
CFG="${CFG:-7.5}"
TRAIN_MISSING_UCE="${TRAIN_MISSING_UCE:-1}"
TRAIN_SPACE_V1="${TRAIN_SPACE_V1:-1}"
TRAIN_SPACE_V2R="${TRAIN_SPACE_V2R:-1}"
TRAIN_SPACE_V3="${TRAIN_SPACE_V3:-1}"
FORCE_SPACE_REBUILD="${FORCE_SPACE_REBUILD:-0}"
FORCE_IMAGES="${FORCE_IMAGES:-0}"
FORCE_FID="${FORCE_FID:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
METHOD_KEYS="${METHOD_KEYS:-esd,uce,space,space_v2r_e175,space_v3}"
FID_METHOD_KEYS="${FID_METHOD_KEYS:-esd,uce,space,space_v2r_e175,space_v3}"
GRID_METHODS="${GRID_METHODS:-ESD-x,UCE,SPACE,v2R e175,SPACE-v3}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/workspace/space-checkpoints/final_multi_artist}"

cd "$ROOT"
mkdir -p logs results/evaluation results/comparisons "$ARCHIVE_DIR"

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  bash "$ROOT/baselines/preflight_env.sh"
fi

slugify() {
  echo "$1" | sed 's/://g' | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g'
}

sanitize() {
  echo "$1" | sed 's/://g' | sed 's/ /_/g'
}

artist_config() {
  local artist="$1"
  case "$artist" in
    "Van Gogh")
      CSV="data/vangogh_prompts.csv"
      FILTER="Vincent van Gogh"
      ERASE_CONCEPT="Vincent van Gogh"
      SLUG="Van_Gogh"
      DIR_SLUG="vangogh"
      ESD_DIR="vangogh"
      ESD_WEIGHT="diffusers-VanGogh-ESDx1-UNET.pt"
      ;;
    "Tyler Edlin")
      CSV="data/short_niche_art_prompts.csv"
      FILTER="Tyler Edlin"
      ERASE_CONCEPT="Tyler Edlin"
      SLUG="Tyler_Edlin"
      DIR_SLUG="tyler_edlin"
      ESD_DIR="tyler_edlin"
      ESD_WEIGHT="diffusers-TylerEdlin-ESDx1-UNET.pt"
      ;;
    "Thomas Kinkade")
      CSV="data/short_niche_art_prompts.csv"
      FILTER="Thomas Kinkade"
      ERASE_CONCEPT="Thomas Kinkade"
      SLUG="Thomas_Kinkade"
      DIR_SLUG="thomas_kinkade"
      ESD_DIR="thomas_kinkade"
      ESD_WEIGHT="diffusers-ThomasKinkade-ESDx1-UNET.pt"
      ;;
    "Kilian Eng")
      CSV="data/short_niche_art_prompts.csv"
      FILTER="Kilian Eng"
      ERASE_CONCEPT="Kilian Eng"
      SLUG="Kilian_Eng"
      DIR_SLUG="kilian_eng"
      ESD_DIR="kilian_eng"
      ESD_WEIGHT="diffusers-KilianEng-ESDx1-UNET.pt"
      ;;
    "Ajin: Demi Human")
      CSV="data/short_niche_art_prompts.csv"
      FILTER="Ajin: Demi Human"
      ERASE_CONCEPT="Ajin: Demi Human"
      SLUG="Ajin_Demi_Human"
      DIR_SLUG="ajin_demi_human"
      ESD_DIR="ajin"
      ESD_WEIGHT="diffusers-AjinDemiHuman-ESDx1-UNET.pt"
      ;;
    *)
      echo "Unsupported artist for this final harness: $artist"
      echo "Supported: Van Gogh, Tyler Edlin, Thomas Kinkade, Kilian Eng, Ajin: Demi Human"
      exit 1
      ;;
  esac
}

resolve_file() {
  local label="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "Missing $label. Checked:" >&2
  printf '  %s\n' "$@" >&2
  return 1
}

count_pngs() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find -L "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d " "
}

prompt_count() {
  local csv="$1"
  local filter="$2"
  "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$csv")
if "$filter" and "artist" in df.columns:
    df = df[df["artist"].astype(str).str.strip().str.lower() == "$filter".strip().lower()]
print(len(df))
PY
}

require_expected_pairs() {
  local label="$1"
  local dir="$2"
  local expected="$3"
  local count
  count="$(count_pngs "$dir")"
  if [ "$count" != "$expected" ]; then
    echo "$label has $count PNGs in $dir; expected $expected"
    return 1
  fi
}

generate_case_images_if_needed() {
  local label="$1"
  local dir="$2"
  local expected="$3"
  shift 3
  local count
  count="$(count_pngs "$dir")"
  echo "[$(progress_now)] $label prompt images: $count/$expected"
  if [ "$FORCE_IMAGES" = "1" ] && [ -d "$dir" ]; then
    rm -rf "$dir"
    count=0
  fi
  if [ "$count" = "$expected" ]; then
    return
  fi
  run_with_progress "$label prompt images" "$@"
  require_expected_pairs "$label" "$dir" "$expected"
}

generate_fid_if_needed() {
  local label="$1"
  local dir="$2"
  shift 2
  local count
  count="$(count_pngs "$dir")"
  echo "[$(progress_now)] $label COCO-FID images: $count/$FID_IMAGES"
  if [ "$FORCE_FID" = "1" ] && [ -d "$dir" ]; then
    rm -rf "$dir"
    count=0
  fi
  if [ "$count" = "$FID_IMAGES" ]; then
    return
  fi
  mkdir -p "$dir"
  run_with_progress "$label COCO-FID images" "$@"
  require_expected_pairs "$label COCO-FID" "$dir" "$FID_IMAGES"
}

train_space_v1_if_needed() {
  local artist="$1" slug="$2" csv="$3" filter="$4" concept="$5"
  local ckpt="space-models/sd/space-$slug.safetensors"
  if [ "$FORCE_SPACE_REBUILD" = "1" ]; then
    rm -f "$ckpt"
  fi
  if [ "$TRAIN_SPACE_V1" != "1" ] || [ -f "$ckpt" ]; then
    return
  fi
  run_with_progress "SPACE-v1 training: $artist" "$PYTHON_BIN" space_sd.py \
    --erase_concept "$concept" \
    --target_prompts_path "$csv" \
    --target_artist_filter "$filter" \
    --preserve_prompts_path data/coco_30k.csv \
    --neutral_art_template "a high quality painting of {content}" \
    --neutral_image_template "a high quality image of {content}" \
    --style_basis_rank "${STYLE_BASIS_RANK:-4}" \
    --saliency_topk_blocks "${SALIENCY_TOPK_BLOCKS:-0.25}" \
    --trajectory_bands "${TRAJECTORY_BANDS:-0.2,0.5,0.8}" \
    --robust_prompt_mode "${ROBUST_PROMPT_MODE:-full}" \
    --vulnerable_preserve_k "${VULNERABLE_PRESERVE_K:-3}" \
    --lora_rank "${LORA_RANK:-8}" \
    --stage1_steps "${SPACE_V1_STAGE1_STEPS:-300}" \
    --stage2_steps "${SPACE_V1_STAGE2_STEPS:-100}" \
    --lr "${SPACE_V1_LR:-1e-4}" \
    --guidance_scale "${SPACE_TRAIN_GUIDANCE_SCALE:-3.0}" \
    --num_inference_steps "${SPACE_TRAIN_STEPS:-50}" \
    --alpha_art "${SPACE_V1_ALPHA_ART:-0.30}" \
    --erase_scale "${SPACE_V1_ERASE_SCALE:-1.0}" \
    --lambda_style "${SPACE_V1_LAMBDA_STYLE:-0.8}" \
    --lambda_content "${SPACE_V1_LAMBDA_CONTENT:-1.0}" \
    --lambda_image "${SPACE_V1_LAMBDA_IMAGE:-0.5}" \
    --lambda_preserve "${SPACE_V1_LAMBDA_PRESERVE:-0.20}" \
    --lambda_vulnerable "${SPACE_V1_LAMBDA_VULNERABLE:-0.30}" \
    --lambda_lora "${SPACE_LAMBDA_LORA:-1e-4}" \
    --preserve_limit "${PRESERVE_LIMIT:-256}" \
    --save_path space-models/sd \
    --exp_name "$slug" \
    --device "$DEVICE"
}

train_space_v2r_if_needed() {
  local artist="$1" slug="$2" csv="$3" filter="$4" concept="$5"
  local ckpt="space-models/sd-v2r/space-${slug}_v2r_e175.safetensors"
  if [ "$FORCE_SPACE_REBUILD" = "1" ]; then
    rm -f "$ckpt"
  fi
  if [ "$TRAIN_SPACE_V2R" != "1" ] || [ -f "$ckpt" ]; then
    return
  fi
  run_with_progress "SPACE-v2R e175 training: $artist" "$PYTHON_BIN" space_v2/space_sd_v2.py \
    --erase_concept "$concept" \
    --target_prompts_path "$csv" \
    --target_artist_filter "$filter" \
    --preserve_prompts_path data/coco_30k.csv \
    --save_path space-models/sd-v2r \
    --exp_name "${slug}_v2r_e175" \
    --erase_scale "${SPACE_V2R_ERASE_SCALE:-1.75}" \
    --residual_mix "${SPACE_V2R_RESIDUAL_MIX:-0.70}" \
    --alpha_art "${SPACE_V2R_ALPHA_ART:-0.30}" \
    --saliency_topk_blocks "${SPACE_V2R_TOPK:-0.25}" \
    --lr "${SPACE_V2R_LR:-1e-4}" \
    --stage1_steps "${SPACE_V2R_STAGE1_STEPS:-300}" \
    --stage2_steps "${SPACE_V2R_STAGE2_STEPS:-100}" \
    --lora_target_layers attn2 \
    --trajectory_bands "${SPACE_V2R_TRAJECTORY_BANDS:-0.2,0.5,0.8}" \
    --device "$DEVICE"
}

train_space_v3_if_needed() {
  local artist="$1" slug="$2" csv="$3" filter="$4" concept="$5"
  local ckpt="space-models/sd-v3/space-${slug}_v3.safetensors"
  if [ "$FORCE_SPACE_REBUILD" = "1" ]; then
    rm -f "$ckpt"
  fi
  if [ "$TRAIN_SPACE_V3" != "1" ] || [ -f "$ckpt" ]; then
    return
  fi
  run_with_progress "SPACE-v3 training: $artist" "$PYTHON_BIN" space_v3/space_sd_v3.py \
    --erase_concept "$concept" \
    --target_prompts_path "$csv" \
    --target_artist_filter "$filter" \
    --preserve_prompts_path data/coco_30k.csv \
    --save_path space-models/sd-v3 \
    --exp_name "${slug}_v3" \
    --erase_scale "${SPACE_V3_ERASE_SCALE:-1.20}" \
    --residual_mix "${SPACE_V3_RESIDUAL_MIX:-0.90}" \
    --alpha_art "${SPACE_V3_ALPHA_ART:-0.40}" \
    --alpha_img "${SPACE_V3_ALPHA_IMG:-0.25}" \
    --style_gate_mode "${STYLE_GATE_MODE:-residual_norm}" \
    --style_gate_min "${STYLE_GATE_MIN:-0.50}" \
    --style_gate_max "${STYLE_GATE_MAX:-1.10}" \
    --saliency_topk_blocks "${SPACE_V3_TOPK:-0.18}" \
    --lr "${SPACE_V3_LR:-7e-5}" \
    --stage1_steps "${SPACE_V3_STAGE1_STEPS:-350}" \
    --stage2_steps "${SPACE_V3_STAGE2_STEPS:-150}" \
    --lambda_content "${SPACE_V3_LAMBDA_CONTENT:-1.25}" \
    --lambda_image "${SPACE_V3_LAMBDA_IMAGE:-1.0}" \
    --lambda_preserve "${SPACE_V3_LAMBDA_PRESERVE:-0.75}" \
    --lambda_fid_preserve "${SPACE_V3_LAMBDA_FID_PRESERVE:-2.0}" \
    --lambda_vulnerable "${SPACE_V3_LAMBDA_VULNERABLE:-0.50}" \
    --preserve_null_interval "${PRESERVE_NULL_INTERVAL:-10}" \
    --lora_target_layers attn2 \
    --trajectory_bands "${SPACE_V3_TRAJECTORY_BANDS:-0.25,0.40,0.55}" \
    --device "$DEVICE"
}

generate_space_method_images() {
  local label="$1" ckpt="$2" csv="$3" filter="$4" out_root="$5" out_name="$6" samples="$7"
  generate_case_images_if_needed "$label" "$out_root/$out_name" "$EXPECTED_PAIRS" \
    "$PYTHON_BIN" evalscripts/generate-images.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$ckpt" \
      --prompts_path "$csv" \
      --save_path "$out_root" \
      --model_name_override "$out_name" \
      --num_samples "$samples" \
      --num_inference_steps "$PROMPT_STEPS" \
      --guidance_scale "$CFG" \
      --artist_filter "$filter" \
      --device "$DEVICE"
}

ensure_shared_baseline_fid() {
  local shared="results/fid/_shared_baseline"
  generate_fid_if_needed "Vanilla shared" "$shared" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --prompts_path data/coco_30k.csv \
      --save_path "$shared" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
}

link_baseline_fid_for_artist() {
  local artist_slug="$1"
  local dst="results/fid/$artist_slug/baseline"
  mkdir -p "results/fid/$artist_slug"
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    rm -rf "$dst"
  fi
  ln -s "../_shared_baseline" "$dst"
}

archive_checkpoint() {
  local ckpt="$1"
  local artist="$2"
  local method="$3"
  if [ -f "$ckpt" ]; then
    local out_dir="$ARCHIVE_DIR/checkpoints/$(slugify "$artist")/$method"
    mkdir -p "$out_dir"
    cp "$ckpt" "$out_dir/"
  fi
}

selected_artists=()
IFS=',' read -ra raw_artists <<< "$ARTISTS"
for raw in "${raw_artists[@]}"; do
  artist="$(echo "$raw" | sed 's/^ *//;s/ *$//')"
  [ -n "$artist" ] && selected_artists+=("$artist")
done

if [ "${#selected_artists[@]}" -eq 0 ]; then
  echo "No artists selected. Set ARTISTS='Van Gogh,Tyler Edlin,Thomas Kinkade'."
  exit 1
fi

progress_banner "Final multi-artist SPACE benchmark"
echo "Artists: ${selected_artists[*]}"
echo "Expected prompt pairs per artist/method: $EXPECTED_PAIRS"
echo "COCO-FID images per artist/method: $FID_IMAGES"
echo "Methods: $METHOD_KEYS"
echo "CA: excluded"

ensure_shared_baseline_fid

for artist in "${selected_artists[@]}"; do
  artist_config "$artist"
  artist_slug="$(slugify "$artist")"
  prompt_n="$(prompt_count "$CSV" "$FILTER")"
  if [ "$prompt_n" -le 0 ]; then
    echo "No prompts found for $artist in $CSV"
    exit 1
  fi
  if [ $((EXPECTED_PAIRS % prompt_n)) -ne 0 ]; then
    echo "$artist has $prompt_n prompts, which does not divide EXPECTED_PAIRS=$EXPECTED_PAIRS."
    echo "Choose a different EXPECTED_PAIRS or exclude this artist."
    exit 1
  fi
  samples=$((EXPECTED_PAIRS / prompt_n))

  progress_banner "Artist: $artist"
  echo "Prompts: $prompt_n"
  echo "Samples per prompt: $samples"
  echo "CSV: $CSV"
  echo "Artist filter: $FILTER"

  esd_ckpt="$(resolve_file "ESD-x checkpoint for $artist" \
    "$ROOT/esd-weights/art/$ESD_WEIGHT" \
    "$ROOT/../esd-weights/art/$ESD_WEIGHT" \
    "/workspace/space-claude-implementation/esd-weights/art/$ESD_WEIGHT")"

  generate_case_images_if_needed "Vanilla $artist" "results/baseline/$DIR_SLUG" "$EXPECTED_PAIRS" \
    "$PYTHON_BIN" generate_esd_old.py \
      --prompts_path "$CSV" \
      --artist "$FILTER" \
      --save_path "results/baseline/$DIR_SLUG" \
      --num_samples "$samples" \
      --num_inference_steps "$ESD_PROMPT_STEPS" \
      --guidance_scale "$CFG"

  generate_case_images_if_needed "ESD-x $artist" "results/erased/$ESD_DIR" "$EXPECTED_PAIRS" \
    "$PYTHON_BIN" generate_esd_old.py \
      --esd_path "$esd_ckpt" \
      --prompts_path "$CSV" \
      --artist "$FILTER" \
      --save_path "results/erased/$ESD_DIR" \
      --num_samples "$samples" \
      --num_inference_steps "$ESD_PROMPT_STEPS" \
      --guidance_scale "$CFG"

  uce_ckpt="baseline-models/uce/uce-$SLUG.safetensors"
  if [ ! -f "$uce_ckpt" ] && [ "$TRAIN_MISSING_UCE" = "1" ]; then
    ONLY_ARTIST="$artist" bash baselines/run_uce_training.sh
  fi
  uce_ckpt="$(resolve_file "UCE checkpoint for $artist" "$ROOT/$uce_ckpt" "/workspace/space-claude-implementation/baseline-models/uce/uce-$SLUG.safetensors")"
  generate_space_method_images "UCE $artist" "$uce_ckpt" "$CSV" "$FILTER" "results/uce" "uce-$SLUG" "$samples"

  train_space_v1_if_needed "$artist" "$SLUG" "$CSV" "$FILTER" "$ERASE_CONCEPT"
  space_ckpt="$(resolve_file "SPACE-v1 checkpoint for $artist" "$ROOT/space-models/sd/space-$SLUG.safetensors")"
  generate_space_method_images "SPACE-v1 $artist" "$space_ckpt" "$CSV" "$FILTER" "results/space" "space-$SLUG" "$samples"
  archive_checkpoint "$space_ckpt" "$artist" "space_v1"

  train_space_v2r_if_needed "$artist" "$SLUG" "$CSV" "$FILTER" "$ERASE_CONCEPT"
  space_v2r_ckpt="$(resolve_file "SPACE-v2R e175 checkpoint for $artist" "$ROOT/space-models/sd-v2r/space-${SLUG}_v2r_e175.safetensors")"
  generate_space_method_images "SPACE-v2R e175 $artist" "$space_v2r_ckpt" "$CSV" "$FILTER" "results/space_v2r_e175" "space-$SLUG" "$samples"
  archive_checkpoint "$space_v2r_ckpt" "$artist" "space_v2r_e175"

  train_space_v3_if_needed "$artist" "$SLUG" "$CSV" "$FILTER" "$ERASE_CONCEPT"
  space_v3_ckpt="$(resolve_file "SPACE-v3 checkpoint for $artist" "$ROOT/space-models/sd-v3/space-${SLUG}_v3.safetensors")"
  generate_space_method_images "SPACE-v3 $artist" "$space_v3_ckpt" "$CSV" "$FILTER" "results/space_v3" "space-$SLUG" "$samples"
  archive_checkpoint "$space_v3_ckpt" "$artist" "space_v3"

  link_baseline_fid_for_artist "$artist_slug"
  generate_fid_if_needed "ESD-x $artist" "results/fid/$artist_slug/erased" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$esd_ckpt" \
      --prompts_path data/coco_30k.csv \
      --save_path "results/fid/$artist_slug/erased" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
  generate_fid_if_needed "UCE $artist" "results/fid/$artist_slug/uce" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$uce_ckpt" \
      --prompts_path data/coco_30k.csv \
      --save_path "results/fid/$artist_slug/uce" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
  generate_fid_if_needed "SPACE-v1 $artist" "results/fid/$artist_slug/space" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$space_ckpt" \
      --prompts_path data/coco_30k.csv \
      --save_path "results/fid/$artist_slug/space" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
  generate_fid_if_needed "SPACE-v2R e175 $artist" "results/fid/$artist_slug/space_v2r_e175" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$space_v2r_ckpt" \
      --prompts_path data/coco_30k.csv \
      --save_path "results/fid/$artist_slug/space_v2r_e175" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
  generate_fid_if_needed "SPACE-v3 $artist" "results/fid/$artist_slug/space_v3" \
    "$PYTHON_BIN" evalscripts/generate_fid_samples.py \
      --base_model CompVis/stable-diffusion-v1-4 \
      --esd_path "$space_v3_ckpt" \
      --prompts_path data/coco_30k.csv \
      --save_path "results/fid/$artist_slug/space_v3" \
      --n_images "$FID_IMAGES" \
      --device "$DEVICE" \
      --num_inference_steps "$FID_STEPS" \
      --guidance_scale "$CFG"
done

artists_csv="$(IFS=,; echo "${selected_artists[*]}")"
cache_slug="$(echo "$artists_csv" | tr ', :' '___' | tr '[:upper:]' '[:lower:]')"
rm -f "results/evaluation/metrics_cache_v5_${cache_slug}.json"

progress_banner "Evaluate selected artists"
"$PYTHON_BIN" evaluate.py \
  --artists "$artists_csv" \
  --method-keys "$METHOD_KEYS" \
  --expected-pairs "$EXPECTED_PAIRS" \
  --fid-method-keys "$FID_METHOD_KEYS" \
  --disable-fid-fallback \
  --artist-fid-dirs

out_root="results/final_multi_artist"
mkdir -p "$out_root/per_artist_grids" "$out_root/evaluation"
cp results/evaluation/metrics.csv "$out_root/evaluation/metrics.csv"
cp results/evaluation/ablation_table.tex "$out_root/evaluation/ablation_table.tex"
cp results/evaluation/comparison_bars.png "$out_root/evaluation/comparison_bars.png"

for artist in "${selected_artists[@]}"; do
  "$PYTHON_BIN" make_comparison_grids.py --only-artist "$artist" --methods "$GRID_METHODS"
  grid_slug="$(slugify "$artist")"
  cp "results/comparisons/${grid_slug}.png" "$out_root/per_artist_grids/${grid_slug}.png"
  cp "results/comparisons/${grid_slug}.pdf" "$out_root/per_artist_grids/${grid_slug}.pdf"
done

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import pandas as pd

out = Path("results/final_multi_artist")
df = pd.read_csv(out / "evaluation/metrics.csv")
mean = df[df["artist"] == "MEAN"].copy()
artists = [x.strip() for x in os.environ.get("ARTISTS", "").split(",") if x.strip()]
lines = [
    "# Final Multi-Artist SPACE Benchmark",
    "",
    f"Artists: {', '.join(artists)}",
    "Methods: ESD-x, UCE, SPACE, SPACE-v2R e175, SPACE-v3",
    "Prompt-pair metrics: 500 matched pairs per artist/method.",
    "FID/KID: 2000 COCO prompts per artist/method, artist-scoped edited model folders.",
    "Concept Ablation: excluded.",
    "",
    "## Mean Rows",
    "",
    "```text",
    mean.to_string(index=False),
    "```",
    "",
    "## Artifacts",
    "",
    "- Metrics: `results/final_multi_artist/evaluation/metrics.csv`",
    "- Table: `results/final_multi_artist/evaluation/ablation_table.tex`",
    "- Overall bars: `results/final_multi_artist/evaluation/comparison_bars.png`",
    "- Per-artist grids: `results/final_multi_artist/per_artist_grids/`",
]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print((out / "summary.md").read_text())
PY

tar -czf "$ARCHIVE_DIR/final_multi_artist_eval_artifacts_$(date +%Y%m%d_%H%M%S).tar.gz" \
  results/final_multi_artist results/evaluation/metrics.csv results/evaluation/ablation_table.tex results/evaluation/comparison_bars.png

echo ""
echo "Final multi-artist benchmark complete."
echo "Summary: results/final_multi_artist/summary.md"
echo "Metrics: results/final_multi_artist/evaluation/metrics.csv"
echo "Overall chart: results/final_multi_artist/evaluation/comparison_bars.png"
echo "Per-artist grids: results/final_multi_artist/per_artist_grids/"
echo "Checkpoint/eval archive root: $ARCHIVE_DIR"
