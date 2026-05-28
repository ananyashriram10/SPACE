#!/bin/bash
# Train/evaluate the Van Gogh SPACE-v2R rescue sweep against cached Vanilla + ESD-x outputs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/use_venv.sh"
source "$ROOT/baselines/progress.sh"

DEVICE="${DEVICE:-cuda:0}"
ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"
BASELINE_DIR="${BASELINE_DIR:-results/baseline/vangogh}"
ESD_DIR="${ESD_DIR:-results/erased/vangogh}"
SAVE_DIR="${SAVE_DIR:-space-models/sd-v2r}"
EXPECTED_COUNT="${EXPECTED_COUNT:-50}"
FORCE_SPACE_V2R_REBUILD="${FORCE_SPACE_V2R_REBUILD:-0}"
FORCE_SPACE_V2R_IMAGES="${FORCE_SPACE_V2R_IMAGES:-0}"
SPACE_DEBUG_STEPS="${SPACE_DEBUG_STEPS:-0}"

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "SPACE-v2R sweep is currently scoped to Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/$SAVE_DIR" "$ROOT/results/provenance/space_v2r"

count_pngs() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d " "
}

validate_inputs() {
  "$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

expected = int("$EXPECTED_COUNT")
csv_path = Path("$TARGET_PROMPTS_PATH")
df = pd.read_csv(csv_path)
if "artist" in df.columns:
    df = df[df["artist"].astype(str).str.strip().str.lower() == "$TARGET_ARTIST_FILTER".lower()]
rows = len(df)
if rows != expected:
    raise SystemExit(f"{csv_path} has {rows} Van Gogh rows; expected {expected}")
for label, folder in [("Vanilla", "$BASELINE_DIR"), ("ESD-x", "$ESD_DIR")]:
    n = len(list(Path(folder).glob("*.png")))
    if n != expected:
        raise SystemExit(f"{label} folder {folder} has {n} PNGs; expected {expected}")
print(f"Validated {expected} prompts plus cached Vanilla and ESD-x outputs.")
PY
}

train_variant() {
  local variant="$1"
  local erase_scale="$2"
  local residual_mix="$3"
  local topk="$4"
  local lr="$5"
  local stage1="$6"
  local stage2="$7"
  local exp_name="Van_Gogh_${variant}"
  local ckpt="$ROOT/$SAVE_DIR/space-${exp_name}.safetensors"

  if [ "$FORCE_SPACE_V2R_REBUILD" = "1" ] && [ -f "$ckpt" ]; then
    rm -f "$ckpt"
  fi
  if [ -f "$ckpt" ]; then
    echo "Skipping training for $variant; checkpoint already exists: $ckpt"
    return
  fi

  local debug_arg=""
  if [ "$SPACE_DEBUG_STEPS" != "0" ]; then
    debug_arg="--debug_steps $SPACE_DEBUG_STEPS"
  fi

  run_with_progress "SPACE-v2R training $variant" bash -c "
    '$PYTHON_BIN' '$ROOT/space_v2/space_sd_v2.py' \
      --erase_concept 'Vincent van Gogh' \
      --target_prompts_path '$TARGET_PROMPTS_PATH' \
      --target_artist_filter '$TARGET_ARTIST_FILTER' \
      --preserve_prompts_path data/coco_30k.csv \
      --save_path '$SAVE_DIR' \
      --exp_name '$exp_name' \
      --erase_scale '$erase_scale' \
      --residual_mix '$residual_mix' \
      --alpha_art 0.30 \
      --saliency_topk_blocks '$topk' \
      --lr '$lr' \
      --stage1_steps '$stage1' \
      --stage2_steps '$stage2' \
      --lora_target_layers attn2 \
      --trajectory_bands 0.2,0.5,0.8 \
      --device '$DEVICE' \
      $debug_arg \
      2>&1 | tee '$ROOT/logs/space_v2r_train_${variant}.log'
  "
}

generate_variant() {
  local variant="$1"
  local exp_name="Van_Gogh_${variant}"
  local out_root="results/space_${variant}"
  local out_dir="$ROOT/$out_root/space-Van_Gogh"
  local count
  count="$(count_pngs "$out_dir")"

  if [ "$FORCE_SPACE_V2R_IMAGES" = "1" ] && [ -d "$out_dir" ]; then
    rm -rf "$out_dir"
    count=0
  fi
  if [ "$count" = "$EXPECTED_COUNT" ]; then
    echo "Skipping image generation for $variant; found $count PNGs in $out_dir"
    return
  fi

  run_with_progress "SPACE-v2R images $variant" env \
    ONLY_ARTIST="$ONLY_ARTIST" \
    DEVICE="$DEVICE" \
    WEIGHTS_DIR="$SAVE_DIR" \
    OUT_DIR="$out_root" \
    EXP_NAME="$exp_name" \
    OUTPUT_MODEL_NAME="space-Van_Gogh" \
    TARGET_PROMPTS_PATH="$TARGET_PROMPTS_PATH" \
    TARGET_ARTIST_FILTER="$TARGET_ARTIST_FILTER" \
    NUM_SAMPLES=1 \
    bash "$ROOT/space_v2/run_space_v2r_images.sh"

  count="$(count_pngs "$out_dir")"
  if [ "$count" != "$EXPECTED_COUNT" ]; then
    echo "SPACE-v2R image validation failed for $variant: $out_dir has $count PNGs; expected $EXPECTED_COUNT"
    exit 1
  fi
}

write_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

import pandas as pd

metrics_path = Path("results/evaluation/metrics.csv")
summary_path = Path("results/evaluation/space_v2r_sweep_summary.md")
summary_json_path = Path("results/evaluation/space_v2r_sweep_summary.json")
df = pd.read_csv(metrics_path)
rows = df[(df["artist"] == "Van Gogh") & (df["method"].astype(str).str.contains("SPACE-v2R", regex=False))].copy()
payload = {"status": "ok", "candidates": []}
if not rows.empty:
    rows["passes_erasure"] = rows["style_target_rate"] <= 0.32
    candidates = rows.to_dict(orient="records")
    passing = rows[rows["passes_erasure"]]
    chooser = passing if not passing.empty else rows
    sort_cols = ["lpips", "dino_similarity", "fid"]
    ascending = [True, False, True]
    best = chooser.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].to_dict()
    payload["candidates"] = candidates
    payload["selected"] = best
else:
    payload["status"] = "no_space_v2r_rows"

summary_json_path.write_text(json.dumps(payload, indent=2) + "\n")
lines = [
    "# SPACE-v2R Van Gogh Sweep",
    "",
    "Compared cached Vanilla SD v1.4 and ESD-x outputs against SPACE-v2R candidates.",
    "",
    "Primary selection rule: first require `style_target_rate <= 0.32`, then choose the best preservation tradeoff.",
    "",
]
if payload.get("selected"):
    selected = payload["selected"]
    lines.extend(
        [
            "## Selected Candidate",
            "",
            f"- Method: `{selected.get('method')}`",
            f"- style_target_rate: `{selected.get('style_target_rate')}`",
            f"- style_drop: `{selected.get('style_drop')}`",
            f"- clip_drop: `{selected.get('clip_drop')}`",
            f"- LPIPS: `{selected.get('lpips')}`",
            f"- FID: `{selected.get('fid')}`",
            f"- DINO similarity: `{selected.get('dino_similarity')}`",
            "",
        ]
    )
lines.extend(
    [
        "## Artifacts",
        "",
        "- Metrics: `results/evaluation/metrics.csv`",
        "- Table: `results/evaluation/ablation_table.tex`",
        "- Bars: `results/evaluation/comparison_bars.png`",
        "- Grid: `results/comparisons/van_gogh.png`",
        "- Summary JSON: `results/evaluation/space_v2r_sweep_summary.json`",
        "",
    ]
)
summary_path.write_text("\n".join(lines))
print(summary_path.read_text())
PY
}

progress_banner "SPACE-v2R Van Gogh rescue sweep"
progress_step 1 6 "validate cached inputs"
validate_inputs

variants=(
  "v2r_e125|1.25|0.70|0.25|1e-4|300|100"
  "v2r_e175|1.75|0.70|0.25|1e-4|300|100"
  "v2r_e225|2.25|0.70|0.25|1e-4|300|100"
  "v2r_e175_top35|1.75|0.70|0.35|1e-4|375|125"
)

if [ -n "${SPACE_V2R_VARIANTS:-}" ]; then
  requested=",${SPACE_V2R_VARIANTS// /},"
  filtered=()
  for spec in "${variants[@]}"; do
    IFS="|" read -r variant _erase_scale _residual_mix _topk _lr _stage1 _stage2 <<< "$spec"
    if [[ "$requested" == *",$variant,"* ]]; then
      filtered+=("$spec")
    fi
  done
  if [ "${#filtered[@]}" -eq 0 ]; then
    echo "No SPACE-v2R variants matched SPACE_V2R_VARIANTS=$SPACE_V2R_VARIANTS"
    echo "Available variants: v2r_e125, v2r_e175, v2r_e225, v2r_e175_top35"
    exit 1
  fi
  variants=("${filtered[@]}")
fi

method_keys="esd"
grid_methods="ESD-x"
for spec in "${variants[@]}"; do
  IFS="|" read -r variant _erase_scale _residual_mix _topk _lr _stage1 _stage2 <<< "$spec"
  method_keys="$method_keys,space_$variant"
  case "$variant" in
    v2r_e125) grid_methods="$grid_methods,v2R e125" ;;
    v2r_e175) grid_methods="$grid_methods,v2R e175" ;;
    v2r_e225) grid_methods="$grid_methods,v2R e225" ;;
    v2r_e175_top35) grid_methods="$grid_methods,v2R top35" ;;
  esac
done

progress_step 2 6 "train SPACE-v2R candidates"
for spec in "${variants[@]}"; do
  IFS="|" read -r variant erase_scale residual_mix topk lr stage1 stage2 <<< "$spec"
  train_variant "$variant" "$erase_scale" "$residual_mix" "$topk" "$lr" "$stage1" "$stage2"
done

progress_step 3 6 "generate SPACE-v2R images"
for spec in "${variants[@]}"; do
  IFS="|" read -r variant _erase_scale _residual_mix _topk _lr _stage1 _stage2 <<< "$spec"
  generate_variant "$variant"
done

progress_step 4 6 "evaluate ESD-x vs SPACE-v2R"
rm -f "$ROOT/results/evaluation/metrics_cache_v5_van_gogh.json"
run_with_progress "evaluate SPACE-v2R sweep" "$PYTHON_BIN" "$ROOT/evaluate.py" \
  --only-artist "Van Gogh" \
  --method-keys "$method_keys"

progress_step 5 6 "comparison grid"
run_with_progress "grid SPACE-v2R sweep" "$PYTHON_BIN" "$ROOT/make_comparison_grids.py" \
  --only-artist "Van Gogh" \
  --methods "$grid_methods"

progress_step 6 6 "write sweep summary"
write_summary

echo ""
echo "SPACE-v2R sweep complete."
echo "Metrics: results/evaluation/metrics.csv"
echo "Grid: results/comparisons/van_gogh.png"
echo "Summary: results/evaluation/space_v2r_sweep_summary.md"
