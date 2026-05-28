#!/bin/bash
# Train/evaluate SPACE-v3 for Van Gogh against cached ESD-x, UCE, and SPACE-v1 outputs.

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
UCE_DIR="${UCE_DIR:-results/uce/uce-Van_Gogh}"
SPACE_V1_DIR="${SPACE_V1_DIR:-results/space/space-Van_Gogh}"
SAVE_DIR="${SAVE_DIR:-space-models/sd-v3}"
EXP_NAME="${EXP_NAME:-Van_Gogh_v3}"
SPACE_V3_OUT_ROOT="${SPACE_V3_OUT_ROOT:-results/space_v3}"
SPACE_V3_OUT_DIR="$ROOT/$SPACE_V3_OUT_ROOT/space-Van_Gogh"
EXPECTED_COUNT="${EXPECTED_COUNT:-50}"
FORCE_SPACE_V3_REBUILD="${FORCE_SPACE_V3_REBUILD:-0}"
FORCE_SPACE_V3_IMAGES="${FORCE_SPACE_V3_IMAGES:-0}"

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "SPACE-v3 benchmark is currently scoped to Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/$SAVE_DIR" "$ROOT/results/provenance/space_v3"

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
for label, folder in [
    ("Vanilla", "$BASELINE_DIR"),
    ("ESD-x", "$ESD_DIR"),
    ("UCE", "$UCE_DIR"),
    ("SPACE-v1", "$SPACE_V1_DIR"),
]:
    n = len(list(Path(folder).glob("*.png")))
    if n != expected:
        raise SystemExit(f"{label} folder {folder} has {n} PNGs; expected {expected}")
print(f"Validated {expected} prompts plus cached Vanilla, ESD-x, UCE, and SPACE-v1 outputs.")
PY
}

train_space_v3() {
  local ckpt="$ROOT/$SAVE_DIR/space-$EXP_NAME.safetensors"
  if [ "$FORCE_SPACE_V3_REBUILD" = "1" ] && [ -f "$ckpt" ]; then
    rm -f "$ckpt"
  fi
  if [ -f "$ckpt" ]; then
    echo "Skipping SPACE-v3 training; checkpoint already exists: $ckpt"
    return
  fi
  run_with_progress "SPACE-v3 training" env \
    DEVICE="$DEVICE" \
    TARGET_PROMPTS_PATH="$TARGET_PROMPTS_PATH" \
    TARGET_ARTIST_FILTER="$TARGET_ARTIST_FILTER" \
    SAVE_DIR="$SAVE_DIR" \
    EXP_NAME="$EXP_NAME" \
    bash "$ROOT/space_v3/run_space_v3_vangogh.sh"
}

generate_space_v3() {
  local count
  count="$(count_pngs "$SPACE_V3_OUT_DIR")"
  if [ "$FORCE_SPACE_V3_IMAGES" = "1" ] && [ -d "$SPACE_V3_OUT_DIR" ]; then
    rm -rf "$SPACE_V3_OUT_DIR"
    count=0
  fi
  if [ "$count" = "$EXPECTED_COUNT" ]; then
    echo "Skipping SPACE-v3 image generation; found $count PNGs in $SPACE_V3_OUT_DIR"
    return
  fi
  run_with_progress "SPACE-v3 images" env \
    ONLY_ARTIST="$ONLY_ARTIST" \
    DEVICE="$DEVICE" \
    WEIGHTS_DIR="$SAVE_DIR" \
    OUT_DIR="$SPACE_V3_OUT_ROOT" \
    EXP_NAME="$EXP_NAME" \
    OUTPUT_MODEL_NAME="space-Van_Gogh" \
    TARGET_PROMPTS_PATH="$TARGET_PROMPTS_PATH" \
    TARGET_ARTIST_FILTER="$TARGET_ARTIST_FILTER" \
    NUM_SAMPLES=1 \
    bash "$ROOT/space_v3/run_space_v3_images.sh"

  count="$(count_pngs "$SPACE_V3_OUT_DIR")"
  if [ "$count" != "$EXPECTED_COUNT" ]; then
    echo "SPACE-v3 image validation failed: $SPACE_V3_OUT_DIR has $count PNGs; expected $EXPECTED_COUNT"
    exit 1
  fi
}

write_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

import pandas as pd

metrics_path = Path("results/evaluation/metrics.csv")
summary_path = Path("results/evaluation/space_v3_summary.md")
summary_json_path = Path("results/evaluation/space_v3_summary.json")
df = pd.read_csv(metrics_path)
rows = df[(df["artist"] == "Van Gogh") & (df["method"].isin(["ESD-x", "UCE", "SPACE", "SPACE-v3"]))].copy()
payload = {
    "status": "ok",
    "objective": "Preserve better than UCE while matching or beating SPACE-v1/UCE erasure.",
    "rows": rows.to_dict(orient="records"),
    "artifacts": {
        "checkpoint": "space-models/sd-v3/space-Van_Gogh_v3.safetensors",
        "images": "results/space_v3/space-Van_Gogh",
        "metrics": "results/evaluation/metrics.csv",
        "table": "results/evaluation/ablation_table.tex",
        "bars": "results/evaluation/comparison_bars.png",
        "grid": "results/comparisons/van_gogh.png",
    },
}
summary_json_path.write_text(json.dumps(payload, indent=2) + "\n")

def fmt(value):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

lines = [
    "# SPACE-v3 Van Gogh Benchmark",
    "",
    "Compared cached Vanilla SD v1.4, ESD-x, UCE, and SPACE-v1 outputs against SPACE-v3.",
    "",
    "Primary target: SPACE-v3 should preserve better than UCE while matching or beating SPACE-v1/UCE erasure.",
    "",
    "## Metrics",
    "",
    "| Method | n_pairs | CLIP drop | Style drop | Style target rate | LPIPS | CLIP image sim | DINO sim | FID |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for _, row in rows.iterrows():
    lines.append(
        "| {method} | {n_pairs} | {clip_drop} | {style_drop} | {style_target_rate} | {lpips} | {clip_image_similarity} | {dino_similarity} | {fid} |".format(
            method=row["method"],
            n_pairs=int(row["n_pairs"]),
            clip_drop=fmt(row["clip_drop"]),
            style_drop=fmt(row["style_drop"]),
            style_target_rate=fmt(row["style_target_rate"]),
            lpips=fmt(row["lpips"]),
            clip_image_similarity=fmt(row["clip_image_similarity"]),
            dino_similarity=fmt(row["dino_similarity"]),
            fid=fmt(row["fid"]),
        )
    )
lines.extend(
    [
        "",
        "## Artifacts",
        "",
        "- Checkpoint: `space-models/sd-v3/space-Van_Gogh_v3.safetensors`",
        "- Images: `results/space_v3/space-Van_Gogh`",
        "- Metrics: `results/evaluation/metrics.csv`",
        "- Table: `results/evaluation/ablation_table.tex`",
        "- Bars: `results/evaluation/comparison_bars.png`",
        "- Grid: `results/comparisons/van_gogh.png`",
        "- Summary JSON: `results/evaluation/space_v3_summary.json`",
        "",
    ]
)
summary_path.write_text("\n".join(lines))
print(summary_path.read_text())
PY
}

progress_banner "SPACE-v3 Van Gogh benchmark"
progress_step 1 6 "validate cached inputs"
validate_inputs

progress_step 2 6 "train SPACE-v3"
train_space_v3

progress_step 3 6 "generate SPACE-v3 images"
generate_space_v3

progress_step 4 6 "evaluate ESD-x, UCE, SPACE-v1, SPACE-v3"
rm -f "$ROOT/results/evaluation/metrics_cache_v5_van_gogh.json"
run_with_progress "evaluate SPACE-v3 benchmark" "$PYTHON_BIN" "$ROOT/evaluate.py" \
  --only-artist "Van Gogh" \
  --method-keys "esd,uce,space,space_v3"

progress_step 5 6 "comparison grid"
run_with_progress "grid SPACE-v3 benchmark" "$PYTHON_BIN" "$ROOT/make_comparison_grids.py" \
  --only-artist "Van Gogh" \
  --methods "ESD-x,UCE,SPACE,SPACE-v3"

progress_step 6 6 "write SPACE-v3 summary"
write_summary

echo ""
echo "SPACE-v3 benchmark complete."
echo "Metrics: results/evaluation/metrics.csv"
echo "Grid: results/comparisons/van_gogh.png"
echo "Summary: results/evaluation/space_v3_summary.md"
