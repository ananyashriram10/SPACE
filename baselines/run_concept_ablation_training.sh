#!/bin/bash
# Train Concept Ablation style-erasure baselines with the official CompVis implementation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CA_ROOT="$ROOT/baselines/external/concept-ablation"
CA_DIR="$CA_ROOT/compvis"
OUT_ROOT="$ROOT/baseline-models/concept_ablation_compvis"
LOGS="$ROOT/logs"
PROVENANCE="$ROOT/results/provenance/concept_ablation_compvis"
ASSETS="$ROOT/baseline-assets/concept_ablation/pretrained_models"

GPUS="${GPUS:-0,}"
TRAIN_SIZE="${TRAIN_SIZE:-200}"
STEPS="${STEPS:-100}"
LR="${LR:-2e-6}"
SAVE_FREQ="${SAVE_FREQ:-100}"
SD_CKPT="${SD_CKPT:-$ASSETS/sd-v1-4.ckpt}"
PROMPTS="${PROMPTS:-../assets/finetune_prompts/painting.txt}"
ONLY_ARTIST="${ONLY_ARTIST:-}"
source "$ROOT/baselines/progress.sh"

if [ "${BASELINE_REPOS_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/setup_official_repos.sh"
fi
if [ "${BASELINE_PREFLIGHT_READY:-0}" != "1" ]; then
  bash "$ROOT/baselines/preflight_env.sh"
fi

if [ ! -f "$CA_DIR/train.py" ]; then
  echo "Missing official Concept Ablation repo at $CA_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

if [ ! -f "$SD_CKPT" ]; then
  echo "Missing SD v1.4 original checkpoint: $SD_CKPT"
  echo "Run: bash baselines/setup_concept_ablation_compvis.sh"
  exit 1
fi

mkdir -p "$OUT_ROOT/logs" "$LOGS" "$PROVENANCE"

bash "$ROOT/baselines/apply_official_patches.sh"

should_run() {
  local artist="$1"
  [ -z "$ONLY_ARTIST" ] || [ "$artist" = "$ONLY_ARTIST" ]
}

run_ca() {
  local idx="$1"
  local total="$2"
  local concept="$3"
  local exp="$4"
  local caption="$5"
  local official_ckpt="$OUT_ROOT/official_weights/$exp.ckpt"
  local cmd

  echo ""
  progress_step "$idx" "$total" "Concept Ablation training: $concept"

  if [ -f "$official_ckpt" ] && [ "${FORCE_CA_TRAIN:-0}" != "1" ]; then
    echo "Using author-provided official delta: $official_ckpt"
    echo "Set FORCE_CA_TRAIN=1 to train this concept instead."
    python "$ROOT/baselines/record_provenance.py" \
      --method "Concept Ablation" \
      --artist "$concept" \
      --run_type "official-paper" \
      --repo_url "https://github.com/nupurkmr9/concept-ablation" \
      --repo_dir "$CA_ROOT" \
      --command "use official checkpoint $official_ckpt" \
      --outputs "$official_ckpt" \
      --out "$PROVENANCE/train_$exp.json"
    return 0
  fi

  cmd="python train.py -t --gpus $GPUS --concept_type style --caption_target $caption --prompts $PROMPTS --name $exp --train_size $TRAIN_SIZE --train_max_steps $STEPS --base_lr $LR --save_freq $SAVE_FREQ --logdir $OUT_ROOT/logs --resume-from-checkpoint-custom $SD_CKPT"
  (
    cd "$CA_DIR"
    run_with_progress "Concept Ablation official training for $concept" python train.py \
      -t \
      --gpus "$GPUS" \
      --concept_type style \
      --caption_target "$caption" \
      --prompts "$PROMPTS" \
      --name "$exp" \
      --train_size "$TRAIN_SIZE" \
      --train_max_steps "$STEPS" \
      --base_lr "$LR" \
      --save_freq "$SAVE_FREQ" \
      --logdir "$OUT_ROOT/logs" \
      --resume-from-checkpoint-custom "$SD_CKPT"
  ) 2>&1 | tee "$LOGS/concept_ablation_compvis_train_$exp.log"

  python "$ROOT/baselines/record_provenance.py" \
    --method "Concept Ablation" \
    --artist "$concept" \
    --run_type "official-paper" \
    --repo_url "https://github.com/nupurkmr9/concept-ablation" \
    --repo_dir "$CA_ROOT" \
    --command "$cmd" \
    --outputs "$OUT_ROOT/logs" \
    --out "$PROVENANCE/train_$exp.json"
}

progress_banner "Concept Ablation official CompVis baseline training"
echo "Output: $OUT_ROOT"

should_run "Kelly McKernan" && run_ca 1 6 "Kelly McKernan" "concept_ablation-Kelly_McKernan" "kelly mckernan"
should_run "Van Gogh" && run_ca 2 6 "Van Gogh" "concept_ablation-Van_Gogh" "van gogh"
should_run "Tyler Edlin" && run_ca 3 6 "Tyler Edlin" "concept_ablation-Tyler_Edlin" "tyler edlin"
should_run "Thomas Kinkade" && run_ca 4 6 "Thomas Kinkade" "concept_ablation-Thomas_Kinkade" "thomas kinkade"
should_run "Kilian Eng" && run_ca 5 6 "Kilian Eng" "concept_ablation-Kilian_Eng" "kilian eng"
should_run "Ajin: Demi Human" && run_ca 6 6 "Ajin: Demi Human" "concept_ablation-Ajin_Demi_Human" "ajin demi human"

echo ""
echo "Concept Ablation CompVis training complete. Next: bash baselines/run_concept_ablation_images.sh"
