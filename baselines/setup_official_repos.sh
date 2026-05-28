#!/bin/bash
# Clone the official baseline repositories used for replication.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="$ROOT/baselines/external"

CONCEPT_ABLATION_REPO="https://github.com/nupurkmr9/concept-ablation.git"
CONCEPT_ABLATION_REV="c11fb1bdc7cd97ea65877b0d41722f5e3ef2561f"

UCE_REPO="https://github.com/rohitgandikota/unified-concept-editing.git"
UCE_REV="7c724d9d7d19190e9bd33fe62fb656f1df87cd8e"

mkdir -p "$EXTERNAL"

clone_or_update() {
  local name="$1"
  local repo="$2"
  local rev="$3"
  local dir="$EXTERNAL/$name"

  if [ ! -d "$dir/.git" ]; then
    echo "Cloning $name..."
    git clone "$repo" "$dir"
  else
    echo "Updating $name..."
    git -C "$dir" fetch origin
  fi

  git -C "$dir" checkout "$rev"
  git -C "$dir" reset --hard "$rev"
  git -C "$dir" clean -fd
  echo "  $name pinned at $(git -C "$dir" rev-parse --short HEAD)"
}

clone_or_update "concept-ablation" "$CONCEPT_ABLATION_REPO" "$CONCEPT_ABLATION_REV"
clone_or_update "unified-concept-editing" "$UCE_REPO" "$UCE_REV"

if [ "${INSTALL_DEPS:-0}" = "1" ]; then
  echo ""
  echo "INSTALL_DEPS=1 is disabled for this harness because official requirements can pull incompatible torch/diffusers versions."
  echo "Use: bash baselines/bootstrap_runpod_env.sh"
  exit 1
fi

cat <<EOF

Official baseline repos are ready:
  $EXTERNAL/concept-ablation
  $EXTERNAL/unified-concept-editing

Next:
  bash baselines/run_uce_training.sh
  bash baselines/setup_concept_ablation_compvis.sh
  bash baselines/run_concept_ablation_training.sh
EOF
