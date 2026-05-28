# Official Baseline Replication Harness

This directory contains wrappers for reproducing additional concept-erasure baselines with the authors' official repositories.
The wrappers are deliberately thin: they prepare RunPod paths/prompt files, call official scripts, write provenance JSON, and check generated image integrity.

We keep external code out of this repository. `setup_official_repos.sh` clones the official repos into `baselines/external/`, which is ignored by git.

These baselines gate SPACE development. The next SPACE changes should wait until ESD-x, official UCE, and official Concept Ablation CompVis pass smoke checks and produce sane evaluation metrics.

## Baselines

### Concept Ablation

- Official repo: https://github.com/nupurkmr9/concept-ablation
- Pinned revision used by this harness: `c11fb1bdc7cd97ea65877b0d41722f5e3ef2561f`
- Paper: *Ablating Concepts in Text-to-Image Diffusion Models*, ICCV 2023

Concept Ablation trains the target prompt distribution to match an anchor concept distribution. For our style baseline, we use:

```text
target artist/style -> painting
```

The paper reports results from the official `compvis/` implementation, so this harness treats CompVis as the primary paper-faithful baseline. The official `diffusers/` implementation is kept only as a smoke-test path.

### UCE

- Official repo: https://github.com/rohitgandikota/unified-concept-editing
- Pinned revision used by this harness: `7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`
- Paper: *Unified Concept Editing in Diffusion Models*

UCE is a closed-form edit over cross-attention projections. For our style baseline, we use:

```text
edit_concepts = artist
guided_concept = art
preserve_concepts = Monet; Rembrandt; Warhol
expand_prompts = false
```

## Setup

Clone the official repos:

```bash
bash baselines/setup_official_repos.sh
```

One-time RunPod environment bootstrap:

```bash
bash baselines/bootstrap_runpod_env.sh
```

If the persistent `.venv` gets corrupted, rebuild only the baseline packages:

```bash
FORCE_ENV_REBUILD=1 bash baselines/bootstrap_runpod_env.sh
```

Primary one-command pipeline (recommended):

```bash
bash baselines/run_baselines_and_validate.sh
```

This orchestrator runs setup, preflight, asset prep, UCE, Concept Ablation, strict evaluation, and comparison grids in fail-fast order.
It exits immediately on the first failing step and prints the step name.

Smoke mode for one artist (Van Gogh):

```bash
SMOKE=1 STRICT_EVAL=0 bash baselines/run_baselines_and_validate.sh
```

Van Gogh benchmark report with strict UCE/Concept Ablation/SPACE validation:

```bash
ONLY_ARTIST="Van Gogh" \
STRICT_EVAL=1 \
PUSH_RESULTS=1 \
PROGRESS_INTERVAL=15 \
bash baselines/run_vangogh_replication_and_report.sh
```

This Van Gogh-only path expects 50 prompts and writes `results/comparisons/van_gogh.png`,
`results/evaluation/metrics.csv`, `results/evaluation/ablation_table.tex`, and
`results/evaluation/van_gogh_run_summary.md`.

If the UCE column looks semantically broken, regenerate only UCE with the official-style artist
erasure defaults:

```bash
PROGRESS_INTERVAL=15 bash baselines/rerun_vangogh_uce_fixed.sh
```

This uses `expand_prompts=false`, matching the official SD v1.4 README example. The previous
wrapper used expanded prompt variants and can over-edit artwork-title prompts.

## UCE

Train UCE weights for all six artists:

```bash
bash baselines/run_uce_training.sh
```

Generate images with those UCE checkpoints:

```bash
bash baselines/run_uce_images.sh
```

Outputs:

- weights: `baseline-models/uce/`
- images: `results/uce/`

Future concurrent six-artist UCE checkpoint:

```bash
bash baselines/run_uce_six_artist_concurrent_training.sh
```

This produces one edited checkpoint at `baseline-models/uce/uce-six-artists-concurrent.safetensors`
by passing all six artist names to the official UCE edit in one run. It is scaffolding for the
later multi-target erasure experiment; the current validated replication path remains Van Gogh
single-artist first.

## Concept Ablation: Primary CompVis Path

Prepare CompVis assets and the official author-provided Van Gogh delta:

```bash
bash baselines/setup_concept_ablation_compvis.sh
```

Train Concept Ablation weights for all six artists:

```bash
bash baselines/run_concept_ablation_training.sh
```

Generate images with those Concept Ablation deltas:

```bash
bash baselines/run_concept_ablation_images.sh
```

Outputs:

- weights/logs: `baseline-models/concept_ablation_compvis/`
- images: `results/concept_ablation_compvis/`
- persistent assets: `baseline-assets/concept_ablation/pretrained_models/`

The Van Gogh run can use the official downloaded delta. Other artists are trained with official `compvis/train.py` unless author-provided weights are added later.

Concept Ablation CompVis remains a per-target baseline in this harness. We should evaluate its
six-artist setting as six separate official checkpoints unless we later define and label a separate
sequential/merged CA experiment.

## Concept Ablation: Secondary Diffusers Smoke Path

```bash
bash baselines/run_concept_ablation_diffusers_training.sh
bash baselines/run_concept_ablation_diffusers_images.sh
```

Outputs:

- weights/logs: `baseline-models/concept_ablation_diffusers/`
- images: `results/concept_ablation_diffusers/`

Do not use this path as the primary paper comparison unless explicitly labeled as official-diffusers.

## Provenance And Integrity

Each official run writes provenance under:

```text
results/provenance/
```

Prompt manifests are written under:

```text
baselines/cache/official_prompts/
```

## Notes

- These scripts are intentionally thin wrappers around the official repos.
- Baseline entry scripts re-sync official external repos to pinned commits with `git reset --hard` and `git clean -fd` on each run for deterministic behavior.
- `run_uce_training.sh` and `run_uce_images.sh` auto-apply a minimal documented compatibility patch from `baselines/patches/` to the pinned official UCE repo. This avoids a current diffusers runtime crash (`vae=None`) while keeping official logic intact.
- `baselines/bootstrap_runpod_env.sh` installs the persistent `.venv` once. `baselines/preflight_env.sh` only checks the environment and points back to bootstrap if repair is needed.
- Do not use official repo requirements directly. They can pull incompatible torch/diffusers versions for this harness.
- We should first validate each baseline visually and quantitatively against ESD-x before using it as a paper baseline.
- The official Concept Ablation paper reports CompVis-based results; the CompVis path is now the primary baseline.
