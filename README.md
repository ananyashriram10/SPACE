# SPACE: Style-Preserving Attribute-Constrained Erasure

SPACE is our research codebase for artist-style unlearning in Stable Diffusion v1.4.

The goal is:

> Remove a target artist style while preserving prompt content, nearby non-target styles, and the base model's overall image quality.

For a prompt like `Bedroom in Arles by Vincent van Gogh`, the desired output is still a coherent bedroom. It should not collapse into unrelated images, and it should not intentionally become another named artist's style. SPACE targets neutralized style erasure: suppress the recognizable target style while keeping semantic content and visual quality high.

This repository currently contains:

- a six-artist ESD-x replication workflow
- official-code baseline harnesses for UCE and Concept Ablation
- SPACE-v1, our in-house dual-anchor style erasure method
- a Van Gogh cross-method benchmark comparing Vanilla SD v1.4, ESD-x, UCE, Concept Ablation, and SPACE on the same 50 prompts

## Current Status

ESD-x replication is available for six artists:

- Kelly McKernan
- Van Gogh
- Tyler Edlin
- Thomas Kinkade
- Kilian Eng
- Ajin: Demi Human

The current cross-method benchmark is Van Gogh only. It compares:

- Vanilla SD v1.4
- ESD-x
- UCE
- Concept Ablation
- SPACE

Still in progress:

- six-artist official UCE benchmarking
- six-artist official Concept Ablation benchmarking
- SPACE ablations and tuning beyond the first Van Gogh target

## How SPACE Works

SPACE stands for **Style-Preserving Attribute-Constrained Erasure**. It is a preservation-first artist/style erasure method. The core idea is to avoid treating style erasure as either pure deletion or artist-to-artist replacement. Instead, SPACE constructs multiple views of the same prompt and edits the model so the target artist cue is suppressed while the content and general image quality are retained.

For each target prompt, SPACE builds four prompt forms:

| Symbol | Meaning | Example for `Bedroom in Arles by Vincent van Gogh` |
| --- | --- | --- |
| `p_t` | target prompt with artist/style | `Bedroom in Arles by Vincent van Gogh` |
| `p_c` | content-only prompt | `Bedroom in Arles` |
| `p_a` | neutral art anchor | `a high quality painting of Bedroom in Arles` |
| `p_g` | generic image anchor | `a high quality image of Bedroom in Arles` |

The dual-anchor design is the heart of SPACE:

- `p_a` keeps the erased output aesthetically art-like. This is inspired by the way Concept Ablation can produce visually pleasing neutralized paintings instead of harsh artifacts.
- `p_g` protects broad image utility. It prevents the method from over-specializing to paintings and helps preserve generic image quality, content layout, and photograph-like behavior.

SPACE trains against diffusion noise predictions at several denoising timesteps. Let:

- `x_t` be the noisy latent at timestep `t`
- `eps_base(x_t, p)` be the frozen base model's noise prediction under prompt `p`
- `eps_space(x_t, p_t)` be the SPACE-ed model's prediction for the target prompt

The teacher target is intentionally written below as plain text math so GitHub renders it reliably:

```text
neutral_anchor = (1 - alpha_art) * eps_base(x_t, p_c)
               + alpha_art       * eps_base(x_t, p_a)

style_direction = eps_base(x_t, p_t) - eps_base(x_t, p_c)

teacher = neutral_anchor - erase_scale * style_direction
```

This does two things at once:

- moves the target prompt toward a neutral, content-preserving anchor
- explicitly pushes against the target artist style direction

That second term is important. A single-anchor method can simply learn to imitate a generic replacement distribution and may leave too much target style behind. SPACE instead estimates a target-specific style residual and subtracts it while anchoring the result to a high-quality neutral output.

## Losses And Training Signals

The base Stable Diffusion model is frozen. SPACE trains only localized LoRA-style adapters inserted into selected cross-attention projections.

The total training signal is:

```text
L_total =
    L_erase
  + lambda_style      * L_style
  + lambda_content    * L_content
  + lambda_image      * L_image
  + lambda_preserve   * L_preserve
  + lambda_vulnerable * L_vulnerable
  + lambda_lora       * L_lora
```

Each term has a specific job:

- `L_erase`: moves `eps_space(x_t, p_t)` toward the SPACE teacher.
- `L_style`: suppresses the learned low-rank style residual subspace.
- `L_content`: keeps the content-only prompt `p_c` close to the frozen base model.
- `L_image`: keeps the generic image anchor `p_g` close to the frozen base model.
- `L_preserve`: replays general non-target prompts from `data/coco_30k.csv`.
- `L_vulnerable`: preserves nearby non-target artist styles that are most likely to be damaged.
- `L_lora`: regularizes adapter drift so edits remain localized.

SPACE uses multiple trajectory bands, currently `0.2,0.5,0.8`, so the erase/preserve constraints are applied at early, middle, and late denoising regions rather than at a single point.

## What Is Novel

SPACE combines several ideas that are individually useful but not usually unified into one artist-erasure objective:

- **Dual-anchor neutralization:** SPACE uses both a neutral art anchor and a generic image anchor. The art anchor helps produce beautiful erased outputs; the image anchor protects general model utility.
- **Negative style-direction suppression:** SPACE does not only match `content` or `painting`. It explicitly subtracts the target style residual `eps_base(x_t, p_t) - eps_base(x_t, p_c)`.
- **Localized LoRA editing:** SPACE edits saliency-selected cross-attention projections instead of updating the whole model. This keeps the method narrow and makes preservation easier to control.
- **Trajectory-aware supervision:** SPACE trains across several denoising bands, so the edit is constrained throughout the generation path instead of at one sampled timestep.
- **Vulnerable-style preservation:** SPACE mines nearby artists using CLIP text similarity and explicitly preserves them during training.
- **Robust prompt families:** SPACE expands target prompts with aliases and indirect references such as `inspired by`, `as painted by`, and shortened artist names.
- **Benchmark-first design:** SPACE is evaluated on the same prompts and metrics as ESD-x, UCE, and Concept Ablation before scaling to more artists.

The research hypothesis is that this combination should improve the erasure-preservation tradeoff: stronger erasure than Concept Ablation, better content and quality preservation than aggressive ESD-x/UCE settings, and visually pleasing neutralized outputs.

## How SPACE Differs From Existing Baselines

| Method | Edit type | Target erasure mechanism | Preservation mechanism | Output goal | Limitation SPACE targets |
| --- | --- | --- | --- | --- | --- |
| ESD-x | gradient-based cross-attention edit | trains selected cross-attention parameters to move away from target concept predictions | narrow parameter scope reduces collateral damage | remove target style while keeping prompt content | can be harsh; erasure strength may trade off against semantic preservation and visual quality |
| UCE | closed-form cross-attention edit | solves an efficient linear edit for erased and preserved concepts | explicit preserve concept list | fast scalable editing | less trajectory-aware and less content/style factorized for individual prompts |
| Concept Ablation | concept-to-anchor ablation | maps target concept toward an anchor distribution such as generic paintings | anchor distribution preserves some visual plausibility | neutralize concept into a broader class | often visually attractive, but can under-erase and may damage nearby concepts |
| SPACE | localized LoRA adapters on saliency-selected cross-attention projections | dual-anchor teacher plus explicit negative style-direction suppression | content anchor, neutral image anchor, general prompts, vulnerable styles, and LoRA regularization | neutralized style erasure with strong content and quality preservation | designed to combine CA-like beauty with stronger erasure and better preservation control |

SPACE is not an artist replacement method. The target output is not "Van Gogh becomes Monet." Replacement-style anchors may be useful internally, but the paper-facing claim is neutralized artist/style erasure.

## Repo Map

The repo is organized into five main parts.

### 1. Core ESD-x Replication

- [generate_esd_old.py](generate_esd_old.py): SD v1.4 image generation for vanilla and ESD-x checkpoints
- [run_baseline.sh](run_baseline.sh): generate vanilla SD v1.4 images
- [run_all_erased.sh](run_all_erased.sh): generate replicated ESD-x erased outputs
- [esd_sd.py](esd_sd.py): ESD training entrypoint retained for baseline training and comparison
- [evalscripts/generate-images.py](evalscripts/generate-images.py): generic diffusers image generator with optional edited checkpoint

### 2. SPACE Method

- [space_sd.py](space_sd.py): SPACE-v1 training entrypoint
- [utils/space_trainer.py](utils/space_trainer.py): SPACE-v1 trainer with dual anchors, localized adapters, trajectory-aware losses, and preservation replay
- [run_space_training.sh](run_space_training.sh): train SPACE-v1 for the current benchmark target
- [run_space_images.sh](run_space_images.sh): generate images from a SPACE checkpoint into the benchmark layout

### 3. Evaluation And Visualization

- [evaluate.py](evaluate.py): computes CLIP-based erasure and preservation metrics, LPIPS, FID/KID-style scores, ResNet agreement, and optional DINO similarity
- [make_comparison_grids.py](make_comparison_grids.py): builds side-by-side prompt/image grids
- [serve_results.py](serve_results.py): local HTML viewer for tracked comparison artifacts
- [sort_baseline_by_artist.py](sort_baseline_by_artist.py): reorganizes shared baseline folders into artist-specific folders when needed

### 4. Official Baseline Harnesses

Everything for UCE and Concept Ablation lives under [baselines/](baselines/).

Key files:

- [baselines/setup_official_repos.sh](baselines/setup_official_repos.sh): clones and pins official external repos
- [baselines/bootstrap_runpod_env.sh](baselines/bootstrap_runpod_env.sh): builds the pinned RunPod `.venv`
- [baselines/preflight_env.sh](baselines/preflight_env.sh): verifies the baseline environment
- [baselines/run_uce_training.sh](baselines/run_uce_training.sh): official UCE training wrapper
- [baselines/run_uce_images.sh](baselines/run_uce_images.sh): official UCE generation wrapper
- [baselines/setup_concept_ablation_compvis.sh](baselines/setup_concept_ablation_compvis.sh): prepares Concept Ablation assets
- [baselines/run_concept_ablation_training.sh](baselines/run_concept_ablation_training.sh): official CompVis CA training wrapper
- [baselines/run_concept_ablation_images.sh](baselines/run_concept_ablation_images.sh): official CompVis CA generation wrapper
- [baselines/run_vangogh_replication_and_report.sh](baselines/run_vangogh_replication_and_report.sh): Van Gogh-only end-to-end benchmark pipeline
- [baselines/rerun_vangogh_uce_fixed.sh](baselines/rerun_vangogh_uce_fixed.sh): reruns only the UCE Van Gogh branch
- [baselines/run_uce_six_artist_concurrent_training.sh](baselines/run_uce_six_artist_concurrent_training.sh): future six-artist concurrent UCE scaffolding
- [baselines/six_artist_config.sh](baselines/six_artist_config.sh): shared six-artist configuration

### 5. Prompt Data And Documentation

- [data/vangogh_prompts.csv](data/vangogh_prompts.csv): 50 Van Gogh prompts used for the current cross-method benchmark
- [data/kelly_prompts.csv](data/kelly_prompts.csv): Kelly McKernan prompts
- [data/short_niche_art_prompts.csv](data/short_niche_art_prompts.csv): prompts for retained niche artists
- [data/coco_30k.csv](data/coco_30k.csv): general prompt set for preservation and quality checks
- [docs/SPACE_OBJECTIVE.md](docs/SPACE_OBJECTIVE.md): research objective and scope

## Result Files: What Exists And Where

### Vanilla SD v1.4 Outputs

Stored in [results/baseline/](results/baseline/).

Important artist folders:

- [results/baseline/vangogh](results/baseline/vangogh)
- [results/baseline/kelly_mckernan](results/baseline/kelly_mckernan)
- [results/baseline/tyler_edlin](results/baseline/tyler_edlin)
- [results/baseline/thomas_kinkade](results/baseline/thomas_kinkade)
- [results/baseline/kilian_eng](results/baseline/kilian_eng)
- [results/baseline/ajin_demi_human](results/baseline/ajin_demi_human)

These are the unedited SD v1.4 generations used as the reference model.

### ESD-x Erased Outputs

Stored in [results/erased/](results/erased/).

Important folders:

- [results/erased/vangogh](results/erased/vangogh)
- [results/erased/kelly](results/erased/kelly)
- [results/erased/tyler_edlin](results/erased/tyler_edlin)
- [results/erased/thomas_kinkade](results/erased/thomas_kinkade)
- [results/erased/kilian_eng](results/erased/kilian_eng)
- [results/erased/ajin](results/erased/ajin)

These are the current six-artist ESD-x replication outputs.

### Official UCE Outputs

Current validated Van Gogh run:

- [results/uce/uce-Van_Gogh](results/uce/uce-Van_Gogh)

Checkpoint path used to generate them:

- `baseline-models/uce/uce-Van_Gogh.safetensors`

Notes:

- the first UCE Van Gogh run used an over-aggressive wrapper setting and produced semantically bad images
- the current harness defaults to `expand_prompts=false`, matching official SD v1.4 artist-erasure usage more closely
- if UCE looks wrong again, rerun [baselines/rerun_vangogh_uce_fixed.sh](baselines/rerun_vangogh_uce_fixed.sh)

### Official Concept Ablation Outputs

Current validated Van Gogh run:

- [results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples](results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples)

Official downloaded Van Gogh delta:

- `baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt`

The paper-faithful path is the official `compvis/` implementation, not the secondary diffusers smoke path.

### SPACE Outputs

SPACE outputs are written to:

- [results/space/space-Van_Gogh](results/space/space-Van_Gogh)

Checkpoint path:

- `space-models/sd/space-Van_Gogh.safetensors`

### Comparison Grids

All side-by-side grids are in [results/comparisons/](results/comparisons/).

Most important files:

- [results/comparisons/index.html](results/comparisons/index.html): browser entry point for tracked comparisons
- [results/comparisons/van_gogh.png](results/comparisons/van_gogh.png): Van Gogh prompt-by-prompt grid across Vanilla, ESD-x, UCE, CA, and SPACE
- [results/comparisons/van_gogh.pdf](results/comparisons/van_gogh.pdf)
- [results/comparisons/kelly_mckernan.png](results/comparisons/kelly_mckernan.png)
- [results/comparisons/tyler_edlin.png](results/comparisons/tyler_edlin.png)
- [results/comparisons/thomas_kinkade.png](results/comparisons/thomas_kinkade.png)
- [results/comparisons/kilian_eng.png](results/comparisons/kilian_eng.png)
- [results/comparisons/ajin_demi_human.png](results/comparisons/ajin_demi_human.png)

The six artist PNG/PDF files are ESD-x replication grids. `van_gogh.png` and `van_gogh.pdf` are the main cross-method benchmark grids and include SPACE once SPACE outputs are generated.

### Evaluation Outputs

All aggregate metrics and summaries are in [results/evaluation/](results/evaluation/).

Important files:

- [results/evaluation/metrics.csv](results/evaluation/metrics.csv): main metric table
- [results/evaluation/ablation_table.tex](results/evaluation/ablation_table.tex): LaTeX table for paper/report writing
- [results/evaluation/ablation_table.png](results/evaluation/ablation_table.png): rendered table image
- [results/evaluation/comparison_bars.png](results/evaluation/comparison_bars.png): quick bar-chart comparison
- [results/evaluation/van_gogh_run_summary.md](results/evaluation/van_gogh_run_summary.md): end-to-end summary of the Van Gogh cross-method run
- [results/evaluation/van_gogh_run_summary.json](results/evaluation/van_gogh_run_summary.json): machine-readable run summary
- [results/evaluation/metrics_cache.json](results/evaluation/metrics_cache.json): cached intermediate evaluation outputs

### Provenance Files

Official-code baseline and SPACE runs write provenance JSON under [results/provenance/](results/provenance/).

Current files:

- [results/provenance/uce/train_uce-Van_Gogh.json](results/provenance/uce/train_uce-Van_Gogh.json)
- [results/provenance/uce/uce-Van_Gogh.json](results/provenance/uce/uce-Van_Gogh.json)
- [results/provenance/space/train_space-Van_Gogh.json](results/provenance/space/train_space-Van_Gogh.json)
- [results/provenance/space/space-Van_Gogh.json](results/provenance/space/space-Van_Gogh.json)
- [results/provenance/concept_ablation_compvis/train_concept_ablation-Van_Gogh.json](results/provenance/concept_ablation_compvis/train_concept_ablation-Van_Gogh.json)
- [results/provenance/concept_ablation_compvis/images_concept_ablation-Van_Gogh.json](results/provenance/concept_ablation_compvis/images_concept_ablation-Van_Gogh.json)

These files record run mode, command path, output path, official repo metadata where relevant, and other audit details needed to keep official replications separate from local experiments.

## Local-Only Folders

These folders matter during runs but are not meant to be committed as tracked artifacts:

- `baseline-models/`: generated checkpoints and downloaded deltas
- `baseline-assets/`: persistent large assets such as Concept Ablation pretrained models
- `.venv/`: pinned RunPod environment
- `baselines/external/`: official repositories cloned locally and reset to pinned commits

## How To Reproduce The Current Outputs

### ESD-x Six-Artist Replication

Generate vanilla images:

```bash
bash run_baseline.sh
```

Generate all ESD-x erased images:

```bash
bash run_all_erased.sh
```

Rebuild comparison grids:

```bash
python3 make_comparison_grids.py
```

Run evaluation:

```bash
python3 evaluate.py
```

### Van Gogh Benchmark Comparison

This is the cleanest cross-method entry point in the repo right now.

On RunPod:

```bash
bash baselines/bootstrap_runpod_env.sh

ONLY_ARTIST="Van Gogh" \
STRICT_EVAL=1 \
PROGRESS_INTERVAL=15 \
bash baselines/run_vangogh_replication_and_report.sh
```

That pipeline:

- validates baseline inputs
- reuses existing ESD-x, UCE, and Concept Ablation outputs when their 50-image folders are already present
- trains UCE for Van Gogh only if the checkpoint is missing or `FORCE_UCE_REBUILD=1`
- generates UCE Van Gogh images only if the cached 50-image folder is missing
- trains SPACE for Van Gogh if needed
- generates SPACE Van Gogh images if missing, or if `FORCE_SPACE_REBUILD=1` / `FORCE_SPACE_IMAGES=1`
- prepares Concept Ablation assets and images only if the cached delta/images are missing
- evaluates Vanilla vs ESD-x vs UCE vs CA vs SPACE
- writes the Van Gogh comparison chart and summary report

By default, `REUSE_BASELINE_OUTPUTS=1`, so rerunning the full script after baseline replication should spend compute on SPACE and reporting, not on regenerating UCE or Concept Ablation. Set `REUSE_BASELINE_OUTPUTS=0` only when you intentionally want to refresh official baseline outputs.

### SPACE-Only Path

Run a 10-step sanity check:

```bash
SPACE_DEBUG_STEPS=10 bash run_space_training.sh
```

Train SPACE:

```bash
bash run_space_training.sh
```

Generate SPACE images:

```bash
bash run_space_images.sh
```

Useful SPACE knobs:

- `ERASE_SCALE`: strength of the negative target-style push, default `1.0`
- `ALPHA_ART`: blend weight for the neutral art anchor, default `0.30`
- `TRAJECTORY_BANDS`: denoising bands used for supervision, default `0.2,0.5,0.8`
- `ROBUST_PROMPT_MODE`: target prompt expansion mode, default `full`
- `SPACE_DEBUG_STEPS`: short sanity run that verifies nonzero adapter movement and checkpoint reload

### UCE-Only Repair Path

If the UCE column looks semantically wrong, rerun only the UCE branch:

```bash
PROGRESS_INTERVAL=15 bash baselines/rerun_vangogh_uce_fixed.sh
```

This keeps the Vanilla, ESD-x, and Concept Ablation outputs and rewrites only:

- `results/uce/uce-Van_Gogh/`
- `results/comparisons/van_gogh.*`
- `results/evaluation/metrics.csv`
- `results/evaluation/ablation_table.*`
- `results/evaluation/van_gogh_run_summary.*`

## Research Direction

The project thesis is documented in [docs/SPACE_OBJECTIVE.md](docs/SPACE_OBJECTIVE.md).

In short:

- primary task: artist/style erasure, not artist replacement
- desired output: neutralized style with preserved content
- main competition: ESD-x, UCE, Concept Ablation, then later larger baselines
- main success condition: match or improve erasure while improving preservation and general model quality

The current paper-facing claim should be stated carefully: SPACE is an output-level style suppression method under active validation. We do not claim legal or literal data deletion from the model weights.

## Recommended Reading Order

If you are new to the repo:

1. Read [docs/SPACE_OBJECTIVE.md](docs/SPACE_OBJECTIVE.md)
2. Open [results/comparisons/van_gogh.png](results/comparisons/van_gogh.png)
3. Read [results/evaluation/van_gogh_run_summary.md](results/evaluation/van_gogh_run_summary.md)
4. Check [results/evaluation/metrics.csv](results/evaluation/metrics.csv)
5. For harness details, read [baselines/README.md](baselines/README.md)

## Attribution

This repo contains our research workflow and wrappers, but several baselines depend on upstream official work.

- ESD / ESD-x: [erasing.baulab.info](https://erasing.baulab.info/)
- UCE: [unified.baulab.info](https://unified.baulab.info/)
- Concept Ablation: [cs.cmu.edu/~concept-ablation](https://www.cs.cmu.edu/~concept-ablation/)

Official baseline source code is not vendored into the tracked repository. It is cloned into ignored local directories by scripts in [baselines/](baselines/).

## License

See [LICENSE](LICENSE).
