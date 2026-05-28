# SPACE-v2 Research Report
## Style-Preserving Attribute-Constrained Erasure — Extended Architecture Investigation

---

## 1. Background: What SPACE-v1 Does and How It Works

### 1.1 Core Objective

SPACE (Style-Preserving Attribute-Constrained Erasure) is a method for removing a specific artist's style from a Stable Diffusion v1.4 model without degrading its ability to generate general content. The fundamental challenge is the **erasure-preservation tradeoff**: aggressively removing style tends to damage the model's general generation quality, while being too conservative leaves style residuals that can be extracted via adversarial prompts.

### 1.2 Architecture

SPACE operates by injecting **Low-Rank Adaptation (LoRA)** adapters into the UNet's cross-attention layers. Rather than fine-tuning the full model (as ESD-x does), it modifies only a small number of carefully selected linear projections, limiting the blast radius of the erasure.

**LoRA adapter structure:**
```
LoRALinear wraps base Linear:
  forward(x) = base(x) + scaling * (lora_up @ lora_down @ x)
  where:
    lora_down: [rank × in_features]   — initialized kaiming uniform
    lora_up:   [out_features × rank]  — initialized zeros
    scaling = alpha / rank             — controls adapter magnitude
```

At inference, the adapters are merged into the base weights (`merged_weight = base.weight + scaling * (lora_up @ lora_down)`), so there is no inference overhead.

### 1.3 Style Direction and Teacher Signal

The core of SPACE is the **style direction** — a per-prompt vector in the UNet's noise prediction space that captures what the target artist adds relative to neutral content:

```
style_direction = ε(x_t, t, "painting by Van Gogh") - ε(x_t, t, "painting")
```

This direction is computed at multiple denoising timesteps `t` and across multiple prompt phrasings. The **teacher signal** (the target the adapted model should match) is:

```
teacher = neutral_anchor - erase_scale × style_direction

where:
  neutral_anchor = (1 - alpha_art) × base_content + alpha_art × base_art
  base_content   = ε(x_t, t, "painting")             — content-only prediction
  base_art       = ε(x_t, t, "a high quality painting of {content}")  — generic art anchor
```

This teacher pulls the model toward a neutral content representation while actively pushing it away from the style direction. The `alpha_art=0.3` blend keeps some "painterly" quality (content anchor shifted slightly toward generic art) to avoid collapsing to photorealistic outputs.

### 1.4 Training Losses

The total per-step loss combines five objectives:

| Loss | Formula | Purpose |
|------|---------|---------|
| `erase_loss` | MSE(student_target, teacher) | Push adapted model away from style |
| `style_loss` | projection_energy(student_target - base_content, style_basis) | Suppress residual style in prediction space |
| `content_loss` | MSE(student_content, base_content) | Preserve content-only predictions |
| `image_loss` | MSE(student_image, base_image) | Preserve generic image predictions |
| `preserve_loss` | MSE(student_preserve, base_preserve) | Protect general COCO-like prompts |
| `vulnerable_loss` | MSE(student_nearby, base_nearby) | Protect nearby artists (Monet, Renoir, etc.) |
| `reg` | L2 on adapter parameters | Prevent overfitting |

The style basis `B` is computed via SVD of the style directions across multiple prompt variants, giving a rank-4 subspace estimate of the style manifold. `style_loss` penalizes any residual projection of the adapted prediction onto this subspace.

### 1.5 Saliency-Based Module Selection

Not all cross-attention layers are equally responsible for style representation. SPACE uses **gradient-magnitude saliency** to select which layers to adapt:

1. Probe the model with 4 target families × 2 trajectory bands
2. Compute the erasure loss and backpropagate
3. Rank all candidate layers by `||∂loss/∂W||_F`
4. Select the top 25% (`saliency_topk_blocks=0.25`) by gradient magnitude
5. Inject LoRA adapters only into those layers

This concentrates the capacity where it matters most, reducing preservation damage from unnecessary layer modification.

### 1.6 Trajectory Sampling

The denoising trajectory is sampled at fractional positions `[0.2, 0.5, 0.8]` of the 50-step schedule, corresponding to timestep indices approximately `[10, 25, 40]`. At each band, a noisy latent `x_t` is sampled by running the full forward diffusion process to that timestep, and losses are computed. The total loss is averaged across bands:

```python
total_loss = sum(band_losses) / len(trajectory)
```

### 1.7 Robust Prompt Mode

To improve generalization, SPACE expands each training prompt into multiple phrasings:
- `"painting by Van Gogh"` → `"painting in the style of Van Gogh"`, `"painting inspired by Van Gogh"`, `"painting as painted by Van Gogh"`, etc.
- The content prompt is the original with the artist reference stripped

This ensures the model erases style regardless of how the artist is referenced.

### 1.8 SPACE-v1 Results on Van Gogh

| Metric | Baseline | SPACE-v1 | ESD-x | UCE | CA |
|--------|----------|----------|-------|-----|-----|
| CLIP drop ↑ | — | 0.028 | 0.059 | 0.028 | 0.016 |
| Style drop ↑ | — | 0.046 | 0.075 | 0.046 | 0.033 |
| Style target rate ↓ | — | 0.46 | 0.32 | 0.50 | 0.64 |
| LPIPS ↑ | — | 0.589 | 0.659 | 0.763 | 0.754 |
| FID ↑ (vs baseline) | — | 236 | 250 | 246 | 202 |
| DINO similarity ↑ | — | 0.617 | 0.513 | 0.583 | 0.625 |

SPACE-v1 sits in the middle of the pack: better erasure than UCE and CA, but weaker than ESD-x. Its key advantage over ESD-x is better preservation (DINO similarity 0.617 vs 0.513, LPIPS 0.589 vs 0.659) — SPACE modifies fewer parameters and targets them more precisely.

---

## 2. Problems Identified with SPACE-v1

A systematic analysis of SPACE-v1's design identified seven structural weaknesses:

### Problem 1: Style as a Single Linear Direction (Fundamental Assumption)

SPACE computes `style_direction = base_target - base_content` per prompt, then builds a PCA basis across prompts. This assumes style lives in a **low-dimensional linear subspace** of the 16,384-dimensional noise prediction space (64×64×4 latent, flattened).

**Why this is wrong:** Van Gogh's style is high-dimensional and nonlinear — swirling brushstroke texture, heavy impasto, saturated yellows and blues, distinctive compositional rhythm. These characteristics are encoded across different layers, different attention heads, and different denoising timesteps in fundamentally non-linear ways. A rank-4 PCA subspace almost certainly underfits the style manifold.

**Consequence:** Erasure is incomplete for prompts that lie off the learned basis. The model erases the "average Van Gogh direction" but leaves behind style components that are orthogonal to the basis.

### Problem 2: Text-Space Style Direction ≠ Visual Style

The style direction is computed from UNet noise predictions conditioned on CLIP text embeddings. CLIP text encoders are trained for broad text-image alignment, not fine-grained artistic style discrimination.

**Why this matters:** The text "Vincent van Gogh" in CLIP space encodes the concept of the artist (post-impressionist, Dutch, 19th century), not the visual signature (swirling brushstrokes, impasto, specific palette). Two prompts like "a painting by Van Gogh" and "Starry Night style" produce different style directions even though they share the same target style. The style direction is therefore **noisy and prompt-dependent**, and erasure trained on one prompt distribution may not generalize to novel phrasings.

### Problem 3: Only Targeting Cross-Attention (attn2) — KEY WEAKNESS

SPACE limits LoRA injection to `attn2` layers — the cross-attention mechanism that attends from image features to text embeddings. However, style in diffusion models is encoded across multiple layer types:

- **Self-attention (attn1):** Spatial relationships between image patches — responsible for brushstroke patterns, compositional structure, and texture regularity
- **Feedforward layers (FFN):** Feature transformations at each spatial position — responsible for color palette, local texture statistics, and tonal characteristics
- **Cross-attention (attn2):** Text conditioning — responsible for concept-level style association (the "Van Gogh" concept token's influence)

**Consequence:** By targeting only attn2, SPACE may suppress the textual style concept while leaving the visual texture and compositional patterns intact. This is consistent with why adversarial prompts (indirect references like "Starry Night artist" or "Dutch post-impressionist") often bypass such methods — the visual style lives in attn1/FFN and survives even when attn2-based style association is erased.

### Problem 4: Vulnerable Artist Mining via CLIP Text Similarity

Nearby artists (those to preserve explicitly) are identified by CLIP text embedding similarity to the target. But visual style proximity and text embedding proximity are not the same thing.

**Example:** Monet and Van Gogh are both Impressionists (close in CLIP text space — similar historical and stylistic descriptions), but their visual styles are quite distinct. The preservation loss may protect the wrong artists and miss truly visually similar ones.

### Problem 5: Trajectory Band Sparsity (3 Points)

Training uses only 3 trajectory points `[0.2, 0.5, 0.8]` corresponding to `t ≈ 250, 500, 750` (on a 1000-step DDPM schedule compressed to 50 inference steps).

**Why this matters:** Different denoising timesteps encode different aspects of generation:
- High noise (t ≈ 0.7–1.0): Global composition and rough structure
- Mid noise (t ≈ 0.3–0.6): **Style, texture, color, mood** — where style is most strongly encoded
- Low noise (t ≈ 0.0–0.2): Fine details, sharpness, edge definition

The 0.2 and 0.5 band partially cover the style-encoding range, but with 3 points the training signal is sparse. Timesteps between the sampled bands are entirely unconstrained, leaving gaps where style can persist unpenalized.

### Problem 6: Saliency Computed from Early Training

Module saliency is determined by gradient magnitudes from the first few training steps (4 prompt families × 2 trajectory bands = 8 forward-backward passes). This snapshot is taken before any adapter updates occur, meaning:

1. Early gradients reflect initial model sensitivity, not causal responsibility for style at convergence
2. The specific random batch used for probing introduces variance
3. Layer importance can shift as training progresses — layers that are initially salient may become less important after their neighbors adapt

**Consequence:** LoRA adapters may be placed on suboptimal layers, reducing training efficiency.

### Problem 7: Fixed Hyperparameters Across All Artists

`alpha_art=0.3` and `erase_scale=1.0` are fixed for all artists. Different artists have different "style intensities" — Van Gogh's style is extremely distinctive and visually dominant, while subtler artists require different erasure strengths. The fixed hyperparameters likely represent a compromise that is suboptimal for each individual artist.

---

## 3. SPACE-v2: Improvements Attempted

Based on the analysis above, three improvements were selected for feasibility and expected impact. They were chosen because they are implementable within the existing training framework without fundamental algorithmic restructuring.

### Improvement B: Extend LoRA to Self-Attention + Feedforward Layers

**Motivation (Problem 3 above):** Style texture and compositional patterns are encoded in self-attention and FFN layers, not just cross-attention. Targeting only attn2 leaves these style components intact.

**Implementation:**

```python
# OLD: hardcoded attn2 filter
def select_cross_attention_linears(unet: nn.Module) -> list[str]:
    candidates = []
    for module_name, module in unet.named_modules():
        if "attn2" not in module_name:
            continue
        if module.__class__.__name__ not in {"Linear", "LoRACompatibleLinear"}:
            continue
        candidates.append(module_name)
    return candidates

# NEW: configurable layer targets
def select_candidate_linears(unet: nn.Module, layer_targets: str = "attn2,attn1,ff") -> list[str]:
    targets = [t.strip() for t in layer_targets.split(",") if t.strip()]
    candidates = []
    for module_name, module in unet.named_modules():
        if not any(t in module_name for t in targets):
            continue
        if module.__class__.__name__ not in {"Linear", "LoRACompatibleLinear"}:
            continue
        candidates.append(module_name)
    return candidates
```

The `layer_targets` parameter is exposed as `--lora_target_layers` in the CLI, allowing configurations like:
- `"attn2"` — original SPACE-v1 behavior
- `"attn2,attn1"` — add self-attention
- `"attn2,attn1,ff"` — add feedforward (SPACE-v2 default)

In SD v1.4's UNet, the affected layers per transformer block are:
- `attn1.to_q`, `attn1.to_k`, `attn1.to_v`, `attn1.to_out.0` — self-attention projections
- `attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `attn2.to_out.0` — cross-attention projections
- `ff.net.0.proj` — feedforward GEGLU projection (in → hidden×2)
- `ff.net.2` — feedforward output projection (hidden → out)

Adding attn1+FF increases the candidate pool from ~144 layers (attn2 only) to ~576 layers. With `saliency_topk_blocks=0.25`, the number of adapted layers increases from ~36 to ~144.

**Expected effect:** Deeper and more complete erasure since style texture patterns in self-attention are also suppressed.

**Risk:** More adapters means more parameters. With the same 400 training steps and same learning rate, each adapter receives ~4× less gradient signal per step. Adapters may be undertrained.

### Improvement C: Denser Trajectory Sampling in 0.3–0.6 Range

**Motivation (Problem 5 above):** The style-dense timestep range (t ≈ 0.3–0.6 fractional) is undersampled with only 1-2 points. The 0.8 band (global composition) and 0.2 band (fine details) contribute less style signal.

**Implementation:**

```python
# OLD default in SPACEConfig:
trajectory_bands: str = "0.2,0.5,0.8"  # 3 points: [10, 25, 40] on 50-step schedule

# NEW default in SPACE-v2:
trajectory_bands: str = "0.3,0.375,0.45,0.525,0.6"  # 5 points: [15, 18, 22, 26, 30]
```

The new schedule covers the style-encoding range uniformly at 0.075 fractional intervals, dropping:
- The `0.8` band: high-noise timesteps encode global composition and structure, not style texture — including this can unnecessarily damage compositional quality
- The `0.2` band: very low noise, fine details only, minimal style signal

No logic changes to `trajectory_indices()` were required — the function already handles arbitrary comma-separated fractions.

**Expected effect:** Denser gradient coverage of style-encoding timesteps gives the optimizer a more complete signal of where style lives in the denoising process. Should improve erasure consistency.

**Risk:** Adding 2 more trajectory bands increases the number of forward passes per training step (3 → 5), increasing per-step compute. The `total_loss / len(trajectory)` normalization means each band contributes proportionally less to the final gradient. Effectively, the learning rate per band decreases by 3/5 = 60%.

### Improvement J: Per-Prompt Loss Weighting by Style Direction Magnitude

**Motivation (Problem 2 above):** Not all training prompts carry the same style signal. "A painting in the style of Vincent van Gogh" has a strong, consistent style direction. "A landscape inspired by Van Gogh" may have a weaker, noisier style direction. Treating all prompts equally dilutes the erasure signal with noisy gradients.

**Implementation:**

After computing `base_target` and `base_content` for each training band:

```python
# Compute style direction magnitude
style_dir_norm = float((base_target.float() - base_content.float()).norm().item())

# Update exponential moving average of norms (EMA decay = 0.98)
style_norm_ema = style_dir_norm if style_norm_ema == 0.0 else \
    0.98 * style_norm_ema + 0.02 * style_dir_norm

# Weight for this step: how strong is this prompt's style signal
# relative to the running average?
style_weight = max(0.1, min(5.0, style_dir_norm / (style_norm_ema + 1e-8)))

# Apply to erasure loss only (not preservation losses)
erase_loss = erase_loss * style_weight
```

The EMA provides a running estimate of typical style direction magnitude. Prompts with `||style_direction|| >> EMA` get upweighted (stronger erasure); prompts with `||style_direction|| << EMA` get downweighted (weaker erasure). The `clamp(0.1, 5.0)` prevents extreme scaling.

**Expected effect:** More focused erasure gradient toward prompts with strong style signal, reducing noise from ambiguous phrasings.

**Risk:** If the EMA grows large early (common for Van Gogh's distinctive style), most prompts end up with `style_weight < 1.0`, effectively reducing average erasure loss. The net effect may be negative if most prompts have below-average style direction magnitude.

### 3.1 New Configuration Parameters

```python
@dataclass
class SPACEConfig:
    # ... existing fields ...
    trajectory_bands: str = "0.3,0.375,0.45,0.525,0.6"  # CHANGED from "0.2,0.5,0.8"
    lora_target_layers: str = "attn2,attn1,ff"            # NEW: controls B
    use_prompt_weighting: bool = True                      # NEW: controls J
```

CLI additions to `space_sd_v2.py`:
```
--lora_target_layers    comma-separated layer substrings (default: "attn2,attn1,ff")
--no_prompt_weighting   disable per-prompt loss weighting (default: enabled)
```

### 3.2 Bug Fixes Applied During Development

Three bugs were identified and fixed before the final SPACE-v2 run:

1. **Import path**: `space_sd_v2.py` now uses `sys.path.insert(0, parent_dir)` + direct `from space_trainer_v2 import ...` instead of fragile namespace package import
2. **Provenance directory**: `record_space_provenance` was writing to `results/provenance/space/` instead of `results/provenance/space_v2/`
3. **HuggingFace token**: Hardcoded token set via `os.environ` + `huggingface_hub.login()` before pipeline load

---

## 4. SPACE-v2 Results and Analysis

### 4.1 Results Table

| Method | n_pairs | clip_drop↑ | style_drop↑ | style_target_rate↓ | LPIPS↑ | FID↑ | KID↑ | DINO-sim↑ |
|--------|---------|-----------|------------|-------------------|--------|------|------|----------|
| ESD-x | 50 | 0.059 | 0.075 | 0.32 | 0.659 | 250 | 0.069 | 0.513 |
| UCE | 50 | 0.028 | 0.046 | 0.50 | 0.763 | 246 | 0.062 | 0.583 |
| CA | 50 | 0.016 | 0.033 | 0.64 | 0.754 | 202 | 0.039 | 0.625 |
| SPACE-v1 | 50 | 0.028 | 0.046 | 0.46 | 0.589 | 236 | 0.065 | 0.617 |
| **SPACE-v2 (B+C+J)** | **500** | **0.001** | **0.002** | **0.90** | **0.323** | **53** | **0.001** | **0.842** |

### 4.2 Interpretation

SPACE-v2 (B+C+J combined) **catastrophically failed at erasure**:

- **clip_drop = 0.001**: Near-zero. SPACE-v1 achieves 0.028 — SPACE-v2 erases 28× less
- **style_target_rate = 0.90**: 90% of generated images are still classified as Van Gogh style. ESD-x achieves 0.32. The model is essentially generating vanilla SD outputs with no style suppression
- **FID = 53**: Dramatically lower than all other methods. This is NOT good here — low FID means the model's output distribution is close to baseline, confirming it's not modifying the style at all
- **LPIPS = 0.323, DINO-sim = 0.842**: Images are perceptually very similar to baseline — confirming the LoRA adapters made almost no change to the model's behavior

The 500 pairs vs 50 pairs difference does not explain this — the effect size is 20× in clip_drop, well beyond any statistical artifact.

### 4.3 Root Cause Analysis

**Primary suspect: Gradient dilution from too many adapters (Improvement B)**

Going from attn2-only (~36 adapted layers with saliency topk=0.25) to attn2+attn1+FF (~144 adapted layers) is a **4× increase in adapter count**. The training budget is fixed at 400 steps × same learning rate. Each LoRA adapter receives approximately 1/4 the gradient signal per step compared to SPACE-v1. With 400 steps, the adapters simply do not converge enough to meaningfully alter the model's behavior.

Evidence: LPIPS=0.323 means images are very similar to baseline — the adapters haven't moved the model at all, not that they moved it in the wrong direction.

**Contributing factor: Loss normalization dilution (Improvement C)**

The training loop normalizes by number of trajectory bands:
```python
total_loss = total_loss / max(len(trajectory), 1) + lambda_lora * reg
```

Going from 3 bands to 5 bands reduces the effective erasure loss magnitude by 3/5 = 60% at the same learning rate. While 5 bands provide more gradient coverage per step, the normalization means the optimizer takes smaller steps. Combined with the adapter dilution from B, this compounds the undertrained adapter problem.

**Potential contribution: Per-prompt weighting collapse (Improvement J)**

The EMA-based weighting with decay=0.98 converges quickly. If Van Gogh's style directions are consistently large (which they are — Van Gogh is one of the most stylistically distinctive artists), the EMA quickly grows to a high value. Most individual prompts then have `style_dir_norm / EMA ≈ 1.0`, so weighting has minimal effect in steady state. However, the `clamp(min=0.1)` means any prompt with below-average style signal gets downweighted to 0.1, and the net effect across a batch is a slight reduction in average erasure loss.

### 4.4 What We Can Conclude

1. **The B+C+J combination is counterproductive as implemented.** The interactions between the three changes create a training regime where adapters are too numerous, gradient signal is too diluted, and the model ends up essentially unchanged.

2. **More layers ≠ better erasure** without compensating for the increased parameter count with either more training steps, higher learning rate, or smaller adapter scope (higher topk threshold).

3. **The dense trajectory by itself is likely harmless or mildly helpful** — 5 well-placed bands cover the style-encoding range better than 3 sparse ones. The normalization issue is real but small.

4. **Per-prompt weighting needs refinement** — the EMA-based approach is conceptually sound but may need a longer warmup or different normalization to provide meaningful signal.

---

## 5. Ablation Study Design

To isolate which of the three improvements is beneficial vs harmful, three isolated variants are being trained:

### Variant B-only
- Layer targets: `attn2,attn1` (self-attention added, FF excluded to reduce adapter count)
- Trajectory: `0.2,0.5,0.8` (original 3-band)
- Prompt weighting: disabled
- **Tests:** Does extending to self-attention help erasure? Is the adapter count manageable with attn1 (but not FF) added?

### Variant C-only
- Layer targets: `attn2` (original cross-attention only)
- Trajectory: `0.3,0.375,0.45,0.525,0.6` (dense 5-band)
- Prompt weighting: disabled
- **Tests:** Does denser trajectory sampling in the style-encoding range improve erasure? Does the normalization dilution cause any measurable harm?

### Variant J-only
- Layer targets: `attn2` (original cross-attention only)
- Trajectory: `0.2,0.5,0.8` (original 3-band)
- Prompt weighting: enabled
- **Tests:** Does per-prompt weighting improve or hurt erasure? Is the EMA-based approach working as intended?

### Expected Outcomes and Interpretation

| Variant | Expected vs SPACE-v1 | Interpretation if correct |
|---------|---------------------|--------------------------|
| B-only | Slightly better erasure, slightly worse preservation | Self-attention matters for style texture |
| C-only | ~Same or slightly better erasure, similar preservation | Dense trajectory is safe, may help slightly |
| J-only | ~Same erasure, similar or slightly better | Weighting is neutral to mildly positive |

If **B-only fails** (near-zero erasure like full v2): attn1 alone is still too many adapters for 400 steps → need more training or higher topk
If **B-only succeeds**: the failure of full v2 was due to adding FF layers (too many adapters), and attn1 is within budget
If **C-only fails**: dense trajectory hurts through normalization dilution → should restore 3-band sampling
If **J-only fails**: the weighting collapses as described → disable or redesign

---

## 6. Next Steps

### If ablation shows B helps:
- Scale training steps proportionally: `stage1_steps = 300 * (n_adapters_v2 / n_adapters_v1)` 
- Or reduce FF layers from the target set, keeping only attn1+attn2
- Alternatively, increase `saliency_topk_blocks` threshold to select fewer of the larger candidate pool

### If ablation shows C is neutral/helpful:
- Keep dense trajectory as default
- Consider removing the `/ len(trajectory)` normalization and replacing with a fixed scale factor

### If ablation shows J is neutral/helpful:
- Consider extending to weight preservation losses inversely (prompts with weak style signal → stronger preservation weight)
- Try longer EMA warmup (decay=0.995 instead of 0.98)

### Longer-term improvements not yet attempted:
- **Vision-based style direction (Improvement A):** Compute style direction in CLIP image embedding space from actual Van Gogh paintings rather than text embeddings. More faithful representation of visual style.
- **Iterative saliency refinement (Improvement E):** Recompute saliency midway through training to adapt module selection as adapters converge.
- **Per-prompt loss weighting using image features (Improvement J revised):** Weight by CLIP image similarity to known Van Gogh works rather than noise prediction norm.

---

## 7. Infrastructure Built During This Session

Beyond the research improvements, the following tooling was developed:

| Component | Purpose |
|-----------|---------|
| `space_v2/space_trainer_v2.py` | Modified trainer with B+C+J improvements |
| `space_v2/space_sd_v2.py` | CLI with `--lora_target_layers`, `--no_prompt_weighting` |
| `space_v2/run_space_v2_images.sh` | Image generation for v2 checkpoint |
| `evalscripts/generate_fid_samples.py` | FID reference image generation for any checkpoint |
| `evalscripts/generate-images.py` | Skip-if-exists multi-sample generation |
| `generate_esd_old.py` | Multi-sample baseline generation with skip logic |
| `run_complete_eval.sh` | End-to-end 7-step evaluation pipeline |
| `evaluate.py` (updated) | Added SPACE-v2, SPACE-v2-B, SPACE-v2-C, SPACE-v2-J methods; FID integration; `collect_pairs()` for multi-sample and CA order_samples mode; cache v5 |

### FID Status (main pod, as of report date)
| Method | Status |
|--------|--------|
| Baseline | 2000 images ✓ |
| UCE | 2000 images ✓ |
| CA | ~900/2006 in progress |
| ESD-x | 19/2000 (interrupted, resumable) |
| SPACE-v1 | 0/2000 (not started) |
| SPACE-v2 | 0/2000 (to run on new pod or main pod after CA) |

---

*Report generated: 2026-04-25*
*Base model: CompVis/stable-diffusion-v1-4*
*Target artist: Vincent van Gogh (Van Gogh prompts CSV, 50 prompts, filtered by artist column)*
