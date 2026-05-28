# SPACE Research Objective

SPACE stands for **Style-Preserving Attribute-Constrained Erasure**.

The first paper scope is artist/style erasure for SD v1.4. The goal is not artist replacement. Given a prompt that contains a target artist or style, the edited model should suppress the recognizable target style while preserving the prompt's semantic content, nearby non-target styles, and general image quality.

## Expected Behavior

For a prompt such as:

```text
Bedroom in Arles by Vincent van Gogh
```

SPACE should produce a plausible bedroom scene in a neutral, non-attributed visual style. The output should still look like a bedroom. It should not intentionally become a Monet, Rembrandt, or other replacement-artist image.

Anchor concepts such as `art`, `painting`, or `high quality image` may be useful inside the training objective, but they are training mechanisms. The paper-facing objective is neutralized style erasure with high preservation.

## Main Claim

SPACE improves the erasure-preservation tradeoff for artist/style unlearning by:

- factorizing target style attributes away from prompt content
- explicitly preserving vulnerable neighboring styles
- constraining edits across the diffusion trajectory
- testing robustness against paraphrased and adversarial style prompts

The claim should be measurable output-level target-style suppression with preservation, not legal or literal deletion of all latent knowledge from model weights.

## Non-Goals For V1

- broad NSFW or safety unlearning
- object erasure
- artist-to-artist replacement
- style transfer
- claims of legally complete data deletion

## Method Direction

The current implementation target is SPACE-v1. It uses:

- target prompt: `content + target style`
- content anchor: target style phrase removed
- neutral art anchor: `a high quality painting of {content}`
- neutral image anchor: `a high quality image of {content}`
- preservation anchors: nearby artists, unrelated artists, and COCO/general prompts

Next method improvements should focus on:

- vulnerable preservation sets selected from styles most likely to be damaged by erasing the target
- trajectory-aware preservation losses over multiple denoising timesteps
- robustness prompts including artwork titles, paraphrases, `inspired by`, `as painted by`, misspellings, and CLIP-near target prompts
- clear separation between official-replication outputs and fair-eval outputs

## Evaluation Success Condition

SPACE should be considered successful only if it:

- matches or beats ESD-x, UCE, and Concept Ablation on target style erasure
- improves content preservation on target prompts
- reduces damage to nearby non-target artists/styles
- preserves global model quality on COCO/general prompts
- remains robust to paraphrases and indirect artist/style references

Core metrics:

- target erasure: artist/style CLIP score, CLIP classifier accuracy, CLIP drop from vanilla
- content preservation: CLIP image similarity, LPIPS, DINO or DreamSim where available
- nearby-style preservation: non-target artist score/classifier accuracy
- global quality: FID/KID, COCO CLIP score, ResNet top-1/top-5 agreement
- robustness: adversarial/paraphrased prompt erasure rate

## Baseline Gating

SPACE development should pause as the main focus until the current baselines are validated:

- ESD-x replication
- official UCE
- official Concept Ablation CompVis

After smoke checks and metric sanity checks pass, SPACE improvements should be benchmarked against these baselines before adding MACE, SPM, Receler, SalUn, Forget-Me-Not, or other methods.

## Related Work Pointers

- ESD: https://erasing.baulab.info/
- Concept Ablation: https://www.cs.cmu.edu/~concept-ablation/
- UCE: https://unified.baulab.info/
- MACE: https://arxiv.org/abs/2403.06135
- SPEED: https://arxiv.org/abs/2503.07392
- UnlearnCanvas: https://unlearn-canvas.netlify.app/
- Memories of Forgotten Concepts: https://arxiv.org/abs/2412.00782
- Microsoft unlearning policy caution: https://www.microsoft.com/en-us/research/publication/machine-unlearning-doesnt-do-what-you-think-lessons-for-generative-ai-policy-research-and-practice-tr/
