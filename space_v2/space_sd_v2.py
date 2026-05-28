import argparse
import os
import sys

import torch

# Ensure erasing/ (parent of space_v2/) is on sys.path so utils.* imports resolve
_HERE = os.path.dirname(os.path.abspath(__file__))
_ERASING_ROOT = os.path.dirname(_HERE)
for _p in (_ERASING_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    try:
        import huggingface_hub

        huggingface_hub.login(token=_HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TrainSPACE-v2R for SD",
        description=(
            "Train SPACE-v2R for SD v1.4: a recovery path built from SPACE-v1 defaults "
            "with a stronger dual-residual teacher for artist/style erasure."
        ),
    )
    parser.add_argument(
        "--basemodel_id",
        help="HF model id for a Stable Diffusion v1.x-compatible diffusers pipeline",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--erase_concept",
        help="artist/style attribute to neutralize",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target_prompts_path",
        help="CSV with target prompts. Must contain `prompt`; may contain `artist` for filtering.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target_artist_filter",
        help="optional artist filter for target prompt CSVs with an `artist` column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--preserve_prompts_path",
        help="CSV with general non-target preservation prompts. Must contain `prompt`.",
        type=str,
        default="data/coco_30k.csv",
    )
    parser.add_argument(
        "--neutral_art_template",
        help="neutral art anchor template; must include `{content}`",
        type=str,
        default="a high quality painting of {content}",
    )
    parser.add_argument(
        "--neutral_image_template",
        help="generic image anchor template; must include `{content}`",
        type=str,
        default="a high quality image of {content}",
    )
    parser.add_argument(
        "--style_basis_rank",
        help="rank of the style residual subspace estimate",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--saliency_topk_blocks",
        help="fraction or count of projection blocks to train (applied across all targeted layers)",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--trajectory_bands",
        help="comma-separated fractions of the denoising trajectory to supervise",
        type=str,
        default="0.2,0.5,0.8",
    )
    parser.add_argument(
        "--lora_target_layers",
        help=(
            "comma-separated layer name substrings for LoRA injection. "
            "E.g. 'attn2' (v2R default), 'attn2,attn1', or 'attn2,attn1,ff' for audits."
        ),
        type=str,
        default="attn2",
    )
    parser.add_argument(
        "--use_prompt_weighting",
        dest="use_prompt_weighting",
        help="enable the failed v2 per-prompt erasure weighting ablation",
        action="store_true",
    )
    parser.add_argument(
        "--no_prompt_weighting",
        dest="use_prompt_weighting",
        help="explicitly disable per-prompt erasure loss weighting",
        action="store_false",
    )
    parser.set_defaults(use_prompt_weighting=False)
    parser.add_argument(
        "--robust_prompt_mode",
        help="prompt family expansion mode: off, light, or full",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--vulnerable_preserve_k",
        help="how many nearby artists to preserve explicitly",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--lora_rank",
        help="LoRA rank for localized SPACE adapters",
        type=int,
        default=8,
    )
    parser.add_argument("--stage1_steps", help="number of stage-1 optimization steps", type=int, default=300)
    parser.add_argument("--stage2_steps", help="number of stage-2 optimization steps", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", help="number of denoising steps", type=int, default=50)
    parser.add_argument("--guidance_scale", help="guidance scale used to sample xt", type=float, default=3.0)
    parser.add_argument(
        "--resolution",
        help="training resolution. Defaults to the base model native size.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--alpha_art",
        help="blend amount toward the neutral art anchor inside the erase target",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--erase_scale",
        help="negative style-direction strength for SPACE-v2R erasure",
        type=float,
        default=1.75,
    )
    parser.add_argument(
        "--residual_mix",
        help="blend weight for content residual versus null residual in the v2R teacher",
        type=float,
        default=0.70,
    )
    parser.add_argument("--lambda_style", type=float, default=0.8)
    parser.add_argument("--lambda_content", type=float, default=1.0)
    parser.add_argument("--lambda_image", type=float, default=0.5)
    parser.add_argument("--lambda_preserve", type=float, default=0.20)
    parser.add_argument("--lambda_vulnerable", type=float, default=0.30)
    parser.add_argument("--lambda_lora", type=float, default=1e-4)
    parser.add_argument(
        "--preserve_limit",
        help="maximum number of general preserve prompts to pre-encode",
        type=int,
        default=256,
    )
    parser.add_argument("--exp_name", help="checkpoint/output suffix, e.g. Van_Gogh", type=str, default=None)
    parser.add_argument(
        "--save_path",
        help="directory to save SPACE-v2R checkpoints",
        type=str,
        default="space-models/sd-v2r/",
    )
    parser.add_argument("--device", help="device to train on", type=str, default="cuda:0")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for template_name in ("neutral_art_template", "neutral_image_template"):
        if "{content}" not in getattr(args, template_name):
            raise ValueError(f"--{template_name} must include `{{content}}`")

    from space_trainer_v2 import SPACEConfig, run_space_training

    config = SPACEConfig(
        base_model_id=args.basemodel_id,
        erase_concept=args.erase_concept,
        target_prompts_path=args.target_prompts_path,
        target_artist_filter=args.target_artist_filter,
        preserve_prompts_path=args.preserve_prompts_path,
        neutral_art_template=args.neutral_art_template,
        neutral_image_template=args.neutral_image_template,
        style_basis_rank=args.style_basis_rank,
        saliency_topk_blocks=args.saliency_topk_blocks,
        trajectory_bands=args.trajectory_bands,
        lora_target_layers=args.lora_target_layers,
        use_prompt_weighting=args.use_prompt_weighting,
        robust_prompt_mode=args.robust_prompt_mode,
        vulnerable_preserve_k=args.vulnerable_preserve_k,
        lora_rank=args.lora_rank,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        lr=args.lr,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        alpha_art=args.alpha_art,
        erase_scale=args.erase_scale,
        residual_mix=args.residual_mix,
        lambda_style=args.lambda_style,
        lambda_content=args.lambda_content,
        lambda_image=args.lambda_image,
        lambda_preserve=args.lambda_preserve,
        lambda_vulnerable=args.lambda_vulnerable,
        lambda_lora=args.lambda_lora,
        preserve_limit=args.preserve_limit,
        exp_name=args.exp_name,
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        gradient_checkpointing=args.gradient_checkpointing,
        allow_tf32=args.allow_tf32,
        debug_steps=args.debug_steps,
    )
    checkpoint_path = run_space_training(config)
    print(f"Saved SPACE-v2R checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
