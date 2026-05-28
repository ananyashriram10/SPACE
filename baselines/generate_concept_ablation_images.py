#!/usr/bin/env python3
"""Generate images with the official Concept Ablation diffusers pipeline."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch


def make_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from an official Concept Ablation delta.bin")
    parser.add_argument("--repo_dir", type=str, default="baselines/external/concept-ablation/diffusers")
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--delta_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--artist_filter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--from_case", type=int, default=0)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not (repo_dir / "model_pipeline.py").exists():
        raise FileNotFoundError(
            f"Could not find official Concept Ablation diffusers code at {repo_dir}. "
            "Run `bash baselines/setup_official_repos.sh` first."
        )
    if not os.path.exists(args.delta_path):
        raise FileNotFoundError(f"Missing Concept Ablation delta: {args.delta_path}")

    sys.path.insert(0, str(repo_dir))
    from model_pipeline import CustomDiffusionPipeline

    pipe = CustomDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if str(args.device).startswith("cuda") else torch.float32,
        safety_checker=None,
    ).to(args.device)
    pipe.load_model(args.delta_path)
    pipe.set_progress_bar_config(disable=True)

    df = pd.read_csv(args.prompts_path)
    if args.artist_filter and "artist" in df.columns:
        df = df[df["artist"].str.strip().str.lower() == args.artist_filter.strip().lower()].reset_index(drop=True)
        print(f"Filtered to artist '{args.artist_filter}': {len(df)} prompts")

    out_dir = os.path.join(args.save_path, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        case_number = int(row.case_number)
        if case_number < args.from_case:
            continue

        prompt = [str(row.prompt)] * args.num_samples
        seed = int(row.evaluation_seed)
        images = pipe(
            prompt,
            generator=make_generator(args.device, seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
        ).images
        for sample_idx, image in enumerate(images):
            image.save(os.path.join(out_dir, f"{case_number}_{sample_idx}.png"))


if __name__ == "__main__":
    main()
