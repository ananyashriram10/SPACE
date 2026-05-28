#!/usr/bin/env python3
"""
Generate images from SD v1.4, optionally with a pre-trained ESD-x UNet checkpoint.

Usage:
  # Baseline (no ESD weights)
  python generate_esd_old.py \
    --prompts_path data/kelly_prompts.csv \
    --save_path results/baseline/kelly

  # Erased
  python generate_esd_old.py \
    --esd_path esd-weights/art/diffusers-KellyMcKernan-ESDx1-UNET.pt \
    --prompts_path data/kelly_prompts.csv \
    --save_path results/erased/kelly
"""
import torch
import pandas as pd
import argparse
import os
import sys
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = os.path.basename(args.save_path.rstrip("/"))

    print(f"\n{'='*60}")
    print(f"  Job: {label}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    print(f"  [1/3] Loading SD v1.4 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)  # suppress per-image denoising bars
    print(f"  [1/3] Pipeline loaded.")

    if args.esd_path:
        print(f"  [2/3] Loading ESD-x weights from: {args.esd_path}")
        # Load to CPU first — avoids OOM when multiple jobs share the GPU.
        # load_state_dict then patches the already-on-GPU UNet in-place (no extra VRAM).
        unet_weights = torch.load(args.esd_path, map_location="cpu", weights_only=False)
        pipe.unet.load_state_dict(unet_weights)
        del unet_weights  # free CPU RAM immediately
        pipe.unet.eval()
        print(f"  [2/3] ESD-x weights applied.")
    else:
        print(f"  [2/3] No ESD weights — running baseline.")

    print(f"  [3/3] Loading prompts from: {args.prompts_path}")
    df = pd.read_csv(args.prompts_path)

    # Filter to a specific artist if requested
    if args.artist:
        df = df[df["artist"].str.strip().str.lower() == args.artist.strip().lower()].reset_index(drop=True)
        print(f"  [3/3] Filtered to artist '{args.artist}': {len(df)} prompts")
    else:
        print(f"  [3/3] {len(df)} prompts loaded.")

    os.makedirs(args.save_path, exist_ok=True)

    num_samples = args.num_samples
    already_done = sum(
        1 for _, row in df.iterrows()
        if all(
            os.path.exists(os.path.join(args.save_path, f"{int(row['case_number'])}_{i}.png"))
            for i in range(num_samples)
        )
    )
    if already_done:
        print(f"  Resuming: {already_done}/{len(df)} cases fully done ({num_samples} samples each), skipping those.")

    print(f"\n  Steps: {args.num_inference_steps}  |  CFG: {args.guidance_scale}  |  Samples/prompt: {num_samples}  |  Saving to: {args.save_path}\n")

    with tqdm(df.iterrows(), total=len(df), desc=label, unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for _, row in pbar:
            case = int(row["case_number"])
            all_paths = [os.path.join(args.save_path, f"{case}_{i}.png") for i in range(num_samples)]
            if all(os.path.exists(p) for p in all_paths):
                pbar.set_postfix(status="skipped")
                continue

            seed = int(row["evaluation_seed"])
            prompt = str(row["prompt"])
            pbar.set_postfix(case=case, seed=seed)

            generator = torch.Generator(device=device).manual_seed(seed)
            images = pipe(
                [prompt] * num_samples,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images
            for i, image in enumerate(images):
                if not os.path.exists(all_paths[i]):
                    image.save(all_paths[i])

    print(f"\n  Done. {len(df)} cases × {num_samples} samples saved to: {args.save_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate ESD comparison images")
    p.add_argument("--esd_path", default=None,
                   help="Path to ESD-x .pt UNet checkpoint (omit for baseline)")
    p.add_argument("--prompts_path", required=True,
                   help="CSV with columns: case_number, prompt, evaluation_seed, artist")
    p.add_argument("--save_path", required=True,
                   help="Directory to save output PNG images")
    p.add_argument("--artist", default=None,
                   help="Filter CSV to this artist name only (exact match, case-insensitive)")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of images to generate per prompt (default: 1)")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM inference steps (default: 50, paper uses 50)")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Classifier-free guidance scale (default: 7.5)")
    generate(p.parse_args())
