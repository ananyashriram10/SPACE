#!/usr/bin/env python3
"""Generate N diverse images from coco_30k.csv for FID computation.

Saves one image per prompt to --save_path/{image_id}.png.
Skips images that already exist so re-runs are safe.
"""
import argparse
import os
import sys

import pandas as pd
import torch
from diffusers import DiffusionPipeline

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.esd_checkpoint import apply_esd_checkpoint


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def generate_fid_samples(
    base_model,
    esd_path,
    prompts_path,
    save_path,
    n_images=2000,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    guidance_scale=7.5,
    num_inference_steps=20,
    component=None,
):
    os.makedirs(save_path, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    if esd_path is not None:
        if esd_path.endswith(".pt"):
            unet_weights = torch.load(esd_path, map_location="cpu", weights_only=False)
            pipe.unet.load_state_dict(unet_weights)
            pipe.unet.eval()
            print(f"Loaded .pt checkpoint into pipe.unet")
        else:
            metadata, resolved_component, _ = apply_esd_checkpoint(
                pipe, esd_path, device="cpu", component_name=component
            )
            if metadata.get("base_model_id") and metadata["base_model_id"] != base_model:
                print(f"Warning: checkpoint was trained on {metadata['base_model_id']}, running on {base_model}")
            print(f"Loaded checkpoint into pipe.{resolved_component}")

    df = pd.read_csv(prompts_path)
    df = df.head(n_images)

    already = sum(1 for _, row in df.iterrows() if os.path.exists(os.path.join(save_path, f"{row.image_id}.png")))
    if already:
        print(f"Resuming: {already}/{len(df)} already done, skipping those.")

    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(save_path)):
        out_path = os.path.join(save_path, f"{row.image_id}.png")
        if os.path.exists(out_path):
            continue
        image = pipe(
            str(row.prompt),
            generator=make_generator(int(row.evaluation_seed)),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        image.save(out_path)

    print(f"Done. {len(df)} images in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FID reference images from COCO prompts.")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--esd_path", default=None, help="Optional checkpoint to apply")
    parser.add_argument("--component", default=None)
    parser.add_argument("--prompts_path", required=True, help="Path to coco_30k.csv")
    parser.add_argument("--save_path", required=True, help="Output directory")
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    args = parser.parse_args()

    generate_fid_samples(
        base_model=args.base_model,
        esd_path=args.esd_path,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        n_images=args.n_images,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        component=args.component,
    )
