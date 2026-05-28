"""
Optuna hyperparameter search for SPACE-v1 with W&B logging.

Objective: maximise erasure (low style_target_rate) while keeping
           content preservation acceptable (dino_similarity >= 0.55).

Usage (from the erasing/ directory):
    pip install optuna wandb weave
    python optuna_space_v1.py \
        --erase_concept "Van Gogh" \
        --target_prompts_path data/vangogh_prompts.csv \
        --target_artist_filter "Van Gogh" \
        --wandb_api_key <your_key> \
        --n_trials 30 \
        --train_steps 150 \
        --eval_prompts 20 \
        --storage sqlite:///space_hpo.db
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── evaluation models (loaded once, reused across trials) ─────────────────────
_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None


def _load_eval_models(device: str):
    global _clip_model, _clip_processor, _dino_model, _dino_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        print("Loading CLIP for eval …")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel

        print("Loading DINOv2 for eval …")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")


# ── evaluation helpers ────────────────────────────────────────────────────────

def _clip_style_sim(images, style_text: str, device: str) -> float:
    inputs = _clip_processor(
        text=[style_text],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        out = _clip_model(**inputs)
    img_feats = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_feats = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_feats @ txt_feats.T).mean().item())


def _dino_sim_batch(images_a, images_b, device: str) -> float:
    def _feats(imgs):
        inputs = _dino_processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = _dino_model(**inputs)
        f = out.last_hidden_state[:, 0]
        return f / f.norm(dim=-1, keepdim=True)

    fa = _feats(images_a)
    fb = _feats(images_b)
    return float((fa * fb).sum(dim=-1).mean().item())


def _generate_images(pipe, prompts, n_steps, device, seed=42):
    images = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            out = pipe(prompt, num_inference_steps=n_steps, guidance_scale=7.5, generator=gen)
        images.append(out.images[0])
    return images


def _run_mini_eval(checkpoint_path, eval_prompts, style_text, style_threshold,
                   eval_steps, device, base_model_id, vanilla_images) -> dict:
    from diffusers import StableDiffusionPipeline
    from utils.esd_checkpoint import apply_esd_checkpoint

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    apply_esd_checkpoint(pipe, checkpoint_path, device=device)

    erased_images = _generate_images(pipe, eval_prompts, eval_steps, device)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    style_sims = [_clip_style_sim([img], style_text, device) for img in erased_images]
    style_target_rate = float(sum(s > style_threshold for s in style_sims) / len(style_sims))
    dino_similarity = _dino_sim_batch(vanilla_images, erased_images, device)

    return {
        "style_target_rate": style_target_rate,
        "dino_similarity": dino_similarity,
        "mean_style_sim": float(sum(style_sims) / len(style_sims)),
        "erased_images": erased_images,
    }


def _generate_vanilla_images(prompts, base_model_id, eval_steps, device):
    from diffusers import StableDiffusionPipeline

    print(f"Generating {len(prompts)} vanilla reference images …")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    images = _generate_images(pipe, prompts, eval_steps, device)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images


def _load_eval_prompts(target_prompts_path, target_artist_filter, n) -> list[str]:
    import pandas as pd
    import random

    df = pd.read_csv(target_prompts_path)
    if target_artist_filter and "artist" in df.columns:
        df = df[df["artist"].astype(str).str.strip().str.lower() == target_artist_filter.strip().lower()]
    prompts = df["prompt"].dropna().tolist()
    if len(prompts) > n:
        random.seed(0)
        prompts = random.sample(prompts, n)
    return [str(p) for p in prompts]


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _wandb_log_trial(trial_number, params, metrics, score, eval_prompts,
                     vanilla_images, erased_images, wandb_project, erase_concept):
    import wandb

    run = wandb.init(
        project=wandb_project,
        name=f"trial_{trial_number:03d}",
        group=f"optuna_{erase_concept.replace(' ', '_')}",
        config={**params, "trial_number": trial_number},
        reinit=True,
    )

    run.log({
        "score": score,
        "style_target_rate": metrics["style_target_rate"],
        "dino_similarity": metrics["dino_similarity"],
        "mean_style_sim": metrics["mean_style_sim"],
        "preservation_penalty": max(0.0, 0.50 - metrics["dino_similarity"]) * 3.0,
        **{f"param/{k}": v for k, v in params.items()},
    })

    # Log a side-by-side image grid (vanilla vs erased) for the first 6 prompts
    try:
        import wandb
        panels = []
        for i, (prompt, vanilla, erased) in enumerate(
            zip(eval_prompts[:6], vanilla_images[:6], erased_images[:6])
        ):
            panels.append(wandb.Image(vanilla, caption=f"[vanilla] {prompt[:60]}"))
            panels.append(wandb.Image(erased, caption=f"[erased] {prompt[:60]}"))
        run.log({"image_comparison": panels})
    except Exception:
        pass

    run.finish()


def _wandb_log_summary(study, wandb_project, erase_concept):
    import wandb
    import pandas as pd

    run = wandb.init(
        project=wandb_project,
        name="hpo_summary",
        group=f"optuna_{erase_concept.replace(' ', '_')}",
        job_type="summary",
        reinit=True,
    )

    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append({
            "trial": t.number,
            "score": t.value,
            "style_target_rate": t.user_attrs.get("style_target_rate"),
            "dino_similarity": t.user_attrs.get("dino_similarity"),
            **t.params,
        })
    if rows:
        table = wandb.Table(dataframe=pd.DataFrame(rows))
        run.log({"all_trials": table})

    best = study.best_trial
    run.summary.update({
        "best_score": best.value,
        "best_style_target_rate": best.user_attrs.get("style_target_rate"),
        "best_dino_similarity": best.user_attrs.get("dino_similarity"),
        **{f"best/{k}": v for k, v in best.params.items()},
    })
    run.finish()


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(args, eval_prompts, vanilla_images, style_text, style_threshold, tmp_dir):
    from utils.space_trainer import SPACEConfig, run_space_training

    def objective(trial):
        erase_scale          = trial.suggest_float("erase_scale",          0.8,  2.5)
        alpha_art            = trial.suggest_float("alpha_art",            0.10, 0.55)
        lambda_preserve      = trial.suggest_float("lambda_preserve",      0.05, 0.60)
        lambda_style         = trial.suggest_float("lambda_style",         0.3,  1.5)
        lambda_content       = trial.suggest_float("lambda_content",       0.5,  2.0)
        saliency_topk_blocks = trial.suggest_float("saliency_topk_blocks", 0.15, 0.40)

        params = {
            "erase_scale": erase_scale,
            "alpha_art": alpha_art,
            "lambda_preserve": lambda_preserve,
            "lambda_style": lambda_style,
            "lambda_content": lambda_content,
            "saliency_topk_blocks": saliency_topk_blocks,
        }

        config = SPACEConfig(
            base_model_id=args.base_model_id,
            erase_concept=args.erase_concept,
            target_prompts_path=args.target_prompts_path,
            target_artist_filter=args.target_artist_filter,
            preserve_prompts_path=args.preserve_prompts_path,
            neutral_art_template="a high quality painting of {content}",
            neutral_image_template="a high quality image of {content}",
            style_basis_rank=4,
            saliency_topk_blocks=saliency_topk_blocks,
            trajectory_bands="0.2,0.5,0.8",
            robust_prompt_mode="full",
            vulnerable_preserve_k=3,
            lora_rank=8,
            stage1_steps=int(args.train_steps * 0.75),
            stage2_steps=int(args.train_steps * 0.25),
            lr=1e-4,
            num_inference_steps=50,
            guidance_scale=3.0,
            resolution=None,
            alpha_art=alpha_art,
            erase_scale=erase_scale,
            lambda_style=lambda_style,
            lambda_content=lambda_content,
            lambda_image=0.5,
            lambda_preserve=lambda_preserve,
            lambda_vulnerable=0.30,
            lambda_lora=1e-4,
            preserve_limit=128,
            exp_name=f"optuna_trial_{trial.number}",
            save_path=tmp_dir,
            device=args.device,
            torch_dtype=torch.bfloat16,
            gradient_checkpointing=False,
            allow_tf32=True,
            debug_steps=0,
        )

        try:
            checkpoint_path = run_space_training(config)
        except Exception as e:
            print(f"Trial {trial.number} training failed: {e}")
            return 1.0

        gc.collect()
        torch.cuda.empty_cache()

        try:
            metrics = _run_mini_eval(
                checkpoint_path=checkpoint_path,
                eval_prompts=eval_prompts,
                style_text=style_text,
                style_threshold=style_threshold,
                eval_steps=args.eval_steps,
                device=args.device,
                base_model_id=args.base_model_id,
                vanilla_images=vanilla_images,
            )
        except Exception as e:
            print(f"Trial {trial.number} eval failed: {e}")
            return 1.0

        style_target_rate = metrics["style_target_rate"]
        dino_similarity   = metrics["dino_similarity"]
        preservation_penalty = max(0.0, 0.50 - dino_similarity) * 3.0
        score = style_target_rate + preservation_penalty

        trial.set_user_attr("style_target_rate", style_target_rate)
        trial.set_user_attr("dino_similarity",   dino_similarity)
        trial.set_user_attr("mean_style_sim",    metrics["mean_style_sim"])
        trial.set_user_attr("checkpoint",        checkpoint_path)

        # W&B logging per trial
        if args.wandb_project:
            try:
                _wandb_log_trial(
                    trial_number=trial.number,
                    params=params,
                    metrics=metrics,
                    score=score,
                    eval_prompts=eval_prompts,
                    vanilla_images=vanilla_images,
                    erased_images=metrics["erased_images"],
                    wandb_project=args.wandb_project,
                    erase_concept=args.erase_concept,
                )
            except Exception as e:
                print(f"W&B logging failed for trial {trial.number}: {e}")

        print(
            f"Trial {trial.number}: score={score:.4f}  "
            f"style_rate={style_target_rate:.3f}  dino={dino_similarity:.3f}  "
            f"erase_scale={erase_scale:.2f}  alpha_art={alpha_art:.2f}  "
            f"lambda_preserve={lambda_preserve:.3f}"
        )
        return score

    return objective


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Optuna HPO for SPACE-v1 with W&B logging")
    p.add_argument("--erase_concept",         required=True)
    p.add_argument("--target_prompts_path",   required=True)
    p.add_argument("--target_artist_filter",  default=None)
    p.add_argument("--preserve_prompts_path", default="data/coco_30k.csv")
    p.add_argument("--base_model_id",         default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--n_trials",    type=int,   default=30)
    p.add_argument("--train_steps", type=int,   default=150,
                   help="Total training steps per trial")
    p.add_argument("--eval_prompts", type=int,  default=20,
                   help="Number of prompts for mini eval per trial")
    p.add_argument("--eval_steps",   type=int,  default=20,
                   help="DDIM steps for eval image generation")
    p.add_argument("--style_threshold", type=float, default=0.24,
                   help="CLIP sim threshold for 'style still present'")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--study_name",  default="space_v1_hpo")
    p.add_argument("--storage",     default=None,
                   help="Optuna DB URL e.g. sqlite:///space_hpo.db")
    # W&B args
    p.add_argument("--wandb_project",  default="SPACE",
                   help="W&B project name")
    p.add_argument("--wandb_api_key",  default=None,
                   help="W&B API key (or set WANDB_API_KEY env var)")
    p.add_argument("--no_wandb",       action="store_true",
                   help="Disable W&B logging entirely")
    return p


def main():
    import optuna

    args = build_parser().parse_args()

    # W&B login
    if not args.no_wandb:
        import wandb
        api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        else:
            wandb.login()  # prompts interactively if no key
        print(f"W&B logging → project: {args.wandb_project}")
    else:
        args.wandb_project = None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _load_eval_models(args.device)

    eval_prompts = _load_eval_prompts(
        args.target_prompts_path, args.target_artist_filter, args.eval_prompts
    )
    print(f"Eval prompts: {len(eval_prompts)}")

    style_text = f"artwork in the style of {args.erase_concept}"
    vanilla_images = _generate_vanilla_images(
        eval_prompts, args.base_model_id, args.eval_steps, args.device
    )

    vanilla_style_sims = [_clip_style_sim([img], style_text, args.device) for img in vanilla_images]
    vanilla_mean = sum(vanilla_style_sims) / len(vanilla_style_sims)
    style_threshold = args.style_threshold or vanilla_mean * 0.80
    print(f"Vanilla mean style sim: {vanilla_mean:.4f}  threshold: {style_threshold:.4f}")

    # Log vanilla reference images to W&B once
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name="vanilla_reference",
                group=f"optuna_{args.erase_concept.replace(' ', '_')}",
                job_type="reference",
                reinit=True,
            )
            run.log({
                "vanilla_style_sim_mean": vanilla_mean,
                "style_threshold": style_threshold,
                "vanilla_images": [
                    wandb.Image(img, caption=p[:80])
                    for img, p in zip(vanilla_images[:6], eval_prompts[:6])
                ],
            })
            run.finish()
        except Exception as e:
            print(f"W&B vanilla reference logging failed: {e}")

    with tempfile.TemporaryDirectory(prefix="space_optuna_") as tmp_dir:
        objective = make_objective(args, eval_prompts, vanilla_images, style_text, style_threshold, tmp_dir)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            storage=args.storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Final W&B summary
    if args.wandb_project:
        try:
            _wandb_log_summary(study, args.wandb_project, args.erase_concept)
        except Exception as e:
            print(f"W&B summary logging failed: {e}")

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Score:            {best.value:.4f}")
    print(f"  style_target_rate {best.user_attrs.get('style_target_rate', '?'):.3f}")
    print(f"  dino_similarity   {best.user_attrs.get('dino_similarity', '?'):.3f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v:.4f}")

    print("\n=== Top 5 trials by score ===")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 9999)
    for t in trials[:5]:
        print(
            f"  #{t.number}  score={t.value:.4f}  "
            f"style_rate={t.user_attrs.get('style_target_rate', '?')}  "
            f"dino={t.user_attrs.get('dino_similarity', '?')}  "
            f"params={t.params}"
        )

    if args.storage:
        print(f"\nStudy saved to {args.storage}. Resume with --storage {args.storage}")


if __name__ == "__main__":
    main()
