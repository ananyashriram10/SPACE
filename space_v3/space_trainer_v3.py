from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.esd_checkpoint import load_esd_checkpoint, save_esd_checkpoint
from utils.esd_trainer import (
    _suppress_transformers_pipeline_load_noise,
    load_sd_pipeline,
    make_sampling_generator,
    normalize_train_method,
    offload_modules_to_cpu,
    resolve_default_resolution,
    sanitize_checkpoint_name,
    set_module,
)
from utils.sd_utils import esd_sd_call


ARTIST_BANK = [
    "Monet",
    "Rembrandt",
    "Warhol",
    "Picasso",
    "Cezanne",
    "Matisse",
    "Kandinsky",
    "Renoir",
]


@dataclass
class SPACEConfig:
    base_model_id: str
    erase_concept: str
    target_prompts_path: str
    target_artist_filter: Optional[str]
    preserve_prompts_path: Optional[str]
    neutral_art_template: str
    neutral_image_template: str
    style_basis_rank: int
    saliency_topk_blocks: float
    trajectory_bands: str
    lora_target_layers: str
    use_prompt_weighting: bool
    robust_prompt_mode: str
    vulnerable_preserve_k: int
    lora_rank: int
    stage1_steps: int
    stage2_steps: int
    lr: float
    num_inference_steps: int
    guidance_scale: float
    resolution: Optional[int]
    alpha_art: float
    alpha_img: float
    erase_scale: float
    residual_mix: float
    style_gate_mode: str
    style_gate_min: float
    style_gate_max: float
    lambda_style: float
    lambda_content: float
    lambda_image: float
    lambda_preserve: float
    lambda_fid_preserve: float
    lambda_vulnerable: float
    lambda_lora: float
    preserve_null_interval: int
    preserve_null_projection: bool
    preserve_limit: int
    exp_name: Optional[str]
    save_path: str
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.bfloat16
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    debug_steps: int = 0


@dataclass
class PromptVariant:
    target: str
    content: str
    art_anchor: str
    image_anchor: str


@dataclass
class SPACEFamily:
    prompt: str
    variants: list[PromptVariant]
    preserve_variants: list[str]


@dataclass
class EncodedVariant:
    target: str
    content: str
    art_anchor: str
    image_anchor: str
    target_embeds: torch.Tensor
    content_embeds: torch.Tensor
    art_embeds: torch.Tensor
    image_embeds: torch.Tensor


@dataclass
class EncodedFamily:
    prompt: str
    variants: list[EncodedVariant]
    preserve_contexts: list[Dict[str, torch.Tensor | str]]


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,;:])+", r"\1", text)
    text = re.sub(r"^[,;:\-\s]+|[,;:\-\s]+$", "", text)
    return text or "a high quality image"


def strip_concept_from_prompt(prompt: str, concept: str) -> str:
    escaped = re.escape(concept.strip())
    patterns = [
        rf"\s+in\s+the\s+style\s+of\s+{escaped}",
        rf"\s+in\s+{escaped}\s+style",
        rf"\s+by\s+{escaped}",
        rf"\s+from\s+{escaped}",
        rf"\s+with\s+(?:a\s+)?{escaped}\s+touch",
        rf"{escaped}\s*[- ]\s*inspired\s+",
        rf"\s+inspired\s+by\s+{escaped}",
        rf"\s+as\s+painted\s+by\s+{escaped}",
        rf"\s+as\s+drawn\s+by\s+{escaped}",
    ]
    content = prompt
    for pattern in patterns:
        content = re.sub(pattern, " ", content, flags=re.IGNORECASE)
    return _normalize_spaces(content)


def build_prompt_variants(prompt: str, concept: str, mode: str) -> list[str]:
    prompt = _normalize_spaces(prompt)
    variants = [prompt]
    if mode == "off":
        return variants

    concept_regex = re.compile(re.escape(concept), flags=re.IGNORECASE)
    by_regex = re.compile(rf"\bby\s+{re.escape(concept)}\b", flags=re.IGNORECASE)
    style_regex = re.compile(rf"\bin the style of\s+{re.escape(concept)}\b", flags=re.IGNORECASE)

    transformed = []
    if by_regex.search(prompt):
        transformed.append(by_regex.sub(f"in the style of {concept}", prompt))
        transformed.append(by_regex.sub(f"inspired by {concept}", prompt))
        transformed.append(by_regex.sub(f"as painted by {concept}", prompt))
    if style_regex.search(prompt):
        transformed.append(style_regex.sub(f"by {concept}", prompt))
        transformed.append(style_regex.sub(f"inspired by {concept}", prompt))
    if concept_regex.search(prompt):
        transformed.append(concept_regex.sub(concept.replace("Vincent ", ""), prompt))

    if mode == "full":
        transformed.extend(
            [
                f"{prompt}, inspired by {concept}",
                f"{prompt}, as painted by {concept}",
            ]
        )

    deduped = []
    seen = set()
    for item in variants + transformed:
        clean = _normalize_spaces(item)
        if clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def build_anchor_prompt(content_prompt: str, template: str) -> str:
    return _normalize_spaces(template.format(content=content_prompt))


def encode_prompt(pipe, prompt: str, config: SPACEConfig) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            device=config.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt="",
        )
    return prompt_embeds.to(config.device)


def encode_null_prompt(pipe, config: SPACEConfig) -> torch.Tensor:
    with torch.no_grad():
        _, null_embeds = pipe.encode_prompt(
            prompt="",
            device=config.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
    return null_embeds.to(config.device)


def load_preserve_prompts(config: SPACEConfig) -> list[str]:
    if not config.preserve_prompts_path:
        return []
    df = pd.read_csv(config.preserve_prompts_path)
    if "prompt" not in df.columns:
        raise ValueError(f"{config.preserve_prompts_path} must contain a `prompt` column.")
    prompts = [str(prompt) for prompt in df["prompt"].dropna().tolist()]
    if config.preserve_limit > 0:
        prompts = prompts[: config.preserve_limit]
    return prompts


def encode_preserve_prompts(pipe, prompts: list[str], config: SPACEConfig) -> list[Dict[str, torch.Tensor | str]]:
    return [{"prompt": prompt, "embeds": encode_prompt(pipe, prompt, config)} for prompt in prompts]


def mean_text_embedding(embeds: torch.Tensor) -> torch.Tensor:
    pooled = embeds.float().mean(dim=1)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def mine_vulnerable_artists(pipe, concept: str, config: SPACEConfig) -> list[str]:
    candidates = [artist for artist in ARTIST_BANK if artist.lower() != concept.lower()]
    concept_embed = mean_text_embedding(encode_prompt(pipe, f"an artwork in the style of {concept}", config))
    scored = []
    for artist in candidates:
        artist_embed = mean_text_embedding(encode_prompt(pipe, f"an artwork in the style of {artist}", config))
        score = float((concept_embed @ artist_embed.T).item())
        scored.append((score, artist))
    scored.sort(reverse=True)
    return [artist for _score, artist in scored[: max(config.vulnerable_preserve_k, 0)]]


def load_target_families(pipe, config: SPACEConfig) -> list[SPACEFamily]:
    df = pd.read_csv(config.target_prompts_path)
    if config.target_artist_filter and "artist" in df.columns:
        df = df[df["artist"].astype(str).str.strip().str.lower() == config.target_artist_filter.strip().lower()]
    if "prompt" not in df.columns:
        raise ValueError(f"{config.target_prompts_path} must contain a `prompt` column.")

    nearby_artists = mine_vulnerable_artists(pipe, config.erase_concept, config)
    families = []
    for prompt in [str(prompt) for prompt in df["prompt"].dropna().tolist()]:
        variants = []
        for target_prompt in build_prompt_variants(prompt, config.erase_concept, config.robust_prompt_mode):
            content = strip_concept_from_prompt(target_prompt, config.erase_concept)
            variants.append(
                PromptVariant(
                    target=target_prompt,
                    content=content,
                    art_anchor=build_anchor_prompt(content, config.neutral_art_template),
                    image_anchor=build_anchor_prompt(content, config.neutral_image_template),
                )
            )
        preserve_variants = []
        for artist in nearby_artists:
            base_content = strip_concept_from_prompt(prompt, config.erase_concept)
            preserve_variants.append(_normalize_spaces(f"{base_content} by {artist}"))
            preserve_variants.append(_normalize_spaces(f"{base_content} in the style of {artist}"))
        families.append(SPACEFamily(prompt=prompt, variants=variants, preserve_variants=preserve_variants))
    if not families:
        raise ValueError("No SPACE target prompts were found.")
    return families


def get_timestep_cond(pipe, config: SPACEConfig) -> Optional[torch.Tensor]:
    if pipe.unet.config.time_cond_proj_dim is None:
        return None
    guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(1)
    return pipe.get_guidance_scale_embedding(
        guidance_scale_tensor,
        embedding_dim=pipe.unet.config.time_cond_proj_dim,
    ).to(device=config.device, dtype=config.torch_dtype)


def sample_xt(pipe, prompt_embeds, null_embeds, run_till_timestep, seed, resolution, config):
    return esd_sd_call(
        pipe,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=null_embeds,
        num_images_per_prompt=1,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        run_till_timestep=run_till_timestep,
        generator=make_sampling_generator(config.device, seed),
        output_type="latent",
        height=resolution,
        width=resolution,
    ).images


def predict_noise(module: nn.Module, xt, timestep, prompt_embeds, timestep_cond):
    return module(
        xt,
        timestep,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=False,
    )[0]


def make_space_teacher(
    base_target: torch.Tensor,
    base_content: torch.Tensor,
    base_art: torch.Tensor,
    base_image: torch.Tensor,
    base_null: torch.Tensor,
    config: SPACEConfig,
) -> torch.Tensor:
    alpha_art = min(max(float(config.alpha_art), 0.0), 1.0)
    alpha_img = min(max(float(config.alpha_img), 0.0), max(0.0, 1.0 - alpha_art))
    content_weight = max(0.0, 1.0 - alpha_art - alpha_img)
    neutral_anchor = content_weight * base_content + alpha_art * base_art + alpha_img * base_image
    residual_mix = min(max(float(config.residual_mix), 0.0), 1.0)
    content_residual = base_target - base_content
    null_residual = base_target - base_null
    style_direction = residual_mix * content_residual + (1.0 - residual_mix) * null_residual
    style_gate = 1.0
    if config.style_gate_mode == "residual_norm":
        content_norm = content_residual.float().pow(2).mean().sqrt()
        null_norm = null_residual.float().pow(2).mean().sqrt().clamp_min(1e-8)
        style_gate = float((content_norm / null_norm).detach().item())
    style_gate = min(max(style_gate, float(config.style_gate_min)), float(config.style_gate_max))
    return neutral_anchor - config.erase_scale * style_gate * style_direction


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / max(rank, 1)
        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        self.lora_down = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=torch.float32))
        self.lora_up = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)
        nn.init.zeros_(self.lora_up)
        self.to(device=device)
        self.adapters_enabled = True

    def forward(self, x, *args, **kwargs):
        base_scale = kwargs.get("scale", None)
        if base_scale is not None:
            try:
                base_out = self.base(x, *args, **kwargs)
            except TypeError:
                kwargs = dict(kwargs)
                kwargs.pop("scale", None)
                base_out = self.base(x, *args, **kwargs)
        else:
            base_out = self.base(x, *args, **kwargs)
        if not self.adapters_enabled:
            return base_out
        lora_down = self.lora_down.to(device=x.device, dtype=x.dtype)
        lora_up = self.lora_up.to(device=x.device, dtype=x.dtype)
        delta = F.linear(F.linear(x, lora_down), lora_up)
        return base_out + self.scaling * delta

    def merged_weight(self) -> torch.Tensor:
        base = self.base.weight.detach()
        merged = base.float() + self.scaling * (self.lora_up.float() @ self.lora_down.float())
        return merged.to(dtype=base.dtype)

    def merged_bias(self) -> Optional[torch.Tensor]:
        return self.base.bias.detach() if self.base.bias is not None else None


def select_candidate_linears(unet: nn.Module, layer_targets: str = "attn2") -> list[str]:
    """Return names of Linear layers whose module path contains any of the layer_targets substrings."""
    targets = [t.strip() for t in layer_targets.split(",") if t.strip()]
    candidates = []
    for module_name, module in unet.named_modules():
        if not any(t in module_name for t in targets):
            continue
        if module.__class__.__name__ not in {"Linear", "LoRACompatibleLinear"}:
            continue
        candidates.append(module_name)
    return candidates


def resolve_topk_modules(candidates: list[str], topk_value: float) -> int:
    if not candidates:
        return 0
    if topk_value <= 0:
        return 1
    if topk_value <= 1:
        return max(1, int(round(len(candidates) * topk_value)))
    return min(len(candidates), int(topk_value))


def trajectory_indices(config: SPACEConfig) -> list[int]:
    indices = []
    for token in str(config.trajectory_bands).split(","):
        token = token.strip()
        if not token:
            continue
        frac = float(token)
        frac = min(max(frac, 0.0), 0.999)
        indices.append(min(config.num_inference_steps - 1, int(frac * config.num_inference_steps)))
    return sorted(set(indices)) or [config.num_inference_steps // 2]


def ranked_saliency_modules(pipe, families, null_embeds, timestep_cond, resolution, config) -> list[str]:
    candidate_names = select_candidate_linears(pipe.unet, config.lora_target_layers)
    if not candidate_names:
        raise ValueError(f"No linear modules found for SPACE saliency selection (layer_targets={config.lora_target_layers!r}).")

    saliency = {name: 0.0 for name in candidate_names}
    tracked_weights = {f"{name}.weight": dict(pipe.unet.named_parameters())[f"{name}.weight"] for name in candidate_names}
    probe_families = families[: min(4, len(families))]
    band_indices = trajectory_indices(config)[: min(2, len(trajectory_indices(config)))]

    pipe.unet.zero_grad(set_to_none=True)
    pipe.unet.train()
    for family in probe_families:
        variant = family.variants[0]
        target_embeds = encode_prompt(pipe, variant.target, config)
        content_embeds = encode_prompt(pipe, variant.content, config)
        art_embeds = encode_prompt(pipe, variant.art_anchor, config)
        image_embeds = encode_prompt(pipe, variant.image_anchor, config)
        for idx in band_indices:
            seed = idx + 1234
            xt = sample_xt(pipe, target_embeds, null_embeds, idx, seed, resolution, config)
            timestep = pipe.scheduler.timesteps[idx]
            with torch.no_grad():
                base_target = predict_noise(pipe.unet, xt, timestep, target_embeds, timestep_cond).detach()
                base_content = predict_noise(pipe.unet, xt, timestep, content_embeds, timestep_cond).detach()
                base_art = predict_noise(pipe.unet, xt, timestep, art_embeds, timestep_cond).detach()
                base_image = predict_noise(pipe.unet, xt, timestep, image_embeds, timestep_cond).detach()
                base_null = predict_noise(pipe.unet, xt, timestep, null_embeds, timestep_cond).detach()
                teacher = make_space_teacher(base_target, base_content, base_art, base_image, base_null, config)
            student = predict_noise(pipe.unet, xt, timestep, target_embeds, timestep_cond)
            loss = F.mse_loss(student.float(), teacher.float())
            loss.backward()
            for name, param in tracked_weights.items():
                if param.grad is not None:
                    saliency[name[:-7]] += float(param.grad.detach().float().norm().item())
            pipe.unet.zero_grad(set_to_none=True)

    ranked = sorted(saliency.items(), key=lambda item: item[1], reverse=True)
    topk = resolve_topk_modules(candidate_names, config.saliency_topk_blocks)
    return [name for name, _score in ranked[:topk]]


def inject_lora_adapters(unet: nn.Module, module_names: list[str], rank: int) -> dict[str, LoRALinear]:
    adapters = {}
    for module_name in module_names:
        module = dict(unet.named_modules())[module_name]
        adapter = LoRALinear(module, rank=rank)
        set_module(unet, module_name, adapter)
        adapters[module_name] = adapter
    return adapters


def encode_space_families(pipe, families: list[SPACEFamily], config: SPACEConfig) -> list[EncodedFamily]:
    encoded = []
    for family in families:
        encoded_variants = []
        for variant in family.variants:
            encoded_variants.append(
                EncodedVariant(
                    target=variant.target,
                    content=variant.content,
                    art_anchor=variant.art_anchor,
                    image_anchor=variant.image_anchor,
                    target_embeds=encode_prompt(pipe, variant.target, config),
                    content_embeds=encode_prompt(pipe, variant.content, config),
                    art_embeds=encode_prompt(pipe, variant.art_anchor, config),
                    image_embeds=encode_prompt(pipe, variant.image_anchor, config),
                )
            )
        encoded.append(
            EncodedFamily(
                prompt=family.prompt,
                variants=encoded_variants,
                preserve_contexts=encode_preserve_prompts(pipe, family.preserve_variants, config),
            )
        )
    return encoded


def lora_parameters(adapters: dict[str, LoRALinear]):
    for adapter in adapters.values():
        yield adapter.lora_down
        yield adapter.lora_up


def set_adapters_enabled(adapters: dict[str, LoRALinear], enabled: bool) -> None:
    for adapter in adapters.values():
        adapter.adapters_enabled = enabled


def adapter_regularization(adapters: dict[str, LoRALinear]) -> torch.Tensor:
    penalties = []
    for adapter in adapters.values():
        penalties.append(adapter.lora_down.float().pow(2).mean())
        penalties.append(adapter.lora_up.float().pow(2).mean())
    if not penalties:
        return torch.zeros(())
    return torch.stack([pen.to(next(iter(adapters.values())).lora_down.device) for pen in penalties]).mean()


def clone_adapter_parameters(adapters: dict[str, LoRALinear]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (adapter.lora_down.detach().clone(), adapter.lora_up.detach().clone())
        for name, adapter in adapters.items()
    }


def adapter_parameter_norm(adapters: dict[str, LoRALinear]) -> torch.Tensor:
    values = []
    for adapter in adapters.values():
        values.append(adapter.lora_down.float().pow(2).mean())
        values.append(adapter.lora_up.float().pow(2).mean())
    if not values:
        return torch.zeros(())
    return torch.stack(values).mean().sqrt()


def adapter_delta_norm(adapters: dict[str, LoRALinear]) -> torch.Tensor:
    values = []
    for adapter in adapters.values():
        delta = adapter.scaling * (adapter.lora_up.float() @ adapter.lora_down.float())
        values.append(delta.pow(2).mean())
    if not values:
        return torch.zeros(())
    return torch.stack(values).mean().sqrt()


def adapter_update_norm(
    adapters: dict[str, LoRALinear],
    initial_parameters: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    values = []
    for name, adapter in adapters.items():
        initial_down, initial_up = initial_parameters[name]
        values.append((adapter.lora_down.detach().float() - initial_down.float()).pow(2).mean())
        values.append((adapter.lora_up.detach().float() - initial_up.float()).pow(2).mean())
    if not values:
        return torch.zeros(())
    return torch.stack(values).mean().sqrt()


def trainable_lora_parameters(adapters: dict[str, LoRALinear]) -> list[torch.nn.Parameter]:
    return [param for param in lora_parameters(adapters)]


def capture_gradients(params: list[torch.nn.Parameter]) -> list[Optional[torch.Tensor]]:
    return [None if param.grad is None else param.grad.detach().clone() for param in params]


def restore_gradients(params: list[torch.nn.Parameter], grads: list[Optional[torch.Tensor]]) -> None:
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach().clone()


def project_gradients_away(
    params: list[torch.nn.Parameter],
    total_grads: list[Optional[torch.Tensor]],
    preserve_grads: list[Optional[torch.Tensor]],
) -> float:
    dot = torch.zeros((), device=params[0].device)
    denom = torch.zeros((), device=params[0].device)
    for total_grad, preserve_grad in zip(total_grads, preserve_grads):
        if total_grad is None or preserve_grad is None:
            continue
        total = total_grad.to(device=params[0].device, dtype=torch.float32)
        preserve = preserve_grad.to(device=params[0].device, dtype=torch.float32)
        dot = dot + (total * preserve).sum()
        denom = denom + preserve.pow(2).sum()
    if denom.item() <= 1e-20:
        restore_gradients(params, total_grads)
        return 0.0

    coeff = torch.clamp(dot / denom.clamp_min(1e-20), min=0.0)
    for param, total_grad, preserve_grad in zip(params, total_grads, preserve_grads):
        if total_grad is None:
            param.grad = None
            continue
        if preserve_grad is None:
            param.grad = total_grad.detach().clone()
            continue
        projected = total_grad.float() - coeff * preserve_grad.float()
        param.grad = projected.to(device=param.device, dtype=param.dtype)
    return float(coeff.detach().item())


def style_vector(tensor: torch.Tensor) -> torch.Tensor:
    reduced = tensor.float()
    while reduced.ndim > 2:
        reduced = reduced.mean(dim=-1)
    return reduced.reshape(-1)


def build_style_basis(deltas: list[torch.Tensor], rank: int) -> Optional[torch.Tensor]:
    if not deltas:
        return None
    mat = torch.stack([style_vector(delta) for delta in deltas], dim=0)
    if mat.numel() == 0:
        return None
    mat = mat - mat.mean(dim=0, keepdim=True)
    if mat.shape[0] == 1:
        vec = mat[0]
        norm = vec.norm().clamp_min(1e-6)
        return (vec / norm).unsqueeze(1)
    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
    basis = vh[: min(rank, vh.shape[0])].T.contiguous()
    return basis


def projection_energy(residual: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None:
        return residual.float().pow(2).mean()
    flat = style_vector(residual)
    coeff = basis.T @ flat
    projected = basis @ coeff
    return projected.pow(2).mean()


def build_space_checkpoint_path(config: SPACEConfig) -> str:
    suffix = config.exp_name or sanitize_checkpoint_name(config.erase_concept)
    filename = f"space-{suffix}.safetensors"
    return os.path.join(config.save_path, filename)


def build_space_metadata(config: SPACEConfig, selected_modules: list[str]) -> Dict[str, str]:
    return {
        "format": "space-v3",
        "family": "sd",
        "component": "unet",
        "base_model_id": config.base_model_id,
        "method": "SPACE-v3",
        "erase_concept": config.erase_concept,
        "target_prompts_path": config.target_prompts_path,
        "target_artist_filter": config.target_artist_filter or "",
        "preserve_prompts_path": config.preserve_prompts_path or "",
        "neutral_art_template": config.neutral_art_template,
        "neutral_image_template": config.neutral_image_template,
        "alpha_art": str(config.alpha_art),
        "alpha_img": str(config.alpha_img),
        "erase_scale": str(config.erase_scale),
        "residual_mix": str(config.residual_mix),
        "style_gate_mode": str(config.style_gate_mode),
        "style_gate_min": str(config.style_gate_min),
        "style_gate_max": str(config.style_gate_max),
        "style_basis_rank": str(config.style_basis_rank),
        "trajectory_bands": str(config.trajectory_bands),
        "lora_target_layers": str(config.lora_target_layers),
        "use_prompt_weighting": str(config.use_prompt_weighting),
        "robust_prompt_mode": config.robust_prompt_mode,
        "vulnerable_preserve_k": str(config.vulnerable_preserve_k),
        "lambda_fid_preserve": str(config.lambda_fid_preserve),
        "preserve_null_interval": str(config.preserve_null_interval),
        "preserve_null_projection": str(config.preserve_null_projection),
        "lora_rank": str(config.lora_rank),
        "stage1_steps": str(config.stage1_steps),
        "stage2_steps": str(config.stage2_steps),
        "selected_modules": json.dumps(selected_modules),
    }


def save_space_checkpoint(adapters: dict[str, LoRALinear], filename: str, metadata: Dict[str, str]) -> None:
    state = {}
    for name, adapter in adapters.items():
        state[f"{name}.weight"] = adapter.merged_weight().cpu().contiguous()
        if adapter.merged_bias() is not None:
            state[f"{name}.bias"] = adapter.merged_bias().cpu().contiguous()
    save_esd_checkpoint(state, filename, metadata=metadata)


def record_space_provenance(config: SPACEConfig, checkpoint_path: str, selected_modules: list[str]) -> None:
    out_dir = os.path.join("results", "provenance", "space_v3")
    os.makedirs(out_dir, exist_ok=True)
    suffix = config.exp_name or sanitize_checkpoint_name(config.erase_concept)
    payload = {
        "method": "SPACE-v3",
        "artist": config.exp_name or config.erase_concept,
        "run_type": "research-method",
        "base_model_id": config.base_model_id,
        "checkpoint": checkpoint_path,
        "target_prompts_path": config.target_prompts_path,
        "target_artist_filter": config.target_artist_filter,
        "neutral_art_template": config.neutral_art_template,
        "neutral_image_template": config.neutral_image_template,
        "alpha_art": config.alpha_art,
        "alpha_img": config.alpha_img,
        "erase_scale": config.erase_scale,
        "residual_mix": config.residual_mix,
        "style_gate_mode": config.style_gate_mode,
        "style_gate_min": config.style_gate_min,
        "style_gate_max": config.style_gate_max,
        "style_basis_rank": config.style_basis_rank,
        "saliency_topk_blocks": config.saliency_topk_blocks,
        "trajectory_bands": config.trajectory_bands,
        "lora_target_layers": config.lora_target_layers,
        "use_prompt_weighting": config.use_prompt_weighting,
        "robust_prompt_mode": config.robust_prompt_mode,
        "vulnerable_preserve_k": config.vulnerable_preserve_k,
        "lambda_fid_preserve": config.lambda_fid_preserve,
        "preserve_null_interval": config.preserve_null_interval,
        "preserve_null_projection": config.preserve_null_projection,
        "lora_rank": config.lora_rank,
        "stage1_steps": config.stage1_steps,
        "stage2_steps": config.stage2_steps,
        "selected_modules": selected_modules,
    }
    with open(os.path.join(out_dir, f"train_space-{suffix}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def run_space_training(config: SPACEConfig) -> str:
    normalize_train_method("esd-x")
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    with _suppress_transformers_pipeline_load_noise():
        pipe = load_sd_pipeline(config)
    pipe.set_progress_bar_config(disable=True)

    if config.gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    resolution = config.resolution or resolve_default_resolution(pipe)
    null_embeds = encode_null_prompt(pipe, config)
    timestep_cond = get_timestep_cond(pipe, config)
    families = load_target_families(pipe, config)
    general_preserve_contexts = encode_preserve_prompts(pipe, load_preserve_prompts(config), config)

    selected_modules = ranked_saliency_modules(
        pipe=pipe,
        families=families,
        null_embeds=null_embeds,
        timestep_cond=timestep_cond,
        resolution=resolution,
        config=config,
    )
    adapters = inject_lora_adapters(pipe.unet, selected_modules, config.lora_rank)
    pipe.unet.to(config.device)
    trainable_params = trainable_lora_parameters(adapters)
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    initial_adapter_parameters = clone_adapter_parameters(adapters)
    encoded_families = encode_space_families(pipe, families, config)

    offload_modules_to_cpu(config.device, pipe.vae, pipe.safety_checker)
    trajectory = trajectory_indices(config)
    requested_steps = config.stage1_steps + config.stage2_steps
    total_steps = min(requested_steps, config.debug_steps) if config.debug_steps > 0 else requested_steps
    pbar = tqdm(range(total_steps), desc="Training SPACE v3 (sd)")
    last_total_loss = None
    last_target_shift = 0.0
    warned_static_adapter = False
    style_norm_ema: float = 0.0  # exponential moving average of per-band style direction norm (improvement J)

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        family = random.choice(encoded_families)
        variant = random.choice(family.variants)
        stage2 = step >= config.stage1_steps

        target_embeds = variant.target_embeds
        content_embeds = variant.content_embeds
        art_embeds = variant.art_embeds
        image_embeds = variant.image_embeds

        style_variant_embeds = []
        family_variants = family.variants if stage2 else family.variants[: min(2, len(family.variants))]
        for extra_variant in family_variants[: max(config.style_basis_rank, 2)]:
            style_variant_embeds.append((extra_variant.target_embeds, extra_variant.content_embeds))

        total_loss = torch.zeros((), device=config.device)
        erase_meter = []
        style_meter = []
        content_meter = []
        preserve_meter = []
        image_meter = []
        fid_preserve_meter = []

        for band_idx in trajectory:
            seed = random.randint(0, 2**15)
            set_adapters_enabled(adapters, False)
            xt = sample_xt(pipe, target_embeds, null_embeds, band_idx, seed, resolution, config)
            timestep = pipe.scheduler.timesteps[band_idx]

            with torch.no_grad():
                base_target = predict_noise(pipe.unet, xt, timestep, target_embeds, timestep_cond).detach()
                base_content = predict_noise(pipe.unet, xt, timestep, content_embeds, timestep_cond).detach()
                base_art = predict_noise(pipe.unet, xt, timestep, art_embeds, timestep_cond).detach()
                base_image = predict_noise(pipe.unet, xt, timestep, image_embeds, timestep_cond).detach()
                base_null = predict_noise(pipe.unet, xt, timestep, null_embeds, timestep_cond).detach()

            set_adapters_enabled(adapters, True)
            student_target = predict_noise(pipe.unet, xt, timestep, target_embeds, timestep_cond)
            student_content = predict_noise(pipe.unet, xt, timestep, content_embeds, timestep_cond)
            student_image = predict_noise(pipe.unet, xt, timestep, image_embeds, timestep_cond)

            teacher = make_space_teacher(base_target, base_content, base_art, base_image, base_null, config)
            erase_loss = F.mse_loss(student_target.float(), teacher.float())

            # Improvement J: weight erasure loss by style direction magnitude
            if config.use_prompt_weighting:
                residual_mix = min(max(float(config.residual_mix), 0.0), 1.0)
                style_direction = (
                    residual_mix * (base_target.float() - base_content.float())
                    + (1.0 - residual_mix) * (base_target.float() - base_null.float())
                )
                style_dir_norm = float(style_direction.norm().item())
                style_norm_ema = style_dir_norm if style_norm_ema == 0.0 else 0.98 * style_norm_ema + 0.02 * style_dir_norm
                style_weight = max(0.1, min(5.0, style_dir_norm / (style_norm_ema + 1e-8)))
                erase_loss = erase_loss * style_weight
            content_loss = F.mse_loss(student_content.float(), base_content.float())
            image_loss = F.mse_loss(student_image.float(), base_image.float())
            target_shift = F.mse_loss(student_target.float(), base_target.float())

            style_deltas = []
            set_adapters_enabled(adapters, False)
            with torch.no_grad():
                for variant_target_embeds, variant_content_embeds in style_variant_embeds:
                    base_t = predict_noise(pipe.unet, xt, timestep, variant_target_embeds, timestep_cond).detach()
                    base_c = predict_noise(pipe.unet, xt, timestep, variant_content_embeds, timestep_cond).detach()
                    style_deltas.append(base_t - base_c)
            set_adapters_enabled(adapters, True)
            basis = build_style_basis(style_deltas, config.style_basis_rank)
            style_loss = projection_energy(student_target - base_content, basis)

            preserve_loss = torch.zeros((), device=config.device)
            fid_preserve_loss = torch.zeros((), device=config.device)
            if general_preserve_contexts and (config.lambda_preserve > 0 or config.lambda_fid_preserve > 0):
                preserve_ctx = random.choice(general_preserve_contexts)
                preserve_embeds = preserve_ctx["embeds"]
                set_adapters_enabled(adapters, False)
                with torch.no_grad():
                    preserve_xt = sample_xt(pipe, preserve_embeds, null_embeds, band_idx, seed + 1, resolution, config)
                    base_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond).detach()
                set_adapters_enabled(adapters, True)
                student_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond)
                preserve_loss = F.mse_loss(student_preserve.float(), base_preserve.float())
                fid_preserve_loss = preserve_loss

            vulnerable_loss = torch.zeros((), device=config.device)
            if family.preserve_contexts and config.lambda_vulnerable > 0:
                preserve_ctx = random.choice(family.preserve_contexts)
                preserve_embeds = preserve_ctx["embeds"]
                set_adapters_enabled(adapters, False)
                with torch.no_grad():
                    preserve_xt = sample_xt(pipe, preserve_embeds, null_embeds, band_idx, seed + 2, resolution, config)
                    base_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond).detach()
                set_adapters_enabled(adapters, True)
                student_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond)
                vulnerable_loss = F.mse_loss(student_preserve.float(), base_preserve.float())

            total_loss = total_loss + (
                erase_loss
                + config.lambda_style * style_loss
                + config.lambda_content * content_loss
                + config.lambda_image * image_loss
                + config.lambda_preserve * preserve_loss
                + config.lambda_fid_preserve * fid_preserve_loss
                + config.lambda_vulnerable * vulnerable_loss
            )
            erase_meter.append(float(erase_loss.detach().item()))
            style_meter.append(float(style_loss.detach().item()))
            content_meter.append(float(content_loss.detach().item()))
            image_meter.append(float(image_loss.detach().item()))
            preserve_meter.append(float((preserve_loss + vulnerable_loss).detach().item()))
            fid_preserve_meter.append(float(fid_preserve_loss.detach().item()))
            last_target_shift = float(target_shift.detach().item())

        reg = adapter_regularization(adapters).to(config.device)
        total_loss = total_loss / max(len(trajectory), 1) + config.lambda_lora * reg
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"SPACE loss became non-finite at step {step}: {total_loss.item()}")
        total_loss.backward()
        projection_coeff = 0.0
        if (
            config.preserve_null_projection
            and config.preserve_null_interval > 0
            and general_preserve_contexts
            and step > 0
            and step % config.preserve_null_interval == 0
        ):
            total_grads = capture_gradients(trainable_params)
            optimizer.zero_grad(set_to_none=True)
            preserve_ctx = random.choice(general_preserve_contexts)
            preserve_embeds = preserve_ctx["embeds"]
            band_idx = random.choice(trajectory)
            seed = random.randint(0, 2**15)
            set_adapters_enabled(adapters, False)
            preserve_xt = sample_xt(pipe, preserve_embeds, null_embeds, band_idx, seed + 17, resolution, config)
            timestep = pipe.scheduler.timesteps[band_idx]
            with torch.no_grad():
                base_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond).detach()
            set_adapters_enabled(adapters, True)
            student_preserve = predict_noise(pipe.unet, preserve_xt, timestep, preserve_embeds, timestep_cond)
            projection_loss = F.mse_loss(student_preserve.float(), base_preserve.float())
            projection_loss.backward()
            preserve_grads = capture_gradients(trainable_params)
            projection_coeff = project_gradients_away(trainable_params, total_grads, preserve_grads)
        optimizer.step()
        last_total_loss = float(total_loss.detach().item())
        adapter_norm = float(adapter_parameter_norm(adapters).detach().item())
        delta_norm = float(adapter_delta_norm(adapters).detach().item())
        update_norm = float(adapter_update_norm(adapters, initial_adapter_parameters).detach().item())
        if step >= 9 and not warned_static_adapter and update_norm < 1e-9:
            warned_static_adapter = True
            print(
                "Warning: SPACE adapter update norm is still near zero after 10 steps; "
                "the training signal may be too weak."
            )
        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.2e}",
                "erase": f"{sum(erase_meter)/len(erase_meter):.2e}",
                "style": f"{sum(style_meter)/len(style_meter):.2e}",
                "content": f"{sum(content_meter)/len(content_meter):.2e}",
                "image": f"{sum(image_meter)/len(image_meter):.2e}",
                "preserve": f"{sum(preserve_meter)/len(preserve_meter):.2e}",
                "fidp": f"{sum(fid_preserve_meter)/len(fid_preserve_meter):.2e}",
                "adapter": f"{adapter_norm:.2e}",
                "delta": f"{delta_norm:.2e}",
                "update": f"{update_norm:.2e}",
                "teacher": f"{sum(erase_meter)/len(erase_meter):.2e}",
                "shift": f"{last_target_shift:.2e}",
                "proj": f"{projection_coeff:.1e}",
                "stage": "s2" if stage2 else "s1",
            }
        )

    final_update_norm = float(adapter_update_norm(adapters, initial_adapter_parameters).detach().item())
    if config.debug_steps > 0:
        if last_total_loss is None or not torch.isfinite(torch.tensor(last_total_loss)):
            raise RuntimeError("SPACE debug run did not produce a finite loss.")
        if final_update_norm < 1e-9:
            raise RuntimeError("SPACE debug run failed: LoRA parameters did not change.")
        if last_target_shift <= 0:
            raise RuntimeError("SPACE debug run failed: student target prediction did not move from base target.")

    checkpoint_path = build_space_checkpoint_path(config)
    metadata = build_space_metadata(config, selected_modules)
    save_space_checkpoint(adapters, checkpoint_path, metadata)
    if config.debug_steps > 0:
        checkpoint_tensors, _ = load_esd_checkpoint(checkpoint_path, device="cpu")
        if not checkpoint_tensors:
            raise RuntimeError("SPACE debug run failed: saved checkpoint did not reload any tensors.")
    record_space_provenance(config, checkpoint_path, selected_modules)
    return checkpoint_path
