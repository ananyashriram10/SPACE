from typing import Any, Callable, Dict, List, Optional, Union

import torch

# ── diffusers version-compat shims ───────────────────────────────────────────
# sd_utils.py was written against diffusers ≥ 0.28. Every import below is
# wrapped so the module loads on any diffusers version the pod ships with.

# Callbacks (added ~0.25)
try:
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
except ImportError:
    class PipelineCallback:        # type: ignore[no-redef]
        tensor_inputs: list = []
    class MultiPipelineCallbacks:  # type: ignore[no-redef]
        tensor_inputs: list = []

# PipelineImageInput type alias (added ~0.26) — only used in annotations
try:
    from diffusers.image_processor import PipelineImageInput
except ImportError:
    PipelineImageInput = None

# deprecate helper (present in all recent versions, but guard anyway)
try:
    from diffusers.utils import deprecate
except ImportError:
    def deprecate(*args, **kwargs): pass  # type: ignore[misc]

# XLA support detection
try:
    from diffusers.utils import is_torch_xla_available
    _XLA_AVAILABLE = is_torch_xla_available()
except ImportError:
    _XLA_AVAILABLE = False

# Pipeline output dataclass
try:
    from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
except ImportError:
    from dataclasses import dataclass, field
    from typing import List as _List
    @dataclass
    class StableDiffusionPipelineOutput:  # type: ignore[no-redef]
        images: object
        nsfw_content_detected: object = None

# rescale_noise_cfg and retrieve_timesteps (added ~0.26)
try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        rescale_noise_cfg,
        retrieve_timesteps,
    )
except ImportError:
    def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # type: ignore[misc]
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg  = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        rescaled = noise_cfg * (std_text / std_cfg)
        return guidance_rescale * rescaled + (1 - guidance_rescale) * noise_cfg

    def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None):  # type: ignore[misc]
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler.timesteps, num_inference_steps

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

XLA_AVAILABLE = _XLA_AVAILABLE


@torch.no_grad()
def esd_sd_call(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    run_from_timestep=0,
    run_till_timestep=None,
    start_latents=None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.
    """

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    if not height or not width:
        _sample_size = self.unet.config.sample_size
        _sz_is_int = isinstance(_sample_size, int)
        height = _sample_size if _sz_is_int else _sample_size[0]
        width  = _sample_size if _sz_is_int else _sample_size[1]
        height, width = height * self.vae_scale_factor, width * self.vae_scale_factor

    # 1. Check inputs — signature varies by diffusers version; fall back gracefully
    import inspect as _inspect
    _ci_params = set(_inspect.signature(self.check_inputs).parameters)
    _ci_kwargs = dict(
        prompt=prompt, height=height, width=width,
        callback_steps=callback_steps if callback_steps is not None else 1,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    if "ip_adapter_image" in _ci_params:
        _ci_kwargs["ip_adapter_image"] = ip_adapter_image
    if "ip_adapter_image_embeds" in _ci_params:
        _ci_kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds
    if "callback_on_step_end_tensor_inputs" in _ci_params:
        _ci_kwargs["callback_on_step_end_tensor_inputs"] = callback_on_step_end_tensor_inputs
    self.check_inputs(**_ci_kwargs)

    # Use local variables for pipeline state — avoids relying on private
    # properties (_guidance_scale etc) that only exist in diffusers ≥ 0.26
    _do_cfg = guidance_scale > 1.0

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self.unet.device

    # 3. Encode input prompt
    lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )

    import inspect as _inspect
    _ep_sig = set(_inspect.signature(self.encode_prompt).parameters)
    _ep_kw = dict(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=_do_cfg,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    if "lora_scale" in _ep_sig:
        _ep_kw["lora_scale"] = lora_scale
    if "clip_skip" in _ep_sig:
        _ep_kw["clip_skip"] = clip_skip
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(**_ep_kw)

    if _do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            _do_cfg,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    timesteps = timesteps[run_from_timestep: run_till_timestep]
    if start_latents is not None:
        latents = start_latents
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    try:
        self._num_timesteps = len(timesteps)
    except Exception:
        pass
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if _do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if _do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if _do_cfg and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    if hasattr(self, "maybe_free_model_hooks"):
        self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


