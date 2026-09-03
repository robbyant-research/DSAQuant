import torch
import imageio
import numpy as np

from typing import List, Union

from diffusers import (
    AutoencoderKLWan, 
    UniPCMultistepScheduler,
)
from diffusers.video_processor import VideoProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)


def load_vae(
    vae_path: str,
    vae_type: str = "vdm_wan",
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    if "vdm_wan" in vae_type:
        vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            torch_dtype=torch_dtype,
            device_map="cpu"
        )
    else:
        raise ValueError("Unknown vae type")
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path: str,
    text_encoder_type: str = "u5",
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    if text_encoder_type == "u5":
        text_encoder = UMT5EncoderModel.from_pretrained(
            text_encoder_path,
            dtype=torch_dtype,
            device_map="cpu"
        )
    else:
        raise ValueError("Unknown text encoder type")
    return text_encoder.to(torch_device)


def load_scheduler(
    flow_shift: float = 5.0,
    scheduler_type: str = "multi_step",
):
    if scheduler_type == "multi_step":
        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=flow_shift,
        )
    else:
        raise ValueError("Unknown scheduler type")
    return scheduler


def load_tokenizer(
    tokenizer_path: str,
):
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
    )
    return tokenizer


def data_seq_to_patch(
    patch_size: list,
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch

@torch.no_grad()
def prepare_latents(
    vae,
    text_encoder,
    videos: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor = None,
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    # note 1. prepare video latents
    latents = vae._encode(videos.to(device=torch_device, dtype=torch_dtype))
    mu, logvar = torch.chunk(latents, 2, dim=1)
    
    latents_mean = torch.tensor(vae.config.latents_mean).to(mu.device)
    latents_std = torch.tensor(vae.config.latents_std).to(mu.device)
    mu = normalize_latents(mu, latents_mean, 1.0 / latents_std)
    logvar = normalize_latents(logvar, latents_mean, 1.0 / latents_std)
    latents = torch.cat([mu, logvar], dim=1)

    posterior = DiagonalGaussianDistribution(latents)
    latents = posterior.sample()

    # note 2. prepare text latents
    input_ids = token_ids.squeeze(1)
    attention_mask = token_mask.squeeze(1) if token_mask is not None else None
    text_embs = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    return latents, text_embs


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def flow_match_xt(x0: torch.Tensor, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Forward process of flow matching."""
    return (1.0 - t) * x0 + t * n


def flow_match_target(n: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    r"""Loss target for flow matching."""
    return n - x0


def expand_tensor_dims(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    assert len(tensor.shape) <= ndim
    return tensor.reshape(tensor.shape + (1,) * (ndim - len(tensor.shape)))


def set_requires_grad(models: Union[torch.nn.Module, List[torch.nn.Module]], value: bool) -> None:
    if isinstance(models, torch.nn.Module):
        models = [models]
    for model in models:
        if model is not None:
            model.requires_grad_(value)


def save_video(
    video_frames: np.ndarray,
    output_path: str,
):
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    imageio.mimwrite(output_path, video_frames, fps=15)


@torch.no_grad()
def decode_latents_to_video(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
):
    if latents.ndim == 3:
        latents = latents[None, ...]
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    video = video_processor.postprocess_video(video, output_type="np")
    return video


def sample_timestep_id(
    batch_size: int = 1,
    sample_scheme: str = "logit_normal",
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    do_fix_timestep: bool = False,
    fix_timestep_weight: float = 0.2,
    num_train_timesteps: int = 1000,
):
    if do_fix_timestep:
        u = torch.ones([batch_size], dtype=torch.int64) * fix_timestep_weight
    else:
        if sample_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=[batch_size])
            u = torch.nn.functional.sigmoid(u)
        elif sample_scheme == "uniform":
            u = torch.rand(size=[batch_size])
        else:
            raise ValueError("Unknown timestep sample scheme")
        u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd

    timestep_id = (u * num_train_timesteps).to(torch.int64)
    return timestep_id
