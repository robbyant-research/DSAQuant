import logging
import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import Any, List, Dict, Optional, Union

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from vdm_infer.models.base.vdm_attn import build_attn_mask

from .args import WanTransformer3DModelArgs
from .dit_block import WanTransformerBlock
from .embedding import WanTimeTextImageEmbedding, WanRotaryPosEmbed


def set_requires_grad(models: Union[torch.nn.Module, List[torch.nn.Module]], value: bool) -> None:
    if isinstance(models, torch.nn.Module):
        models = [models]
    for model in models:
        if model is not None:
            model.requires_grad_(value)


logger = logging.getLogger("vdm_infer")


def pad_tensor(x: torch.Tensor, dim: int, padding_size: int):
    if padding_size <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = padding_size
    pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def unpad_tensor(x: torch.Tensor, dim: int, padding_size: int):
    if padding_size <= 0:
        return x
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]


def gather_outputs(x: torch.Tensor, gather_dim: int):
    return x


def slice_input_tensor_scale_grad(x: torch.Tensor, dim: int):
    return x


class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding_mlp", "patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    def __init__(
        self,
        model_args: WanTransformer3DModelArgs,
    ) -> None:
        super().__init__()
        self.model_args = model_args

        inner_dim = model_args.num_attention_heads * model_args.attention_head_dim
        out_channels = model_args.out_channels or model_args.in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(model_args.attention_head_dim, model_args.patch_size, model_args.rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(model_args.in_channels * model_args.patch_size[0] * model_args.patch_size[1] * model_args.patch_size[2], inner_dim)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=model_args.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=model_args.text_dim,
            image_embed_dim=model_args.image_dim,
            pos_embed_seq_len=model_args.pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, 
                    model_args.ffn_dim, 
                    model_args.num_attention_heads, 
                    model_args.qk_norm, 
                    model_args.cross_attn_norm, 
                    model_args.eps, 
                    model_args.added_kv_proj_dim,
                    attn_mode=model_args.attn_mode,
                )
                for _ in range(model_args.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, model_args.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(model_args.patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def init_weights(
        self, 
        buffer_device=None,
    ):
        """Initialize model weights following the original Wan implementation."""
        
        # Initialize patch embedding mlp layer
        nn.init.normal_(self.patch_embedding_mlp.weight, std=0.02)
        if self.patch_embedding_mlp.bias is not None:
            nn.init.constant_(self.patch_embedding_mlp.bias, 0)

        # Initialize condition embedder (time/text/image embeddings)
        self.condition_embedder.init_weights()
        
        # Initialize transformer blocks
        for block in self.blocks:
            block.init_weights()
        
        # Initialize output layers
        self.norm_out.reset_parameters()
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0)
        
        # Initialize scale_shift_table
        nn.init.normal_(self.scale_shift_table, std=0.02)
        
        # Initialize rope embeddings
        self.rope.init_weights()

    def load_pretrained_weight(self):
        # note this is only for debug
        import os
        from safetensors.torch import load_file
        def load_multi_safetensors(paths):
            if isinstance(paths, str):
                paths = sorted([
                    os.path.join(paths, f)
                    for f in os.listdir(paths)
                    if f.endswith(".safetensors")
                ])
            
            state_dict = {}
            for path in paths:
                part = load_file(path)
                overlapping_keys = set(state_dict.keys()).intersection(part.keys())
                if overlapping_keys:
                    print(f"[Warning] Overlapping keys in {path}: {overlapping_keys}")
                state_dict.update(part)
            
            return state_dict
        if self.model_args.pretrained_model_name_or_path is not None:
            state_dict = load_multi_safetensors(f"{self.model_args.pretrained_model_name_or_path}/transformer")
            m, n = self.load_state_dict(state_dict, strict=False)
            print(f"Missing key: {m}")
            print(f"Unexpected key: {n}")

    def forward_pre_process(
        self,
        timestep: torch.Tensor,
        hidden_states: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        latent_num_frames_lst: List[int] = None,
        latent_height_lst: List[int] = None,
        latent_width_lst: List[int] = None,
        text_emb_len_lst: List[int] = None,
        ulysses_enabled: bool = False,
    ):

        # note rope embedding
        rotary_emb = [
            self.rope(int(f)//self.model_args.patch_size[0], int(h)//self.model_args.patch_size[1], int(w)//self.model_args.patch_size[2], device=timestep.device)
            for f, h, w in zip(latent_num_frames_lst, latent_height_lst, latent_width_lst)
        ]
        rotary_emb = torch.cat(rotary_emb)[None, ...]

        # note reshape hidden states & patch embedding
        hidden_states = [
            rearrange(
                i, 
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.model_args.patch_size[0],
                c2=self.model_args.patch_size[1],
                c3=self.model_args.patch_size[2],
            ) for i in hidden_states
        ]
        hidden_states = torch.cat(hidden_states, dim=1) # [1, (L1+...+Ln), C]
        hidden_states = self.patch_embedding_mlp(hidden_states)

        # note q/kv seq_len for building mask
        q_seq_len_list = [
            int((f//self.model_args.patch_size[0]) * (h//self.model_args.patch_size[1]) * (w//self.model_args.patch_size[2]))
            for f, h, w in zip(latent_num_frames_lst, latent_height_lst, latent_width_lst)
        ]
        kv_seq_len_list = [int(i) for i in text_emb_len_lst]
        n_seq = len(q_seq_len_list)

        # note condition embedding
        # timestep shape: [L1,L2,...,Ln] (5B wan2.2 ti2v)
        # timestep shape: [1,1,...,1] (1.3B 14B wan2.1&wan2.2 t2v&i2v)
        (
            temb, 
            timestep_proj, 
            encoder_hidden_states, 
        ) = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        # note repeat temb/timestep_proj into [L1+L2+...+Ln] (1.3B 14B wan2.1&wan2.2 t2v&i2v)
        if temb.shape[0] != sum(q_seq_len_list):
            temb_cache = []
            for item, q_len in zip(temb, q_seq_len_list):
                temb_cache.append(item[None, ...].repeat(q_len, 1))
            temb = torch.cat(temb_cache)
        if timestep_proj.shape[0] != sum(q_seq_len_list):
            timestep_proj_cache = []
            for item, q_len in zip(timestep_proj, q_seq_len_list):
                timestep_proj_cache.append(item[None, ...].repeat(q_len, 1, 1))
            timestep_proj = torch.cat(timestep_proj_cache)

        # note init attn mask
        build_attn_mask(
            attn_mode=self.model_args.attn_mode,
            n_seq=n_seq,
            q_seq_len_list=q_seq_len_list,
            kv_seq_len_list=kv_seq_len_list,
            device=hidden_states.device,
        )

        # note cp pre-process
        padding_size = 0
        hidden_states_length = hidden_states.shape[1]
        if ulysses_enabled:
            padding_size = math.ceil(hidden_states_length / self.model_args.cp_size) * self.model_args.cp_size - hidden_states_length
            if padding_size > 0:
                hidden_states = pad_tensor(hidden_states, dim=1, padding_size=padding_size)
                encoder_hidden_states = pad_tensor(encoder_hidden_states, dim=1, padding_size=512)
                rotary_emb_cos = pad_tensor(rotary_emb_cos, dim=1, padding_size=padding_size)
                rotary_emb_sin = pad_tensor(rotary_emb_sin, dim=1, padding_size=padding_size)
                rotary_emb = (rotary_emb_cos, rotary_emb_sin)
                temb = pad_tensor(temb, dim=0, padding_size=padding_size)
                timestep_proj = pad_tensor(timestep_proj, dim=0, padding_size=padding_size)
                q_seq_len_list.append(padding_size)
                kv_seq_len_list.append(512)
                n_seq += 1
                # re-init attention mask
                build_attn_mask(
                    attn_mode=self.model_args.attn_mode,
                    n_seq=n_seq,
                    q_seq_len_list=q_seq_len_list,
                    kv_seq_len_list=kv_seq_len_list,
                    device=hidden_states.device,
                )
            hidden_states = slice_input_tensor_scale_grad(hidden_states, dim=1)
            encoder_hidden_states = slice_input_tensor_scale_grad(encoder_hidden_states, dim=1)
            temb = slice_input_tensor_scale_grad(temb, dim=0)
            timestep_proj = slice_input_tensor_scale_grad(timestep_proj, dim=0)

        return hidden_states, encoder_hidden_states, temb, timestep_proj, rotary_emb, padding_size

    def forward_post_process(
        self,
        temb: torch.Tensor,
        hidden_states: torch.Tensor,
        ulysses_enabled: bool = False,
        padding_size: int = 0,
    ):
        temb_scale_shift_table = self.scale_shift_table + temb[:, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, 'l n c -> n l c').chunk(2, dim=0)

        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # note cp post-process
        if ulysses_enabled:
            hidden_states = gather_outputs(hidden_states, gather_dim=1)
            if padding_size > 0:
                hidden_states = hidden_states[:, :-padding_size, :]
        return hidden_states
    
    def forward(
        self,
        input_dict: Optional[Dict[str, Any]],
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
        timestep = input_dict["timestep"]
        hidden_states = input_dict["hidden_states"]
        encoder_hidden_states = input_dict.get("encoder_hidden_states", None)
        ulysses_enabled = True if self.model_args.cp_size > 1 else False

        # 1. Input patchify / embedding
        if (
            self.model_args.attn_mode == "flashattn_varlen"
            or self.model_args.attn_mode == "sageattn_varlen"
            or self.model_args.attn_mode == "sdpa"
        ):
            # note prepare input data (sequnce mode)
            (
                hidden_states, 
                encoder_hidden_states, 
                temb, 
                timestep_proj, 
                rotary_emb,
                padding_size, 
            ) = self.forward_pre_process(
                timestep=timestep,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                latent_num_frames_lst=input_dict["latent_num_frames_lst"],
                latent_height_lst=input_dict["latent_height_lst"],
                latent_width_lst=input_dict["latent_width_lst"],
                text_emb_len_lst=input_dict["text_emb_len_lst"],
                ulysses_enabled=ulysses_enabled,
            )
        else:
            raise ValueError("Unknown attn mode")

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, ulysses_enabled
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, ulysses_enabled)

        # 3. Output norm, projection & unpatchify
        if (
            self.model_args.attn_mode == "flashattn_varlen"
            or self.model_args.attn_mode == "sageattn_varlen"
            or self.model_args.attn_mode == "sdpa"
        ):
            # note prepare output hidden states
            hidden_states = self.forward_post_process(temb, hidden_states, ulysses_enabled=ulysses_enabled, padding_size=padding_size)
            # note hidden_states is pathified by 4 (2*2)
            # note '1 l (n c) -> 1 (n l) c' is wrong! pls be careful about the order
            output = rearrange(hidden_states, '1 l (n c) -> 1 (l n) c', n=4)
        else:
            raise ValueError("Unknown attn mode")

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
