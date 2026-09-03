import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.models.attention import AttentionModuleMixin
from vdm_infer.models.base.vdm_attn import build_attn_op


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


_USE_FAST_ROPE = _env_bool("USE_FAST_ROPE", False)
_USE_NATIVE_RMSNORM = _env_bool("USE_NATIVE_RMSNORM", False)

def gather_seq_scatter_heads(x: torch.Tensor, seq_dim: int, head_dim: int):
    return x


def gather_heads_scatter_seq(x: torch.Tensor, seq_dim: int, head_dim: int):
    return x


def gather_seq_scatter_heads_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int, head_dim: int):
    q = gather_seq_scatter_heads(q, seq_dim, head_dim)
    k = gather_seq_scatter_heads(k, seq_dim, head_dim)
    v = gather_seq_scatter_heads(v, seq_dim, head_dim)
    return q, k, v


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    # VDM fused QKV kernel path.
    if getattr(attn, 'use_fused_qkv_kernel', False) and getattr(attn, 'qkv_kernel_backend', None) == 'vdm':
        from vdm_infer.kernels.vdm_w4a4 import linear_vdm_w4a4, linear_vdm_w4a4_qkv
        if attn.cross_attention_dim_head is None:
            if not hasattr(attn, '_vdm_qkv_weight_packed') or attn._vdm_qkv_weight_packed is None:
                attn._vdm_qkv_weight_packed = torch.cat([
                    attn.to_q.vdm_weight_packed,
                    attn.to_k.vdm_weight_packed,
                    attn.to_v.vdm_weight_packed,
                ], dim=0).contiguous()
                attn._vdm_qkv_weight_scale = torch.cat([
                    attn.to_q.vdm_weight_scale_packed,
                    attn.to_k.vdm_weight_scale_packed,
                    attn.to_v.vdm_weight_scale_packed,
                ], dim=0).contiguous()
            if not hasattr(attn, '_vdm_qkv_bias_half') or attn._vdm_qkv_bias_half is None:
                N = attn.to_q.vdm_weight_packed.shape[0]
                has_bias = (attn.to_q._bias_half is not None or attn.to_k._bias_half is not None or attn.to_v._bias_half is not None)
                if has_bias:
                    device = attn.to_q.vdm_weight_packed.device
                    q_bias = attn.to_q._bias_half if attn.to_q._bias_half is not None else torch.zeros(N, dtype=torch.float16, device=device)
                    k_bias = attn.to_k._bias_half if attn.to_k._bias_half is not None else torch.zeros(N, dtype=torch.float16, device=device)
                    v_bias = attn.to_v._bias_half if attn.to_v._bias_half is not None else torch.zeros(N, dtype=torch.float16, device=device)
                    attn._vdm_qkv_bias_half = torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous()
                else:
                    attn._vdm_qkv_bias_half = None
            query, key, value = linear_vdm_w4a4_qkv(
                hidden_states,
                attn.to_q.vdm_weight_packed, attn.to_q.vdm_weight_scale_packed,
                attn.to_k.vdm_weight_packed, attn.to_k.vdm_weight_scale_packed,
                attn.to_v.vdm_weight_packed, attn.to_v.vdm_weight_scale_packed,
                clip_ratio=attn.to_q.kernel_clip_ratio,
                w_qkv_packed=attn._vdm_qkv_weight_packed,
                w_qkv_scale=attn._vdm_qkv_weight_scale,
                qkv_bias_half=attn._vdm_qkv_bias_half,
            )
        else:
            query = attn.to_q(hidden_states)
            if not hasattr(attn, '_vdm_kv_weight_packed') or attn._vdm_kv_weight_packed is None:
                attn._vdm_kv_weight_packed = torch.cat([
                    attn.to_k.vdm_weight_packed,
                    attn.to_v.vdm_weight_packed,
                ], dim=0).contiguous()
                attn._vdm_kv_weight_scale = torch.cat([
                    attn.to_k.vdm_weight_scale_packed,
                    attn.to_v.vdm_weight_scale_packed,
                ], dim=0).contiguous()
            if not hasattr(attn, '_vdm_kv_bias_half') or attn._vdm_kv_bias_half is None:
                N = attn.to_k.vdm_weight_packed.shape[0]
                has_bias = (attn.to_k._bias_half is not None or attn.to_v._bias_half is not None)
                if has_bias:
                    device = attn.to_k.vdm_weight_packed.device
                    k_bias = attn.to_k._bias_half if attn.to_k._bias_half is not None else torch.zeros(N, dtype=torch.float16, device=device)
                    v_bias = attn.to_v._bias_half if attn.to_v._bias_half is not None else torch.zeros(N, dtype=torch.float16, device=device)
                    attn._vdm_kv_bias_half = torch.cat([k_bias, v_bias], dim=0).contiguous()
                else:
                    attn._vdm_kv_bias_half = None
            kv = linear_vdm_w4a4(
                encoder_hidden_states,
                attn._vdm_kv_weight_packed,
                attn._vdm_kv_weight_scale,
                bias=None,
                clip_ratio=attn.to_k.kernel_clip_ratio,
                bias_half=attn._vdm_kv_bias_half,
            )
            N = attn.to_k.vdm_weight_packed.shape[0]
            key, value = kv.split(N, dim=-1)
        return query, key, value

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def apply_rotary_emb(x, freqs):
    rope_dtype = torch.float32 if _USE_FAST_ROPE else torch.float64
    x_out = torch.view_as_complex(x.to(rope_dtype).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)


class WanAttnProcessor:
    def __init__(
        self, 
        attn_mode="sdpa",
        attn_type="self-attention",
    ):
        self.attn_type = attn_type
        self.attn_op = build_attn_op(attn_mode=attn_mode, attn_type=attn_type)

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ulysses_enabled: bool = False
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if ulysses_enabled:
            query, key, value = gather_seq_scatter_heads_qkv(query, key, value, seq_dim=1, head_dim=2)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self.attn_op(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self.attn_op(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        if ulysses_enabled:
            hidden_states = gather_heads_scatter_seq(hidden_states, seq_dim=1, head_dim=2)
            
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class RMSNorm(nn.RMSNorm):
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        if _USE_NATIVE_RMSNORM and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and self.weight is not None:
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None
        self.use_fused_qkv_kernel = False
        self.qkv_kernel_backend = None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def init_weights(self):
        """Initialize weights for the attention module."""
        # Initialize projection layers
        nn.init.normal_(self.to_q.weight, std=0.02)
        if self.to_q.bias is not None:
            nn.init.constant_(self.to_q.bias, 0.0)
            
        nn.init.normal_(self.to_k.weight, std=0.02)
        if self.to_k.bias is not None:
            nn.init.constant_(self.to_k.bias, 0.0)
            
        nn.init.normal_(self.to_v.weight, std=0.02)
        if self.to_v.bias is not None:
            nn.init.constant_(self.to_v.bias, 0.0)
            
        # Initialize output projection
        nn.init.normal_(self.to_out[0].weight, std=0.02)
        if self.to_out[0].bias is not None:
            nn.init.constant_(self.to_out[0].bias, 0.0)
            
        # Initialize added projection layers if they exist
        if self.add_k_proj is not None:
            nn.init.normal_(self.add_k_proj.weight, std=0.02)
            if self.add_k_proj.bias is not None:
                nn.init.constant_(self.add_k_proj.bias, 0.0)
                
        if self.add_v_proj is not None:
            nn.init.normal_(self.add_v_proj.weight, std=0.02)
            if self.add_v_proj.bias is not None:
                nn.init.constant_(self.add_v_proj.bias, 0.0)
                
        # Initialize fused projections if they exist
        if hasattr(self, 'to_qkv'):
            nn.init.normal_(self.to_qkv.weight, std=0.02)
            if self.to_qkv.bias is not None:
                nn.init.constant_(self.to_qkv.bias, 0.0)
                
        if hasattr(self, 'to_kv'):
            nn.init.normal_(self.to_kv.weight, std=0.02)
            if self.to_kv.bias is not None:
                nn.init.constant_(self.to_kv.bias, 0.0)
                
        if hasattr(self, 'to_added_kv'):
            nn.init.normal_(self.to_added_kv.weight, std=0.02)
            if self.to_added_kv.bias is not None:
                nn.init.constant_(self.to_added_kv.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ulysses_enabled: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, ulysses_enabled, **kwargs)
