import torch
import torch.nn as nn

from typing import Optional
from einops import rearrange

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm

from .attention import WanAttention, WanAttnProcessor


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        # note support attn_mode
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(attn_mode=attn_mode, attn_type="self-attention"),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(attn_mode=attn_mode, attn_type="cross-attention"),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        """Initialize weights for all submodules in the transformer block."""
        # Initialize attention modules
        self.attn1.init_weights()
        self.attn2.init_weights()
            
        # Initialize feed-forward network
        # Recursively initialize all linear layers in the FeedForward module
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
        # Initialize scale_shift_table
        nn.init.normal_(self.scale_shift_table, std=0.02)
        
        # Initialize normalization layers
        self.norm1.reset_parameters()
        if not isinstance(self.norm2, nn.Identity):
            self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        ulysses_enabled: bool = False,
    ) -> torch.Tensor:
        
        if (
            self.attn_mode == "flashattn_varlen"
            or self.attn_mode == "sageattn_varlen"
            or self.attn_mode == "sdpa"
        ):
            temb_scale_shift_table = self.scale_shift_table.to(dtype=temb.dtype, device=temb.device) + temb
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(temb_scale_shift_table, 'l n c -> n l c').chunk(6, dim=0)
        else:
            raise RuntimeError("Unkown attn mode!")
        
        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, ulysses_enabled)
        hidden_states = hidden_states + attn_output * gate_msa

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None, ulysses_enabled)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output * c_gate_msa

        return hidden_states
    

class WanVACETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
        # note support attn_mode
        attn_mode: str = "flashattn",
    ):
        super().__init__()

        # 1. Input projection
        self.proj_in = None
        if apply_input_projection:
            self.proj_in = nn.Linear(dim, dim)

        # 2. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=WanAttnProcessor(attn_mode=attn_mode, attn_type="self-attention"),
        )

        # 3. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanAttnProcessor(attn_mode=attn_mode, attn_type="cross-attention"),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. Output projection
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        """Initialize weights for all submodules in the transformer block."""
        # Initialize attention modules
        self.attn1.init_weights()
        self.attn2.init_weights()
            
        # Initialize feed-forward network
        # Recursively initialize all linear layers in the FeedForward module
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
        # Initialize scale_shift_table
        nn.init.normal_(self.scale_shift_table, std=0.02)
        
        # Initialize proj_in/proj_out layers
        if self.proj_in is not None:
            nn.init.normal_(self.proj_in.weight, std=0.02)
            if self.proj_in.bias is not None:
                nn.init.constant_(self.proj_in.bias, 0.0)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0.0)

        # Initialize normalization layers
        self.norm1.reset_parameters()
        if not isinstance(self.norm2, nn.Identity):
            self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        if (
            self.attn_mode == "flashattn_varlen"
            or self.attn_mode == "sageattn_varlen"
            or self.attn_mode == "sdpa"
        ):
            temb_scale_shift_table = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(temb_scale_shift_table, 'l n c -> n l c').chunk(6, dim=0)
        elif (
            self.attn_mode == "flashattn"
            or self.attn_mode == "sageattn"
        ):
            if temb.ndim == 4:
                # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                    self.scale_shift_table.unsqueeze(0) + temb.float()
                ).chunk(6, dim=2)
                # batch_size, seq_len, 1, inner_dim
                shift_msa = shift_msa.squeeze(2)
                scale_msa = scale_msa.squeeze(2)
                gate_msa = gate_msa.squeeze(2)
                c_shift_msa = c_shift_msa.squeeze(2)
                c_scale_msa = c_scale_msa.squeeze(2)
                c_gate_msa = c_gate_msa.squeeze(2)
            else:
                # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                    self.scale_shift_table + temb.float()
                ).chunk(6, dim=1)


        # 1. Self-attention
        norm_hidden_states = (self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            control_hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        control_hidden_states = (control_hidden_states.float() + attn_output * gate_msa).type_as(control_hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            control_hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        control_hidden_states = (control_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
            control_hidden_states
        )

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states
