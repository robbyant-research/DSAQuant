from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn as nn


@dataclass
class WanTransformer3DModelArgs:
    pretrained_model_name_or_path: str = ""
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: Optional[int] = None
    added_kv_proj_dim: Optional[int] = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: Optional[int] = None
    max_seq_len: int = 2048
    attn_mode: str = "sdpa"
    in_channels_org: int = 0
    cp_size: int = 1

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, float]:
        nparams = sum(p.numel() for p in model.parameters())
        d_model = self.num_attention_heads * self.attention_head_dim
        self_attn_params = 4 * d_model * d_model
        cross_attn_params = 4 * d_model * d_model
        ffn_params = 2 * d_model * self.ffn_dim
        gelu_norm_params = 4 * self.ffn_dim
        total_per_layer = self_attn_params + cross_attn_params + ffn_params + gelu_norm_params
        return nparams, float(6 * self.num_layers * total_per_layer)
