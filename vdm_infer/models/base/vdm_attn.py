import torch
import torch.nn.functional as torch_F

from einops import rearrange
from typing_extensions import List

try:
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    _flash_attn_import_error = None
except Exception as exc:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _flash_attn_import_error = exc

from sageattention import sageattn, sageattn_varlen


def _require_optional_dependency(value, package_name: str, attn_mode: str, import_error=None):
    if value is None:
        message = (
            f"{package_name} is required for attn_mode={attn_mode}. "
            "Install it or choose another ATTN_MODE."
        )
        if import_error is not None:
            message += f" Original import error: {import_error}"
        raise ModuleNotFoundError(message) from import_error
    return value


class BaseAttnVarlen(torch.nn.Module):
    def __init__(
        self, 
        attn_type: str = "self-attention",
    ) -> None:
        super().__init__()
        if attn_type not in ["self-attention", "cross-attention"]:
            raise ValueError(f"Unrecognized attn_type {attn_type}.")
        self.attn_type = attn_type

    def convert_type(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = query[0]
        k_varlen = key[0]
        v_varlen = value[0]

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)
        return q_varlen, k_varlen, v_varlen

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        dtype=torch.bfloat16,
    ):
        q_varlen, k_varlen, v_varlen = self.convert_type(query, key, value, dtype)
        if self.attn_type == "self-attention":
            x_out = self.self_attn(q_varlen, k_varlen, v_varlen)
        elif self.attn_type == "cross-attention":
            x_out = self.cross_attn(q_varlen, k_varlen, v_varlen)
        else:
            raise ValueError(f"Unrecognized attn_type {self.attn_type}.")
        return x_out

    def self_attn(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        raise NotImplementedError("Not implement self.self_attn")

    def cross_attn(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        raise NotImplementedError("Not implement self.cross_attn")

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        self,
    ) -> None:
        raise NotImplementedError("Not implement self.init_attn_params")


class FlashAttnVarlen(BaseAttnVarlen):
    is_causal_selfattn: bool = False
    is_causal_crossattn: bool = False
    # flash attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None

    def self_attn(
        self,   
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        x_out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=FlashAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=FlashAttnVarlen.cu_seqlens_q,
            max_seqlen_q=FlashAttnVarlen.max_seqlen_q,
            max_seqlen_k=FlashAttnVarlen.max_seqlen_q,
            causal=FlashAttnVarlen.is_causal_selfattn,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        x_out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=FlashAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=FlashAttnVarlen.cu_seqlens_kv,
            max_seqlen_q=FlashAttnVarlen.max_seqlen_q,
            max_seqlen_k=FlashAttnVarlen.max_seqlen_kv,
            causal=FlashAttnVarlen.is_causal_crossattn,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        is_causal_selfattn: bool = False,
        is_causal_crossattn: bool = False,
        torch_device: str = "cuda",
    ) -> None:
        seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
        FlashAttnVarlen.max_seqlen_q = max(seqlen_q)
        FlashAttnVarlen.cu_seqlens_q = cu_seqlens_q

        seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
        FlashAttnVarlen.max_seqlen_kv = max(seqlen_kv)
        FlashAttnVarlen.cu_seqlens_kv = cu_seqlens_kv

        FlashAttnVarlen.is_causal_selfattn = is_causal_selfattn
        FlashAttnVarlen.is_causal_crossattn = is_causal_crossattn


class SageAttnVarlen(BaseAttnVarlen):
    is_causal_selfattn: bool = False
    is_causal_crossattn: bool = False
    # sage attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None

    def self_attn(
        self,   
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        x_out = sageattn_varlen(
            query, key, value,
            cu_seqlens_q=SageAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=SageAttnVarlen.cu_seqlens_q,
            max_seqlen_q=SageAttnVarlen.max_seqlen_q,
            max_seqlen_k=SageAttnVarlen.max_seqlen_q,
            causal=SageAttnVarlen.is_causal_selfattn,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
    ):
        x_out = sageattn_varlen(
            query, key, value,
            cu_seqlens_q=SageAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=SageAttnVarlen.cu_seqlens_kv,
            max_seqlen_q=SageAttnVarlen.max_seqlen_q,
            max_seqlen_k=SageAttnVarlen.max_seqlen_kv,
            causal=SageAttnVarlen.is_causal_crossattn,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        is_causal_selfattn: bool = False,
        is_causal_crossattn: bool = False,
        torch_device: str = "cuda",
    ) -> None:
        seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
        SageAttnVarlen.max_seqlen_q = max(seqlen_q)
        SageAttnVarlen.cu_seqlens_q = cu_seqlens_q

        seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
        SageAttnVarlen.max_seqlen_kv = max(seqlen_kv)
        SageAttnVarlen.cu_seqlens_kv = cu_seqlens_kv

        SageAttnVarlen.is_causal_selfattn = is_causal_selfattn
        SageAttnVarlen.is_causal_crossattn = is_causal_crossattn
        

class SDPAAttn(torch.nn.Module):
    """Run mask-free SDPA on Wan's [batch, seq, heads, dim] tensors."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # PyTorch SDPA expects [batch, heads, seq, head_dim].  No mask or
        # causal flag is needed for Wan's bidirectional inference attention.
        output = torch_F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return output.transpose(1, 2).contiguous()


def build_attn_op(
    attn_mode: str = "sdpa",
    attn_type: str = "self-attention",
):
    if attn_mode == "flashattn_varlen":
        _require_optional_dependency(flash_attn_varlen_func, "flash_attn_interface", attn_mode, _flash_attn_import_error)
        return FlashAttnVarlen(attn_type)
    elif attn_mode == "sageattn_varlen":
        return SageAttnVarlen(attn_type)
    elif attn_mode == "sdpa":
        return SDPAAttn()
    elif attn_mode == "flash_attn":
        return _require_optional_dependency(flash_attn_func, "flash_attn_interface", attn_mode, _flash_attn_import_error)
    elif attn_mode == "sage_attn":
        return sageattn
    else:
        raise ValueError("Unknown attn mode")


def build_attn_mask(
    attn_mode: str = "sdpa",
    n_seq: int = None,
    q_seq_len_list: List = None,
    kv_seq_len_list: List = None,
    is_causal_selfattn: bool = False,
    is_causal_crossattn: bool = False,
    device: str = "cuda",
):
    if attn_mode == "flashattn_varlen":
        _require_optional_dependency(flash_attn_varlen_func, "flash_attn_interface", attn_mode, _flash_attn_import_error)
        FlashAttnVarlen.init_attn_params(n_seq, q_seq_len_list, kv_seq_len_list, is_causal_selfattn, is_causal_crossattn, device)
    elif attn_mode == "sageattn_varlen":
        SageAttnVarlen.init_attn_params(n_seq, q_seq_len_list, kv_seq_len_list, is_causal_selfattn, is_causal_crossattn, device)
    elif attn_mode == "sdpa":
        # Dense SDPA receives regular tensors directly; no varlen metadata or
        # attention mask is needed for single-sample inference.
        return
    else:
        raise ValueError("Unknown attn mode")


if __name__ == '__main__':
    batch_size = 4
    q_seq_len = 4096
    kv_seq_len = 512
    num_head = 12
    head_dim = 128

    torch_device = 'cuda'
    torch_dtype = torch.bfloat16

    q = torch.randn(batch_size, num_head, q_seq_len, head_dim, dtype=torch_dtype, device=torch_device)
    k = torch.randn(batch_size, num_head, kv_seq_len, head_dim, dtype=torch_dtype, device=torch_device)
    v = torch.randn(batch_size, num_head, kv_seq_len, head_dim, dtype=torch_dtype, device=torch_device)

    # * flash attn batch
    flash_attn = build_attn_op(attn_mode="flash_attn")
    q_batch = rearrange(q, "b n s d -> b s n d")
    k_batch = rearrange(k, "b n s d -> b s n d")
    v_batch = rearrange(v, "b n s d -> b s n d")
    x_selfattn_batch = flash_attn(q_batch, q_batch, q_batch)
    x_crossattn_batch = flash_attn(q_batch, k_batch, v_batch)
    
    # * flash attn varlen
    attn_mode = "flashattn_varlen"
    q_seq_len_list = [q_seq_len] * batch_size
    kv_seq_len_list = [kv_seq_len] * batch_size
    build_attn_mask(
        attn_mode=attn_mode,
        n_seq=batch_size,
        q_seq_len_list=q_seq_len_list,
        kv_seq_len_list=kv_seq_len_list,
    )
    selfattn_varlen = build_attn_op(attn_mode=attn_mode, attn_type="self-attention")
    crossattn_varlen = build_attn_op(attn_mode=attn_mode, attn_type="cross-attention")
    
    q_varlen = rearrange(q, "b n s d -> (b s) n d")[None, ...]
    k_varlen = rearrange(k, "b n s d -> (b s) n d")[None, ...]
    v_varlen = rearrange(v, "b n s d -> (b s) n d")[None, ...]
    x_varlen_selfattn = selfattn_varlen(q_varlen, q_varlen, q_varlen)
    x_varlen_crossattn = crossattn_varlen(q_varlen, k_varlen, v_varlen)

    x_varlen_selfattn_gt = rearrange(x_selfattn_batch, "b s n d -> (b s) n d", b=batch_size, n=num_head)
    x_varlen_crossattn_gt = rearrange(x_crossattn_batch, "b s n d -> (b s) n d", b=batch_size, n=num_head)
    
    # * compute loss
    mean_diff_selfattn = (x_varlen_selfattn_gt - x_varlen_selfattn).abs().mean()
    mean_diff_crossattn = (x_varlen_crossattn_gt - x_varlen_crossattn).abs().mean()
    print(f"mean Abs Diff (flash self attention): {mean_diff_selfattn:.6f}")
    print(f"mean Abs Diff (flash cross attention): {mean_diff_crossattn:.6f}")
