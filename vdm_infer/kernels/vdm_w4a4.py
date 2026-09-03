"""Standalone DSAQuant W4A4 kernel wrapper.

This module builds the DSAQuant W4A4 CUDA extension used by inference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "csrc" / "vdm_w4a4_kernel.cu"
_BUILD_ROOT = Path(
    os.environ.get("VDM_W4A4_EXTENSIONS_DIR", str(_ROOT / ".torch_extensions_vdm_w4a4"))
)
_EXT = None
_LOAD_ERROR: Optional[BaseException] = None
_BUILD_CONFIG = {}

BLOCK_M = 256
BLOCK_N = 128
WARP_K = 64


def _configure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    cuda_version = torch.version.cuda
    if cuda_version:
        candidate = Path(f"/usr/local/cuda-{cuda_version}")
        if candidate.exists():
            os.environ["CUDA_HOME"] = str(candidate)
            return
    fallback = Path("/usr/local/cuda")
    if fallback.exists():
        os.environ["CUDA_HOME"] = str(fallback)


def _detect_arch_list() -> str:
    explicit = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if explicit:
        return explicit

    target = os.environ.get("VDM_W4A4_TARGET_ARCH", "auto").strip().lower()
    aliases = {
        "a6000": "8.6",
        "rtx_a6000": "8.6",
        "sm86": "8.6",
        "sm_86": "8.6",
        "8.6": "8.6",
        "4090": "8.9",
        "rtx4090": "8.9",
        "rtx_4090": "8.9",
        "ada": "8.9",
        "sm89": "8.9",
        "sm_89": "8.9",
        "8.9": "8.9",
    }
    if target in aliases:
        return aliases[target]

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"

    return "8.6"


def _arch_tag(arch_list: str) -> str:
    tokens = arch_list.replace(";", " ").replace(",", " ").split()
    tags = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        has_ptx = "+ptx" in token.lower()
        token = token.replace("+PTX", "").replace("+ptx", "")
        if token.startswith("sm_"):
            tag = "sm" + token.split("_", 1)[1]
        elif token.startswith("sm") and token[2:].isdigit():
            tag = token
        elif "." in token:
            major, minor = token.split(".", 1)
            tag = f"sm{major}{minor}"
        else:
            tag = token.lower().replace("-", "_")
        if has_ptx:
            tag += "ptx"
        tags.append(tag)
    return "_".join(tags) if tags else "sm86"


def _remove_unwanted_pytorch_nvcc_flags() -> None:
    # These macros make half/half2 CUDA operators available in the extension build.
    # operators unavailable and can break architecture-specific experiments.
    try:
        import torch.utils.cpp_extension as torch_cpp_ext
    except Exception:
        return
    for flag in (
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ):
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def build_config():
    if _BUILD_CONFIG:
        return dict(_BUILD_CONFIG)
    arch_list = _detect_arch_list()
    tag = _arch_tag(arch_list)
    return {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "torch_cuda": torch.version.cuda,
        "arch_list": arch_list,
        "arch_tag": tag,
        "build_dir": str(_BUILD_ROOT / tag),
        "extension_name": os.environ.get("VDM_W4A4_EXT_NAME", f"vdm_w4a4_ext_{tag}"),
    }


def _load_ext():
    global _EXT, _LOAD_ERROR, _BUILD_CONFIG
    if _EXT is not None:
        return _EXT
    _configure_cuda_home()
    _remove_unwanted_pytorch_nvcc_flags()
    arch_list = _detect_arch_list()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    tag = _arch_tag(arch_list)
    build_dir = _BUILD_ROOT / tag
    ext_name = os.environ.get("VDM_W4A4_EXT_NAME", f"vdm_w4a4_ext_{tag}")
    build_dir.mkdir(parents=True, exist_ok=True)
    _BUILD_CONFIG = {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "torch_cuda": torch.version.cuda,
        "arch_list": arch_list,
        "arch_tag": tag,
        "build_dir": str(build_dir),
        "extension_name": ext_name,
    }
    try:
        _EXT = load(
            name=ext_name,
            sources=[str(_SRC)],
            build_directory=str(build_dir),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=bool(int(os.environ.get("VDM_W4A4_VERBOSE", "0"))),
        )
        _LOAD_ERROR = None
        return _EXT
    except BaseException as exc:  # Keep the original exception for diagnostics.
        _LOAD_ERROR = exc
        raise


def is_available() -> bool:
    try:
        _load_ext()
        return True
    except BaseException:
        return False


def load_error() -> Optional[BaseException]:
    return _LOAD_ERROR


def _as_2d_half_contiguous(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]).half().contiguous(), orig_shape


def _pad_m(x_2d: torch.Tensor) -> Tuple[torch.Tensor, int]:
    m, k = x_2d.shape
    padded_m = ((m + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if padded_m == m:
        return x_2d, m
    padded = torch.empty((padded_m, k), device=x_2d.device, dtype=x_2d.dtype)
    padded[:m].copy_(x_2d)
    padded[m:].zero_()
    return padded, m


def _check_linear_shapes(x_2d: torch.Tensor, qweight: torch.Tensor, wscales: torch.Tensor) -> None:
    if x_2d.shape[1] % WARP_K != 0:
        raise ValueError(f"K={x_2d.shape[1]} must be a multiple of {WARP_K}")
    if qweight.shape[0] % BLOCK_N != 0:
        raise ValueError(f"N={qweight.shape[0]} must be a multiple of {BLOCK_N}")
    if qweight.shape[1] != x_2d.shape[1] // 2:
        raise ValueError("qweight shape must be [N, K/2]")
    if wscales.shape != (qweight.shape[0], x_2d.shape[1] // WARP_K):
        raise ValueError("wscales shape must be [N, K/64]")


def pack_weight_with_scale(weight: torch.Tensor, channel_scale: torch.Tensor):
    """Pack weight with caller-provided per-channel scale from the QuantLinear checkpoint."""
    ext = _load_ext()
    weight_h = weight.detach().half().contiguous()
    scale_h = channel_scale.detach().view(-1).half().contiguous().to(weight_h.device)
    return ext.pack_weight_with_scale(weight_h, scale_h)


def quantize_weight_per_channel(weight: torch.Tensor):
    ext = _load_ext()
    return ext.quantize_weight_per_channel(weight.detach().half().contiguous())


def linear_vdm_w4a4(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    clip_ratio: float = 1.0,
    bias_half: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ext = _load_ext()
    x_2d, orig_shape = _as_2d_half_contiguous(x)
    _check_linear_shapes(x_2d, qweight, wscales)
    b = bias_half if bias_half is not None else bias
    if b is not None:
        out = ext.linear_dynamic_bias(
            x_2d,
            qweight.contiguous(),
            wscales.contiguous(),
            b.half().contiguous().view(-1),
            float(clip_ratio),
        )
    else:
        out = ext.linear_dynamic(x_2d, qweight.contiguous(), wscales.contiguous(), float(clip_ratio))
    return out.view(*orig_shape, -1)


def quantize_activation_per_token(x: torch.Tensor, clip_ratio: float = 1.0):
    ext = _load_ext()
    x_2d, _ = _as_2d_half_contiguous(x)
    return ext.quantize_activation_per_token(x_2d, float(clip_ratio))


def linear_prequantized(qact: torch.Tensor, ascales: torch.Tensor, qweight: torch.Tensor, wscales: torch.Tensor) -> torch.Tensor:
    ext = _load_ext()
    # qact/ascales are expected to already be padded to BLOCK_M if needed by caller.
    return ext.linear_prequantized(qact.contiguous(), ascales.contiguous(), qweight.contiguous(), wscales.contiguous())


def linear_vdm_w4a4_qkv(
    x: torch.Tensor,
    q_weight_packed: torch.Tensor,
    q_weight_scale: torch.Tensor,
    k_weight_packed: torch.Tensor,
    k_weight_scale: torch.Tensor,
    v_weight_packed: torch.Tensor,
    v_weight_scale: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    clip_ratio: float = 1.0,
    w_qkv_packed: Optional[torch.Tensor] = None,
    w_qkv_scale: Optional[torch.Tensor] = None,
    qkv_bias_half: Optional[torch.Tensor] = None,
):
    if w_qkv_packed is None or w_qkv_scale is None:
        w_qkv_packed = torch.cat([q_weight_packed, k_weight_packed, v_weight_packed], dim=0).contiguous()
        w_qkv_scale = torch.cat([q_weight_scale, k_weight_scale, v_weight_scale], dim=0).contiguous()
    if qkv_bias_half is None:
        n = q_weight_packed.shape[0]
        has_bias = q_bias is not None or k_bias is not None or v_bias is not None
        if has_bias:
            device = w_qkv_packed.device
            zeros = lambda: torch.zeros(n, dtype=torch.float16, device=device)
            qkv_bias_half = torch.cat([
                q_bias.half().contiguous() if q_bias is not None else zeros(),
                k_bias.half().contiguous() if k_bias is not None else zeros(),
                v_bias.half().contiguous() if v_bias is not None else zeros(),
            ], dim=0)
    qkv = linear_vdm_w4a4(
        x,
        w_qkv_packed,
        w_qkv_scale,
        bias=None,
        clip_ratio=clip_ratio,
        bias_half=qkv_bias_half,
    )
    n = q_weight_packed.shape[0]
    return qkv.split(n, dim=-1)
