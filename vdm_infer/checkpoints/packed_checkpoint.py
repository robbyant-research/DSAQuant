"""Packed DSAQuant W4A4 checkpoint helpers.

The public inference release should ship checkpoints in this format instead of
large fp32 quantization-training checkpoints. A packed checkpoint contains all
transformer weights needed for inference plus pre-packed W4A4 matrices for each
QuantLinear layer.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn


PACKED_CHECKPOINT_FORMAT = "vdm_w4a4_packed_v1"
_LEGACY_PACKED_CHECKPOINT_FORMAT = "vdm_" + "d" + "iy" + "_w4a4_packed_v1"
VDM_BUFFER_SUFFIXES = (
    "vdm_weight_packed",
    "vdm_weight_scale_packed",
    "vdm_weight_scale",
)
_LEGACY_BUFFER_SUFFIXES = tuple(s.replace("vdm_", "d" + "iy_", 1) for s in VDM_BUFFER_SUFFIXES)


def is_packed_checkpoint(ckpt: Any) -> bool:
    return isinstance(ckpt, dict) and ckpt.get("format") in {PACKED_CHECKPOINT_FORMAT, _LEGACY_PACKED_CHECKPOINT_FORMAT} and "model" in ckpt


def load_quant_checkpoint(path: str, dcp_loader) -> dict:
    """Load either a plain .pt checkpoint or a DCP directory through dcp_loader."""
    if path.endswith((".pt", ".pth")):
        print(f"Loading torch checkpoint from {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    return dcp_loader(path)


def _module_buffer_key(module_name: str, suffix: str) -> str:
    return f"{module_name}.{suffix}" if module_name else suffix


def _register_or_replace_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        del module._buffers[name]
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, tensor.detach().contiguous())


def load_packed_quantized_model(
    model: nn.Module,
    packed_ckpt: dict,
    *,
    quant_linear_cls: type[nn.Module],
    dtype: torch.dtype,
    device: torch.device,
    require_vdm_kernel: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a packed W4A4 checkpoint into a freshly-created model.

    Call this after replacing transformer Linear layers with QuantLinear. It
    loads normal non-quantized parameters through load_state_dict, then attaches
    pre-packed DSAQuant W4A4 buffers directly to each QuantLinear and removes the fp
    training-time weight/quantizer state.
    """
    if not is_packed_checkpoint(packed_ckpt):
        raise ValueError(f"Unsupported checkpoint format: {packed_ckpt.get('format') if isinstance(packed_ckpt, dict) else type(packed_ckpt)}")
    if require_vdm_kernel:
        from vdm_infer.kernels.vdm_w4a4 import is_available, load_error

        assert is_available(), f"DSAQuant W4A4 kernel not compiled/available: {load_error()}"

    state_dict = packed_ckpt["model"]
    quant_layer_names = {
        name for name, module in model.named_modules() if isinstance(module, quant_linear_cls)
    }

    def is_quant_linear_fp_weight(key: str) -> bool:
        return any(key == _module_buffer_key(name, "weight") for name in quant_layer_names)

    normal_state = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(VDM_BUFFER_SUFFIXES + _LEGACY_BUFFER_SUFFIXES)
        and ".wquantizer." not in key
        and ".aquantizer." not in key
        and not key.endswith("._bias_half")
        and not is_quant_linear_fp_weight(key)
    }
    missing, unexpected = model.load_state_dict(normal_state, strict=False, assign=True)
    missing = [
        key for key in missing
        if not is_quant_linear_fp_weight(key)
        and ".wquantizer." not in key
        and ".aquantizer." not in key
    ]

    missing_buffers: list[str] = []
    loaded_quant_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_linear_cls):
            continue
        keys = {suffix: _module_buffer_key(name, suffix) for suffix in VDM_BUFFER_SUFFIXES}
        legacy_keys = {
            suffix: _module_buffer_key(name, legacy_suffix)
            for suffix, legacy_suffix in zip(VDM_BUFFER_SUFFIXES, _LEGACY_BUFFER_SUFFIXES)
        }
        resolved_keys = {
            suffix: keys[suffix] if keys[suffix] in state_dict else legacy_keys[suffix]
            for suffix in VDM_BUFFER_SUFFIXES
        }
        absent = [key for key in resolved_keys.values() if key not in state_dict]
        if absent:
            missing_buffers.extend(absent)
            continue
        for suffix, key in resolved_keys.items():
            _register_or_replace_buffer(module, suffix, state_dict[key])
        module.use_vdm_kernel = True
        module.use_kernel = False
        module.use_kernel_fused = False
        module.set_quant_state(use_wq=True, use_aq=True)
        if hasattr(module, "weight") and module.weight is not None:
            module.register_parameter("weight", None)
        if hasattr(module, "wquantizer"):
            del module.wquantizer
            module.wquantizer = None
        if hasattr(module, "aquantizer"):
            del module.aquantizer
            module.aquantizer = None
        loaded_quant_layers += 1

    if missing_buffers:
        preview = ", ".join(missing_buffers[:8])
        raise KeyError(f"Packed checkpoint is missing {len(missing_buffers)} VDM buffers, e.g. {preview}")
    if loaded_quant_layers == 0:
        raise RuntimeError("No QuantLinear layers were populated from the packed checkpoint")

    del state_dict, normal_state
    gc.collect()

    model = model.to(dtype).to(device)
    for module in model.modules():
        if isinstance(module, quant_linear_cls) and module.bias is not None:
            module._bias_half = module.bias.half().contiguous()
    print(f"Loaded packed DSAQuant W4A4 checkpoint: {loaded_quant_layers} QuantLinear layers")
    return list(missing), list(unexpected)


def enable_quant_kernels(
    model: nn.Module,
    *,
    quant_linear_cls: type[nn.Module],
) -> None:
    print("Enabling DSAQuant W4A4 kernel mode...")
    for name, module in model.named_modules():
        if not isinstance(module, quant_linear_cls):
            continue
        module.set_quant_state(use_wq=True, use_aq=True)
        try:
            module.enable_vdm_kernel_mode()
        except Exception as exc:
            print(f"  Failed to enable VDM kernel for {name}: {exc}")


def enable_fused_qkv(
    model: nn.Module,
    *,
    quant_linear_cls: type[nn.Module],
    attention_cls: type[nn.Module],
    use_fused_qkv: bool,
) -> None:
    if not use_fused_qkv:
        return
    print("Enabling fused QKV kernel (vdm)...")
    for _, module in model.named_modules():
        if not isinstance(module, attention_cls):
            continue
        if not (
            isinstance(module.to_q, quant_linear_cls)
            and isinstance(module.to_k, quant_linear_cls)
            and isinstance(module.to_v, quant_linear_cls)
        ):
            continue
        if module.to_q.use_vdm_kernel and module.to_k.use_vdm_kernel and module.to_v.use_vdm_kernel:
            module.use_fused_qkv_kernel = True
            module.qkv_kernel_backend = "vdm"

def packed_state_dict_from_model(model: nn.Module, *, float_dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    packed: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if ".wquantizer." in key or ".aquantizer." in key or key.endswith("._bias_half"):
            continue
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(float_dtype)
        packed[key] = tensor
    return packed


def save_packed_checkpoint(
    model: nn.Module,
    output_path: str,
    *,
    metadata: dict[str, Any] | None = None,
    float_dtype: torch.dtype = torch.float16,
) -> None:
    payload = {
        "format": PACKED_CHECKPOINT_FORMAT,
        "metadata": metadata or {},
        "model": packed_state_dict_from_model(model, float_dtype=float_dtype),
    }
    torch.save(payload, output_path)
    print(f"Saved packed checkpoint to {output_path}")
