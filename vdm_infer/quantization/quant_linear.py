import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformQuantizer(nn.Module):
    def __init__(
        self,
        bit: int = 4,
        sym: bool = True,
        granularity: str = "per_channel",
        group_size: int = -1,
        round_zero: bool = True,
        cali: str = "mse",
        clip_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.bit = bit
        self.sym = sym
        self.granularity = granularity
        self.group_size = group_size
        self.round_zero = round_zero
        self.cali = cali
        self.clip_ratio = clip_ratio
        self.P = 2 ** (bit - 1) - 1 if sym else 2**bit - 1
        self.N = -(2 ** (bit - 1)) if sym else 0
        self.scale = None
        self.zero_point = None
        self.init = False

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        if self.granularity == "per_tensor":
            return x.reshape(1, -1)
        if self.granularity == "per_channel":
            return x.reshape(x.shape[0], -1)
        if self.granularity == "per_token":
            return x.reshape(-1, x.shape[-1])
        if self.granularity == "per_group":
            return x.reshape(-1, self.group_size)
        raise ValueError(f"Unsupported granularity: {self.granularity}")

    def init_qparams(self, x: torch.Tensor) -> None:
        x2 = self.reshape(x.float())
        min_x = x2.amin(dim=-1, keepdim=True) * self.clip_ratio
        max_x = x2.amax(dim=-1, keepdim=True) * self.clip_ratio
        if self.sym:
            scale = torch.maximum(min_x.abs(), max_x.abs()).clamp(min=1e-5) / self.P
            zero_point = None
        else:
            scale = ((max_x - min_x) / self.P).clamp_min(1e-9)
            zero_point = (-torch.round(min_x / scale)).clamp(self.N, self.P)
        if hasattr(self, "scale"):
            del self.scale
        self.register_buffer("scale", scale.to(dtype=x.dtype))
        if hasattr(self, "zero_point"):
            del self.zero_point
        if zero_point is not None:
            self.register_buffer("zero_point", zero_point.to(dtype=x.dtype))
        else:
            self.zero_point = None
        self.init = True

    def quant(self, x: torch.Tensor, scale: torch.Tensor, zero_point):
        if zero_point is None:
            return torch.clamp(torch.round(x / scale), self.N, self.P)
        if self.round_zero:
            return torch.clamp(torch.round(x / scale) + zero_point, self.N, self.P)
        return torch.clamp(torch.round(x / scale + zero_point), self.N, self.P)

    def dequant(self, x: torch.Tensor, scale: torch.Tensor, zero_point):
        if zero_point is None:
            return x * scale
        return scale * (x - zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.init or self.scale is None:
            self.init_qparams(x)
        org_shape = x.shape
        x2 = self.reshape(x)
        out = self.dequant(self.quant(x2, self.scale, self.zero_point), self.scale, self.zero_point)
        return out.reshape(org_shape)

    def build(self) -> None:
        if self.scale is not None:
            self.init = True


QuantizerMap = {
    "uniform": UniformQuantizer,
    "lsq": UniformQuantizer,
    "lsq+": UniformQuantizer,
    "dynamic": UniformQuantizer,
    "learnable_clipped_dynamic": UniformQuantizer,
}


class QuantLinear(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        quantizer_type: dict = {"w": "uniform", "act": "uniform"},
        q_params: dict = {"w": {}, "act": {}},
    ) -> None:
        super().__init__()
        self.kwd_func = F.linear
        self.weight = nn.Parameter(layer.weight.detach().clone())
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None
        self.use_wq = False
        self.use_aq = False
        self.wquantizer = QuantizerMap[quantizer_type["w"]](**q_params["w"])
        self.aquantizer = QuantizerMap[quantizer_type["act"]](**q_params["act"])
        self.extra_repr_prefix = layer.extra_repr()
        self.use_hadamard = False
        self.use_timestep_scaling = False
        self.use_kernel = False
        self.use_kernel_fused = False
        self.use_vdm_kernel = False
        self.kernel_clip_ratio = 0.95
        self._bias_half = None

    def set_quant_state(self, use_wq: bool = False, use_aq: bool = False) -> None:
        self.use_wq = use_wq
        self.use_aq = use_aq

    def set_cur_timestep(self, t) -> None:
        self.cur_timestep = t

    def set_hadamard(self, use_hadamard: bool = False) -> None:
        if use_hadamard:
            raise NotImplementedError("Hadamard rotation is not included in the inference-only release.")
        self.use_hadamard = False

    def prepare_vdm_kernel_weights(self):
        from vdm_infer.kernels.vdm_w4a4 import pack_weight_with_scale
        assert self.wquantizer is not None and self.wquantizer.scale is not None, "Weight quantizer scale is missing"
        assert self.wquantizer.sym, "Only symmetric quantization supported for VDM kernel"
        scale = self.wquantizer.scale.detach().view(-1)
        qweight, packed_scales, channel_scales = pack_weight_with_scale(self.weight.detach(), scale)
        self.register_buffer("vdm_weight_packed", qweight)
        self.register_buffer("vdm_weight_scale_packed", packed_scales)
        self.register_buffer("vdm_weight_scale", channel_scales.half())

    def enable_vdm_kernel_mode(self):
        from vdm_infer.kernels.vdm_w4a4 import is_available, load_error
        assert is_available(), f"DSAQuant W4A4 kernel not compiled/available: {load_error()}"
        if not hasattr(self, "vdm_weight_packed"):
            self.prepare_vdm_kernel_weights()
        self.use_vdm_kernel = True
        self.use_kernel = False
        self.use_kernel_fused = False
        if self.bias is not None:
            self._bias_half = self.bias.half().contiguous()
        if hasattr(self, "weight") and self.weight is not None:
            self.register_parameter("weight", None)
        if hasattr(self, "wquantizer") and self.wquantizer is not None:
            del self.wquantizer
            self.wquantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_vdm_kernel:
            from vdm_infer.kernels.vdm_w4a4 import linear_vdm_w4a4
            return linear_vdm_w4a4(
                x,
                self.vdm_weight_packed,
                self.vdm_weight_scale_packed,
                self.bias,
                self.kernel_clip_ratio,
                bias_half=self._bias_half,
            )
        w = self.weight
        if self.use_wq and self.wquantizer is not None:
            w = self.wquantizer(w)
        if self.use_aq and self.aquantizer is not None:
            x = self.aquantizer(x)
        return self.kwd_func(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"{self.extra_repr_prefix}, use_aq={self.use_aq}, use_wq={self.use_wq}, use_vdm_kernel={self.use_vdm_kernel}"
