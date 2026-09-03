try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from tests._bootstrap import bootstrap_project_root
bootstrap_project_root()

import torch
from scripts import eval_quant_vdm_single as e
from vdm_infer.quantization.quant_linear import QuantLinear
from vdm_infer.models.wan_vdm.modules.attention import WanAttention


def main():
    model, model_args, device = e.build_model_quantized()
    ql = [m for m in model.modules() if isinstance(m, QuantLinear)]
    vdm = sum(1 for m in ql if getattr(m, 'use_vdm_kernel', False))
    attn = [m for m in model.modules() if isinstance(m, WanAttention)]
    fused_vdm = sum(
        1 for m in attn
        if getattr(m, 'use_fused_qkv_kernel', False) and getattr(m, 'qkv_kernel_backend', None) == 'vdm'
    )
    print('quantlinear_total', len(ql))
    print('vdm_kernel_enabled', vdm)
    print('attention_total', len(attn))
    print('vdm_fused_qkv_enabled', fused_vdm)
    print('cuda_allocated_gb', torch.cuda.memory_allocated() / 1e9)


if __name__ == '__main__':
    main()
