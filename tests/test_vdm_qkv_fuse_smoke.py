try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from tests._bootstrap import bootstrap_project_root
bootstrap_project_root()

import torch
from vdm_infer.models.wan_vdm.modules.attention import WanAttention, _get_qkv_projections
from vdm_infer.quantization.quant_linear import QuantLinear


def make_quant(linear):
    qparams = {
        'w': {'bit': 4, 'sym': True, 'granularity': 'per_channel', 'group_size': -1, 'round_zero': True, 'use_grad_scaling': True, 'cali': 'mse'},
        'act': {'bit': 4, 'sym': True},
    }
    ql = QuantLinear(linear, {'w': 'lsq', 'act': 'dynamic'}, qparams).cuda().half()
    ql.wquantizer(ql.weight.detach())
    ql.wquantizer.build()
    ql.aquantizer.build()
    ql = ql.half()
    ql.set_quant_state(use_wq=True, use_aq=True)
    ql.enable_vdm_kernel_mode()
    return ql


def main():
    torch.manual_seed(2)
    attn = WanAttention(dim=1536, heads=12, dim_head=128).cuda().half()
    attn.to_q = make_quant(attn.to_q)
    attn.to_k = make_quant(attn.to_k)
    attn.to_v = make_quant(attn.to_v)
    x = torch.randn(1, 257, 1536, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        q0 = attn.to_q(x)
        k0 = attn.to_k(x)
        v0 = attn.to_v(x)
        attn.use_fused_qkv_kernel = True
        attn.qkv_kernel_backend = 'vdm'
        q1, k1, v1 = _get_qkv_projections(attn, x, None)
    for name, a, b in [('q', q1, q0), ('k', k1, k0), ('v', v1, v0)]:
        diff = (a.float() - b.float()).abs()
        rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b.float())
        print(name, 'max_abs', diff.max().item(), 'mean_abs', diff.mean().item(), 'rel_l2', rel.item())


if __name__ == '__main__':
    main()
