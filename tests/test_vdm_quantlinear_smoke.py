try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from tests._bootstrap import bootstrap_project_root
bootstrap_project_root()

import torch
import torch.nn as nn
from vdm_infer.quantization.quant_linear import QuantLinear


def main():
    torch.manual_seed(1)
    qparams = {
        'w': {'bit': 4, 'sym': True, 'granularity': 'per_channel', 'group_size': -1, 'round_zero': True, 'use_grad_scaling': True, 'cali': 'mse'},
        'act': {'bit': 4, 'sym': True},
    }
    ql = QuantLinear(nn.Linear(1536, 1536, bias=True), {'w': 'lsq', 'act': 'dynamic'}, qparams).cuda().half()
    ql.wquantizer(ql.weight.detach())
    ql.wquantizer.build()
    ql.aquantizer.build()
    ql = ql.half()
    ql.set_quant_state(use_wq=True, use_aq=True)
    x = torch.randn(257, 1536, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        fake = ql(x)
        ql.enable_vdm_kernel_mode()
        out = ql(x)
    diff = (out.float() - fake.float()).abs()
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(fake.float())
    print('use_vdm_kernel', ql.use_vdm_kernel)
    print('shape', tuple(out.shape))
    print('max_abs', diff.max().item())
    print('mean_abs', diff.mean().item())
    print('rel_l2', rel_l2.item())


if __name__ == '__main__':
    main()
