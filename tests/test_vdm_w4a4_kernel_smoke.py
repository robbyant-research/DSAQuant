try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from tests._bootstrap import bootstrap_project_root
bootstrap_project_root()

import torch
from vdm_infer.kernels import vdm_w4a4 as vdm


def main():
    print('torch', torch.__version__, 'cuda', torch.version.cuda)
    print('device', torch.cuda.get_device_name(0))
    print('vdm available', vdm.is_available(), vdm.load_error())
    torch.manual_seed(0)
    m, k, n = 257, 1536, 1536
    clip = 0.95
    x = torch.randn(m, k, device='cuda', dtype=torch.float16)
    w = torch.randn(n, k, device='cuda', dtype=torch.float16)
    b = torch.randn(n, device='cuda', dtype=torch.float16)
    wscale = (w.abs().amax(dim=1).clamp_min(1e-6) / 7).half()
    qw, wpacked_scale, _ = vdm.pack_weight_with_scale(w, wscale)
    out = vdm.linear_vdm_w4a4(x, qw, wpacked_scale, b, clip_ratio=clip)

    ascale = (x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7 * clip).float()
    qx = torch.round(x.float() / ascale).clamp(-8, 7)
    qwi = torch.round(w.float() / wscale.float().view(-1, 1)).clamp(-8, 7)
    ref = (qx @ qwi.t()) * ascale * wscale.float().view(1, -1) + b.float().view(1, -1)
    diff = (out.float() - ref).abs()
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)
    print('shape', tuple(out.shape))
    print('max_abs', diff.max().item())
    print('mean_abs', diff.mean().item())
    print('rel_l2', rel_l2.item())


if __name__ == '__main__':
    main()
