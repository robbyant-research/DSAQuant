#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export USE_VDM_W4A4_KERNEL=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export VDM_W4A4_TARGET_ARCH="${VDM_W4A4_TARGET_ARCH:-auto}"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST
fi
export ATTN_MODE="${ATTN_MODE:-flashattn_varlen}"
export USE_FAST_ROPE="${USE_FAST_ROPE:-0}"
export USE_NATIVE_RMSNORM="${USE_NATIVE_RMSNORM:-0}"

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1 && [[ -x /home/shuaiting/miniconda3/bin/conda ]]; then
  CONDA_BIN=/home/shuaiting/miniconda3/bin/conda
fi
CONDA_ENV="${CONDA_ENV:-vdm}"
MODE="${1:-eval}"

case "${MODE}" in
  eval|run)
    exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u scripts/eval_quant_vdm_single.py
    ;;
  smoke|build)
    exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u tests/test_vdm_eval_build_smoke.py
    ;;
  kernel-smoke)
    exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u tests/test_vdm_w4a4_kernel_smoke.py
    ;;
  qkv-smoke)
    exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u tests/test_vdm_qkv_fuse_smoke.py
    ;;
  *)
    echo "Usage: $0 [eval|smoke|kernel-smoke|qkv-smoke]" >&2
    exit 2
    ;;
esac
