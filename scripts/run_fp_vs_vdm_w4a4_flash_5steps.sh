#!/usr/bin/env bash
set -uo pipefail
trap 'echo "Interrupted" >&2; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export NUM_INFERENCE_STEPS=5
export ATTN_MODE=flashattn_varlen

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1 && [[ -x /home/shuaiting/miniconda3/bin/conda ]]; then
  CONDA_BIN=/home/shuaiting/miniconda3/bin/conda
fi
LOG_DIR="${LOG_DIR:-logs/fp_vs_vdm_w4a4_flash_5steps}"
mkdir -p "${LOG_DIR}"

overall_status=0

run_one() {
  local model="$1"
  local precision="$2"
  local log="${LOG_DIR}/${model}_${precision}_flashattn_varlen.log"
  echo "===== RUN model=${model} precision=${precision} attn=flashattn_varlen gpu=${CUDA_VISIBLE_DEVICES} $(date '+%F %T') =====" | tee "${log}"
  ATTN_MODE=flashattn_varlen BENCH_OUTPUT_DIR="test_videos/fp_vs_vdm_w4a4_${model}_${precision}_flashattn_varlen" \
    "${CONDA_BIN}" run --no-capture-output -n vdm \
    python -u scripts/bench_fp_vs_vdm_w4a4_5steps.py --model "${model}" --precision "${precision}" --attn flashattn_varlen --steps 5 \
    2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  echo "===== END model=${model} precision=${precision} attn=flashattn_varlen status=${status} $(date '+%F %T') =====" | tee -a "${log}"
  if [[ ${status} -ne 0 ]]; then
    overall_status=${status}
  fi
}

for model in 1.3b 5b 14b; do
  run_one "${model}" fp
  run_one "${model}" w4a4
done

exit ${overall_status}
