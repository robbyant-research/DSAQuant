# DSAQuant

**DSAQuant: Denoising-Stage-Aligned Quantization-Aware Training for Video Generation**

[**Paper**](https://arxiv.org/abs/2609.04031) | [**Project Page**](https://robbyant-research.github.io/DSAQuant/) | [**Model Page**](https://huggingface.co/Robbyant-Research/DSAQuant)

Shuaiting Li [1,2], Zelin Gao [1,4], Haibin Shen [2], Yujun Shen [1], Haotong Qin<sup>†</sup> [3], Yinghao Xu<sup>†</sup> [4,1]

[1] Robby Ant, [2] Zhejiang University, [3] PolyU, [4] HKUST

<sup>†</sup> Corresponding authors.

This repository provides the inference implementation for DSAQuant, including
W4A4 quantized inference for Wan video generation models, packed checkpoint
loading, and the CUDA kernel path used for efficient deployment.

## Overview

DSAQuant targets efficient video generation by combining quantization-aware
training with inference-time optimizations aligned to the denoising process. The
implementation is organized around reproducible inference, checkpoint
export, and latency evaluation. It includes the runtime components needed to
evaluate DSAQuant checkpoints and compare them with FP16 baselines.

Supported inference entrypoints:

- Wan 2.1 T2V 1.3B
- Wan 2.1 T2V 14B
- Wan 2.2 5B T2V

This repository does not include model weights, quantized checkpoints, training
data, or training infrastructure.

## Features

- Packed DSAQuant W4A4 checkpoints for inference-only deployment.
- Dynamic A4 activation quantization and symmetric W4 weight quantization.
- Fused QKV support for the DSAQuant W4A4 CUDA kernel path.
- CFG-drop support for low-noise denoising steps.
- FP16 baseline path for latency and quality comparison.
- CUDA JIT build support for sm86 and sm89 targets.

## Installation

Create or activate a CUDA-compatible Python environment, then install the pinned
runtime dependencies:

```bash
pip install -r requirements.txt
```

The tested environment uses CUDA 12.6 and PyTorch CUDA 12.6 development wheels.
CUDA/PyTorch-specific packages such as `torch`, `torchvision`, `flash_attn_3`,
and `sageattention` may need to be installed from the matching wheel index or
built locally for your platform.

## Checkpoints

Pretrained packed DSAQuant checkpoints for inference are available on the
[DSAQuant Hugging Face model page](https://huggingface.co/Robbyant-Research/DSAQuant).
Download the checkpoint for the desired model variant, then set
`QUANT_CHECKPOINT_PATH` to the local `.pt` file when running an inference
entrypoint.

For W4A4 inference, use packed checkpoint `.pt` files. Packed checkpoints
use:

```text
format = "vdm_w4a4_packed_v1"
```

A packed checkpoint contains the transformer weights and pre-packed DSAQuant W4A4
tensors required by the inference scripts. When this format is used, the inference
path does not load base transformer safetensors and does not repack fp32 quantized
weights at startup.

## Usage

Set `PRETRAINED_MODEL_PATH` to the corresponding Wan Diffusers model directory
and `QUANT_CHECKPOINT_PATH` to the packed DSAQuant checkpoint.

```bash
# 1.3B DSAQuant W4A4 inference
USE_QUANTIZATION=1 \
USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan1.3b_w4a4_packed.pt \
PRETRAINED_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
python scripts/eval_quant_vdm_single.py

# 5B DSAQuant W4A4 inference
USE_QUANTIZATION=1 \
USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan2.2_5b_w4a4_packed.pt \
PRETRAINED_MODEL_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers \
python scripts/eval_quant_vdm_single_wan22_5b.py

# 14B DSAQuant W4A4 inference
USE_QUANTIZATION=1 \
USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan2.1_14b_w4a4_packed.pt \
PRETRAINED_MODEL_PATH=/path/to/Wan2.1-T2V-14B-Diffusers \
python scripts/eval_quant_vdm_single_14b.py
```

To run the FP16 baseline, disable quantization:

```bash
BOUNDARY_RATIO=none \
USE_QUANTIZATION=0 \
PRETRAINED_MODEL_PATH=/path/to/Wan2.1-T2V-14B-Diffusers \
python scripts/eval_quant_vdm_single_14b.py
```

## Boundary Ratio and CFG Drop

The inference scripts support CFG drop through `BOUNDARY_RATIO` and
`GUIDANCE_SCALE_2`. The default inference setting used for the released
checkpoints is:

```bash
BOUNDARY_RATIO=0.4 GUIDANCE_SCALE_2=1.0
```

With this setting, the pipeline uses normal CFG (`guidance_scale=5.0`) for the
high-noise part of the sampling trajectory. Once the scheduler timestep falls
below `BOUNDARY_RATIO * num_train_timesteps`, the active guidance scale switches
to `GUIDANCE_SCALE_2`. When the active guidance scale is `1.0` or lower, the
pipeline skips the unconditional forward pass and uses only the conditional model
output. This is the actual CFG-drop optimization: it reduces inference work in
low-noise steps instead of only setting the CFG formula scale to 1 while still
computing the unconditional branch.

`boundary_ratio` is an engineering implementation detail for the guidance
schedule. It does not mean that the checkpoint is split into timestep-specific
models, and it does not mean that different denoising timesteps use different
trained model weights. For the single-transformer entrypoints provided here,
the same transformer object is reused on both sides of the boundary; only the
guidance behavior changes.

To disable boundary-based CFG drop for either FP16 or W4A4 evaluation, set:

```bash
BOUNDARY_RATIO=none
```

## CUDA Kernel

The repository ships DSAQuant W4A4 CUDA source code rather than prebuilt `.so`
files. On the first W4A4 run, PyTorch JIT-compiles
`vdm_infer/kernels/csrc/vdm_w4a4_kernel.cu` for the target GPU and stores the
build cache under:

```text
vdm_infer/kernels/.torch_extensions_vdm_w4a4/<arch-tag>/
```

The wrapper chooses the target architecture in this order:

1. `TORCH_CUDA_ARCH_LIST`, if set.
2. `VDM_W4A4_TARGET_ARCH`, if set. Supported aliases include `a6000`, `sm86`,
   `4090`, `rtx4090`, `ada`, and `sm89`.
3. The visible CUDA device reported by `torch.cuda.get_device_capability()`.

Examples:

```bash
# RTX A6000 / sm86
TORCH_CUDA_ARCH_LIST=8.6 \
USE_QUANTIZATION=1 USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan1.3b_w4a4_packed.pt \
python scripts/eval_quant_vdm_single.py

# RTX 4090 / sm89
CUDA_HOME=/usr/local/cuda-12.6 \
TORCH_CUDA_ARCH_LIST=8.9 \
VDM_W4A4_TARGET_ARCH=4090 \
USE_QUANTIZATION=1 USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan1.3b_w4a4_packed.pt \
python scripts/eval_quant_vdm_single.py

# Build a local extension for both A6000 and 4090
TORCH_CUDA_ARCH_LIST="8.6;8.9" \
USE_QUANTIZATION=1 USE_VDM_W4A4_KERNEL=1 \
QUANT_CHECKPOINT_PATH=/path/to/wan1.3b_w4a4_packed.pt \
python scripts/eval_quant_vdm_single.py
```

For other NVIDIA GPUs, either leave both architecture variables unset and let the
wrapper detect the visible device, or set the compute capability explicitly.

To inspect the build configuration without running inference:

```bash
python - <<'PYCODE'
from vdm_infer.kernels import vdm_w4a4
print(vdm_w4a4.build_config())
PYCODE
```

If you change CUDA, PyTorch, or `TORCH_CUDA_ARCH_LIST`, remove the old local cache
before rebuilding:

```bash
rm -rf vdm_infer/kernels/.torch_extensions_vdm_w4a4/sm89
rm -rf vdm_infer/kernels/.torch_extensions_vdm_w4a4/sm86
```

## Exporting Packed Checkpoints

Use the exporter to convert a trained quantized checkpoint into the packed
inference format:

```bash
python scripts/export_packed_checkpoint.py \
  --model 1.3b \
  --pretrained-model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --quant-checkpoint-path /path/to/wan1.3b_w4a4_checkpoint_or_full.pt \
  --output /path/to/wan1.3b_w4a4_packed.pt
```

Use `--model 5b` or `--model 14b` for the other variants. The exporter builds the
quantized model once, enables the DSAQuant W4A4 kernel to pack weights, and saves
only inference-required tensors.

## Quantization Details

- Weight quantization is symmetric W4 per output channel.
- Packed weight scales are per-channel, not per-group.
- Activation quantization is dynamic A4 per token at inference time.
- The fused QKV path uses the same per-channel weight scales and per-token
  activation quantization as the standalone linear path.
- No LoRA path is used by the packed inference checkpoints.

## Acknowledgements

We thank the authors of [Wan](https://github.com/Wan-Video/Wan2.1),
[CogVideo](https://github.com/zai-org/CogVideo), and
[QVGen](https://github.com/ModelTC/QVGen) for making their excellent work
available to the community. Their open-source contributions provided valuable
foundations and references for this project.

## Citation

If you find this repository useful, please cite the DSAQuant paper. Citation
information will be updated when available.
