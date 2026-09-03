#!/usr/bin/env python3
"""Export a public packed DSAQuant W4A4 checkpoint.

This script consumes the legacy/full quantized checkpoint format currently
used by the eval scripts, builds the quantized model once, packs QuantLinear
weights for the DSAQuant W4A4 kernel, and writes a smaller inference-only .pt.
"""

from __future__ import annotations

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap_project_root
bootstrap_project_root()

import argparse
import importlib
import os
from pathlib import Path

import torch

from vdm_infer.checkpoints.packed_checkpoint import PACKED_CHECKPOINT_FORMAT, save_packed_checkpoint


MODEL_TO_MODULE = {
    "1.3b": "scripts.eval_quant_vdm_single",
    "5b": "scripts.eval_quant_vdm_single_wan22_5b",
    "14b": "scripts.eval_quant_vdm_single_14b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export packed DSAQuant W4A4 checkpoint for inference-only release")
    parser.add_argument("--model", choices=sorted(MODEL_TO_MODULE), required=True, help="Model size/flavor to export")
    parser.add_argument("--pretrained-model-path", required=True, help="Diffusers pretrained model directory")
    parser.add_argument("--quant-checkpoint-path", required=True, help="Internal DCP/.pt quant checkpoint to pack")
    parser.add_argument("--output", required=True, help="Output packed .pt path")
    parser.add_argument(
        "--float-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Floating tensor dtype stored in the packed checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # The eval modules read these globals at import time for kernel toggles.
    os.environ["USE_QUANTIZATION"] = "1"
    os.environ["USE_VDM_W4A4_KERNEL"] = "1"
    os.environ["USE_FUSED_QKV_KERNEL"] = "1"
    os.environ["PRETRAINED_MODEL_PATH"] = args.pretrained_model_path
    os.environ["QUANT_CHECKPOINT_PATH"] = args.quant_checkpoint_path
    os.environ["FORCE_LEGACY_QUANT_CHECKPOINT"] = "1"

    module = importlib.import_module(MODEL_TO_MODULE[args.model])
    module.USE_QUANTIZATION = True
    module.USE_VDM_W4A4_KERNEL = True
    module.USE_FUSED_QKV_KERNEL = True
    module.PRETRAINED_MODEL_PATH = args.pretrained_model_path
    module.QUANT_CHECKPOINT_PATH = args.quant_checkpoint_path

    print(f"Exporting {args.model} as {PACKED_CHECKPOINT_FORMAT}")
    model, model_args, _ = module.build_model_quantized()
    float_dtype = torch.float16 if args.float_dtype == "float16" else torch.bfloat16
    save_packed_checkpoint(
        model,
        str(output),
        metadata={
            "model": args.model,
            "model_flavor": getattr(module, "MODEL_FLAVOR", None),
            "attn_mode": getattr(module, "ATTN_MODE", None),
            "source_quant_checkpoint": args.quant_checkpoint_path,
            "source_pretrained_model": args.pretrained_model_path,
            "format": PACKED_CHECKPOINT_FORMAT,
        },
        float_dtype=float_dtype,
    )

    size_gib = output.stat().st_size / 1024**3
    print(f"Packed checkpoint size: {size_gib:.3f} GiB")


if __name__ == "__main__":
    main()
