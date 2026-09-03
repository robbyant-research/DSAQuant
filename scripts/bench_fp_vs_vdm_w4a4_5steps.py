from __future__ import annotations

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap_project_root
bootstrap_project_root()


import argparse
import importlib
import os
import time


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


MODELS = {
    "1.3b": "scripts.eval_quant_vdm_single",
    "5b": "scripts.eval_quant_vdm_single_wan22_5b",
    "14b": "scripts.eval_quant_vdm_single_14b",
}


def configure_module(mod, model: str, attn: str, steps: int, precision: str, skip_export: bool) -> None:
    use_quant = precision == "w4a4"
    mod.USE_QUANTIZATION = use_quant
    mod.NUM_INFERENCE_STEPS = steps
    mod.ATTN_MODE = attn

    if hasattr(mod, "USE_VDM_W4A4_KERNEL"):
        mod.USE_VDM_W4A4_KERNEL = use_quant
    if hasattr(mod, "USE_FUSED_QKV_KERNEL"):
        mod.USE_FUSED_QKV_KERNEL = use_quant
    if hasattr(mod, "GENERATION_MODE"):
        mod.GENERATION_MODE = os.environ.get("GENERATION_MODE", "t2v")
    if hasattr(mod, "USE_MOE"):
        mod.USE_MOE = _env_bool("USE_MOE", getattr(mod, "USE_MOE", False))
    if hasattr(mod, "OUTPUT_DIR"):
        mod.OUTPUT_DIR = os.environ.get(
            "BENCH_OUTPUT_DIR",
            f"test_videos/fp_vs_vdm_w4a4_{model}_{precision}_{attn}",
        )
    if skip_export and hasattr(mod, "export_to_video"):
        def _skip_export(*args, **kwargs):
            print("[bench] skip export_to_video")
            return None
        mod.export_to_video = _skip_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--precision", choices=["fp", "w4a4"], required=True)
    parser.add_argument("--attn", choices=["sdpa", "flashattn_varlen", "sageattn_varlen"], default="sdpa")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-export", action="store_true", default=True)
    args = parser.parse_args()

    use_quant = args.precision == "w4a4"
    os.environ["ATTN_MODE"] = args.attn
    os.environ["USE_QUANTIZATION"] = "1" if use_quant else "0"
    os.environ["USE_VDM_W4A4_KERNEL"] = "1" if use_quant else "0"
    os.environ["USE_FUSED_QKV_KERNEL"] = "1" if use_quant else "0"
    os.environ["NUM_INFERENCE_STEPS"] = str(args.steps)

    mod = importlib.import_module(MODELS[args.model])
    configure_module(mod, args.model, args.attn, args.steps, args.precision, args.skip_export)

    print("=" * 80)
    print(f"BENCH_START model={args.model} precision={args.precision} attn={args.attn} steps={args.steps}")
    print(f"script={MODELS[args.model]}")
    print(f"USE_QUANTIZATION={getattr(mod, 'USE_QUANTIZATION', None)}")
    print(f"USE_VDM_W4A4_KERNEL={getattr(mod, 'USE_VDM_W4A4_KERNEL', None)}")
    print(f"USE_FUSED_QKV_KERNEL={getattr(mod, 'USE_FUSED_QKV_KERNEL', None)}")
    print(f"ATTN_MODE={getattr(mod, 'ATTN_MODE', None)}")
    print(f"GENERATION_MODE={getattr(mod, 'GENERATION_MODE', None)}")
    start = time.time()
    mod.main()
    elapsed = time.time() - start
    print(f"BENCH_TOTAL model={args.model} precision={args.precision} attn={args.attn} steps={args.steps} wall_s={elapsed:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
