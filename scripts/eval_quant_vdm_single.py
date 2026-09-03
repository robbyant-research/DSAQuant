"""
Single-GPU inference script supporting FP16 and quantized comparisons.

Usage:
    # Quantized inference (W4A4 fused kernel + fused QKV)
    python scripts/eval_quant_vdm_single.py

    # Floating-point inference (original model without quantization)
    Set USE_QUANTIZATION = False below before running.

Run directly with Python; torchrun is not required.
"""

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap_project_root
bootstrap_project_root()

import os
import gc
import time
import torch
import torch.nn as nn

# ---- Project imports ----
from vdm_infer.models.wan_vdm import WanTransformer3DModel, vdm_wan_2_1_configs
from vdm_infer.models.wan_vdm.modules.attention import WanAttention
from vdm_infer.models.wan_vdm.pipelines import WanPipeline
from vdm_infer.models.wan_vdm.schedulers import FlowUniPCMultistepScheduler
from vdm_infer.models.base.vdm_utils import (
    load_vae,
    load_tokenizer,
    load_text_encoder,
)
from vdm_infer.models.wan_vdm.modules.state_dict_adapter import WanStateDictAdapter

from diffusers.utils import export_to_video


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# ============================================================
# Configuration - update paths and parameters here
# ============================================================

# ---- Main switches ----
USE_QUANTIZATION = _env_bool("USE_QUANTIZATION", False)   # False = FP16 inference; True = W4A4 quantized inference

# Pretrained model path (Diffusers format)
PRETRAINED_MODEL_PATH = os.environ.get("PRETRAINED_MODEL_PATH", "/path/to/Wan2.1-T2V-1.3B-Diffusers")

# Quantized checkpoint path (DCP format is converted to PyTorch format and cached automatically)
# Used only when USE_QUANTIZATION = True
QUANT_CHECKPOINT_PATH = os.environ.get("QUANT_CHECKPOINT_PATH", "/path/to/wan1.3b_w4a4_checkpoint")

# Output directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/wan1.3b")

# Inference settings
PROMPT = "A cat sitting on a windowsill watching the rain outside."
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
FLOW_SHIFT = 3.0
_BOUNDARY_RATIO_ENV = os.environ.get("BOUNDARY_RATIO", "0.4")
BOUNDARY_RATIO = None if _BOUNDARY_RATIO_ENV.lower() in ("none", "null", "off") else float(_BOUNDARY_RATIO_ENV)
GUIDANCE_SCALE_2 = float(os.environ.get("GUIDANCE_SCALE_2", "1.0"))

# Quantized kernel settings (effective only when USE_QUANTIZATION = True)
USE_VDM_W4A4_KERNEL = _env_bool("USE_VDM_W4A4_KERNEL", USE_QUANTIZATION)
USE_FUSED_QKV_KERNEL = _env_bool("USE_FUSED_QKV_KERNEL", USE_QUANTIZATION)

# Attention mode: mask-free SDPA (PyTorch selects a FlashAttention kernel when available).
ATTN_MODE = os.environ.get("ATTN_MODE", "sdpa")

# Precompute text embeddings and delete the text encoder (saves ~8 GB of peak VRAM)
PRECOMPUTE_TEXT_EMBEDDING = True

# Model configuration
MODEL_FLAVOR = "t2v_1.3B"
DTYPE = torch.float16

# ============================================================


def load_full_state_dict_from_safetensors(transformer_dir: str) -> dict:
    """Build a complete state dict from Diffusers safetensors shards."""
    from safetensors.torch import load_file as load_safetensors_file
    import json as _json

    index_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    print(f"Loading pretrained weights from {index_path}")
    with open(index_path, "r") as f:
        index_data = _json.load(f)

    weight_map = index_data.get("weight_map", {})
    shard_files = sorted({v for v in weight_map.values()})

    full_state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        shard_path = os.path.join(transformer_dir, shard)
        if not os.path.exists(shard_path):
            print(f"[Warning] Shard file not found: {shard_path}")
            continue
        print(f"  Loading shard: {shard}")
        shard_state = load_safetensors_file(shard_path, device="cpu")
        full_state_dict.update(shard_state)

    return full_state_dict


def convert_dcp_to_torch(dcp_path: str, cache_path: str = None) -> dict:
    """Convert a DCP checkpoint to standard PyTorch format."""
    if cache_path is None:
        cache_path = dcp_path.rstrip("/") + ".pt"

    if os.path.exists(cache_path):
        print(f"Loading cached torch checkpoint from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"Converting DCP checkpoint from {dcp_path} to torch format...")
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    # Find the latest step directory.
    if os.path.isdir(dcp_path):
        step_dirs = sorted([d for d in os.listdir(dcp_path) if d.startswith("step-")])
        if step_dirs:
            dcp_path = os.path.join(dcp_path, step_dirs[-1])
            print(f"  Using latest step: {dcp_path}")

    dcp_to_torch_save(dcp_path, cache_path)
    print(f"  Saved torch checkpoint to {cache_path}")

    result = torch.load(cache_path, map_location="cpu", weights_only=True)
    return result


def quant_module_refactor(model: nn.Module, ignore_list: list = None):
    """Replace nn.Linear modules with QuantLinear modules."""
    from vdm_infer.quantization.quant_linear import QuantLinear

    if ignore_list is None:
        ignore_list = []

    quantizer_type = {"w": "lsq", "act": "dynamic"}
    q_params = {
        "w": {
            "bit": 4, "sym": True, "granularity": "per_channel",
            "group_size": -1, "round_zero": True, "use_grad_scaling": True, "cali": "mse"
        },
        "act": {"bit": 4, "sym": True},
    }

    def find_module_name(model, target_module):
        for name, module in model.named_modules():
            if module is target_module:
                return name
        return None

    for name, child_module in model.named_children():
        module_name = find_module_name(model, child_module)
        if any([ignore_key in module_name for ignore_key in ignore_list]):
            continue
        if isinstance(child_module, nn.Linear):
            quantized_module = QuantLinear(
                layer=child_module,
                quantizer_type=quantizer_type,
                q_params=q_params,
            )
            setattr(model, name, quantized_module)
            print(f"  Refactored: {module_name} -> QuantLinear")
        else:
            quant_module_refactor(child_module, ignore_list)


def build_model_fp16():
    """Build the floating-point model without quantization."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load the model configuration.
    model_args = vdm_wan_2_1_configs[MODEL_FLAVOR]
    model_args.pretrained_model_name_or_path = PRETRAINED_MODEL_PATH
    model_args.cp_size = 1
    model_args.attn_mode = ATTN_MODE

    # 2. Create the model.
    print(f"Creating model (fp16, no quantization, attn={ATTN_MODE})...")
    model = WanTransformer3DModel(model_args)

    # 3. Load pretrained weights.
    print("Loading pretrained weights...")
    transformer_dir = f"{PRETRAINED_MODEL_PATH}/transformer"
    full_state_dict = load_full_state_dict_from_safetensors(transformer_dir)
    sd_adapter = WanStateDictAdapter(model_args)
    full_state_dict = sd_adapter.from_hf(full_state_dict)
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False, assign=True)
    del full_state_dict
    gc.collect()
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # 4. Cast to the target dtype and move to the GPU.
    model = model.to(DTYPE).to(device)
    model.eval()
    print("FP16 model ready for inference!")
    return model, model_args, device


def build_model_quantized():
    """Build the quantized model with the W4A4 kernel."""
    from vdm_infer.quantization.quant_linear import QuantLinear
    from vdm_infer.checkpoints.packed_checkpoint import (
        enable_fused_qkv,
        enable_quant_kernels,
        is_packed_checkpoint,
        load_packed_quantized_model,
        load_quant_checkpoint,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load the model configuration.
    model_args = vdm_wan_2_1_configs[MODEL_FLAVOR]
    model_args.pretrained_model_name_or_path = PRETRAINED_MODEL_PATH
    model_args.cp_size = 1
    model_args.attn_mode = ATTN_MODE

    # 2. Create the model on the CPU.
    print(f"Creating model (attn={ATTN_MODE})...")
    model = WanTransformer3DModel(model_args)

    # A packed release checkpoint contains all transformer weights and prepacked W4A4 buffers.
    # It avoids loading base safetensors or repacking FP32 weights at inference startup.
    packed_ckpt = None
    if QUANT_CHECKPOINT_PATH.endswith((".pt", ".pth")) and not _env_bool("FORCE_LEGACY_QUANT_CHECKPOINT", False):
        packed_ckpt = load_quant_checkpoint(QUANT_CHECKPOINT_PATH, convert_dcp_to_torch)
        if is_packed_checkpoint(packed_ckpt):
            if not USE_VDM_W4A4_KERNEL:
                raise ValueError("Packed W4A4 checkpoint requires USE_VDM_W4A4_KERNEL=1")
            print("Loading packed DSAQuant W4A4 checkpoint...")
            print("Replacing Linear with QuantLinear...")
            quant_module_refactor(model.blocks, ignore_list=[])
            missing, unexpected = load_packed_quantized_model(
                model,
                packed_ckpt,
                quant_linear_cls=QuantLinear,
                dtype=DTYPE,
                device=device,
                require_vdm_kernel=True,
            )
            del packed_ckpt
            gc.collect()
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            torch.cuda.empty_cache()
            gc.collect()
            enable_fused_qkv(
                model,
                quant_linear_cls=QuantLinear,
                attention_cls=WanAttention,
                use_fused_qkv=USE_FUSED_QKV_KERNEL,
            )
            model.eval()
            print("Quantized model ready for inference! (packed checkpoint)")
            return model, model_args, device
        del packed_ckpt
        packed_ckpt = None
        gc.collect()

    # 3. Load pretrained weights.
    print("Loading pretrained weights...")
    transformer_dir = f"{PRETRAINED_MODEL_PATH}/transformer"
    full_state_dict = load_full_state_dict_from_safetensors(transformer_dir)
    sd_adapter = WanStateDictAdapter(model_args)
    full_state_dict = sd_adapter.from_hf(full_state_dict)
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False, assign=True)
    del full_state_dict
    gc.collect()
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # 4. Replace linear layers with QuantLinear.
    print("Replacing Linear with QuantLinear...")
    quant_module_refactor(model.blocks, ignore_list=[])

    model = model.to(device)
    # 5. Initialize the quantizers.
    print("Initializing quantizers...")
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.wquantizer(module.weight.detach())
            module.wquantizer.build()
            module.aquantizer.build()
            module.set_quant_state(use_wq=True, use_aq=False)

    # 6. Cast to the target dtype.
    model = model.to(DTYPE)

    # 7. Load the quantized checkpoint.
    print("Loading quantized checkpoint...")
    ckpt_data = load_quant_checkpoint(QUANT_CHECKPOINT_PATH, convert_dcp_to_torch)
    if is_packed_checkpoint(ckpt_data):
        raise ValueError("Packed checkpoint should have been loaded by the fast path above")
    if "model" in ckpt_data:
        q_state_dict = ckpt_data["model"]
    else:
        q_state_dict = ckpt_data

    missing, unexpected = model.load_state_dict(q_state_dict, strict=False, assign=True)
    del q_state_dict, ckpt_data
    gc.collect()
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # 8. Move to the GPU and restore the target dtype (assign=True may move parameters to the CPU or change dtype).
    model = model.to(DTYPE).to(device)

    # 9. Enable kernel mode.
    enable_quant_kernels(
        model,
        quant_linear_cls=QuantLinear,
        
    )

    # Release unused GPU memory.
    torch.cuda.empty_cache()
    gc.collect()

    # 10. Enable the fused QKV kernel.
    enable_fused_qkv(
        model,
        quant_linear_cls=QuantLinear,
        attention_cls=WanAttention,
        use_fused_qkv=USE_FUSED_QKV_KERNEL,
    )

    model.eval()
    print("Quantized model ready for inference!")
    return model, model_args, device

def _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, max_sequence_length=226, device=None, dtype=None):
    """Compute T5 text embeddings."""
    import re
    def prompt_clean(text):
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        return text.strip()

    device = device or torch.device("cuda:0")
    dtype = dtype or torch.float16

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

    return prompt_embeds


def run_inference(model, model_args, device):
    """Run inference."""
    mode_str = "W4A4 Quantized" if USE_QUANTIZATION else "FP16"
    print("\n" + "=" * 60)
    print(f"Starting inference ({mode_str})...")
    print("=" * 60)

    # Load the VAE.
    print("Loading VAE...")
    vae = load_vae(
        f"{PRETRAINED_MODEL_PATH}/vae",
        vae_type="vdm_wan",
        torch_dtype=DTYPE,
        torch_device=device,
    )
    tokenizer = load_tokenizer(f"{PRETRAINED_MODEL_PATH}/tokenizer")

    if PRECOMPUTE_TEXT_EMBEDDING:
        # Precompute text embeddings, then delete the text encoder to save VRAM.
        print("Precomputing text embeddings...")
        text_encoder = load_text_encoder(
            f"{PRETRAINED_MODEL_PATH}/text_encoder",
            text_encoder_type="u5",
            torch_dtype=DTYPE,
            torch_device=device,
        )

        prompt_embeds = _get_t5_prompt_embeds(tokenizer, text_encoder, PROMPT, device=device, dtype=DTYPE)
        negative_prompt_embeds = _get_t5_prompt_embeds(tokenizer, text_encoder, "", device=device, dtype=DTYPE)

        # Delete the text encoder to release VRAM.
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1e9
        print(f"  Text encoder deleted. GPU memory: {mem_after:.3f} GB")

        # Create the pipeline without a text encoder.
        scheduler = FlowUniPCMultistepScheduler(shift=FLOW_SHIFT)
        pipeline = WanPipeline(
            tokenizer=tokenizer,
            text_encoder=None,  # Already deleted.
            transformer=model,
            transformer_2=model,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=BOUNDARY_RATIO,
            expand_timesteps=False,
            pipe_dtype=DTYPE,
        )

        # Run inference.
        torch.manual_seed(42)
        start_time = time.time()

        try:
            with torch.no_grad():
                result = pipeline.__call__(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE_2 if BOUNDARY_RATIO is not None else None,
                    generator=torch.manual_seed(42),
                )
            elapsed = time.time() - start_time
            print(f"\n[{mode_str}] Inference completed in {elapsed:.2f}s ({elapsed/60:.1f}min)")
            print(f"  Per-step: {elapsed/NUM_INFERENCE_STEPS:.3f}s")

            # Save the result.
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            mode_tag = "quant_vdm" if (USE_QUANTIZATION and USE_VDM_W4A4_KERNEL) else ("quant" if USE_QUANTIZATION else "fp16")
            output_path = os.path.join(OUTPUT_DIR, f"output_{mode_tag}.mp4")
            export_to_video(result.frames[0], output_path, fps=16)
            print(f"Video saved to {output_path}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[{mode_str}] Inference failed after {elapsed:.2f}s")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Original path: keep the text encoder in GPU memory.
        print("Loading text encoder (kept in GPU memory)...")
        text_encoder = load_text_encoder(
            f"{PRETRAINED_MODEL_PATH}/text_encoder",
            text_encoder_type="u5",
            torch_dtype=DTYPE,
            torch_device=device,
        )

        scheduler = FlowUniPCMultistepScheduler(shift=FLOW_SHIFT)
        pipeline = WanPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=model,
            transformer_2=model,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=BOUNDARY_RATIO,
            expand_timesteps=False,
            pipe_dtype=DTYPE,
        )

        # Run inference.
        torch.manual_seed(42)
        start_time = time.time()

        try:
            result = pipeline.__call__(
                prompt=PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                guidance_scale_2=GUIDANCE_SCALE_2 if BOUNDARY_RATIO is not None else None,
                generator=torch.manual_seed(42),
            )
            elapsed = time.time() - start_time
            print(f"\n[{mode_str}] Inference completed in {elapsed:.2f}s ({elapsed/60:.1f}min)")
            print(f"  Per-step: {elapsed/NUM_INFERENCE_STEPS:.3f}s")

            # Save the result.
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            mode_tag = "quant_vdm" if (USE_QUANTIZATION and USE_VDM_W4A4_KERNEL) else ("quant" if USE_QUANTIZATION else "fp16")
            output_path = os.path.join(OUTPUT_DIR, f"output_{mode_tag}.mp4")
            export_to_video(result.frames[0], output_path, fps=16)
            print(f"Video saved to {output_path}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[{mode_str}] Inference failed after {elapsed:.2f}s")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    mode_str = "W4A4 Quantized" if USE_QUANTIZATION else "FP16 (no quantization)"
    print("=" * 60)
    print(f"Single-GPU DSAQuant Inference - {mode_str}")
    print("=" * 60)
    print(f"Pretrained model: {PRETRAINED_MODEL_PATH}")
    if USE_QUANTIZATION:
        print(f"Quant checkpoint: {QUANT_CHECKPOINT_PATH}")
        print(f"Use VDM kernel: {USE_VDM_W4A4_KERNEL}")
        print(f"Use fused QKV: {USE_FUSED_QKV_KERNEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Dtype: {DTYPE}")
    print(f"Prompt: {PROMPT}")
    print(f"Resolution: {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames, {NUM_INFERENCE_STEPS} steps")
    print()

    if USE_QUANTIZATION:
        model, model_args, device = build_model_quantized()
    else:
        model, model_args, device = build_model_fp16()

    run_inference(model, model_args, device)


if __name__ == "__main__":
    main()
