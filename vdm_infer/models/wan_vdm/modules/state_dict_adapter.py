# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import torch
from typing import Any
from einops import rearrange


from ..modules.args import WanTransformer3DModelArgs


class WanStateDictAdapter:
    def __init__(self, model_args: WanTransformer3DModelArgs):
        self.model_args = model_args
        # Map from HF format to internal format
        self.from_hf_map = {
            # Patch embedding
            "patch_embedding.bias": "patch_embedding_mlp.bias",
            "patch_embedding.weight": "patch_embedding_mlp.weight",

            # Condition embedder
            "condition_embedder.text_embedder.linear_1.bias": "condition_embedder.text_embedder.linear_1.bias",
            "condition_embedder.text_embedder.linear_1.weight": "condition_embedder.text_embedder.linear_1.weight",
            "condition_embedder.text_embedder.linear_2.bias": "condition_embedder.text_embedder.linear_2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "condition_embedder.text_embedder.linear_2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "condition_embedder.time_embedder.linear_1.bias",
            "condition_embedder.time_embedder.linear_1.weight": "condition_embedder.time_embedder.linear_1.weight",
            "condition_embedder.time_embedder.linear_2.bias": "condition_embedder.time_embedder.linear_2.bias",
            "condition_embedder.time_embedder.linear_2.weight":  "condition_embedder.time_embedder.linear_2.weight",
            "condition_embedder.time_proj.bias": "condition_embedder.time_proj.bias",
            "condition_embedder.time_proj.weight": "condition_embedder.time_proj.weight",
            
            # Transformer blocks - norms
            "blocks.{}.norm2.bias": "blocks.{}.norm2.bias",
            "blocks.{}.norm2.weight": "blocks.{}.norm2.weight",
            
            # Transformer blocks - scale shift tables
            "blocks.{}.scale_shift_table": "blocks.{}.scale_shift_table",
            
            # Output layers
            "proj_out.weight": "proj_out.weight",
            "proj_out.bias": "proj_out.bias",
            "scale_shift_table": "scale_shift_table",
        }
        attn_dict = {
            # Transformer blocks - self attention
            "blocks.{}.attn1.norm_k.weight": "blocks.{}.attn1.norm_k.weight",
            "blocks.{}.attn1.norm_q.weight": "blocks.{}.attn1.norm_q.weight",
            "blocks.{}.attn1.to_q.bias": "blocks.{}.attn1.to_q.bias",
            "blocks.{}.attn1.to_q.weight": "blocks.{}.attn1.to_q.weight",
            "blocks.{}.attn1.to_k.bias": "blocks.{}.attn1.to_k.bias",
            "blocks.{}.attn1.to_k.weight": "blocks.{}.attn1.to_k.weight",
            "blocks.{}.attn1.to_v.bias": "blocks.{}.attn1.to_v.bias",
            "blocks.{}.attn1.to_v.weight": "blocks.{}.attn1.to_v.weight",
            "blocks.{}.attn1.to_out.0.bias": "blocks.{}.attn1.to_out.0.bias",
            "blocks.{}.attn1.to_out.0.weight": "blocks.{}.attn1.to_out.0.weight",

            # Transformer blocks - cross attention
            "blocks.{}.attn2.norm_k.weight": "blocks.{}.attn2.norm_k.weight",
            "blocks.{}.attn2.norm_q.weight": "blocks.{}.attn2.norm_q.weight",
            "blocks.{}.attn2.to_q.bias": "blocks.{}.attn2.to_q.bias",
            "blocks.{}.attn2.to_q.weight": "blocks.{}.attn2.to_q.weight",
            "blocks.{}.attn2.to_k.bias": "blocks.{}.attn2.to_k.bias",
            "blocks.{}.attn2.to_k.weight": "blocks.{}.attn2.to_k.weight",
            "blocks.{}.attn2.to_v.bias": "blocks.{}.attn2.to_v.bias",
            "blocks.{}.attn2.to_v.weight": "blocks.{}.attn2.to_v.weight",
            "blocks.{}.attn2.to_out.0.bias": "blocks.{}.attn2.to_out.0.bias",
            "blocks.{}.attn2.to_out.0.weight": "blocks.{}.attn2.to_out.0.weight",

            # Transformer blocks - feed forward
            "blocks.{}.ffn.net.0.proj.bias": "blocks.{}.ffn.net.0.proj.bias",
            "blocks.{}.ffn.net.0.proj.weight": "blocks.{}.ffn.net.0.proj.weight",
            "blocks.{}.ffn.net.2.bias": "blocks.{}.ffn.net.2.bias",
            "blocks.{}.ffn.net.2.weight": "blocks.{}.ffn.net.2.weight",
        }
        self.from_hf_map.update(attn_dict)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert internal state dict to HF format"""
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "blocks" in key:
                # Handle layer-specific keys
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                if abstract_key in to_hf_map:
                    new_key = to_hf_map[abstract_key]
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                # Handle non-layer keys
                if key in to_hf_map:
                    # note we use patch embedding mlp, but need load pretrianed model fisrt
                    if key == "patch_embedding_mlp.weight":
                        org_weight = rearrange(
                            value.clone(), 
                            'c_out (c_in p_1 p_2 p_3) -> c_out c_in p_1 p_2 p_3', 
                            c_in=self.model_args.in_channels, 
                            p_1=self.model_args.patch_size[0],
                            p_2=self.model_args.patch_size[1],
                            p_3=self.model_args.patch_size[2],
                        )
                        # note we inflate patch embedding, but need load pretrianed model fisrt
                        if self.model_args.in_channels_org > 0:
                            org_weight = org_weight[:, :self.model_args.in_channels_org, ...]
                        hf_state_dict[to_hf_map[key]] = org_weight
                    else:
                        hf_state_dict[to_hf_map[key]] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to internal format"""
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "blocks" in key:
                # Handle layer-specific keys
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                if abstract_key in self.from_hf_map:
                    new_key = self.from_hf_map[abstract_key]
                    new_key = new_key.format(layer_num)
                    state_dict[new_key] = value
            else:
                # Handle non-layer keys
                if key in self.from_hf_map:
                    # note inlfate & convert conv3d into mlp (if inflate, we inflate!)
                    if key == "patch_embedding.weight":
                        hf_weight = hf_state_dict["patch_embedding.weight"].clone()
                        if self.model_args.in_channels_org > 0:
                            repeat_times = int(self.model_args.in_channels // self.model_args.in_channels_org) - 1
                            hf_weight = torch.cat(
                                [hf_weight] + 
                                [torch.zeros_like(hf_weight)] * repeat_times,
                                dim=1,
                            )
                        state_dict["patch_embedding_mlp.weight"] = rearrange(hf_weight.clone(), 'c_out c_in p_1 p_2 p_3 -> c_out (c_in p_1 p_2 p_3)')
                    elif key == "patch_embedding.bias":
                        state_dict["patch_embedding_mlp.bias"] = hf_state_dict["patch_embedding.bias"].clone()
                    else:
                        state_dict[self.from_hf_map[key]] = value
        return state_dict