# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Released Flash checkpoint mappings, independent of model construction."""

from collections.abc import Mapping
from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.glm5next.config_adapter import parse_flash_config
from megatron.bridge.models.glm5next.param_mapping import HCAlphaMapping, KDAProjectionMapping


def build_flash_mapping_registry(hf_config: Mapping[str, Any]) -> MegatronMappingRegistry:
    """Map every released text/vision parameter, with explicit HF block indices."""
    raw = hf_config
    parsed = parse_flash_config(raw)
    text = raw["text_config"]
    mappings = [
        AutoMapping("language_model.embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"),
        ReplicatedMapping("language_model.decoder.final_norm.weight", "model.language_model.norm.weight"),
        AutoMapping("language_model.output_layer.weight", "lm_head.weight"),
    ]
    section_size = parsed.kda.num_heads * parsed.kda.head_dim
    for index, (attention, mlp) in enumerate(zip(parsed.layer_types, parsed.mlp_layer_types, strict=True)):
        hf = f"model.language_model.layers.{index}"
        attn = f"language_model.decoder.layers.{2 * index}"
        ffn = f"language_model.decoder.layers.{2 * index + 1}"
        for layer, kind in ((attn, "attn"), (ffn, "ffn")):
            hc = f"{layer}.hyper_connection"
            mappings.extend(
                [
                    ReplicatedMapping(f"{hc}.mapping_proj.weight", f"{hf}.hc_{kind}_fn"),
                    ReplicatedMapping(f"{hc}.bias", f"{hf}.hc_{kind}_base"),
                    *[
                        HCAlphaMapping(f"{hc}.alpha_{part}", f"{hf}.hc_{kind}_scale", i)
                        for i, part in enumerate(("pre", "post", "res"))
                    ],
                ]
            )
        sa, source = f"{attn}.inner_layer.self_attention", f"{hf}.self_attn"
        mappings.append(
            ReplicatedMapping(f"{attn}.inner_layer.input_layernorm.weight", f"{hf}.input_layernorm.weight")
        )
        if attention == "linear_attention":
            for target, suffix in (("in_proj", "proj"), ("conv1d", "conv1d")):
                mappings.append(
                    KDAProjectionMapping(
                        f"{sa}.{target}.weight",
                        {part: f"{source}.{part}_{suffix}.weight" for part in ("q", "k", "v")},
                        (section_size,) * 3,
                    )
                )
            for target, source_suffix, mapping in (
                ("beta_proj.weight", "b_proj.weight", ColumnParallelMapping),
                ("f_a_proj.weight", "f_a_proj.weight", ReplicatedMapping),
                ("f_b_proj.weight", "f_b_proj.weight", ColumnParallelMapping),
                ("g_a_proj.weight", "g_a_proj.weight", ReplicatedMapping),
                ("g_b_proj.weight", "g_b_proj.weight", ColumnParallelMapping),
                ("A_log", "A_log", ColumnParallelMapping),
                ("dt_bias", "dt_bias", ColumnParallelMapping),
                ("out_norm.weight", "o_norm.weight", ReplicatedMapping),
                ("out_proj.weight", "o_proj.weight", RowParallelMapping),
            ):
                mappings.append(mapping(f"{sa}.{target}", f"{source}.{source_suffix}"))
        else:
            if parsed.mla.q_lora_rank is None:
                mappings.append(ColumnParallelMapping(f"{sa}.linear_q_proj.weight", f"{source}.q_proj.weight"))
            else:
                mappings.extend(
                    [
                        ReplicatedMapping(f"{sa}.linear_q_down_proj.weight", f"{source}.q_a_proj.weight"),
                        ColumnParallelMapping(f"{sa}.linear_q_up_proj.weight", f"{source}.q_b_proj.weight"),
                        ReplicatedMapping(f"{sa}.q_layernorm.weight", f"{source}.q_a_layernorm.weight"),
                        ReplicatedMapping(
                            f"{sa}.linear_q_up_proj.layer_norm_weight", f"{source}.q_a_layernorm.weight"
                        ),
                    ]
                )
            mappings.extend(
                [
                    ReplicatedMapping(f"{sa}.linear_kv_down_proj.weight", f"{source}.kv_a_proj_with_mqa.weight"),
                    ColumnParallelMapping(f"{sa}.linear_kv_up_proj.weight", f"{source}.kv_b_proj.weight"),
                    ReplicatedMapping(f"{sa}.kv_layernorm.weight", f"{source}.kv_a_layernorm.weight"),
                    ReplicatedMapping(f"{sa}.linear_kv_up_proj.layer_norm_weight", f"{source}.kv_a_layernorm.weight"),
                    RowParallelMapping(f"{sa}.linear_proj.weight", f"{source}.o_proj.weight"),
                ]
            )
            for target, source_suffix in (
                ("linear_wq_b.weight", "wq_b.weight"),
                ("linear_wk.weight", "wk.weight"),
                ("k_norm.weight", "k_norm.weight"),
                ("k_norm.bias", "k_norm.bias"),
                ("linear_weights_proj.weight", "weights_proj.weight"),
                ("index_kpool_compress_ape", "index_kpool_compress_ape"),
                ("index_kpool_compress_gate", "index_kpool_compress_gate"),
            ):
                mappings.append(
                    ReplicatedMapping(f"{sa}.core_attention.indexer.{target}", f"{source}.indexer.{source_suffix}")
                )
        inner = f"{ffn}.inner_layer"
        mappings.append(
            ReplicatedMapping(f"{inner}.pre_mlp_layernorm.weight", f"{hf}.post_attention_layernorm.weight")
        )
        if mlp == "dense":
            # Explicit physical indices do not receive registry wildcard
            # layernorm aliases. Support both fused TE and separate norms.
            mappings.append(
                ReplicatedMapping(f"{inner}.mlp.linear_fc1.layer_norm_weight", f"{hf}.post_attention_layernorm.weight")
            )
            mappings.extend(_mlp_mappings(f"{inner}.mlp", f"{hf}.mlp"))
        else:
            mappings.extend(
                [
                    ReplicatedMapping(f"{inner}.mlp.router.weight", f"{hf}.mlp.gate.weight"),
                    ReplicatedMapping(f"{inner}.mlp.router.expert_bias", f"{hf}.mlp.gate.e_score_correction_bias"),
                ]
            )
            if text.get("n_shared_experts", 0):
                mappings.extend(_mlp_mappings(f"{inner}.mlp.shared_experts", f"{hf}.mlp.shared_experts"))
            for expert_prefix, weight_suffix in (("experts", "*"), ("experts.local_experts.*", "")):
                mappings.extend(
                    _mlp_mappings(f"{inner}.mlp.{expert_prefix}", f"{hf}.mlp.experts.*", weight_suffix=weight_suffix)
                )
    mappings.append(ReplicatedMapping("visual.**", "model.visual.**"))
    return MegatronMappingRegistry(*mappings)


def _mlp_mappings(target: str, source: str, *, weight_suffix: str = "") -> list[MegatronParamMapping]:
    return [
        GatedMLPMapping(
            f"{target}.linear_fc1.weight{weight_suffix}",
            gate=f"{source}.gate_proj.weight",
            up=f"{source}.up_proj.weight",
        ),
        RowParallelMapping(f"{target}.linear_fc2.weight{weight_suffix}", f"{source}.down_proj.weight"),
    ]
