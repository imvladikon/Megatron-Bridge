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

"""GLM-5.3-Flash language-stack provider.

This configures the native KDA/DSA/mHC decoder. A VLM wrapper must own vision
insertion and packing; this provider alone does not convert the vision tower.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Self

from megatron.core.transformer import ModuleSpec
from torch.nn import functional as F

from megatron.bridge.models.glm5next.config_adapter import parse_flash_config
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider, get_default_hybrid_stack_spec
from megatron.bridge.models.transformer_config import MLATransformerConfig


if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel


def glm53_hybrid_stack_spec(config: "GLM53FlashTextModelProvider") -> ModuleSpec:
    """Install the checkpoint's Q/KV RMSNorms in the native DSA layer spec."""
    spec = get_default_hybrid_stack_spec(config)
    dsa_spec = spec.submodules.dsa_layer
    layer = dsa_spec.submodules
    # Copy the dataclass path being changed, keeping factory objects intact.
    # Optional-dependency factories may contain closures that cannot be pickled
    # by deepcopy. No shared spec or factory is mutated here.
    attention = replace(
        layer.self_attention.submodules, q_layernorm=layer.input_layernorm, kv_layernorm=layer.input_layernorm
    )
    layer = replace(layer, self_attention=replace(layer.self_attention, submodules=attention))
    return replace(spec, submodules=replace(spec.submodules, dsa_layer=replace(dsa_spec, submodules=layer)))


def _expert_count(text: Mapping[str, Any]) -> int:
    values = [text[key] for key in ("n_routed_experts", "num_local_experts") if key in text]
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Flash requires a positive n_routed_experts / num_local_experts")
    if len(set(values)) != 1:
        raise ValueError("Conflicting n_routed_experts and num_local_experts")
    return values[0]


@dataclass
class GLM53FlashTextModelProvider(HybridModelProvider, MLATransformerConfig):
    """Native Flash language decoder with HF geometry and deferred validation.

    ``from_hf_config`` does not instantiate a model or allocate its weights.
    Checkpoint MTP metadata is retained. Until the native MTP architecture is
    mapped, an explicit ``mtp_num_layers=0`` override is required for a base
    decoder run on a configuration declaring MTP.
    """

    source_hf_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hf_config(cls, hf_config: Mapping[str, Any]) -> Self:
        """Read a VLM or text config without constructing a Transformers model.

        Architecture fields come from the checkpoint. Runtime parallelism and
        sequence length may be overridden before ``finalize`` as in Bridge's
        other providers. The complete source config remains available for the
        conversion layer, including tokenizer, vision and quantization fields.
        """
        parsed = parse_flash_config(hf_config)
        original = parsed.original_hf_config()
        text = original.get("text_config", original)
        if parsed.mla.rope_head_dim != 0:
            raise NotImplementedError("Flash provider currently implements the published NoPE geometry")
        if parsed.hc_epsilon != 1e-6:
            raise NotImplementedError("Native mHC uses hc_eps=1e-6; refusing a different checkpoint stabilizer")
        if not parsed.index_kpool_compress:
            raise NotImplementedError("Flash KPool requires learned key compression")
        if any(
            indexer != "full"
            for kind, indexer in zip(parsed.layer_types, parsed.indexer_types, strict=True)
            if kind == "deepseek_sparse_attention"
        ):
            raise NotImplementedError("Flash shared indexers require an explicit physical-layer source schedule")
        if text.get("hidden_act") != "silu" or text.get("attention_bias") is not False:
            raise ValueError("Flash requires bias-free attention and SwiGLU")
        if text.get("norm_topk_prob") is not True or text["num_experts_per_tok"] <= 1:
            raise NotImplementedError("Native sigmoid routing matches Flash with normalized top-k > 1")
        if text.get("scoring_func", "sigmoid") != "sigmoid":
            raise ValueError("Flash uses sigmoid routing")
        # HF group choice sums the top two scores. MCore sums topk/group_topk.
        # This difference is irrelevant if every group is selected.
        num_groups, group_topk = text["n_group"], text["topk_group"]
        if any(type(value) is not int or value < 1 for value in (num_groups, group_topk)) or group_topk > num_groups:
            raise ValueError("Flash routing requires integer 1 <= topk_group <= n_group")
        if num_groups != group_topk and text["num_experts_per_tok"] // group_topk != 2:
            raise NotImplementedError("Flash and MCore group-routing score reductions differ for this geometry")

        mla, kda = parsed.mla, parsed.kda
        shared_width = text["moe_intermediate_size"] * text["n_shared_experts"]
        return cls(
            source_hf_config=original,
            num_layers=len(parsed.hybrid_pattern),
            hidden_size=text["hidden_size"],
            ffn_hidden_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            seq_length=text["max_position_embeddings"],
            share_embeddings_and_output_weights=original.get(
                "tie_word_embeddings", text.get("tie_word_embeddings", False)
            ),
            should_pad_vocab=False,
            hybrid_layer_pattern=parsed.hybrid_pattern,
            hybrid_stack_spec=glm53_hybrid_stack_spec,
            normalization="RMSNorm",
            layernorm_epsilon=text["rms_norm_eps"],
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=text["attention_dropout"],
            init_method_std=text["initializer_range"],
            activation_func_clamp_value=text["swiglu_limit"],
            use_te_activation_func=False,
            num_moe_experts=_expert_count(text),
            moe_ffn_hidden_size=text["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=shared_width or None,
            moe_shared_expert_gate=False,
            moe_shared_expert_overlap=False,
            moe_layer_freq=[flag for kind in parsed.mlp_layer_types for flag in (0, int(kind == "sparse"))],
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_pre_softmax=True,
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_router_topk=text["num_experts_per_tok"],
            moe_router_topk_scaling_factor=text["routed_scaling_factor"],
            moe_router_num_groups=num_groups,
            moe_router_group_topk=group_topk,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=text["router_aux_loss_coef"],
            multi_latent_attention=True,
            num_attention_heads=mla.num_heads,
            kv_channels=mla.value_head_dim,
            q_lora_rank=mla.q_lora_rank,
            kv_lora_rank=mla.kv_lora_rank,
            qk_head_dim=mla.nope_head_dim,
            qk_pos_emb_head_dim=mla.rope_head_dim,
            v_head_dim=mla.value_head_dim,
            qk_layernorm=True,
            position_embedding_type="none",
            apply_rope_fusion=False,
            cp_comm_type="allgather",
            experimental_attention_variant="dsa",
            dsa_indexer_head_dim=text["index_head_dim"],
            dsa_indexer_n_heads=text["index_n_heads"],
            dsa_indexer_topk=parsed.index_topk,
            dsa_indexer_kpool=parsed.index_kpool,
            dsa_indexer_kpool_always_select_tail=parsed.index_kpool_always_select_tail,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_rotate_activation=False,
            dsa_indexer_rope_interleaved=text["indexer_rope_interleave"],
            dsa_indexer_k_norm_epsilon=1e-6,
            dsa_indexer_topk_freq=1,
            dsa_indexer_skip_topk_offset=0,
            linear_num_key_heads=kda.num_heads,
            linear_num_value_heads=kda.num_heads,
            linear_key_head_dim=kda.head_dim,
            linear_value_head_dim=kda.head_dim,
            linear_conv_kernel_dim=kda.conv_kernel_size,
            kda_two_stage_gates=True,
            kda_safe_gate=True,
            kda_lower_bound=kda.gate_lower_bound,
            gdn_pre_gated_delta_rule_fusion=False,
            gdn_conv_pad_alignment=None,
            enable_mhc_connections=True,
            mhc_num_residual_streams=parsed.hc_streams,
            mhc_sinkhorn_iterations=parsed.hc_sinkhorn_iterations,
            mhc_learned_output_contract=False,
            mhc_rms_epsilon_inside_sqrt=True,
            mhc_mapping_proj_fp32=False,
            use_fused_mhc=False,
            mtp_num_layers=text.get("num_nextn_predict_layers", 0),
        )

    def finalize(self) -> None:
        """Validate supported native execution before any model allocation."""
        if self.mtp_num_layers:
            raise NotImplementedError(
                "Flash native MTP mapping is pending; explicitly set mtp_num_layers=0 for base-decoder training"
            )
        super().finalize()


@dataclass
class GLM53FlashModelProvider(GLM53FlashTextModelProvider):
    """Flash VLM provider; compose the native language stack with HF vision."""

    vision_config: dict[str, Any] | None = None
    scatter_embedding_sequence_parallel: bool = False
    image_token_id: int | None = None
    video_start_token_id: int | None = None
    video_end_token_id: int | None = None
    pad_token_id: int = 0
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @classmethod
    def from_hf_config(cls, hf_config: Mapping[str, Any]) -> Self:
        """Retain the checkpoint vision architecture and media token markers."""
        if hf_config.get("model_type") != "glm5_next" or not isinstance(hf_config.get("vision_config"), Mapping):
            raise ValueError("Flash VLM provider requires glm5_next with vision_config")
        provider = super().from_hf_config(hf_config)
        provider.vision_config = deepcopy(dict(hf_config["vision_config"]))
        if provider.vision_config.get("out_hidden_size") != provider.hidden_size:
            raise ValueError("Flash vision output width must match language hidden_size")
        for name in ("image_token_id", "video_start_token_id", "video_end_token_id"):
            value = hf_config.get(name)
            if type(value) is not int or not 0 <= value < provider.vocab_size:
                raise ValueError(f"Flash {name} must be a valid vocabulary ID")
            setattr(provider, name, value)
        pad_token = hf_config["text_config"].get("pad_token_id")
        if pad_token is not None and (type(pad_token) is not int or not 0 <= pad_token < provider.vocab_size):
            raise ValueError("Flash pad_token_id must be a valid vocabulary ID or None")
        provider.pad_token_id = pad_token if pad_token is not None else 0
        return provider

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> "GLM53FlashModel":
        """Build the wrapper using Bridge's explicit pipeline process group."""
        from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

        from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp)
        model = GLM53FlashModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_vision_model=self.freeze_vision_model,
            freeze_vision_projection=self.freeze_vision_projection,
        )
        return model

    def provide_language_model(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> "HybridModel":
        """Build only the native decoder, retaining this provider as its config."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
