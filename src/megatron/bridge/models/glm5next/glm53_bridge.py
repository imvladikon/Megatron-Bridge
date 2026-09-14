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

"""Native Flash bridge for the released per-expert checkpoint schema."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.glm5next.config_adapter import parse_flash_config
from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider
from megatron.bridge.models.glm5next.mapping_registry import build_flash_mapping_registry
from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel


if TYPE_CHECKING:
    from megatron.bridge.models.hf_pretrained.base import PreTrainedBase


@MegatronModelBridge.register_bridge(
    source="Glm5NextForConditionalGeneration",
    target=GLM53FlashModel,
    provider=GLM53FlashModelProvider,
    model_type="glm5_next",
)
class GLM53FlashBridge(MegatronModelBridge):
    """Convert released Flash weights to native KDA/DSA/mHC physical layers.

    Each HF block maps to an attention layer and an FFN layer. Expert weights
    remain separate on export; conversion never buffers a layer's entire expert
    bank. Native Transformers packed-expert checkpoints and quantized import
    require separate loading support and are rejected rather than cast blindly.
    """

    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained: "PreTrainedBase") -> GLM53FlashModelProvider:
        """Preserve the complete HF configuration without allocating a model."""
        return GLM53FlashModelProvider.from_hf_config(hf_pretrained.config.to_dict())

    def _split_qkv_linear_out_weight(
        self,
        megatron_model: torch.nn.Module | list[torch.nn.Module],
        linear_out_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split gathered KDA LoRA B for both adapter and merged HF export.

        Only KDA uses fused QKV in Flash; DSA uses separate MLA projections.
        Generic adapter materialization concatenates TP ranks, each containing
        [Q_rank | K_rank | V_rank]. This is not the head-interleaved QKV layout
        of ordinary attention, and KDA head geometry differs from MLA geometry.
        """
        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        config = model.config
        tp = config.tensor_model_parallel_size
        qk_width = config.linear_num_key_heads * config.linear_key_head_dim
        v_width = config.linear_num_value_heads * config.linear_value_head_dim
        widths = (qk_width, qk_width, v_width)
        if tp < 1 or any(width < 1 or width % tp for width in widths):
            raise ValueError(f"KDA projection widths {widths} must be positive and divisible by TP={tp}")
        if linear_out_weight.ndim != 2 or linear_out_weight.shape[0] != sum(widths):
            raise ValueError(f"KDA LoRA B must have {sum(widths)} rows, got {tuple(linear_out_weight.shape)}")
        rank = linear_out_weight.shape[1]
        per_rank = linear_out_weight.reshape(tp, sum(widths) // tp, rank)
        result = {}
        offset = 0
        for name, width in zip(("q_proj", "k_proj", "v_proj"), widths, strict=True):
            local_width = width // tp
            result[name] = per_rank[:, offset : offset + local_width, :].reshape(width, rank)
            offset += local_width
        return result

    @classmethod
    def megatron_to_hf_config(cls, provider: GLM53FlashModelProvider) -> dict[str, Any]:
        """Export source geometry and the actually constructed base/MTP depth."""
        if not isinstance(provider, GLM53FlashModelProvider):
            raise TypeError("Flash export requires a GLM53FlashModelProvider")
        original = parse_flash_config(provider.source_hf_config).original_hf_config()
        expected = GLM53FlashModelProvider.from_hf_config(original)
        # Runtime topology/sequence-length overrides are legitimate. Shape or
        # attention-math overrides must instead be made in the source HF config.
        for name in (
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "vocab_size",
            "hybrid_layer_pattern",
            "num_moe_experts",
            "moe_ffn_hidden_size",
            "moe_shared_expert_intermediate_size",
            "moe_layer_freq",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "kda_lower_bound",
            "moe_router_topk",
            "moe_router_topk_scaling_factor",
            "moe_router_num_groups",
            "moe_router_group_topk",
            "dsa_indexer_head_dim",
            "dsa_indexer_n_heads",
            "dsa_indexer_topk",
            "dsa_indexer_kpool",
            "dsa_indexer_kpool_always_select_tail",
            "dsa_indexer_rope_interleaved",
            "vision_config",
            "layernorm_epsilon",
            "mhc_num_residual_streams",
            "mhc_sinkhorn_iterations",
            "activation_func_clamp_value",
            "share_embeddings_and_output_weights",
        ):
            if getattr(provider, name) != getattr(expected, name):
                raise ValueError(f"Flash export configuration disagrees with source geometry: {name}")
        if provider.mtp_num_layers:
            raise NotImplementedError("Flash MTP checkpoint conversion is not implemented")
        original["text_config"]["num_nextn_predict_layers"] = 0
        # This path exports ordinary floating-point weights. Keeping an FP8
        # source quantization declaration would make HF interpret them wrongly.
        for config in (original, original["text_config"]):
            config.pop("quantization_config", None)
            config["dtype"] = str(provider.params_dtype).removeprefix("torch.")
            if "torch_dtype" in config:
                config["torch_dtype"] = config["dtype"]
        return original

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load one mapping; dequantize block-scaled FP8 weights, reject other quantized dtypes.

        The released GLM-5.3-Flash checkpoint stores expert/MLP/MLA projections as FP8 E4M3 with a
        companion ``<name>_scale_inv`` block-scale tensor (128x128). The generic loader ships the
        companion in ``hf_state_dict`` (see ``model_bridge`` companion handling), so the weight is
        dequantized to bf16 here, before any TP cast, exactly like ``GLM5Bridge._maybe_dequantize_fp8``.
        Weights listed in ``modules_to_not_convert`` arrive as bf16 and pass through unchanged.
        """

        def load(name: str) -> torch.Tensor:
            weight = hf_state_dict[name]
            if quantization_utils.is_fp8_tensor(weight):
                scale_inv = hf_state_dict.get(name + "_scale_inv")
                if scale_inv is None:
                    raise NotImplementedError(f"Flash FP8 weight without block scale: {name}")
                return quantization_utils.maybe_dequantize_fp8_blockwise(weight, scale_inv)
            if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise NotImplementedError(f"Flash quantized checkpoint import needs scale-aware loading: {name}")
            return weight

        return load(hf_param) if isinstance(hf_param, str) else {role: load(name) for role, name in hf_param.items()}

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map released text/vision weights to explicit native physical layers."""
        return build_flash_mapping_registry(self.hf_config.to_dict())
