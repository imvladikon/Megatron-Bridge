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

"""Replicated HF vision tower and native Megatron Flash language decoder."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.glm5next.modeling_glm53.layout import FlashInputLayout, prepare_flash_input_layout
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


if TYPE_CHECKING:
    from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider


class GLM53FlashModel(MegatronModule):
    """Insert image/video features into CP-local embeddings before TP scattering.

    The caller's token IDs retain enough information to distinguish image and
    video placeholders. Only token/boolean/index arrays are global; language
    embeddings are allocated after packing and CP selection. Already CP-local
    THD text inputs are accepted without a second split.
    """

    def __init__(
        self, config: "GLM53FlashModelProvider", *, pre_process: bool, post_process: bool, vp_stage: int | None = None
    ) -> None:
        super().__init__(config=config)
        if config.scatter_embedding_sequence_parallel:
            raise ValueError("Flash VLM must insert features before embedding sequence-parallel scatter")
        self.pre_process, self.post_process = pre_process, post_process
        self.vp_stage = vp_stage
        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        self.pg_collection = self.language_model.pg_collection
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        if pre_process:
            from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

            self.visual = Glm5NextVisionModel._from_config(
                Glm5NextVisionConfig(**config.vision_config), dtype=config.params_dtype
            )
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

    def set_input_tensor(self, input_tensor: torch.Tensor | list[torch.Tensor]) -> None:
        """Pass pipeline input through to the native decoder."""
        self.language_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        """Expose the native tied-embedding contract to Megatron schedules."""
        return self.language_model.shared_embedding_or_output_weight()

    def _vision_features(self, pixels: torch.Tensor, grid: torch.Tensor | None, *, video: bool) -> torch.Tensor:
        if grid is None or grid.ndim != 2 or grid.size(1) != 3 or grid.dtype not in (torch.int32, torch.int64):
            raise ValueError("Flash vision requires integer [items, 3] grid_thw")
        if bool(torch.any(grid <= 0)):
            raise ValueError("Flash vision grid dimensions must be positive")
        if video:
            # HF treats each video frame as an image with temporal grid size 1.
            hw = torch.repeat_interleave(grid[:, 1:], grid[:, 0], dim=0)
            grid = torch.cat([grid.new_ones((hw.size(0), 1)), hw], dim=1)
        return self.visual(pixels.to(dtype=self.visual.dtype), grid_thw=grid, return_dict=True).pooler_output

    @staticmethod
    def _local_features(
        features: torch.Tensor, source_mask: torch.Tensor, layout: FlashInputLayout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or int(source_mask.sum()) != features.size(0):
            raise ValueError("Flash vision feature count does not match the source placeholder count")
        source_mask = source_mask.flatten()
        safe_indices = layout.source_indices.clamp_min(0)
        local_mask = source_mask.index_select(0, safe_indices) & (layout.source_indices >= 0)
        local_rows = local_mask.nonzero().flatten()
        slots = source_mask.to(torch.long).cumsum(0) - 1
        feature_indices = slots.index_select(0, safe_indices).index_select(0, local_rows)
        return local_rows, features.index_select(0, feature_indices)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        output_processor: Callable[..., Any] | None = None,
        output_processor_context: Any = None,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        compute_mtp_loss: bool = True,
    ) -> Any:
        """Preserve native labels/loss/output hooks while preparing input rows.

        ``inputs_embeds`` uses the same B×S layout as input IDs. Labels and loss
        masks are already model-local, as supplied by VERL; they are never
        repacked here. ``padding_mask`` is CP-local or already SP-local router
        padding (True means invalid), distinct from the source 2D keep mask.
        Padded/CP training requires THD metadata. Unpadded BSHD with CP=1 is
        supported. Flash uses NoPE, so position IDs do not affect embeddings.
        """
        has_vision = pixel_values is not None or pixel_values_videos is not None
        if packed_seq_params is not None and packed_seq_params.cp_group not in (None, self.pg_collection.cp):
            raise NotImplementedError("Flash wrapper does not yet support changing its CP group per microbatch")
        if self.pre_process and input_ids is None:
            raise ValueError("Flash input_ids are required for layout and placeholder validation")
        decoder_input = None
        lm_input_ids = input_ids
        lm_attention_mask = attention_mask
        if input_ids is not None:
            layout = prepare_flash_input_layout(
                input_ids,
                attention_mask,
                packed_seq_params,
                cp_size=self.pg_collection.cp.size(),
                cp_rank=self.pg_collection.cp.rank(),
                pad_token_id=self.config.pad_token_id,
            )
            lm_input_ids, packed_seq_params = layout.input_ids, layout.packed_seq_params
            if packed_seq_params is not None or (attention_mask is not None and attention_mask.ndim == 2):
                lm_attention_mask = None
            if has_vision and layout.source_cp_sharded:
                raise ValueError("Flash image/video inputs require unsharded source IDs to locate all placeholders")
            cp_padding = layout.padding_mask
            sp_padding = None
            if padding_mask is not None:
                if padding_mask.dtype != torch.bool:
                    raise ValueError("Flash padding_mask must be boolean (True means padding)")
                if padding_mask.shape == cp_padding.shape:
                    cp_padding = cp_padding | padding_mask
                else:
                    sp_padding = padding_mask
            if self.config.sequence_parallel:
                if lm_input_ids.size(1) % self.pg_collection.tp.size():
                    raise ValueError("Flash CP-local token rows must be divisible by TP for sequence parallelism")
                cp_padding = (
                    scatter_to_sequence_parallel_region(
                        cp_padding.transpose(0, 1).contiguous(), group=self.pg_collection.tp
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
            if sp_padding is not None:
                if sp_padding.shape != cp_padding.shape:
                    raise ValueError("Flash router padding_mask has neither CP-local nor SP-local shape")
                cp_padding = cp_padding | sp_padding
            padding_mask = cp_padding if bool(cp_padding.any()) else None

            if self.pre_process:
                if inputs_embeds is None:
                    embeddings = self.language_model.embedding(input_ids=lm_input_ids, position_ids=None).transpose(
                        0, 1
                    )
                else:
                    embeddings = layout.select(inputs_embeds)
                if embeddings.size(-1) != self.config.hidden_size:
                    raise ValueError("Flash language embeddings disagree with hidden_size")
                embeddings = embeddings.masked_fill(layout.padding_mask.unsqueeze(-1), 0)
                if has_vision:
                    # HF uses image_token_id for both media types; video spans
                    # are identified by start/end markers, not video_token_id.
                    in_video = (input_ids == self.config.video_start_token_id).cumsum(-1) > (
                        input_ids == self.config.video_end_token_id
                    ).cumsum(-1)
                    placeholders = (input_ids == self.config.image_token_id) & layout.source_valid
                    replacements = []
                    if pixel_values is not None:
                        features = self._vision_features(pixel_values, image_grid_thw, video=False)
                        replacements.append(self._local_features(features, placeholders & ~in_video, layout))
                    if pixel_values_videos is not None:
                        features = self._vision_features(pixel_values_videos, video_grid_thw, video=True)
                        replacements.append(self._local_features(features, placeholders & in_video, layout))
                    rows = torch.cat([item[0] for item in replacements])
                    values = torch.cat([item[1] for item in replacements]).to(embeddings)
                    embeddings = embeddings.flatten(0, 1).index_copy(0, rows, values).view_as(embeddings)
                decoder_input = embeddings.transpose(0, 1).contiguous()
                if self.config.sequence_parallel:
                    decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.pg_collection.tp)
        return self.language_model(
            input_ids=lm_input_ids,
            position_ids=None,
            attention_mask=lm_attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            compute_mtp_loss=compute_mtp_loss,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
        )

    def freeze(
        self, *, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ) -> None:
        """Freeze requested parameters; vision-model freeze includes its projection."""
        if freeze_language_model:
            self.language_model.requires_grad_(False)
        if hasattr(self, "visual"):
            if freeze_vision_model:
                self.visual.requires_grad_(False)
            elif freeze_vision_projection:
                self.visual.downsample.requires_grad_(False)
                self.visual.merger.requires_grad_(False)
