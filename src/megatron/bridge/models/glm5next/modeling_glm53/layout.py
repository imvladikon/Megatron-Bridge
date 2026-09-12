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

"""Token-only layout planning before allocating Flash language embeddings."""

from copy import copy
from dataclasses import dataclass

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions,
)


@dataclass
class FlashInputLayout:
    """Local CP token rows and their indices in the caller's flattened input.

    SP scattering is deliberately separate. ``source_indices`` uses -1 for
    alignment padding; ``source_valid`` describes the full caller tensor for
    counting image/video placeholders before selecting the local features.
    """

    input_ids: torch.Tensor
    source_indices: torch.Tensor
    source_valid: torch.Tensor
    padding_mask: torch.Tensor
    packed_seq_params: PackedSeqParams | None
    source_cp_sharded: bool = False

    def select(self, source: torch.Tensor) -> torch.Tensor:
        """Select caller B×S×... values into CP-local B×S×..., zeroing padding."""
        if source.shape[:2] != self.source_valid.shape:
            raise ValueError("Flash inputs must share the source input_ids batch/sequence shape")
        values = source.flatten(0, 1).index_select(0, self.source_indices.clamp_min(0))
        invalid = self.source_indices < 0
        values = values.masked_fill(invalid.reshape(-1, *([1] * (values.ndim - 1))), 0)
        return values.reshape(*self.input_ids.shape, *source.shape[2:])


def _bounds(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"Flash {name} requires a 1D cumulative-length tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Flash {name} must have integer dtype")
    if bool(value[0] != 0 or torch.any(value[1:] < value[:-1])):
        raise ValueError(f"Flash {name} must start at zero and be nondecreasing")
    return value


def prepare_flash_input_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    packed_seq_params: PackedSeqParams | None,
    *,
    cp_size: int,
    cp_rank: int,
    pad_token_id: int,
) -> FlashInputLayout:
    """Honor producer padding and select zigzag CP rows before embedding.

    A 2D keep mask identifies VERL's unprocessed BSHD inputs, including B=1.
    Without that mask, packed input must be a full or already CP-local THD row.
    Physical offsets are authoritative; alignment/bucket padding is never
    recomputed. No B×S×hidden or sequence-squared tensors are created here.
    """
    if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Flash input_ids must be a nonempty integer B×S tensor")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("Invalid Flash context-parallel rank or size")
    keep = None
    if attention_mask is not None and attention_mask.ndim == 2:
        if attention_mask.shape != input_ids.shape or bool(torch.any((attention_mask != 0) & (attention_mask != 1))):
            raise ValueError("Flash 2D attention_mask must be a B×S zero/one keep mask")
        keep = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    if packed_seq_params is None:
        if cp_size != 1 or (keep is not None and not bool(keep.all())):
            raise ValueError("Flash padded or CP inputs require packed_seq_params; use THD training")
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        return FlashInputLayout(
            input_ids, torch.arange(input_ids.numel(), device=input_ids.device), valid, ~valid, None
        )
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("Flash packed inputs require qkv_format='thd'")
    if attention_mask is not None and keep is None:
        raise ValueError("Flash THD layout accepts a 2D keep mask or attention_mask=None")
    local_cp_size = packed_seq_params.local_cp_size
    if local_cp_size not in (None, cp_size):
        raise NotImplementedError("Flash wrapper requires packed and model CP group sizes to agree")
    logical = _bounds(packed_seq_params.cu_seqlens_q, "cu_seqlens_q").to(input_ids.device)
    physical = packed_seq_params.cu_seqlens_q_padded
    physical = logical if physical is None else _bounds(physical, "cu_seqlens_q_padded").to(input_ids.device)
    kv = _bounds(packed_seq_params.cu_seqlens_kv, "cu_seqlens_kv").to(input_ids.device)
    kv_physical = packed_seq_params.cu_seqlens_kv_padded
    kv_physical = kv if kv_physical is None else _bounds(kv_physical, "cu_seqlens_kv_padded").to(input_ids.device)
    if physical.shape != logical.shape or not torch.equal(logical, kv) or not torch.equal(physical, kv_physical):
        raise ValueError("Flash self-attention requires matching query/key logical and physical boundaries")
    physical_lengths = physical.diff()
    logical_lengths = logical.diff()
    total_tokens = int(physical[-1])
    if total_tokens < 1 or bool(torch.any(logical_lengths > physical_lengths)):
        raise ValueError("Flash logical lengths must fit in a nonempty physical THD stream")
    if packed_seq_params.total_tokens not in (None, total_tokens):
        raise ValueError("Flash total_tokens must match the final physical boundary")
    if cp_size > 1 and bool(torch.any(physical_lengths % (2 * cp_size))):
        raise ValueError("Flash zigzag CP requires physical document lengths divisible by 2*CP")

    normalized = copy(packed_seq_params)
    if keep is not None:
        if logical.numel() != input_ids.size(0) + 1:
            raise ValueError("Flash BSHD rows must match the number of packed documents")
        lengths = keep.sum(-1, dtype=logical.dtype)
        # Legacy VERL stores physical lengths in both fields. Recover true
        # lengths from its keep mask, preserving all physical row coordinates.
        if not (torch.equal(logical_lengths, lengths) or torch.equal(logical_lengths, physical_lengths)):
            raise ValueError("Flash keep-mask lengths disagree with packed metadata")
        if bool(torch.any(lengths > physical_lengths)):
            raise ValueError("Flash physical padding is shorter than the real document")
        logical = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=logical.dtype)])
        logical_lengths = lengths
        source_valid = keep
    else:
        if input_ids.size(0) != 1:
            raise ValueError("Flash THD without a keep mask must have batch size 1")
        source_valid = None
    normalized.cu_seqlens_q = normalized.cu_seqlens_kv = logical
    normalized.cu_seqlens_q_padded = normalized.cu_seqlens_kv_padded = physical

    positions = build_packed_allgather_cp_local_positions(
        physical,
        cp_size,
        cp_rank,
        input_ids.device,
        output_size=total_tokens // cp_size,
        cu_seqlens_cover_output=True,
    )
    segments = torch.bucketize(positions, physical[1:], right=True)
    offsets = positions - physical.index_select(0, segments)
    valid = offsets < logical_lengths.index_select(0, segments)
    source_cp_sharded = False
    if keep is not None:
        compact_source = keep.flatten().nonzero().flatten()
        compact_rows = logical.index_select(0, segments) + offsets
        source_indices = torch.full_like(positions, -1)
        source_indices[valid] = compact_source.index_select(0, compact_rows[valid].long())
    elif input_ids.size(1) == total_tokens:
        source_indices = positions.masked_fill(~valid, -1)
        full_positions = torch.arange(total_tokens, device=input_ids.device)
        full_segments = torch.bucketize(full_positions, physical[1:], right=True)
        full_offsets = full_positions - physical.index_select(0, full_segments)
        source_valid = (full_offsets < logical_lengths.index_select(0, full_segments)).unsqueeze(0)
    elif cp_size > 1 and input_ids.size(1) == total_tokens // cp_size:
        # Caller already owns exactly the current CP shard; never partition it
        # again. Image/video insertion requires the unsharded placeholder map.
        source_cp_sharded = True
        source_indices = torch.arange(positions.numel(), device=input_ids.device).masked_fill(~valid, -1)
        source_valid = valid.unsqueeze(0)
    else:
        raise ValueError("Flash THD input length is neither the full physical stream nor its CP-local shard")
    local_ids = input_ids.flatten().index_select(0, source_indices.clamp_min(0)).masked_fill(~valid, pad_token_id)
    return FlashInputLayout(
        local_ids.unsqueeze(0), source_indices, source_valid, (~valid).unsqueeze(0), normalized, source_cp_sharded
    )
