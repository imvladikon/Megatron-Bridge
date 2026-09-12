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

"""GLM Flash mappings for KDA's rank-local Q/K/V concatenation."""

from collections.abc import Mapping
from typing import Sequence

import torch
from torch.distributed.tensor import DTensor

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping, _module_uses_fsdp
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name


class KDAProjectionMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Separate HF tensors ↔ rank-local ``[Q_rank | K_rank | V_rank]``.

    Applies to the input projection and depthwise convolution. Section sizes
    come from the validated KDA configuration, not from MLA head dimensions.
    Communication proceeds one section at a time: import does not first create
    a full fused QKV tensor or a second rank-major copy of that tensor.
    """

    def __init__(self, megatron_param: str, hf_params: Mapping[str, str], section_sizes: tuple[int, int, int]):
        if set(hf_params) != {"q", "k", "v"}:
            raise ValueError("KDA mapping requires exactly q, k and v source names")
        if len(section_sizes) != 3 or any(type(size) is not int or size < 1 for size in section_sizes):
            raise ValueError("KDA section sizes must be three positive integers")
        super().__init__(megatron_param, {key: hf_params[key] for key in ("q", "k", "v")})
        self.section_sizes = tuple(section_sizes)

    def resolve(self, captures):
        """Preserve section geometry when resolving layer wildcards."""
        megatron_param, hf_params = self._resolve_names(captures)
        return type(self)(megatron_param, hf_params, self.section_sizes)

    def local_hf_param_specs(self, global_param_name=None):
        """Use explicit conversion until local transport is qualified for KDA."""
        return ()

    def _local_section_sizes(self):
        if any(size % self.tp_size for size in self.section_sizes):
            raise ValueError(f"Every KDA section must be divisible by TP size {self.tp_size}: {self.section_sizes}")
        return tuple(size // self.tp_size for size in self.section_sizes)

    def hf_to_megatron(self, hf_weights, megatron_module):
        """Scatter Q, K and V independently into a single local destination."""
        local_sizes = self._local_section_sizes()
        _, target = get_module_and_param_from_name(megatron_module, self.megatron_param)
        if isinstance(target, DTensor):
            raise NotImplementedError("KDA DTensor import requires a qualified section-aware placement")
        if target.shape[0] != sum(local_sizes):
            raise ValueError("KDA target row count disagrees with the configured local sections")
        error = None
        if self.tp_rank == 0:
            if not isinstance(hf_weights, Mapping) or set(hf_weights) != set(self.hf_param):
                error = "KDA source weights must contain exactly q, k and v"
            else:
                for key, rows in zip(self.hf_param, self.section_sizes, strict=True):
                    source = hf_weights[key]
                    if not isinstance(source, torch.Tensor) or tuple(source.shape) != (rows, *target.shape[1:]):
                        error = f"KDA {key} shape must be {(rows, *target.shape[1:])}"
                        break
                    if source.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                        error = "KDA source weights must be dequantized before conversion"
                        break
        # A malformed source must fail on all TP ranks before scatter, so peers
        # are not stranded in a collective after rank zero raises.
        if self.tp_size > 1:
            errors = [error]
            torch.distributed.broadcast_object_list(
                errors, src=torch.distributed.get_global_rank(self.tp_group, 0), group=self.tp_group
            )
            error = errors[0]
        if error is not None:
            raise ValueError(error)

        output = torch.empty_like(target)
        offset = 0
        for key, rows in zip(self.hf_param, local_sizes, strict=True):
            splits = None
            if self.tp_rank == 0:
                source = hf_weights[key].to(dtype=target.dtype)
                splits = source.split(rows, dim=0)
            if self.tp_size == 1:
                section = splits[0]
            else:
                section = self.scatter_to_tp_ranks(splits, (rows, *target.shape[1:]), target.dtype, target.device)
            output.narrow(0, offset, rows).copy_(section)
            offset += rows
            del section, splits
            if self.tp_rank == 0:
                del source
        return output

    def megatron_to_hf(self, megatron_weights, megatron_module):
        """Broadcast across PP, then gather each Q/K/V section across TP."""
        uses_fsdp = self.broadcast_obj_from_pp_rank(
            _module_uses_fsdp(megatron_module) if megatron_module is not None else None,
            cache_key="kda_fsdp_layout",
        )
        if uses_fsdp:
            raise NotImplementedError("KDA FSDP export requires a qualified section-aware placement")
        weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if weights is None:
            return {}
        weights = self.maybe_dequantize(weights)
        local_sizes = self._local_section_sizes()
        if weights.shape[0] != sum(local_sizes):
            raise ValueError("KDA export row count disagrees with the configured local sections")
        result = {}
        for name, section in zip(self.hf_param.values(), weights.split(local_sizes, dim=0), strict=True):
            if self.tp_size == 1:
                result[name] = section
            else:
                result[name] = torch.cat(self.gather_from_tp_ranks(section.contiguous()), dim=0)
        return result


class HCAlphaMapping(MegatronParamMapping[torch.Tensor]):
    """Map HF's three mHC scales to replicated Megatron scalar parameters.

    Only the pre-scale task emits the joined HF tensor. It reads the current
    sibling parameters from their owning module, so repeated exports neither
    retain stale values nor need a process-global accumulation cache.
    """

    _names = ("alpha_pre", "alpha_post", "alpha_res")

    def __init__(self, megatron_param: str, hf_param: str, index: int) -> None:
        if type(index) is not int or index not in range(3):
            raise ValueError("mHC scale index must be 0, 1 or 2")
        if megatron_param.rsplit(".", 1)[-1] != self._names[index]:
            raise ValueError("mHC scale index disagrees with the target parameter name")
        super().__init__(megatron_param, hf_param)
        self.index = index

    def resolve(self, captures: Sequence[str]) -> "HCAlphaMapping":
        """Preserve the scalar index when resolving layer wildcards."""
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, self.index)

    def local_hf_param_specs(self, global_param_name: str | None = None) -> tuple:
        """Require conversion until joined local scale transport is qualified."""
        return ()

    def hf_to_megatron(self, hf_weights: torch.Tensor | None, megatron_module: torch.nn.Module) -> torch.Tensor:
        """Broadcast one scale without replacing or modifying the Parameter."""
        _, target = get_module_and_param_from_name(megatron_module, self.megatron_param)
        if isinstance(target, DTensor) or _module_uses_fsdp(megatron_module):
            raise NotImplementedError("mHC scale DTensor/FSDP conversion is not qualified")
        if tuple(target.shape) != (1,):
            raise ValueError("mHC target scale must have shape [1]")
        error = None
        if self.tp_rank == 0:
            if not isinstance(hf_weights, torch.Tensor) or tuple(hf_weights.shape) != (3,):
                error = "HF mHC scale must have shape [3]"
            elif hf_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                error = "HF mHC scale must be dequantized before conversion"
        if self.tp_size > 1:
            errors = [error]
            torch.distributed.broadcast_object_list(
                errors, src=torch.distributed.get_global_rank(self.tp_group, 0), group=self.tp_group
            )
            error = errors[0]
        if error is not None:
            raise ValueError(error)
        value = torch.empty_like(target)
        if self.tp_rank == 0:
            value.copy_(hf_weights[self.index : self.index + 1])
        if self.tp_size > 1:
            torch.distributed.broadcast(
                value, src=torch.distributed.get_global_rank(self.tp_group, 0), group=self.tp_group
            )
        return value

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: torch.nn.Module | None
    ) -> dict[str, torch.Tensor]:
        """Join current pre/post/res scales on the owner, then broadcast over PP."""
        if self.index != 0:
            return {}
        values = None
        error = None
        if megatron_module is not None:
            if _module_uses_fsdp(megatron_module):
                error = "mHC scale DTensor/FSDP conversion is not qualified"
            else:
                owner, _ = get_module_and_param_from_name(megatron_module, self.megatron_param)
                siblings = [megatron_weights, owner.alpha_post, owner.alpha_res]
                if any(isinstance(v, DTensor) or v is None or tuple(v.shape) != (1,) for v in siblings):
                    error = "mHC export requires three current scalar [1] parameters"
                else:
                    values = torch.cat([v.detach() for v in siblings])
        # Include a sentinel for success: None means a non-owning PP stage.
        status = self.broadcast_obj_from_pp_rank((error,) if megatron_module is not None else None, cache_key=None)
        if status[0] is not None:
            raise ValueError(status[0])
        values = self.broadcast_from_pp_rank(values, cache_key=self.hf_param)
        return {self.hf_param: values}
