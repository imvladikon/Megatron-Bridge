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

"""Dependency-free configuration contract for GLM-5.3-Flash conversion.

The published and Transformers layouts differ. Resolve aliases before creating
a provider, keep the complete original config for export, and do not apply the
ordinary GLM-5.2 QK-width workaround to this model family.
"""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _schedule(value: Any, *, name: str, layers: int, choices: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != layers:
        raise ValueError(f"{name} must contain exactly {layers} layer entries")
    if any(item not in choices for item in value):
        raise ValueError(f"{name} only supports {choices}, got {value!r}")
    return tuple(value)


@dataclass(frozen=True)
class KdaGeometry:
    """Equal Q/K/V geometry and the two-stage forget-gate lower bound."""

    num_heads: int
    head_dim: int
    conv_kernel_size: int
    gate_lower_bound: float


@dataclass(frozen=True)
class MlaGeometry:
    """Projection geometry; the non-RoPE and RoPE widths are authoritative."""

    num_heads: int
    nope_head_dim: int
    rope_head_dim: int
    value_head_dim: int
    q_lora_rank: int | None
    kv_lora_rank: int

    @property
    def query_head_dim(self) -> int:
        """Total Q/K width used by Transformers' projection and softmax scale."""
        return self.nope_head_dim + self.rope_head_dim


@dataclass(frozen=True)
class FlashConfig:
    """Validated geometry with lossless source configuration for later export."""

    kda: KdaGeometry
    mla: MlaGeometry
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]
    hc_streams: int
    hc_sinkhorn_iterations: int
    hc_epsilon: float
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool
    diagnostics: tuple[str, ...]
    _original: dict[str, Any] = field(repr=False, compare=False)

    @property
    def hybrid_pattern(self) -> str:
        """Physical attention/FFN pairs; HF layer i maps to 2*i and 2*i+1."""
        attention = {"linear_attention": "K", "deepseek_sparse_attention": "D"}
        mlp = {"dense": "-", "sparse": "E"}
        return "".join(attention[a] + mlp[m] for a, m in zip(self.layer_types, self.mlp_layer_types, strict=True))

    def original_hf_config(self) -> dict[str, Any]:
        """Return a fresh complete copy, including vision, tokens and quantization."""
        return deepcopy(self._original)


def parse_flash_config(config: Mapping[str, Any]) -> FlashConfig:
    """Validate a raw HF config or ``PretrainedConfig.to_dict()`` without models.

    Args:
        config: A ``glm5_next`` VLM config or ``glm5_next_text`` config mapping.

    Returns:
        Normalized Flash geometry. No input dictionaries are modified.

    Raises:
        ValueError: Required fields are absent, aliases conflict, or schedules
            and attention geometry disagree.
    """
    if config.get("model_type") not in ("glm5_next", "glm5_next_text"):
        raise ValueError("Expected glm5_next or glm5_next_text; ordinary GLM uses GLM5Bridge")
    original = deepcopy(dict(config))
    text = original.get("text_config", original)
    if not isinstance(text, Mapping) or text.get("model_type") != "glm5_next_text":
        raise ValueError("GLM-5.3-Flash requires a glm5_next_text configuration")
    nested = text.get("linear_attn_config", {})
    if not isinstance(nested, Mapping):
        raise ValueError("linear_attn_config must be a mapping")
    aliases = (
        ("num_heads", "linear_num_heads"),
        ("head_dim", "linear_head_dim"),
        ("short_conv_kernel_size", "linear_conv_kernel_dim"),
        ("gate_lower_bound", "linear_lower_bound"),
    )
    values = []
    for nested_name, flat_name in aliases:
        nv, fv = nested.get(nested_name), text.get(flat_name)
        if nv is not None and fv is not None and nv != fv:
            raise ValueError(f"Conflicting KDA fields: {nested_name}={nv!r}, {flat_name}={fv!r}")
        value = nv if nv is not None else fv
        if value is None:
            raise ValueError(f"Missing KDA field {nested_name} / {flat_name}")
        values.append(value)
    bound = _real(values[3], "gate_lower_bound")
    if bound >= 0:
        raise ValueError("gate_lower_bound must be negative")
    kda = KdaGeometry(
        _integer(values[0], "linear_num_heads"),
        _integer(values[1], "linear_head_dim"),
        _integer(values[2], "linear_conv_kernel_dim"),
        bound,
    )
    mla = MlaGeometry(
        _integer(text.get("num_attention_heads"), "num_attention_heads"),
        _integer(text.get("qk_nope_head_dim"), "qk_nope_head_dim"),
        _integer(text.get("qk_rope_head_dim"), "qk_rope_head_dim", minimum=0),
        _integer(text.get("v_head_dim"), "v_head_dim"),
        None if text.get("q_lora_rank") is None else _integer(text["q_lora_rank"], "q_lora_rank"),
        _integer(text.get("kv_lora_rank"), "kv_lora_rank"),
    )
    if _boolean(text.get("mla_use_nope"), "mla_use_nope") != (mla.rope_head_dim == 0):
        raise ValueError("mla_use_nope disagrees with qk_rope_head_dim")
    diagnostics = []
    # Transformers 5.16.1 computes QK width from these two components. The
    # published 0.1B fixture retains a stale qk_head_dim=256 while its real
    # q_b_proj is [4*64, 128]. Subtraction would wrongly invent 192 RoPE dims.
    if text.get("qk_head_dim") is not None and text["qk_head_dim"] != mla.query_head_dim:
        diagnostics.append("Ignoring redundant qk_head_dim; using qk_nope_head_dim + qk_rope_head_dim")
    layers = _integer(text.get("num_hidden_layers"), "num_hidden_layers")
    attention = _schedule(
        text.get("layer_types"),
        name="layer_types",
        layers=layers,
        choices=("linear_attention", "deepseek_sparse_attention"),
    )
    for field_name, kind in (("kda_layers", "linear_attention"), ("full_attn_layers", "deepseek_sparse_attention")):
        if field_name in nested:
            expected = [i for i, item in enumerate(attention) if item == kind]
            actual = nested[field_name]
            if (
                not isinstance(actual, (list, tuple))
                or any(type(i) is not int for i in actual)
                or list(actual) != expected
            ):
                raise ValueError(f"linear_attn_config.{field_name} disagrees with layer_types")
    mlp_types = text.get("mlp_layer_types")
    if mlp_types is None:
        dense = _integer(text.get("first_k_dense_replace"), "first_k_dense_replace", minimum=0)
        if dense > layers:
            raise ValueError("first_k_dense_replace exceeds num_hidden_layers")
        mlp_types = ["dense"] * dense + ["sparse"] * (layers - dense)
    mlp = _schedule(mlp_types, name="mlp_layer_types", layers=layers, choices=("dense", "sparse"))
    indexers = _schedule(text.get("indexer_types"), name="indexer_types", layers=layers, choices=("full", "shared"))
    have_indexer_source = False
    for kind, indexer in zip(attention, indexers, strict=True):
        if kind == "deepseek_sparse_attention":
            if indexer == "shared" and not have_indexer_source:
                raise ValueError("Shared DSA indexer has no preceding full DSA source")
            have_indexer_source |= indexer == "full"
    if not _boolean(text.get("mhc"), "mhc"):
        raise ValueError("The Flash bridge requires mHC")
    epsilon = _real(text.get("hc_eps"), "hc_eps")
    if epsilon <= 0:
        raise ValueError("hc_eps must be positive")
    return FlashConfig(
        kda=kda,
        mla=mla,
        layer_types=attention,
        mlp_layer_types=mlp,
        indexer_types=indexers,
        hc_streams=_integer(text.get("hc_mult"), "hc_mult"),
        hc_sinkhorn_iterations=_integer(text.get("hc_sinkhorn_iters"), "hc_sinkhorn_iters"),
        hc_epsilon=epsilon,
        index_topk=_integer(text.get("index_topk"), "index_topk"),
        index_kpool=_integer(text.get("index_kpool"), "index_kpool"),
        index_kpool_compress=_boolean(text.get("index_kpool_compress"), "index_kpool_compress"),
        index_kpool_always_select_tail=_boolean(
            text.get("index_kpool_always_select_tail"), "index_kpool_always_select_tail"
        ),
        diagnostics=tuple(diagnostics),
        _original=original,
    )
