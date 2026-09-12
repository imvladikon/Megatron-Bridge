"""Coverage of the pinned released checkpoint header, without weight downloads."""

import hashlib
import json
from pathlib import Path

from megatron.bridge.models.glm5next.mapping_registry import build_flash_mapping_registry
from megatron.bridge.models.glm5next.param_mapping import HCAlphaMapping, KDAProjectionMapping


FIXTURES = Path(__file__).parent / "fixtures"


def test_released_tiny_header_is_fully_covered_without_grouped_expert_buffers():
    registry = build_flash_mapping_registry(json.loads((FIXTURES / "flash_tiny.json").read_text()))
    header_bytes = (FIXTURES / "flash_tiny_header.json").read_bytes()
    assert (
        hashlib.sha256(header_bytes).hexdigest() == "7d1c1995e13dd1ec083907ea5effe887169f4afead43b48bd778ffb21b72ed21"
    )
    header = json.loads(header_bytes)
    weights = {name: spec for name, spec in header.items() if name != "__metadata__"}
    assert len(weights) == 223
    assert sum(name.startswith("model.visual.") for name in weights) == 39
    assert all(not getattr(mapping, "is_grouped_export", False) for mapping in registry.mappings)
    assert not any("mtp" in mapping.megatron_param for mapping in registry.mappings)
    resolved = {}
    for name in weights:
        mapping = registry.hf_to_megatron_lookup(name)
        assert mapping is not None, name
        resolved[mapping.megatron_param] = mapping
        expected_names = [mapping.hf_param] if isinstance(mapping.hf_param, str) else list(mapping.hf_param.values())
        assert all(key in weights for key in expected_names)
    for index in range(5):
        for kind, physical in (("attn", 2 * index), ("ffn", 2 * index + 1)):
            prefix = f"language_model.decoder.layers.{physical}"
            for i, part in enumerate(("pre", "post", "res")):
                mapping = registry.megatron_to_hf_lookup(f"{prefix}.hyper_connection.alpha_{part}")
                assert isinstance(mapping, HCAlphaMapping) and mapping.index == i
                assert mapping.hf_param == f"model.language_model.layers.{index}.hc_{kind}_scale"
        # Both fused TE and separate FFN norms need concrete physical-index aliases.
        prefix = f"language_model.decoder.layers.{2 * index + 1}.inner_layer"
        separate = registry.megatron_to_hf_lookup(f"{prefix}.pre_mlp_layernorm.weight")
        assert separate.hf_param == f"model.language_model.layers.{index}.post_attention_layernorm.weight"
        if index < 3:
            assert (
                registry.megatron_to_hf_lookup(f"{prefix}.mlp.linear_fc1.layer_norm_weight").hf_param
                == separate.hf_param
            )
    for index in (0, 1, 2, 4):
        prefix = f"language_model.decoder.layers.{2 * index}.inner_layer.self_attention"
        for name in ("in_proj", "conv1d"):
            mapping = registry.megatron_to_hf_lookup(f"{prefix}.{name}.weight")
            assert isinstance(mapping, KDAProjectionMapping)
            assert mapping.section_sizes == (256, 256, 256)
    for index in (3, 4):
        for expert in range(8):
            prefix = f"language_model.decoder.layers.{2 * index + 1}.inner_layer.mlp.experts"
            grouped = registry.megatron_to_hf_lookup(f"{prefix}.linear_fc1.weight{expert}")
            sequential = registry.megatron_to_hf_lookup(f"{prefix}.local_experts.{expert}.linear_fc1.weight")
            assert (
                grouped.hf_param
                == sequential.hf_param
                == {
                    "gate": f"model.language_model.layers.{index}.mlp.experts.{expert}.gate_proj.weight",
                    "up": f"model.language_model.layers.{index}.mlp.experts.{expert}.up_proj.weight",
                }
            )
