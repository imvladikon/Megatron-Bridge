"""Config-only tests with real published metadata and no model imports.

Load the entire dependency-free module by path so these checks can also run
without importing the Bridge/TE package on a development laptop.
"""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).parent / "fixtures"
SPEC = importlib.util.spec_from_file_location(
    "glm53_config_under_test", ROOT / "src/megatron/bridge/models/glm5next/config_adapter.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
parse_flash_config = MODULE.parse_flash_config
pytestmark = pytest.mark.unit


def fixture(name="tiny"):
    return json.loads((FIXTURES / f"flash_{name}.json").read_text())


def test_published_tiny_geometry_matches_actual_checkpoint_header():
    config = parse_flash_config(fixture())
    assert config.mla.nope_head_dim == 64
    assert config.mla.rope_head_dim == 0
    assert config.mla.query_head_dim == 64
    assert config.diagnostics  # Redundant qk_head_dim=256 is stale in this fixture.
    shapes = json.loads((FIXTURES / "flash_tiny_mla_shapes.json").read_text())
    prefix = "model.language_model.layers.3.self_attn."
    mla = config.mla
    assert shapes[prefix + "q_b_proj.weight"] == [mla.num_heads * mla.query_head_dim, mla.q_lora_rank]
    assert shapes[prefix + "kv_a_proj_with_mqa.weight"] == [mla.kv_lora_rank + mla.rope_head_dim, 256]
    assert shapes[prefix + "kv_b_proj.weight"] == [
        mla.num_heads * (mla.nope_head_dim + mla.value_head_dim),
        mla.kv_lora_rank,
    ]
    assert shapes[prefix + "o_proj.weight"] == [256, mla.num_heads * mla.value_head_dim]
    assert config.hybrid_pattern == "K-K-K-DEKE"


def test_published_full_geometry_and_physical_layers():
    source = fixture("full")
    config = parse_flash_config(source)
    assert config.kda.num_heads == 64
    assert config.kda.head_dim == 128
    assert config.kda.conv_kernel_size == 4
    assert config.kda.gate_lower_bound == -5
    assert config.mla.query_head_dim == 256
    assert config.mla.rope_head_dim == 0
    assert len(config.layer_types) == 45
    assert len(config.hybrid_pattern) == 90
    assert config.hybrid_pattern.count("K") == 34
    assert config.hybrid_pattern.count("D") == 11
    assert config.hybrid_pattern.count("E") == 42
    assert (config.hc_streams, config.hc_sinkhorn_iterations, config.hc_epsilon) == (4, 20, 1e-6)
    assert (config.index_topk, config.index_kpool) == (2048, 4)
    assert config.index_kpool_compress and config.index_kpool_always_select_tail
    assert not config.diagnostics
    assert config.original_hf_config() == source


@pytest.mark.parametrize("layout", ["nested", "flat", "both"])
def test_kda_schema_aliases(layout):
    config = fixture()
    text = config["text_config"]
    if layout == "flat":
        text.pop("linear_attn_config")
    elif layout == "nested":
        for key in ("linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim", "linear_lower_bound"):
            text.pop(key)
    parsed = parse_flash_config(config)
    assert (parsed.kda.num_heads, parsed.kda.head_dim, parsed.kda.conv_kernel_size) == (4, 64, 4)


def test_conflicting_aliases_are_not_silently_selected():
    config = fixture()
    config["text_config"]["linear_head_dim"] = 128
    with pytest.raises(ValueError, match="Conflicting KDA"):
        parse_flash_config(config)


def test_original_config_and_export_copy_are_independent():
    config = fixture("full")
    original = deepcopy(config)
    parsed = parse_flash_config(config)
    assert config == original
    config["text_config"]["num_hidden_layers"] = 90
    export = parsed.original_hf_config()
    export["text_config"]["num_hidden_layers"] = 1
    assert parsed.original_hf_config() == original
    assert "vision_config" in original and "quantization_config" in original


def test_flat_text_config_is_supported():
    text = fixture()["text_config"]
    parsed = parse_flash_config(text)
    assert len(parsed.layer_types) == 5
    assert parsed.original_hf_config() == text


@pytest.mark.parametrize(
    "name,value",
    [
        ("layer_types", ["linear_attention"]),
        ("mlp_layer_types", ["unknown"] * 5),
        ("indexer_types", ["shared"] * 5),
        ("hc_mult", True),
        ("hc_eps", 0),
        ("hc_eps", float("nan")),
        ("qk_rope_head_dim", 64),
        ("index_kpool", 0),
        ("index_kpool_always_select_tail", "false"),
        ("num_hidden_layers", 0),
    ],
)
def test_invalid_contracts_fail_before_provider_construction(name, value):
    config = fixture()
    config["text_config"][name] = value
    with pytest.raises(ValueError):
        parse_flash_config(config)


def test_redundant_kda_schedule_must_agree():
    config = fixture("full")
    config["text_config"]["linear_attn_config"]["kda_layers"].pop()
    with pytest.raises(ValueError, match="disagrees with layer_types"):
        parse_flash_config(config)


def test_mlp_schedule_fallback_requires_explicit_dense_count():
    config = fixture()
    config["text_config"].pop("mlp_layer_types")
    assert parse_flash_config(config).mlp_layer_types == ("dense", "dense", "dense", "sparse", "sparse")
    config["text_config"]["first_k_dense_replace"] = 6
    with pytest.raises(ValueError, match="exceeds"):
        parse_flash_config(config)


def test_ordinary_glm_is_never_routed_to_flash_config():
    with pytest.raises(ValueError, match="ordinary GLM"):
        parse_flash_config({"model_type": "glm_moe_dsa"})
