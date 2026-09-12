"""KDA adapter export must preserve projection and tensor-parallel row order."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.glm5next.glm53_bridge import GLM53FlashBridge
from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider


@pytest.mark.parametrize("fixture", ["flash_tiny.json", "flash_full.json"])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_kda_adapter_export_and_merge_keep_projection_rows(fixture, tp):
    config = json.loads((Path(__file__).parent / "fixtures" / fixture).read_text())
    provider = GLM53FlashModelProvider.from_hf_config(config)
    provider.tensor_model_parallel_size = tp
    provider.mtp_num_layers = 0
    provider.finalize()
    width = provider.linear_num_key_heads * provider.linear_key_head_dim
    rank, input_width = 4, 7
    # Independent full HF factors, distinct across both rows and projections.
    expected = {
        key: torch.arange(width * rank, dtype=torch.float32).reshape(width, rank) / (width * rank) + i
        for i, key in enumerate(("q_proj", "k_proj", "v_proj"), start=1)
    }
    local_rows = width // tp
    # ColumnParallelMapping materializes B by concatenating rank-local tensors.
    # KDA stores [Q_rank | K_rank | V_rank] within each of those tensors.
    gathered_b = torch.cat(
        [expected[key][worker * local_rows : (worker + 1) * local_rows] for worker in range(tp) for key in expected]
    )
    hf_names = [f"model.language_model.layers.0.self_attn.{key}.weight" for key in expected]
    model = SimpleNamespace(config=provider)
    bridge = GLM53FlashBridge()
    actual = bridge._get_fused_adapter_linear_out_slices([model], hf_names, gathered_b)
    assert set(actual) == set(hf_names)
    for name, reference in zip(hf_names, expected.values(), strict=True):
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0)

    # Check the actual merged-export path too; no neural model is constructed.
    a = torch.arange(rank * input_width, dtype=torch.float32).reshape(rank, input_width) / 32
    base = {name: torch.full((width, input_width), 0.25) for name in hf_names}
    weight = SimpleNamespace(
        global_base_prefix="language_model.decoder.layers.0.self_attention.in_proj",
        adapter_key=None,
        alpha=8,
        dim=rank,
        linear_in_weight=SimpleNamespace(weight=a),
        linear_out_weight=SimpleNamespace(weight=gathered_b),
    )
    merged = bridge._merge_lora_adapter_weights([model], base, [weight])
    for name, reference in zip(hf_names, expected.values(), strict=True):
        torch.testing.assert_close(merged[name], 0.25 + 2 * (reference @ a))
