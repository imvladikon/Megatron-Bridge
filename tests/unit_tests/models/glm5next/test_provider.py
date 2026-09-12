"""Native Flash provider and small FFN contracts; run on the remote host only."""

import importlib
import json
from copy import deepcopy
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import AbsorbedMLASelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from torch.nn import functional as F
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashTextModelProvider, glm53_hybrid_stack_spec
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider, get_default_hybrid_stack_spec


FIXTURES = Path(__file__).parent / "fixtures"


def _source(name="tiny"):
    return json.loads((FIXTURES / f"flash_{name}.json").read_text())


@pytest.mark.parametrize("name", ["tiny", "full"])
@pytest.mark.parametrize("hf_normalized", [False, True])
def test_real_provider_finalizes_published_geometry(name, hf_normalized):
    source = _source(name)
    if hf_normalized:
        source = Glm5NextConfig(**source).to_dict()
    before = deepcopy(source)
    provider = GLM53FlashTextModelProvider.from_hf_config(source)
    # Both source JSONs declare MTP=1; this test explicitly requests the base
    # decoder. Construction must not silently clear the checkpoint metadata.
    assert provider.mtp_num_layers == 1
    provider.mtp_num_layers = 0
    provider.seq_length = 128
    provider.finalize()
    assert source == before == provider.source_hf_config
    assert provider.num_layers == (10 if name == "tiny" else 90)
    assert provider.hybrid_layer_pattern.count("D") == (1 if name == "tiny" else 11)
    assert provider.hybrid_layer_pattern.count("K") == (4 if name == "tiny" else 34)
    assert provider.hybrid_layer_pattern.count("E") == (2 if name == "tiny" else 42)
    assert len(provider.moe_layer_freq) == provider.num_layers
    assert [i for i, enabled in enumerate(provider.moe_layer_freq) if enabled] == [
        i for i, kind in enumerate(provider.hybrid_layer_pattern) if kind == "E"
    ]
    mla = create_layer_config(provider, "D")
    kda = create_layer_config(provider, "K")
    assert mla.qk_pos_emb_head_dim == 0
    assert mla.qk_head_dim == (64 if name == "tiny" else 256)
    assert mla.num_attention_heads * mla.qk_head_dim == (256 if name == "tiny" else 16384)
    assert kda.linear_num_value_heads * kda.linear_value_head_dim == (256 if name == "tiny" else 8192)
    assert provider.mhc_rms_epsilon_inside_sqrt and not provider.mhc_mapping_proj_fp32
    assert not provider.mhc_learned_output_contract
    assert provider.params_dtype == torch.bfloat16
    assert not provider.dsa_indexer_loss_coeff
    # The PR used phantom attributes for these. Every supplied architecture
    # option must reach a declared MCore/Bridge dataclass field.
    assert "norm_topk_prob" not in vars(provider)
    assert "dsa_indexer_kpool_fp8" not in vars(provider)
    assert "mhc_keep_mappings_in_fp32" not in vars(provider)
    assert not torch.cuda.is_initialized()


def test_native_spec_has_real_mla_norms_without_mutating_shared_spec():
    provider = GLM53FlashTextModelProvider.from_hf_config(_source())
    shared = get_default_hybrid_stack_spec(provider)
    original_norm = shared.submodules.dsa_layer.submodules.self_attention.submodules.q_layernorm
    spec = glm53_hybrid_stack_spec(provider)
    layer = spec.submodules.dsa_layer.submodules
    assert layer.self_attention.module is AbsorbedMLASelfAttention
    assert layer.self_attention.submodules.core_attention.module is DSAttention
    assert layer.self_attention.submodules.q_layernorm is layer.input_layernorm
    assert layer.self_attention.submodules.kv_layernorm is layer.input_layernorm
    assert layer.input_layernorm is not IdentityOp
    assert shared.submodules.dsa_layer.submodules.self_attention.submodules.q_layernorm is original_norm


def test_mtp_requires_an_explicit_base_decoder_override():
    provider = GLM53FlashTextModelProvider.from_hf_config(_source())
    with pytest.raises(NotImplementedError, match="MTP mapping"):
        provider.finalize()
    assert provider.mtp_num_layers == 1


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("hc_eps", 1e-5, "hc_eps"),
        ("index_kpool_compress", False, "compression"),
        ("norm_topk_prob", False, "normalized"),
        ("num_experts_per_tok", 1, "top-k"),
        ("num_local_experts", 99, "Conflicting"),
        ("topk_group", 0, "topk_group"),
        ("n_group", 2, "group-routing"),
    ],
)
def test_unimplemented_semantics_are_rejected_before_building(key, value, match):
    source = _source()
    source["text_config"][key] = value
    with pytest.raises((ValueError, NotImplementedError), match=match):
        GLM53FlashTextModelProvider.from_hf_config(source)


def test_checkpoint_copy_is_independent():
    source = _source()
    provider = GLM53FlashTextModelProvider.from_hf_config(source)
    source["text_config"]["hidden_size"] = 999
    assert provider.source_hf_config["text_config"]["hidden_size"] == provider.hidden_size == 256


def test_embedding_scatter_reaches_hybrid_constructor(monkeypatch):
    # Constructor-boundary check only: the future VLM wrapper must insert
    # features before SP scattering, while ordinary Hybrid keeps its default.
    assert next(f for f in fields(HybridModelProvider) if f.name == "scatter_embedding_sequence_parallel").default
    provider = GLM53FlashTextModelProvider.from_hf_config(_source())
    provider.scatter_embedding_sequence_parallel = False
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return observed

    monkeypatch.setattr(
        importlib.import_module("megatron.bridge.models.hybrid.hybrid_provider"), "MCoreHybridModel", capture
    )
    result = provider.provide(pre_process=True, post_process=True)
    assert result["scatter_embedding_sequence_parallel"] is False
    assert result["config"] is provider


class _Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, *, config, bias=False, **_kwargs):
        super().__init__(in_features, out_features, bias=bias, dtype=config.params_dtype)

    def forward(self, inputs):
        return F.linear(inputs, self.weight), None


def test_provider_drives_real_dense_mlp_with_hf_clamp_and_gradients(tmp_path):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    dist = torch.distributed
    own_world = not dist.is_initialized()
    if own_world:
        dist.init_process_group(
            "gloo",
            init_method=(tmp_path / "mlp-gloo").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=30),
        )
    group = dist.group.WORLD if own_world else dist.new_group([dist.get_rank()], backend="gloo")
    try:
        torch.manual_seed(129)
        source = _source()
        provider = GLM53FlashTextModelProvider.from_hf_config(source)
        provider.params_dtype = torch.float64
        provider.bf16 = False
        provider.mtp_num_layers = 0
        provider.bias_activation_fusion = False
        provider.finalize()
        reference = Glm5NextTextMLP(Glm5NextConfig(**source).text_config).double()
        actual = MLP(provider, MLPSubmodules(linear_fc1=_Linear, linear_fc2=_Linear), tp_group=group)
        with torch.no_grad():
            actual.linear_fc1.weight.copy_(torch.cat([reference.gate_proj.weight, reference.up_proj.weight]))
            actual.linear_fc2.weight.copy_(reference.down_proj.weight)
        x = (80 * torch.randn(7, 1, provider.hidden_size, dtype=torch.float64)).requires_grad_()
        expected = reference(x)
        output, bias = actual(x)
        assert bias is None
        torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)
        cotangent = torch.randn_like(output)
        actual_grads = torch.autograd.grad((output * cotangent).sum(), (x, *actual.parameters()))
        ref_grads = torch.autograd.grad((expected * cotangent).sum(), (x, *reference.parameters()))
        for got, wanted in zip(actual_grads, (ref_grads[0], torch.cat(ref_grads[1:3]), ref_grads[3]), strict=True):
            torch.testing.assert_close(got, wanted, atol=1e-9, rtol=1e-9)
        with torch.no_grad():
            actual.activation_func = F.gelu
            provider.activation_func = F.gelu
            assert (actual(x)[0] - expected).abs().max() > 1e-3
            actual.activation_func = F.silu
            provider.activation_func = F.silu
            provider.activation_func_clamp_value = None
            assert (actual(x)[0] - expected).abs().max() > 1
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group(group)
