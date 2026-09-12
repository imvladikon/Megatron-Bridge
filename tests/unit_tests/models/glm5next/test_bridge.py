"""Flash registration and released checkpoint coverage, executed remotely."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.glm5next.glm53_bridge import GLM53FlashBridge
from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider
from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel


FIXTURES = Path(__file__).parent / "fixtures"


def _source(name="tiny"):
    return json.loads((FIXTURES / f"flash_{name}.json").read_text())


def _bridge(name="tiny", *, normalized=False):
    source = _source(name)
    config = Glm5NextConfig(**source) if normalized else SimpleNamespace(to_dict=lambda: deepcopy(source))
    bridge = get_model_bridge("Glm5NextForConditionalGeneration", hf_config=config)
    return bridge, config


@pytest.mark.parametrize("name", ["tiny", "full"])
@pytest.mark.parametrize("normalized", [False, True])
def test_registered_bridge_preserves_hf_blocks_and_vlm_config(name, normalized):
    bridge, config = _bridge(name, normalized=normalized)
    assert isinstance(bridge, GLM53FlashBridge)
    assert bridge.PROVIDER_CLASS is GLM53FlashModelProvider
    assert bridge.MODEL_TYPE == "glm5_next"
    provider = bridge.provider_bridge(SimpleNamespace(config=config))
    assert provider.source_hf_config == config.to_dict()
    provider.mtp_num_layers = 0
    provider.seq_length = 128
    provider.finalize()
    exported = bridge.megatron_to_hf_config(provider)
    assert exported["text_config"]["num_hidden_layers"] == provider.num_layers // 2
    assert exported["vision_config"] == config.to_dict()["vision_config"]
    assert exported["text_config"]["num_nextn_predict_layers"] == 0
    assert provider.source_hf_config["text_config"]["num_nextn_predict_layers"] == 1
    assert exported["text_config"]["dtype"] == "bfloat16"
    assert exported["architectures"] == ["Glm5NextForConditionalGeneration"]
    assert not torch.cuda.is_initialized()


def test_missing_checkpoint_source_fails_task_validation():
    bridge, _ = _bridge()
    registry = bridge.mapping_registry()
    params = ["language_model.decoder.final_norm.weight"]
    bridge._validate_conversion_mappings(registry, params, ["model.language_model.norm.weight"])
    with pytest.raises(ValueError, match="missing mapped"):
        bridge._validate_conversion_mappings(registry, params, [])


def test_export_rejects_geometry_override_and_unimplemented_mtp():
    bridge, config = _bridge()
    provider = bridge.provider_bridge(SimpleNamespace(config=config))
    with pytest.raises(NotImplementedError, match="MTP"):
        bridge.megatron_to_hf_config(provider)
    provider.mtp_num_layers = 0
    provider.hidden_size += 8
    with pytest.raises(ValueError, match="hidden_size"):
        bridge.megatron_to_hf_config(provider)


def test_raw_quantized_sources_never_become_unscaled_floating_weights():
    bridge, _ = _bridge()
    weight = torch.arange(8).float().reshape(2, 4)
    assert bridge.maybe_modify_loaded_hf_weight("weight", {"weight": weight}) is weight
    assert bridge.maybe_modify_loaded_hf_weight({"q": "weight"}, {"weight": weight})["q"] is weight
    with pytest.raises(NotImplementedError, match="scale-aware"):
        bridge.maybe_modify_loaded_hf_weight("weight", {"weight": weight.to(torch.float8_e4m3fn)})


@pytest.mark.parametrize("pad_token", [-1, True, 154880])
def test_vlm_pad_id_is_validated_before_embedding(pad_token):
    source = _source()
    source["text_config"]["pad_token_id"] = pad_token
    with pytest.raises(ValueError, match="pad_token_id"):
        GLM53FlashModelProvider.from_hf_config(source)


def test_target_type_is_native_flash_vlm():
    assert GLM53FlashBridge.SOURCE_NAME == "Glm5NextForConditionalGeneration"
    assert GLM53FlashModel.__module__.endswith("modeling_glm53.model")
