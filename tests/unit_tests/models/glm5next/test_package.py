"""Normal package initialization and public AutoBridge path; remote CPU only."""

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

import megatron.bridge
from megatron.bridge import AutoBridge
from megatron.bridge.models import GLM53FlashBridge, GLM53FlashModelProvider


@pytest.mark.parametrize("name", ["tiny", "full"])
def test_public_auto_bridge_config_path_uses_registered_native_flash(name):
    # Both modules must come from the same normal package tree. A bootstrap
    # namespace without __file__ is not evidence of running its initializer.
    assert Path(megatron.bridge.__file__).name == "__init__.py"
    assert Path(megatron.bridge.__file__).parent == Path(inspect.getfile(GLM53FlashModelProvider)).parents[2]
    source = json.loads((Path(__file__).parent / "fixtures" / f"flash_{name}.json").read_text())
    hf_config = Glm5NextConfig(**source)
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, GLM53FlashModelProvider)
    assert provider.source_hf_config == hf_config.to_dict()
    assert provider.mtp_num_layers == 1
    provider.mtp_num_layers = 0
    provider.seq_length = 128
    provider.finalize()
    exported = GLM53FlashBridge.megatron_to_hf_config(provider)
    assert exported["text_config"]["num_hidden_layers"] == provider.num_layers // 2
    assert exported["vision_config"] == provider.vision_config
    assert not torch.cuda.is_initialized()
