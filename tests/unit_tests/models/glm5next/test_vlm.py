"""Flash input-layout and VLM composition tests; run on the remote host only."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel

from megatron.bridge.models.glm5next.glm53_provider import GLM53FlashModelProvider
from megatron.bridge.models.glm5next.modeling_glm53.layout import prepare_flash_input_layout
from megatron.bridge.models.glm5next.modeling_glm53.model import GLM53FlashModel


def _packed(*, legacy=False, lengths=(5, 0, 11), physical=(0, 8, 8, 32)):
    physical = torch.tensor(physical, dtype=torch.int32)
    logical = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=physical if legacy else logical,
        cu_seqlens_kv=physical if legacy else logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=int(physical.diff().max()),
        max_seqlen_kv=int(physical.diff().max()),
    )


def _oracle_positions(physical, cp_size, cp_rank):
    positions = []
    for start, end in zip(physical[:-1], physical[1:], strict=True):
        if cp_size == 1:
            positions.extend(range(start, end))
        else:
            chunk = (end - start) // (2 * cp_size)
            positions.extend(range(start + chunk * cp_rank, start + chunk * (cp_rank + 1)))
            positions.extend(range(end - chunk * (cp_rank + 1), end - chunk * cp_rank))
    return torch.tensor(positions, dtype=torch.long)


def _dense_pack(source, keep, physical):
    result = source.new_zeros((physical[-1], *source.shape[2:]))
    for row, start in enumerate(physical[:-1]):
        valid = source[row, keep[row].bool()]
        result[start : start + valid.size(0)] = valid
    return result


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int32])
def test_layout_matches_independent_dense_pack_and_gradients(cp_size, cp_rank, legacy, mask_dtype):
    ids = torch.arange(1, 37).reshape(3, 12)
    keep = torch.zeros_like(ids, dtype=mask_dtype)
    keep[0, -5:], keep[2, :11] = 1, 1
    packed = _packed(legacy=legacy)
    original_q = packed.cu_seqlens_q.clone()
    layout = prepare_flash_input_layout(ids, keep, packed, cp_size=cp_size, cp_rank=cp_rank, pad_token_id=0)
    positions = _oracle_positions([0, 8, 8, 32], cp_size, cp_rank)
    expected_ids = _dense_pack(ids, keep, [0, 8, 8, 32])[positions]
    torch.testing.assert_close(layout.input_ids, expected_ids.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(layout.packed_seq_params.cu_seqlens_q, torch.tensor([0, 5, 5, 16], dtype=torch.int32))
    torch.testing.assert_close(packed.cu_seqlens_q, original_q)
    assert layout.input_ids.size(1) == 32 // cp_size  # includes producer's bucket padding
    source = torch.randn(3, 12, 7, dtype=torch.float64, requires_grad=True)
    output = layout.select(source)
    expected = _dense_pack(source, keep, [0, 8, 8, 32])[positions].unsqueeze(0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(output)
    actual_grad = torch.autograd.grad((output * cotangent).sum(), source)[0]
    expected_grad = torch.autograd.grad((expected * cotangent).sum(), source)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_full_and_pre_sharded_thd_have_the_same_local_rows(cp_rank):
    packed = _packed()
    full_ids = torch.arange(1, 33).unsqueeze(0)
    positions = _oracle_positions([0, 8, 8, 32], 2, cp_rank)
    full = prepare_flash_input_layout(full_ids, None, packed, cp_size=2, cp_rank=cp_rank, pad_token_id=0)
    local = prepare_flash_input_layout(
        full_ids[:, positions], None, packed, cp_size=2, cp_rank=cp_rank, pad_token_id=0
    )
    assert not full.source_cp_sharded and local.source_cp_sharded
    torch.testing.assert_close(local.input_ids, full.input_ids)
    torch.testing.assert_close(local.padding_mask, full.padding_mask)


class _Group:
    def __init__(self, size=1, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Embedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.params_dtype)
        self.seen_shapes = []

    def forward(self, input_ids, position_ids):
        self.seen_shapes.append(tuple(input_ids.shape))
        return self.word_embeddings(input_ids).transpose(0, 1).contiguous()


class _Language(torch.nn.Module):
    def __init__(self, config, groups):
        super().__init__()
        self.pg_collection = groups
        self.embedding = _Embedding(config)
        self.seen = None

    def forward(self, **kwargs):
        self.seen = kwargs
        return kwargs["decoder_input"]

    def set_input_tensor(self, tensor):
        self.input_tensor = tensor

    def shared_embedding_or_output_weight(self):
        return self.embedding.word_embeddings.weight


def _provider():
    source = json.loads((Path(__file__).parent / "fixtures/flash_tiny.json").read_text())
    source["text_config"].update(hidden_size=16, vocab_size=64, pad_token_id=0, num_nextn_predict_layers=0)
    source.update(image_token_id=3, video_start_token_id=4, video_end_token_id=5)
    source["vision_config"].update(
        depth=1,
        hidden_size=16,
        num_heads=2,
        out_hidden_size=16,
        intermediate_size=32,
        projection_intermediate_size=32,
        patch_size=2,
        temporal_patch_size=1,
    )
    config = GLM53FlashModelProvider.from_hf_config(source)
    config.params_dtype, config.bf16 = torch.float32, False
    return config


def _language_only_source(**flags):
    source = json.loads((Path(__file__).parent / "fixtures/flash_tiny.json").read_text())
    source["text_config"].update(num_nextn_predict_layers=0)
    source.update(image_token_id=3, video_start_token_id=4, video_end_token_id=5, **flags)
    # Surgery checkpoints keep a depth-0 placeholder tower narrower than the decoder.
    source["vision_config"].update(depth=0, hidden_size=32, out_hidden_size=32)
    return source


@pytest.mark.parametrize("flags", [{"language_only": True}, {"vision_disabled": True}])
def test_language_only_checkpoint_accepts_placeholder_vision_width(flags):
    config = GLM53FlashModelProvider.from_hf_config(_language_only_source(**flags))
    assert config.language_only and config.vision_config["out_hidden_size"] == 32 != config.hidden_size


def test_vlm_checkpoint_still_rejects_mismatched_vision_width():
    with pytest.raises(ValueError, match="vision output width"):
        GLM53FlashModelProvider.from_hf_config(_language_only_source())


def _model(config, groups, *, pre_process=True):
    # Only the language decoder is a capture module. The wrapper constructor
    # and its HF vision tower are real, including dtype and TP metadata setup.
    language = _Language(config, groups)
    config.provide_language_model = lambda **_kwargs: language
    model = GLM53FlashModel(config, pre_process=pre_process, post_process=True)
    return model


class _HFVisionReference(torch.nn.Module):
    """Actual HF feature/placeholder helpers around the shared real vision tower."""

    get_image_features = Glm5NextModel.get_image_features
    get_video_features = Glm5NextModel.get_video_features
    get_placeholder_mask = Glm5NextModel.get_placeholder_mask

    def __init__(self, config, visual):
        super().__init__()
        self.config, self.visual = Glm5NextConfig(**config.source_hf_config), visual


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_real_vision_wrapper_matches_hf_helpers_and_gradients(cp_size, cp_rank):
    torch.manual_seed(613)
    config = _provider()
    model = _model(config, SimpleNamespace(tp=_Group(), cp=_Group(cp_size, cp_rank)))
    assert model.visual.dtype == config.params_dtype
    assert all(p.average_gradients_across_tp_domain for p in model.visual.parameters())
    ids = torch.tensor([[0, 9, 3, 8, 7, 6, 0, 0], [9, 4, 3, 3, 5, 8, 7, 0]])
    keep = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]])
    packed = _packed(legacy=True, lengths=(5, 7), physical=(0, 8, 24))
    image = torch.randn(4, 12, requires_grad=True)
    video = torch.randn(8, 12, requires_grad=True)
    image_grid, video_grid = torch.tensor([[1, 2, 2]]), torch.tensor([[2, 2, 2]])
    positions = _oracle_positions([0, 8, 24], cp_size, cp_rank)
    labels = torch.arange(positions.numel()).unsqueeze(0)
    loss_mask = torch.ones_like(labels)
    hook, context = lambda *_args: None, object()
    output = model(
        input_ids=ids,
        attention_mask=keep,
        packed_seq_params=packed,
        pixel_values=image,
        image_grid_thw=image_grid,
        pixel_values_videos=video,
        video_grid_thw=video_grid,
        labels=labels,
        loss_mask=loss_mask,
        output_processor=hook,
        output_processor_context=context,
    )
    # Only local token rows reached the embedding module; no full padded BSHD
    # hidden tensor was built on the production path.
    assert model.language_model.embedding.seen_shapes == [(1, 24 // cp_size)]
    received = model.language_model.seen
    assert received["labels"] is labels and received["loss_mask"] is loss_mask
    assert received["output_processor"] is hook and received["output_processor_context"] is context
    assert received["compute_mtp_loss"] is True
    hf = _HFVisionReference(config, model.visual)
    embedding = model.language_model.embedding.word_embeddings
    full = embedding(ids)
    img_features = torch.cat(hf.get_image_features(image, image_grid, return_dict=True).pooler_output)
    vid_features = torch.cat(hf.get_video_features(video, video_grid, return_dict=True).pooler_output)
    img_mask, vid_mask = hf.get_placeholder_mask(ids, full, img_features, vid_features)
    full = full.masked_scatter(img_mask, img_features).masked_scatter(vid_mask, vid_features)
    expected = _dense_pack(full, keep, [0, 8, 24])[positions].unsqueeze(1)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    cotangent = torch.randn_like(output)
    parameters = (embedding.weight, image, video, *model.visual.parameters())
    got = torch.autograd.grad((output * cotangent).sum(), parameters, retain_graph=True)
    wanted = torch.autograd.grad((expected * cotangent).sum(), parameters)
    for actual, reference in zip(got, wanted, strict=True):
        torch.testing.assert_close(actual, reference, rtol=5e-5, atol=3e-5)
    assert not torch.cuda.is_initialized()


def test_language_only_wrapper_rejects_media_before_vision():
    config = _provider()
    config.language_only = True
    model = _model(config, SimpleNamespace(cp=None, tp=None, pp=None))
    model.visual = None  # any access to the tower would fail loudly
    with pytest.raises(ValueError, match="language-only"):
        model(input_ids=torch.tensor([[3, 1]]), pixel_values=torch.zeros(1, 12))


def test_non_preprocess_stage_does_not_access_vision_or_unbound_inputs():
    config = _provider()
    model = _model(config, SimpleNamespace(tp=_Group(), cp=_Group()), pre_process=False)
    assert not hasattr(model, "visual")
    token = torch.randn(2, 1, 16)
    model.set_input_tensor(token)
    assert model.language_model.input_tensor is token
    assert model(input_ids=None) is None
    assert model.language_model.seen["decoder_input"] is None


def test_freeze_vision_covers_norms_and_projection():
    config = _provider()
    model = _model(config, SimpleNamespace(tp=_Group(), cp=_Group()))
    model.freeze(freeze_language_model=False, freeze_vision_model=True, freeze_vision_projection=False)
    assert not any(p.requires_grad for p in model.visual.parameters())
    assert all(p.requires_grad for p in model.language_model.parameters())


def _sp_worker(rank, rendezvous):
    patch = pytest.MonkeyPatch()
    patch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        singleton = [dist.new_group([i]) for i in range(2)]
        groups = ProcessGroupCollection(tp=dist.group.WORLD, cp=singleton[rank])
        config = _provider()
        config.sequence_parallel = True
        config.tensor_model_parallel_size = 2
        model = _model(config, groups)
        ids = torch.tensor([[0, 0, 11, 12, 13]])
        keep = torch.tensor([[False, False, True, True, True]])
        packed = _packed(legacy=True, lengths=(3,), physical=(0, 8))
        source = torch.arange(80, dtype=torch.float32).reshape(1, 5, 16).requires_grad_()
        output = model(input_ids=ids, attention_mask=keep, packed_seq_params=packed, inputs_embeds=source)
        expected_cp = _dense_pack(source, keep, [0, 8]).unsqueeze(1)
        torch.testing.assert_close(output, expected_cp.chunk(2)[rank])
        cotangent = torch.arange(128, dtype=torch.float32).reshape(8, 1, 16)
        actual_grad = torch.autograd.grad((output * cotangent.chunk(2)[rank]).sum(), source)[0]
        reference_grad = torch.autograd.grad((expected_cp * cotangent).sum(), source)[0]
        torch.testing.assert_close(actual_grad, reference_grad)
        mask = model.language_model.seen["padding_mask"]
        assert mask.shape == (1, 4)
        assert mask.tolist() == ([[False, False, False, True]] if rank == 0 else [[True] * 4])
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()
        patch.undo()


def test_wrapper_uses_real_sequence_parallel_scatter_and_adjoint(tmp_path):
    mp.spawn(_sp_worker, args=(str(tmp_path / "sp-gloo"),), nprocs=2, join=True)


def test_bad_mask_and_conflicting_metadata_fail_before_embedding():
    ids = torch.ones(1, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="zero/one"):
        prepare_flash_input_layout(
            ids, ids * 2, _packed(lengths=(8,), physical=(0, 8)), cp_size=1, cp_rank=0, pad_token_id=0
        )
    bad = _packed(lengths=(4,), physical=(0, 8))
    with pytest.raises(ValueError, match="disagree"):
        prepare_flash_input_layout(ids, ids, bad, cp_size=1, cp_rank=0, pad_token_id=0)
    bad.cu_seqlens_q_padded = torch.tensor([0, 7], dtype=torch.int32)
    with pytest.raises(ValueError, match="matching"):
        prepare_flash_input_layout(ids, None, bad, cp_size=2, cp_rank=0, pad_token_id=0)
