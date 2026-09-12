# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KDA conversion tests with real Gloo TP/PP collectives, run remotely."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.bridge.models.glm5next.param_mapping import HCAlphaMapping, KDAProjectionMapping


def _worker(rank, tp_size, pp_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=tp_size * pp_size,
        init_method="file://" + rendezvous,
        timeout=timedelta(seconds=20),
    )
    try:
        tp_group = pp_group = None
        for stage in range(pp_size):
            ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                tp_group = group
        for local_rank in range(tp_size):
            ranks = list(range(local_rank, tp_size * pp_size, tp_size))
            group = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                pp_group = group
        groups = SimpleNamespace(tp=tp_group, pp=pp_group)
        owner = rank // tp_size == pp_size - 1
        local_rank = rank % tp_size
        sizes = (16, 16, 16)
        names = {key: f"model.layers.3.self_attn.{key}_proj.weight" for key in ("q", "k", "v")}
        # 2D projection/LoRA-B payload and 3D depthwise convolution, including
        # noncontiguous sources and a source/target dtype conversion.
        for trailing, dtype in [((8,), torch.float32), ((1, 4), torch.float32), ((8,), torch.bfloat16)]:
            mapping = KDAProjectionMapping("weight", names, sizes)
            mapping.set_process_groups_from_pg_collection(groups)
            hf = {}
            for index, (key, rows) in enumerate(zip(names, sizes, strict=True)):
                count = rows * torch.Size(trailing).numel()
                hf[key] = (torch.arange(count * 2, dtype=torch.float32)[::2] + index * 1000).view(rows, *trailing)
            module = torch.nn.Module() if owner else None
            local = None
            if owner:
                module.weight = torch.nn.Parameter(torch.full((sum(sizes) // tp_size, *trailing), -1, dtype=dtype))
                identity = id(module.weight)
                local = mapping.hf_to_megatron(hf if local_rank == 0 else None, module)
                expected = torch.cat(
                    [
                        tensor.narrow(0, local_rank * rows // tp_size, rows // tp_size).to(dtype)
                        for tensor, rows in zip(hf.values(), sizes, strict=True)
                    ]
                )
                torch.testing.assert_close(local, expected, rtol=0, atol=0)
                assert id(module.weight) == identity
                assert torch.all(module.weight == -1), (
                    "Conversion must not replace or mutate the destination Parameter"
                )
            exported = mapping.megatron_to_hf(local, module)
            assert set(exported) == set(names.values()), "Non-owning PP stages must also receive all HF sections"
            for key, name in names.items():
                torch.testing.assert_close(exported[name], hf[key].to(dtype), rtol=0, atol=0)
            if owner:
                bad = dict(hf, q=hf["q"][:-1]) if local_rank == 0 else None
                with pytest.raises(ValueError, match="KDA q shape"):
                    mapping.hf_to_megatron(bad, module)
                raw_fp8 = dict(hf, q=hf["q"].to(torch.float8_e4m3fn)) if local_rank == 0 else None
                with pytest.raises(ValueError, match="dequantized before conversion"):
                    mapping.hf_to_megatron(raw_fp8, module)
            dist.barrier()
        _check_mhc_scales(groups, owner, local_rank)
    finally:
        dist.destroy_process_group()


def _check_mhc_scales(groups, owner, local_rank):
    root = torch.nn.Module() if owner else None
    if owner:
        root.hc = torch.nn.Module()
        for name in ("pre", "post", "res"):
            root.hc.register_parameter("alpha_" + name, torch.nn.Parameter(torch.full((1,), -9.0)))
    mappings = [
        HCAlphaMapping("hc.alpha_" + name, "model.hc_attn_scale", index)
        for index, name in enumerate(("pre", "post", "res"))
    ]
    for mapping in mappings:
        mapping.set_process_groups_from_pg_collection(groups)
    # Import is TP replicated, with no source tensor on non-primary TP ranks.
    # Use distinct nontrivial values so swaps or a repeated coefficient fail.
    source = torch.tensor([0.125, -0.75, 1.375], dtype=torch.bfloat16)
    if owner:
        for mapping in mappings:
            param = getattr(root.hc, mapping._names[mapping.index])
            identity = id(param)
            converted = mapping.hf_to_megatron(source if local_rank == 0 else None, root)
            assert id(param) == identity and param.item() == -9.0
            torch.testing.assert_close(converted, source[mapping.index : mapping.index + 1].float(), rtol=0, atol=0)
            with torch.no_grad():
                param.copy_(converted)
        with pytest.raises(ValueError, match="shape"):
            mappings[0].hf_to_megatron(source[:2] if local_rank == 0 else None, root)
        with pytest.raises(ValueError, match="dequantized"):
            mappings[0].hf_to_megatron(source.to(torch.float8_e4m3fn) if local_rank == 0 else None, root)
    for step in range(2):
        if step and owner:
            with torch.no_grad():
                for param in root.parameters():
                    param.add_(0.25)
        exported = {}
        # Order does not matter; only the pre task emits the joined current scales.
        for mapping in reversed(mappings):
            param = getattr(root.hc, mapping._names[mapping.index]) if owner else None
            # Bridge hands the owning leaf module to a resolved absolute mapping.
            value = mapping.megatron_to_hf(param, root.hc if owner else None)
            assert not (set(exported) & set(value))
            exported.update(value)
        torch.testing.assert_close(exported["model.hc_attn_scale"], source.float() + step * 0.25, rtol=0, atol=0)
    if owner:
        root.hc.alpha_post = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="scalar"):
        mappings[0].megatron_to_hf(root.hc.alpha_pre if owner else None, root.hc if owner else None)


@pytest.mark.parametrize("tp_size,pp_size", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_kda_round_trip_uses_section_order_and_pipeline_broadcast(tmp_path, tp_size, pp_size):
    # This focused CPU test intentionally avoids CUDA and model construction.
    # The remote harness sets CPU/memory/time limits for the process tree.
    assert not torch.cuda.is_initialized()
    mp.start_processes(
        _worker,
        args=(tp_size, pp_size, str(tmp_path / "rendezvous")),
        nprocs=tp_size * pp_size,
        join=True,
        start_method="spawn",
    )


def test_kda_mapping_resolve_preserves_geometry():
    mapping = KDAProjectionMapping(
        "decoder.layers.*.weight", {key: f"model.layers.*.{key}.weight" for key in ("q", "k", "v")}, (8, 8, 8)
    ).resolve(("7",))
    assert mapping.megatron_param == "decoder.layers.7.weight"
    assert mapping.hf_param["k"] == "model.layers.7.k.weight"
    assert mapping.section_sizes == (8, 8, 8)
    assert mapping.local_hf_param_specs() == ()


def test_mhc_mapping_resolve_preserves_scale_index():
    mapping = HCAlphaMapping("decoder.layers.*.hc.alpha_post", "model.layers.*.hc_attn_scale", 1).resolve(("7",))
    assert mapping.megatron_param == "decoder.layers.7.hc.alpha_post"
    assert mapping.hf_param == "model.layers.7.hc_attn_scale"
    assert mapping.index == 1
    assert mapping.local_hf_param_specs() == ()
    with pytest.raises(ValueError, match="disagrees"):
        HCAlphaMapping("alpha_res", "scale", 0)
