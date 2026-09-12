"""Real collective regression for the direct effective-weight path.

Run on the validation server with TP=2 and TP=4, for example:
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_lora_merge_distributed.py -q

Only small CPU tensors are used; no model is constructed. The regular LoRA
forward tests do not cover consumers of LoRALinear.weight (e.g. absorbed MLA).
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.peft.lora_merge import LoRAMerge


@pytest.fixture(scope="module")
def tp_group():
    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4):
        pytest.skip("requires torch.distributed.run with 2 or 4 ranks")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    try:
        assert dist.get_backend() == "gloo"
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("partition", ["column", "row"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zero_b", [False, True])
def test_effective_weight_gradients_match_unsharded(tp_group, partition, dtype, zero_b):
    """Check forward, accumulated A/B gradients, and the no-grad export path."""
    world = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    base = torch.randn(16, 12, generator=generator, dtype=dtype) * 0.2
    a = (torch.randn(8, 12, generator=generator, dtype=dtype) * 0.2).requires_grad_()
    b = (torch.randn(16, 8, generator=generator, dtype=dtype) * 0.2).requires_grad_()
    if zero_b:
        with torch.no_grad():
            b.zero_()

    weight_dim, a_dim = (0, 0) if partition == "column" else (1, 1)
    local_base = base.chunk(world, dim=weight_dim)[rank]
    local_a = a.detach().chunk(world, dim=a_dim)[rank].clone().requires_grad_()
    local_b = b.detach().chunk(world, dim=0)[rank].clone().requires_grad_()
    merge = LoRAMerge()
    tolerances = {"rtol": 1e-5, "atol": 3e-5} if dtype == torch.float32 else {"rtol": 1e-12, "atol": 1e-12}

    # A different loss on every output shard forces the backward collective to
    # include remote contributions, not just restore a local autograd edge.
    for microbatch in range(2):
        target = torch.randn(16, 12, generator=generator, dtype=dtype) + microbatch
        expected = base + 2 * b @ a
        (expected - target).square().sum().backward()
        actual = merge.merge(local_base, local_b, local_a, 16, 8, tp_group=tp_group)
        local_target = target.chunk(world, dim=weight_dim)[rank]
        (actual - local_target).square().sum().backward()
        torch.testing.assert_close(actual, expected.chunk(world, dim=weight_dim)[rank], **tolerances)
        assert local_a.grad is not None
        assert local_b.grad is not None
        torch.testing.assert_close(local_a.grad, a.grad.chunk(world, dim=a_dim)[rank], **tolerances)
        torch.testing.assert_close(local_b.grad, b.grad.chunk(world, dim=0)[rank], **tolerances)

    with torch.no_grad():
        exported = merge.merge(local_base, local_b, local_a, 16, 8, tp_group=tp_group)
    assert not exported.requires_grad
    torch.testing.assert_close(exported, actual.detach(), **tolerances)
