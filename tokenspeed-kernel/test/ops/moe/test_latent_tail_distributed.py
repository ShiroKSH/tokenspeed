# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multicast latent tail: AR(latent)+norm+RS, sharded up-projection with NVLS
multicast all-gather, Lamport gather — must match the unfused reference and
survive CUDA-graph capture/replay.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    _world_size() not in {8, 16},
    reason="launch with torchrun world size 8 or 16",
)


def _setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank)


def _reference(routed, shared, rms_w, up_w):
    lat = routed.float().clone()
    dist.all_reduce(lat)
    sh = shared.float().clone()
    dist.all_reduce(sh)
    var = lat.pow(2).mean(-1, keepdim=True)
    lat_n = (lat * torch.rsqrt(var + EPS)).to(torch.bfloat16) * rms_w
    return (lat_n.float() @ up_w.float().T + sh).to(torch.bfloat16)


def _inputs(rank, dev, m, seed):
    torch.manual_seed(1234)  # weights identical across ranks
    up_w = (torch.randn(H, L, dtype=torch.bfloat16, device=dev) * 0.02).contiguous()
    rms_w = torch.randn(L, dtype=torch.bfloat16, device=dev).contiguous()
    torch.manual_seed(seed + rank)  # per-rank partials
    routed = (torch.randn(m, L, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    shared = (torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
    return routed, shared, rms_w, up_w


TOPK = 16


def _deferred_inputs(rank, dev, m, seed):
    """Per-rank synthetic trtllm-gen deferred-finalize triple.

    Rows are shuffled across the (padded) permuted buffer and one slot per
    token is dropped (-1) to exercise the dropped-slot skip.
    """
    torch.manual_seed(seed + 31 * rank)
    padded_rows = m * TOPK + 32
    gemm2 = (
        torch.randn(padded_rows, L, dtype=torch.bfloat16, device=dev) * 0.1
    ).contiguous()
    perm = torch.randperm(padded_rows, device=dev)[: m * TOPK].to(torch.int32)
    idx = perm.clone()
    idx[TOPK - 1 :: TOPK] = -1  # last slot of every token dropped
    weights = (
        torch.softmax(torch.randn(m, TOPK, device=dev), dim=-1)
        .to(torch.bfloat16)
        .contiguous()
    )
    return gemm2, idx.contiguous(), weights


def _torch_finalize(gemm2, idx, weights):
    m, top_k = weights.shape
    rows = idx.view(m, top_k).to(torch.long)
    valid = rows >= 0
    gathered = gemm2[rows.clamp(min=0)].to(torch.float32)
    scale = weights.to(torch.float32) * valid.to(torch.float32)
    return (gathered * scale.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 4, 16])
def test_latent_tail_matches_reference(m):
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, m, seed=100)
    ref = _reference(routed, shared, rms_w, up_w)
    out = op(routed, shared, rms_w, up_w)
    torch.cuda.synchronize()
    scale = ref.float().abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05 * max(scale, 1.0), f"m={m}: err {err} vs scale {scale}"


@pytest.mark.parametrize("m", [1, 4, 16])
def test_latent_tail_deferred_finalize_matches_reference(m):
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        finalize_top_k=TOPK,
    )
    assert op.supports_deferred_finalize
    _, shared, rms_w, up_w = _inputs(rank, dev, m, seed=400)
    gemm2, idx, weights = _deferred_inputs(rank, dev, m, seed=400)
    routed = _torch_finalize(gemm2, idx, weights)
    ref = _reference(routed, shared, rms_w, up_w)
    out = op.deferred(gemm2, idx, weights, shared, rms_w, up_w)
    torch.cuda.synchronize()
    scale = ref.float().abs().max().item()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05 * max(scale, 1.0), f"m={m}: err {err} vs scale {scale}"
    # the plain path must keep working on the same (finalize-enabled) op
    out_plain = op(routed, shared, rms_w, up_w)
    torch.cuda.synchronize()
    err = (out_plain.float() - ref.float()).abs().max().item()
    assert err < 0.05 * max(scale, 1.0), f"plain fallback: err {err}"


def test_latent_tail_deferred_finalize_graph_replay():
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
        finalize_top_k=TOPK,
    )
    _, shared, rms_w, up_w = _inputs(rank, dev, 1, seed=500)
    gemm2, idx, weights = _deferred_inputs(rank, dev, 1, seed=500)
    for _ in range(3):
        op.deferred(gemm2, idx, weights, shared, rms_w, up_w)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.deferred(gemm2, idx, weights, shared, rms_w, up_w)
    for seed in (600, 601):
        torch.manual_seed(seed + rank)
        gemm2.copy_(
            torch.randn_like(gemm2, dtype=torch.float32).to(torch.bfloat16) * 0.1
        )
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(_torch_finalize(gemm2, idx, weights), shared, rms_w, up_w)
        scale = ref.float().abs().max().item()
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 0.05 * max(scale, 1.0), f"seed={seed}: err {err}"


def test_latent_tail_graph_replay():
    from tokenspeed_kernel.ops.moe.latent_tail import (
        KimiK3LatentTailOp,
        latent_tail_supported,
    )

    rank, dev = _setup()
    if not latent_tail_supported(
        tp_size=_world_size(), hidden_size=H, latent_size=L, dtype=torch.bfloat16
    ):
        pytest.skip("platform does not support the multicast tail")
    op = KimiK3LatentTailOp.initialize(
        group=dist.group.WORLD,
        hidden_size=H,
        latent_size=L,
        rms_eps=EPS,
        device=dev,
    )
    routed, shared, rms_w, up_w = _inputs(rank, dev, 1, seed=200)
    for _ in range(3):
        op(routed, shared, rms_w, up_w)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(routed, shared, rms_w, up_w)
    # two replays with fresh inputs: the Lamport state must self-reset
    scale = None
    for seed in (300, 301):
        torch.manual_seed(seed + rank)
        routed.copy_(torch.randn(1, L, dtype=torch.bfloat16, device=dev) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(routed, shared, rms_w, up_w)
        scale = ref.float().abs().max().item()
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 0.05 * max(scale, 1.0), f"seed={seed}: err {err}"
