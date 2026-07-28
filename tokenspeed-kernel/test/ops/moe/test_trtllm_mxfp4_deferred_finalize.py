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

"""SiTU sidecar deferred finalize (``do_finalize=False``) contract tests.

Verifies that the raw-FFI deferred path returns the trtllm-gen finalize
triple whose torch-side reduction reproduces the in-op finalized output,
that the permuted-row count is routing-independent (CUDA-graph static),
and that the deferred call survives graph capture/replay.
"""

from __future__ import annotations

import pytest
import torch

E, TOPK, HID, ISPP = 896, 16, 3584, 384
T = 4


def _deferred_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
            mxfp4_situ_deferred_finalize_supported,
        )
    except ImportError:
        return False
    return mxfp4_situ_deferred_finalize_supported()


pytestmark = pytest.mark.skipif(
    not _deferred_available(),
    reason="requires the SiTU sidecar raw FFI on sm103",
)


@pytest.fixture(scope="module")
def situ_module():
    """Synthetic expert weights in the sidecar's final layout.

    Weight bytes are random (every MXFP4 nibble is finite) and all scale
    bytes are 127 (e8m0 => 1.0), so the finalized and deferred paths see the
    identical, well-defined problem.
    """
    torch.manual_seed(0)
    dev = "cuda"
    w = torch.nn.Module()
    w.w13_weight = torch.nn.Parameter(
        torch.randint(0, 256, (E, 2 * ISPP, HID // 2), dtype=torch.uint8, device=dev),
        requires_grad=False,
    )
    w.w13_weight_scale = torch.nn.Parameter(
        torch.full((E, 2 * ISPP, HID // 32), 127, dtype=torch.uint8, device=dev).view(
            torch.float8_e4m3fn
        ),
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        torch.randint(0, 256, (E, HID, ISPP // 2), dtype=torch.uint8, device=dev),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.full((E, HID, ISPP // 32), 127, dtype=torch.uint8, device=dev).view(
            torch.float8_e4m3fn
        ),
        requires_grad=False,
    )
    w.gemm1_alpha = torch.nn.Parameter(
        torch.ones(E, dtype=torch.float32, device=dev), requires_grad=False
    )
    w.gemm1_beta = torch.nn.Parameter(
        torch.ones(E, dtype=torch.float32, device=dev), requires_grad=False
    )
    w.num_experts = E
    w.top_k = TOPK
    w.intermediate_size_per_partition = ISPP
    w.hidden_size_padded = HID
    w.hidden_size_original = HID
    return w


def _routing(seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)])
    weights = torch.softmax(torch.randn(T, TOPK, generator=g), dim=-1)
    return ids.to(torch.int32).cuda(), weights.to(torch.bfloat16).cuda()


def _finalized(w, ids, weights):
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        _call_mxfp4_situ_routed_moe,
    )

    out = torch.empty(T, HID, dtype=torch.bfloat16, device="cuda")
    return _call_mxfp4_situ_routed_moe(w, weights, ids, _x(), out, False).clone()


def _deferred(w, ids, weights):
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
        _call_mxfp4_situ_routed_moe_deferred,
    )

    return _call_mxfp4_situ_routed_moe_deferred(w, weights, ids, _x(), False)


_X = None


def _x() -> torch.Tensor:
    global _X
    if _X is None:
        torch.manual_seed(7)
        _X = torch.randn(T, HID, dtype=torch.bfloat16, device="cuda") * 0.1
    return _X


def torch_finalize(
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx: torch.Tensor,
) -> torch.Tensor:
    """Reference: ``out[t] = sum_k w[t, k] * gemm2[idx[t * K + k]]``."""
    num_tokens, top_k = expert_weights.shape
    idx = expanded_idx.view(num_tokens, top_k).to(torch.long)
    valid = idx >= 0
    rows = gemm2_out[idx.clamp(min=0)].to(torch.float32)
    scale = expert_weights.to(torch.float32) * valid.to(torch.float32)
    return (rows * scale.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


def test_deferred_triple_matches_finalized(situ_module):
    ids, weights = _routing(1)
    reference = _finalized(situ_module, ids, weights)
    gemm2_out, expert_weights, expanded_idx = _deferred(situ_module, ids, weights)
    torch.cuda.synchronize()

    assert gemm2_out.dtype == torch.bfloat16 and gemm2_out.shape[1] == HID
    assert expert_weights.shape == (T, TOPK)
    assert expanded_idx.dtype == torch.int32 and expanded_idx.numel() == T * TOPK

    mine = torch_finalize(gemm2_out, expert_weights, expanded_idx)
    diff = (mine.float() - reference.float()).abs().max().item()
    scale = reference.float().abs().max().clamp(min=1.0).item()
    assert diff <= 0.02 * scale, f"finalize mismatch: {diff} vs scale {scale}"


def test_deferred_shapes_are_routing_independent(situ_module):
    ids1, weights1 = _routing(2)
    ids2, weights2 = _routing(3)
    g1, _, i1 = _deferred(situ_module, ids1, weights1)
    g2, _, i2 = _deferred(situ_module, ids2, weights2)
    torch.cuda.synchronize()
    assert g1.shape == g2.shape, "gemm2_output rows depend on routing content"
    assert i1.shape == i2.shape


def test_deferred_graph_capture_replay(situ_module):
    ids, weights = _routing(4)
    ids_buf, weights_buf = ids.clone(), weights.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _deferred(situ_module, ids_buf, weights_buf)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm2_out, expert_weights, expanded_idx = _deferred(
            situ_module, ids_buf, weights_buf
        )

    ids2, weights2 = _routing(5)
    ids_buf.copy_(ids2)
    weights_buf.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()

    reference = _finalized(situ_module, ids2, weights2)
    # expert_weights was captured from the pre-copy warm inputs; the replayed
    # graph recomputes it from ids_buf/weights_buf contents.
    mine = torch_finalize(gemm2_out, weights_buf, expanded_idx)
    diff = (mine.float() - reference.float()).abs().max().item()
    scale = reference.float().abs().max().clamp(min=1.0).item()
    assert diff <= 0.02 * scale, f"replay mismatch: {diff} vs scale {scale}"
