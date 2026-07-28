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
#
# Orchestrates the CuTe-DSL kernels vendored under
# thirdparty/cute_dsl/latent_moe_tail/ (from the vLLM project, Apache-2.0).

"""Multicast latent-MoE tail for Kimi-K3 decode.

Replaces the ``all-reduce(latent+shared lanes) -> replicated up-projection``
tail with three fused stages: one kernel doing AR(latent)+RMSNorm+RS(shared),
a *sharded* up-projection (each rank computes ``hidden/tp`` rows — 1/tp of the
weight traffic) whose epilogue multicast-stores the shard into every rank's
mailbox (NVLS), and a barrier-free Lamport gather. Buffers come from stock
``torch.distributed._symmetric_memory``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAX_NUM_TOKENS = 16
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 8
_LAMPORT_COPY_CTAS = 32
_LAMPORT_COPY_THREADS = 224
_SUPPORTED_TP_SIZES = (8, 16)


def latent_tail_supported(
    *,
    tp_size: int,
    hidden_size: int,
    latent_size: int,
    dtype: torch.dtype,
) -> bool:
    """Cheap, non-collective eligibility probe (no rendezvous)."""
    if tp_size not in _SUPPORTED_TP_SIZES:
        return False
    if (hidden_size, latent_size) != (7168, 3584) or dtype != torch.bfloat16:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from torch.distributed import _symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class _Contract:
    group_id: int
    tp_size: int
    device: torch.device
    hidden_size: int
    latent_size: int
    rms_eps: float
    finalize_top_k: int | None


class KimiK3LatentTailOp:
    """Process-wide multicast tail; one instance (and mailbox) per contract.

    Construction performs a collective symmetric-memory rendezvous — every
    rank in ``group`` must construct with identical arguments in lockstep
    (model-layer initialization satisfies this).
    """

    _instances: dict[_Contract, "KimiK3LatentTailOp"] = {}

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        rms_eps: float,
        device: torch.device,
        finalize_top_k: int | None = None,
    ) -> "KimiK3LatentTailOp":
        """Get or build the tail op for one contract (collective rendezvous).

        Args:
            group: Tensor-parallel process group (every rank constructs in
                lockstep with identical arguments).
            hidden_size: Model hidden size (7168 for K3).
            latent_size: Routed-expert latent width (3584 for K3).
            rms_eps: Latent RMSNorm epsilon.
            device: This rank's CUDA device.
            finalize_top_k: When set (K3: 16), additionally compile the
                deferred-finalize collective variant so :meth:`deferred` can
                consume the sidecar's ``do_finalize=False`` triple directly;
                the plain :meth:`__call__` path stays available as fallback.

        Returns:
            The process-wide op instance for the contract.
        """
        contract = _Contract(
            group_id=id(group),
            tp_size=dist.get_world_size(group),
            device=device,
            hidden_size=hidden_size,
            latent_size=latent_size,
            rms_eps=float(rms_eps),
            finalize_top_k=finalize_top_k,
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group)
            cls._instances[contract] = op
        return op

    def __init__(self, contract: _Contract, group: dist.ProcessGroup) -> None:
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )

        self.contract = contract
        self.rank = dist.get_rank(group)
        with torch.accelerator.device_index(contract.device.index):
            self._collective = CollectiveKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=False,
                finalize_top_k=contract.finalize_top_k,
            )
            self._up_projection = AdaptiveUpProjectionKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                skinny_max_m=_SKINNY_MAX_NUM_TOKENS,
                mma_tiler_mn=_MMA_TILER_MN,
                cluster_shape_mn=_GEMM_CLUSTER_MN,
                b_prime_stages=_B_PRIME_STAGES,
            )
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )

    @property
    def max_num_tokens(self) -> int:
        return _MAX_NUM_TOKENS

    @property
    def supports_deferred_finalize(self) -> bool:
        """True when :meth:`deferred` is available (built with a top_k)."""
        return self.contract.finalize_top_k is not None

    def _finish(
        self,
        latent: torch.Tensor,
        shared_shard: torch.Tensor,
        up_weight: torch.Tensor,
        m: int,
    ) -> torch.Tensor:
        local_hidden = self.contract.hidden_size // self.contract.tp_size
        local_up_weight = up_weight.narrow(0, self.rank * local_hidden, local_hidden)
        mailbox = self._up_projection(latent, local_up_weight, shared_shard)
        return self._lamport_copy(mailbox, m=m).squeeze(0)

    def deferred(
        self,
        gemm2_out: torch.Tensor,
        expanded_idx: torch.Tensor,
        expert_weights: torch.Tensor,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Fused tail consuming the sidecar's deferred-finalize triple.

        The collective's staging pass performs the top-k weighted gather
        itself, so the rank-local finalized latent (and the trtllm-gen
        finalizeKernel that would produce it) is eliminated.

        Args:
            gemm2_out: Deferred ``gemm2_output`` ``[P, 3584]`` bf16 in
                permuted layout (``P`` is CUDA-graph static for fixed M).
            expanded_idx: ``expanded_idx_to_permuted_idx`` ``[M * top_k]``
                int32; ``-1`` marks dropped slots.
            expert_weights: ``[M, top_k]`` bf16 routing weights.
            shared_partial: This rank's shared-expert partial ``[M, 7168]``.
            rms_weight: Latent RMSNorm weight ``[3584]``.
            up_weight: Replicated up-projection weight ``[7168, 3584]``.

        Returns:
            ``[M, 7168]`` post-communication hidden (up-projection + shared).
        """
        m = shared_partial.shape[0]
        self._up_projection.ensure_compiled(m)
        latent, shared_shard = self._collective(
            gemm2_out,
            shared_partial,
            rms_weight,
            fin_idx=expanded_idx,
            fin_weights=expert_weights,
        )
        return self._finish(latent, shared_shard, up_weight, m)

    def __call__(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Fused tail for one decode step.

        Args:
            routed_partial: This rank's routed-expert partial ``[M, 3584]``
                (contiguous BF16, pre-all-reduce).
            shared_partial: This rank's shared-expert partial ``[M, 7168]``.
            rms_weight: Latent RMSNorm weight ``[3584]``.
            up_weight: Replicated up-projection weight ``[7168, 3584]``; this
                rank's ``hidden/tp`` row shard is consumed.

        Returns:
            ``[M, 7168]`` post-communication hidden (up-projection + shared);
            the caller still owns the residual accumulate.
        """
        m = routed_partial.shape[0]
        self._up_projection.ensure_compiled(m)
        latent, shared_shard = self._collective(
            routed_partial,
            shared_partial,
            rms_weight,
        )
        return self._finish(latent, shared_shard, up_weight, m)


__all__ = ["KimiK3LatentTailOp", "latent_tail_supported"]
