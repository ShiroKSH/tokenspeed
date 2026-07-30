# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/
# of https://github.com/vllm-project/vllm (Apache-2.0).

"""CuTe DSL kernels for the Kimi-K3 latent-MoE tail fusion.

Only ``primitives`` (cluster/DSM helpers, shared with the fused KDA decode)
is vendored so far; the collective kernels land with the tail fusion itself.
"""
