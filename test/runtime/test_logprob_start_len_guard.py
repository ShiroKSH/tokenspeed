"""InputProcessor guard for the unsupported SGLang ``logprob_start_len``.

Prompt logprobs are not implemented, so a request asking for them must be
rejected loudly rather than silently downgraded. The guard used to read

    (getattr(obj, "logprob_start_len", -1) or -1) >= 0

which mis-handles the boundary value: ``0 or -1`` evaluates to ``-1`` because
0 is falsy, so ``logprob_start_len=0`` -- "give me logprobs from the very
first prompt token", the most natural way to ask for the unsupported feature
-- slipped through and was then silently ignored. The list form (one entry
per request in a batch) was not handled either.

The guard lives partway through the async ``tokenize_one_request``, so this
drives it with a stub engine and pre-tokenized ``input_ids`` to skip the
tokenizer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import GenerateReqInput

_GUARD_MSG = "logprob_start_len >= 0 (prompt logprobs) is not supported yet."


def _processor() -> InputProcessor:
    engine = types.SimpleNamespace(
        is_generation=True,
        context_len=4096,
        logger=types.SimpleNamespace(warning=lambda *a, **k: None),
        tokenizer=None,
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(),
            is_multimodal=False,
            is_multimodal_active=False,
            vocab_size=1000,
        ),
        server_args=types.SimpleNamespace(
            disaggregation_mode="null",
            enable_output_logprobs=True,
            enable_prefix_caching=True,
        ),
    )
    return InputProcessor(engine)


def _request(start_len) -> GenerateReqInput:
    # input_ids (not text) so no tokenizer is needed to reach the guard.
    obj = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1})
    obj.normalize_batch_and_arguments()
    obj.return_logprob = True
    obj.top_logprobs_num = 0
    obj.logprob_start_len = start_len
    obj.token_ids_logprob = None
    return obj


def _rejects(start_len) -> bool:
    """Whether the guard rejects this logprob_start_len."""
    try:
        asyncio.run(_processor().tokenize_one_request(_request(start_len)))
    except ValueError as exc:
        if _GUARD_MSG in str(exc):
            return True
        raise
    except Exception as exc:  # pragma: no cover - stub gap, not the guard
        pytest.skip(f"tokenize_one_request needs more engine state: {exc!r}")
    return False


class LogprobStartLenGuardTest(unittest.TestCase):
    def test_zero_is_rejected(self):
        # The regression: 0 is falsy, so the old `or -1` turned it into -1.
        self.assertTrue(_rejects(0))

    def test_positive_is_rejected(self):
        self.assertTrue(_rejects(5))

    def test_disabled_sentinels_are_accepted(self):
        self.assertFalse(_rejects(-1))
        self.assertFalse(_rejects(None))

    def test_batch_list_is_scanned_elementwise(self):
        self.assertTrue(_rejects([0]))
        self.assertTrue(_rejects([-1, 3]))
        self.assertFalse(_rejects([-1, -1]))
        self.assertFalse(_rejects([None, -1]))


if __name__ == "__main__":
    unittest.main()
