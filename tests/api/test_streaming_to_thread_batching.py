"""Regression test: PERF-07-STREAM — coalesce per-token CPU work off the
event loop so streaming N tokens makes exactly N asyncio.to_thread hops
(one hop per token: the stateful model call plus sample/decode/logprob CPU
work together), instead of the pre-fix two-or-more hops per token.

The model call itself stays one hop per token (it is stateful: it consumes
and updates past_key_values), and the sample/decode/logprob CPU work is
moved into that same hop instead of blocking the event loop separately.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import torch

import distllm.api.streaming as streaming_mod
from distllm.api.streaming import _generate_tokens


def _make_coord(n_tokens: int = 100):
    """Coordinator mock whose model yields deterministic, decodeable logits."""

    def decode_side_effect(ids, **kw):
        if isinstance(ids, torch.Tensor):
            ids = int(ids.item())
        if isinstance(ids, list):
            return "".join(chr(65 + (t % 26)) for t in ids)
        return chr(65 + (ids % 26))

    tok = MagicMock()
    tok.eos_token_id = None
    tok.decode.side_effect = decode_side_effect

    def encode_side_effect(text, return_tensors=None):
        return torch.tensor([[1, 2, 3]])

    tok.encode.side_effect = encode_side_effect

    state = {"step": 0}

    def model_fn(*args, **kwargs):
        state["step"] += 1
        logits = torch.full((1, 1, 256), -1e9)
        logits[0, 0, 65 + ((state["step"] - 1) % 26)] = 10.0
        out = MagicMock()
        out.logits = logits
        out.past_key_values = None
        return out

    mock_model = MagicMock()
    mock_model.side_effect = model_fn
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    coord = MagicMock()
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord.tokenizer = tok
    coord.node_order = []
    coord._param_update_channel = None
    coord._inference_engine = None
    return coord


async def _run_and_count(n_tokens: int):
    coord = _make_coord(n_tokens)
    original = streaming_mod.g.coordinator
    streaming_mod.g.coordinator = coord

    calls = {"n": 0}
    real_to_thread = asyncio.to_thread

    async def counting_to_thread(func, *args, **kwargs):
        calls["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    streaming_mod.asyncio.to_thread = counting_to_thread
    try:
        gen = _generate_tokens(
            "hello", "req-1", max_tokens=n_tokens,
            temperature=0.7, top_p=0.9, top_k=0,
        )
        tokens = [t async for t in gen]
    finally:
        streaming_mod.asyncio.to_thread = real_to_thread
        streaming_mod.g.coordinator = original
    return tokens, calls["n"]


class TestStreamToThreadBatching:
    def test_100_tokens_makes_one_to_thread_call_per_token(self):
        """PERF-07-STREAM: sample/decode/logprobs coalesced into the model hop.

        The autoregressive model call stays one hop per token (it consumes
        and updates past_key_values), so 100 tokens make exactly 100 hops.
        The regression this guards: the pre-fix code made an ADDITIONAL hop
        per token for the CPU work (sample/decode/logprobs on the event
        loop or a second to_thread), i.e. >= 2*N hops. Assert == N so the
        CPU work cannot drift back onto a separate hop.
        """
        tokens, n_calls = asyncio.run(_run_and_count(100))
        assert len(tokens) == 100, "each token must still be yielded"
        assert n_calls == 100, (
            f"expected exactly one to_thread hop per token (100), got {n_calls}; "
            f"a higher count means the per-token CPU work split onto extra hops"
        )

    def test_100_tokens_each_token_yielded(self):
        """Every token still yields (SSE per-token semantics preserved)."""
        tuples, n_calls = asyncio.run(_run_and_count(100))
        # _generate_tokens yields (token_text, logprob_data, ttft) tuples
        assert len(tuples) == 100
        assert all(isinstance(t, tuple) and len(t) == 3 for t in tuples)
        assert all(isinstance(t[0], str) and t[0] for t in tuples)

    def test_single_token_stream(self):
        """A 1-token stream still works and yields exactly one token."""
        tokens, n_calls = asyncio.run(_run_and_count(1))
        assert len(tokens) == 1
        assert n_calls >= 1

    @pytest.mark.asyncio
    async def test_model_called_once_per_token(self):
        """Model (KV-cache stateful) is still invoked once per token — order intact."""
        coord = _make_coord(50)
        original = streaming_mod.g.coordinator
        streaming_mod.g.coordinator = coord
        try:
            gen = _generate_tokens(
                "x", "req-1", max_tokens=50,
                temperature=0.7, top_p=0.9, top_k=0,
            )
            tuples = [t async for t in gen]
        finally:
            streaming_mod.g.coordinator = original
        assert len(tuples) == 50
        assert coord.local_partitioner.full_model.call_count == 50, (
            "model must run once per token (autoregressive KV cache)"
        )
