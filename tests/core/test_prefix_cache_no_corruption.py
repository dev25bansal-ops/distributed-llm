"""Regression tests for C2: prefix-cache hits corrupted output.

``request_pipeline`` used to adopt ``match_len`` from
``CacheManager.lookup_prefix()`` while DISCARDING the matched ``kv_data``;
the scheduler then emitted only ``prompt_tokens[match_len:]`` and every
node's KV caches started empty, so attention ran over a missing prefix and
produced garbage whenever any cache hit occurred (and a full-prompt hit
crashed outright).

Fix: hits are treated as misses (``prefix_match_len`` stays 0, full prompt
prefetched) until cached-KV reuse is actually wired — see
``PREFIX_CACHE_KV_REUSE_WIRED`` in request_pipeline.py.

These tests drive the REAL local generation loop (BatchScheduler +
RequestPipeline._generate_local_batch) over a deterministic stub model
whose sampled token depends on the observed input width, so any silent
token-skipping changes the output.

Run: pytest tests/core/test_prefix_cache_no_corruption.py -v
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch

from distllm.core.batch_scheduler import BatchScheduler, Sequence
from distllm.core.cache_manager import CacheManager
from distllm.core.request_pipeline import RequestPipeline, PREFIX_CACHE_KV_REUSE_WIRED


# ---------------------------------------------------------------------------
# Deterministic stubs
# ---------------------------------------------------------------------------


class _WidthSensitiveModel(torch.nn.Module):
    """Tiny model whose greedy token depends on the input width.

    The argmax winner is token 5 for even widths and token 9 for odd
    widths, so if the pipeline silently drops the cached prefix span
    (changing the width parity), the generated output changes too.
    """

    VOCAB = 32

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.seen_widths: list[int] = []

    def logits_for(self, width: int) -> torch.Tensor:
        winner = 5 if width % 2 == 0 else 9
        logits = torch.zeros(1, max(width, 0), self.VOCAB)
        if width > 0:
            logits[0, -1, winner] = 10.0
        return logits

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None):
        width = int(input_ids.shape[-1])
        self.seen_widths.append(width)
        return type("Out", (), {"logits": self.logits_for(width)})()


class _FakePartitioner:

    def __init__(self, model: _WidthSensitiveModel) -> None:
        self.full_model = model


class _GreedyTokenGen:
    """Argmax sampling — fully deterministic."""

    tokenizer = None

    def sample_batch(self, logits, sequences, tokenizer=None):
        # request_pipeline hands us last-position logits [B, V].
        winners = logits.argmax(dim=-1)
        return winners, []

    def sample(self, logits, temperature=1.0, top_p=1.0, top_k=0):
        return logits.argmax(dim=-1).reshape(1, 1)


class _StubParamChannel:

    def get(self, request_id):
        return None


class _DistributedStubPipeline:
    """Stands in for dist.pipeline in _run_distributed_pipeline_batch.

    Mirrors the real contract: ``run_pipeline`` returns full-position
    logits ``[batch, seq, vocab]``; the caller slices ``[:, -1, :]``.
    """

    enable_overlap = False

    def __init__(self, model: _WidthSensitiveModel) -> None:
        self.model = model

    def create_node_kv_caches(self):
        return {}

    def run_pipeline(self, input_ids, node_kv_caches, request_id=None, **_kw):
        # Same width-parity trick as the local model: skipping the cached
        # span changes the width, hence the greedy token.
        width = int(input_ids.shape[-1])
        self.model.seen_widths.append(width)
        return self.model.logits_for(width)


class _StubCoordinator:

    def __init__(self, cache_mgr: CacheManager) -> None:
        self.model = _WidthSensitiveModel()
        self.local_partitioner = _FakePartitioner(self.model)
        self._spec_decoder = None
        self._cache_mgr = cache_mgr
        self.prefix_cache = cache_mgr.prefix_cache is not None
        self.scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=4096)
        self._batch_kv_caches: dict = {}
        self._batch_kv_caches_lock = threading.Lock()
        self._token_gen = _GreedyTokenGen()
        self._param_update_channel = _StubParamChannel()
        self._pipeline = _DistributedStubPipeline(self.model)
        self._async_pipeline = None
        self.tokenizer = None


PROMPT = list(range(100, 124))  # 24 tokens (even)


def _run_once(coord: _StubCoordinator, request_id: str) -> tuple[list[int], Sequence]:
    seq = Sequence(
        request_id=request_id, prompt_tokens=list(PROMPT), max_new_tokens=2,
    )
    coord.scheduler.add(seq)
    for _ in range(50):
        if seq.is_complete:
            break
        batch = coord.scheduler.schedule()
        assert batch is not None, "scheduler starved while sequence active"
        pipe = coord.pipeline
        pipe._generate_local_batch(batch)
    assert seq.is_complete
    return list(seq.generated_tokens), seq


@pytest.fixture()
def coord():
    cache_mgr = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=8)
    c = _StubCoordinator(cache_mgr)
    c.pipeline = RequestPipeline(c)  # type: ignore[attr-defined]
    return c


def test_kv_reuse_flag_is_off():
    """The reuse switch must stay off until KV restoration is wired (C2)."""
    assert PREFIX_CACHE_KV_REUSE_WIRED is False


def test_full_prompt_hit_second_output_equals_first(coord):
    """Identical prompt twice (full-prompt cache entry): outputs must match."""
    first, seq1 = _run_once(coord, "req-1")

    # Simulate the post-completion store that step()/pipeline would do.
    coord._cache_mgr.store_prefix(PROMPT, {"kv": "full-prompt-blob"})

    # Sanity: the cache genuinely serves a full-prompt hit now...
    match_len, kv = coord._cache_mgr.lookup_prefix(PROMPT)
    assert match_len == len(PROMPT) and kv is not None

    second, seq2 = _run_once(coord, "req-2")

    assert first == second, (
        f"output changed on repeated prompt: {first} != {second} "
        f"(widths seen: {coord.model.seen_widths})"
    )
    # ...but the pipeline must NOT act on it: full prompt processed again.
    # Widths per run: one 24-token prefill + one 1-token decode.
    assert seq2.prefix_match_len == 0
    assert coord.model.seen_widths == [24, 1, 24, 1]


def test_partial_hit_second_output_equals_first(coord):
    """Partial prefix hit (17 of 24 tokens): outputs must still match."""
    first, _ = _run_once(coord, "req-1")

    # Store only a 17-token prefix of the prompt -> partial hit on lookup.
    coord._cache_mgr.store_prefix(PROMPT[:17], {"kv": "partial-blob"})
    match_len, kv = coord._cache_mgr.lookup_prefix(PROMPT)
    assert match_len == 17 and kv is not None

    second, seq2 = _run_once(coord, "req-2")

    assert first == second, (
        f"output changed on repeated prompt: {first} != {second} "
        f"(widths seen: {coord.model.seen_widths})"
    )
    assert seq2.prefix_match_len == 0
    # Both runs fed the FULL prompt (skipping would make the prefill width
    # 7 -> odd -> a different greedy token, caught by the equality check).
    # Widths per run: one 24-token prefill + one 1-token decode.
    assert coord.model.seen_widths == [24, 1, 24, 1]


def test_miss_path_unaffected_by_gate(coord):
    """With no cache entries the loop behaves exactly as before the gate."""
    out, seq = _run_once(coord, "req-1")
    assert len(out) == 2
    assert seq.prefix_match_len == 0
    # One 24-token prefill + one 1-token decode.
    assert coord.model.seen_widths == [24, 1]


# ---------------------------------------------------------------------------
# Distributed-path drivers.
#
# _run_distributed_pipeline_batch re-slices prompt_tokens[prefix_match_len:]
# AFTER the lookup has (pre-fix) adopted match_len, so this is the path
# where a bare hit actually corrupts the model input.
# ---------------------------------------------------------------------------


def _run_once_distributed(coord, request_id: str) -> tuple[list[int], Sequence]:
    seq = Sequence(
        request_id=request_id, prompt_tokens=list(PROMPT), max_new_tokens=2,
    )
    coord.scheduler.add(seq)
    for _ in range(50):
        if seq.is_complete:
            break
        batch = coord.scheduler.schedule()
        assert batch is not None, "scheduler starved while sequence active"
        coord.pipeline._run_distributed_pipeline_batch(batch)
    assert seq.is_complete
    return list(seq.generated_tokens), seq


def test_distributed_full_prompt_hit_second_output_equals_first(coord):
    first, _ = _run_once_distributed(coord, "req-1")
    coord._cache_mgr.store_prefix(PROMPT, {"kv": "full-prompt-blob"})
    second, seq2 = _run_once_distributed(coord, "req-2")

    assert first == second, (
        f"output changed on repeated prompt: {first} != {second} "
        f"(widths seen: {coord.model.seen_widths})"
    )
    assert seq2.prefix_match_len == 0
    # Widths per run: 24-token prefill + 1-token decode, twice.
    assert coord.model.seen_widths == [24, 1, 24, 1]


def test_distributed_partial_hit_second_output_equals_first(coord):
    first, _ = _run_once_distributed(coord, "req-1")

    # 17 of 24 tokens cached -> partial hit. Pre-fix the pipeline adopted
    # match_len=17 and fed only prompt[17:] (7 tokens -> odd width ->
    # different greedy token => corrupted output).
    coord._cache_mgr.store_prefix(PROMPT[:17], {"kv": "partial-blob"})
    assert coord._cache_mgr.lookup_prefix(PROMPT)[0] == 17

    second, seq2 = _run_once_distributed(coord, "req-2")

    assert first == second, (
        f"output changed on repeated prompt: {first} != {second} "
        f"(widths seen: {coord.model.seen_widths})"
    )
    assert seq2.prefix_match_len == 0
    assert coord.model.seen_widths == [24, 1, 24, 1]
