"""Regression tests: prompt-lookup partial-accept crop under transformers>=5.

Bug: ``_PromptLookupStrategy._generate_tokens`` trimmed the KV cache after a
partially accepted draft round with ``past.crop(prefix_len + accepted)`` — an
ABSOLUTE target length.  That matches transformers 4.x ``DynamicCache.crop
(max_length)``, where the argument was the final cache size.  In
transformers>=5 (``crop(tokens_to_remove)``) the argument is a RELATIVE count
that must be negative; a positive value raises::

    ValueError: Calling `crop` with a positive value (`129`) is not supported.
    Please use a negative integer to remove that number of tokens from the
    cache instead.

so any request whose prompt-lookup draft round partially accepted drafts
(e.g. the background ``/v1/batch/submit`` worker) died at execution while
no-crop rounds succeeded.

The main test drives the real ``_PromptLookupStrategy`` against a rule-based
stub model whose logits are a pure function of the attended prefix, with a
fake cache implementing transformers>=5 semantics (``crop(tokens_to_remove)``,
negative-only).  The attended history is checkpointed at every forward pass:
each checkpoint must be a prefix of the independently computed greedy chain,
so a skipped, mis-signed, or mis-sized crop diverges and fails.
"""

from __future__ import annotations

import inspect
import re

import pytest

torch = pytest.importorskip("torch")

from distllm.core.inference_engine import (  # noqa: E402
    InferenceEngine,
    _PromptLookupStrategy,
)

VOCAB = 16


# ---------------------------------------------------------------------------
# transformers>=5 fake cache: relative, negative-only crop.
# ---------------------------------------------------------------------------


class _T5Cache:
    """Duck-typed ``DynamicCache`` with transformers>=5 crop semantics.

    ``crop`` takes a NEGATIVE relative count; a positive argument raises the
    exact ``ValueError`` the installed library raises.  As with ``_CacheStub``
    in ``test_kv_reuse.py``, cropping also truncates the owning model's token
    history, mirroring how ``DynamicCache.crop`` drops KV entries.
    """

    def __init__(self, model: "_RuleModel") -> None:
        self._model = model
        self.recorded_lens: list[int] = []
        # (length_before, tokens_to_remove) for every crop() call.
        self.crop_calls: list[tuple[int, int]] = []

    def get_seq_length(self) -> int:
        return len(self._model.history)

    def crop(self, tokens_to_remove: int) -> None:
        before = self.get_seq_length()
        self.crop_calls.append((before, tokens_to_remove))
        if tokens_to_remove > 0:
            raise ValueError(
                f"Calling `crop` with a positive value (`{tokens_to_remove}`) "
                "is not supported. Please use a negative integer to remove "
                "that number of tokens from the cache instead."
            )
        if tokens_to_remove == 0:
            return
        keep = max(0, before - abs(tokens_to_remove))
        del self._model.history[keep:]
        self.recorded_lens = [min(n, keep) for n in self.recorded_lens]

    def record(self, n_in: int) -> None:
        self.recorded_lens.append(self.get_seq_length())


class _Out:
    def __init__(self, logits: torch.Tensor, past) -> None:
        self.logits = logits
        self.past_key_values = past


class _RuleModel:
    """Causal LM stand-in: next = (last + second-last) mod 16, spiked with a
    +100 logit.  Logits are a pure function of the attended prefix (the full
    token history, which the cache crops on rejection), so any KV-threading
    or crop mistake is impossible to hide."""

    VOCAB = VOCAB

    def __init__(self) -> None:
        self.cache = _T5Cache(self)
        self.history: list[int] = []
        self.call_input_lens: list[int] = []
        # Snapshot of the attended history at every forward-call entry, i.e.
        # one checkpoint per generation round.
        self.checkpoints: list[list[int]] = []

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        seq = input_ids[0].tolist()
        self.call_input_lens.append(len(seq))
        self.checkpoints.append(list(self.history))
        cache = past_key_values if past_key_values is not None else self.cache
        start = len(self.history)
        self.history.extend(seq)
        rows = []
        for i in range(len(seq)):
            h = self.history[: start + i + 1]
            row = torch.zeros(self.VOCAB)
            nxt = (h[-2] + h[-1]) % self.VOCAB if len(h) >= 2 else h[-1]
            row[nxt] = 100.0
            rows.append(row)
        cache.record(len(seq))
        return _Out(torch.stack(rows).unsqueeze(0), cache)


def _rule_plain_chain(prompt: list[int], n: int) -> list[int]:
    """Independent greedy reference: t_{p+1} = (t_p + t_{p-1}) mod 16."""
    seq = list(prompt)
    out = []
    for _ in range(n):
        nxt = (seq[-1] + seq[-2]) % VOCAB
        out.append(nxt)
        seq.append(nxt)
    return out


def _make_stub_engine(model) -> InferenceEngine:
    engine = InferenceEngine(model_name="stub")
    engine.tokenizer = type(
        "T",
        (),
        {
            "encode": staticmethod(
                lambda text, **kw: torch.tensor(
                    [[int(t) for t in text.split(",")]]
                )
            ),
            "decode": staticmethod(
                lambda ids, **kw: "".join(f"<{int(i)}>" for i in ids)
            ),
            "eos_token_id": None,
        },
    )()
    engine.local_partitioner = type("P", (), {"full_model": model})()
    return engine


# Periodic Fibonacci-mod-16 prompt.  With max_new_tokens=16 the last draft
# round proposes k=6 drafts and the token-budget clamp turns it into a partial
# accept — the only path that reaches the crop (unclamped rounds keep all KV
# entries).
_PROMPT_TEXT = "0,4,4,8,12,4,0,4"
_PROMPT_IDS = [0, 4, 4, 8, 12, 4, 0, 4]
_MAX_NEW = 16


def _run_with_round_capture():
    """Run one generation, returning (ids, rounds, model, strategy).

    ``rounds`` is one (k, accepted) pair per draft-verification round.
    """
    model = _RuleModel()
    engine = _make_stub_engine(model)
    strat = _PromptLookupStrategy(engine)

    rounds: list[tuple[int, int]] = []
    orig = _PromptLookupStrategy._accept_drafts

    def counting(self_inner, *a, **kw):
        r = orig(self_inner, *a, **kw)
        k = int(a[1].shape[1] if len(a) == 3 else a[2].shape[1])
        rounds.append((k, int(r)))
        return r

    import unittest.mock as mock

    with mock.patch.object(_PromptLookupStrategy, "_accept_drafts", counting):
        text = strat.generate(_PROMPT_TEXT, _MAX_NEW, 0.0, 1.0, 0)
    ids = [int(m) for m in re.findall(r"<(\d+)>", text)]
    return ids, rounds, model, strat


# ---------------------------------------------------------------------------
# 1. End-to-end: partial accept must complete, crop negatively, stay correct.
# ---------------------------------------------------------------------------


class TestPartialAcceptCropT5Semantics:
    # Note on the trigger: with this deterministic rule model a proposed draft
    # is always the model's own true continuation, so a RAW partial accept can
    # never occur.  The crop branch still fires exactly as in production: the
    # token-budget clamp turns a full raw accept into a partial one
    # (``accepted = min(accepted, target_len - prefix_len)`` < len(draft_list)),
    # which is the same ``accepted < len(draft_list)`` branch the buggy call
    # sat in.

    def test_partial_accept_round_completes_with_correct_output(self) -> None:
        """Fails on the pre-fix code with the exact ``crop`` positive-value
        ValueError; must pass once the relative-crop conversion is in."""
        ids, rounds, model, _ = _run_with_round_capture()

        # Non-vacuousness: the crop branch really executed, and the raw draft
        # accepts overran the budget (so a boundary round was clamped to a
        # partial accept).
        assert model.cache.crop_calls, "crop branch never executed"
        assert sum(acc for _, acc in rounds) > _MAX_NEW, (
            f"test premise lost: no budget-clamped draft round (rounds={rounds})"
        )
        # Output must track the plain greedy chain exactly.
        want = _rule_plain_chain(_PROMPT_IDS, _MAX_NEW)
        assert ids == want, (
            f"prompt-lookup diverged after partial accept: got {ids}, want {want}"
        )
        assert len(ids) == _MAX_NEW

    def test_every_round_attends_only_the_correct_prefix(self) -> None:
        """Per-round KV correctness: at each forward-call entry the attended
        history must be a prefix of the reference sequence (prompt + greedy
        chain; empty before the prefill).  A skipped crop leaves rejected
        draft tokens in the history (not a prefix); an over-crop drops
        accepted tokens (logits diverge downstream)."""
        _, _, model, _ = _run_with_round_capture()
        reference = _PROMPT_IDS + _rule_plain_chain(_PROMPT_IDS, _MAX_NEW)
        for cp in model.checkpoints:
            assert cp == reference[: len(cp)], (
                f"attended history is not a reference prefix: {cp}"
            )

    def test_crop_uses_negative_relative_deltas(self) -> None:
        """Every crop must pass a NEGATIVE relative count and leave exactly
        the emitted prefix in the cache (the boundary round keeps the full
        budget), never a positive absolute length."""
        ids, _, model, _ = _run_with_round_capture()

        crop_calls = model.cache.crop_calls
        assert crop_calls, "crop branch never executed"
        for before, delta in crop_calls:
            assert delta < 0, (
                f"transformers>=5 crop requires negative deltas, got {crop_calls}"
            )
            keep = before + delta
            assert keep == len(_PROMPT_IDS) + len(ids), (
                "crop kept the wrong entry count: "
                f"before={before}, delta={delta}, keep={keep}, "
                f"expected {len(_PROMPT_IDS) + len(ids)}"
            )

    def test_cache_length_matches_history_after_each_round(self) -> None:
        """The cache's remaining length must equal the model's valid history
        after every round (recorded per forward pass), so no rejected-token
        entries leak and none of the accepted prefix is dropped."""
        ids, _, model, _ = _run_with_round_capture()
        assert model.cache.recorded_lens, "no forward passes recorded"
        for recorded in model.cache.recorded_lens:
            assert recorded <= len(model.history)
        assert model.cache.get_seq_length() == len(model.history), (
            f"cache length {model.cache.get_seq_length()} != history "
            f"length {len(model.history)}"
        )
        assert len(model.history) == len(_PROMPT_IDS) + len(ids), (
            "final cache leaked or dropped tokens: "
            f"{len(model.history)} != {len(_PROMPT_IDS)} + {len(ids)}"
        )


# ---------------------------------------------------------------------------
# 2. Premise guard against the real transformers DynamicCache.
# ---------------------------------------------------------------------------


class TestRealDynamicCachePremise:
    def test_real_dynamiccache_crop_requires_negative_delta(self) -> None:
        cache_utils = pytest.importorskip("transformers.cache_utils")
        DynamicCache = cache_utils.DynamicCache

        params = inspect.signature(DynamicCache.crop).parameters
        if "tokens_to_remove" not in params:
            pytest.skip(
                "installed transformers uses legacy absolute crop(max_length)"
            )

        cache = DynamicCache()
        seq = 8
        cache.update(
            torch.zeros(1, 2, seq, 4), torch.zeros(1, 2, seq, 4), 0
        )
        assert cache.get_seq_length() == seq

        # This is precisely the pre-fix call shape: an absolute target
        # length, which is positive after any partial accept.
        keep = 5
        to_remove = cache.get_seq_length() - keep
        assert to_remove > 0
        with pytest.raises(ValueError, match="positive value"):
            cache.crop(to_remove)

        # The fixed code path converts to the relative, negative form.
        cache.crop(-to_remove)
        assert cache.get_seq_length() == keep, (
            "negative relative crop did not trim to the kept prefix"
        )
