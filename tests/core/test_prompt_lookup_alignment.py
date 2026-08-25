"""W2-30: prompt-lookup acceptance alignment tests.

Proves and guards the +1 logit-row alignment fix in
``_PromptLookupStrategy._accept_drafts`` (docs/WAVE2-PLAN.md item 30;
follow-up flagged by B1-1 in docs/benchmark-results.md).

Background: a causal LM's logits row at absolute position ``p`` predicts
the token at position ``p + 1``.  In the cached verify pass, draft token
``i`` is fed at absolute position ``prefix_len + i``, so the model's
opinion about draft ``i`` lives in logits row ``i - 1`` of the verify pass
— or in the pre-verify pending row (``prev_logits``) for ``i == 0``.
The pre-fix code compared draft ``i`` against verify row ``i``, which is
the prediction for position ``prefix_len + i + 1``: a systematic off-by-one.

Consequences of the old comparison (both demonstrated below):
- True continuations were systematically rejected (depressed acceptance),
  unless the continuation happened to be locally constant.
- Worse, on partial acceptance the old code emitted tokens plain greedy
  decoding never would (the last accepted draft was duplicated as the
  "correction" token) — a genuine correctness hazard, not just a speed issue.

Three layers of proof:
1. Sentinel-row unit tests on ``_accept_drafts`` — each row's argmax encodes
   its identity, so the accepted count reveals exactly which row each draft
   was scored against, in both directions (missed accepts AND spurious
   accepts).  A small signature shim lets these tests fail on VALUES against
   the pre-fix code rather than raising TypeError, documenting the delta.
2. An interpretable rule-based stub model (next = (a+b) mod 16) with a
   hand-computed prompt where the misaligned algorithm provably emits tokens
   the plain greedy chain never contains; the engine must reproduce the
   plain chain exactly.
3. Real-model (TinyStories-1M) parity on repetitive prompts chosen to fire
   draft rounds (asserted non-vacuous via a round counter): prompt-lookup
   output must equal plain local output and raw HF ``generate()`` output
   byte-for-byte.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest
import torch

from distllm.core.inference_engine import (
    InferenceEngine,
    _PromptLookupStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V = 16


def _one_hot_row(token: int, weight: float = 100.0) -> torch.Tensor:
    """[V] logits whose argmax is exactly *token*."""
    row = torch.zeros(V)
    row[token] = weight
    return row


def _make_stub_engine(model) -> InferenceEngine:
    engine = InferenceEngine(model_name="stub")
    engine.tokenizer = type(
        "T",
        (),
        {
            "encode": staticmethod(
                lambda text, **kw: torch.tensor([[int(t) for t in text.split(",")]])
            ),
            "decode": staticmethod(
                lambda ids, **kw: "".join(f"<{int(i)}>" for i in ids)
            ),
            "eos_token_id": None,
        },
    )()
    engine.local_partitioner = type("P", (), {"full_model": model})()
    return engine


def _call_accept(strat, prev, verify, drafts, temperature):
    """Call ``_accept_drafts`` under either the pre-fix or post-fix signature.

    Exists so the sentinel tests below fail on VALUES when run against the
    pre-fix code (showing accepted=0 where the textbook answer is nonzero,
    and vice versa) instead of erroring out with a TypeError.  Once the fix
    landed, only the 4-parameter branch ever runs.
    """
    n_params = len(inspect.signature(strat._accept_drafts).parameters)
    if n_params == 3:  # pre-fix: (verify_logits, draft_ids, temperature)
        return strat._accept_drafts(verify, drafts, temperature)
    return strat._accept_drafts(prev, verify, drafts, temperature)


# ---------------------------------------------------------------------------
# 1. Sentinel-row unit tests: which row is each draft scored against?
#
# Layout for every case: prev_logits has argmax 2; verify rows carry their
# own sentinel argmaxes.  Draft tokens are chosen so the textbook answer and
# the misaligned answer differ.
# ---------------------------------------------------------------------------


class TestAcceptDraftsRowAlignment:
    def setup_method(self) -> None:
        self.engine = InferenceEngine(model_name="stub")
        self.strat = _PromptLookupStrategy(self.engine)

    def _prev(self, token: int = 2) -> torch.Tensor:
        return _one_hot_row(token).unsqueeze(0)  # [1, V]

    def _verify(self, argmaxes: list[int]) -> torch.Tensor:
        return torch.stack([_one_hot_row(t) for t in argmaxes]).unsqueeze(0)  # [1, k, V]

    def test_greedy_true_continuation_fully_accepted(self) -> None:
        """True continuation [2,3,4]: prev predicts 2, row0 predicts 3,
        row1 predicts 4, row2 predicts 5 -> all three drafts accepted."""
        prev = self._prev(2)
        verify = self._verify([3, 4, 5])
        drafts = torch.tensor([[2, 3, 4]])
        assert _call_accept(self.strat, prev, verify, drafts, 0.0) == 3

    def test_greedy_first_draft_scored_against_pending_row(self) -> None:
        """Draft [2, ...] matches prev_logits (the pending prediction for the
        first draft position), even though verify row 0 predicts something
        else entirely.  Textbook answer: accept the first draft.
        Pre-fix code compared draft 0 against row 0 and accepted NOTHING."""
        prev = self._prev(2)
        verify = self._verify([5, 3, 6])
        drafts = torch.tensor([[2, 9, 9]])
        assert _call_accept(self.strat, prev, verify, drafts, 0.0) == 1

    def test_greedy_shifted_match_is_not_accepted(self) -> None:
        """Draft 0 == argmax(verify row 0) but != argmax(prev): the match is
        an artifact of looking one position ahead.  Textbook answer: reject.
        Pre-fix code accepted it (spurious acceptance)."""
        prev = self._prev(2)
        verify = self._verify([5, 3, 6])
        drafts = torch.tensor([[5, 9, 9]])
        assert _call_accept(self.strat, prev, verify, drafts, 0.0) == 0

    def test_greedy_single_draft_uses_only_the_pending_row(self) -> None:
        """k=1: there are no usable verify rows; the sole draft must be scored
        against prev_logits alone."""
        prev = self._prev(2)
        verify = self._verify([0])  # argmax 0, irrelevant under correct alignment
        drafts = torch.tensor([[2]])
        assert _call_accept(self.strat, prev, verify, drafts, 0.0) == 1

    def test_sampled_mode_scores_each_draft_against_its_own_position(self) -> None:
        """Sampled path: draft 0 is near-certain under prev (accept), draft 1
        is near-impossible under verify row 0 (reject) -> exactly 1 accepted,
        deterministically, with the aligned rows."""
        prev = _one_hot_row(2, weight=1000.0).unsqueeze(0)
        verify = torch.stack([
            _one_hot_row(7, weight=1000.0),  # p(draft=3) ~ 0 here
            _one_hot_row(1),
        ]).unsqueeze(0)
        drafts = torch.tensor([[2, 3]])
        assert _call_accept(self.strat, prev, verify, drafts, 0.7) == 1


# ---------------------------------------------------------------------------
# 2. Rule-model end-to-end: the misaligned algorithm provably leaves the
#    greedy chain; the fixed engine must not.
# ---------------------------------------------------------------------------


class _RuleCache:
    """Duck-typed DynamicCache: total length == owning model's history."""

    def __init__(self, model: "_RuleModel") -> None:
        self._model = model
        self.recorded_lens: list[int] = []

    def get_seq_length(self) -> int:
        return len(self._model.history)

    def crop(self, max_length: int) -> None:
        del self._model.history[max_length:]
        self.recorded_lens = [min(n, max_length) for n in self.recorded_lens]

    def record(self, n_in: int) -> None:
        self.recorded_lens.append(self.get_seq_length())


class _Out:
    def __init__(self, logits: torch.Tensor, past) -> None:
        self.logits = logits
        self.past_key_values = past


class _RuleModel:
    """Causal LM stand-in: next token = (last + second-last) mod 16, encoded
    as a +100 logit spike (everything else 0).  Logits are a pure function of
    the attended prefix, so KV threading mistakes are impossible to hide."""

    VOCAB = 16

    def __init__(self) -> None:
        self.cache = _RuleCache(self)
        self.history: list[int] = []
        self.call_input_lens: list[int] = []

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        seq = input_ids[0].tolist()
        self.call_input_lens.append(len(seq))
        cache = past_key_values if past_key_values is not None else self.cache
        start = len(self.history)
        self.history.extend(seq)
        rows = []
        for i in range(len(seq)):
            h = self.history[: start + i + 1]
            row = torch.zeros(self.VOCAB)
            nxt = (
                (h[-2] + h[-1]) % self.VOCAB if len(h) >= 2 else h[-1]
            )
            row[nxt] = 100.0
            rows.append(row)
        cache.record(len(seq))
        return _Out(torch.stack(rows).unsqueeze(0), cache)


def _rule_plain_chain(prompt: list[int], n: int) -> list[int]:
    """Independent greedy reference: t_{p+1} = (t_p + t_{p-1}) mod 16."""
    seq = list(prompt)
    out = []
    for _ in range(n):
        nxt = (seq[-1] + seq[-2]) % _RuleModel.VOCAB
        out.append(nxt)
        seq.append(nxt)
    return out


# Hand-computed triggering prompt.  Suffix [0,3,7,0] recurs at start=1, so a
# draft [3,7,0] is proposed.  Under the MISALIGNED rule, draft 0 (=3) matches
# the prediction for the NEXT position ((0+3)%16=3) and is spuriously
# accepted, after which the "correction" row re-emits 3: the output begins
# [3, 3, ...].  The true greedy chain begins [7, 7, 14, ...] ((7+0),(0+7),...).
_PROMPT_TEXT = "5,0,3,7,0,3,7,0"

# Periodic prompt (Fibonacci mod 16, cycle 0,4,4,8,12,4): after three
# fallback tokens the sequence is [0,4,4,8,12,4,0,4,4,8,12]; its last-4
# suffix [4,4,8,12] matches at start=1, proposing the SIX-token draft
# [4,0,4,4,8,12] — which is exactly the true continuation, so a full
# k=6 accept fires.  With max_new_tokens=8 only 5 slots remain: this is
# the budget-boundary scenario.
_PERIODIC_PROMPT_TEXT = "0,4,4,8,12,4,0,4"


class TestRuleModelEndToEnd:
    def test_engine_tracks_plain_greedy_chain_through_draft_rounds(
        self, monkeypatch,
    ) -> None:
        model = _RuleModel()
        engine = _make_stub_engine(model)
        strat = _PromptLookupStrategy(engine)

        calls = {"rounds": 0, "accepted": 0, "proposed": 0}
        orig = _PromptLookupStrategy._accept_drafts

        def counting(self_inner, *a, **kw):
            r = orig(self_inner, *a, **kw)
            calls["rounds"] += 1
            calls["accepted"] += int(r)
            calls["proposed"] += int(a[1].shape[1] if len(a) == 3 else a[2].shape[1])
            return r

        monkeypatch.setattr(_PromptLookupStrategy, "_accept_drafts", counting)

        got_ids = [
            int(m) for m in re.findall(
                r"<(\d+)>", strat.generate(_PROMPT_TEXT, 12, 0.0, 1.0, 0))
        ]

        # Non-vacuousness: at least one draft round actually fired.
        assert calls["rounds"] >= 1, "prompt never triggered a draft round"

        want = _rule_plain_chain([5, 0, 3, 7, 0, 3, 7, 0], 12)
        assert got_ids == want, (
            "prompt-lookup diverged from the plain greedy chain "
            f"(draft rounds={calls['rounds']}, "
            f"accepted={calls['accepted']}/{calls['proposed']}): "
            f"got {got_ids}, want {want}"
        )

    def test_engine_output_equals_plain_local_strategy(self) -> None:
        engine = _make_stub_engine(_RuleModel())
        lookup = _PromptLookupStrategy(engine).generate(
            _PROMPT_TEXT, 12, 0.0, 1.0, 0)
        plain = "".join(
            engine.tokenizer.decode([t]) for t in _iter_plain(
                engine, _PROMPT_TEXT, 12))
        assert lookup == plain

    def test_stream_yields_every_accepted_token(self) -> None:
        """Streaming fidelity: with correct alignment a full-accept round can
        accept k>1 drafts; every accepted token must be yielded (the
        pre-W2-30 loop yielded only the round's last token)."""
        engine = _make_stub_engine(_RuleModel())
        strat = _PromptLookupStrategy(engine)
        events = list(strat.generate_stream(_PROMPT_TEXT, 12, 0.0, 1.0, 0))
        joined = strat.generate(_PROMPT_TEXT, 12, 0.0, 1.0, 0)
        assert len(events) == 12, (
            f"expected one event per generated token, got {len(events)}"
        )
        assert "".join(events) == joined

    def test_full_accept_round_respects_token_budget(self) -> None:
        """A full-accept draft round must never push output past
        max_new_tokens (overshoot became reachable once alignment made
        k>1 acceptance possible; parity vs generate() requires the cap).

        Scenario: the periodic Fibonacci-mod-16 prompt fires a full SIX-token
        accept with only 5 budget slots remaining."""
        model = _RuleModel()
        engine = _make_stub_engine(model)
        strat = _PromptLookupStrategy(engine)

        rounds = {"full_accepts": 0}
        orig = _PromptLookupStrategy._accept_drafts

        def counting(self_inner, *a, **kw):
            r = orig(self_inner, *a, **kw)
            k = int(a[1].shape[1] if len(a) == 3 else a[2].shape[1])
            if r == k:
                rounds["full_accepts"] += 1
            return r

        import unittest.mock as mock
        with mock.patch.object(
            _PromptLookupStrategy, "_accept_drafts", counting,
        ):
            ids = [
                int(m) for m in re.findall(r"<(\d+)>", strat.generate(
                    _PERIODIC_PROMPT_TEXT, 8, 0.0, 1.0, 0))
            ]
        assert rounds["full_accepts"] >= 1, (
            "test premise lost: no full-accept round fired"
        )
        assert len(ids) == 8, f"budget overshoot: emitted {len(ids)} tokens"
        want = _rule_plain_chain([0, 4, 4, 8, 12, 4, 0, 4], 8)
        assert ids == want

    def test_periodic_prompt_stream_events_match_tokens(self) -> None:
        """On the full-accept scenario the stream emits exactly one event per
        generated token (12 tokens -> 12 events), and matches generate()."""
        events = list(_PromptLookupStrategy(_make_stub_engine(_RuleModel()))
                      .generate_stream(_PERIODIC_PROMPT_TEXT, 8, 0.0, 1.0, 0))
        joined = _PromptLookupStrategy(_make_stub_engine(_RuleModel())).generate(
            _PERIODIC_PROMPT_TEXT, 8, 0.0, 1.0, 0)
        assert len(events) == 8, f"got {len(events)} events for 8 tokens"
        assert "".join(events) == joined


def _iter_plain(engine: InferenceEngine, prompt_text: str, n: int) -> list[int]:
    input_ids = engine.tokenizer.encode(prompt_text)
    return list(engine._iter_local_tokens(
        input_ids, n, temperature=0.0, top_p=1.0, top_k=0,
        stop_token_ids=set(),
    ))


# ---------------------------------------------------------------------------
# 3. Real-model parity (auto-skips without the local HF cache).
# ---------------------------------------------------------------------------

_HF_SNAPSHOT = Path(os.environ.get(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface"),
)) / "hub" / "models--roneneldan--TinyStories-1M"


def _real_model_available() -> bool:
    env_flag = os.environ.get("DISTLLM_KV_REUSE_REAL_MODEL")
    if env_flag is not None:
        return env_flag not in ("0", "", "false")
    return _HF_SNAPSHOT.exists()


@pytest.mark.skipif(not _real_model_available(), reason="TinyStories-1M not in local HF cache")
class TestRealModelAlignedParity:
    """Repetitive prompts engineered to fire draft rounds; prompt-lookup must
    stay byte-identical to plain local decode and raw HF greedy generate."""

    PROMPTS = [
        "the bear said the bear said the bear said",
        "Lily said hello to Ben. Lily said hello to Ben.",
        "Once upon a time there was a little girl named Lily. Lily loved to",
        "and then and then and then and then",
    ]

    @pytest.fixture(scope="class")
    def engine(self):
        eng = InferenceEngine(model_name="roneneldan/TinyStories-1M", dtype="float32")
        eng.load_local_model()
        return eng

    def test_parity_with_nonvacuous_draft_rounds(self, engine) -> None:
        stats = {"rounds": 0, "proposed": 0, "accepted": 0}
        orig = _PromptLookupStrategy._accept_drafts

        def counting(self_inner, *a, **kw):
            r = orig(self_inner, *a, **kw)
            stats["rounds"] += 1
            stats["accepted"] += int(r)
            stats["proposed"] += int(a[1].shape[1] if len(a) == 3 else a[2].shape[1])
            return r

        _PromptLookupStrategy._accept_drafts = counting
        try:
            for p in self.PROMPTS:
                plain = engine._generate_local(p, 48, 0.0, 1.0, 0)
                lookup = _PromptLookupStrategy(engine).generate(p, 48, 0.0, 1.0, 0)
                assert lookup == plain, f"prompt-lookup drift on prompt {p!r}"

                ids = engine.tokenizer.encode(p, return_tensors="pt").to(
                    next(engine.local_partitioner.full_model.parameters()).device)
                with torch.no_grad():
                    out = engine.local_partitioner.full_model.generate(
                        ids, max_new_tokens=48, do_sample=False,
                        pad_token_id=engine.tokenizer.eos_token_id or 0,
                    )
                hf = engine.tokenizer.decode(
                    out[0, ids.shape[-1]:], skip_special_tokens=True)
                assert lookup == hf, f"HF parity drift on prompt {p!r}"
        finally:
            _PromptLookupStrategy._accept_drafts = orig

        assert stats["rounds"] >= 1, (
            "no draft rounds fired — parity would be vacuous"
        )
