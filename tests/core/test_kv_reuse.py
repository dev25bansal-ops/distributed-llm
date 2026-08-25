"""KV-reuse correctness tests for the local decode paths.

Guards the prefill-once + ``past_key_values`` threading fix in
``inference_engine.py``:

- ``_iter_local_tokens`` (shared by ``_LocalStrategy.generate_stream`` and
  ``InferenceEngine._generate_local``)
- ``_PromptLookupStrategy._generate_tokens``

Hermetic tests use a deterministic stub model whose logits are a pure
function of the full attended prefix, so any KV-threading mistake changes the
greedy output and fails.  Naive full-reforward references implement the OLD
decode algorithms independently and must produce identical outputs.  The
real-model test runs TinyStories-1M from the local HF cache and skips when
unavailable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest
import torch

from distllm.core.inference_engine import (
    InferenceEngine,
    _LocalStrategy,
    _PromptLookupStrategy,
)


# ---------------------------------------------------------------------------
# Deterministic stub model: logits at absolute position t are a pure function
# of tokens [0..t].  A causal LM satisfies this; encoding it makes KV-cache
# threading observable — if the cached path ever attends over the wrong
# prefix, its logits diverge from the full-reforward reference.
# ---------------------------------------------------------------------------

VOCAB = 64


def _prefix_logits(seq: list[int]) -> torch.Tensor:
    """[VOCAB] logits, deterministic pure function of *seq*."""
    digest = hashlib.sha256(bytes(seq)).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
    return torch.randn(VOCAB, generator=gen)


class _CacheStub:
    """Minimal duck-typed DynamicCache: tracks total cached length.

    ``crop`` truncates the owning model's token history too, mirroring how
    DynamicCache.crop drops KV entries.
    """

    def __init__(self, model: "_SeqFunctionModel") -> None:
        self._model = model
        self.recorded_lens: list[int] = []

    def get_seq_length(self) -> int:
        return len(self._model.history)

    def crop(self, max_length: int) -> None:
        del self._model.history[max_length:]
        self.recorded_lens = [
            min(n, max_length) for n in self.recorded_lens
        ]

    def record(self, n_in: int) -> None:
        self.recorded_lens.append(self.get_seq_length())


class _Out:
    def __init__(self, logits: torch.Tensor, past) -> None:
        self.logits = logits
        self.past_key_values = past


class _SeqFunctionModel:
    """Causal-LM stand-in accumulating the full token history."""

    def __init__(self) -> None:
        self.cache = _CacheStub(self)
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
        rows = [_prefix_logits(self.history[: start + i + 1]) for i in range(len(seq))]
        logits = torch.stack(rows).unsqueeze(0)  # [1, T, V]
        cache.record(len(seq))
        return _Out(logits, cache)


# ---------------------------------------------------------------------------
# Stub engine wiring
# ---------------------------------------------------------------------------


def _stub_encode(text: str) -> torch.Tensor:
    d = hashlib.sha256(text.encode()).digest()
    return torch.tensor([[d[0] % VOCAB, d[1] % VOCAB, d[2] % VOCAB]])


def _decode_stub_ids(text: str) -> list[int]:
    """Inverse of the stub tokenizer decode (ids rendered as ``<n>``)."""
    return [int(m) for m in re.findall(r"<(\d+)>", text)]


def _make_engine(model=None) -> InferenceEngine:
    engine = InferenceEngine(model_name="stub")
    engine.tokenizer = type(
        "T",
        (),
        {
            "encode": staticmethod(lambda text, **kw: _stub_encode(text)),
            "decode": staticmethod(
                lambda ids, **kw: "".join(f"<{i}>" for i in ids)
            ),
            "eos_token_id": None,
        },
    )()
    engine.local_partitioner = type(
        "P", (), {"full_model": model or _SeqFunctionModel()}
    )()
    return engine


# ---------------------------------------------------------------------------
# Reference implementations of the OLD (full-reforward) algorithms,
# written directly against the pure logits function.
# ---------------------------------------------------------------------------


def _naive_reforward_greedy(input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Old _generate_local / stream loop: full-sequence logits every step."""
    ids_list = input_ids[0].tolist()
    out: list[int] = []
    for _ in range(max_new_tokens):
        nxt = int(_prefix_logits(ids_list).argmax())
        out.append(nxt)
        ids_list.append(nxt)
    return out


def _find_match_ref(ids_list: list[int], min_match: int = 4, max_draft: int = 10):
    n = len(ids_list)
    if n < min_match + 1:
        return None
    suffix = ids_list[-min_match:]
    for start in range(n - min_match - 1, 0, -1):
        if ids_list[start:start + min_match] == suffix:
            end = start + min_match
            avail = n - end
            if avail > 0:
                return end, min(avail, max_draft)
    return None


def _naive_prompt_lookup_greedy(
    input_ids: torch.Tensor, max_new_tokens: int,
) -> list[int]:
    """Old _PromptLookupStrategy greedy loop (full re-forward every round),
    with the W2-30-corrected acceptance alignment: draft i is scored against
    the prediction one position earlier (pending row for i == 0, verify row
    i - 1 otherwise), since a causal LM's row at position p predicts p + 1."""
    ids_list = input_ids[0].tolist()
    out: list[int] = []
    while len(out) < max_new_tokens:
        match = _find_match_ref(ids_list)
        if match is not None:
            match_end, num_draft = match
            draft = ids_list[match_end:match_end + num_draft]
        else:
            draft = None

        if draft:
            n = len(ids_list)
            k = len(draft)
            full = ids_list + draft
            # Pending row: prediction for position n (= draft 0's slot).
            # Rows n .. n+k-2 of the full-sequence forward predict positions
            # n+1 .. n+k-1 (= drafts 1 .. k-1's slots).
            preds = [_prefix_logits(full[:n])] + [
                _prefix_logits(full[: n + i]) for i in range(1, k)
            ]
            accepted = 0
            for i in range(k):
                if int(preds[i].argmax()) != draft[i]:
                    break
                accepted += 1
            # Budget cap (W2-30): never emit more than max_new_tokens.
            accepted = min(accepted, max_new_tokens - len(out))
            if accepted == k:
                out.extend(draft)
                ids_list.extend(draft)
                continue
            out.extend(draft[:accepted])
            ids_list.extend(draft[:accepted])
            if len(out) >= max_new_tokens:
                break
            # Correction: prediction for the position after the last kept
            # token — verify row accepted - 1 (or the pending row if none).
            correction = (
                _prefix_logits(full[: n + accepted]) if accepted > 0
                else _prefix_logits(ids_list[:n])
            )
            nxt = int(correction.argmax())
            out.append(nxt)
            ids_list.append(nxt)
        else:
            nxt = int(_prefix_logits(ids_list).argmax())
            out.append(nxt)
            ids_list.append(nxt)
    return out


# ---------------------------------------------------------------------------
# _iter_local_tokens tests
# ---------------------------------------------------------------------------


class TestIterLocalTokensKvThreading:
    """_iter_local_tokens: prefill once, then single-token forwards."""

    def test_prefill_once_then_single_token_forwards(self) -> None:
        model = _SeqFunctionModel()
        engine = _make_engine(model)
        input_ids = _stub_encode("abc")
        toks = list(engine._iter_local_tokens(
            input_ids, 6, temperature=0.0, top_p=1.0, top_k=0,
            stop_token_ids=set(),
        ))
        assert len(toks) == 6
        # First call covers the whole prompt; every later call exactly one token.
        assert model.call_input_lens[0] == input_ids.shape[-1]
        assert all(n == 1 for n in model.call_input_lens[1:])
        assert len(model.call_input_lens) == 6

    def test_same_cache_object_threaded_and_growing(self) -> None:
        model = _SeqFunctionModel()
        engine = _make_engine(model)
        input_ids = _stub_encode("abc")
        list(engine._iter_local_tokens(
            input_ids, 4, temperature=0.0, top_p=1.0, top_k=0,
            stop_token_ids=set(),
        ))
        # Cache saw the prompt once, then one token per decode step.  Four
        # generated tokens = prefill(3) + three single-token forwards.
        assert model.cache.get_seq_length() == 6
        assert model.cache.recorded_lens == [3, 4, 5, 6]

    def test_greedy_output_matches_full_reforward_reference(self) -> None:
        engine = _make_engine()
        input_ids = _stub_encode("kv-reuse probe")
        got = list(engine._iter_local_tokens(
            input_ids, 12, temperature=0.0, top_p=1.0, top_k=0,
            stop_token_ids=frozenset(),  # no EOS configured on this stub
        ))
        want = _naive_reforward_greedy(input_ids, 12)
        assert got == want

    def test_stop_token_breaks_loop(self) -> None:
        engine = _make_engine()
        input_ids = _stub_encode("stop early")
        ref = _naive_reforward_greedy(input_ids, 3)
        stop = {ref[0]}
        toks = list(engine._iter_local_tokens(
            input_ids, 10, temperature=0.0, top_p=1.0, top_k=0,
            stop_token_ids=stop,
        ))
        assert toks[0] == ref[0]
        assert len(toks) == 1

    def test_logit_bias_applied(self) -> None:
        engine = _make_engine()
        input_ids = _stub_encode("bias me")
        toks = list(engine._iter_local_tokens(
            input_ids, 3, temperature=0.0, top_p=1.0, top_k=0,
            logit_bias={7: 1e9}, stop_token_ids=set(),
        ))
        assert toks == [7, 7, 7]

    def test_stream_and_generate_paths_agree(self) -> None:
        prompt = "hello world"
        stream_text = "".join(_LocalStrategy(_make_engine()).generate_stream(
            prompt, 8, 0.0, 1.0, 0))
        generate_text = _make_engine()._generate_local(prompt, 8, 0.0, 1.0, 0)
        assert stream_text == generate_text


# ---------------------------------------------------------------------------
# _PromptLookupStrategy tests
# ---------------------------------------------------------------------------


class TestPromptLookupKvReuse:
    """Cached verify/fallback passes with preserved accept semantics."""

    def test_prefill_once_then_small_forwards(self) -> None:
        model = _SeqFunctionModel()
        engine = _make_engine(model)
        strat = _PromptLookupStrategy(engine)
        out = strat.generate("a b c d e f g h", 10, 0.0, 1.0, 0)
        assert isinstance(out, str)
        lens = model.call_input_lens
        # First call is the full prompt (3 stub tokens); later calls carry
        # only drafts (bounded by _max_draft) or single correction/fallback
        # tokens — never the full regenerated sequence.
        assert lens[0] == 3
        assert all(0 < n <= strat._max_draft for n in lens[1:])

    def test_greedy_output_matches_naive_prompt_lookup(self) -> None:
        engine = _make_engine()
        strat = _PromptLookupStrategy(engine)
        prompt_text = "x" * 24  # repetitive -> repeated n-gram matches
        input_ids = _stub_encode(prompt_text)
        got_ids = _decode_stub_ids(strat.generate(prompt_text, 20, 0.0, 1.0, 0))
        want_ids = _naive_prompt_lookup_greedy(input_ids, 20)
        assert got_ids == want_ids

    def test_generate_matches_stream(self) -> None:
        # Fresh model per generation: the stub accumulates token history, and
        # a real model would likewise start each request with an empty cache.
        joined = _PromptLookupStrategy(_make_engine()).generate(
            "ababab ababab", 15, 0.0, 1.0, 0)
        streamed = "".join(_PromptLookupStrategy(_make_engine()).generate_stream(
            "ababab ababab", 15, 0.0, 1.0, 0))
        assert joined == streamed


# ---------------------------------------------------------------------------
# Real-model correctness (auto-skips without the local HF cache).
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
class TestRealModelGreedyParity:
    """Fixed-prompt greedy parity vs raw HF generate() on a real model."""

    PROMPTS = [
        "Once upon a time, there was a",
        "The quick brown fox jumps over",
        "One day, Lily and Ben went to",
        "The little robot wanted to",
    ]

    @pytest.fixture(scope="class")
    def engine(self):
        eng = InferenceEngine(model_name="roneneldan/TinyStories-1M", dtype="float32")
        eng.load_local_model()
        return eng

    def _hf_greedy(self, engine, prompt: str, n: int) -> str:
        ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(
            next(engine.local_partitioner.full_model.parameters()).device)
        with torch.no_grad():
            out = engine.local_partitioner.full_model.generate(
                ids, max_new_tokens=n, do_sample=False,
                pad_token_id=engine.tokenizer.eos_token_id or 0,
            )
        return engine.tokenizer.decode(
            out[0, ids.shape[-1]:], skip_special_tokens=True)

    def test_stream_matches_hf_greedy(self, engine) -> None:
        for p in self.PROMPTS:
            got = "".join(engine.generate_stream(p, max_new_tokens=32, temperature=0.0))
            want = self._hf_greedy(engine, p, 32)
            assert got == want, f"stream drift on prompt {p!r}"

    def test_generate_matches_hf_greedy(self, engine) -> None:
        for p in self.PROMPTS:
            got = engine._generate_local(p, 32, 0.0, 1.0, 0)
            want = self._hf_greedy(engine, p, 32)
            assert got == want, f"_generate_local drift on prompt {p!r}"

    def test_prompt_lookup_matches_plain_local(self, engine) -> None:
        for p in self.PROMPTS:
            plain = engine._generate_local(p, 32, 0.0, 1.0, 0)
            lookup = _PromptLookupStrategy(engine).generate(p, 32, 0.0, 1.0, 0)
            assert lookup == plain, f"prompt-lookup drift on prompt {p!r}"
