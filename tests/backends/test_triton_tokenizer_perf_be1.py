"""Tests for PERF-BE-1 / CQ-019 / B-12: Triton tokenizer + decode loop.

Run with:  PYTHONPATH=src python -m pytest tests/backends/test_triton_tokenizer_perf_be1.py -v
(PYTHONPATH matters: the shared venv has an editable install pointing at
D:/distributed-llm/src which would otherwise shadow this worktree.)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_WORKTREE_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))

from distllm.backends.triton_backend import TritonNodeAdapter  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────


class FakeTokenizer:
    """Tiny real tokenizer: char->id via ord capped, id->char via chr."""

    def __init__(self):
        self.eos_token_id = 0
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return [ord(c) % 126 + 1 for c in text]  # 1..126, 0 reserved for EOS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 1) if 1 <= i <= 126 else "" for i in ids)


class FakeResults:
    def __init__(self, logits):
        self._logits = logits

    def as_numpy(self, name):
        return self._logits


class FakeClient:
    """Records each Infer call; returns logits peaking at token 66 ('B').

    FakeTokenizer maps id->char as chr(id-1), so id 65 decodes to '@'.
    Peak 66 decodes to 'A', keeping expectations readable.
    """

    def __init__(self, peak=66, vocab=256):
        self.infer_calls = 0
        self.batches = []
        self.peak = peak
        self.vocab = vocab

    def infer(self, **kwargs):
        self.infer_calls += 1
        inputs = kwargs["inputs"]
        batch = getattr(inputs[0], "_data", None)
        if batch is not None:
            self.batches.append(batch.copy())
            b, t = batch.shape
        else:
            b, t = 1, 1
        logits = np.zeros((b, t, self.vocab), dtype=np.float32)
        logits[:, -1, self.peak] = 1.0
        return FakeResults(logits)

    def close(self):
        pass


class _FakeInferInput:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self._data = None

    def set_data_from_numpy(self, arr):
        self._data = np.asarray(arr)


class _FakeInferRequestedOutput:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def triton_client_stubs(monkeypatch):
    """Stub tritonclient.http/grpc so the adapter's lazy imports work."""
    tc = ModuleType("tritonclient")
    http = ModuleType("tritonclient.http")
    grpc = ModuleType("tritonclient.grpc")
    for mod in (http, grpc):
        mod.InferInput = _FakeInferInput
        mod.InferRequestedOutput = _FakeInferRequestedOutput
    tc.http = http
    tc.grpc = grpc
    monkeypatch.setitem(sys.modules, "tritonclient", tc)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc)
    return http


def _make_adapter(tokenizer=None, client=None, **kwargs):
    adapter = TritonNodeAdapter(
        model_name="fake-model", protocol="http", tokenizer=tokenizer, **kwargs
    )
    adapter._client = client if client is not None else FakeClient()
    adapter._model_metadata = {
        "inputs": [{"name": "input_ids"}],
        "outputs": [{"name": "logits"}],
    }
    return adapter


# ── Tests ────────────────────────────────────────────────────────────


def test_generate_without_tokenizer_raises_clear_error(triton_client_stubs):
    import distllm.backends.triton_backend as _m

    assert str(_WORKTREE_SRC) in str(_m.__file__), (
        f"wrong distllm loaded: {_m.__file__}; run with PYTHONPATH=src"
    )
    adapter = _make_adapter(tokenizer=None)
    with pytest.raises(RuntimeError, match="real tokenizer"):
        adapter.generate("Hello", max_tokens=4)
    assert adapter._client.infer_calls == 0


def test_generate_encodes_once_and_extends_ids(triton_client_stubs):
    tokenizer = FakeTokenizer()
    client = FakeClient()
    adapter = _make_adapter(tokenizer=tokenizer, client=client)
    out = adapter.generate("Hi", max_tokens=5)
    assert tokenizer.encode_calls == 1
    assert client.infer_calls == 5
    assert out == "AAAAA"
    widths = [b.shape[1] for b in client.batches]
    assert widths == [2, 3, 4, 5, 6]


def test_generate_respects_eos(triton_client_stubs):
    tokenizer = FakeTokenizer()
    client = FakeClient(peak=0)  # EOS immediately
    adapter = _make_adapter(tokenizer=tokenizer, client=client)
    out = adapter.generate("Hi", max_tokens=10)
    assert out == ""
    assert client.infer_calls == 1


def test_batch_generate_shares_rpc_per_step(triton_client_stubs):
    tokenizer = FakeTokenizer()
    client = FakeClient()
    adapter = _make_adapter(tokenizer=tokenizer, client=client)
    outs = adapter.batch_generate(["Hi", "Yo", "Ok"], max_tokens=4)
    assert client.infer_calls == 4  # 3 seq x 4 steps -> 4 RPCs, not 12
    assert outs == ["AAAA", "AAAA", "AAAA"]
    assert all(b.shape[0] == 3 for b in client.batches)


def test_generate_stream_yields_incrementally(triton_client_stubs):
    tokenizer = FakeTokenizer()
    client = FakeClient()
    adapter = _make_adapter(tokenizer=tokenizer, client=client)
    chunks = list(adapter.generate_stream("Hi", max_tokens=3))
    assert chunks == ["A", "A", "A"]
    assert "".join(chunks) == adapter.generate("Hi", max_tokens=3)


def test_get_tokenizer_roundtrip(triton_client_stubs):
    tokenizer = FakeTokenizer()
    adapter = _make_adapter(tokenizer=tokenizer)
    assert adapter.get_tokenizer() is tokenizer


def test_no_ord_chr_fallback_in_source():
    """Guard: the byte-wise fallback must never come back."""
    src = (_WORKTREE_SRC / "distllm/backends/triton_backend.py").read_text(
        encoding="utf-8"
    )
    # Docstring/error mentions of the removal are fine; executable
    # fallback patterns are not.
    assert "ord(c) for c in" not in src
    assert "chr(token_id)" not in src
    assert "full_output[-512:]" not in src
