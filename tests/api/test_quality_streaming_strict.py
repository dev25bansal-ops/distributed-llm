"""Strict, xfail-free behavioural tests for SSE streaming.

These tests replace the stale ``xfail`` markers in
``tests/api/test_chat_streaming.py`` and ``tests/api/test_completion.py``.
The xfails were added for "``_stream_response`` accesses ``request.state`` on
a Pydantic model instead of a FastAPI Request"; that crash is fixed in
``src/distllm/api/streaming.py`` (the tenant lookup is now guarded with
``getattr(request, 'state', None)``) and both route handlers pass the
Pydantic body into the generator, so the SSE contract below holds today.

Known, still-open bug (NOT asserted here): every content chunk carries
``"usage": {"_cost": ...}`` unconditionally, because
``streaming.py::_stream_response`` injects the streaming-cost event into
``chunk.usage`` for every token regardless of ``stream_options``. So
"usage is absent unless ``stream_options.include_usage`` is true" does not
hold; only the ``include_usage=True`` direction is asserted below.

Fixtures mirror the self-contained helpers in the two legacy streaming test
files so this file stays independent of edits to them.
"""

import json
import os
import re
import secrets
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from the legacy streaming test files)
# ---------------------------------------------------------------------------

# Stack of env snapshots taken by _make_client(), consumed by _cleanup_auth().
_SAVED_ENV: list[dict[str, str | None]] = []


def _make_client():
    """Install a per-test API key, remembering whatever was there before.

    Two ordering hazards are handled here:

    * The previous version only *popped* API_KEY on cleanup, so a value
      installed by an earlier module was destroyed rather than restored.
    * ``API_KEYS`` (a JSON list) takes precedence over ``API_KEY`` in
      ApiKeyStore._load. Three other test modules set ``API_KEYS`` and never
      unset it, so once they ran, the per-test ``API_KEY`` was ignored and
      every request in this file 401'd. ``API_KEYS`` is cleared for the
      duration of the test and restored afterwards.

    Callers must pair this with _cleanup_auth(), which restores both.
    """
    _SAVED_ENV.append({
        "API_KEY": os.environ.get("API_KEY"),
        "API_KEY_WAS_SET": os.environ.get("API_KEY_WAS_SET"),
        "API_KEYS": os.environ.get("API_KEYS"),
        "API_KEYS_FILE": os.environ.get("API_KEYS_FILE"),
    })
    for name in ("API_KEY_WAS_SET", "API_KEYS", "API_KEYS_FILE"):
        os.environ.pop(name, None)
    test_api_key = secrets.token_urlsafe(32)
    os.environ["API_KEY"] = test_api_key
    reset_api_key_store()
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {test_api_key}"
    return client


def _cleanup_auth():
    """Restore the environment _make_client() replaced, then reset the store."""
    saved = _SAVED_ENV.pop() if _SAVED_ENV else {}
    for name in ("API_KEY", "API_KEY_WAS_SET", "API_KEYS", "API_KEYS_FILE"):
        prior = saved.get(name)
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior
    reset_api_key_store()


def make_mock_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None

    def encode_fn(text, **kwargs):
        tokens = [1, 2, 3, 4, 5]
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn

    def decode_side_effect(tokens, **kwargs):
        if isinstance(tokens, int):
            token_list = [tokens]
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            token_list = tokens.tolist()
        return " ".join(f"tok-{t}" for t in token_list)

    coord.tokenizer.decode.side_effect = decode_side_effect
    coord.tokenizer.eos_token_id = 0
    coord.generate.return_value = "Hello! This is a test response."

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord.list_models.return_value = ["test-model"]
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._model_router = None
    coord._shutting_down = False
    coord.tokenizer.chat_template = None
    return coord


_TOKEN_RE = re.compile(r"^tok-\d+$")


def _sse_payloads(resp):
    """Parse the SSE body into (json_chunks, raw_data_lines).

    ``json_chunks`` excludes the ``[DONE]`` sentinel; ``raw_data_lines``
    keeps every line starting with ``data: `` for framing assertions.
    """
    raw = [ln for ln in resp.text.split("\n") if ln.startswith("data: ")]
    chunks = [json.loads(ln[len("data: "):]) for ln in raw if ln != "data: [DONE]"]
    return chunks, raw


def _content_deltas(chunks):
    """Concatenated assistant content from chat-completion chunk deltas."""
    return [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c.get("choices") and c["choices"][0].get("delta", {}).get("content")
    ]


def _text_deltas(chunks):
    """Concatenated completion text from text-completion chunks."""
    return [
        c["choices"][0].get("text")
        for c in chunks
        if c.get("choices") and c["choices"][0].get("text")
    ]


class TestChatStreamingStrict:
    """Strict SSE contract for POST /v1/chat/completions with stream=true."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _stream(self, **extra):
        payload = {
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "stream": True,
        }
        payload.update(extra)
        return self.client.post("/v1/chat/completions", json=payload)

    def test_stream_status_and_content_type(self):
        resp = self._stream()
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_role_event_is_first_chunk(self):
        chunks, _ = _sse_payloads(self._stream())
        assert chunks, "stream produced no JSON chunks before [DONE]"
        first = chunks[0]
        assert first["object"] == "chat.completion.chunk"
        delta = first["choices"][0]["delta"]
        assert delta.get("role") == "assistant"
        assert "content" not in delta or not delta["content"]

    def test_stream_content_deltas_assemble_into_generated_text(self):
        chunks, _ = _sse_payloads(self._stream())
        deltas = _content_deltas(chunks)
        assert len(deltas) >= 1, "no content deltas emitted"
        # Every delta is a decoded token from the mock generator; joining
        # them must reconstruct the generated text exactly (no dropped or
        # duplicated tokens between buffer and SSE frame).
        for d in deltas:
            assert _TOKEN_RE.match(d), f"unexpected delta text: {d!r}"
        # With max_tokens=5 the generator cannot emit more than 5 tokens; the
        # assembled client-visible text is the in-order concatenation of the
        # decoded tokens with no frame-level truncation or duplication.
        assert len(deltas) <= 5
        assembled = "".join(deltas)
        assert assembled.count("tok-") == len(deltas)

    def test_stream_terminates_with_done_signal(self):
        resp = self._stream()
        _, raw = _sse_payloads(resp)
        assert raw[-1] == "data: [DONE]"
        assert raw.count("data: [DONE]") == 1
        # SSE framing: every event line — including the [DONE] sentinel — must
        # be immediately followed by a blank line. Checking line-by-line (not
        # just .endswith) so a lost inter-event separator is caught.
        lines = resp.text.split("\n")
        data_line_idx = [i for i, ln in enumerate(lines) if ln.startswith("data:")]
        assert data_line_idx, "stream body has no data lines"
        for i in data_line_idx:
            assert i + 1 < len(lines) and lines[i + 1] == "", (
                f"event at line {i} not terminated by a blank line: {lines[i]!r}"
            )
        assert resp.text.endswith("data: [DONE]\n\n")

    def test_stream_finish_reason_present_exactly_once(self):
        chunks, _ = _sse_payloads(self._stream())
        finishes = [
            c["choices"][0].get("finish_reason")
            for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason") is not None
        ]
        assert finishes == ["stop"]
        # The finish chunk is the last JSON chunk before [DONE].
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop"

    def test_stream_response_id_consistent_across_chunks(self):
        chunks, _ = _sse_payloads(self._stream())
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1
        (rid,) = ids
        assert rid.startswith("chatcmpl-")

    def test_stream_include_usage_emits_final_usage_chunk(self):
        chunks, _ = _sse_payloads(
            self._stream(stream_options={"include_usage": True})
        )
        usage_chunks = [c for c in chunks if c.get("usage")]
        assert usage_chunks, "include_usage=True produced no usage chunk"
        final = usage_chunks[-1]
        assert final["choices"] == []
        usage = final["usage"]
        assert isinstance(usage.get("prompt_tokens"), int)
        assert isinstance(usage.get("completion_tokens"), int)
        assert isinstance(usage.get("total_tokens"), int)
        # Cross-check: completion_tokens must equal the number of content
        # deltas actually streamed (usage accounting matches the wire).
        assert usage["completion_tokens"] == len(_content_deltas(chunks))
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


class TestCompletionStreamingStrict:
    """Strict SSE contract for POST /v1/completions with stream=true."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _stream(self, **extra):
        payload = {
            "model": "distributed-llm",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": True,
        }
        payload.update(extra)
        return self.client.post("/v1/completions", json=payload)

    def test_stream_status_and_content_type(self):
        resp = self._stream()
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_text_deltas_assemble_into_generated_text(self):
        chunks, _ = _sse_payloads(self._stream())
        assert chunks, "stream produced no JSON chunks before [DONE]"
        assert chunks[0]["object"] == "text_completion.chunk"
        deltas = _text_deltas(chunks)
        assert len(deltas) >= 1, "no text deltas emitted"
        for d in deltas:
            assert _TOKEN_RE.match(d), f"unexpected delta text: {d!r}"
        assert len(deltas) <= 5

    def test_stream_terminates_with_done_signal(self):
        resp = self._stream()
        _, raw = _sse_payloads(resp)
        assert raw[-1] == "data: [DONE]"
        assert raw.count("data: [DONE]") == 1
        # SSE framing: every event line — including the [DONE] sentinel — must
        # be immediately followed by a blank line. Checking line-by-line (not
        # just .endswith) so a lost inter-event separator is caught.
        lines = resp.text.split("\n")
        data_line_idx = [i for i, ln in enumerate(lines) if ln.startswith("data:")]
        assert data_line_idx, "stream body has no data lines"
        for i in data_line_idx:
            assert i + 1 < len(lines) and lines[i + 1] == "", (
                f"event at line {i} not terminated by a blank line: {lines[i]!r}"
            )
        assert resp.text.endswith("data: [DONE]\n\n")

    def test_stream_finish_reason_present_exactly_once(self):
        chunks, _ = _sse_payloads(self._stream())
        finishes = [
            c["choices"][0].get("finish_reason")
            for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason") is not None
        ]
        assert finishes == ["stop"]
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop"

    def test_stream_response_id_consistent_across_chunks(self):
        chunks, _ = _sse_payloads(self._stream())
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1
        (rid,) = ids
        assert rid.startswith("cmpl-")
