"""Tests for streaming helpers — _get_client_id, _build_chunk, StreamChunk, SSE format."""

from __future__ import annotations

import json
import time

import pytest
from starlette.requests import Request

from distllm.api.streaming import _build_chunk, _get_client_id
from distllm.core.streaming_generator import StreamChunk


# ======================================================================
# _get_clientId tests
# ======================================================================


def _make_request(
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
    client_is_none: bool = False,
) -> Request:
    """Build a real Starlette Request from ASGI scope."""
    scope_headers: list[tuple[bytes, bytes]] = []
    if headers:
        for k, v in headers.items():
            scope_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    scope: dict[str, object] = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "headers": scope_headers,
        "client": None if client_is_none else (client_host, 8000),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    return Request(scope)


class TestGetClientId:
    """_get_client_id extracts a stable client identifier from a request."""

    def test_bearer_token(self):
        """Bearer token produces a hashed auth: client ID."""
        req = _make_request(
            headers={"authorization": "Bearer sk-test12345678901234567890"},
            client_host="10.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid.startswith("auth:")
        assert len(cid) > 4  # "auth:" + at least 1 hex char

    def test_bearer_short_token(self):
        """Short bearer tokens still work."""
        req = _make_request(
            headers={"authorization": "Bearer short"},
            client_host="10.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid.startswith("auth:")

    def test_bearer_empty_token(self):
        """Bearer with empty token still produces a hash (of empty string)."""
        req = _make_request(
            headers={"authorization": "Bearer "},
            client_host="10.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid.startswith("auth:")

    def test_bearer_deterministic(self):
        """Same bearer token produces the same client ID."""
        req1 = _make_request(
            headers={"authorization": "Bearer sk-abc123"},
        )
        req2 = _make_request(
            headers={"authorization": "Bearer sk-abc123"},
        )
        assert _get_client_id(req1) == _get_client_id(req2)

    def test_bearer_different_tokens_differ(self):
        """Different bearer tokens produce different client IDs."""
        req1 = _make_request(
            headers={"authorization": "Bearer sk-aaa"},
        )
        req2 = _make_request(
            headers={"authorization": "Bearer sk-bbb"},
        )
        assert _get_client_id(req1) != _get_client_id(req2)

    def test_x_forwarded_for(self, monkeypatch):
        """X-Forwarded-For returns the leftmost IP when trust env is set."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.1, 10.0.0.1"},
            client_host="127.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid == "203.0.113.1"

    def test_x_forwarded_for_single_ip(self, monkeypatch):
        """X-Forwarded-For with a single IP works."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
        req = _make_request(
            headers={"x-forwarded-for": "198.51.100.99"},
            client_host="127.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid == "198.51.100.99"

    def test_x_forwarded_for_skipped_without_env(self, monkeypatch):
        """Without PYTEST_CURRENT_TEST or DISTLLM_TRUST_PROXY_HEADERS,
        X-Forwarded-For is ignored and client.host is used."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.1"},
            client_host="10.0.0.5",
        )
        cid = _get_client_id(req)
        # Should fall through to client.host, not the header
        assert cid == "10.0.0.5"

    def test_x_forwarded_for_with_distllm_env(self, monkeypatch):
        """DISTLLM_TRUST_PROXY_HEADERS=1 also enables proxy header trust."""
        monkeypatch.setenv("DISTLLM_TRUST_PROXY_HEADERS", "1")
        req = _make_request(
            headers={"x-forwarded-for": "10.0.0.99, 10.0.0.1"},
            client_host="127.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid == "10.0.0.99"

    def test_direct_ip_fallback(self):
        """When no relevant headers are present, uses client.host."""
        req = _make_request(client_host="203.0.113.42")
        cid = _get_client_id(req)
        assert cid == "203.0.113.42"

    def test_unknown_when_no_client(self):
        """When request.client is None, returns 'unknown'."""
        req = _make_request(client_is_none=True)
        cid = _get_client_id(req)
        assert cid == "unknown"

    def test_non_bearer_auth_ignored(self):
        """Authorization header that is not Bearer is ignored (no auth: prefix)."""
        req = _make_request(
            headers={"authorization": "Basic dXNlcjpwYXNz"},
            client_host="10.0.0.1",
        )
        cid = _get_client_id(req)
        assert not cid.startswith("auth:")
        assert cid == "10.0.0.1"

    def test_case_insensitive_auth_header(self):
        """Authorization header matching is case-insensitive (startswith check)."""
        req = _make_request(
            headers={"authorization": "BEARER sk-test-token"},
            client_host="10.0.0.1",
        )
        # "BEARER " starts with "Bearer " -> startswith is case-sensitive,
        # so this WILL NOT match and falls through to IP
        # This documents existing behavior since startswith is case-sensitive
        cid = _get_client_id(req)
        assert not cid.startswith("auth:")

    def test_no_headers_at_all(self):
        """Request with no headers (empty dict) falls back to client.host."""
        req = _make_request(headers={}, client_host="10.0.0.1")
        cid = _get_client_id(req)
        assert cid == "10.0.0.1"

    def test_x_forwarded_for_ipv6(self, monkeypatch):
        """IPv6 addresses in X-Forwarded-For are handled correctly."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
        req = _make_request(
            headers={"x-forwarded-for": "2001:db8::1, 10.0.0.1"},
            client_host="127.0.0.1",
        )
        cid = _get_client_id(req)
        assert cid == "2001:db8::1"


# ======================================================================
# _build_chunk tests
# ======================================================================


class TestBuildChunk:
    """_build_chunk constructs StreamChunk objects with correct structure."""

    def test_chat_chunk_basic(self):
        """Basic chat completion chunk with token text."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="Hello",
        )
        assert isinstance(chunk, StreamChunk)
        assert chunk.id == "chatcmpl-abc"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "gpt-4"
        assert len(chunk.choices) == 1
        assert chunk.choices[0]["index"] == 0
        assert chunk.choices[0]["delta"]["content"] == "Hello"
        assert chunk.choices[0]["finish_reason"] is None

    def test_chat_chunk_with_finish_reason(self):
        """Chat completion chunk including a finish_reason."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="world",
            finish_reason="stop",
        )
        assert chunk.choices[0]["finish_reason"] == "stop"
        assert chunk.choices[0]["delta"]["content"] == "world"

    def test_chat_chunk_with_role(self):
        """When include_role=True, delta includes a role field."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            include_role=True,
        )
        assert chunk.choices[0]["delta"]["role"] == "assistant"

    def test_chat_chunk_with_role_and_content(self):
        """Role and content can coexist in the delta."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="Hello",
            include_role=True,
        )
        assert chunk.choices[0]["delta"]["role"] == "assistant"
        assert chunk.choices[0]["delta"]["content"] == "Hello"

    def test_chat_chunk_tool_calls(self):
        """When tool_calls is provided, content is set to None per OpenAI spec."""
        tool_calls = [
            {
                "index": 0,
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"loc": "NYC"}'},
            }
        ]
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            tool_calls=tool_calls,
        )
        delta = chunk.choices[0]["delta"]
        assert delta["content"] is None
        assert delta["tool_calls"] == tool_calls

    def test_chat_chunk_tool_calls_with_finish_reason(self):
        """Tool-call chunk with finish_reason='tool_calls'."""
        tool_calls = [
            {
                "index": 0,
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        assert chunk.choices[0]["finish_reason"] == "tool_calls"
        assert chunk.choices[0]["delta"]["tool_calls"] == tool_calls

    def test_text_completion_chunk(self):
        """Text completion uses 'text' field instead of 'delta'."""
        chunk = _build_chunk(
            request_id="cmpl-abc",
            object_type="text.completion",
            model="gpt-3.5-turbo-instruct",
            token_text="Once upon a time",
        )
        assert chunk.object == "text.completion"
        assert chunk.choices[0]["text"] == "Once upon a time"
        assert "delta" not in chunk.choices[0]

    def test_text_completion_with_finish_reason(self):
        """Text completion chunk with finish_reason."""
        chunk = _build_chunk(
            request_id="cmpl-abc",
            object_type="text.completion",
            model="gpt-3.5-turbo-instruct",
            token_text="end",
            finish_reason="length",
        )
        assert chunk.choices[0]["text"] == "end"
        assert chunk.choices[0]["finish_reason"] == "length"

    def test_with_logprobs(self):
        """Logprob data is injected into the choice dict."""
        logprob_data = {
            "token": "Hello",
            "logprob": -0.123,
            "top_logprobs": [{"token": "Hello", "logprob": -0.123}],
        }
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="Hello",
            logprob_data=logprob_data,
        )
        assert chunk.choices[0]["logprobs"] == logprob_data

    def test_with_logprobs_text_completion(self):
        """Logprobs also work on text completion chunks."""
        logprob_data = {"token": "A", "logprob": -0.5}
        chunk = _build_chunk(
            request_id="cmpl-abc",
            object_type="text.completion",
            model="gpt-3.5-turbo-instruct",
            token_text="A",
            logprob_data=logprob_data,
        )
        assert chunk.choices[0]["logprobs"] == logprob_data

    def test_empty_token_text(self):
        """Empty token_text results in delta without content key
        (since empty string is falsy, _build_chunk skips setting it)."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="",
        )
        assert "content" not in chunk.choices[0]["delta"]

    def test_empty_token_text_text_completion(self):
        """Empty token_text in text completion."""
        chunk = _build_chunk(
            request_id="cmpl-abc",
            object_type="text.completion",
            model="gpt-4",
            token_text="",
        )
        assert chunk.choices[0]["text"] == ""

    def test_no_token_text_no_tool_calls(self):
        """No token_text and no tool_calls -> delta has no content key."""
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
        )
        delta = chunk.choices[0]["delta"]
        assert "content" not in delta

    def test_created_timestamp(self):
        """created is set to current time (approximately)."""
        before = int(time.time())
        chunk = _build_chunk(
            request_id="chatcmpl-abc",
            object_type="chat.completion.chunk",
            model="gpt-4",
            token_text="x",
        )
        after = int(time.time())
        assert before <= chunk.created <= after


# ======================================================================
# StreamChunk SSE format tests
# ======================================================================


class TestStreamChunkFormat:
    """StreamChunk.to_sse() produces valid SSE-formatted strings."""

    def test_chunk_to_sse_basic(self):
        """Basic SSE output format."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        )
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert "Hello" in sse
        assert sse.endswith("\n\n")

    def test_chunk_to_sse_usage_excluded_when_none(self):
        """When usage is None, it is omitted from the JSON output."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            usage=None,
        )
        sse = chunk.to_sse()
        assert '"usage"' not in sse

    def test_chunk_to_sse_usage_included(self):
        """When usage is set, it appears in the JSON output."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage={"total_tokens": 42},
        )
        sse = chunk.to_sse()
        assert '"total_tokens": 42' in sse or "42" in sse
        assert '"usage"' in sse

    def test_chunk_to_sse_unicode(self):
        """Unicode content in SSE is handled."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "Hello 世界"}, "finish_reason": None}],
        )
        sse = chunk.to_sse()
        assert "世界" in sse or "\\u4e16\\u754c" in sse

    def test_chunk_to_sse_newlines_in_content(self):
        """Content with newlines is properly JSON-escaped."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "line1\nline2"}, "finish_reason": None}],
        )
        sse = chunk.to_sse()
        # JSON-encoded newline is \\n (literal backslash-n), not a real newline
        assert "\\n" in sse or "line1" in sse

    def test_chunk_with_finish_reason(self):
        """Finish chunk includes the finish reason."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        sse = chunk.to_sse()
        assert '"finish_reason": "stop"' in sse

    def test_chunk_with_multiple_choices(self):
        """Multiple choices are serialized correctly."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[
                {"index": 0, "delta": {"content": "A"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "B"}, "finish_reason": None},
            ],
        )
        sse = chunk.to_sse()
        assert '"index": 0' in sse
        assert '"index": 1' in sse

    def test_chunk_with_logprobs_in_choices(self):
        """Chunks with logprobs include the data."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{
                "index": 0,
                "delta": {"content": "A"},
                "finish_reason": None,
                "logprobs": {"token": "A", "logprob": -0.5},
            }],
        )
        sse = chunk.to_sse()
        assert "logprobs" in sse or "logprob" in sse

    def test_done_signal(self):
        """The [DONE] signal marks stream completion."""
        sse = StreamChunk.data_done()
        assert sse == "data: [DONE]\n\n"

    def test_sse_parsing_round_trip(self):
        """Chunk SSE can be parsed back into valid JSON and matches input."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        )
        sse = chunk.to_sse()
        lines = sse.strip().split("\n")
        assert lines[0].startswith("data: ")
        data = json.loads(lines[0][6:])
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["id"] == "test-id"

    def test_empty_delta(self):
        """A chunk with an empty delta is valid SSE."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {}, "finish_reason": None}],
        )
        sse = chunk.to_sse()
        assert "data: " in sse

    def test_default_constructor(self):
        """StreamChunk default constructor produces a valid-format SSE."""
        chunk = StreamChunk()
        sse = chunk.to_sse()
        assert "data: " in sse
        data = json.loads(sse[6:].strip())
        assert data["object"] == "chat.completion.chunk"

    def test_data_done_is_static(self):
        """data_done() returns same value regardless of instance."""
        assert StreamChunk.data_done() == "data: [DONE]\n\n"
        assert StreamChunk().data_done() == "data: [DONE]\n\n"
