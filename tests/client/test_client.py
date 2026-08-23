"""Tests for ``DistLLMClient`` -- connection pooling, retry, timeouts, error parsing.

All tests use the ``load_module`` pattern from ``tests._import_helper`` to
bypass ``distllm/__init__.py`` and its circular-import chain, then use
real ``httpx.MockTransport`` so no real network I/O occurs.

No MagicMock -- real httpx transport, real dataclass instances.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_client_mod = load_module("distllm/client/client.py")
DistLLMClient = _client_mod.DistLLMClient
SyncDistLLMClient = _client_mod.SyncDistLLMClient
CompletionResponse = _client_mod.CompletionResponse
ChatResponse = _client_mod.ChatResponse
ChatMessage = _client_mod.ChatMessage
CompletionChoice = _client_mod.CompletionChoice
ChatChoice = _client_mod.ChatChoice
ModelInfo = _client_mod.ModelInfo
NodeInfo = _client_mod.NodeInfo
ClusterMetrics = _client_mod.ClusterMetrics


# -- Transport helpers --------------------------------------------------------


def _install(client, handler):
    """Replace the transport on an existing DistLLMClient with *handler*.

    Keeps ``base_url`` and other settings intact by replacing only the
    ``_transport`` attribute on the existing ``httpx.AsyncClient``.
    """
    client._client._transport = httpx.MockTransport(handler)
    return client


def _json_response(payload, status=200):
    """Build a handler that returns *payload* as JSON."""
    def handler(request):
        return httpx.Response(status, json=payload)
    return handler


def _text_response(text, status=200, content_type="application/json"):
    """Build a handler that returns plain text."""
    def handler(request):
        return httpx.Response(status, text=text,
                              headers={"content-type": content_type})
    return handler


def _sequence_handler(responses):
    """Build a handler that returns responses in sequence."""
    it = iter(responses)
    def handler(request):
        try:
            return next(it)
        except StopIteration:
            return httpx.Response(200, json={})
    return handler


# ============================================================================
# 1.  Connection Pooling & Initialisation
# ============================================================================


class TestConnectionInitialisation:
    """Client construction, auth headers, base URL normalisation, factory."""

    def test_default_timeout(self):
        client = DistLLMClient("http://localhost:8000")
        assert client._timeout == DistLLMClient.DEFAULT_TIMEOUT == 60.0

    def test_default_max_retries(self):
        client = DistLLMClient("http://localhost:8000")
        assert client._max_retries == DistLLMClient.MAX_RETRIES == 3

    def test_custom_timeout_and_retries(self):
        client = DistLLMClient("http://localhost:8000", timeout=15.0, max_retries=5)
        assert client._timeout == 15.0
        assert client._max_retries == 5

    def test_base_url_strips_trailing_slash(self):
        client = DistLLMClient("http://localhost:8000/")
        assert client._base_url == "http://localhost:8000"

    def test_auth_header_set_when_api_key_provided(self):
        client = DistLLMClient("http://localhost:8000", api_key="sk-abc")
        assert client._client.headers.get("Authorization") == "Bearer sk-abc"

    def test_auth_header_absent_when_no_api_key(self):
        client = DistLLMClient("http://localhost:8000")
        assert "Authorization" not in client._client.headers

    def test_content_type_headers_present(self):
        client = DistLLMClient("http://localhost:8000")
        assert client._client.headers["Content-Type"] == "application/json"
        assert client._client.headers["Accept"] == "application/json"

    def test_http_client_base_url_set(self):
        client = DistLLMClient("http://10.0.0.1:8000")
        assert str(client._client.base_url) == "http://10.0.0.1:8000"

    def test_http_client_timeout_set(self):
        client = DistLLMClient("http://localhost:8000", timeout=42.0)
        assert client._client.timeout.connect == 42.0

    @pytest.mark.asyncio
    async def test_connect_creates_client_and_checks_health(self):
        def handler(request):
            if str(request.url).endswith("/health"):
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, json={})
        client = DistLLMClient("http://localhost:8000", api_key="sk-test")
        _install(client, handler)
        await client._check_connection()
        assert client.is_connected
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_raises_on_health_failure(self):
        def handler(request):
            raise httpx.ConnectError("refused")
        client = DistLLMClient("http://localhost:8000")
        _install(client, handler)
        with pytest.raises(ConnectionError, match="Could not connect"):
            await client._check_connection()

    @pytest.mark.asyncio
    async def test_is_connected_false_after_close(self):
        client = DistLLMClient("http://localhost:8000")
        assert client.is_connected
        await client.close()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_connect_inherits_parameters(self):
        """Factory passes all args to the constructor."""
        original_check = DistLLMClient._check_connection
        DistLLMClient._check_connection = lambda self: None
        try:
            client = DistLLMClient(
                coordinator_url="http://10.0.0.1:9000",
                api_key="sk-factory-key",
                timeout=99.0,
                max_retries=7,
            )
            assert client._base_url == "http://10.0.0.1:9000"
            assert client._api_key == "sk-factory-key"
            assert client._timeout == 99.0
            assert client._max_retries == 7
        finally:
            DistLLMClient._check_connection = original_check


# ============================================================================
# 2.  Request Retry Behaviour
# ============================================================================


class TestRequestRetry:
    """Transient HTTP errors (429, 503, 502, 504) trigger retry with backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_503_then_succeed(self):
        """503 triggers retry; eventual 200 is returned."""
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(503, text="overloaded")
            return httpx.Response(200, json={"id": "cmpl-1", "choices": [{"text": "ok"}]})
        client = DistLLMClient("http://localhost:8000", max_retries=3, timeout=5.0)
        _install(client, handler)
        result = await client.generate("hello")
        assert result.text == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_429_then_succeed(self):
        """429 (rate limit) triggers retry."""
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={})
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        client = DistLLMClient("http://localhost:8000", max_retries=2, timeout=5.0)
        _install(client, handler)
        models = await client.list_models()
        assert len(models) == 1
        assert models[0].id == "m1"

    @pytest.mark.asyncio
    async def test_retry_on_502_and_504(self):
        """502 and 504 both trigger retry."""
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(502, json={})
            if call_count == 2:
                return httpx.Response(504, json={})
            return httpx.Response(200, json=[{"node_id": "n1"}])
        client = DistLLMClient("http://localhost:8000", max_retries=3, timeout=5.0)
        _install(client, handler)
        nodes = await client.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises_runtime_error(self):
        """After exhausting retries a ``RuntimeError`` is raised."""
        def handler(request):
            return httpx.Response(503, text="always overloaded")
        client = DistLLMClient("http://localhost:8000", max_retries=2, timeout=5.0)
        _install(client, handler)
        with pytest.raises(RuntimeError, match="DistLLM API error 503"):
            await client.generate("hello")

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        """4xx errors (client errors) are not retried."""
        def handler(request):
            return httpx.Response(400, json={"error": "bad request"})
        client = DistLLMClient("http://localhost:8000", max_retries=3, timeout=5.0)
        _install(client, handler)
        with pytest.raises(RuntimeError, match="DistLLM API error 400"):
            await client.generate("hello")

    @pytest.mark.asyncio
    async def test_retry_connection_error(self):
        """Connection/timeout errors are retried."""
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timed out", request=request)
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        client = DistLLMClient("http://localhost:8000", max_retries=2, timeout=5.0)
        _install(client, handler)
        models = await client.list_models()
        assert len(models) == 1
        assert models[0].id == "m1"

    @pytest.mark.asyncio
    async def test_exhaust_retries_on_connection_error(self):
        """Exhausted retries on connection errors raise ``RuntimeError``."""
        def handler(request):
            raise httpx.TimeoutException("timed out", request=request)
        client = DistLLMClient("http://localhost:8000", max_retries=2, timeout=5.0)
        _install(client, handler)
        with pytest.raises(RuntimeError, match="DistLLM request failed after 2 retries"):
            await client.list_models()

    @pytest.mark.asyncio
    async def test_retry_backoff_increases(self):
        """Delay between retries grows exponentially (1.5 ** attempt)."""
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, text="rate limited")
        client = DistLLMClient("http://localhost:8000", max_retries=3, timeout=5.0)
        _install(client, handler)
        async def no_sleep(delay):
            return None
        original_sleep = asyncio.sleep
        asyncio.sleep = no_sleep
        try:
            with pytest.raises(RuntimeError):
                await client.list_models()
        finally:
            asyncio.sleep = original_sleep
        # With max_retries=3, the loop runs 3 times total (initial + 2 retries)
        assert call_count == 3


# ============================================================================
# 3.  Timeout Handling
# ============================================================================


class TestTimeoutHandling:
    """Timeout exceptions are caught and retried appropriately."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        call_count = 0
        def handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("read timed out")
            return httpx.Response(200, json={"id": "cmpl-1", "choices": [{"text": "done"}]})
        client = DistLLMClient("http://localhost:8000", max_retries=2, timeout=5.0)
        _install(client, handler)
        result = await client.generate("hello")
        assert result.text == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_exhausted_raises_runtime_error(self):
        def handler(request):
            raise httpx.TimeoutException("connect timed out")
        client = DistLLMClient("http://localhost:8000", max_retries=1, timeout=5.0)
        _install(client, handler)
        with pytest.raises(RuntimeError, match="DistLLM request failed after 1 retries"):
            await client.generate("hello")


# ============================================================================
# 4.  Error Response Parsing
# ============================================================================


class TestErrorResponseParsing:
    """Parsing of HTTP-level errors and malformed response bodies."""

    @pytest.mark.asyncio
    async def test_401_raises_runtime_error(self):
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        _install(client, _json_response({"error": {"message": "Invalid API key"}}, 401))
        with pytest.raises(RuntimeError, match="DistLLM API error 401"):
            await client.list_models()

    @pytest.mark.asyncio
    async def test_500_raises_runtime_error(self):
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        _install(client, _text_response("Internal Server Error", 500, "text/plain"))
        with pytest.raises(RuntimeError, match="DistLLM API error 500"):
            await client.get_metrics()

    @pytest.mark.asyncio
    async def test_malformed_json_parse_error(self):
        """A non-200 response with unparseable body still raises ``RuntimeError``."""
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        _install(client, _text_response("<html>bad gateway</html>", 502, "text/html"))
        with pytest.raises(RuntimeError, match="DistLLM API error 502"):
            await client.list_models()


# ============================================================================
# 5.  Successful Response Parsing
# ============================================================================


class TestCompletionResponseParsing:
    """Parsing of ``/v1/completions`` responses."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, completion_payload):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(completion_payload))
        result = await client.generate("hello")
        assert isinstance(result, CompletionResponse)
        assert result.text == "Hello world"
        assert result.model == "test-model"
        assert result.request_id == "cmpl-abc123"
        assert result.usage == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        assert len(result.choices) == 2

    def test_completion_choice_fields(self, completion_payload):
        choices = completion_payload["choices"]
        parsed = [
            CompletionChoice(
                text=c.get("text", ""),
                index=c.get("index", 0),
                finish_reason=c.get("finish_reason", ""),
            )
            for c in choices
        ]
        assert parsed[0].text == "Hello world"
        assert parsed[0].index == 0
        assert parsed[0].finish_reason == "length"
        assert parsed[1].text == "Second choice"
        assert parsed[1].index == 1

    @pytest.mark.asyncio
    async def test_completion_no_choices(self):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response({"id": "cmpl-empty", "choices": []}))
        result = await client.generate("hello")
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_completion_text_plain_response(self):
        """text/plain responses are wrapped as ``{"text": ...}``."""
        client = DistLLMClient("http://localhost:8000")
        _install(client, _text_response("raw plain text output", content_type="text/plain"))
        result = await client.generate("hello")
        assert result.text == "raw plain text output"


class TestChatResponseParsing:
    """Parsing of ``/v1/chat/completions`` responses."""

    @pytest.mark.asyncio
    async def test_basic_chat(self, chat_payload):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(chat_payload))
        result = await client.generate_chat([{"role": "user", "content": "How are you?"}])
        assert isinstance(result, ChatResponse)
        assert result.message.content == "I am fine."
        assert result.message.role == "assistant"
        assert result.model == "test-model"
        assert result.request_id == "chatcmpl-xyz789"
        assert len(result.choices) == 1

    def test_chat_choice_fields(self, chat_payload):
        choices = chat_payload["choices"]
        parsed = [
            ChatChoice(
                message=_client_mod.ChatMessage(
                    role=c.get("message", {}).get("role", ""),
                    content=c.get("message", {}).get("content", ""),
                ),
                index=c.get("index", 0),
                finish_reason=c.get("finish_reason", ""),
            )
            for c in choices
        ]
        assert parsed[0].message.content == "I am fine."
        assert parsed[0].message.role == "assistant"
        assert parsed[0].index == 0
        assert parsed[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_no_choices(self):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response({"id": "chatcmpl-empty", "choices": []}))
        result = await client.generate_chat([{"role": "user", "content": "hi"}])
        assert result.message.content == ""


class TestModelListParsing:
    """Parsing of ``/v1/models`` responses."""

    @pytest.mark.asyncio
    async def test_list_models(self, models_payload):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(models_payload))
        models = await client.list_models()
        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)
        assert models[0].id == "model-a"
        assert models[0].owned_by == "org-1"
        assert models[1].id == "model-b"

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response({"data": []}))
        models = await client.list_models()
        assert models == []


class TestNodeListParsing:
    """Parsing of ``/api/v1/nodes`` responses."""

    @pytest.mark.asyncio
    async def test_list_nodes(self, nodes_payload):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(nodes_payload))
        nodes = await client.list_nodes()
        assert len(nodes) == 1
        node = nodes[0]
        assert isinstance(node, NodeInfo)
        assert node.node_id == "node-0"
        assert node.host == "10.0.0.1"
        assert node.port == 50051
        assert node.start_layer == 0
        assert node.end_layer == 5
        assert node.healthy is True
        assert node.gpu_name == "Tesla T4"
        assert node.gpu_utilization == 0.45
        assert node.memory_free_mb == 6.0 * 1024

    @pytest.mark.asyncio
    async def test_list_nodes_empty(self):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response([]))
        nodes = await client.list_nodes()
        assert nodes == []


class TestMetricsParsing:
    """Parsing of ``/api/v1/metrics`` responses."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, metrics_payload):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(metrics_payload))
        metrics = await client.get_metrics()
        assert isinstance(metrics, ClusterMetrics)
        assert metrics.requests_total == 1000
        assert metrics.tokens_generated == 50000
        assert metrics.active_requests == 5
        assert metrics.pending_requests == 2
        assert metrics.node_count == 3
        assert metrics.p95_latency_ms == 250.0
        assert metrics.errors_total == 10
        assert metrics.cache_hit_rate == 0.85

    @pytest.mark.asyncio
    async def test_get_metrics_empty(self):
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response({}))
        metrics = await client.get_metrics()
        assert metrics.requests_total == 0
        assert metrics.node_count == 0
        assert metrics.cache_hit_rate == 0.0


# ============================================================================
# 6.  Streaming
# ============================================================================


class _StubSSEResponse:
    """Minimal httpx response stub for streaming."""

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _StubSSEClient:
    """Minimal httpx-like client stub for SSE streaming tests.

    Implements just enough of the AsyncClient interface that
    ``stream_generate()`` can work: it provides ``stream()``
    which returns an async context manager with ``aiter_lines()``.
    """

    def __init__(self, sse_lines):
        self._sse_lines = sse_lines
        self.stream_kwargs = None

    def stream(self, method, url, **kwargs):
        """Return a context manager for SSE streaming.

        Note: this is a synchronous method returning a context manager,
        matching httpx.AsyncClient.stream() which is also synchronous.
        """
        self.stream_kwargs = kwargs
        return _StubSSEResponse(self._sse_lines)

    async def close(self):
        pass


class TestStreaming:
    """SSE streaming via ``stream_generate``."""

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self):
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        sse_lines = [
            "data: " + json.dumps({"choices": [{"text": "Hello"}], "index": 0}) + "\n\n",
            "data: " + json.dumps({"choices": [{"text": " world"}], "index": 1}) + "\n\n",
            "data: " + json.dumps({"choices": [{"text": ""}], "index": 2}) + "\n\n",
            "data: [DONE]\n\n",
        ]
        client._client = _StubSSEClient(sse_lines)
        chunks = []
        async for chunk in client.stream_generate("hello"):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_heartbeat_and_empty_lines(self):
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        sse_lines = [
            ": heartbeat\n\n",
            "\n",
            "data: " + json.dumps({"choices": [{"text": "A"}]}) + "\n\n",
            "data: [DONE]\n\n",
        ]
        client._client = _StubSSEClient(sse_lines)
        chunks = [c async for c in client.stream_generate("x")]
        assert chunks == ["A"]

    @pytest.mark.asyncio
    async def test_stream_ignores_malformed_json(self):
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        sse_lines = [
            "data: not valid json\n\n",
            "data: [DONE]\n\n",
        ]
        client._client = _StubSSEClient(sse_lines)
        chunks = [c async for c in client.stream_generate("x")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_requests_with_stream_true(self):
        """stream=True is passed in the payload."""
        client = DistLLMClient("http://localhost:8000", max_retries=1)
        stub = _StubSSEClient(["data: [DONE]\n\n"])
        client._client = stub
        async for _ in client.stream_generate("hello", max_tokens=10, temperature=0.5):
            pass
        assert stub.stream_kwargs is not None
        assert stub.stream_kwargs["json"]["stream"] is True
        assert stub.stream_kwargs["json"]["max_tokens"] == 10
        assert stub.stream_kwargs["json"]["temperature"] == 0.5


# ============================================================================
# 7.  Payload Construction
# ============================================================================


class TestPayloadConstruction:
    """Verifies request payloads are formed correctly."""

    @pytest.mark.asyncio
    async def test_generate_payload_includes_all_fields(self):
        def handler(request):
            body = json.loads(request.read())
            assert body["prompt"] == "hello"
            assert body["max_tokens"] == 100
            assert body["temperature"] == 0.8
            assert body["top_p"] == 0.95
            assert body["model"] == "gpt-3"
            return httpx.Response(200, json={"choices": [{"text": "ok"}]})
        client = DistLLMClient("http://localhost:8000")
        _install(client, handler)
        await client.generate("hello", max_tokens=100, temperature=0.8, top_p=0.95, model="gpt-3")

    @pytest.mark.asyncio
    async def test_generate_payload_defaults(self):
        def handler(request):
            body = json.loads(request.read())
            assert body["max_tokens"] == 256
            assert body["temperature"] == 0.7
            assert body["top_p"] == 0.9
            assert "model" not in body
            return httpx.Response(200, json={"choices": [{"text": "ok"}]})
        client = DistLLMClient("http://localhost:8000")
        _install(client, handler)
        await client.generate("hello")

    @pytest.mark.asyncio
    async def test_generate_extra_kwargs_forwarded(self):
        def handler(request):
            body = json.loads(request.read())
            assert body["stop"] == ["\n"]
            assert body["frequency_penalty"] == 0.5
            return httpx.Response(200, json={"choices": [{"text": "ok"}]})
        client = DistLLMClient("http://localhost:8000")
        _install(client, handler)
        await client.generate("hello", stop=["\n"], frequency_penalty=0.5)

    @pytest.mark.asyncio
    async def test_chat_payload_includes_all_fields(self):
        def handler(request):
            body = json.loads(request.read())
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            assert body["max_tokens"] == 200
            assert body["temperature"] == 0.5
            assert body["model"] == "gpt-4"
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        client = DistLLMClient("http://localhost:8000")
        _install(client, handler)
        msgs = [{"role": "user", "content": "hi"}]
        await client.generate_chat(msgs, max_tokens=200, temperature=0.5, model="gpt-4")


# ============================================================================
# 8.  SyncDistLLMClient Wrapper
# ============================================================================


class TestSyncClient:
    """Synchronous wrapper delegates to ``asyncio.run``."""

    def _make_async_client(self, response_json):
        """Create a DistLLMClient with mock transport returning *response_json*."""
        client = DistLLMClient("http://localhost:8000")
        _install(client, _json_response(response_json))
        return client

    def test_sync_generate(self):
        aclient = self._make_async_client({"choices": [{"text": "sync result"}]})
        sync = SyncDistLLMClient("http://localhost:8000")
        sync._client = aclient
        result = sync.generate("hello")
        assert result.text == "sync result"
        sync.close()

    def test_sync_generate_chat(self):
        aclient = self._make_async_client(
            {"choices": [{"message": {"role": "assistant", "content": "sync chat"}}]}
        )
        sync = SyncDistLLMClient("http://localhost:8000")
        sync._client = aclient
        result = sync.generate_chat([{"role": "user", "content": "hi"}])
        assert result.message.content == "sync chat"
        sync.close()

    def test_sync_list_models(self):
        aclient = self._make_async_client({"data": [{"id": "m1", "object": "model", "owned_by": "org-1", "created": 0}]})
        sync = SyncDistLLMClient("http://localhost:8000")
        sync._client = aclient
        models = sync.list_models()
        assert len(models) == 1
        assert models[0].id == "m1"
        sync.close()

    def test_sync_list_nodes(self):
        aclient = self._make_async_client([{"node_id": "n1"}])
        sync = SyncDistLLMClient("http://localhost:8000")
        sync._client = aclient
        nodes = sync.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"
        sync.close()

    def test_sync_get_metrics(self):
        aclient = self._make_async_client({"requests_total": 42})
        sync = SyncDistLLMClient("http://localhost:8000")
        sync._client = aclient
        metrics = sync.get_metrics()
        assert metrics.requests_total == 42
        sync.close()

    def test_sync_close_idempotent(self):
        sync = SyncDistLLMClient("http://localhost:8000")
        sync.close()
        sync.close()

    def test_sync_client_creates_async_client_lazily(self):
        """The underlying async client is created on first access."""
        sync = SyncDistLLMClient("http://localhost:8000")
        assert sync._client is None
        aclient = self._make_async_client({"choices": [{"text": ""}]})
        sync._client = aclient
        sync.generate("hello")
        assert sync._client is not None
        sync.close()


# ============================================================================
# 9.  Dataclass Defaults
# ============================================================================


class TestDataclassDefaults:
    """Response model dataclasses have sensible default values."""

    def test_completion_response_defaults(self):
        r = CompletionResponse()
        assert r.text == ""
        assert r.choices == []
        assert r.model == ""
        assert r.usage == {}
        assert r.request_id == ""

    def test_chat_response_defaults(self):
        r = ChatResponse()
        assert r.message.content == ""
        assert r.message.role == ""
        assert r.choices == []

    def test_model_info_defaults(self):
        m = ModelInfo()
        assert m.id == ""
        assert m.object == "model"
        assert m.owned_by == ""

    def test_node_info_defaults(self):
        n = NodeInfo()
        assert n.node_id == ""
        assert n.healthy is True
        assert n.gpu_utilization == 0.0

    def test_cluster_metrics_defaults(self):
        m = ClusterMetrics()
        assert m.requests_total == 0
        assert m.node_count == 0
        assert m.p95_latency_ms == 0.0
        assert m.cache_hit_rate == 0.0
