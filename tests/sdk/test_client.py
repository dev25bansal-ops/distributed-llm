"""Tests for DistLLM SDK client and types."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import httpx

from distllm.sdk.client import _BaseClient, DistLLMClient, DistLLMClientSync, _compute_delay, RetryConfig, PoolConfig
from distllm.sdk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState
from distllm.sdk.streaming import parse_sse_stream, parse_sse_stream_sync
from distllm.sdk.types import (
    ChatMessage, ChatCompletionRequest, ChatChoice, ChatCompletionResponse,
    CompletionRequest, CompletionChoice, CompletionResponse,
    ModelInfo, ModelList,
)
from distllm.sdk.streaming import parse_sse_stream


# --- SDK Types Tests ---

class TestSDKTypes:
    def test_chat_message_validation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.model == "distributed-llm"
        assert req.temperature == 0.7
        assert req.stream is False

    def test_chat_completion_response_parsing(self):
        resp = ChatCompletionResponse(
            id="chat-1", created=1234567890, model="test",
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="Hi"))],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        assert resp.id == "chat-1"
        assert resp.usage["total_tokens"] == 8

    def test_completion_response_parsing(self):
        resp = CompletionResponse(
            id="cmpl-1", created=1234567890, model="test",
            choices=[CompletionChoice(text="Hello world")],
        )
        assert resp.choices[0].text == "Hello world"

    def test_model_list_parsing(self):
        ml = ModelList(data=[ModelInfo(id="model1", created=123)])
        assert ml.object == "list"
        assert len(ml.data) == 1
        assert ml.data[0].id == "model1"

    def test_response_format_field(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            response_format={"type": "json_object"}
        )
        assert req.response_format["type"] == "json_object"

    def test_adapter_field(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            adapter="my-lora"
        )
        assert req.adapter == "my-lora"


# --- Base Client Tests ---

class TestBaseClientHelpers:
    def test_base_url_strips_trailing_slash(self):
        class TestClient(_BaseClient):
            def _request(self, *a, **k): pass
            def _stream_request(self, *a, **k): pass

        client = TestClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_api_key_sets_authorization_header(self):
        class TestClient(_BaseClient):
            def _request(self, *a, **k): pass
            def _stream_request(self, *a, **k): pass

        client = TestClient(api_key="test-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_header(self):
        class TestClient(_BaseClient):
            def _request(self, *a, **k): pass
            def _stream_request(self, *a, **k): pass

        client = TestClient()
        headers = client._build_headers()
        assert "Authorization" not in headers

    def test_build_chat_payload(self):
        class TestClient(_BaseClient):
            def _request(self, *a, **k): pass
            def _stream_request(self, *a, **k): pass

        client = TestClient()
        payload = client._build_chat_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model", temperature=0.5, top_p=0.9,
            max_tokens=128, stream=False, response_format=None, adapter=None,
        )
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["stream"] is False
        assert "response_format" not in payload
        assert "adapter" not in payload

    def test_build_chat_payload_with_response_format(self):
        class TestClient(_BaseClient):
            def _request(self, *a, **k): pass
            def _stream_request(self, *a, **k): pass

        client = TestClient()
        payload = client._build_chat_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="test", temperature=0.7, top_p=0.9, max_tokens=256,
            stream=False, response_format={"type": "json_object"}, adapter="lora-1",
        )
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["adapter"] == "lora-1"


# --- Async Client Tests ---

class TestDistLLMClientInit:
    def test_default_initialization(self):
        client = DistLLMClient()
        assert client.base_url == "http://localhost:8000"
        assert client._timeout == 120.0

    def test_custom_base_url(self):
        client = DistLLMClient(base_url="http://api.example.com:9000/")
        assert client.base_url == "http://api.example.com:9000"

    def test_api_key_stored(self):
        client = DistLLMClient(api_key="secret")
        assert client._api_key == "secret"

    def test_timeout_configuration(self):
        client = DistLLMClient(timeout=60.0)
        assert client._timeout == 60.0

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with DistLLMClient() as client:
            assert client._client is not None
        # Client should be closed after exit


# --- Async Chat Tests (Mocked HTTP) ---

class TestDistLLMClientChat:
    @pytest.mark.asyncio
    async def test_chat_completion_basic(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "chat-1", "object": "chat.completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            resp = await client.chat_completions(
                messages=[{"role": "user", "content": "Hi"}]
            )
            assert resp.choices[0].message.content == "Hello"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["temperature"] == 0.9
            return httpx.Response(200, json={
                "id": "chat-1", "object": "chat.completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            await client.chat_completions(
                messages=[{"role": "user", "content": "Hi"}], temperature=0.9
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_response_format(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["response_format"] == {"type": "json_object"}
            return httpx.Response(200, json={
                "id": "chat-1", "object": "chat.completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            await client.chat_completions(
                messages=[{"role": "user", "content": "JSON"}],
                response_format={"type": "json_object"}
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_completion_with_adapter(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["adapter"] == "lora-v1"
            return httpx.Response(200, json={
                "id": "chat-1", "object": "chat.completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            await client.chat_completions(
                messages=[{"role": "user", "content": "Hi"}], adapter="lora-v1"
            )
        finally:
            await client.close()


# --- Async Completions Tests ---

class TestDistLLMClientCompletions:
    @pytest.mark.asyncio
    async def test_completion_basic(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "cmpl-1", "object": "text_completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "text": "Hello world"}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            resp = await client.completions(prompt="Once upon")
            assert resp.choices[0].text == "Hello world"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_completion_with_parameters(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["temperature"] == 0.5
            assert body["max_tokens"] == 64
            return httpx.Response(200, json={
                "id": "cmpl-1", "object": "text_completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "text": "done"}],
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            await client.completions(prompt="Test", temperature=0.5, max_tokens=64)
        finally:
            await client.close()


# --- Async Models/Health Tests ---

class TestDistLLMClientModels:
    @pytest.mark.asyncio
    async def test_list_models(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "object": "list", "data": [{"id": "model1", "object": "model", "created": 123, "owned_by": "distllm"}]
            })

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            models = await client.list_models()
            assert len(models.data) == 1
            assert models.data[0].id == "model1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_check(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "healthy", "model": "test"})

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            health = await client.health_check()
            assert health["status"] == "healthy"
        finally:
            await client.close()


# --- Sync Client Tests ---

class TestDistLLMClientSync:
    def test_sync_context_manager(self):
        with DistLLMClientSync() as client:
            assert client._client is not None

    def test_sync_chat_completion(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "chat-1", "object": "chat.completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            })

        transport = httpx.MockTransport(mock_handler)
        with httpx.Client(transport=transport, base_url="http://localhost:8000") as http_client:
            client = DistLLMClientSync()
            client._client = http_client
            resp = client.chat_completions(messages=[{"role": "user", "content": "Hi"}])
            assert resp.choices[0].message.content == "Hi"

    def test_sync_completion(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "cmpl-1", "object": "text_completion", "created": 123,
                "model": "test", "choices": [{"index": 0, "text": "done"}],
            })

        transport = httpx.MockTransport(mock_handler)
        with httpx.Client(transport=transport, base_url="http://localhost:8000") as http_client:
            client = DistLLMClientSync()
            client._client = http_client
            resp = client.completions(prompt="Test")
            assert resp.choices[0].text == "done"

    def test_sync_close(self):
        client = DistLLMClientSync()
        client.close()
        # Should not raise


# --- Error Handling Tests ---

class TestDistLLMClientErrors:
    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "Internal error"})

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat_completions(messages=[{"role": "user", "content": "Hi"}])
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_404_error_raises(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Not found"})

        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.list_models()
        finally:
            await client.close()

    def test_sync_http_error_raises(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "Unavailable"})

        client = DistLLMClientSync()
        client._client = httpx.Client(transport=httpx.MockTransport(mock_handler), base_url="http://localhost:8000")
        try:
            with pytest.raises(httpx.HTTPStatusError):
                client.chat_completions(messages=[{"role": "user", "content": "Hi"}])
        finally:
            client.close()


# --- Streaming Tests ---

class TestDistLLMClientStreaming:
    @pytest.mark.asyncio
    async def test_chat_completions_stream_yields_tokens(self):
        async def mock_stream():
            for line in ['data: {"choices": [{"delta": {"content": "Hello"}}]}',
                         'data: {"choices": [{"delta": {"content": " world"}}]}',
                         'data: [DONE]']:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream

        tokens = []
        async for event in parse_sse_stream(mock_response):
            if "choices" in event and event["choices"]:
                delta = event["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    tokens.append(content)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_handles_done_marker(self):
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Hi"}}]}'
            yield 'data: [DONE]'
            yield 'data: {"choices": [{"delta": {"content": "SHOULD NOT APPEAR"}}]}'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream

        events = []
        async for event in parse_sse_stream(mock_response):
            events.append(event)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_json(self):
        async def mock_stream():
            yield 'data: {invalid json}'
            yield 'data: {"choices": [{"delta": {"content": "OK"}}]}'
            yield 'data: [DONE]'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream

        events = []
        async for event in parse_sse_stream(mock_response):
            events.append(event)

        assert len(events) == 1
        assert events[0]["choices"][0]["delta"]["content"] == "OK"

    @pytest.mark.asyncio
    async def test_stream_ignores_non_data_lines(self):
        async def mock_stream():
            yield ': comment'
            yield ''
            yield 'data: {"test": 1}'
            yield 'data: [DONE]'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream

        events = []
        async for event in parse_sse_stream(mock_response):
            events.append(event)

        assert len(events) == 1
        assert events[0] == {"test": 1}


# ===========================================================================
# Embeddings
# ===========================================================================


def _make_handler(response_json: dict, status_code: int = 200):
    return httpx.MockTransport(lambda req: httpx.Response(status_code, json=response_json))

def _make_binary_handler(content: bytes, status_code: int = 200, headers: dict | None = None):
    return httpx.MockTransport(lambda req: httpx.Response(status_code, content=content, headers=headers))


class TestDistLLMClientEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings_single_input(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "model": "test-model",
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }), base_url="http://test")
        resp = await client.embeddings("hello world")
        assert resp.model == "test-model"
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]
        await client.close()

    @pytest.mark.asyncio
    async def test_embeddings_batch_input(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "model": "test",
            "data": [
                {"index": 0, "embedding": [0.1]},
                {"index": 1, "embedding": [0.2]},
            ],
        }), base_url="http://test")
        resp = await client.embeddings(["text a", "text b"])
        assert len(resp.data) == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_embeddings_returns_valid_response(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "model": "m", "data": [{"index": 0, "embedding": [0.5]}],
            "usage": {"prompt_tokens": 4},
        }), base_url="http://test")
        resp = await client.embeddings("test")
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.5]
        await client.close()


# ===========================================================================
# List Models
# ===========================================================================


class TestDistLLMClientModels:
    @pytest.mark.asyncio
    async def test_list_models_parsed_correctly(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "object": "list",
            "data": [
                {"id": "model-a", "owned_by": "org1"},
                {"id": "model-b", "owned_by": "org2"},
            ],
        }), base_url="http://test")
        ml = await client.list_models()
        assert len(ml.data) == 2
        assert ml.data[0].id == "model-a"
        assert ml.data[1].owned_by == "org2"
        await client.close()


# ===========================================================================
# Completions
# ===========================================================================


class TestDistLLMClientCompletionsExtended:
    @pytest.mark.asyncio
    async def test_completion_returns_text(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "cmp-1", "model": "m",
            "choices": [{"index": 0, "text": "Hello world", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }), base_url="http://test")
        resp = await client.completions("Hi")
        assert resp.choices[0].text == "Hello world"
        await client.close()

    @pytest.mark.asyncio
    async def test_completion_custom_temperature(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "cmp-2", "model": "m",
            "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
        }), base_url="http://test")
        resp = await client.completions("test", temperature=0.5, max_tokens=100)
        assert resp.choices[0].text == "ok"
        await client.close()


# ===========================================================================
# Sync client
# ===========================================================================


class TestDistLLMClientSyncExtended:
    def test_sync_chat_completions(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({
            "id": "chat-1", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"total_tokens": 5},
        }), base_url="http://test")
        resp = client.chat_completions([{"role": "user", "content": "Hi"}])
        assert resp.choices[0].message.content == "Hello"
        assert resp.model == "m"
        client.close()

    def test_sync_completions(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({
            "id": "cmp-sync", "model": "m",
            "choices": [{"index": 0, "text": "sync response", "finish_reason": "stop"}],
        }), base_url="http://test")
        resp = client.completions("test")
        assert resp.choices[0].text == "sync response"
        client.close()

    def test_sync_list_models(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({
            "data": [{"id": "sync-model"}],
        }), base_url="http://test")
        ml = client.list_models()
        assert ml.data[0].id == "sync-model"
        client.close()


# ===========================================================================
# Health Check
# ===========================================================================


class TestDistLLMClientHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "status": "healthy", "model": "test", "nodes": 2,
        }), base_url="http://test")
        resp = await client.health_check()
        assert resp["status"] == "healthy"
        assert resp["nodes"] == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "status": "unhealthy",
        }), base_url="http://test")
        resp = await client.health_check()
        assert resp["status"] == "unhealthy"
        await client.close()


# ===========================================================================
# Batch API
# ===========================================================================


class TestDistLLMClientBatch:
    @pytest.mark.asyncio
    async def test_submit_batch(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "batch-1", "status": "validating",
            "input_file_id": "file-1", "created_at": 1000,
        }), base_url="http://test")
        job = await client.submit_batch("file-1", "/v1/chat/completions")
        assert job.id == "batch-1"
        assert job.status == "validating"
        await client.close()

    @pytest.mark.asyncio
    async def test_get_batch_by_id(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "batch-1", "status": "in_progress",
            "input_file_id": "file-1", "created_at": 1000,
        }), base_url="http://test")
        job = await client.get_batch("batch-1")
        assert job.id == "batch-1"
        assert job.status == "in_progress"
        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_batch(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "batch-1", "status": "cancelled",
            "input_file_id": "file-1", "created_at": 1000,
        }), base_url="http://test")
        job = await client.cancel_batch("batch-1")
        assert job.status == "cancelled"
        await client.close()


# ===========================================================================
# Moderations
# ===========================================================================


class TestDistLLMClientModerations:
    @pytest.mark.asyncio
    async def test_moderations_single_input(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "mod-1", "model": "text-moderation",
            "results": [{
                "flagged": True,
                "categories": {"hate": True, "violence": False},
                "category_scores": {"hate": 0.95, "violence": 0.1},
            }],
        }), base_url="http://test")
        resp = await client.moderations("bad text")
        assert resp.id == "mod-1"
        assert resp.results[0].flagged is True
        assert resp.results[0].categories["hate"] is True
        await client.close()

    @pytest.mark.asyncio
    async def test_moderations_not_flagged(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "mod-2", "model": "text-moderation",
            "results": [{
                "flagged": False,
                "categories": {},
                "category_scores": {},
            }],
        }), base_url="http://test")
        resp = await client.moderations("safe text")
        assert resp.results[0].flagged is False
        await client.close()


# ===========================================================================
# Audio — transcribe
# ===========================================================================


class TestDistLLMClientAudioTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "text": "Hello world",
        }), base_url="http://test")
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            path = f.name
        try:
            resp = await client.transcribe(path)
            assert resp.text == "Hello world"
        finally:
            os.unlink(path)
        await client.close()


# ===========================================================================
# Audio — speech
# ===========================================================================


class TestDistLLMClientAudioSpeech:
    @pytest.mark.asyncio
    async def test_speech_returns_bytes(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_binary_handler(
            b"fake audio bytes",
            headers={"content-type": "audio/mpeg"},
        ), base_url="http://test")
        resp = await client.speech("Hello", model="tts-1", voice="alloy")
        assert isinstance(resp.content, bytes)
        assert len(resp.content) > 0
        assert resp.content_type == "audio/mpeg"
        await client.close()


# ===========================================================================
# Images — generate
# ===========================================================================


class TestDistLLMClientImages:
    @pytest.mark.asyncio
    async def test_generate_images_returns_urls(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "created": 12345,
            "data": [
                {"url": "http://example.com/img1.png"},
                {"url": "http://example.com/img2.png"},
            ],
        }), base_url="http://test")
        resp = await client.generate_images("a cat", n=2)
        assert len(resp.data) == 2
        assert resp.data[0].url == "http://example.com/img1.png"
        assert resp.created == 12345
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_images_b64(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "created": 12346,
            "data": [
                {"b64_json": "aW1hZ2UgZGF0YQ=="},
            ],
        }), base_url="http://test")
        resp = await client.generate_images("a dog", response_format="b64_json")
        assert resp.data[0].b64_json == "aW1hZ2UgZGF0YQ=="
        await client.close()


# ===========================================================================
# Files — upload, list, delete
# ===========================================================================


class TestDistLLMClientFiles:
    @pytest.mark.asyncio
    async def test_upload_file(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "file-1", "filename": "test.jsonl",
            "purpose": "fine-tune", "bytes": 100, "created_at": 1000,
        }), base_url="http://test")
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(b'{"messages": []}\n')
            path = f.name
        try:
            info = await client.upload_file(path, "fine-tune")
            assert info.id == "file-1"
            assert info.purpose == "fine-tune"
        finally:
            os.unlink(path)
        await client.close()

    @pytest.mark.asyncio
    async def test_list_files(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "data": [
                {"id": "f1", "filename": "a.jsonl", "purpose": "fine-tune", "bytes": 50, "created_at": 1},
                {"id": "f2", "filename": "b.jsonl", "purpose": "fine-tune", "bytes": 60, "created_at": 2},
            ],
        }), base_url="http://test")
        files = await client.list_files()
        assert len(files) == 2
        assert files[0].id == "f1"
        assert files[1].id == "f2"
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_file(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "deleted": True,
        }), base_url="http://test")
        deleted = await client.delete_file("file-1")
        assert deleted is True
        await client.close()


# ===========================================================================
# Fine-Tuning — create, list, cancel
# ===========================================================================


class TestDistLLMClientFineTuning:
    @pytest.mark.asyncio
    async def test_create_fine_tuning(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "ft-1", "status": "queued",
            "model": "base-model", "training_file": "file-1",
            "created_at": 1000,
        }), base_url="http://test")
        job = await client.create_fine_tuning("file-1")
        assert job.id == "ft-1"
        assert job.status == "queued"
        await client.close()

    @pytest.mark.asyncio
    async def test_list_fine_tuning(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "data": [
                {"id": "ft-1", "status": "succeeded", "model": "m", "training_file": "f1", "created_at": 1},
                {"id": "ft-2", "status": "failed", "model": "m", "training_file": "f2", "created_at": 2,
                 "error": "OOM"},
            ],
        }), base_url="http://test")
        jobs = await client.list_fine_tuning()
        assert len(jobs) == 2
        assert jobs[1].error == "OOM"
        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_fine_tuning(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "id": "ft-1", "status": "cancelled",
            "model": "m", "training_file": "f1", "created_at": 1,
        }), base_url="http://test")
        job = await client.cancel_fine_tuning("ft-1")
        assert job.status == "cancelled"
        await client.close()


# ===========================================================================
# Retry logic
# ===========================================================================


class TestRetryTransientErrors:
    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={
                "id": "chat-1", "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            })
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        resp = await client.chat_completions([{"role": "user", "content": "hi"}])
        assert resp.choices[0].message.content == "ok"
        assert call_count == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(200, json={
                "id": "cmp-1", "model": "m",
                "choices": [{"text": "recovered"}],
            })
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        resp = await client.completions("test")
        assert resp.choices[0].text == "recovered"
        assert call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_503(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        ml = await client.list_models()
        assert len(ml.data) == 1
        assert call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(503, json={"error": "still down"})
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        with pytest.raises(httpx.HTTPStatusError):
            await client.completions("test")
        assert call_count > 1
        await client.close()

    @pytest.mark.asyncio
    async def test_non_retryable_400_no_retry(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": "bad request"})
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completions([{"role": "user", "content": "hi"}])
        assert call_count == 1
        await client.close()


# ===========================================================================
# Sync client — remaining operations
# ===========================================================================


class TestDistLLMClientSyncOperations:
    def test_sync_health_check(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({"status": "ok"}), base_url="http://test")
        resp = client.health_check()
        assert resp["status"] == "ok"
        client.close()

    def test_sync_retry_on_429(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
        ml = client.list_models()
        assert len(ml.data) == 1
        assert call_count == 2
        client.close()

    def test_sync_non_retryable_400(self):
        call_count = 0
        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(400)
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
        with pytest.raises(httpx.HTTPStatusError):
            client.list_models()
        assert call_count == 1
        client.close()


# ===========================================================================
# Retry — jitter
# ===========================================================================


class TestRetryJitter:
    def test_compute_delay_has_jitter(self):
        cfg = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0)
        delays = [_compute_delay(0, cfg) for _ in range(50)]
        assert max(delays) > min(delays)
        assert all(0.5 <= d <= 1.0 for d in delays)
        assert len(set(round(d, 2) for d in delays)) > 1

    def test_delay_increases_with_attempt(self):
        cfg = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0)
        d0 = _compute_delay(0, cfg)
        d1 = _compute_delay(1, cfg)
        assert d0 * 1.0 <= d1 * 2.0

    def test_delay_capped_at_max(self):
        cfg = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=5.0)
        d10 = _compute_delay(10, cfg)
        assert d10 <= 5.0


# ===========================================================================
# Circuit Breaker
# ===========================================================================


class TestCircuitBreakerClosedToOpen:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_can_execute_when_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_n_failures_opens_circuit(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_cannot_execute_when_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False


class TestCircuitBreakerOpenToHalfOpen:
    def test_transitions_after_recovery_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0))
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.OPEN  # internal state before property access
        assert cb.state == CircuitState.HALF_OPEN  # property auto-transitions

    def test_half_open_can_execute(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0))
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True


class TestCircuitBreakerHalfOpenToClosed:
    def test_successes_close_when_threshold_met(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, success_threshold=2, recovery_timeout=0))
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        cb.can_execute()  # triggers HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, success_threshold=3, recovery_timeout=1))
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        cb._half_open_calls = 0
        cb._success_count = 0
        cb._state = CircuitState.HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb.can_execute() is False


class TestCircuitBreakerRequestRejected:
    def test_open_raises_circuit_breaker_error(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False

    def test_metrics_includes_state(self):
        cb = CircuitBreaker()
        m = cb.get_metrics()
        assert m["state"] == CircuitState.CLOSED.value
        assert m["total_successes"] == 0
        assert m["total_failures"] == 0

    def test_reset_clears(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True


# ===========================================================================
# SSE Parser
# ===========================================================================


class TestSSEParser:
    @pytest.mark.asyncio
    async def test_async_parses_valid_events(self):
        async def mock_lines():
            yield 'data: {"key": "value"}'
            yield 'data: {"num": 42}'
            yield 'data: [DONE]'
            yield 'data: {"should": "not appear"}'
        response = MagicMock()
        response.aiter_lines = mock_lines
        events = []
        async for event in parse_sse_stream(response):
            events.append(event)
        assert len(events) == 2
        assert events[0] == {"key": "value"}
        assert events[1] == {"num": 42}

    @pytest.mark.asyncio
    async def test_async_ignores_non_data_lines(self):
        async def mock_lines():
            yield ':comment'
            yield ''
            yield 'data: {"test": 1}'
            yield 'data: [DONE]'
        response = MagicMock()
        response.aiter_lines = mock_lines
        events = []
        async for event in parse_sse_stream(response):
            events.append(event)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_async_handles_malformed_json(self):
        async def mock_lines():
            yield 'data: {invalid}'
            yield 'data: {"valid": true}'
            yield 'data: [DONE]'
        response = MagicMock()
        response.aiter_lines = mock_lines
        events = []
        async for event in parse_sse_stream(response):
            events.append(event)
        assert len(events) == 1
        assert events[0] == {"valid": True}

    def test_sync_parses_valid_events(self):
        def mock_lines():
            yield 'data: {"msg": "hello"}'
            yield 'data: [DONE]'
        response = MagicMock()
        response.iter_lines = mock_lines
        events = list(parse_sse_stream_sync(response))
        assert len(events) == 1
        assert events[0] == {"msg": "hello"}

    def test_sync_empty_stream(self):
        def mock_lines():
            yield 'data: [DONE]'
        response = MagicMock()
        response.iter_lines = mock_lines
        events = list(parse_sse_stream_sync(response))
        assert len(events) == 0


# ===========================================================================
# Stats tracking
# ===========================================================================


class TestStatsTracking:
    def test_call_recorded_stats_update(self):
        handler = httpx.MockTransport(lambda req: httpx.Response(
            200, json={
                "id": "chat-1", "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            }
        ))
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=handler, base_url="http://test")
        client.chat_completions([{"role": "user", "content": "hi"}])
        stats = client.stats
        assert stats.total_calls >= 1
        assert stats.total_prompt_tokens >= 3
        assert stats.total_completion_tokens >= 5
        assert len(stats.call_log) >= 1
        client.close()

    def test_stats_tokens_per_second(self):
        handler = httpx.MockTransport(lambda req: httpx.Response(
            200, json={
                "id": "cmp-1", "model": "m",
                "choices": [{"text": "hello world"}],
                "usage": {"completion_tokens": 3},
                "generation_time": 0.5,
            }
        ))
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=handler, base_url="http://test")
        client.completions("test")
        stats = client.stats
        tps = stats.tokens_per_second
        assert isinstance(tps, float)

    def test_stats_avg_latency(self):
        handler = httpx.MockTransport(lambda req: httpx.Response(
            200, json={
                "id": "chat-1", "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        ))
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=handler, base_url="http://test")
        client.chat_completions([{"role": "user", "content": "hi"}])
        stats = client.stats
        avg = stats.avg_latency
        assert isinstance(avg, float)

    def test_reset_stats(self):
        handler = httpx.MockTransport(lambda req: httpx.Response(
            200, json={
                "id": "chat-1", "model": "m",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        ))
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=handler, base_url="http://test")
        client.chat_completions([{"role": "user", "content": "hi"}])
        assert client.stats.total_calls > 0
        client.reset_stats()
        assert client.stats.total_calls == 0
        client.close()

    def test_cost_estimation(self):
        from distllm.sdk.types import ClientStats, CallStats
        stats = ClientStats()
        stats.total_completion_tokens = 1000
        stats.total_prompt_tokens = 500
        cost = stats.estimate_cost(price_per_million_input=0.50, price_per_million_output=1.50)
        expected = (500 / 1e6 * 0.50) + (1000 / 1e6 * 1.50)
        assert cost == pytest.approx(expected)

    def test_cost_estimation_zero_tokens(self):
        from distllm.sdk.types import ClientStats
        stats = ClientStats()
        cost = stats.estimate_cost()
        assert cost == 0.0

    def test_cost_estimation_call_stats(self):
        from distllm.sdk.types import CallStats
        cs = CallStats(endpoint="chat", latency=0.5, prompt_tokens=10, completion_tokens=20)
        assert cs.endpoint == "chat"
        assert cs.latency == 0.5
        assert cs.prompt_tokens == 10
        assert cs.completion_tokens == 20


# ===========================================================================
# Client — context manager cleanup
# ===========================================================================


class TestClientContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager_cleans_up(self):
        client = DistLLMClient()
        client._client = httpx.AsyncClient(transport=_make_handler({
            "data": [{"id": "m"}],
        }), base_url="http://test")
        async with client as c:
            ml = await c.list_models()
            assert len(ml.data) == 1

    def test_sync_context_manager_cleans_up(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({
            "data": [{"id": "m"}],
        }), base_url="http://test")
        with client as c:
            ml = c.list_models()
            assert len(ml.data) == 1

    def test_close_is_idempotent(self):
        client = DistLLMClientSync()
        client._client = httpx.Client(transport=_make_handler({"data": []}), base_url="http://test")
        client.close()
        client.close()


# ===========================================================================
# Connection pool config
# ===========================================================================


class TestPoolConfig:
    def test_pool_config_defaults(self):
        p = PoolConfig()
        assert p.max_connections == 100
        assert p.max_keepalive_connections == 20
        assert p.keepalive_expiry == 5.0

    def test_pool_config_custom(self):
        p = PoolConfig(max_connections=50, max_keepalive_connections=10, keepalive_expiry=10.0)
        assert p.max_connections == 50

    @pytest.mark.asyncio
    async def test_async_client_pool_applied(self):
        client = DistLLMClient(pool=PoolConfig(max_connections=10))
        assert client._pool.max_connections == 10
        await client.close()

    def test_sync_client_pool_applied(self):
        client = DistLLMClientSync(pool=PoolConfig(max_connections=20))
        assert client._pool.max_connections == 20
        client.close()


# ===========================================================================
# API key header
# ===========================================================================


class TestApiKeyHeader:
    def test_api_key_sets_bearer_token(self):
        client = DistLLMClient(api_key="my-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-key"
        client.close()

    def test_no_api_key_no_auth_header(self):
        client = DistLLMClient()
        headers = client._build_headers()
        assert "Authorization" not in headers
        client.close()

    def test_content_type_always_present(self):
        client = DistLLMClient()
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_builds_headers(self):
        client = DistLLMClient(api_key="test-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        await client.close()


# ===========================================================================
# Types — dataclasses
# ===========================================================================


class TestTypesDataclasses:
    def test_usage_info_defaults(self):
        from distllm.sdk.types import UsageInfo
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_chat_message(self):
        from distllm.sdk.types import ChatMessage
        m = ChatMessage(role="assistant", content="Hello")
        assert m.role == "assistant"
        assert m.content == "Hello"

    def test_chat_choice_with_delta(self):
        from distllm.sdk.types import ChatChoice
        c = ChatChoice(index=0, delta="hello")
        assert c.delta == "hello"

    def test_chat_choice_with_message(self):
        from distllm.sdk.types import ChatChoice, ChatMessage
        msg = ChatMessage(role="assistant", content="world")
        c = ChatChoice(index=0, message=msg)
        assert c.message.content == "world"

    def test_chat_completion_response(self):
        from distllm.sdk.types import ChatCompletionResponse, ChatChoice
        c = ChatCompletionResponse(id="chat-1", model="m", choices=[ChatChoice(index=0)])
        assert c.id == "chat-1"
        assert c.object == "chat.completion"

    def test_completion_choice(self):
        from distllm.sdk.types import CompletionChoice
        c = CompletionChoice(index=0, text="output")
        assert c.text == "output"

    def test_completion_response(self):
        from distllm.sdk.types import CompletionResponse, CompletionChoice
        c = CompletionResponse(id="cmp-1", model="m", choices=[CompletionChoice(index=0, text="out")])
        assert c.id == "cmp-1"
        assert c.object == "text_completion"

    def test_embedding_object(self):
        from distllm.sdk.types import EmbeddingObject
        e = EmbeddingObject(index=0, embedding=[0.1, 0.2])
        assert len(e.embedding) == 2

    def test_embedding_response(self):
        from distllm.sdk.types import EmbeddingResponse, EmbeddingObject
        r = EmbeddingResponse(model="m", data=[EmbeddingObject(index=0, embedding=[0.5])])
        assert r.model == "m"

    def test_model_info(self):
        from distllm.sdk.types import ModelInfo
        m = ModelInfo(id="test-model")
        assert m.id == "test-model"

    def test_model_list(self):
        from distllm.sdk.types import ModelInfo, ModelList
        ml = ModelList(data=[ModelInfo(id="a"), ModelInfo(id="b")])
        assert len(ml.data) == 2

    def test_usage_info_with_tps(self):
        from distllm.sdk.types import UsageInfo
        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30, tokens_per_second=15.5)
        assert u.tokens_per_second == 15.5


# ===========================================================================
# Types — exception hierarchy
# ===========================================================================


class TestTypesExceptions:
    def test_api_error_defaults(self):
        from distllm.sdk.types import ApiError
        e = ApiError("something broke")
        assert str(e) == "something broke"
        assert e.status_code == 500
        assert e.error_type == "api_error"

    def test_authentication_error_subclass(self):
        from distllm.sdk.types import ApiError, AuthenticationError
        e = AuthenticationError()
        assert isinstance(e, ApiError)
        assert e.status_code == 401
        assert e.error_type == "authentication_error"

    def test_rate_limit_error_subclass(self):
        from distllm.sdk.types import ApiError, RateLimitError
        e = RateLimitError()
        assert isinstance(e, ApiError)
        assert e.status_code == 429
        assert e.error_type == "rate_limit_error"
        assert e.retry_after is None

    def test_rate_limit_with_retry_after(self):
        from distllm.sdk.types import RateLimitError
        e = RateLimitError(retry_after=30.0)
        assert e.retry_after == 30.0

    def test_timeout_error_subclass(self):
        from distllm.sdk.types import ApiError, TimeoutError_
        e = TimeoutError_()
        assert isinstance(e, ApiError)
        assert e.status_code == 504
        assert e.error_type == "timeout_error"

    def test_isinstance_check_works(self):
        from distllm.sdk.types import ApiError, AuthenticationError, RateLimitError, TimeoutError_
        assert issubclass(AuthenticationError, ApiError)
        assert issubclass(RateLimitError, ApiError)
        assert issubclass(TimeoutError_, ApiError)

    def test_custom_request_id(self):
        from distllm.sdk.types import ApiError
        e = ApiError("msg", request_id="req-1")
        assert e.request_id == "req-1"
