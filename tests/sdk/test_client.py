"""Tests for DistLLM SDK client and types."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import httpx

from distllm.sdk.client import _BaseClient, DistLLMClient, DistLLMClientSync
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
