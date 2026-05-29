"""Tests for the standalone distllm_sdk package."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from distllm_sdk import constants
from distllm_sdk import errors
from distllm_sdk import types
from distllm_sdk import streaming
from distllm_sdk import circuit_breaker as cb
from distllm_sdk import client as sdk_client
from distllm_sdk import __all__ as public_api


class TestConstants:
    def test_has_timeout(self):
        assert constants.DEFAULT_HTTP_TIMEOUT == 120.0

    def test_has_retries(self):
        assert constants.MAX_RETRIES == 3
        assert constants.RETRY_DELAY == 1.0


class TestErrors:
    def test_api_error_base(self):
        err = errors.ApiError("test error", status_code=400, error_type="invalid", request_id="req-1")
        assert err.message == "test error"
        assert err.status_code == 400
        assert err.error_type == "invalid"
        assert err.request_id == "req-1"
        assert "test error" in str(err)

    def test_authentication_error(self):
        err = errors.AuthenticationError()
        assert err.status_code == 401
        assert err.error_type == "authentication_error"

    def test_rate_limit_error(self):
        err = errors.RateLimitError(retry_after=5.0)
        assert err.status_code == 429
        assert err.retry_after == 5.0

    def test_timeout_error(self):
        err = errors.TimeoutError()
        assert err.status_code == 504
        assert err.error_type == "timeout_error"

    def test_model_not_found_error(self):
        err = errors.ModelNotFoundError("gpt-5")
        assert err.status_code == 404
        assert err.model == "gpt-5"
        assert "gpt-5" in err.message
        assert err.error_type == "model_not_found"

    def test_service_unavailable_error(self):
        err = errors.ServiceUnavailableError(retry_after=30.0)
        assert err.status_code == 503
        assert err.retry_after == 30.0

    def test_invalid_request_error(self):
        err = errors.InvalidRequestError(param="temperature", request_id="req-2")
        assert err.status_code == 400
        assert err.param == "temperature"

    def test_all_error_types_raise(self):
        for exc in [
            errors.AuthenticationError(),
            errors.RateLimitError(),
            errors.TimeoutError(),
            errors.ModelNotFoundError("test"),
            errors.ServiceUnavailableError(),
            errors.InvalidRequestError(),
        ]:
            assert isinstance(exc, errors.ApiError)
            assert isinstance(exc, Exception)


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        c = cb.CircuitBreaker()
        assert c.state == cb.CircuitState.CLOSED
        assert c.can_execute() is True

    def test_opens_after_failures(self):
        c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=3, recovery_timeout=999))
        assert c.state == cb.CircuitState.CLOSED
        c.record_failure()
        assert c.state == cb.CircuitState.CLOSED
        c.record_failure()
        assert c.state == cb.CircuitState.CLOSED
        c.record_failure()
        assert c.state == cb.CircuitState.OPEN
        assert c.can_execute() is False

    def test_recovers_after_timeout(self):
        import time
        c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01))
        c.record_failure()
        assert c.state == cb.CircuitState.OPEN
        time.sleep(0.02)
        assert c.state == cb.CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        import time
        c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01, success_threshold=2))
        c.record_failure()
        time.sleep(0.02)
        assert c.state == cb.CircuitState.HALF_OPEN
        assert c.can_execute() is True
        c.record_success()
        assert c.state == cb.CircuitState.HALF_OPEN
        assert c.can_execute() is True
        c.record_success()
        assert c.state == cb.CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        import time
        c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01))
        c.record_failure()
        time.sleep(0.02)
        assert c.state == cb.CircuitState.HALF_OPEN
        c.record_failure()
        assert c.state == cb.CircuitState.OPEN

    def test_circuit_breaker_error(self):
        err = cb.CircuitBreakerError("test", cb.CircuitState.OPEN)
        assert err.state == cb.CircuitState.OPEN

    def test_get_metrics(self):
        c = cb.CircuitBreaker()
        m = c.get_metrics()
        assert m["state"] == "closed"
        assert m["failure_count"] == 0
        assert m["total_successes"] == 0

    def test_reset(self):
        c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1))
        c.record_failure()
        assert c.state == cb.CircuitState.OPEN
        c.reset()
        assert c.state == cb.CircuitState.CLOSED
        assert c.can_execute() is True


class TestStreaming:
    @pytest.mark.asyncio
    async def test_async_parse_sse(self):
        class FakeAsyncResponse:
            async def aiter_lines(self):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]})
                yield "data: [DONE]"

        results = []
        async for event in streaming.parse_sse_stream_async(FakeAsyncResponse()):
            results.append(event)
        assert len(results) == 1
        assert results[0]["choices"][0]["delta"]["content"] == "hello"

    def test_sync_parse_sse(self):
        class FakeSyncResponse:
            def iter_lines(self):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "world"}}]})
                yield "data: [DONE]"

        results = list(streaming.parse_sse_stream_sync(FakeSyncResponse()))
        assert len(results) == 1
        assert results[0]["choices"][0]["delta"]["content"] == "world"

    def test_parse_ignores_bad_json(self):
        class FakeSyncResponse:
            def iter_lines(self):
                yield "data: not json"
                yield "data: [DONE]"

        results = list(streaming.parse_sse_stream_sync(FakeSyncResponse()))
        assert results == []

    @pytest.mark.asyncio
    async def test_async_parse_ignores_non_data_lines(self):
        class FakeAsyncResponse:
            async def aiter_lines(self):
                yield ":comment"
                yield "event: custom"
                yield "data: " + json.dumps({"x": 1})
                yield "data: [DONE]"

        results = []
        async for event in streaming.parse_sse_stream_async(FakeAsyncResponse()):
            results.append(event)
        assert len(results) == 1


class TestTypes:
    def test_usage_info_defaults(self):
        u = types.UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.tokens_per_second == 0.0

    def test_chat_message(self):
        m = types.ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_chat_completion_request(self):
        msg = types.ChatMessage(role="user", content="hi")
        r = types.ChatCompletionRequest(messages=[msg], temperature=0.5)
        assert r.model == "distributed-llm"
        assert r.temperature == 0.5
        assert r.messages[0].content == "hi"

    def test_chat_completion_response(self):
        msg = types.ChatMessage(role="assistant", content="hello!")
        choice = types.ChatChoice(index=0, message=msg)
        resp = types.ChatCompletionResponse(id="chat-1", model="distributed-llm", choices=[choice])
        assert resp.id == "chat-1"
        assert resp.choices[0].message.content == "hello!"

    def test_completion_response(self):
        choice = types.CompletionChoice(index=0, text="world")
        resp = types.CompletionResponse(id="cmp-1", model="distributed-llm", choices=[choice])
        assert resp.choices[0].text == "world"

    def test_embedding_response(self):
        obj = types.EmbeddingObject(index=0, embedding=[0.1, 0.2, 0.3])
        resp = types.EmbeddingResponse(model="distributed-llm", data=[obj])
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]

    def test_model_list(self):
        info = types.ModelInfo(id="model-1")
        ml = types.ModelList(data=[info])
        assert ml.data[0].id == "model-1"

    def test_batch_job(self):
        job = types.BatchJob(id="batch-1", status="completed", input_file_id="file-1", created_at=1000)
        assert job.status == "completed"

    def test_transcription_response(self):
        r = types.TranscriptionResponse(text="hello world")
        assert r.text == "hello world"

    def test_speech_response(self):
        r = types.SpeechResponse(content=b"audio data")
        assert r.content == b"audio data"

    def test_image_response(self):
        img = types.ImageObject(url="https://example.com/img.png")
        resp = types.ImageGenerationResponse(created=123, data=[img])
        assert resp.data[0].url == "https://example.com/img.png"

    def test_moderation_response(self):
        result = types.ModerationResult(flagged=True, categories={"harassment": True}, category_scores={"harassment": 0.99})
        resp = types.ModerationResponse(id="mod-1", model="distributed-llm", results=[result])
        assert resp.results[0].flagged is True

    def test_file_info(self):
        f = types.FileInfo(id="file-1", filename="train.jsonl", purpose="fine-tune", bytes=1024, created_at=1000)
        assert f.filename == "train.jsonl"

    def test_fine_tuning_job(self):
        job = types.FineTuningJob(id="ft-1", status="running", model="distributed-llm", training_file="file-1", created_at=1000)
        assert job.status == "running"

    def test_client_stats(self):
        stats = types.ClientStats()
        stats.total_calls = 10
        stats.total_latency = 5.0
        stats.total_completion_tokens = 100
        assert stats.tokens_per_second == 20.0
        assert stats.avg_latency == 0.5
        cost = stats.estimate_cost()
        assert cost > 0.0

    def test_api_error_base_type(self):
        err = types.ApiError("base error")
        assert err.status_code == 500
        assert err.error_type == "api_error"


class TestClientHelpers:
    def test_retry_config_defaults(self):
        cfg = sdk_client.RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.initial_delay == 1.0

    def test_pool_config_defaults(self):
        cfg = sdk_client.PoolConfig()
        assert cfg.max_connections == 100

    def test_compute_delay(self):
        cfg = sdk_client.RetryConfig(initial_delay=1.0, max_delay=10.0)
        delay = sdk_client._compute_delay(0, cfg)
        assert 0.5 <= delay <= 1.0
        delay2 = sdk_client._compute_delay(3, cfg)
        assert delay2 <= 10.0

    def test_parse_usage(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, "generation_time": 2.0}
        u = sdk_client._parse_usage(data)
        assert u is not None
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.tokens_per_second == 10.0

    def test_parse_usage_none(self):
        assert sdk_client._parse_usage({}) is None

    def test_build_chat_payload(self):
        payload = sdk_client._BaseClient._build_chat_payload(
            [{"role": "user", "content": "hi"}], "test-model", 0.5, 0.8, 100, False, response_format={"type": "json_object"}
        )
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["stream"] is False
        assert payload["response_format"] == {"type": "json_object"}

    def test_build_chat_payload_with_logprobs(self):
        payload = sdk_client._BaseClient._build_chat_payload(
            [{"role": "user", "content": "hi"}], "test-model", 0.7, 0.9, 256, False, logprobs={"top_n": 5}
        )
        assert payload["logprobs"] is True
        assert payload["top_logprobs"] == 5

    def test_build_chat_payload_stream_with_usage(self):
        payload = sdk_client._BaseClient._build_chat_payload(
            [{"role": "user", "content": "hi"}], "test-model", 0.7, 0.9, 256, True, include_usage=True
        )
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}


def _make_sync_mock_response(status_code: int, json_data: dict):
    """Create a plain object that mimics httpx.Response sync methods."""
    class FakeResponse:
        def __init__(self):
            self.status_code = status_code
            self.headers = {}
        def json(self):
            return json_data
        def raise_for_status(self):
            if status_code >= 400:
                raise __import__("httpx").HTTPStatusError(
                    f"HTTP {status_code}", request=None, response=self
                )
    return FakeResponse()


class TestClientIntegration:
    """Integration tests with mocked HTTP server."""

    @pytest.fixture
    def mock_async_client(self):
        client = sdk_client.DistLLMClient(base_url="http://test:8000", api_key="test-key", timeout=5.0)
        return client

    @pytest.mark.asyncio
    async def test_health_check(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(200, {"status": "ok"})
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        result = await mock_async_client.health_check()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_list_models(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(200, {"data": [{"id": "model-1", "owned_by": "distributed-llm"}]})
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        result = await mock_async_client.list_models()
        assert result.data[0].id == "model-1"

    @pytest.mark.asyncio
    async def test_chat_completions(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(200, {
            "id": "chat-1", "model": "distributed-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "created": 1000,
        })
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        result = await mock_async_client.chat_completions(messages=[{"role": "user", "content": "Hi"}])
        assert result.choices[0].message.content == "Hello!"
        assert result.model == "distributed-llm"

    @pytest.mark.asyncio
    async def test_embeddings(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(200, {
            "model": "distributed-llm",
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        })
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        result = await mock_async_client.embeddings(input="hello")
        assert result.data[0].embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_chat_stream(self, mocker, mock_async_client):
        class FakeStreamResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            async def aiter_lines(self):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
                yield "data: [DONE]"

        mock_async_client._client.stream = mocker.MagicMock()
        mock_async_client._client.stream.return_value.__aenter__.return_value = FakeStreamResponse()

        chunks = []
        async for chunk in mock_async_client.chat_completions_stream(messages=[{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        assert chunks == ["Hello"]

    @pytest.mark.asyncio
    async def test_authentication_error(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(401, {"error": {"message": "Invalid API key"}})
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        with pytest.raises(errors.AuthenticationError) as exc_info:
            await mock_async_client.health_check()
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, mocker, mock_async_client):
        mock_resp = _make_sync_mock_response(429, {})
        mock_async_client._client.request = mocker.AsyncMock(return_value=mock_resp)

        with pytest.raises(errors.RateLimitError) as exc_info:
            await mock_async_client.health_check()
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_retry_then_success(self, mocker, mock_async_client):
        mock_async_client._retry.max_retries = 1
        mock_async_client._retry.retryable_status_codes = (503,)
        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_sync_mock_response(503, {})
            return _make_sync_mock_response(200, {"status": "ok"})

        mock_async_client._client.request = mock_request
        with mocker.patch("asyncio.sleep", mocker.AsyncMock()):
            result = await mock_async_client.health_check()
        assert result == {"status": "ok"}
        assert call_count[0] == 2


def test_public_api_exports():
    expected = {
        "DistLLMClient", "DistLLMClientSync", "RetryConfig", "PoolConfig",
        "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerError", "CircuitState",
        "ChatCompletionResponse", "ChatMessage", "ChatChoice",
        "CompletionResponse", "CompletionChoice",
        "ModelInfo", "ModelList",
        "EmbeddingResponse", "EmbeddingObject",
        "BatchJob", "BatchList",
        "TranscriptionResponse", "SpeechResponse",
        "ImageGenerationResponse", "ImageObject",
        "ModerationResponse", "ModerationResult",
        "FileInfo", "FineTuningJob",
        "UsageInfo", "ClientStats", "CallStats",
        "ApiError", "AuthenticationError", "RateLimitError", "TimeoutError",
        "ModelNotFoundError", "ServiceUnavailableError", "InvalidRequestError",
    }
    assert set(public_api) == expected
