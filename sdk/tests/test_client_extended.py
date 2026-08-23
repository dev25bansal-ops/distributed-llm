"""Extended tests for the HTTP client (distllm_sdk.client).

Covers circuit-breaker threading/concurrency, 429 Retry-After backoff, large
payloads and concurrent calls. All network access is mocked via a fake
httpx.Response and a mocked ``client._client.request`` transport.
"""

import asyncio
import json
import sys
import threading

import httpx
import pytest

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))

from distllm_sdk import client as sdk_client
from distllm_sdk import circuit_breaker as cb
from distllm_sdk import errors


# --------------------------------------------------------------------------- #
# Fake httpx response
# --------------------------------------------------------------------------- #
def _make_resp(status_code, json_data, headers=None):
    class FakeResponse:
        def __init__(self):
            self.status_code = status_code
            self.headers = headers or {}
            self._json = json_data

        def json(self):
            return self._json

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}", request=None, response=self
                )

    return FakeResponse()


def _chat_ok():
    return _make_resp(200, {
        "id": "chat-1", "model": "distributed-llm", "created": 1,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                     "finish_reason": "stop"}],
    })


def _embed_ok():
    return _make_resp(200, {"model": "distributed-llm",
                            "data": [{"index": 0, "embedding": [0.1, 0.2]}]})


def _completion_ok():
    return _make_resp(200, {"id": "c1", "model": "distributed-llm", "created": 1,
                            "choices": [{"index": 0, "text": "world", "finish_reason": "stop"}]})


@pytest.fixture
def client():
    c = sdk_client.DistLLMClient(base_url="http://test:8000", api_key="k", timeout=5.0)
    return c


# --------------------------------------------------------------------------- #
# Circuit breaker — direct threading / concurrency
# --------------------------------------------------------------------------- #
def test_cb_concurrent_record_failure():
    c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1000))
    threads = [threading.Thread(target=c.record_failure) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.state == cb.CircuitState.CLOSED
    assert c.get_metrics()["total_failures"] == 50
    assert c.get_metrics()["failure_count"] == 50


def test_cb_concurrent_record_success():
    c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1000))
    threads = [threading.Thread(target=c.record_success) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.get_metrics()["total_successes"] == 50


def test_cb_half_open_concurrent_admission():
    import time
    c = cb.CircuitBreaker(cb.CircuitBreakerConfig(
        failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=3))
    c.record_failure()
    assert c.state == cb.CircuitState.OPEN
    time.sleep(0.02)
    assert c.state == cb.CircuitState.HALF_OPEN

    results = []
    lock = threading.Lock()

    def worker():
        with lock:
            results.append(c.can_execute())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 3
    assert results.count(False) == 17
    assert c.state == cb.CircuitState.HALF_OPEN


def test_cb_reset_threadsafe():
    c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=2))

    def worker():
        for _ in range(5):
            c.reset()
            c.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c.reset()
    assert c.state == cb.CircuitState.CLOSED


def test_cb_metrics_threadsafe_counts():
    c = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=1000, success_threshold=1000))

    def worker():
        c.record_failure()
        c.record_success()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m = c.get_metrics()
    assert m["total_failures"] == 10
    assert m["total_successes"] == 10


# --------------------------------------------------------------------------- #
# Circuit breaker — client integration
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_client_cb_opens_after_failures(client, mocker):
    client._retry.max_retries = 0
    client._circuit_breaker = cb.CircuitBreaker(
        cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=999))
    client._client.request = mocker.AsyncMock(
        return_value=_make_resp(500, {"error": {"message": "fail"}}))

    with pytest.raises(errors.ApiError):
        await client.health_check()

    calls_before = client._client.request.call_count
    # Once open, the breaker must reject without touching the transport.
    with pytest.raises(cb.CircuitBreakerError):
        await client.health_check()
    assert client._client.request.call_count == calls_before


@pytest.mark.asyncio
async def test_client_cb_allows_after_reset(client, mocker):
    client._retry.max_retries = 0
    client._circuit_breaker = cb.CircuitBreaker(
        cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=999))
    client._client.request = mocker.AsyncMock(
        return_value=_make_resp(500, {"error": {"message": "fail"}}))
    with pytest.raises(errors.ApiError):
        await client.health_check()

    client.circuit_breaker.reset()
    client._client.request.return_value = _make_resp(200, {"status": "ok"})
    result = await client.health_check()
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_client_cb_records_success_on_200(client, mocker):
    client._circuit_breaker = cb.CircuitBreaker(cb.CircuitBreakerConfig(failure_threshold=5))
    client._client.request = mocker.AsyncMock(return_value=_make_resp(200, {"status": "ok"}))
    await client.health_check()
    assert client.circuit_breaker.get_metrics()["total_successes"] >= 1


# --------------------------------------------------------------------------- #
# 429 Retry-After backoff
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_429_retry_after_backoff(client, mocker):
    sleep_calls = []
    async def _sleep(d):
        sleep_calls.append(d)
    mocker.patch("asyncio.sleep", _sleep)

    client._retry.max_retries = 1
    client._client.request = mocker.AsyncMock(side_effect=[
        _make_resp(429, {}, headers={"retry-after": "0.01"}),
        _make_resp(200, {"status": "ok"}),
    ])
    result = await client.health_check()
    assert result == {"status": "ok"}
    assert client._client.request.call_count == 2
    assert 0.25 <= sleep_calls[0] <= 0.5  # max(0.5, 0.01) * jitter


@pytest.mark.asyncio
async def test_429_retry_after_invalid_falls_back(client, mocker):
    sleep_calls = []
    async def _sleep(d):
        sleep_calls.append(d)
    mocker.patch("asyncio.sleep", _sleep)

    client._retry.max_retries = 1
    client._client.request = mocker.AsyncMock(side_effect=[
        _make_resp(429, {}, headers={"retry-after": "not-a-number"}),
        _make_resp(200, {"status": "ok"}),
    ])
    result = await client.health_check()
    assert result == {"status": "ok"}
    assert len(sleep_calls) == 1


@pytest.mark.asyncio
async def test_429_exhausts_retries_raises(client, mocker):
    client._retry.max_retries = 2
    client._client.request = mocker.AsyncMock(
        side_effect=[_make_resp(429, {}) for _ in range(3)])
    with pytest.raises(errors.RateLimitError) as exc:
        await client.health_check()
    assert exc.value.status_code == 429
    assert client._client.request.call_count == 3


@pytest.mark.asyncio
async def test_429_non_retryable_no_retry(client, mocker):
    client._retry = sdk_client.RetryConfig(retryable_status_codes=(500, 502, 503, 504))
    client._client.request = mocker.AsyncMock(return_value=_make_resp(429, {}))
    with pytest.raises(errors.RateLimitError):
        await client.health_check()
    assert client._client.request.call_count == 1


def test_compute_delay_with_retry_after_header():
    cfg = sdk_client.RetryConfig(initial_delay=1.0, max_delay=10.0)
    delay = sdk_client._compute_delay(0, cfg, {"retry-after": "2"})
    assert 1.0 <= delay <= 2.0


def test_compute_delay_exponential_no_header():
    cfg = sdk_client.RetryConfig(initial_delay=1.0, max_delay=10.0, exponential_base=2.0)
    delay = sdk_client._compute_delay(3, cfg)
    assert 4.0 <= delay <= 8.0  # min(1*2^3, 10) * jitter


# --------------------------------------------------------------------------- #
# Large payloads
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_large_chat_payload(client, mocker):
    big = "x" * 100_000
    client._client.request = mocker.AsyncMock(return_value=_chat_ok())
    result = await client.chat_completions(messages=[{"role": "user", "content": big}])
    assert result.choices[0].message.content == "Hello"
    sent = client._client.request.call_args.kwargs["json"]
    assert sent["messages"][0]["content"] == big


@pytest.mark.asyncio
async def test_large_embeddings_batch(client, mocker):
    batch = [f"doc-{i}" for i in range(1000)]
    client._client.request = mocker.AsyncMock(return_value=_embed_ok())
    result = await client.embeddings(input=batch)
    assert result.data[0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_large_completion_prompt(client, mocker):
    big = "y" * 50_000
    client._client.request = mocker.AsyncMock(return_value=_completion_ok())
    result = await client.completions(prompt=big)
    assert result.choices[0].text == "world"
    sent = client._client.request.call_args.kwargs["json"]
    assert sent["prompt"] == big


# --------------------------------------------------------------------------- #
# Concurrent calls
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_concurrent_chat_calls(client, mocker):
    client._client.request = mocker.AsyncMock(side_effect=[_chat_ok() for _ in range(10)])
    tasks = [client.chat_completions(messages=[{"role": "user", "content": "hi"}])
             for _ in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert all(r.choices[0].message.content == "Hello" for r in results)


@pytest.mark.asyncio
async def test_concurrent_calls_circuit_breaker_open(client, mocker):
    client._circuit_breaker = cb.CircuitBreaker(
        cb.CircuitBreakerConfig(failure_threshold=1, recovery_timeout=999))
    client._client.request = mocker.AsyncMock(return_value=_make_resp(200, {"status": "ok"}))
    client.circuit_breaker.record_failure()  # open the breaker

    tasks = [client.health_check() for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, cb.CircuitBreakerError) for r in results)
    assert client._client.request.call_count == 0


@pytest.mark.asyncio
async def test_concurrent_mixed_results(client, mocker):
    client._retry.max_retries = 0  # fail fast, no retries
    responses = [_chat_ok() if i % 2 == 0 else _make_resp(500, {"error": {"message": "x"}})
                 for i in range(10)]
    client._client.request = mocker.AsyncMock(side_effect=responses)

    tasks = [client.chat_completions(messages=[{"role": "user", "content": "hi"}])
             for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(successes) == 5
    assert len(failures) == 5


# --------------------------------------------------------------------------- #
# Timeout / connect error retries
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_timeout_exception_retries_then_success(client, mocker):
    sleep_calls = []
    async def _sleep(d):
        sleep_calls.append(d)
    mocker.patch("asyncio.sleep", _sleep)

    client._retry.max_retries = 2
    client._client.request = mocker.AsyncMock(side_effect=[
        httpx.TimeoutException("t1"),
        httpx.TimeoutException("t2"),
        _make_resp(200, {"status": "ok"}),
    ])
    result = await client.health_check()
    assert result == {"status": "ok"}
    assert client._client.request.call_count == 3
    assert len(sleep_calls) == 2


@pytest.mark.asyncio
async def test_connect_error_retries_then_success(client, mocker):
    sleep_calls = []
    async def _sleep(d):
        sleep_calls.append(d)
    mocker.patch("asyncio.sleep", _sleep)

    client._retry.max_retries = 2
    client._client.request = mocker.AsyncMock(side_effect=[
        httpx.ConnectError("c1"),
        httpx.ConnectError("c2"),
        _make_resp(200, {"status": "ok"}),
    ])
    result = await client.health_check()
    assert result == {"status": "ok"}
    assert client._client.request.call_count == 3


# --------------------------------------------------------------------------- #
# Differential privacy disabled passthrough
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dp_disabled_payload_passthrough(client, mocker):
    client._client.request = mocker.AsyncMock(return_value=_chat_ok())
    await client.chat_completions(messages=[{"role": "user", "content": "hi"}])
    sent = client._client.request.call_args.kwargs["json"]
    assert sent["model"] == "distributed-llm"
    assert sent["messages"][0]["content"] == "hi"
