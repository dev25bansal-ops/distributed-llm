"""HTTP transport layer with retry and circuit-breaker support.

Extracted from client.py to share retry-loop logic between sync and async clients.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from distllm.constants import DEFAULT_HTTP_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from distllm.sdk.circuit_breaker import CircuitBreaker, CircuitBreakerError


@dataclass
class RetryConfig:
    """Configuration for automatic retries."""

    max_retries: int = MAX_RETRIES
    initial_delay: float = RETRY_DELAY
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
    """Compute exponential backoff delay with jitter."""
    import random

    delay = min(cfg.initial_delay * (cfg.exponential_base**attempt), cfg.max_delay)
    return delay * (0.5 + random.random() * 0.5)  # noqa: S311 - retry jitter


def _check_circuit_breaker(circuit_breaker: CircuitBreaker | None) -> None:
    """Check if the circuit breaker allows execution; raise if open."""
    if circuit_breaker and not circuit_breaker.can_execute():
        raise CircuitBreakerError("Request rejected: circuit breaker is open")


# ---------------------------------------------------------------------------
# _request  -- return parsed JSON dict
# ---------------------------------------------------------------------------


async def _async_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    circuit_breaker: CircuitBreaker | None,
    retry: RetryConfig,
    sleep_fn: Callable[[float], Any],
    **kwargs: Any,
) -> dict:
    """Async HTTP request with retry and circuit breaker, returning parsed JSON."""
    _check_circuit_breaker(circuit_breaker)

    last_exc: BaseException | None = None
    for attempt in range(retry.max_retries + 1):
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            if circuit_breaker:
                circuit_breaker.record_success()
            return response.json()
        except httpx.HTTPStatusError as e:
            if (
                e.response.status_code in retry.retryable_status_codes
                and attempt < retry.max_retries
            ):
                await sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < retry.max_retries:
                await sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
    if circuit_breaker:
        circuit_breaker.record_failure()
    raise last_exc  # type: ignore[misc]


def _sync_request_with_retry(
    client: httpx.Client,
    method: str,
    path: str,
    circuit_breaker: CircuitBreaker | None,
    retry: RetryConfig,
    sleep_fn: Callable[[float], Any],
    **kwargs: Any,
) -> dict:
    """Sync HTTP request with retry and circuit breaker, returning parsed JSON."""
    _check_circuit_breaker(circuit_breaker)

    last_exc: BaseException | None = None
    for attempt in range(retry.max_retries + 1):
        try:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            if circuit_breaker:
                circuit_breaker.record_success()
            return response.json()
        except httpx.HTTPStatusError as e:
            if (
                e.response.status_code in retry.retryable_status_codes
                and attempt < retry.max_retries
            ):
                sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < retry.max_retries:
                sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
    if circuit_breaker:
        circuit_breaker.record_failure()
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _request_raw  -- return raw httpx.Response (for binary endpoints)
# ---------------------------------------------------------------------------


async def _async_request_raw_with_retry(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    circuit_breaker: CircuitBreaker | None,
    retry: RetryConfig,
    sleep_fn: Callable[[float], Any],
    **kwargs: Any,
) -> httpx.Response:
    """Async raw HTTP request with retry and circuit breaker, returning the response object."""
    _check_circuit_breaker(circuit_breaker)

    last_exc: BaseException | None = None
    for attempt in range(retry.max_retries + 1):
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            if circuit_breaker:
                circuit_breaker.record_success()
            return response
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as e:
            if attempt < retry.max_retries:
                await sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
    if circuit_breaker:
        circuit_breaker.record_failure()
    raise last_exc  # type: ignore[misc]


def _sync_request_raw_with_retry(
    client: httpx.Client,
    method: str,
    path: str,
    circuit_breaker: CircuitBreaker | None,
    retry: RetryConfig,
    sleep_fn: Callable[[float], Any],
    **kwargs: Any,
) -> httpx.Response:
    """Sync raw HTTP request with retry and circuit breaker, returning the response object."""
    _check_circuit_breaker(circuit_breaker)

    last_exc: BaseException | None = None
    for attempt in range(retry.max_retries + 1):
        try:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            if circuit_breaker:
                circuit_breaker.record_success()
            return response
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as e:
            if attempt < retry.max_retries:
                sleep_fn(_compute_delay(attempt, retry))
                last_exc = e
                continue
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise
    if circuit_breaker:
        circuit_breaker.record_failure()
    raise last_exc  # type: ignore[misc]


__all__ = [
    "RetryConfig",
    "_compute_delay",
    "_check_circuit_breaker",
    "_async_request_with_retry",
    "_sync_request_with_retry",
    "_async_request_raw_with_retry",
    "_sync_request_raw_with_retry",
]
