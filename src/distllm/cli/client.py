"""Unified HTTP client for the DistLLM API.

Eliminates 15+ duplications of the ``try/except/httpx`` pattern across CLI
modules. Provides a single consistent interface for all API calls with
retry, error handling, and response parsing.

Usage::

    from distllm.cli.client import DistLLMClient

    client = DistLLMClient(base_url="http://localhost:8000", api_key="...")
    result = client.get("/v1/models")
    result = client.post("/v1/chat/completions", json={"model": "..."}, timeout=60.0)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ClientConfig:
    """Configuration for the DistLLM API client.

    Args:
        base_url: Coordinator API base URL.
        api_key: API key for authentication.
        timeout: Default request timeout in seconds.
        max_retries: Number of retries on 5xx errors.
        retry_delay: Base delay for exponential backoff (seconds).
    """
    base_url: str = ""
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0


class DistLLMError(Exception):
    """Base exception for DistLLM API errors."""
    def __init__(self, message: str, status_code: int = 0, response: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class DistLLMClient:
    """Unified HTTP client for DistLLM coordinator API.

    Provides ``get``, ``post``, ``put``, ``delete`` methods with
    automatic retry, error handling, and response parsing.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        config: ClientConfig | None = None,
    ):
        if config:
            self._config = config
        else:
            self._config = ClientConfig(
                base_url=base_url or os.environ.get("DISTLLM_API_URL", "http://localhost:8000"),
                api_key=api_key or os.environ.get("DISTLLM_API_KEY", ""),
            )

        self._session: Any = None  # httpx.Client, lazily created
        self._async_session: Any = None

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _get_session(self) -> Any:
        """Lazy-initialize httpx.Client for connection pooling."""
        if self._session is None:
            import httpx
            self._session = httpx.Client(
                base_url=self._config.base_url,
                headers=self._headers,
                timeout=self._config.timeout,
            )
        return self._session

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method.
            path: URL path (e.g. ``/v1/models``).
            **kwargs: Passed to httpx.Client.request().

        Returns:
            Parsed JSON response.

        Raises:
            DistLLMError: On HTTP error after exhausting retries.
        """
        session = self._get_session()
        timeout = kwargs.pop("timeout", self._config.timeout)

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                resp = session.request(method, path, timeout=timeout, **kwargs)

                if resp.status_code == 429:
                    # Rate limited — wait and retry
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    time.sleep(retry_after)
                    continue

                if 200 <= resp.status_code < 300:
                    try:
                        return resp.json()
                    except (json.JSONDecodeError, ValueError):
                        return resp.text

                # Handle error responses
                try:
                    detail = resp.json()
                except (json.JSONDecodeError, ValueError):
                    detail = resp.text

                if 500 <= resp.status_code < 600 and attempt < self._config.max_retries:
                    delay = self._config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

                raise DistLLMError(
                    f"{method} {path} returned {resp.status_code}: {detail}",
                    status_code=resp.status_code,
                    response=detail,
                )

            except DistLLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay * (2 ** attempt)
                    logger.debug(f"Request failed (attempt {attempt + 1}): {e}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise DistLLMError(f"Request failed after {self._config.max_retries + 1} attempts: {e}") from e

        raise DistLLMError(f"Request failed after retries: {last_error}")  # pragma: no cover

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, path: str, **kwargs: Any) -> Any:
        """Send a GET request."""
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        """Send a POST request."""
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        """Send a PUT request."""
        return self._request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        """Send a DELETE request."""
        return self._request("DELETE", path, **kwargs)

    # ── Async ─────────────────────────────────────────────────────────────

    async def _get_async_session(self) -> Any:
        """Lazy-initialize httpx.AsyncClient."""
        if self._async_session is None:
            import httpx
            self._async_session = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=self._headers,
                timeout=self._config.timeout,
            )
        return self._async_session

    async def async_get(self, path: str, **kwargs: Any) -> Any:
        """Send an async GET request."""
        session = await self._get_async_session()
        resp = await session.get(path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def async_post(self, path: str, **kwargs: Any) -> Any:
        """Send an async POST request."""
        session = await self._get_async_session()
        resp = await session.post(path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    async def aclose(self) -> None:
        """Close the underlying async HTTP session."""
        if self._async_session:
            try:
                await self._async_session.aclose()
            except Exception:
                pass
            self._async_session = None

    def __enter__(self) -> DistLLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
