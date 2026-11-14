"""Regression tests for the async-close leak in ``TestApiKeyHeader`` (Division C).

``tests/sdk/test_client.py::TestApiKeyHeader`` instantiates the *async*
``DistLLMClient`` but calls ``client.close()`` without ``await`` in three
tests (test_api_key_sets_bearer_token, test_no_api_key_no_auth_header,
test_content_type_always_present).  ``close()`` is a coroutine, so it never
runs and the underlying ``httpx.AsyncClient`` transport is never released;
pytest reports ``RuntimeWarning: coroutine 'DistLLMClient.close' was never
awaited`` at those call sites.

These tests restate the same header contract against the async client while
closing it correctly (try/finally and ``async with``), and pin down the
close contract itself:

* ``close()`` sets ``httpx.AsyncClient.is_closed`` to True (transport released)
* ``close()`` awaits ``httpx.AsyncClient.aclose`` exactly once (not skipped)
* ``close()`` is idempotent (a second awaited close does not raise)
* a request after close raises RuntimeError instead of silently proceeding

Verified against httpx 0.28.1, where ``AsyncClient.is_closed`` exists and a
request after ``aclose()`` raises ``RuntimeError: Cannot send a request, as
the client has been closed.``
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from distllm.sdk.client import DistLLMClient

pytestmark = pytest.mark.sdk


_MODELS_PAYLOAD = {"data": [{"id": "m-1", "owned_by": "distributed-llm"}]}


class _RecordingTransport:
    """MockTransport handler that records every outgoing request."""

    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self.requests: list[httpx.Request] = []
        self._payload = _MODELS_PAYLOAD if payload is None else payload
        self._status_code = status_code

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self._status_code, json=self._payload)

    @property
    def last_headers(self) -> httpx.Headers:
        assert self.requests, "no request reached the transport"
        return self.requests[-1].headers


def _make_client(api_key: str | None = None):
    """Build an async DistLLMClient wired to a recording httpx.MockTransport.

    The transport is swapped on the client's real ``httpx.AsyncClient`` so the
    headers, timeout, and pool limits configured by ``DistLLMClient.__init__``
    stay in effect -- the wire-level headers therefore prove the client's own
    header construction, not a test-supplied override.
    """
    client = DistLLMClient(api_key=api_key)
    handler = _RecordingTransport()
    # httpx 0.28.1 exposes the active transport as AsyncClient._transport;
    # swapping it leaves the client object (and its close state) intact, so
    # close() can be exercised on exactly the client under test.
    assert hasattr(client._client, "_transport"), "httpx AsyncClient has no _transport"
    client._client._transport = httpx.MockTransport(handler)
    return client, handler


# ===========================================================================
# Header behaviour (async client, close correctly awaited)
# ===========================================================================


class TestAsyncApiKeyHeader:
    @pytest.mark.asyncio
    async def test_api_key_sets_bearer_token(self):
        client, handler = _make_client(api_key="my-key")
        try:
            headers = client._build_headers()
            assert headers["Authorization"] == "Bearer my-key"
            await client.list_models()
            assert handler.last_headers["authorization"] == "Bearer my-key"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_no_api_key_no_auth_header(self):
        client, handler = _make_client()
        try:
            headers = client._build_headers()
            assert "Authorization" not in headers
            await client.list_models()
            assert "authorization" not in handler.last_headers
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_content_type_always_present(self):
        client, handler = _make_client()
        try:
            headers = client._build_headers()
            assert headers["Content-Type"] == "application/json"
            await client.list_models()
            assert handler.last_headers["content-type"] == "application/json"
        finally:
            await client.close()


# ===========================================================================
# Async close contract: transport actually released, close actually awaited
# ===========================================================================


class TestAsyncCloseContract:
    @pytest.mark.asyncio
    async def test_close_releases_transport(self):
        client, _ = _make_client(api_key="k")
        try:
            assert client._client.is_closed is False
            await client.list_models()  # transport usable before close
        finally:
            await client.close()
        assert client._client.is_closed is True

    @pytest.mark.asyncio
    async def test_close_awaits_aclose_exactly_once(self):
        client, _ = _make_client()
        try:
            real_aclose = client._client.aclose
            with patch.object(
                client._client, "aclose", new=AsyncMock(wraps=real_aclose)
            ) as aclose_spy:
                await client.close()
            assert aclose_spy.await_count == 1
            assert client._client.is_closed is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        client, _ = _make_client()
        await client.close()
        await client.close()
        assert client._client.is_closed is True

    @pytest.mark.asyncio
    async def test_request_after_close_raises(self):
        client, _ = _make_client()
        await client.close()
        with pytest.raises(RuntimeError, match="closed"):
            await client.list_models()

    @pytest.mark.asyncio
    async def test_async_context_manager_releases_transport(self):
        client, _ = _make_client(api_key="ctx-key")
        async with client as c:
            assert c is client
            await c.list_models()
            assert c._client.is_closed is False
        assert client._client.is_closed is True
