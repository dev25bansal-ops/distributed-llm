"""Tests for TGI backend adapter."""
import httpx
import pytest
from distllm.backends.tgi_backend import TGIBackendAdapter, TGIBackendConfig


class _StubPostResponse:
    """Stub for httpx response with .json(), .raise_for_status()."""
    def __init__(self, json_data=None):
        self._json = json_data or {"generated_text": "Hello"}

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class _StubAsyncClient:
    """Stub for httpx.AsyncClient used as async context manager."""
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        return _StubPostResponse()

    async def stream(self, method, url, **kwargs):
        return _StubStreamResponse()


class _StubStreamResponse:
    """Stub for streaming response."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        return
        yield  # pragma: no cover


class TestTGIBackendAdapter:
    def test_config_defaults(self):
        adapter = TGIBackendAdapter()
        assert adapter.config.base_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_generate(self, monkeypatch):
        monkeypatch.setattr(httpx, "AsyncClient", _StubAsyncClient)
        adapter = TGIBackendAdapter()
        result = await adapter.generate("Hi")
        assert result["text"] == "Hello"
