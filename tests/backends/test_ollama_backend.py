"""Tests for Ollama backend adapter."""
import httpx
import pytest
from distllm.backends.ollama_backend import OllamaBackendAdapter


class _StubAsyncPost:
    """Stub response with .json() and .raise_for_status()."""
    def __init__(self, json_data=None):
        self._json = json_data or {"response": "Hello from Ollama"}

    async def json(self):
        return self._json

    def raise_for_status(self):
        pass


class _StubAsyncClient:
    """Stub for httpx.AsyncClient used as async context manager."""
    def __init__(self, **kwargs):
        self._timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        return _StubAsyncPost()

    async def stream(self, method, url, **kwargs):
        return _StubAsyncResponseStream()


class _StubAsyncResponseStream:
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


class TestOllamaBackendAdapter:
    def test_config_defaults(self):
        adapter = OllamaBackendAdapter()
        assert adapter.config.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_generate(self, monkeypatch):
        monkeypatch.setattr(httpx, "AsyncClient", _StubAsyncClient)
        adapter = OllamaBackendAdapter()
        result = await adapter.generate("Hi")
        assert "Ollama" in result["text"]
