"""Integration tests for gateway with mock HTTP backends.

Spins up real aiohttp web servers simulating vLLM, TGI, and Ollama,
then routes requests through GatewayRouter to verify fallback chains,
health checking, and model listing with actual HTTP traffic.
"""

import asyncio
import json
import time
from typing import Optional

import pytest
from aiohttp import web

from distllm.gateway.models import BackendConfig, BackendType, GatewayConfig, ModelRoute
from distllm.gateway.router import GatewayRouter


class MockBackendServer:
    """An aiohttp web server that mimics a model backend (vLLM/TGI/Ollama)."""

    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.port: int = 0
        self.requests_log: list[dict] = []
        self._fail_chat = False
        self._fail_health = False
        self._latency_ms = 0

        self.app.router.add_post("/v1/chat/completions", self._handle_chat_completion)
        self.app.router.add_get("/v1/models", self._handle_list_models)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_post("/api/chat", self._handle_ollama_chat)
        self.app.router.add_get("/api/tags", self._handle_ollama_tags)
        self.app.router.add_get("/", self._handle_health)

    def set_fail_chat(self, fail: bool = True):
        self._fail_chat = fail

    def set_fail_health(self, fail: bool = True):
        self._fail_health = fail

    def set_latency(self, ms: float):
        self._latency_ms = ms

    async def _delay(self):
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        self.requests_log.append({"endpoint": "chat", "method": request.method})
        await self._delay()
        if self._fail_chat:
            return web.Response(text="Service Unavailable", status=503)
        body = await request.json()
        return web.json_response({
            "id": f"{self.backend_type.value}-resp",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"reply from {self.backend_type.value}",
                },
                "finish_reason": "stop",
            }],
        })

    async def _handle_list_models(self, request: web.Request) -> web.Response:
        self.requests_log.append({"endpoint": "models", "method": request.method})
        names = {
            BackendType.VLLM: ["llama-3-8b", "mistral-7b"],
            BackendType.TGI: ["llama-3-70b"],
            BackendType.OLLAMA: ["llama3", "mistral"],
        }
        data = [{"id": m} for m in names.get(self.backend_type, [])]
        return web.json_response({"data": data})

    async def _handle_health(self, request: web.Request) -> web.Response:
        self.requests_log.append({"endpoint": "health", "method": request.method})
        await self._delay()
        if self._fail_health:
            return web.Response(text="Unhealthy", status=503)
        return web.json_response({"status": "ok"})

    async def _handle_ollama_chat(self, request: web.Request) -> web.Response:
        self.requests_log.append({"endpoint": "ollama-chat", "method": request.method})
        await self._delay()
        if self._fail_chat:
            return web.Response(text="Service Unavailable", status=503)
        return web.json_response({
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "ollama reply"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20,
        })

    async def _handle_ollama_tags(self, request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": "llama3"}, {"name": "mistral"}]})

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def clear_log(self):
        self.requests_log.clear()


@pytest.fixture
async def mock_vllm():
    srv = MockBackendServer(BackendType.VLLM)
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def mock_tgi():
    srv = MockBackendServer(BackendType.TGI)
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def mock_ollama():
    srv = MockBackendServer(BackendType.OLLAMA)
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
class TestGatewayMockBackendIntegration:

    async def test_route_to_vllm_backend(self, mock_vllm):
        config = GatewayConfig(
            backends=[BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url)],
            routes=[ModelRoute(model_name="test-model", primary_backend="v1")],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            result = await router.route_chat_completion({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
            assert result["id"] == "vllm-resp"
            assert result["choices"][0]["message"]["content"] == "reply from vllm"
            assert len(mock_vllm.requests_log) >= 1
        finally:
            await router.stop()

    async def test_route_to_tgi_backend(self, mock_tgi):
        config = GatewayConfig(
            backends=[BackendConfig(name="t1", backend_type=BackendType.TGI, base_url=mock_tgi.base_url)],
            routes=[ModelRoute(model_name="test-model", primary_backend="t1")],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            result = await router.route_chat_completion({"model": "test-model"})
            assert result["id"] == "tgi-resp"
        finally:
            await router.stop()

    async def test_route_to_ollama_backend(self, mock_ollama):
        config = GatewayConfig(
            backends=[BackendConfig(name="o1", backend_type=BackendType.OLLAMA, base_url=mock_ollama.base_url)],
            routes=[ModelRoute(model_name="test-model", primary_backend="o1")],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            result = await router.route_chat_completion({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
            assert result["id"] == "2024-01-01T00:00:00Z"
            assert result["choices"][0]["message"]["content"] == "ollama reply"
        finally:
            await router.stop()

    async def test_fallback_chain_primary_fails(self, mock_vllm, mock_tgi):
        mock_vllm.set_fail_chat(True)
        config = GatewayConfig(
            backends=[
                BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url),
                BackendConfig(name="t1", backend_type=BackendType.TGI, base_url=mock_tgi.base_url),
            ],
            routes=[ModelRoute(model_name="test-model", primary_backend="v1", fallback_chain=["t1"])],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            result = await router.route_chat_completion({"model": "test-model"})
            assert result["id"] == "tgi-resp"
            assert len(mock_vllm.requests_log) >= 1
            assert len(mock_tgi.requests_log) >= 1
        finally:
            await router.stop()

    async def test_all_backends_fail_raises(self, mock_vllm, mock_tgi):
        mock_vllm.set_fail_chat(True)
        mock_tgi.set_fail_chat(True)
        config = GatewayConfig(
            backends=[
                BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url),
                BackendConfig(name="t1", backend_type=BackendType.TGI, base_url=mock_tgi.base_url),
            ],
            routes=[ModelRoute(model_name="test-model", primary_backend="v1", fallback_chain=["t1"])],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            with pytest.raises(RuntimeError, match="No backend succeeded"):
                await router.route_chat_completion({"model": "test-model"})
        finally:
            await router.stop()

    async def test_health_detects_unhealthy_backend(self, mock_vllm, mock_tgi):
        mock_vllm.set_fail_health(True)
        config = GatewayConfig(
            backends=[
                BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url),
                BackendConfig(name="t1", backend_type=BackendType.TGI, base_url=mock_tgi.base_url),
            ],
            routes=[ModelRoute(model_name="test-model", primary_backend="t1")],
            health_check_interval_s=0.5,
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            for _ in range(10):
                health = router.get_health()
                if not health["v1"].healthy and health["t1"].healthy:
                    break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Health check did not update within timeout")
        finally:
            await router.stop()

    async def test_list_models_aggregates_all_backends(self, mock_vllm, mock_tgi, mock_ollama):
        config = GatewayConfig(
            backends=[
                BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url),
                BackendConfig(name="t1", backend_type=BackendType.TGI, base_url=mock_tgi.base_url),
                BackendConfig(name="o1", backend_type=BackendType.OLLAMA, base_url=mock_ollama.base_url),
            ],
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            models = await router.list_models()
            assert "llama-3-8b" in models
            assert "llama-3-70b" in models
            assert "llama3" in models
            assert "mistral-7b" in models
        finally:
            await router.stop()

    async def test_route_no_healthy_backends_raises(self, mock_vllm):
        mock_vllm.set_fail_health(True)
        config = GatewayConfig(
            backends=[BackendConfig(name="v1", backend_type=BackendType.VLLM, base_url=mock_vllm.base_url)],
            health_check_interval_s=0.5,
        )
        router = GatewayRouter(config)
        await router.start()
        try:
            for _ in range(10):
                health = router.get_health()
                if not health["v1"].healthy:
                    break
                await asyncio.sleep(0.2)
            with pytest.raises(RuntimeError, match="No healthy backends available"):
                await router.route_chat_completion({"model": "test-model"})
        finally:
            await router.stop()
