"""Tests for the model-as-a-service gateway module."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from distllm.gateway.models import (
    BackendConfig,
    BackendType,
    GatewayConfig,
    ModelRoute,
    BackendHealth,
)
from distllm.gateway.backend import (
    create_backend,
    NativeBackend,
    VLLMBackend,
    TGIBackend,
    OllamaBackend,
)
from distllm.gateway.router import GatewayRouter
from distllm.gateway.fallback import FallbackManager


class TestGatewayModels:
    def test_backend_config_defaults(self):
        bc = BackendConfig(name="test")
        assert bc.backend_type == BackendType.NATIVE
        assert bc.timeout_s == 120.0
        assert bc.weight == 100
        assert bc.tags == {}

    def test_backend_config_model_dump(self):
        bc = BackendConfig(name="t1", backend_type=BackendType.VLLM, base_url="http://vllm:8000")
        d = bc.model_dump()
        assert d["name"] == "t1"
        assert d["backend_type"] == "vllm"
        assert d["base_url"] == "http://vllm:8000"

    def test_gateway_config_defaults(self):
        gc = GatewayConfig()
        assert gc.enabled is True
        assert gc.backends == []
        assert gc.routes == []
        assert gc.default_fallback == []

    def test_model_route(self):
        mr = ModelRoute(model_name="llama", primary_backend="b1", fallback_chain=["b2", "b3"])
        assert mr.model_name == "llama"
        assert mr.fallback_chain == ["b2", "b3"]

    def test_backend_health_defaults(self):
        bh = BackendHealth(backend_name="b1")
        assert bh.healthy is True
        assert bh.active_requests == 0


class TestBackendFactory:
    def test_create_native(self):
        bc = BackendConfig(name="n1", backend_type=BackendType.NATIVE)
        b = create_backend(bc)
        assert isinstance(b, NativeBackend)
        assert b.name == "n1"

    def test_create_vllm(self):
        bc = BackendConfig(name="v1", backend_type=BackendType.VLLM)
        b = create_backend(bc)
        assert isinstance(b, VLLMBackend)

    def test_create_tgi(self):
        bc = BackendConfig(name="t1", backend_type=BackendType.TGI)
        b = create_backend(bc)
        assert isinstance(b, TGIBackend)

    def test_create_ollama(self):
        bc = BackendConfig(name="o1", backend_type=BackendType.OLLAMA)
        b = create_backend(bc)
        assert isinstance(b, OllamaBackend)

    def test_unknown_backend(self):
        bc = BackendConfig(name="x1", backend_type="unknown")  # type: ignore
        with pytest.raises(ValueError):
            create_backend(bc)

    def test_ollama_chat_converts_format(self):
        bc = BackendConfig(name="o1", backend_type=BackendType.OLLAMA, base_url="http://ollama:11434")
        backend = OllamaBackend(bc)
        assert backend.name == "o1"
        assert backend.base_url == "http://ollama:11434"


@pytest.mark.asyncio
class TestNativeBackend:
    async def test_list_models_empty_on_error(self):
        bc = BackendConfig(name="n1")
        backend = NativeBackend(bc)
        models = await backend.list_models()
        assert models == []

    async def test_health_returns_error_on_connection_refused(self):
        bc = BackendConfig(name="n1", base_url="http://127.0.0.1:1")
        backend = NativeBackend(bc)
        healthy, lat, err = await backend.health()
        assert healthy is False
        assert len(err) > 0


@pytest.mark.asyncio
class TestGatewayRouter:
    async def test_start_stop(self):
        config = GatewayConfig(
            backends=[BackendConfig(name="b1")],
            routes=[ModelRoute(model_name="m1", primary_backend="b1")],
        )
        router = GatewayRouter(config)
        await router.start()
        await router.stop()

    async def test_no_healthy_backends_raises(self):
        config = GatewayConfig()
        router = GatewayRouter(config)
        await router.start()
        with pytest.raises(RuntimeError, match="No healthy backends available"):
            await router.route_chat_completion({"model": "test"})
        await router.stop()

    async def test_route_uses_primary_backend(self):
        config = GatewayConfig(
            backends=[BackendConfig(name="b1")],
            routes=[ModelRoute(model_name="m1", primary_backend="b1")],
        )
        router = GatewayRouter(config)
        await router.start()
        router._backends["b1"].chat_completion = AsyncMock(return_value={"id": "ok"})
        result = await router.route_chat_completion({"model": "m1"})
        assert result["id"] == "ok"
        await router.stop()

    async def test_get_health(self):
        config = GatewayConfig(backends=[BackendConfig(name="b1")])
        router = GatewayRouter(config)
        await router.start()
        health = router.get_health()
        assert "b1" in health
        await router.stop()

    async def test_list_models_empty(self):
        config = GatewayConfig()
        router = GatewayRouter(config)
        await router.start()
        models = await router.list_models()
        assert models == []
        await router.stop()

    async def test_stream_fallback(self):
        config = GatewayConfig(
            backends=[BackendConfig(name="b1"), BackendConfig(name="b2")],
            routes=[ModelRoute(model_name="m1", primary_backend="b1", fallback_chain=["b2"])],
        )
        router = GatewayRouter(config)
        await router.start()
        router._backends["b1"]._healthy = False

        async def fake_stream(body, headers):
            yield "chunk"

        router._backends["b2"].chat_completion_stream = fake_stream
        router._backends["b2"]._healthy = True
        chunks = []
        async for c in router.route_chat_completion_stream({"model": "m1"}):
            chunks.append(c)
        assert len(chunks) > 0
        await router.stop()

    async def test_route_fallback_on_failure(self):
        config = GatewayConfig(
            backends=[BackendConfig(name="b1"), BackendConfig(name="b2")],
            routes=[ModelRoute(model_name="m1", primary_backend="b1", fallback_chain=["b2"])],
        )
        router = GatewayRouter(config)
        await router.start()
        router._backends["b1"].chat_completion = AsyncMock(side_effect=RuntimeError("fail"))
        router._backends["b1"]._healthy = True
        router._backends["b2"].chat_completion = AsyncMock(return_value={"id": "fallback"})
        router._backends["b2"]._healthy = True
        result = await router.route_chat_completion({"model": "m1"})
        assert result["id"] == "fallback"
        await router.stop()

    async def test_all_fallback_fail_raises(self):
        config = GatewayConfig(
            backends=[BackendConfig(name="b1"), BackendConfig(name="b2")],
            routes=[ModelRoute(model_name="m1", primary_backend="b1", fallback_chain=["b2"])],
        )
        router = GatewayRouter(config)
        await router.start()
        router._backends["b1"].chat_completion = AsyncMock(side_effect=RuntimeError("fail"))
        router._backends["b1"]._healthy = True
        router._backends["b2"].chat_completion = AsyncMock(side_effect=RuntimeError("fail2"))
        router._backends["b2"]._healthy = True
        with pytest.raises(RuntimeError, match="No backend succeeded"):
            await router.route_chat_completion({"model": "m1"})
        await router.stop()


class TestFallbackManager:
    def test_record_and_count(self):
        fm = FallbackManager()
        assert fm.get_failure_count("b1") == 0
        fm.record_failure("b1")
        assert fm.get_failure_count("b1") == 1

    def test_circuit_breaker_opens_at_5(self):
        fm = FallbackManager()
        for _ in range(5):
            fm.record_failure("b1")
        assert fm.is_circuit_open("b1") is True

    def test_success_closes_circuit(self):
        fm = FallbackManager()
        for _ in range(5):
            fm.record_failure("b1")
        assert fm.is_circuit_open("b1") is True
        fm.record_success("b1")
        assert fm.is_circuit_open("b1") is False

    def test_sorted_backends(self):
        fm = FallbackManager()
        fm.record_failure("b1")
        fm.record_failure("b1")
        fm.record_failure("b2")
        sorted_list = fm.get_sorted_backends(["b1", "b2", "b3"])
        assert sorted_list[0] == "b3"
        assert sorted_list[1] == "b2"
        assert sorted_list[2] == "b1"

    def test_stats(self):
        fm = FallbackManager()
        fm.record_failure("b1")
        stats = fm.stats()
        assert stats["b1"]["count"] == 1

    def test_reset_all(self):
        fm = FallbackManager()
        fm.record_failure("b1")
        fm.reset_all()
        assert fm.get_failure_count("b1") == 0


def aiter(iterable):
    """Convert iterable to async generator."""
    async def gen():
        for item in iterable:
            yield item
    return gen()
