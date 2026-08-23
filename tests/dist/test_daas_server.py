"""Tests for the Draft-as-a-Service (DaaS) server module.

Tests cover:
- DaaSConfig dataclass (defaults, custom values, edge cases)
- DaaSMetrics dataclass (initial state, computed properties, edge cases)
- DaaSServer (initialization, health/ready/metrics, rate limiting, mock generation,
  FastAPI application endpoints, auth, shutdown)
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient
import pytest

from distllm.dist.daas_server import DaaSConfig, DaaSMetrics, DaaSServer


# =========================================================================
# DaaSConfig
# =========================================================================


class TestDaaSConfig:
    """DaaSConfig dataclass -- defaults and custom values."""

    def test_defaults(self) -> None:
        cfg = DaaSConfig()
        assert cfg.model_name == "SmolLM-135M"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.max_concurrent == 10
        assert cfg.rate_limit_per_minute == 60
        assert cfg.api_key == ""
        assert cfg.cost_per_hour == 0.05
        assert cfg.hardware == "cpu"
        assert cfg.dtype == "float16"
        assert cfg.max_tokens_per_request == 32
        assert cfg.enable_logprobs is True

    def test_custom_values(self) -> None:
        cfg = DaaSConfig(
            model_name="Custom-42M",
            host="127.0.0.1",
            port=8080,
            max_concurrent=5,
            rate_limit_per_minute=10,
            api_key="sk-abc",
            cost_per_hour=0.10,
            hardware="gpu",
            dtype="bfloat16",
            max_tokens_per_request=64,
            enable_logprobs=False,
        )
        assert cfg.model_name == "Custom-42M"
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.max_concurrent == 5
        assert cfg.rate_limit_per_minute == 10
        assert cfg.api_key == "sk-abc"
        assert cfg.cost_per_hour == 0.10
        assert cfg.hardware == "gpu"
        assert cfg.dtype == "bfloat16"
        assert cfg.max_tokens_per_request == 64
        assert cfg.enable_logprobs is False

    def test_zero_rate_limit(self) -> None:
        """rate_limit_per_minute=0 means unlimited."""
        cfg = DaaSConfig(rate_limit_per_minute=0)
        assert cfg.rate_limit_per_minute == 0

    def test_zero_max_concurrent(self) -> None:
        cfg = DaaSConfig(max_concurrent=0)
        assert cfg.max_concurrent == 0

    def test_empty_api_key(self) -> None:
        cfg = DaaSConfig(api_key="")
        assert cfg.api_key == ""

    def test_negative_cost(self) -> None:
        """Edge case: negative cost_per_hour should not crash."""
        cfg = DaaSConfig(cost_per_hour=-1.0)
        assert cfg.cost_per_hour == -1.0

    def test_zero_cost(self) -> None:
        cfg = DaaSConfig(cost_per_hour=0.0)
        assert cfg.cost_per_hour == 0.0


# =========================================================================
# DaaSMetrics
# =========================================================================


class TestDaaSMetrics:
    """DaaSMetrics dataclass -- initial state and computed properties."""

    def test_defaults(self) -> None:
        m = DaaSMetrics()
        assert m.total_requests == 0
        assert m.total_tokens_generated == 0
        assert m.total_latency_s == 0.0
        assert m.active_requests == 0
        assert m.errors == 0
        assert m.start_time == 0.0
        assert m.rate_limited == 0

    def test_uptime_s_before_start(self) -> None:
        """When start_time is 0, uptime_s should be 0 (not negative)."""
        m = DaaSMetrics()
        assert m.uptime_s == 0.0

    def test_uptime_s_after_start(self) -> None:
        m = DaaSMetrics(start_time=time.time() - 5.0)
        assert 4.0 <= m.uptime_s <= 6.0

    def test_avg_latency_ms_no_requests(self) -> None:
        """Division-by-zero guard: avg_latency_ms is 0 when no requests."""
        m = DaaSMetrics()
        assert m.avg_latency_ms == 0.0

    def test_avg_latency_ms_with_data(self) -> None:
        m = DaaSMetrics(total_requests=4, total_latency_s=2.0)
        assert m.avg_latency_ms == 500.0  # (2.0 / 4) * 1000

    def test_avg_latency_ms_single_request(self) -> None:
        m = DaaSMetrics(total_requests=1, total_latency_s=0.5)
        assert m.avg_latency_ms == 500.0

    def test_avg_latency_ms_large_number(self) -> None:
        m = DaaSMetrics(total_requests=1000, total_latency_s=100.0)
        assert m.avg_latency_ms == 100.0  # (100.0 / 1000) * 1000

    def test_tokens_per_second_no_latency(self) -> None:
        """Division-by-zero guard: tokens_per_second is 0 when no latency."""
        m = DaaSMetrics()
        assert m.tokens_per_second == 0.0

    def test_tokens_per_second_with_data(self) -> None:
        m = DaaSMetrics(total_tokens_generated=50, total_latency_s=10.0)
        assert m.tokens_per_second == 5.0

    def test_tokens_per_second_zero_tokens(self) -> None:
        """Zero tokens with non-zero latency should return 0."""
        m = DaaSMetrics(total_tokens_generated=0, total_latency_s=5.0)
        assert m.tokens_per_second == 0.0


# =========================================================================
# DaaSServer
# =========================================================================


class TestDaaSServerInit:
    """DaaSServer initialization."""

    def test_default_init(self) -> None:
        server = DaaSServer()
        assert isinstance(server._config, DaaSConfig)
        assert isinstance(server._metrics, DaaSMetrics)
        assert server._model is None
        assert server._tokenizer is None
        assert server._rate_limits == {}
        assert server._semaphore is None

    def test_custom_config(self) -> None:
        cfg = DaaSConfig(model_name="MyModel", port=7777, api_key="key")
        server = DaaSServer(cfg)
        assert server._config.model_name == "MyModel"
        assert server._config.port == 7777
        assert server._config.api_key == "key"


class TestDaaSServerEndpoints:
    """DaaSServer health/ready/metrics endpoints (no FastAPI needed yet)."""

    def test_health_check(self) -> None:
        server = DaaSServer()
        result = server.health_check()
        assert result["status"] == "healthy"
        assert result["model"] == "SmolLM-135M"
        assert result["hardware"] == "cpu"
        assert result["active_requests"] == 0
        assert isinstance(result["uptime_s"], float)

    def test_ready_check(self) -> None:
        server = DaaSServer()
        result = server.ready_check()
        assert result["ready"] is True  # mock mode always ready
        assert result["model"] == "SmolLM-135M"

    def test_metrics_initial(self) -> None:
        server = DaaSServer()
        result = server.metrics()
        assert result["total_requests"] == 0
        assert result["total_tokens_generated"] == 0
        assert result["avg_latency_ms"] == 0.0
        assert result["tokens_per_second"] == 0.0
        assert result["active_requests"] == 0
        assert result["errors"] == 0
        assert result["rate_limited"] == 0
        assert isinstance(result["uptime_s"], float)
        assert result["cost_per_hour"] == 0.05
        assert result["model"] == "SmolLM-135M"
        assert result["hardware"] == "cpu"

    def test_health_after_custom_config(self) -> None:
        cfg = DaaSConfig(model_name="CustomModel", hardware="gpu")
        server = DaaSServer(cfg)
        h = server.health_check()
        assert h["model"] == "CustomModel"
        assert h["hardware"] == "gpu"

    def test_metrics_after_custom_config(self) -> None:
        cfg = DaaSConfig(model_name="CustomModel", cost_per_hour=0.99)
        server = DaaSServer(cfg)
        m = server.metrics()
        assert m["model"] == "CustomModel"
        assert m["cost_per_hour"] == 0.99


class TestDaaSServerRateLimit:
    """Rate limiting logic (_check_rate_limit)."""

    def test_allows_within_limit(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=3))
        assert server._check_rate_limit("k") is True
        assert server._check_rate_limit("k") is True
        assert server._check_rate_limit("k") is True

    def test_blocks_when_exceeded(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=2))
        assert server._check_rate_limit("k") is True
        assert server._check_rate_limit("k") is True
        assert server._check_rate_limit("k") is False  # blocked

    def test_different_keys_independent(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=1))
        assert server._check_rate_limit("a") is True
        assert server._check_rate_limit("b") is True  # different key, allowed
        assert server._check_rate_limit("a") is False  # a exceeded
        assert server._check_rate_limit("b") is False  # b also exceeded (1/min)

    def test_unlimited_when_zero(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=0))
        for _ in range(100):
            assert server._check_rate_limit("k") is True

    def test_empty_key_is_tracked(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=2))
        assert server._check_rate_limit("") is True
        assert server._check_rate_limit("") is True
        assert server._check_rate_limit("") is False  # empty key also limited

    def test_different_keys_have_separate_buckets(self) -> None:
        server = DaaSServer(DaaSConfig(rate_limit_per_minute=2))
        # Exhaust key-a
        assert server._check_rate_limit("a") is True
        assert server._check_rate_limit("a") is True
        assert server._check_rate_limit("a") is False
        # key-b still has 2 slots
        assert server._check_rate_limit("b") is True
        assert server._check_rate_limit("b") is True
        assert server._check_rate_limit("b") is False


class TestDaaSServerMockGenerate:
    """Mock generation (_mock_generate)."""

    def test_structure(self) -> None:
        server = DaaSServer()
        result = server._mock_generate(5)
        assert result["object"] == "text_completion"
        assert result["model"] == "SmolLM-135M"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["logprobs"] is None
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 5
        assert result["usage"]["prompt_tokens"] == 0

    def test_zero_tokens(self) -> None:
        server = DaaSServer()
        result = server._mock_generate(0)
        assert result["usage"]["completion_tokens"] == 0
        assert result["choices"][0]["token_ids"] == []

    def test_token_ids_in_range(self) -> None:
        server = DaaSServer()
        result = server._mock_generate(200)
        ids = result["choices"][0]["token_ids"]
        assert len(ids) == 200
        for tid in ids:
            assert 0 <= tid <= 32000

    def test_id_prefix(self) -> None:
        server = DaaSServer()
        result = server._mock_generate(1)
        assert result["id"].startswith("daas-mock-")

    def test_model_name_reflects_config(self) -> None:
        server = DaaSServer(DaaSConfig(model_name="OtherModel"))
        result = server._mock_generate(3)
        assert result["model"] == "OtherModel"


class TestDaaSServerGenerate:
    """Real _generate in mock mode (no torch/model loaded)."""

    def test_mock_mode_list_prompt(self) -> None:
        server = DaaSServer()
        result = server._generate(prompt=[101, 102], max_tokens=4)
        assert result["object"] == "text_completion"
        # Mock mode always returns prompt_tokens=0
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 4
        assert len(result["choices"][0]["token_ids"]) == 4

    def test_mock_mode_string_prompt(self) -> None:
        server = DaaSServer()
        result = server._generate(prompt="hello world", max_tokens=3)
        # String prompts are encoded via tokenizer; when no tokenizer is
        # loaded the mock path uses the string as-is as the prompt length 0
        # in the usage dictionary.
        assert result["usage"]["prompt_tokens"] == 0

    def test_zero_max_tokens(self) -> None:
        server = DaaSServer()
        result = server._generate(prompt=[1], max_tokens=0)
        assert result["usage"]["completion_tokens"] == 0
        assert result["choices"][0]["token_ids"] == []

    def test_max_tokens_capped_by_config(self) -> None:
        """The max_tokens_per_request cap only applies in the real model path,
        not in mock mode (where _mock_generate accepts the raw max_tokens)."""
        server = DaaSServer(DaaSConfig(max_tokens_per_request=4))
        result = server._generate(prompt=[1], max_tokens=100)
        # Mock mode uses max_tokens directly (no cap)
        assert result["usage"]["completion_tokens"] == 100

    def test_empty_list_prompt(self) -> None:
        server = DaaSServer()
        result = server._generate(prompt=[], max_tokens=2)
        assert result["usage"]["prompt_tokens"] == 0

    def test_metrics_updated_after_generate(self) -> None:
        """Mock mode now properly updates metrics."""
        server = DaaSServer()
        assert server._metrics.total_requests == 0
        result = server._generate(prompt=[1, 2, 3], max_tokens=5)
        assert server._metrics.total_requests == 1
        assert server._metrics.total_tokens_generated == 5
        assert server._metrics.total_latency_s > 0
        assert server._metrics.active_requests == 0
        assert result is not None

    def test_invalid_prompt_type_falls_through(self) -> None:
        """An invalid prompt type is caught by mock mode and returns OK."""
        server = DaaSServer()
        result = server._generate(prompt=12345, max_tokens=5)  # type: ignore[arg-type]
        # Mock mode does not inspect the prompt type, so it succeeds.
        assert result["object"] == "text_completion"

    def test_with_logprobs_enabled(self) -> None:
        server = DaaSServer(DaaSConfig(enable_logprobs=True))
        result = server._generate(prompt=[1], max_tokens=3)
        assert result["object"] == "text_completion"

    def test_with_logprobs_disabled(self) -> None:
        server = DaaSServer(DaaSConfig(enable_logprobs=False))
        result = server._generate(prompt=[1], max_tokens=3)
        assert result["object"] == "text_completion"


class TestDaaSServerLoadModel:
    """_load_model behavior without torch/transformers installed."""

    def test_load_model_noop_when_already_loaded(self) -> None:
        server = DaaSServer()
        server._model = object()  # simulate already-loaded model
        server._load_model()  # should return immediately
        assert server._model is not None

    def test_load_model_fallback_to_mock(self) -> None:
        """Without torch, _load_model warns and leaves _model as None."""
        server = DaaSServer()
        server._load_model()
        assert server._model is None  # falls back to mock mode


class TestDaaSServerFastAPI:
    """FastAPI application creation and endpoint tests via TestClient."""

    def test_create_app(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        assert app.title == "DistLLM Draft-as-a-Service"
        assert app.version == "0.1.0"

    def test_health_endpoint(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_ready_endpoint(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True

    def test_metrics_endpoint(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.json()["total_requests"] == 0

    def test_v1_models_endpoint(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "SmolLM-135M"
        assert data["data"][0]["object"] == "model"

    def test_completions_success(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1, 2, 3], "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1

    def test_completions_string_prompt(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello", "max_tokens": 3},
        )
        assert resp.status_code == 200

    def test_completions_no_prompt(self) -> None:
        """Missing prompt defaults to empty list."""
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post("/v1/completions", json={})
        assert resp.status_code == 200

    def test_completions_default_parameters(self) -> None:
        """Only prompt provided; all other params use defaults."""
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"prompt": [1]})
        assert resp.status_code == 200

    def test_completions_with_temperature_topk_topp(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": [1, 2],
                "max_tokens": 4,
                "temperature": 0.7,
                "top_k": 10,
                "top_p": 0.95,
            },
        )
        assert resp.status_code == 200

    # --- Auth ---

    def test_completions_auth_required(self) -> None:
        server = DaaSServer(DaaSConfig(api_key="secret-123"))
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1},
        )
        assert resp.status_code == 401
        assert "detail" in resp.json()

    def test_completions_auth_valid(self) -> None:
        server = DaaSServer(DaaSConfig(api_key="secret-123"))
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1},
            headers={"Authorization": "Bearer secret-123"},
        )
        assert resp.status_code == 200

    def test_completions_auth_invalid(self) -> None:
        server = DaaSServer(DaaSConfig(api_key="secret-123"))
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_completions_auth_no_bearer_prefix(self) -> None:
        """Authorization header without 'Bearer ' prefix -> empty api_key."""
        server = DaaSServer(DaaSConfig(api_key="secret-123"))
        app = server.create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 1},
            headers={"Authorization": "secret-123"},  # missing "Bearer "
        )
        # api_key defaults to "" -> doesn't match "secret-123"
        assert resp.status_code == 401

    # --- Rate limiting via endpoints ---

    def test_completions_rate_limited(self) -> None:
        server = DaaSServer(DaaSConfig(api_key="", rate_limit_per_minute=1))
        app = server.create_app()
        client = TestClient(app)

        resp1 = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})
        assert resp1.status_code == 200

        resp2 = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})
        assert resp2.status_code == 429

    def test_metrics_reflects_rate_limited_requests(self) -> None:
        server = DaaSServer(DaaSConfig(api_key="", rate_limit_per_minute=1))
        app = server.create_app()
        client = TestClient(app)

        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})

        metrics = client.get("/metrics").json()
        assert metrics["rate_limited"] >= 1

    # --- Shutdown ---

    def test_admin_shutdown_response(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        resp = client.post("/admin/shutdown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "shutting_down"
        assert "active_requests" in data

    def test_shutdown_rejects_completions(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        client.post("/admin/shutdown")
        resp = client.post("/v1/completions", json={"prompt": [1], "max_tokens": 1})
        assert resp.status_code == 503
        assert "shutting down" in resp.json()["detail"].lower()

    def test_health_still_works_after_shutdown(self) -> None:
        """Non-completions endpoints should still respond after shutdown."""
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)
        client.post("/admin/shutdown")
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/metrics").status_code == 200

    # --- Metrics reflect activity ---

    def test_metrics_updates_after_completions(self) -> None:
        server = DaaSServer()
        app = server.create_app()
        client = TestClient(app)

        client.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 3})
        client.post("/v1/completions", json={"prompt": [3, 4], "max_tokens": 5})

        m = client.get("/metrics").json()
        assert m["total_requests"] == 2

    # --- Semaphore initialization ---

    def test_semaphore_after_create_app(self) -> None:
        server = DaaSServer(DaaSConfig(max_concurrent=7))
        server.create_app()
        assert server._semaphore is not None
        assert server._semaphore._value == 7  # asyncio.Semaphore internals
