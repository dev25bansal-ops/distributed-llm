"""AuthMiddleware, RequestIDMiddleware, ObservabilityMiddleware & TimeoutMiddleware tests."""

import asyncio
import os
import re
import time
import secrets
import unittest
import uuid as uuid_mod
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.server import app, state
from distllm.api.middleware import _rate_limiter, RequestIDMiddleware


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    _rate_limiter._attempts.clear()


@pytest.fixture
def patch_coordinator():
    """Set a mock coordinator on server state and restore on teardown."""
    from unittest.mock import MagicMock
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._shutting_down = False
    coord.tokenizer = MagicMock()
    coord.tokenizer.eos_token_id = 0
    coord.list_models.return_value = ["test-model"]

    original = state.coordinator
    state.coordinator = coord
    yield coord
    state.coordinator = original


# ---------------------------------------------------------------------------
# Fixtures: clean env state per scenario
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_auth(mock_coordinator):
    """Auth enabled with an explicitly set API_KEY."""
    key = secrets.token_hex(32)
    os.environ.pop("DISABLE_AUTH", None)
    os.environ.pop("DISTLLM_DEV_MODE", None)
    os.environ["API_KEY"] = key
    os.environ["API_KEY_WAS_SET"] = "1"
    client = TestClient(app)
    client._test_api_key = key
    yield client
    del os.environ["API_KEY"]
    os.environ.pop("API_KEY_WAS_SET", None)


@pytest.fixture
def client_dev_bypass(mock_coordinator):
    """Dev bypass: DISABLE_AUTH=1 + DISTLLM_DEV_MODE=1 + auto-generated key."""
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ["DISABLE_AUTH"] = "1"
    os.environ["DISTLLM_DEV_MODE"] = "1"
    client = TestClient(app)
    yield client
    os.environ.pop("DISABLE_AUTH", None)
    os.environ.pop("DISTLLM_DEV_MODE", None)
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)


@pytest.fixture
def client_dev_bypass_with_explicit_key(mock_coordinator):
    """Dev bypass disabled when API_KEY was explicitly set."""
    key = secrets.token_hex(32)
    os.environ["API_KEY"] = key
    os.environ["API_KEY_WAS_SET"] = "1"
    os.environ["DISABLE_AUTH"] = "1"
    os.environ["DISTLLM_DEV_MODE"] = "1"
    client = TestClient(app)
    client._test_api_key = key
    yield client
    del os.environ["API_KEY"]
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ.pop("DISABLE_AUTH", None)
    os.environ.pop("DISTLLM_DEV_MODE", None)


# ===================================================================
# Valid API key
# ===================================================================


class TestValidApiKey:
    def test_valid_key_returns_200(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {client_with_auth._test_api_key}"},
        )
        assert resp.status_code == 200

    def test_valid_key_response_contains_model_list(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {client_with_auth._test_api_key}"},
        )
        body = resp.json()
        assert body["object"] == "list"
        assert "data" in body


# ===================================================================
# Invalid / missing API key
# ===================================================================


class TestInvalidApiKey:
    def test_wrong_bearer_token_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_wrong_token_error_type(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        body = resp.json()
        assert body["error"]["type"] == "auth_error"
        assert "Unauthorized" in body["error"]["message"]

    def test_missing_authorization_header_returns_401(self, client_with_auth):
        resp = client_with_auth.get("/v1/models")
        assert resp.status_code == 401

    def test_missing_auth_header_error_type(self, client_with_auth):
        resp = client_with_auth.get("/v1/models")
        body = resp.json()
        assert body["error"]["type"] == "auth_error"

    def test_bearer_without_token_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_no_bearer_prefix_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": client_with_auth._test_api_key},
        )
        assert resp.status_code == 401

    def test_wrong_scheme_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": f"Basic {client_with_auth._test_api_key}"},
        )
        assert resp.status_code == 401


# ===================================================================
# Brute-force rate limiting (30 failed attempts / 60s per IP)
# ===================================================================


class TestBruteForceRateLimiting:
    def test_rate_limited_after_30_failures(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer bad-key"},
        )
        assert resp.status_code == 429

    def test_rate_limit_error_type(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer bad-key"},
        )
        body = resp.json()
        assert body["error"]["type"] == "auth_rate_limit"

    def test_rate_limit_has_retry_after(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer bad-key"},
        )
        body = resp.json()
        assert "retry_after" in body["error"]
        assert body["error"]["retry_after"] > 0

    def test_rate_limit_resets_after_window(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        _rate_limiter._attempts.clear()
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {client_with_auth._test_api_key}"},
        )
        assert resp.status_code == 200

    def test_valid_requests_not_rate_limited(self, client_with_auth):
        for _ in range(35):
            resp = client_with_auth.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {client_with_auth._test_api_key}"},
            )
            assert resp.status_code == 200

    def test_rate_limiter_prunes_old_entries(self):
        from distllm.api.middleware import _RateLimiter
        rl = _RateLimiter(max_attempts=3, window_seconds=1)
        rl.record_attempt("1.2.3.4")
        rl.record_attempt("1.2.3.4")
        rl.record_attempt("1.2.3.4")
        assert rl.is_rate_limited("1.2.3.4") is True
        time.sleep(1.1)
        assert rl.is_rate_limited("1.2.3.4") is False

    def test_different_ips_independent(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        resp = client_with_auth.get(
            "/v1/models",
            headers={
                "Authorization": f"Bearer {client_with_auth._test_api_key}",
                "X-Forwarded-For": "10.0.0.99",
            },
        )
        assert resp.status_code == 200


# ===================================================================
# Dev bypass: DISABLE_AUTH=1 + DISTLLM_DEV_MODE=1
# ===================================================================


class TestDevBypass:
    def test_dev_bypass_allows_request(self, client_dev_bypass):
        resp = client_dev_bypass.get("/v1/models")
        assert resp.status_code == 200

    def test_dev_bypass_no_auth_header_needed(self, client_dev_bypass):
        resp = client_dev_bypass.get("/v1/models", headers={})
        assert resp.status_code == 200

    def test_dev_bypass_only_with_both_flags(self, mock_coordinator):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ.pop("DISTLLM_DEV_MODE", None)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        os.environ.pop("DISABLE_AUTH", None)

    def test_dev_bypass_disabled_with_only_dev_mode(self, mock_coordinator):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ.pop("DISABLE_AUTH", None)
        os.environ["DISTLLM_DEV_MODE"] = "1"
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_dev_bypass_disabled_with_explicit_api_key(self, client_dev_bypass_with_explicit_key):
        resp = client_dev_bypass_with_explicit_key.get("/v1/models")
        assert resp.status_code == 401

    def test_dev_bypass_explicit_key_accepts_valid_token(self, client_dev_bypass_with_explicit_key):
        resp = client_dev_bypass_with_explicit_key.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {client_dev_bypass_with_explicit_key._test_api_key}"},
        )
        assert resp.status_code == 200

    def test_dev_bypass_logs_warning_only_once(self, mock_coordinator, monkeypatch):
        import logging
        logs = []
        monkeypatch.setattr("distllm.api.middleware.logger.warning", lambda msg: logs.append(msg))
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        from distllm.api.middleware import AuthMiddleware
        client = TestClient(app)
        for _ in range(3):
            client.get("/v1/models")
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)
        assert len(logs) == 1


# ===================================================================
# Error response format
# ===================================================================


class TestErrorResponseFormat:
    def test_error_has_openai_compat_structure(self, client_with_auth):
        resp = client_with_auth.get("/v1/models")
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert "type" in body["error"]
        assert "code" in body["error"]

    def test_error_code_is_string_status(self, client_with_auth):
        resp = client_with_auth.get("/v1/models")
        body = resp.json()
        assert body["error"]["code"] == "401"

    def test_429_error_includes_retry_after(self, client_with_auth):
        for _ in range(30):
            client_with_auth.get(
                "/v1/models",
                headers={"Authorization": "Bearer bad-key"},
            )
        resp = client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer bad-key"},
        )
        body = resp.json()
        assert body["error"]["code"] == "429"
        assert "retry_after" in body["error"]

    def test_request_id_in_error_when_available(self, mock_coordinator):
        key = secrets.token_hex(32)
        os.environ["API_KEY"] = key
        os.environ["API_KEY_WAS_SET"] = "1"
        client = TestClient(app)
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        del os.environ["API_KEY"]
        os.environ.pop("API_KEY_WAS_SET", None)
        body = resp.json()
        assert "request_id" in body
        assert len(body["request_id"]) > 0


# ===================================================================
# Dev bypass — each single flag alone still requires auth
# ===================================================================


class TestDevBypassSingleFlag:
    """Only one of DISABLE_AUTH / DISTLLM_DEV_MODE set → auth still enforced."""

    def test_disable_auth_alone_requires_auth(self, mock_coordinator):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ.pop("DISTLLM_DEV_MODE", None)
        client = TestClient(app)
        resp = client.get("/v1/models")
        os.environ.pop("DISABLE_AUTH", None)
        assert resp.status_code == 401

    def test_dev_mode_alone_requires_auth(self, mock_coordinator):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ.pop("DISABLE_AUTH", None)
        os.environ["DISTLLM_DEV_MODE"] = "1"
        client = TestClient(app)
        resp = client.get("/v1/models")
        os.environ.pop("DISTLLM_DEV_MODE", None)
        assert resp.status_code == 401

    def test_neither_flag_requires_auth(self, mock_coordinator):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401


# ===================================================================
# RequestIDMiddleware
# ===================================================================


@pytest.fixture
def request_id_test_app():
    """Minimal app with only RequestIDMiddleware + echo route for state inspection."""
    test_app = FastAPI()

    @test_app.get("/echo-state")
    async def echo_state(request: Request):
        return {
            "request_id": getattr(request.state, "request_id", None),
            "request_timeout": getattr(request.state, "request_timeout", None),
            "request_priority": getattr(request.state, "request_priority", None),
        }

    test_app.add_middleware(RequestIDMiddleware)
    return test_app


class TestRequestIDUUID:
    """UUID generation when no X-Request-ID is sent."""

    def test_response_has_x_request_id(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        assert "X-Request-ID" in resp.headers

    def test_x_request_id_is_uuid(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        val = resp.headers["X-Request-ID"]
        # uuid hex with dashes: 8-4-4-4-12
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            val,
            re.I,
        )

    def test_request_id_is_valid_uuid4(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        val = resp.headers["X-Request-ID"]
        parsed = uuid_mod.UUID(val)
        assert parsed.version == 4

    def test_request_state_matches_response_header(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        body = resp.json()
        assert body["request_id"] == resp.headers["X-Request-ID"]


class TestRequestIDCustom:
    """Incoming X-Request-ID is propagated."""

    CUSTOM_ID = "my-custom-request-001"

    def test_custom_id_appears_in_response(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-ID": self.CUSTOM_ID})
        assert resp.headers["X-Request-ID"] == self.CUSTOM_ID

    def test_custom_id_in_request_state(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-ID": self.CUSTOM_ID})
        body = resp.json()
        assert body["request_id"] == self.CUSTOM_ID

    def test_multiple_custom_ids_unique(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        id_a = "req-a"
        id_b = "req-b"
        resp_a = client.get("/echo-state", headers={"X-Request-ID": id_a})
        resp_b = client.get("/echo-state", headers={"X-Request-ID": id_b})
        assert resp_a.headers["X-Request-ID"] == id_a
        assert resp_b.headers["X-Request-ID"] == id_b


class TestRequestTimeout:
    """X-Request-Timeout header parsing."""

    def test_no_header_returns_none(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        assert resp.json()["request_timeout"] is None

    def test_valid_float_timeout(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": "30.5"})
        assert resp.json()["request_timeout"] == 30.5

    def test_valid_int_timeout(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": "60"})
        assert resp.json()["request_timeout"] == 60.0

    def test_zero_timeout_returns_none(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": "0"})
        assert resp.json()["request_timeout"] is None

    def test_negative_timeout_returns_none(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": "-5"})
        assert resp.json()["request_timeout"] is None

    def test_nonnumeric_timeout_returns_none(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": "abc"})
        assert resp.json()["request_timeout"] is None

    def test_empty_timeout_returns_none(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Request-Timeout": ""})
        assert resp.json()["request_timeout"] is None


class TestRequestPriority:
    """X-Priority header parsing."""

    def test_no_header_defaults_mid(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state")
        assert resp.json()["request_priority"] == 2

    def test_critical_priority(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "critical"})
        assert resp.json()["request_priority"] == 0

    def test_high_priority(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "high"})
        assert resp.json()["request_priority"] == 1

    def test_normal_priority(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "normal"})
        assert resp.json()["request_priority"] == 2

    def test_low_priority(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "low"})
        assert resp.json()["request_priority"] == 3

    def test_case_insensitive(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "HIGH"})
        assert resp.json()["request_priority"] == 1

    def test_unknown_priority_defaults_mid(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": "urgent"})
        assert resp.json()["request_priority"] == 2

    def test_empty_priority_defaults_mid(self, request_id_test_app):
        client = TestClient(request_id_test_app)
        resp = client.get("/echo-state", headers={"X-Priority": ""})
        assert resp.json()["request_priority"] == 2


# ===================================================================
# ObservabilityMiddleware
# ===================================================================


class _InMemorySpanExporter(SpanExporter):
    """Stores exported spans in memory for test assertions."""

    def __init__(self):
        self.spans = []

    def export(self, spans, timeout_millis=30000):
        self.spans.extend(spans)
        return None

    def shutdown(self):
        self.spans.clear()

    def clear(self):
        self.spans.clear()


@pytest.fixture
def tracer_and_exporter():
    """Returns (tracer, in_memory_exporter) for a fresh TracerProvider per test."""
    exporter = _InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.fixture
def mock_metrics_exporter():
    return MagicMock()


@pytest.fixture
def obs_app(mock_metrics_exporter, tracer_and_exporter):
    """Minimal app with ObservabilityMiddleware + success/fail routes."""
    from distllm.api.observability_middleware import ObservabilityMiddleware

    test_app = FastAPI()

    @test_app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @test_app.get("/error")
    async def error():
        raise ValueError("test error")

    tracer, _exporter = tracer_and_exporter

    test_app.add_middleware(
        ObservabilityMiddleware,
        metrics_exporter=mock_metrics_exporter,
        cost_tracker=None,
        anomaly_detector=None,
        tracer=tracer,
    )
    test_app.add_middleware(RequestIDMiddleware)
    return test_app


@pytest.fixture
def obs_app_with_cost(mock_metrics_exporter, tracer_and_exporter):
    """App with cost tracker and route that sets _request_cost."""
    from distllm.api.observability_middleware import ObservabilityMiddleware

    test_app = FastAPI()

    @test_app.get("/predict")
    async def predict(request: Request):
        request.state._request_cost = {"cost": 0.05, "gpu_hours": 0.001}
        return {"cost": 0.05}

    @test_app.get("/predict-no-cost")
    async def predict_no_cost(request: Request):
        return {"cost": None}

    tracer, _exporter = tracer_and_exporter

    test_app.add_middleware(
        ObservabilityMiddleware,
        metrics_exporter=mock_metrics_exporter,
        cost_tracker=MagicMock(),
        anomaly_detector=None,
        tracer=tracer,
    )
    test_app.add_middleware(RequestIDMiddleware)
    return test_app


@pytest.fixture
def obs_app_with_anomaly(tracer_and_exporter):
    """App with mock anomaly detector."""
    from distllm.api.observability_middleware import ObservabilityMiddleware

    test_app = FastAPI()

    @test_app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @test_app.get("/error")
    async def error():
        raise ValueError("boom")

    tracer, _exporter = tracer_and_exporter
    anomaly_detector = MagicMock()

    test_app.add_middleware(
        ObservabilityMiddleware,
        metrics_exporter=None,
        cost_tracker=None,
        anomaly_detector=anomaly_detector,
        tracer=tracer,
    )
    test_app.add_middleware(RequestIDMiddleware)
    return test_app, anomaly_detector


@pytest.fixture
def rate_limit_app():
    """Minimal app with RateLimitMiddleware using very low RPM."""
    from distllm.api.rate_limiter import RateLimiter
    from distllm.api.rate_limit_middleware import RateLimitMiddleware

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route():
        return {"ok": True}

    @test_app.get("/health")
    async def health():
        return {"status": "healthy"}

    @test_app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @test_app.get("/live")
    async def live():
        return {"status": "alive"}

    @test_app.get("/metrics")
    async def metrics():
        return {"status": "metrics"}

    @test_app.get("/docs")
    async def docs():
        return {"status": "docs"}

    @test_app.get("/openapi.json")
    async def openapi():
        return {"status": "openapi"}

    @test_app.get("/redoc")
    async def redoc():
        return {"status": "redoc"}

    limiter = RateLimiter(default_rpm=3.0, burst_multiplier=1.0)
    test_app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)
    test_app.add_middleware(RequestIDMiddleware)
    return test_app, limiter


class TestObservabilitySpanCreation:
    """OTel span created per request with correct attributes."""

    def test_span_created_for_success(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        assert len(exporter.spans) >= 1

    def test_span_has_http_attributes(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        span = exporter.spans[0]
        attrs = span.attributes
        assert attrs["http.method"] == "GET"
        assert attrs["http.target"] == "/ok"
        assert attrs["http.status_code"] == 200

    def test_span_has_status_code_on_success(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        assert exporter.spans[0].attributes["http.status_code"] == 200

    def test_span_has_duration(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        assert exporter.spans[0].attributes["http.duration_s"] > 0

    def test_span_has_request_id_attribute(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok", headers={"X-Request-ID": "span-request-abc"})
        _tracer, exporter = tracer_and_exporter
        span = exporter.spans[0]
        assert span.attributes["http.request_id"] == "span-request-abc"

    def test_span_name_contains_method_and_path(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        assert exporter.spans[0].name == "HTTP GET /ok"


class TestObservabilityREDMetrics:
    """Rate, Errors, Duration metrics recorded per endpoint."""

    def test_requests_total_incremented(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        mock_metrics_exporter.requests_total.labels.assert_called_once_with(
            method="GET", status="success", model="distributed-llm", tenant="default",
        )

    def test_request_latency_observed(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        mock_metrics_exporter.request_latency.labels.assert_called_once_with(
            method="GET", model="distributed-llm", tenant="default",
        )

    def test_request_duration_observed(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        mock_metrics_exporter.request_duration_seconds.labels.assert_called_once_with(
            method="GET", model="distributed-llm", tenant="default",
        )


class TestObservabilityErrorTracking:
    """Exception → error metric incremented."""

    def test_errors_total_on_exception(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        try:
            client.get("/error")
        except ValueError:
            pass
        mock_metrics_exporter.errors_total.labels.assert_called_once_with(
            type="http_500", model="distributed-llm", tenant="default",
        )

    def test_requests_total_marks_error_status(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        try:
            client.get("/error")
        except ValueError:
            pass
        mock_metrics_exporter.requests_total.labels.assert_called_once_with(
            method="GET", status="error", model="distributed-llm", tenant="default",
        )

    def test_span_status_set_to_error(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        try:
            client.get("/error")
        except ValueError:
            pass
        _tracer, exporter = tracer_and_exporter
        span = exporter.spans[0]
        assert span.status.status_code == trace.StatusCode.ERROR

    def test_span_records_exception(self, obs_app, tracer_and_exporter):
        client = TestClient(obs_app)
        try:
            client.get("/error")
        except ValueError:
            pass
        _tracer, exporter = tracer_and_exporter
        span = exporter.spans[0]
        events = [e for e in span.events if e.name == "exception"]
        assert len(events) >= 1


# ===================================================================
# Cost tracking
# ===================================================================


class TestObservabilityCostTracking:
    def test_cost_recorded_with_labels(self, obs_app_with_cost, mock_metrics_exporter):
        client = TestClient(obs_app_with_cost)
        client.get("/predict")
        mock_metrics_exporter.request_cost_total.labels.assert_called_once_with(
            model="distributed-llm", tenant="default",
        )

    def test_cost_value_incremented(self, obs_app_with_cost, mock_metrics_exporter):
        client = TestClient(obs_app_with_cost)
        client.get("/predict")
        label_call = mock_metrics_exporter.request_cost_total.labels.return_value
        label_call.inc.assert_called_once_with(0.05)

    def test_gpu_hours_recorded(self, obs_app_with_cost, mock_metrics_exporter):
        client = TestClient(obs_app_with_cost)
        client.get("/predict")
        mock_metrics_exporter.request_gpu_hours.labels.assert_called_once_with(
            model="distributed-llm", tenant="default",
        )

    def test_no_cost_when_state_not_set(self, obs_app_with_cost, mock_metrics_exporter):
        client = TestClient(obs_app_with_cost)
        client.get("/predict-no-cost")
        mock_metrics_exporter.request_cost_total.labels.assert_not_called()
        mock_metrics_exporter.request_gpu_hours.labels.assert_not_called()

    def test_no_cost_when_tracker_is_none(self, obs_app, mock_metrics_exporter):
        client = TestClient(obs_app)
        client.get("/ok")
        mock_metrics_exporter.request_cost_total.labels.assert_not_called()


# ===================================================================
# Anomaly detection
# ===================================================================


class TestObservabilityAnomalyDetection:
    def test_duration_recorded_on_success(self, obs_app_with_anomaly):
        app, detector = obs_app_with_anomaly
        client = TestClient(app)
        client.get("/ok")
        calls = [c for c in detector.record.call_args_list if c[0][0] == "http_request_duration"]
        assert len(calls) >= 1
        _, duration = calls[0][0]
        assert duration > 0

    def test_error_rate_recorded_on_exception(self, obs_app_with_anomaly):
        app, detector = obs_app_with_anomaly
        client = TestClient(app)
        try:
            client.get("/error")
        except ValueError:
            pass
        detector.record.assert_any_call("http_error_rate", 1.0)

    def test_http_duration_not_recorded_on_error(self, obs_app_with_anomaly):
        """Exception path only records error_rate, not duration."""
        app, detector = obs_app_with_anomaly
        client = TestClient(app)
        try:
            client.get("/error")
        except ValueError:
            pass
        duration_calls = [c for c in detector.record.call_args_list if c[0][0] == "http_request_duration"]
        assert len(duration_calls) == 0

    def test_span_not_recorded_when_detector_is_none(self, obs_app, tracer_and_exporter):
        """When anomaly_detector is None, no crash on success."""
        client = TestClient(obs_app)
        client.get("/ok")
        _tracer, exporter = tracer_and_exporter
        assert len(exporter.spans) >= 1


# ===================================================================
# RateLimitMiddleware
# ===================================================================


@pytest.fixture
def rate_limit_endpoint_app():
    """App with /restricted (3 RPM) and everything else at 100 RPM."""
    from distllm.api.rate_limiter import RateLimiter
    from distllm.api.rate_limit_middleware import RateLimitMiddleware

    test_app = FastAPI()

    @test_app.get("/restricted")
    async def restricted():
        return {"ok": True}

    @test_app.get("/generous")
    async def generous():
        return {"ok": True}

    limiter = RateLimiter(
        default_rpm=100.0,
        endpoint_limits={"/restricted": 3.0},
        burst_multiplier=1.0,
    )
    test_app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)
    test_app.add_middleware(RequestIDMiddleware)
    return test_app, limiter


@pytest.fixture
def rate_limit_auth_app():
    """App with low RPM and auth multiplier to test authenticated client benefit."""
    from distllm.api.rate_limiter import RateLimiter
    from distllm.api.rate_limit_middleware import RateLimitMiddleware

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route():
        return {"ok": True}

    # RPM=3, burst=3, auth multiplier=2 → auth limit = 6 RPM, burst = 6
    limiter = RateLimiter(
        default_rpm=3.0,
        burst_multiplier=1.0,
        auth_rpm_multiplier=2.0,
    )
    test_app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)
    test_app.add_middleware(RequestIDMiddleware)
    return test_app, limiter


class TestRateLimitBasic:
    def test_exceed_rpm_returns_429(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        # Default RPM=3, burst=3 → 4th request is blocked
        for _ in range(4):
            resp = client.get("/test")
        assert resp.status_code == 429

    def test_rate_limit_error_type(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        body = resp.json()
        assert body["error"]["type"] == "rate_limit_error"

    def test_rate_limit_exceeded_code(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        body = resp.json()
        assert body["error"]["code"] == "429"

    def test_burst_allows_exact_burst(self, rate_limit_app):
        app, limiter = rate_limit_app
        limiter.default_rpm = 3.0
        # burst = 3 * 1.0 = 3 → first 3 succeed
        client = TestClient(app)
        for _ in range(3):
            resp = client.get("/test")
            assert resp.status_code == 200

    def test_rate_limit_headers_present_on_success(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        resp = client.get("/test")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_headers_values(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        resp = client.get("/test")
        assert int(resp.headers["X-RateLimit-Limit"]) > 0
        assert int(resp.headers["X-RateLimit-Remaining"]) >= 0

    def test_excluded_paths_not_rate_limited(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rate_limit_resets_after_refill(self, rate_limit_app):
        app, limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        assert resp.status_code == 429
        limiter.reset_all()
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_different_endpoints_independent(self, rate_limit_app):
        app, limiter = rate_limit_app
        client = TestClient(app)
        @app.get("/other")
        async def other():
            return {"ok": True}
        for _ in range(4):
            client.get("/test")
        resp = client.get("/other")
        assert resp.status_code == 200


class TestRateLimitExcludedPaths:
    """Excluded paths bypass rate limiting entirely."""

    EXCLUDED_PATHS = ["/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json", "/redoc"]

    def test_all_excluded_paths_bypass(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for path in self.EXCLUDED_PATHS:
            for _ in range(10):
                resp = client.get(path)
                assert resp.status_code == 200, f"{path} was rate limited"

    def test_no_rate_limit_headers_on_excluded(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for path in self.EXCLUDED_PATHS:
            resp = client.get(path)
            assert "X-RateLimit-Limit" not in resp.headers, f"{path} has rate limit header"
            assert "X-RateLimit-Remaining" not in resp.headers

    def test_non_excluded_path_still_rate_limited(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        assert resp.status_code == 429


class TestRateLimitPerEndpoint:
    """Different endpoints have independent rate limits."""

    def test_low_limit_endpoint_blocked(self, rate_limit_endpoint_app):
        app, _limiter = rate_limit_endpoint_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/restricted")
        assert resp.status_code == 429

    def test_high_limit_endpoint_unaffected(self, rate_limit_endpoint_app):
        app, _limiter = rate_limit_endpoint_app
        client = TestClient(app)
        for _ in range(4):
            client.get("/restricted")  # exhaust restricted
        resp = client.get("/generous")
        assert resp.status_code == 200

    def test_endpoint_limit_header_reflects_correct_limit(self, rate_limit_endpoint_app):
        app, _limiter = rate_limit_endpoint_app
        client = TestClient(app)
        resp_restricted = client.get("/restricted")
        resp_generous = client.get("/generous")
        assert int(resp_restricted.headers["X-RateLimit-Limit"]) == 3
        assert int(resp_generous.headers["X-RateLimit-Limit"]) == 100


class TestRateLimitAuthMultiplier:
    """Authenticated clients get higher rate limits via auth_rpm_multiplier."""

    def test_unauthenticated_blocked_after_burst(self, rate_limit_auth_app):
        app, _limiter = rate_limit_auth_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        assert resp.status_code == 429

    def test_authenticated_allowed_more_requests(self, rate_limit_auth_app):
        app, _limiter = rate_limit_auth_app
        client = TestClient(app)
        for _ in range(6):
            resp = client.get("/test", headers={"Authorization": "Bearer mykey"})
            if resp.status_code == 429:
                break
        # Auth multiplier 2.0 × RPM 3 × burst 1.0 = 6 allowed
        assert resp.status_code == 200, f"Blocked after {_ + 1} requests"

    def test_authenticated_eventually_blocked(self, rate_limit_auth_app):
        app, _limiter = rate_limit_auth_app
        client = TestClient(app)
        for _ in range(7):
            resp = client.get("/test", headers={"Authorization": "Bearer mykey"})
        assert resp.status_code == 429

    def test_auth_and_unauth_independent_buckets(self, rate_limit_auth_app):
        """Auth and non-auth clients have separate token buckets."""
        app, _limiter = rate_limit_auth_app
        client = TestClient(app)
        # Exhaust unauth bucket
        for _ in range(4):
            client.get("/test")
        # Auth client should still have full bucket
        resp = client.get("/test", headers={"Authorization": "Bearer mykey"})
        assert resp.status_code == 200


class TestRateLimitRetryAfter:
    """429 response includes retry_after field."""

    def test_retry_after_in_body(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        body = resp.json()
        assert "retry_after" in body["error"]
        assert body["error"]["retry_after"] > 0

    def test_retry_after_is_number(self, rate_limit_app):
        app, _limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        body = resp.json()
        assert isinstance(body["error"]["retry_after"], (int, float))

    def test_retry_after_decreases_over_time(self, rate_limit_app):
        app, limiter = rate_limit_app
        client = TestClient(app)
        for _ in range(4):
            resp = client.get("/test")
        body1 = resp.json()
        import time
        time.sleep(0.5)
        limiter.reset_all()
        for _ in range(4):
            resp = client.get("/test")
        body2 = resp.json()
        assert body2["error"]["retry_after"] > 0


class TestRateLimitLRUEviction:
    """Oldest clients are evicted when max_clients is exceeded."""

    def test_oldest_client_evicted_first(self):
        from distllm.api.rate_limiter import RateLimiter
        limiter = RateLimiter(default_rpm=100, max_clients=3, burst_multiplier=1.0)
        # Create 3 clients
        limiter.is_allowed("client-a", "/test")
        limiter.is_allowed("client-b", "/test")
        limiter.is_allowed("client-c", "/test")
        assert limiter.active_clients == 3
        # 4th client causes LRU eviction of oldest ("client-a")
        limiter.is_allowed("client-d", "/test")
        assert limiter.active_clients == 3
        # client-a should be evicted
        assert "client-a" not in limiter._buckets

    def test_recently_used_client_not_evicted(self):
        from distllm.api.rate_limiter import RateLimiter
        limiter = RateLimiter(default_rpm=100, max_clients=3, burst_multiplier=1.0)
        limiter.is_allowed("client-a", "/test")
        limiter.is_allowed("client-b", "/test")
        limiter.is_allowed("client-c", "/test")
        # Re-use client-a to move it to end of LRU order
        limiter.is_allowed("client-a", "/test")
        # 4th client evicts client-b (oldest)
        limiter.is_allowed("client-d", "/test")
        assert "client-b" not in limiter._buckets
        assert "client-a" in limiter._buckets
        assert limiter.active_clients == 3

    def test_lru_eviction_preserves_other_clients(self):
        from distllm.api.rate_limiter import RateLimiter
        limiter = RateLimiter(default_rpm=100, max_clients=3, burst_multiplier=1.0)
        limiter.is_allowed("client-a", "/test")
        limiter.is_allowed("client-b", "/test")
        limiter.is_allowed("client-c", "/test")
        limiter.is_allowed("client-d", "/test")
        assert "client-b" in limiter._buckets
        assert "client-c" in limiter._buckets
        assert "client-d" in limiter._buckets

    def test_evicted_client_gets_fresh_bucket(self):
        from distllm.api.rate_limiter import RateLimiter
        limiter = RateLimiter(default_rpm=3, max_clients=2, burst_multiplier=1.0)
        limiter.is_allowed("client-a", "/test")
        limiter.is_allowed("client-a", "/test")
        limiter.is_allowed("client-a", "/test")
        assert not limiter.is_allowed("client-a", "/test")  # exhausted
        # client-b evicts client-a
        limiter.is_allowed("client-b", "/test")
        limiter.is_allowed("client-c", "/test")
        # client-a was evicted, re-creating gives a fresh bucket
        assert limiter.is_allowed("client-a", "/test") is True


# ===================================================================
# TimeoutMiddleware
# ===================================================================


@pytest.fixture
def timeout_app():
    """App with TimeoutMiddleware set to 50ms, a slow and a fast route."""
    from distllm.api.server import TimeoutMiddleware

    class _TestTimeoutMiddleware(TimeoutMiddleware):
        DEFAULT_TIMEOUT = 0.05
        ENDPOINT_TIMEOUTS = {}

    test_app = FastAPI()

    @test_app.get("/fast")
    async def fast():
        return {"status": "ok"}

    @test_app.get("/slow")
    async def slow():
        await asyncio.sleep(1.0)
        return {"status": "done"}

    test_app.add_middleware(_TestTimeoutMiddleware)
    test_app.add_middleware(RequestIDMiddleware)
    return test_app


class TestTimeoutMiddlewareDefault:
    """Default timeout applied to requests."""

    def test_fast_request_succeeds(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/fast")
        assert resp.status_code == 200

    def test_slow_request_times_out(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/slow")
        assert resp.status_code == 504

    def test_timeout_error_type(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/slow")
        body = resp.json()
        assert body["error"]["type"] == "timeout_error"

    def test_timeout_error_message_includes_seconds(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/slow")
        body = resp.json()
        assert "timeout limit" in body["error"]["message"]

    def test_timeout_error_code_is_504(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/slow")
        body = resp.json()
        assert body["error"]["code"] == "504"

    def test_fast_request_has_no_timeout_error(self, timeout_app):
        client = TestClient(timeout_app)
        resp = client.get("/fast")
        assert resp.status_code == 200


class TestTimeoutHeaderOverride:
    """X-Request-Timeout header overrides the default timeout."""

    def test_custom_timeout_increases_limit(self, timeout_app):
        client = TestClient(timeout_app)
        # Default 50ms is too short, but header sets 2s → request succeeds
        resp = client.get("/slow", headers={"X-Request-Timeout": "2.0"})
        assert resp.status_code == 200

    def test_custom_timeout_reduces_limit(self, timeout_app):
        client = TestClient(timeout_app)
        # Even faster route fails with 10ms timeout
        resp = client.get("/slow", headers={"X-Request-Timeout": "0.01"})
        assert resp.status_code == 504

    def test_custom_timeout_zero_falls_back_to_default(self, timeout_app):
        client = TestClient(timeout_app)
        # X-Request-Timeout: 0 is parsed as None by RequestIDMiddleware
        # so it falls back to the default 50ms → still times out
        resp = client.get("/slow", headers={"X-Request-Timeout": "0"})
        assert resp.status_code == 504


# ===================================================================
# RequestSizeLimitMiddleware
# ===================================================================


@pytest.fixture
def size_limit_app():
    """App with RequestSizeLimitMiddleware set to 100 bytes for testing."""
    from distllm.api.server import RequestSizeLimitMiddleware

    class _SmallLimitMiddleware(RequestSizeLimitMiddleware):
        MAX_REQUEST_SIZE = 100  # bytes

    test_app = FastAPI()

    @test_app.post("/submit")
    async def submit():
        return {"status": "ok"}

    @test_app.put("/update")
    async def update():
        return {"status": "updated"}

    @test_app.get("/fetch")
    async def fetch():
        return {"status": "fetched"}

    test_app.add_middleware(_SmallLimitMiddleware)
    return test_app


class TestRequestSizeLimit:
    """Request body size limits enforced."""

    def test_post_exceeding_limit_returns_413(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 200)
        assert resp.status_code == 413

    def test_put_exceeding_limit_returns_413(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.put("/update", content=b"x" * 200)
        assert resp.status_code == 413

    def test_post_under_limit_succeeds(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 50)
        assert resp.status_code == 200

    def test_put_under_limit_succeeds(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.put("/update", content=b"x" * 50)
        assert resp.status_code == 200

    def test_get_bypasses_size_check(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.get("/fetch", headers={"Content-Length": "500"})
        assert resp.status_code == 200

    def test_413_error_type(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 200)
        body = resp.json()
        assert body["error"]["type"] == "request_too_large"

    def test_413_error_code(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 200)
        body = resp.json()
        assert body["error"]["code"] == "413"

    def test_413_error_message_includes_mb(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 200)
        body = resp.json()
        assert "MB" in body["error"]["message"]

    def test_patch_exceeding_limit_returns_413(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.patch("/submit", content=b"x" * 200)
        assert resp.status_code == 413

    def test_exact_limit_boundary_succeeds(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 100)
        assert resp.status_code == 200

    def test_one_byte_over_limit_returns_413(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 101)
        assert resp.status_code == 413

    def test_one_byte_under_limit_succeeds(self, size_limit_app):
        client = TestClient(size_limit_app)
        resp = client.post("/submit", content=b"x" * 99)
        assert resp.status_code == 200


# ===================================================================
# BackpressureMiddleware
# ===================================================================


@pytest.fixture
def backpressure_app():
    """App with BackpressureMiddleware. Saves/restores state.coordinator."""
    from distllm.api.server import state, BackpressureMiddleware

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route():
        return {"ok": True}

    @test_app.get("/health")
    async def health():
        return {"status": "healthy"}

    test_app.add_middleware(BackpressureMiddleware)
    original = state.coordinator
    yield test_app
    state.coordinator = original


class TestBackpressureQueueFull:
    """Scheduler overloaded with pending requests."""

    def make_coord(self, pending=1000, shutting_down=False):
        from unittest.mock import MagicMock
        coord = MagicMock()
        coord.scheduler = MagicMock()
        coord.scheduler.stats.return_value = {"pending_requests": pending}
        coord._shutting_down = shutting_down
        return coord

    def test_queue_full_returns_503(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(pending=1000)
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 503

    def test_queue_full_error_type(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(pending=1000)
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.json()["error"]["type"] == "backpressure_error"

    def test_slightly_below_full_still_succeeds(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(pending=999)
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_queue_full_excluded_path_bypasses(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(pending=1000)
        client = TestClient(backpressure_app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_no_coordinator_skips_check(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = None
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_no_scheduler_skips_check(self, backpressure_app):
        from distllm.api.server import state
        from unittest.mock import MagicMock
        coord = MagicMock()
        coord.scheduler = None
        coord._shutting_down = False
        state.coordinator = coord
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_scheduler_stats_malformed_skips(self, backpressure_app):
        from distllm.api.server import state
        from unittest.mock import MagicMock
        coord = MagicMock()
        coord.scheduler = MagicMock()
        coord.scheduler.stats.side_effect = TypeError("bad stats")
        coord._shutting_down = False
        state.coordinator = coord
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200


class TestBackpressureShutdown:
    """Server shutting down → all non-excluded requests rejected."""

    def make_coord(self, shutting_down=True):
        from unittest.mock import MagicMock
        coord = MagicMock()
        coord.scheduler = None
        coord._shutting_down = shutting_down
        return coord

    def test_shutdown_returns_503(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=True)
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 503

    def test_shutdown_error_type(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=True)
        client = TestClient(backpressure_app)
        body = client.get("/test").json()
        assert body["error"]["type"] == "shutdown_error"

    def test_shutdown_error_message(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=True)
        client = TestClient(backpressure_app)
        body = client.get("/test").json()
        assert "shutting down" in body["error"]["message"].lower()

    def test_shutdown_excluded_path_bypasses(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=True)
        client = TestClient(backpressure_app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_not_shutting_down_succeeds(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=False)
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_shutdown_error_code(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = self.make_coord(shutting_down=True)
        client = TestClient(backpressure_app)
        body = client.get("/test").json()
        assert body["error"]["code"] == "503"

    def test_coordinator_none_during_shutdown_graceful(self, backpressure_app):
        from distllm.api.server import state
        state.coordinator = None
        client = TestClient(backpressure_app)
        resp = client.get("/test")
        assert resp.status_code == 200


# ===================================================================
# SecurityHeadersMiddleware
# ===================================================================


@pytest.fixture
def security_headers_app():
    """App with SecurityHeadersMiddleware."""
    from distllm.api.server import SecurityHeadersMiddleware

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route():
        return {"ok": True}

    test_app.add_middleware(SecurityHeadersMiddleware)
    return test_app


class TestSecurityHeaders:
    """Security headers added to all responses."""

    def test_csp_header_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert "Content-Security-Policy" in resp.headers

    def test_csp_header_value(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"

    def test_x_frame_options_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_x_content_type_options_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_xss_protection_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"

    def test_referrer_policy_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_permissions_policy_present(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert resp.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


class TestSecurityHSTS:
    """HSTS only added when DISTLLM_TLS_ENABLED=true."""

    def test_hsts_not_present_by_default(self, security_headers_app):
        client = TestClient(security_headers_app)
        resp = client.get("/test")
        assert "Strict-Transport-Security" not in resp.headers

    def test_hsts_present_when_tls_enabled(self, security_headers_app):
        os.environ["DISTLLM_TLS_ENABLED"] = "true"
        try:
            client = TestClient(security_headers_app)
            resp = client.get("/test")
            assert "Strict-Transport-Security" in resp.headers
            assert resp.headers["Strict-Transport-Security"].startswith("max-age=")
            assert "includeSubDomains" in resp.headers["Strict-Transport-Security"]
        finally:
            os.environ.pop("DISTLLM_TLS_ENABLED", None)

    def test_hsts_not_present_when_tls_disabled(self, security_headers_app):
        os.environ["DISTLLM_TLS_ENABLED"] = "false"
        try:
            client = TestClient(security_headers_app)
            resp = client.get("/test")
            assert "Strict-Transport-Security" not in resp.headers
        finally:
            os.environ.pop("DISTLLM_TLS_ENABLED", None)
