"""Regression: feature-gated 503s must be NEUTRAL to the circuit breaker.

P0 finding: every response >= 500 was recorded as a breaker failure, so sweeping
deliberately-disabled features (exchange, marketplace, federated, plugins — all
legitimately raising ``HTTPException(503, "... not available")`` when off by
config) opened the GLOBAL ``_breaker`` and then 503'd unrelated routes for the
whole recovery window.

``FeatureDisabledError`` marks such responses with the
``X-DistLLM-Feature-Disabled`` header; the middleware records those as neutral
(no success, no failure — a success would wrongly reset the failure streak).
Plain 503s (coordinator missing, no model loaded, …) must still count.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.circuit_breaker_middleware import (
    CircuitBreakerMiddleware,
    CircuitState,
    ServerCircuitBreaker,
    get_circuit_breaker,
)
from distllm.api.errors import FEATURE_DISABLED_HEADER, FeatureDisabledError
from distllm.api.server_errors import register_exception_handlers

ADMIN_KEY = "test-feature-503-admin-key"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_breaker():
    """The breaker is a process-wide singleton — reset it around every test."""
    breaker = get_circuit_breaker()
    breaker.reset()
    yield
    breaker.reset()


def _feature_app() -> FastAPI:
    """Minimal app: one route raising the typed feature-off exception."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/feature-off")
    async def feature_off():
        raise FeatureDisabledError("demo-feature", "Demo feature not available")

    @app.get("/plain-503")
    async def plain_503():
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    app.add_middleware(CircuitBreakerMiddleware)
    return app


# ---------------------------------------------------------------------------
# Unit: marker => neutral; no marker => failure
# ---------------------------------------------------------------------------


class TestMiddlewareNeutralRecording:
    def test_feature_503_with_marker_never_trips(self):
        """N marked 503s must not change state, counts, or the streak."""
        client = TestClient(_feature_app())
        breaker = get_circuit_breaker()
        breaker._threshold = 2  # trip fast if the marked 503s were failures

        for _ in range(6):
            resp = client.get("/feature-off")
            assert resp.status_code == 503

        assert breaker.state == CircuitState.CLOSED
        stats = breaker.stats()
        assert stats["failures"] == 0
        assert stats["successes"] == 0
        assert stats["total_trips"] == 0
        assert breaker._recent_results == []

        # A normal request still flows through.
        assert client.get("/ok").status_code == 200

    def test_plain_503_without_marker_still_trips(self):
        """An ordinary 503 (coordinator missing, …) is still a failure."""
        client = TestClient(_feature_app())
        breaker = get_circuit_breaker()
        breaker._threshold = 3

        for _ in range(3):
            assert client.get("/plain-503").status_code == 503

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats()["failures"] >= 3

    def test_marker_response_body_keeps_detail_and_header(self):
        client = TestClient(_feature_app())
        resp = client.get("/feature-off")
        assert resp.status_code == 503
        assert resp.headers.get(FEATURE_DISABLED_HEADER) == "1"
        # The body uses the same shape as the app's regular HTTPException
        # handler (the structured envelope) — no special-cased wire format.
        body = resp.json()
        message = body.get("detail") or body.get("error", {}).get("message")
        assert message == "Demo feature not available"


class TestRecordNeutral:
    def test_record_neutral_releases_half_open_probe_slot(self):
        cb = ServerCircuitBreaker(
            failure_threshold=1, recovery_timeout_s=0.01, half_open_max_calls=1
        )
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        assert cb.allow_request() is True   # reserves the only probe slot
        assert cb.allow_request() is False  # capacity exhausted

        failures_before = cb.stats()["failures"]
        cb.record_neutral()

        # Slot released: a new probe is admitted.
        assert cb.allow_request() is True
        stats = cb.stats()
        assert stats["state"] == "half_open"
        assert stats["successes"] == 0
        assert stats["half_open_successes"] == 0
        assert stats["failures"] == failures_before

    def test_record_neutral_does_not_change_closed_counters(self):
        cb = ServerCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        before = (cb.stats()["failures"], cb.stats()["successes"], cb.state)

        cb.record_neutral()

        after = (cb.stats()["failures"], cb.stats()["successes"], cb.state)
        assert after == before, "neutral must neither count as failure nor reset the streak"


# ---------------------------------------------------------------------------
# End-to-end: sweeping disabled features must not open the global breaker
# ---------------------------------------------------------------------------


DISABLED_FEATURE_PATHS = [
    "/v1/marketplace/listings",
    "/v1/marketplace/stats",
    "/v1/exchange/prompts",
    "/v1/exchange/stats",
    "/v1/federated/stats",
    "/v1/federated/versions",
    "/v1/plugins",
]


@pytest.fixture
def real_app_client(monkeypatch):
    """Real server app with auth pinned to a fresh admin key."""
    from distllm.api.api_state import reset_app_state_for_testing
    from distllm.core.api_key_store import reset_api_key_store

    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv(
        "API_KEYS",
        '{"keys": [{"key": "' + ADMIN_KEY + '", "role": "admin", "label": "t"}]}',
    )
    reset_api_key_store()
    reset_app_state_for_testing()

    from distllm.api.server import app

    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = f"Bearer {ADMIN_KEY}"
    yield client

    reset_app_state_for_testing()
    reset_api_key_store()


class TestEndToEndFeatureSweep:
    def test_sweep_of_disabled_features_does_not_open_global_breaker(
        self, real_app_client
    ):
        breaker = get_circuit_breaker()

        swept = 0
        for _ in range(3):  # 21 deliberately-disabled requests
            for path in DISABLED_FEATURE_PATHS:
                resp = real_app_client.get(path)
                assert resp.status_code == 503, (path, resp.status_code, resp.text)
                assert resp.headers.get(FEATURE_DISABLED_HEADER) == "1", (
                    f"{path} must carry the feature-disabled marker"
                )
                swept += 1

        assert swept > breaker.stats()["threshold"]
        assert breaker.state == CircuitState.CLOSED, (
            "feature-gated 503s must not trip the global circuit breaker"
        )
        assert breaker.stats()["failures"] == 0

        # An unrelated normal route is not rejected by the breaker.
        normal = real_app_client.get("/v1/models")
        assert normal.status_code == 200, normal.text

    def test_feature_disabled_detail_text_preserved(self, real_app_client):
        """The live app's error envelope still carries the original detail."""
        resp = real_app_client.get("/v1/marketplace/listings")
        assert resp.status_code == 503
        assert resp.headers.get(FEATURE_DISABLED_HEADER) == "1"
        # The live app customises the HTTPException handler; the message must
        # survive in whatever envelope that handler produces.
        body = resp.json()
        message = body.get("detail") or body.get("error", {}).get("message")
        assert message == "Marketplace not available"

    def test_gossip_disabled_is_neutral(self, real_app_client):
        """Config-off gossip 503 is the same class: marker + neutral.

        ``Gossip protocol is not enabled`` is a feature switch, not a
        downstream outage — the verifier found it still counted as a breaker
        failure after the initial fix.
        """
        breaker = get_circuit_breaker()
        breaker._threshold = 2
        for _ in range(4):
            resp = real_app_client.post(
                "/api/v1/gossip/exchange", json={"node_id": "peer-x"}
            )
            assert resp.status_code == 503, resp.text
            assert resp.headers.get(FEATURE_DISABLED_HEADER) == "1"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats()["failures"] == 0
