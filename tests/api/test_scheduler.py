"""Tests for the scheduler tuning API (routes/scheduler.py).

Uses plain Python classes (no MagicMock/Mock/AsyncMock) for
coordinator and scheduler mocks.
"""

import os

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

_TEST_ADMIN_KEY = "test-scheduler-admin-key-999"


# ── Mock classes (zero MagicMock) ─────────────────────────────────────────


class _Lock:
    """No-op context manager used in place of threading.Lock in tests."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Budget:
    """Scheduler budget mock — plain class, no mocks."""

    def __init__(self):
        self.max_batch_size = 32
        self.max_total_tokens = 65536
        self.prefill_slack_ratio = 0.1


class _SchedulerMock:
    """Minimal batch scheduler with all attributes the routes access."""

    def __init__(self):
        self._lock = _Lock()
        self._budget = _Budget()
        self.max_batch_size = 32
        self.max_tokens_per_batch = 65536
        self._aging_interval_s = 5.0
        self._aging_max_boost = 3
        self._aging_enabled = True
        self._max_preempted = 4
        self._starvation_threshold_s = 120.0
        self._enable_chunked_prefill = True
        self._adapt_prefill_budget = True
        self._priority_weights = {"default": 1.0, "high": 2.0}

    def stats(self):
        return {
            "active_requests": 3,
            "pending_requests": 7,
            "preempted_requests": 1,
            "max_batch_size": 32,
            "max_tokens_per_batch": 65536,
            "iteration": 99,
            "total_prefill_tokens": 15000,
            "total_decode_tokens": 72000,
            "chunked_prefill_active": 1,
            "adaptive_batching": True,
            "advanced": {"spec_decoding": True},
        }


class _CoordinatorMock:
    """Minimal coordinator mock.

    Only provides the ``_batch_scheduler`` attribute that the scheduler
    route accesses.  No MagicMock, no unittest.mock whatsoever.
    """

    def __init__(self):
        self._batch_scheduler = _SchedulerMock()


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the singleton key store and configure a test admin key."""
    os.environ["API_KEYS"] = (
        '{"keys": ['
        '{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-scheduler-admin"},'
        '{"key": "test-read-scheduler", "role": "read-only", "label": "test-read"}'
        "]}"
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture(autouse=True)
def _patch_state():
    """Set ``g.coordinator`` (via ``_state``) to a coordinator mock.

    Each test starts with a fresh ``_CoordinatorMock`` so state from one
    test cannot leak into another.
    """
    from distllm.api.api_state import _state

    _state.coordinator = _CoordinatorMock()
    yield
    _state.coordinator = None


@pytest.fixture
def auth_header():
    """Valid admin ``Authorization`` header."""
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


@pytest.fixture
def client():
    """FastAPI ``TestClient`` with the scheduler router and test auth middleware."""
    from distllm.api.routes.scheduler import router

    app = FastAPI()

    class _TestAuthMiddleware(BaseHTTPMiddleware):
        """Simulate ``AuthMiddleware``: validate Bearer token, set ``api_key_role``."""

        async def dispatch(self, request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]
                store = get_api_key_store()
                result = store.authenticate(token)
                if result:
                    request.state.api_key_role = result[1]
                    request.state.api_key_id = result[0]
            return await call_next(request)

    app.add_middleware(_TestAuthMiddleware)
    app.include_router(router)
    return TestClient(app)


# ── Auth & error tests ────────────────────────────────────────────────────


class TestSchedulerAuth:
    """Verify scheduler endpoints require valid admin authentication."""

    def test_no_auth_returns_401(self, client):
        resp = client.get("/v1/scheduler/stats")
        assert resp.status_code == 401

    def test_wrong_role_returns_403(self, client):
        resp = client.get(
            "/v1/scheduler/stats",
            headers={"Authorization": "Bearer test-read-scheduler"},
        )
        assert resp.status_code == 403

    def test_invalid_key_returns_401(self, client):
        resp = client.get(
            "/v1/scheduler/stats",
            headers={"Authorization": "Bearer totally-wrong-key"},
        )
        assert resp.status_code == 401


# ── GET /v1/scheduler/stats ────────────────────────────────────────────────


class TestGetSchedulerStats:
    """GET /v1/scheduler/stats"""

    def test_returns_stats(self, client, auth_header):
        resp = client.get("/v1/scheduler/stats", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_requests"] == 3
        assert data["pending_requests"] == 7
        assert data["preempted_requests"] == 1
        assert data["max_batch_size"] == 32
        assert data["max_tokens_per_batch"] == 65536
        assert data["iteration"] == 99
        assert data["total_prefill_tokens"] == 15000
        assert data["total_decode_tokens"] == 72000
        assert data["chunked_prefill_active"] == 1
        assert data["adaptive_batching"] is True
        assert data["advanced"]["spec_decoding"] is True

    def test_no_scheduler_returns_503(self, client, auth_header):
        from distllm.api.api_state import _state

        coord = _state.coordinator
        coord._batch_scheduler = None
        resp = client.get("/v1/scheduler/stats", headers=auth_header)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Batch scheduler not configured"

    def test_no_coordinator_returns_503(self, client, auth_header):
        from distllm.api.api_state import _state

        orig = _state.coordinator
        _state.coordinator = None
        try:
            resp = client.get("/v1/scheduler/stats", headers=auth_header)
            assert resp.status_code == 503
        finally:
            _state.coordinator = orig


# ── PATCH /v1/scheduler/config ─────────────────────────────────────────────


class TestUpdateSchedulerConfig:
    """PATCH /v1/scheduler/config"""

    def test_update_single_field(self, client, auth_header):
        resp = client.patch(
            "/v1/scheduler/config",
            json={"max_batch_size": 64},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated"] == {"max_batch_size": 64}

    def test_update_multiple_fields(self, client, auth_header):
        resp = client.patch(
            "/v1/scheduler/config",
            json={
                "max_batch_size": 48,
                "max_tokens_per_batch": 131072,
                "aging_enabled": False,
            },
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated"]["max_batch_size"] == 48
        assert data["updated"]["max_tokens_per_batch"] == 131072
        assert data["updated"]["aging_enabled"] is False

    def test_update_budget_fields(self, client, auth_header):
        """Verify ``_budget`` attributes are updated alongside top-level ones."""
        resp = client.patch(
            "/v1/scheduler/config",
            json={
                "max_batch_size": 128,
                "max_tokens_per_batch": 262144,
                "prefill_slack_ratio": 0.25,
            },
            headers=auth_header,
        )
        assert resp.status_code == 200

        from distllm.api.api_state import _state

        sched = _state.coordinator._batch_scheduler
        assert sched._budget.max_batch_size == 128
        assert sched._budget.max_total_tokens == 262144
        assert sched._budget.prefill_slack_ratio == 0.25

    def test_update_empty_body(self, client, auth_header):
        resp = client.patch("/v1/scheduler/config", json={}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated"] == {}

    def test_update_no_scheduler_returns_503(self, client, auth_header):
        from distllm.api.api_state import _state

        coord = _state.coordinator
        scheduler_was = coord._batch_scheduler
        coord._batch_scheduler = None
        try:
            resp = client.patch(
                "/v1/scheduler/config",
                json={"max_batch_size": 64},
                headers=auth_header,
            )
            assert resp.status_code == 503
            assert resp.json()["detail"] == "Batch scheduler not configured"
        finally:
            coord._batch_scheduler = scheduler_was

    def test_update_all_fields(self, client, auth_header):
        """Update every field the ``SchedulerConfigUpdate`` model accepts."""
        payload = {
            "max_batch_size": 64,
            "max_tokens_per_batch": 8192,
            "prefill_slack_ratio": 0.5,
            "aging_interval_s": 30.0,
            "aging_max_boost": 5,
            "aging_enabled": False,
            "max_preempted": 8,
            "starvation_threshold_s": 300.0,
        }
        resp = client.patch(
            "/v1/scheduler/config",
            json=payload,
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        for key, value in payload.items():
            assert data["updated"][key] == value, f"Mismatch for {key}"


# ── GET /v1/scheduler/config ───────────────────────────────────────────────


class TestGetSchedulerConfig:
    """GET /v1/scheduler/config"""

    def test_returns_all_config_fields(self, client, auth_header):
        resp = client.get("/v1/scheduler/config", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_batch_size"] == 32
        assert data["max_tokens_per_batch"] == 65536
        assert data["prefill_slack_ratio"] == 0.1
        assert data["aging_interval_s"] == 5.0
        assert data["aging_max_boost"] == 3
        assert data["aging_enabled"] is True
        assert data["max_preempted"] == 4
        assert data["starvation_threshold_s"] == 120.0
        assert data["enable_chunked_prefill"] is True
        assert data["adapt_prefill_budget"] is True
        assert data["priority_weights"] == {"default": 1.0, "high": 2.0}

    def test_config_no_scheduler_returns_503(self, client, auth_header):
        from distllm.api.api_state import _state

        coord = _state.coordinator
        scheduler_was = coord._batch_scheduler
        coord._batch_scheduler = None
        try:
            resp = client.get("/v1/scheduler/config", headers=auth_header)
            assert resp.status_code == 503
            assert resp.json()["detail"] == "Batch scheduler not configured"
        finally:
            coord._batch_scheduler = scheduler_was

    def test_config_no_coordinator_returns_503(self, client, auth_header):
        from distllm.api.api_state import _state

        orig = _state.coordinator
        _state.coordinator = None
        try:
            resp = client.get("/v1/scheduler/config", headers=auth_header)
            assert resp.status_code == 503
        finally:
            _state.coordinator = orig
