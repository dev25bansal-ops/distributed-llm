"""Tests for WebSocket metrics authentication (Authorization header and
Sec-WebSocket-Protocol browser fallback). See I-07 follow-up.

These verify the /ws/metrics endpoint authenticates via the
Sec-WebSocket-Protocol subprotocol (browser clients, which cannot set
headers on a WebSocket) as well as the Authorization header (non-browser
clients), and that missing/invalid keys are rejected with code 4001.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

from tests.api.stubs import CoordinatorStub

TEST_KEY = "sk-test-ws-auth-12345"
API_KEYS_JSON = '{"keys": [{"key": "' + TEST_KEY + '", "role": "admin", "label": "test"}]}'


@pytest.fixture(autouse=True)
def _configure_key_store(monkeypatch):
    monkeypatch.setenv("API_KEYS", API_KEYS_JSON)
    monkeypatch.delenv("DISTLLM_ALLOW_ANONYMOUS_METRICS", raising=False)
    reset_api_key_store()
    # Force the singleton to reload from the env var we just set.
    get_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture
def coordinator():
    coord = CoordinatorStub()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord._shutting_down = False
    coord.get_metrics = lambda: {"requests_total": 42}
    coord.metrics_exporter = None
    coord.scheduler = None
    coord.prefix_cache = None
    return coord


@pytest.fixture(autouse=True)
def _coordinator(coordinator):
    original = g.coordinator
    g.coordinator = coordinator
    yield
    g.coordinator = original


class TestWsMetricsAuth:
    def test_sec_websocket_protocol_auth_succeeds(self):
        """Browser path: key carried via Sec-WebSocket-Protocol subprotocol."""
        with TestClient(app).websocket_connect(
            "/ws/metrics",
            subprotocols=["Bearer", TEST_KEY],
        ) as ws:
            # A successful, authenticated connection should receive a snapshot.
            data = ws.receive_json()
            assert data["type"] == "metrics"

    def test_authorization_header_auth_succeeds(self):
        """Non-browser path: key carried via Authorization header."""
        with TestClient(app).websocket_connect(
            "/ws/metrics",
            headers={"Authorization": f"Bearer {TEST_KEY}"},
        ) as ws:
            data = ws.receive_json()
            assert data["type"] == "metrics"

    def test_missing_key_rejected(self):
        """No credential of any kind → connection closed with code 4001."""
        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(app).websocket_connect("/ws/metrics") as ws:
                ws.receive_json()
        assert exc.value.code == 4001

    def test_invalid_key_rejected(self):
        """Wrong key via subprotocol → connection closed with code 4001."""
        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(app).websocket_connect(
                "/ws/metrics",
                subprotocols=["Bearer", "sk-wrong-key"],
            ) as ws:
                ws.receive_json()
        assert exc.value.code == 4001
