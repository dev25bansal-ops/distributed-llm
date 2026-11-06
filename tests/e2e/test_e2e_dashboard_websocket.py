"""E2E: Dashboard WebSocket connectivity.

Tests the real-time metrics WebSocket:
1. WebSocket connect/disconnect lifecycle
2. Subscribe to metric categories
3. Ping/pong keep-alive
4. Metrics broadcaster integration
5. SSE metrics stream
6. Dashboard HTML page serving
7. REST API endpoints on dashboard app
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e]


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def dashboard_coordinator():
    """Coordinator with dashboard-accessible attributes."""
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.scheduler = None
    coord.prefix_cache = None
    coord._startup_time = time.time()
    coord.metrics_exporter = None
    coord._shutting_down = False
    coord._spec_decoder = None
    # get_metrics returns a dict
    coord.get_metrics.return_value = {
        "latency": {"p50": 42, "p95": 150, "p99": 300},
        "requests_served": 100,
    }
    return coord


@pytest.fixture
def dashboard_client(dashboard_coordinator):
    """FastAPI TestClient with the API server (dashboard embedded)."""
    from fastapi.testclient import TestClient
    from distllm.api.server import app
    from distllm.api.app_state import AppState

    # Set coordinator on the server application state
    import distllm.api.server as server_module
    server_module.state.coordinator = dashboard_coordinator
    return TestClient(app)





# ====================================================================
# WebSocket Tests
# ====================================================================

class TestDashboardWebSocketConnectivity:
    """WebSocket connection lifecycle on the dashboard app."""

    def test_websocket_connect(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            assert ws is not None

    def test_websocket_ping_pong(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"
            assert "timestamp" in resp

    def test_websocket_subscribe(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "subscribe",
                "metrics": ["latency", "gpu"],
                "interval": 2.0,
            })
            resp = ws.receive_json()
            assert resp["type"] == "subscribed"
            assert resp["metrics"] == ["latency", "gpu"]
            assert resp["interval"] == 2.0

    def test_websocket_subscribe_all_metrics(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe"})
            resp = ws.receive_json()
            assert resp["type"] == "subscribed"

    def test_websocket_invalid_json(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_websocket_unknown_command(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "unknown"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_websocket_invalid_metrics_type(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe", "metrics": "not_a_list"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_multiple_websocket_connections(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as ws1:
            with dashboard_client.websocket_connect("/ws") as ws2:
                ws1.send_json({"type": "ping"})
                ws2.send_json({"type": "ping"})
                r1 = ws1.receive_json()
                r2 = ws2.receive_json()
                assert r1["type"] == "pong"
                assert r2["type"] == "pong"

    def test_websocket_disconnect_cleanup(self, dashboard_client):
        from distllm.dashboard.ws_handler import manager
        initial = manager.connection_count
        with dashboard_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            ws.receive_json()
        assert manager.connection_count == initial


# Dashboard REST API Tests
# ====================================================================

class TestDashboardRESTAPI:
    """REST API endpoints served by the dashboard app."""

    def test_dashboard_index(self, dashboard_client):
        resp = dashboard_client.get("/")
        assert resp.status_code == 200

    def test_dashboard_api_status(self, dashboard_client):
        resp = dashboard_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "nodes" in data

    def test_dashboard_api_nodes(self, dashboard_client):
        resp = dashboard_client.get("/api/nodes")
        assert resp.status_code == 200

    def test_dashboard_api_metrics(self, dashboard_client):
        resp = dashboard_client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency" in data

    def test_dashboard_api_waterfall(self, dashboard_client):
        resp = dashboard_client.get("/api/requests/waterfall")
        assert resp.status_code == 200

    def test_dashboard_api_config(self, dashboard_client):
        resp = dashboard_client.post(
            "/api/config",
            json={"batch_size": 32},
        )
        assert resp.status_code in (200, 503)

    def test_dashboard_no_coordinator(self):
        from fastapi.testclient import TestClient
        from distllm.api.server import app
        import distllm.api.server as svr
        svr.state.coordinator = None
        client = TestClient(app)
        resp = client.get("/api/requests/waterfall")
        assert resp.status_code == 200
        assert resp.json() == []


# ====================================================================
# SSE Metrics Stream Tests
# ====================================================================

class TestSSEMetricsStream:
    """SSE (Server-Sent Events) stream for metrics."""

    def test_sse_stream_connected_event(self, e2e_coordinator):
        from distllm.dashboard.ws_handler import stream_metrics_sse
        import asyncio

        async def run():
            gen = stream_metrics_sse(e2e_coordinator, interval=10.0)
            events = []
            async for event in gen:
                events.append(event)
                if len(events) >= 2:
                    break
            return events

        events = asyncio.run(run())
        assert len(events) >= 1
        assert "connected" in events[0] or "event: connected" in events[0]

    def test_sse_stream_with_null_coordinator(self):
        from distllm.dashboard.ws_handler import stream_metrics_sse
        import asyncio

        async def run():
            gen = stream_metrics_sse(None, interval=10.0)
            events = []
            async for event in gen:
                events.append(event)
                break
            return events

        events = asyncio.run(run())
        assert len(events) == 1


# ====================================================================
# MetricsCollector Tests
# ====================================================================

class TestMetricsCollector:
    """In-memory metrics collector used by the dashboard."""

    def test_collector_record_and_summary(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector(max_history=100)
        mc.record_request(latency_ms=50, ttft_ms=10, tokens_per_sec=20, cost=0.01)
        mc.record_kv_cache(hit=True)
        mc.record_kv_cache(hit=True)
        mc.record_kv_cache(hit=False)
        summary = mc.summary()
        assert summary["latency"]["avg"] == 50
        assert summary["kv_cache"]["hit_rate"] == 2 / 3

    def test_collector_gpu_util(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        mc.record_gpu_util("node-1", 85.0)
        mc.record_gpu_util("node-1", 90.0)
        assert len(mc._gpu_util["node-1"]) == 2

    def test_collector_queue_depth(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        mc.record_queue_depth(3)
        mc.record_queue_depth(5)
        summary = mc.summary()
        assert summary["queue_depth"]["max"] == 5

    def test_collector_spec_decoder(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        mc.record_speculative(draft_count=10, accepted_count=8)
        assert mc.spec_acceptance_rate() == 0.8

    def test_collector_empty_summary(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        summary = mc.summary()
        assert summary["latency"]["avg"] == 0
        assert summary["throughput"]["tokens_per_sec_avg"] == 0

    def test_collector_latency_histogram(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        mc.record_request(latency_ms=30)
        mc.record_request(latency_ms=150)
        mc.record_request(latency_ms=600)
        hist = mc.latency_histogram()
        assert hist["50"] >= 1
        assert hist["200"] >= 1
        assert hist["1000"] >= 1

    def test_collector_requests_by_model(self):
        from distllm.dashboard.ws_handler import MetricsCollector
        mc = MetricsCollector()
        mc.record_request(latency_ms=10, model="gpt2")
        mc.record_request(latency_ms=20, model="gpt2")
        mc.record_request(latency_ms=30, model="llama")
        summary = mc.summary()
        assert summary["requests_by_model"]["gpt2"] == 2
        assert summary["requests_by_model"]["llama"] == 1
