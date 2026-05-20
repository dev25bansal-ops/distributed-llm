"""Unit tests for distllm.dashboard.ws_handler."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distllm.dashboard.ws_handler import (
    KNOWN_METRIC_CATEGORIES,
    ConnectionManager,
    MetricsCollector,
    parse_client_message,
)


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class TestConnectionManager:
    @pytest.fixture
    def mgr(self):
        return ConnectionManager()

    @pytest.fixture
    def ws(self):
        m = AsyncMock()
        m.__class__.__name__ = "WebSocket"
        return m

    @pytest.mark.asyncio
    async def test_connect_adds_to_active(self, mgr, ws):
        await mgr.connect(ws)
        assert ws in mgr.active_connections
        assert mgr.connection_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes(self, mgr, ws):
        await mgr.connect(ws)
        mgr.disconnect(ws)
        assert ws not in mgr.active_connections
        assert mgr.connection_count == 0

    @pytest.mark.asyncio
    async def test_subscribe_all_metrics(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, metric_types=None, interval=2.0)
        assert mgr.wants_metric(ws, "latency") is True
        assert mgr.wants_metric(ws, "gpu") is True
        assert mgr.get_interval(ws) == 2.0

    @pytest.mark.asyncio
    async def test_subscribe_filtered(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, metric_types=["latency", "gpu"], interval=0.5)
        assert mgr.wants_metric(ws, "latency") is True
        assert mgr.wants_metric(ws, "gpu") is True
        assert mgr.wants_metric(ws, "nodes") is False
        assert mgr.wants_metric(ws, "cost") is False

    @pytest.mark.asyncio
    async def test_clients_due_returns_ready(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, interval=0.01)
        due = mgr.clients_due(time.time() + 1)
        assert ws in due

    @pytest.mark.asyncio
    async def test_clients_due_skips_recent(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, interval=10.0)
        mgr.mark_sent(ws, time.time())
        due = mgr.clients_due(time.time())
        assert ws not in due

    @pytest.mark.asyncio
    async def test_interval_clamped(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, interval=0.05)
        assert mgr.get_interval(ws) == 0.2
        mgr.subscribe(ws, interval=20.0)
        assert mgr.get_interval(ws) == 10.0

    @pytest.mark.asyncio
    async def test_broadcast_filtered_respects_subscription(self, mgr, ws):
        await mgr.connect(ws)
        mgr.subscribe(ws, metric_types=["latency"])
        msg = {"test": True}
        await mgr.broadcast_filtered(msg, metric_category="gpu")
        ws.send_text.assert_not_called()
        await mgr.broadcast_filtered(msg, metric_category="latency")
        ws.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_to_handles_disconnect(self, mgr, ws):
        await mgr.connect(ws)
        ws.send_text = AsyncMock(side_effect=Exception("gone"))
        await mgr.send_to(ws, {"test": True})
        assert ws not in mgr.active_connections


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector(max_history=10)

    def test_record_request(self, collector):
        collector.record_request(
            latency_ms=150.0, ttft_ms=50.0,
            tokens_per_sec=25.0, model="test-model",
            endpoint="/v1/chat", cost=0.001,
        )
        summary = collector.summary()
        assert summary["latency"]["avg"] == 150.0
        assert summary["ttft"]["avg"] == 50.0
        assert summary["throughput"]["tokens_per_sec_avg"] == 25.0
        assert summary["requests_by_model"]["test-model"] == 1
        assert summary["requests_by_endpoint"]["/v1/chat"] == 1

    def test_kv_cache(self, collector):
        collector.record_kv_cache(hit=True)
        collector.record_kv_cache(hit=True)
        collector.record_kv_cache(hit=False)
        assert collector.kv_hit_rate() == 2 / 3

    def test_speculative(self, collector):
        collector.record_speculative(draft_count=10, accepted_count=8)
        assert collector.spec_acceptance_rate() == 0.8

    def test_queue_and_active(self, collector):
        collector.record_queue_depth(5)
        collector.record_queue_depth(8)
        collector.record_active_requests(3)
        collector.record_active_requests(4)
        summary = collector.summary()
        assert summary["queue_depth"]["avg"] == 6.5
        assert summary["active_requests"]["avg"] == 3.5

    def test_latency_histogram(self, collector):
        collector.record_request(latency_ms=30)
        collector.record_request(latency_ms=80)
        collector.record_request(latency_ms=300)
        hist = collector.latency_histogram()
        assert hist["50"] == 1
        assert hist["100"] == 1
        assert hist["500"] == 1


# ---------------------------------------------------------------------------
# parse_client_message
# ---------------------------------------------------------------------------

class TestParseClientMessage:
    def test_subscribe_all(self):
        msg = json.dumps({"type": "subscribe", "interval": 2.0})
        result = parse_client_message(msg)
        assert result["type"] == "subscribe"
        assert result["metrics"] is None
        assert result["interval"] == 2.0

    def test_subscribe_filtered(self):
        msg = json.dumps({"type": "subscribe", "metrics": ["latency", "gpu"]})
        result = parse_client_message(msg)
        assert result["type"] == "subscribe"
        assert result["metrics"] == ["latency", "gpu"]

    def test_ping(self):
        result = parse_client_message('{"type": "ping"}')
        assert result["type"] == "ping"

    def test_pong(self):
        result = parse_client_message('{"type": "pong"}')
        assert result["type"] == "pong"

    def test_invalid_json(self):
        result = parse_client_message("not-json")
        assert result["type"] == "error"

    def test_unknown_command(self):
        result = parse_client_message('{"type": "unknown"}')
        assert result["type"] == "error"

    def test_invalid_metrics_type(self):
        result = parse_client_message('{"type": "subscribe", "metrics": "latency"}')
        assert result["type"] == "error"


# ---------------------------------------------------------------------------
# KNOWN_METRIC_CATEGORIES
# ---------------------------------------------------------------------------

class TestKnownCategories:
    def test_contains_expected(self):
        for cat in ("latency", "ttft", "throughput", "kv_cache",
                    "speculative", "cost", "nodes", "gpu"):
            assert cat in KNOWN_METRIC_CATEGORIES

    def test_no_empty_string(self):
        assert "" not in KNOWN_METRIC_CATEGORIES


# ---------------------------------------------------------------------------
# stream_metrics_sse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStreamMetricsSSE:
    async def test_no_coordinator(self):
        from distllm.dashboard.ws_handler import stream_metrics_sse
        events = []
        async for line in stream_metrics_sse(None, requested_metrics=None, interval=0.01):
            events.append(line)
            if len(events) >= 2:
                break
        assert any("No coordinator available" in e for e in events)

    async def test_connected_event_on_start(self):
        coord = MagicMock()
        coord.model_name = "test"
        coord.nodes = {}
        coord.scheduler = None
        coord.prefix_cache = None
        from distllm.dashboard.ws_handler import stream_metrics_sse
        gen = stream_metrics_sse(coord, requested_metrics=None, interval=0.01)
        first = await anext(gen)
        assert "event: connected" in first

    async def test_filtered_metrics(self):
        coord = MagicMock()
        coord.model_name = "test"
        coord.nodes = {}
        coord.scheduler = None
        coord.prefix_cache = None
        from distllm.dashboard.ws_handler import stream_metrics_sse
        gen = stream_metrics_sse(coord, requested_metrics={"nodes"}, interval=0.01)
        await anext(gen)  # skip "connected" event
        second = await anext(gen)
        assert "event: metric" in second
        data = json.loads(second.split("data: ", 1)[1])
        assert "timestamp" in data
        assert "nodes" in data
        assert "model" not in data  # filtered out

    async def test_all_metrics_errors_gracefully(self):
        """When full snapshot serialization fails, SSE sends an error event."""
        coord = MagicMock()
        coord.model_name = "test"
        coord.nodes = {}
        coord.scheduler = None
        coord.prefix_cache = None
        from distllm.dashboard.ws_handler import stream_metrics_sse
        gen = stream_metrics_sse(coord, requested_metrics=None, interval=0.01)
        await anext(gen)  # skip "connected"
        second = await anext(gen)
        assert "event: error" in second

    async def test_includes_gpu_utilization(self):
        node = MagicMock()
        node.healthy = True
        node.host = "10.0.0.1"
        node.port = 5001
        node.start_layer = 0
        node.end_layer = 5
        node.gpu_utilization = 75.0
        node.role = "worker"
        node.kv_cache_stats = None
        coord = MagicMock()
        coord.model_name = "test"
        coord.nodes = {"node-0": node}
        coord.scheduler = None
        coord.prefix_cache = None
        from distllm.dashboard.ws_handler import stream_metrics_sse
        gen = stream_metrics_sse(coord, requested_metrics={"nodes"}, interval=0.01)
        await anext(gen)  # skip "connected"
        second = await anext(gen)
        data = json.loads(second.split("data: ", 1)[1])
        assert "nodes" in data
        assert data["nodes"]["node-0"]["gpu_utilization"] == 75.0
