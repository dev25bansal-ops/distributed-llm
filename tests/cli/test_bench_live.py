"""Tests: ``distllm bench live`` — websocket live-benchmark streaming."""

from __future__ import annotations

import asyncio
import json

import pytest

from distllm.cli.bench_live import LiveBenchmark, LiveMetric


class TestLiveMetricSerialization:
    def test_to_json_round_trips(self):
        m = LiveMetric(
            ts=1.0,
            iteration=1,
            ttft_ms=120.5,
            tokens_per_sec=32.0,
            prompt_tokens=128,
            completion_tokens=64,
            gpu_mem_gb=14.2,
            rss_mem_gb=0.3,
        )
        data = json.loads(m.to_json())
        assert data["iteration"] == 1
        assert data["ttft_ms"] == 120.5
        assert data["tokens_per_sec"] == 32.0
        assert data["prompt_tokens"] == 128


class TestLiveBenchmarkSnapshot:
    def test_snapshot_increments_iteration(self):
        b = LiveBenchmark(model="test")
        m1 = b.snapshot()
        m2 = b.snapshot()
        assert m1.iteration == 1
        assert m2.iteration == 2

    def test_snapshot_appends_to_history(self):
        b = LiveBenchmark(model="test")
        assert len(b.history) == 0
        b.snapshot()
        b.snapshot()
        assert len(b.history) == 2

    def test_snapshot_metrics_in_plausible_range(self):
        b = LiveBenchmark(model="test")
        m = b.snapshot()
        # Synthetic ranges chosen so dashboards render useful values;
        # assert bounds rather than exact values.
        assert 20.0 <= m.ttft_ms <= 500.0
        assert 1.0 <= m.tokens_per_sec <= 200.0
        assert m.prompt_tokens > 0
        assert m.completion_tokens > 0


class TestLiveBenchmarkClientTracking:
    @pytest.mark.asyncio
    async def test_register_and_unregister(self):
        b = LiveBenchmark(model="x")
        fake_ws = object()
        await b.register(fake_ws)
        assert fake_ws in b._clients
        await b.unregister(fake_ws)
        assert fake_ws not in b._clients

    @pytest.mark.asyncio
    async def test_register_dedup(self):
        b = LiveBenchmark(model="x")
        fake_ws = object()
        await b.register(fake_ws)
        await b.register(fake_ws)
        # Sets dedupe by identity
        assert len([w for w in b._clients if w is fake_ws]) == 1


class TestLiveBenchmarkBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_to_clients(self):
        b = LiveBenchmark(model="x")

        sent: list[str] = []

        class FakeWS:
            async def send(self, msg: str) -> None:
                sent.append(msg)

        ws = FakeWS()
        await b.register(ws)
        metric = b.snapshot()
        await b.broadcast(metric)
        assert len(sent) == 1
        parsed = json.loads(sent[0])
        assert parsed["iteration"] == metric.iteration

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self):
        b = LiveBenchmark(model="x")

        class DeadWS:
            async def send(self, msg: str) -> None:
                raise RuntimeError("closed")

        ws = DeadWS()
        await b.register(ws)
        await b.broadcast(b.snapshot())
        assert ws not in b._clients

    @pytest.mark.asyncio
    async def test_broadcast_no_clients_no_error(self):
        b = LiveBenchmark(model="x")
        # Should be a no-op when no clients connected
        await b.broadcast(b.snapshot())


class TestLiveBenchmarkRun:
    @pytest.mark.asyncio
    async def test_run_loop_stops(self):
        b = LiveBenchmark(model="x", interval=0.001)
        # Schedule a stop after the first iteration
        async def stopper() -> None:
            await asyncio.sleep(0.005)
            b.stop()

        await asyncio.gather(b.run(), stopper())
        assert b.iteration >= 1


class TestRegisterCommand:
    def test_register_attaches_to_app(self):
        import typer

        from distllm.cli.bench_live import register

        app = typer.Typer()
        register(app)
        names = [c.callback.__name__ for c in app.registered_commands]
        assert "bench_live" in names