"""Tests for WAN disaggregated orchestrator and token accumulator."""

from __future__ import annotations

import asyncio

import pytest

from distllm.dist.pipeline.token_accumulator import TokenAccumulator
from distllm.dist.pipeline.wan_disagg_orchestrator import (
    WanDisaggConfig,
    WanDisaggOrchestrator,
)
from tests.core._stubs import _Stub


class TestTokenAccumulatorBatchAndFlush:
    """TokenAccumulator batch, flush, and should_flush behaviour."""

    def test_batch_and_flush(self) -> None:
        acc = TokenAccumulator(min_batch_size=5, max_tokens=20, flush_interval_s=10)
        for i in range(5):
            acc.add(i)

        assert acc.buffer_size == 5
        assert acc.should_flush() is True

        batch = acc.flush()
        assert batch == [0, 1, 2, 3, 4]
        assert acc.buffer_size == 0
        assert acc.metrics.batch_count == 1

    def test_flush_empty_buffer(self) -> None:
        acc = TokenAccumulator(min_batch_size=2, max_tokens=10)
        batch = acc.flush()
        assert batch == []
        assert acc.metrics.batch_count == 0


@pytest.mark.asyncio
async def test_orchestrator_route_request() -> None:
    """WanDisaggOrchestrator.route_request succeeds for normal conditions."""
    config = WanDisaggConfig(
        prefill_endpoints=("node-a:8001",),
        decode_endpoints=("node-b:8002",),
        wan_timeout_ms=5000,
    )
    orch = WanDisaggOrchestrator(config)

    async def measure_rtt(prefill_node, decode_node):
        return 50.0

    async def run_disagg(prompt, model_name, prefill_node, decode_node, request_id):
        return {
            "output": "Paris",
            "prefill_node": "node-a:8001",
            "decode_node": "node-b:8002",
            "wan_latency_ms": 50.0,
            "prefill_time_ms": 100.0,
            "kv_transfer_time_ms": 200.0,
        }

    orch._measure_wan_rtt = measure_rtt
    orch._run_disaggregated = run_disagg

    result = await orch.route_request("What is the capital of France?", "llama-3-8b")

    assert result["output"] == "Paris"
    assert result["prefill_node"] == "node-a:8001"
    assert result["decode_node"] == "node-b:8002"
    assert result["fallback"] is False


@pytest.mark.asyncio
async def test_orchestrator_fallback_on_high_latency() -> None:
    """WanDisaggOrchestrator falls back to local when WAN latency exceeds threshold."""
    config = WanDisaggConfig(
        prefill_endpoints=("node-a:8001",),
        decode_endpoints=("node-b:8002",),
        wan_timeout_ms=100.0,
    )
    orch = WanDisaggOrchestrator(config)

    async def measure_rtt(prefill_node, decode_node):
        return 500.0

    async def run_local(prompt, model_name, prefill_node):
        return {
            "output": "Paris (local)",
            "prefill_node": "node-a:8001",
            "decode_node": "node-a:8001",
            "wan_latency_ms": 0.0,
        }

    orch._measure_wan_rtt = measure_rtt
    orch._run_local = run_local

    result = await orch.route_request("What is the capital of France?", "llama-3-8b")

    assert result["fallback"] is True
    assert result["output"] == "Paris (local)"
    assert orch.metrics["fallback_count"] == 1
