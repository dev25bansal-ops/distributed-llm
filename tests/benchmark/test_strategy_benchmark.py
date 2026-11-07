"""Per-strategy throughput benchmark for CI.

Measures and compares the throughput of each pipeline strategy
(sequential, overlap, staged, async_1F1B) using the PipelineSimulator
and mock gRPC nodes with configurable latency.

Run with:
    pytest tests/benchmark/test_strategy_benchmark.py -v
    CI=1 pytest tests/benchmark/test_strategy_benchmark.py -v --benchmark-enable
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import pytest
import torch

from distllm.dist.pipeline import (
    PipelineOrchestrator,
    PipelineSimulator,
    PipelineStrategy,
    StrategySelector,
)

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not os.environ.get("CI") and not os.environ.get("DISTLLM_RUN_BENCHMARKS"),
        reason="Benchmarks require CI=1 or DISTLLM_RUN_BENCHMARKS=1",
    ),
]


def _make_pipeline(num_nodes, total_layers, node_latency_ms=5.0):
    """Build an orchestrator with mock gRPC clients via the public API.

    Uses ``register_node`` (orchestrator.py:256) so layer-ordered
    ``node_order`` is maintained by the orchestrator itself — the old
    ``p.node_order.append(...)`` was a silent no-op because the property
    returns a copy (orchestrator.py:245-248).  The removed
    ``_rebuild_fallback_map`` / ``_rebuild_stages`` seams are gone: the
    orchestrator now resolves fallback per call via ``_find_fallback_node``
    (orchestrator.py:333) and walks ``node_order`` in ``run_pipeline``
    (orchestrator.py:398).
    """
    p = PipelineOrchestrator(resource_mgr=MagicMock(), total_layers=total_layers)

    layers_per_node = total_layers // num_nodes
    for i in range(num_nodes):
        node_id = f"n{i}"
        p.register_node(
            node_id, "localhost", 50051 + i,
            start_layer=i * layers_per_node,
            end_layer=(i + 1) * layers_per_node - 1,
            total_layers=total_layers,
        )
        node = p.nodes[node_id]

        def _forward(hidden_states=None, kv_cache=None, request_id=None,
                     _latency_ms=node_latency_ms):
            time.sleep(_latency_ms / 1000.0)
            return torch.zeros_like(hidden_states)

        node.client = MagicMock()
        node.client.cluster_key = None
        # run_pipeline / run_pipeline_microbatched call
        # client.forward(hidden_states=..., kv_cache=..., request_id=...)
        # and pass the return value on as the hidden states.
        node.client.forward = MagicMock(side_effect=_forward)
        node.gpu_compute_tflops = 312.0

    return p


# ─── Analytical Simulator Benchmarks ────────────────────────────────


class TestSimulatorStrategyComparison:
    """Compare strategy latencies using the analytical PipelineSimulator."""

    @pytest.fixture
    def simulator(self):
        return PipelineSimulator(
            model_size="7B",
            gpu_tflops=312.0,
            gpu_bandwidth_gbps=600.0,
            interconnect_gbps=50.0,
        )

    def test_2_nodes(self, simulator):
        result = simulator.simulate(num_nodes=2, num_layers=32, batch_size=1, seq_len=2048)
        strategies = result["strategies"]
        assert strategies["sequential"]["latency_ms"] > 0
        assert strategies["overlap"]["latency_ms"] > 0
        print(f"\n  2 nodes: seq={strategies['sequential']['latency_ms']:.1f}ms, "
              f"overlap={strategies['overlap']['latency_ms']:.1f}ms")

    def test_4_nodes(self, simulator):
        result = simulator.simulate(num_nodes=4, num_layers=32, batch_size=1, seq_len=2048)
        strategies = result["strategies"]
        assert strategies["sequential"]["latency_ms"] > 0
        assert strategies["staged"]["latency_ms"] > 0
        print(f"\n  4 nodes: seq={strategies['sequential']['latency_ms']:.1f}ms, "
              f"staged={strategies['staged']['latency_ms']:.1f}ms")

    def test_8_nodes(self, simulator):
        result = simulator.simulate(num_nodes=8, num_layers=32, batch_size=1, seq_len=2048)
        strategies = result["strategies"]
        assert strategies["async_1f1b"]["latency_ms"] > 0
        print(f"\n  8 nodes: async_1f1b={strategies['async_1f1b']['latency_ms']:.1f}ms, "
              f"bubble={strategies['async_1f1b']['bubble_ratio']:.3f}")

    def test_recommendation(self, simulator):
        result = simulator.simulate(num_nodes=4, num_layers=32, batch_size=1, seq_len=2048)
        assert "recommendation" in result


# ─── StrategySelector Benchmarks ────────────────────────────────────


class TestStrategySelectorBenchmark:
    """Benchmark the StrategySelector's per-request selection latency."""

    def test_selection_latency(self):
        selector = StrategySelector(model_size="7B")
        nodes = {}
        for i in range(4):
            node = MagicMock()
            node.gpu_compute_tflops = 312.0
            nodes[f"n{i}"] = node

        import timeit
        elapsed = timeit.timeit(
            lambda: selector.select_strategy(
                num_nodes=4, total_layers=32, batch_size=1, seq_len=2048,
                current_load=2, nodes=nodes,
            ),
            number=1000,
        )
        avg_us = elapsed / 1000 * 1e6
        print(f"\n  StrategySelector: {avg_us:.1f}µs per selection")
        assert avg_us < 1000  # Should be under 1ms

    def test_selection_with_history(self):
        selector = StrategySelector(model_size="7B")
        nodes = {}
        for i in range(4):
            node = MagicMock()
            node.gpu_compute_tflops = 312.0
            nodes[f"n{i}"] = node

        for _ in range(50):
            selector.record_strategy_latency("overlap", 15.0 + (hash(str(_)) % 10))

        strategy = selector.select_strategy(
            num_nodes=4, total_layers=32, batch_size=1, seq_len=2048,
            current_load=2, nodes=nodes,
        )
        assert strategy in (PipelineStrategy.OVERLAP, PipelineStrategy.STAGED,
                            PipelineStrategy.SEQUENTIAL)


# ─── Mock-Based Throughput Benchmarks ───────────────────────────────


class TestMockThroughputBenchmark:
    """Measure throughput of each strategy using mock gRPC nodes."""

    @pytest.fixture(params=[2, 4, 8], ids=lambda n: f"{n}nodes")
    def pipeline(self, request):
        n = request.param
        return _make_pipeline(n, 32, node_latency_ms=5.0)

    def test_sequential_throughput(self, pipeline, benchmark):
        input_ids = torch.randint(0, 1000, (1, 64))
        kv = pipeline.create_node_kv_caches()

        def run():
            return pipeline.run_pipeline(input_ids, kv, "bench-seq")

        result = benchmark(run)
        assert tuple(result.shape) == (1, 64)

    def test_overlap_throughput(self, pipeline, benchmark):
        input_ids = torch.randint(0, 1000, (1, 64))
        kv = pipeline.create_node_kv_caches()

        def run():
            return pipeline.run_pipeline_overlap(input_ids, kv, "bench-ov")

        result = benchmark(run)
        assert tuple(result.shape) == (1, 64)


# ─── Batch Size Scaling ─────────────────────────────────────────────


class TestBatchSizeScaling:
    """Measure how throughput scales with batch size."""

    @pytest.fixture
    def pipeline(self):
        return _make_pipeline(4, 32, node_latency_ms=2.0)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_scaling(self, pipeline, batch_size):
        import torch
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        kv = pipeline.create_node_kv_caches()

        start = time.monotonic()
        for _ in range(5):
            pipeline.run_pipeline(input_ids, kv, f"bench-b{batch_size}")
        elapsed = (time.monotonic() - start) / 5

        print(f"\n  batch={batch_size}: {elapsed*1000:.1f}ms/step, "
              f"{batch_size/elapsed:.0f} tok/s")


# ─── Sequence Length Scaling ────────────────────────────────────────


class TestSeqLenScaling:
    """Measure how latency scales with sequence length."""

    @pytest.fixture
    def pipeline(self):
        return _make_pipeline(4, 32, node_latency_ms=2.0)

    @pytest.mark.parametrize("seq_len", [32, 128, 512, 2048])
    def test_seq_scaling(self, pipeline, seq_len):
        import torch
        input_ids = torch.randint(0, 1000, (1, seq_len))
        kv = pipeline.create_node_kv_caches()

        start = time.monotonic()
        pipeline.run_pipeline(input_ids, kv, "bench-seq")
        elapsed = (time.monotonic() - start) * 1000

        print(f"\n  seq_len={seq_len}: {elapsed:.1f}ms")
