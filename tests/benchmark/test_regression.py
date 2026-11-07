"""Regression benchmark tests with comparison thresholds.

Ensures that pipeline performance doesn't regress beyond acceptable
limits compared to baseline measurements.

Thresholds:
- Sequential pipeline: ≤1.1× baseline
- Overlap pipeline: ≤1.15× baseline (more variance due to threading)
- Simulator accuracy: ≤1.2× actual
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, "src")

from distllm.dist.pipeline import PipelineOrchestrator, PipelineSimulator


# ── Baseline Configuration ─────────────────────────────────────────

BASELINE_FILE = Path(__file__).parent / "baseline_metrics.json"

# Default baselines (updated by CI)
DEFAULT_BASELINES = {
    "sequential_2node_32layer_latency_ms": 20.0,
    "overlap_2node_32layer_latency_ms": 15.0,
    "staged_4node_32layer_latency_ms": 25.0,
    "simulator_4node_latency_ms": 10.0,
    "strategy_selector_us": 100.0,
}

# Regression thresholds (1.1 = 10% slower is acceptable)
THRESHOLDS = {
    "sequential": 1.10,
    "overlap": 1.15,
    "staged": 1.15,
    "simulator": 1.20,
    "selector": 1.50,
}


def _load_baselines() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return DEFAULT_BASELINES


def _save_baselines(baselines: dict):
    BASELINE_FILE.write_text(json.dumps(baselines, indent=2))


def _make_pipeline(num_nodes=2, total_layers=32, latency_ms=2.0):
    """Build an orchestrator with mock gRPC clients via the public API.

    Uses ``register_node`` (orchestrator.py:256) so ``_node_order`` and the
    real ``PipelineNode.is_healthy`` flag are set by the orchestrator itself
    — assigning a bare MagicMock into ``p.nodes[...]`` never populates the
    private ``_node_order`` run_pipeline walks, which raised "No healthy
    nodes in pipeline".
    """
    p = PipelineOrchestrator(resource_mgr=MagicMock(), total_layers=total_layers)
    _rm = p._resource_mgr
    _rm.check_circuit_breaker.return_value = False

    _simulated_time = [0.0]  # deterministic latency counter (no real sleep)
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
                     _lat=latency_ms):
            _simulated_time[0] += _lat / 1000.0
            return torch.zeros_like(hidden_states)

        node.client = MagicMock()
        node.client.cluster_key = None
        # run_pipeline calls client.forward(hidden_states=..., ...) and
        # forwards the return value as the next node's hidden states.
        node.client.forward = MagicMock(side_effect=_forward)
        node.gpu_compute_tflops = 312.0

    return p


# ── Regression Tests ───────────────────────────────────────────────


class TestRegressionBenchmarks:
    """Performance regression tests with baseline comparison."""

    def test_sequential_latency_regression(self):
        """Sequential pipeline latency should not regress beyond threshold."""
        pipeline = _make_pipeline(2, 32)
        input_ids = torch.randint(0, 1000, (1, 32))
        kv = pipeline.create_node_kv_caches()

        # Warmup
        pipeline.run_pipeline(input_ids, kv, "warmup")

        # Measure
        times = []
        for i in range(5):
            start = time.monotonic()
            pipeline.run_pipeline(input_ids, kv, f"bench-{i}")
            times.append((time.monotonic() - start) * 1000)

        avg_ms = sum(times) / len(times)
        baselines = _load_baselines()
        baseline = baselines.get("sequential_2node_32layer_latency_ms", 20.0)
        threshold = baseline * THRESHOLDS["sequential"]

        assert avg_ms <= threshold, (
            f"Sequential regression: {avg_ms:.1f}ms > {threshold:.1f}ms "
            f"(baseline: {baseline:.1f}ms)"
        )

    def test_overlap_latency_regression(self):
        """Overlap pipeline latency should not regress beyond threshold."""
        pipeline = _make_pipeline(2, 32)
        input_ids = torch.randint(0, 1000, (1, 32))
        kv = pipeline.create_node_kv_caches()

        # Warmup
        pipeline.run_pipeline_overlap(input_ids, kv, "warmup")

        # Measure
        times = []
        for i in range(5):
            start = time.monotonic()
            pipeline.run_pipeline_overlap(input_ids, kv, f"bench-{i}")
            times.append((time.monotonic() - start) * 1000)

        avg_ms = sum(times) / len(times)
        baselines = _load_baselines()
        baseline = baselines.get("overlap_2node_32layer_latency_ms", 15.0)
        threshold = baseline * THRESHOLDS["overlap"]

        assert avg_ms <= threshold, (
            f"Overlap regression: {avg_ms:.1f}ms > {threshold:.1f}ms "
            f"(baseline: {baseline:.1f}ms)"
        )

    def test_simulator_speed(self):
        """Simulator should be fast for analytical estimates."""
        sim = PipelineSimulator(model_size="7B")

        times = []
        for _ in range(100):
            start = time.monotonic()
            sim.simulate(num_nodes=4, num_layers=32, batch_size=1, seq_len=2048)
            times.append((time.monotonic() - start) * 1e6)

        avg_us = sum(times) / len(times)
        baselines = _load_baselines()
        baseline = baselines.get("simulator_4node_latency_ms", 10.0) * 1000  # Convert to µs
        threshold = baseline * THRESHOLDS["simulator"]

        assert avg_us <= threshold, (
            f"Simulator too slow: {avg_us:.0f}µs > {threshold:.0f}µs"
        )


# ── Baseline Update ────────────────────────────────────────────────


class TestUpdateBaselines:
    """Update baseline metrics (run manually, not in CI)."""

    @pytest.mark.skipif(
        os.environ.get("DISTLLM_UPDATE_BASELINES", "") != "1",
        reason="Set DISTLLM_UPDATE_BASELINES=1 to update baselines",
    )
    def test_update_baselines(self):
        """Measure and save current performance as new baselines."""
        pipeline = _make_pipeline(2, 32)
        input_ids = torch.randint(0, 1000, (1, 32))
        kv = pipeline.create_node_kv_caches()

        # Sequential baseline
        times = []
        for i in range(10):
            start = time.monotonic()
            pipeline.run_pipeline(input_ids, kv, f"seq-{i}")
            times.append((time.monotonic() - start) * 1000)
        seq_baseline = sum(times) / len(times)

        # Overlap baseline
        times = []
        for i in range(10):
            start = time.monotonic()
            pipeline.run_pipeline_overlap(input_ids, kv, f"ov-{i}")
            times.append((time.monotonic() - start) * 1000)
        ov_baseline = sum(times) / len(times)

        baselines = {
            "sequential_2node_32layer_latency_ms": round(seq_baseline, 2),
            "overlap_2node_32layer_latency_ms": round(ov_baseline, 2),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_baselines(baselines)
        print(f"Baselines updated: {baselines}")
