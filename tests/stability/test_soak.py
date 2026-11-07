"""Long-running soak tests for PipelineOrchestrator.

Runs continuous pipeline operations for extended periods to detect:
- Memory leaks (node_checkpoints, cached_ones growth)
- Thread pool exhaustion
- KV cache tracking growth
- Event loop leaks

Run manually:
    pytest tests/stability/test_soak.py -v --timeout=3600
    DISTLLM_SOAK_DURATION=60 pytest tests/stability/test_soak.py -v
"""

from __future__ import annotations

import gc
import tracemalloc
import os
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, "src")

from distllm.dist.pipeline import PipelineOrchestrator


# ── Configuration ──────────────────────────────────────────────────

SOAK_DURATION_S = int(os.environ.get("DISTLLM_SOAK_DURATION", "30"))
SOAK_CHECK_INTERVAL = 5  # Check memory every N seconds


def _make_pipeline(num_nodes=2, total_layers=16):
    """Build an orchestrator via the public register_node API.

    register_node (orchestrator.py:256) maintains ``_node_order`` and the
    real ``PipelineNode.is_healthy`` flag. Assigning bare mocks into
    ``p.nodes[...]`` never populated ``_node_order``, so run_pipeline raised
    "No healthy nodes in pipeline".
    """
    p = PipelineOrchestrator(resource_mgr=MagicMock(), total_layers=total_layers)
    p._resource_mgr.check_circuit_breaker.return_value = False
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
        node.client = MagicMock()
        node.client.cluster_key = None
        node.client.forward = MagicMock(
            side_effect=lambda hidden_states=None, kv_cache=None,
            request_id=None, **kw: torch.zeros_like(hidden_states)
        )
        node.gpu_compute_tflops = 312.0

    return p


# ── Soak Tests ─────────────────────────────────────────────────────


@pytest.mark.slow
class TestSoak:
    """Long-running stability tests."""

    def test_memory_stability(self):
        """Memory usage should not grow unbounded over time."""
        pipeline = _make_pipeline(2, 16)

        # Initial measurements
        gc.collect()
        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()
        initial_runs = pipeline.stats()["pipeline_runs"]

        measurements = []
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < SOAK_DURATION_S:
            # Generate random request
            batch_size = 1
            seq_len = 4 + (request_count % 16)
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            kv = pipeline.create_node_kv_caches()

            try:
                pipeline.run_pipeline(
                    input_ids, kv, f"soak-{request_count}",
                )
                request_count += 1
            except Exception:
                pass

            # Periodic measurement
            if request_count % SOAK_CHECK_INTERVAL == 0:
                gc.collect()
                measurements.append({
                    "time_s": time.time() - start_time,
                    "requests": request_count,
                    "pipeline_runs": pipeline.stats()["pipeline_runs"],
                    "errors": pipeline.stats()["errors"],
                })

        # Verify no unbounded growth
        assert request_count > 0, "soak loop never issued a request"
        runs_delta = pipeline.stats()["pipeline_runs"] - initial_runs
        assert runs_delta == request_count, (
            f"stats().pipeline_runs advanced by {runs_delta}, expected {request_count}"
        )

        if len(measurements) >= 2:
            final_snapshot = tracemalloc.take_snapshot()
            growth_bytes = sum(
                stat.size_diff
                for stat in final_snapshot.compare_to(baseline_snapshot, "filename")
            )
            tracemalloc.stop()
            growth_mb = growth_bytes / (1024 * 1024)
            # Steady-state soak: allocations must not balloon. 32 MiB of slack
            # absorbs allocator fragmentation across a short run.
            assert growth_mb < 32, (
                f"Memory grew {growth_mb:.1f} MiB over {request_count} requests "
                "(possible leak in run_pipeline)"
            )
        else:
            tracemalloc.stop()

        print(f"\nSoak test: {request_count} requests in {SOAK_DURATION_S}s")
        print(f"  stats().pipeline_runs: {initial_runs} -> {pipeline.stats()['pipeline_runs']}")

    def test_thread_pool_stability(self):
        """Thread pools should not leak over time."""
        pipeline = _make_pipeline(2, 16)

        initial_thread_count = threading.active_count()
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < SOAK_DURATION_S:
            input_ids = torch.randint(0, 1000, (1, 4))
            kv = pipeline.create_node_kv_caches()

            try:
                pipeline.run_pipeline(
                    input_ids, kv, f"thread-{request_count}",
                )
                request_count += 1
            except Exception:
                pass

        # Thread count should not grow significantly
        final_thread_count = threading.active_count()
        thread_growth = final_thread_count - initial_thread_count

        assert thread_growth < 10, (
            f"Thread leak: {initial_thread_count} → {final_thread_count} "
            f"(+{thread_growth})"
        )

        print(f"\nThread stability: {initial_thread_count} → {final_thread_count} threads")

    def test_event_loop_stability(self):
        """Event loop should not leak over time."""
        pipeline = _make_pipeline(2, 16)

        # Initialize event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assert loop is not None

        start_time = time.time()
        request_count = 0

        while time.time() - start_time < SOAK_DURATION_S:
            input_ids = torch.randint(0, 1000, (1, 4))
            kv = pipeline.create_node_kv_caches()

            try:
                pipeline.run_pipeline(
                    input_ids, kv, f"loop-{request_count}",
                )
                request_count += 1
            except Exception:
                pass

        # Event loop should still be the same instance
        loop2 = asyncio.get_event_loop()
        assert loop is loop2, "Event loop was recreated (leak)"

        print(f"\nEvent loop stability: same loop maintained for {request_count} requests")
