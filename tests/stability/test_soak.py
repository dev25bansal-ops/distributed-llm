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
    p = PipelineOrchestrator(total_layers=total_layers)
    p.resource_mgr = MagicMock()
    p.resource_mgr.check_circuit_breaker.return_value = False
    p.resource_mgr.record_success = MagicMock()
    p.resource_mgr.record_failure = MagicMock()

    for i in range(num_nodes):
        node = MagicMock()
        node.node_id = f"n{i}"
        node.host = "localhost"
        node.port = 50051 + i
        node.start_layer = i * (total_layers // num_nodes)
        node.end_layer = (i + 1) * (total_layers // num_nodes) - 1
        node.healthy = True
        node.max_retries = 1
        node.retry_delay = 0.001
        node.client = MagicMock()
        node.client.cluster_key = None
        node.client.stub.ForwardPass.return_value = MagicMock(
            success=True,
            error_code=0,
            error_message="",
            processing_time_ms=1.0,
            is_logits=True,
            output=MagicMock(
                shape=[1, 1, 32000],
                dtype="torch.float32",
                raw_data=b"\x00" * 128000,
                scale=None,
            ),
            kv_cache=MagicMock(layers=[]),
        )
        p.nodes[f"n{i}"] = node
        p.node_order.append(f"n{i}")

    p._rebuild_fallback_map()
    p._rebuild_stages()
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
        initial_checkpoints = len(pipeline._node_checkpoints)
        initial_cached_ones = len(pipeline._cached_ones)
        initial_kv_tracking = len(pipeline._node_kv_sent_lens)

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
                pipeline._run_sequential_pipeline(
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
                    "checkpoints": len(pipeline._node_checkpoints),
                    "cached_ones": len(pipeline._cached_ones),
                    "kv_tracking": len(pipeline._node_kv_sent_lens),
                    "in_flight": pipeline._in_flight_count,
                })

            # Cleanup completed requests
            if request_count % 100 == 0:
                pipeline.cleanup_request_kv_tracking(f"soak-{request_count - 50}")

        # Verify no unbounded growth
        if len(measurements) >= 2:
            first = measurements[0]
            last = measurements[-1]

            # Checkpoints should not grow unbounded (cleanup should work)
            checkpoint_growth = last["checkpoints"] - initial_checkpoints
            assert checkpoint_growth < 100, (
                f"Checkpoint leak: grew by {checkpoint_growth} "
                f"({initial_checkpoints} → {last['checkpoints']})"
            )

            # Cached tensors should be bounded
            ones_growth = last["cached_ones"] - initial_cached_ones
            assert ones_growth < 50, (
                f"Cached ones leak: grew by {ones_growth} "
                f"({initial_cached_ones} → {last['cached_ones']})"
            )

            # KV tracking should be cleaned up
            kv_growth = last["kv_tracking"] - initial_kv_tracking
            assert kv_growth < 100, (
                f"KV tracking leak: grew by {kv_growth} "
                f"({initial_kv_tracking} → {last['kv_tracking']})"
            )

            # In-flight should always be 0 between requests
            assert last["in_flight"] == 0, (
                f"In-flight semaphore leak: {last['in_flight']}"
            )

        print(f"\nSoak test: {request_count} requests in {SOAK_DURATION_S}s")
        print(f"  Checkpoints: {initial_checkpoints} → {len(pipeline._node_checkpoints)}")
        print(f"  Cached ones: {initial_cached_ones} → {len(pipeline._cached_ones)}")
        print(f"  KV tracking: {initial_kv_tracking} → {len(pipeline._node_kv_sent_lens)}")

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
                pipeline._run_sequential_pipeline(
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
        loop = pipeline._get_or_create_event_loop()
        assert loop is not None

        start_time = time.time()
        request_count = 0

        while time.time() - start_time < SOAK_DURATION_S:
            input_ids = torch.randint(0, 1000, (1, 4))
            kv = pipeline.create_node_kv_caches()

            try:
                pipeline._run_sequential_pipeline(
                    input_ids, kv, f"loop-{request_count}",
                )
                request_count += 1
            except Exception:
                pass

        # Event loop should still be the same instance
        loop2 = pipeline._get_or_create_event_loop()
        assert loop is loop2, "Event loop was recreated (leak)"

        print(f"\nEvent loop stability: same loop maintained for {request_count} requests")
