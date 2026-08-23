"""Benchmark: recovery time under simulated load.

Measures how long the NodeRecoveryManager takes to complete a recovery
cycle (detect → drain → recover → redistribute) with varying numbers
of checkpoints and surviving nodes.  Uses mock callbacks so no real
cluster is required.

Metrics:
    - Recovery time (ms) with 0/10/100/500 checkpoints
    - Recovery time with 2/4/8 survivors (parallel redistribution scaling)
    - P50/P99 recovery time across multiple runs
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.dist.recovery import (
    NodeRecoveryManager,
    compute_redistributions,
    compute_redistributions_capacity_aware,
)


class TestRecoveryBenchmark:
    """Benchmarks the recovery cycle time."""

    @pytest.mark.parametrize("num_checkpoints", [0, 10, 100])
    def test_recovery_time_by_checkpoints(self, benchmark, num_checkpoints):
        """Recovery time scales with checkpoint count."""

        def _run() -> float:
            mgr = NodeRecoveryManager(node_id="bench-coord")
            mgr.set_drain_callback(MagicMock())
            mgr.set_recover_sequences_callback(MagicMock(return_value=[]))
            mgr.set_mark_dead_callback(MagicMock())
            mgr.set_redistribute_layers_callback(MagicMock())

            # Add checkpoints
            for i in range(num_checkpoints):
                mgr.save_checkpoint(
                    f"req-{i}", torch.zeros(100), [1, 2, 3], [4], "failed-node",
                )

            t0 = time.perf_counter_ns()
            plan = mgr.on_node_failure("failed-node")
            elapsed_ns = time.perf_counter_ns() - t0
            return elapsed_ns / 1e6  # ms

        result = benchmark(_run)
        assert result > 0

    @pytest.mark.parametrize("num_survivors", [2, 4, 8])
    def test_redistribution_scaling(self, benchmark, num_survivors):
        """compute_redistributions performance scales linearly with survivors."""

        survivors = {f"n{i}": (i * 8, (i + 1) * 8 - 1) for i in range(num_survivors)}

        def _run() -> int:
            result = compute_redistributions(0, 7, survivors)
            return len(result)

        result = benchmark(_run)
        assert result == num_survivors


class TestCapacityAwareBenchmark:
    """Benchmarks the capacity-aware redistribution algorithm."""

    @pytest.mark.parametrize("num_survivors", [4, 16, 64])
    def test_capacity_aware_scaling(self, benchmark, num_survivors):
        survivors = {f"n{i}": (i * 4, (i + 1) * 4 - 1) for i in range(num_survivors)}
        memory = {f"n{i}": 20.0 + (i % 5) * 10.0 for i in range(num_survivors)}

        def _run() -> int:
            result = compute_redistributions_capacity_aware(
                0, 15, survivors, memory,
            )
            return len(result)

        result = benchmark(_run)
        assert result == num_survivors


class TestDryRunBenchmark:
    """Benchmarks the dry-run recovery path (no callbacks)."""

    @pytest.mark.parametrize("num_checkpoints", [50, 500])
    def test_dry_run_time(self, benchmark, num_checkpoints):
        def _run() -> float:
            mgr = NodeRecoveryManager(node_id="bench-coord")
            for i in range(num_checkpoints):
                mgr.save_checkpoint(
                    f"req-{i}", torch.zeros(100), [1, 2], [3], "target-node",
                )
            t0 = time.perf_counter_ns()
            plan = mgr.dry_run_recovery("target-node")
            return (time.perf_counter_ns() - t0) / 1e6

        result = benchmark(_run)
        assert result > 0
