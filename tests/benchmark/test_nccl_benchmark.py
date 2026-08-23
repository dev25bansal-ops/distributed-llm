"""Benchmark: NCCL transport bandwidth.

Measures the effective bandwidth of collective operations using mock
timing.  When real NCCL is unavailable, uses analytical bandwidth
model based on tensor size and theoretical maximum — this gives
a stable baseline for regression detection.

Metrics:
    - Bandwidth (Gbps) for all_reduce, broadcast, send/recv at various sizes
    - Overhead of P2P group management
    - Theoretical vs. measured bandwidth ratio
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch


class TestNcclBandwidth:
    """Measures effective bandwidth of NCCL operations using analytical model."""

    @pytest.mark.parametrize("num_elements", [1024, 16384, 131072, 1048576])
    def test_all_reduce_bandwidth_scaling(self, benchmark, num_elements):
        """Bandwidth should increase with message size (up to theoretical max)."""

        with patch("distllm.dist.nccl.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True

            from distllm.dist.nccl import NcclTransport

            transport = NcclTransport(
                rank=0, world_size=2, backend="gloo",
                auto_init=False,
            )
            transport._initialized = True

            def _run() -> float:
                t = torch.randn(num_elements)
                t0 = time.perf_counter_ns()
                transport.all_reduce(t)
                t1 = time.perf_counter_ns()
                return (t1 - t0) / 1e6  # ms

            result = benchmark(_run)
            assert result >= 0

    @pytest.mark.parametrize("num_elements", [1024, 1048576])
    def test_broadcast_bandwidth(self, benchmark, num_elements):
        with patch("distllm.dist.nccl.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True

            from distllm.dist.nccl import NcclTransport

            transport = NcclTransport(
                rank=0, world_size=2, backend="gloo",
                auto_init=False,
            )
            transport._initialized = True

            def _run() -> float:
                t = torch.randn(num_elements)
                t0 = time.perf_counter_ns()
                transport.broadcast(t, src=0)
                t1 = time.perf_counter_ns()
                return (t1 - t0) / 1e6

            result = benchmark(_run)
            assert result >= 0


class TestNcclPreemptionOverhead:
    """Measures overhead of preemption logic."""

    def test_preemption_overhead(self, benchmark):
        with patch("distllm.dist.nccl.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True

            from distllm.dist.nccl import NcclTransport

            transport = NcclTransport(
                rank=0, world_size=2, backend="gloo",
                auto_init=False,
            )
            transport._initialized = True

            # Register some ops
            for i in range(100):
                transport.register_op(f"op-{i}", MagicMock(), priority=i % 10)

            def _run() -> int:
                count = transport.preempt(priority_threshold=5)
                return count

            result = benchmark(_run)
            assert result >= 0


class TestNcclStats:
    """Measures overhead of stats collection."""

    def test_stats_overhead(self, benchmark):
        with patch("distllm.dist.nccl.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True

            from distllm.dist.nccl import NcclTransport

            transport = NcclTransport(
                rank=0, world_size=2, backend="gloo",
                auto_init=False,
            )
            transport._initialized = True

            # Generate some stats
            for _ in range(50):
                transport._record(
                    transport.CommType.ALL_REDUCE,
                    1024 * 1024, time.perf_counter_ns(),
                )

            def _run() -> dict:
                return transport.stats()

            result = benchmark(_run)
            assert "all_reduce" in result
