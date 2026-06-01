"""Disaggregated prefill/decode scheduling.

Separates prefill (compute-bound) and decode (memory-bound) phases
onto different worker pools for optimal resource utilization.

In standard continuous batching, prefill and decode share the same
GPU, causing decode latency spikes when long prefills are scheduled.
Disaggregated P&D eliminates this interference by routing prefills
to compute-optimized nodes and decodes to memory-optimized nodes.

Usage::

    scheduler = DisaggregatedBatchScheduler(prefill_fraction=0.7)
    scheduler.set_prefill_nodes(["gpu-0", "gpu-1"])
    scheduler.set_decode_nodes(["gpu-2", "gpu-3"])

    # Schedule a new request
    phase = scheduler.classify_request(seq)
    node = scheduler.route(phase)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class RequestPhase(str, Enum):
    """Phase of a request in the P&D pipeline."""
    PREFILL = "prefill"    # Processing prompt tokens (compute-bound)
    DECODE = "decode"      # Generating tokens (memory-bound)
    TRANSFER = "transfer"  # KV cache transfer between pools


@dataclass
class DisaggregatedBudget:
    """Budget split between prefill and decode pools.

    Attributes:
        prefill_max_tokens: Max tokens per prefill batch.
        decode_max_tokens: Max tokens per decode batch.
        prefill_batch_size: Max concurrent prefills.
        decode_batch_size: Max concurrent decodes.
        prefill_fraction: Fraction of GPU time allocated to prefill (0-1).
        kv_transfer_bandwidth_gbps: Bandwidth for KV cache transfer.
    """
    prefill_max_tokens: int = 4096
    decode_max_tokens: int = 512
    prefill_batch_size: int = 8
    decode_batch_size: int = 32
    prefill_fraction: float = 0.7
    kv_transfer_bandwidth_gbps: float = 10.0


@dataclass
class NodePoolState:
    """State of a worker node pool."""
    node_ids: list[str] = field(default_factory=list)
    active_requests: dict[str, int] = field(default_factory=dict)  # node_id -> count
    total_processed: int = 0
    avg_latency_ms: float = 0.0

    @property
    def available_nodes(self) -> list[str]:
        """Return nodes with available capacity."""
        return [nid for nid in self.node_ids if self.active_requests.get(nid, 0) < 8]

    @property
    def least_loaded(self) -> str | None:
        """Return the least-loaded node."""
        if not self.node_ids:
            return None
        return min(self.node_ids, key=lambda n: self.active_requests.get(n, 0))


class DisaggregatedBatchScheduler:
    """Scheduler that separates prefill and decode onto different nodes.

    Prefill nodes handle the compute-heavy initial processing.
    Decode nodes handle the memory-bound autoregressive generation.
    KV cache is transferred between pools after prefill completes.

    Args:
        prefill_fraction: Fraction of scheduling budget for prefill (0-1).
        min_prefill_tokens: Minimum tokens to justify a prefill batch.
        enable_kv_transfer: Whether to transfer KV cache between pools.
    """

    def __init__(
        self,
        prefill_fraction: float = 0.7,
        min_prefill_tokens: int = 64,
        enable_kv_transfer: bool = True,
    ):
        self._prefill_fraction = prefill_fraction
        self._min_prefill_tokens = min_prefill_tokens
        self._enable_kv_transfer = enable_kv_transfer

        self._prefill_pool = NodePoolState()
        self._decode_pool = NodePoolState()

        self._pending_prefill: list[Any] = []
        self._pending_decode: list[Any] = []

        self._lock = threading.Lock()
        self._stats = {
            "prefill_scheduled": 0,
            "decode_scheduled": 0,
            "kv_transfers": 0,
            "fallback_to_local": 0,
        }

    def set_prefill_nodes(self, node_ids: list[str]) -> None:
        """Set nodes optimized for prefill (compute-bound, high FLOPS)."""
        with self._lock:
            self._prefill_pool.node_ids = list(node_ids)
            self._prefill_pool.active_requests = {nid: 0 for nid in node_ids}
        logger.info(f"Prefill pool: {len(node_ids)} nodes")

    def set_decode_nodes(self, node_ids: list[str]) -> None:
        """Set nodes optimized for decode (memory-bound, high bandwidth)."""
        with self._lock:
            self._decode_pool.node_ids = list(node_ids)
            self._decode_pool.active_requests = {nid: 0 for nid in node_ids}
        logger.info(f"Decode pool: {len(node_ids)} nodes")

    def classify_request(self, sequence: Any) -> RequestPhase:
        """Classify a sequence into prefill or decode phase.

        Sequences with no generated tokens need prefill.
        Sequences with generated tokens are in decode phase.
        """
        generated = getattr(sequence, 'generated_tokens', [])
        if not generated:
            return RequestPhase.PREFILL
        return RequestPhase.DECODE

    def route(self, phase: RequestPhase) -> str | None:
        """Route to the least-loaded node for the given phase.

        Returns:
            Node ID, or None if no nodes available.
        """
        with self._lock:
            if phase == RequestPhase.PREFILL:
                self._stats["prefill_scheduled"] += 1
                node = self._prefill_pool.least_loaded
                if node:
                    self._prefill_pool.active_requests[node] = (
                        self._prefill_pool.active_requests.get(node, 0) + 1
                    )
                return node
            else:
                self._stats["decode_scheduled"] += 1
                node = self._decode_pool.least_loaded
                if node:
                    self._decode_pool.active_requests[node] = (
                        self._decode_pool.active_requests.get(node, 0) + 1
                    )
                return node

    def release(self, node_id: str, phase: RequestPhase) -> None:
        """Release a slot on a node after request completion."""
        with self._lock:
            pool = self._prefill_pool if phase == RequestPhase.PREFILL else self._decode_pool
            pool.active_requests[node_id] = max(
                0, pool.active_requests.get(node_id, 1) - 1
            )
            pool.total_processed += 1

    def compute_budget(self, base_budget: Any) -> DisaggregatedBudget:
        """Compute disaggregated budget from a base budget.

        Splits the total budget between prefill and decode based on
        the configured fraction.
        """
        total_tokens = getattr(base_budget, 'max_total_tokens', 8192)
        return DisaggregatedBudget(
            prefill_max_tokens=int(total_tokens * self._prefill_fraction),
            decode_max_tokens=int(total_tokens * (1 - self._prefill_fraction)),
            prefill_batch_size=getattr(base_budget, 'max_batch_size', 8),
            decode_batch_size=getattr(base_budget, 'max_batch_size', 32) * 4,
        )

    def get_transfer_estimate(self, kv_cache_size_bytes: int) -> float:
        """Estimate KV cache transfer time in milliseconds.

        Args:
            kv_cache_size_bytes: Size of the KV cache to transfer.

        Returns:
            Estimated transfer time in milliseconds.
        """
        bandwidth_bps = self._kv_transfer_bandwidth_gbps * 1e9 / 8
        return (kv_cache_size_bytes / bandwidth_bps) * 1000

    @property
    def _kv_transfer_bandwidth_gbps(self) -> float:
        return 10.0  # Default 10 Gbps

    def stats(self) -> dict[str, Any]:
        """Return scheduler statistics."""
        with self._lock:
            return {
                **self._stats,
                "prefill_nodes": len(self._prefill_pool.node_ids),
                "decode_nodes": len(self._decode_pool.node_ids),
                "prefill_active": sum(self._prefill_pool.active_requests.values()),
                "decode_active": sum(self._decode_pool.active_requests.values()),
                "prefill_processed": self._prefill_pool.total_processed,
                "decode_processed": self._decode_pool.total_processed,
            }
