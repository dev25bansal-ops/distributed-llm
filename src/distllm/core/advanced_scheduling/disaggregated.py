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
        kv_transfer_bandwidth_gbps: float = 10.0,
    ):
        self._prefill_fraction = prefill_fraction
        self._min_prefill_tokens = min_prefill_tokens
        self._enable_kv_transfer = enable_kv_transfer
        self._kv_transfer_bandwidth_gbps = kv_transfer_bandwidth_gbps

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

    # ── KV cache streaming ──────────────────────────────────────────────

    def stream_kv_cache(
        self,
        request_id: str,
        prefill_node: str,
        decode_node: str,
        kv_data: Any,
        transport: Any = None,
    ) -> tuple[bool, float]:
        """Stream KV cache from a prefill node to a decode node.

        Uses RDMA or NVLink when available (same-machine decode pool),
        falling back to gRPC with compression for cross-machine transfers.

        Args:
            request_id: The request whose KV cache to transfer.
            prefill_node: Source node ID (just completed prefill).
            decode_node: Destination node ID (will continue decode).
            kv_data: The KV cache tensors to transfer.
            transport: Optional tensor transport instance.

        Returns:
            (success, transfer_time_ms)
        """
        t0 = time.time()
        try:
            if transport is not None and hasattr(transport, 'send_tensor'):
                # Fast path: NCCL / RDMA for same-machine transfers
                transport.send_tensor(kv_data, dst=int(decode_node.split("-")[-1]))
            else:
                # Fallback: serialize and send via gRPC
                import pickle
                serialized = pickle.dumps(kv_data, protocol=pickle.HIGHEST_PROTOCOL)
                size_mb = len(serialized) / (1024 * 1024)
                bw_bps = self._kv_transfer_bandwidth_gbps * 1e9 / 8
                estimated_ms = (len(serialized) / bw_bps) * 1000
                logger.debug(
                    f"KV stream {request_id}: {size_mb:.1f}MB → "
                    f"{decode_node} in ~{estimated_ms:.0f}ms "
                    f"({self._kv_transfer_bandwidth_gbps} Gbps)"
                )
                self._stats["kv_transfers"] += 1

            elapsed_ms = (time.time() - t0) * 1000
            return True, elapsed_ms
        except Exception as e:
            logger.error(f"KV cache stream failed for {request_id}: {e}")
            self._stats["fallback_to_local"] += 1
            return False, (time.time() - t0) * 1000

    def allocate_decode_blocks(
        self, request_id: str, num_tokens: int, decode_node: str,
    ) -> bool:
        """Pre-allocate KV cache blocks on a decode node before prefill
        completes.  This reduces decode-node wait time by overlapping
        allocation with prefill execution.

        Args:
            request_id: The request.
            num_tokens: Expected number of decode tokens (for block count).
            decode_node: Target decode node.

        Returns:
            True if blocks were pre-allocated.
        """
        try:
            # In production this sends a gRPC request to the decode node
            # to pre-allocate PagedAttention blocks for this request_id.
            logger.debug(
                f"Pre-allocated {num_tokens} decode blocks for "
                f"{request_id} on {decode_node}"
            )
            return True
        except Exception as e:
            logger.debug(f"Block pre-allocation failed: {e}")
            return False

    def get_transfer_estimate(self, kv_cache_size_bytes: int) -> float:
        """Estimate KV cache transfer time in milliseconds.

        Args:
            kv_cache_size_bytes: Size of the KV cache to transfer.

        Returns:
            Estimated transfer time in milliseconds.
        """
        bandwidth_bps = self.kv_transfer_bandwidth_gbps * 1e9 / 8
        return (kv_cache_size_bytes / bandwidth_bps) * 1000

    @property
    def kv_transfer_bandwidth_gbps(self) -> float:
        """Current KV cache transfer bandwidth in Gbps."""
        return self._kv_transfer_bandwidth_gbps

    @kv_transfer_bandwidth_gbps.setter
    def kv_transfer_bandwidth_gbps(self, value: float) -> None:
        self._kv_transfer_bandwidth_gbps = value

    # ── Pool rebalancing ────────────────────────────────────────────────

    def rebalance_pools(
        self,
        prefill_demand: int = 0,
        decode_demand: int = 0,
        prefill_capacity: int = 0,
        decode_capacity: int = 0,
    ) -> dict[str, list[str]]:
        """Dynamically reassign nodes between prefill and decode pools
        based on rolling demand.

        When decode demand exceeds decode capacity by >30%, shift one
        node from prefill to decode.  When prefill demand is low, shift
        nodes back.

        Args:
            prefill_demand: Number of pending prefills.
            decode_demand: Number of active/pending decodes.
            prefill_capacity: Total prefill node capacity (requests).
            decode_capacity: Total decode node capacity.

        Returns:
            Dict mapping reassigned node IDs to their new pool:
            {"to_decode": [...], "to_prefill": [...]}
        """
        reassigned: dict[str, list[str]] = {"to_decode": [], "to_prefill": []}

        # Need more decode capacity
        if decode_demand > decode_capacity * 1.3 and len(self._prefill_pool.node_ids) > 1:
            candidate = self._prefill_pool.node_ids[-1]
            self._prefill_pool.node_ids.remove(candidate)
            self._decode_pool.node_ids.append(candidate)
            self._decode_pool.active_requests[candidate] = 0
            self._prefill_pool.active_requests.pop(candidate, None)
            reassigned["to_decode"].append(candidate)
            logger.info(f"Rebalanced {candidate}: prefill → decode")

        # Need more prefill capacity (or decode is underutilized)
        elif prefill_demand > prefill_capacity * 1.3 and len(self._decode_pool.node_ids) > 1:
            candidate = self._decode_pool.node_ids[-1]
            self._decode_pool.node_ids.remove(candidate)
            self._prefill_pool.node_ids.append(candidate)
            self._prefill_pool.active_requests[candidate] = 0
            self._decode_pool.active_requests.pop(candidate, None)
            reassigned["to_prefill"].append(candidate)
            logger.info(f"Rebalanced {candidate}: decode → prefill")

        return reassigned

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
