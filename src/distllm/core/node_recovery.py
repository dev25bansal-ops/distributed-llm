"""Self-healing cluster: node failure detection, layer redistribution, in-flight recovery.

When a node dies mid-generation:
1. Detect failure (via health probes)
2. Mark the node as draining (stop new requests)
3. Redistribute the dead node's layers to remaining nodes
4. Recompute in-flight sequences from the last checkpoint
5. Flag responses with ``x-distllm-recovered``
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger


# ── Checkpoint ──────────────────────────────────────────────────────────────────

@dataclass
class SequenceCheckpoint:
    """KV cache + token state for a single in-flight sequence.

    Created periodically so that a dead node's sequences can be
    recovered on surviving nodes.
    """
    request_id: str
    kv_cache: Any  # opaque KV cache snapshot (dict of layer → tensors)
    prompt_tokens: list[int]
    generated_tokens: list[int]
    node_id: str  # node that held this sequence last
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        """Rough memory footprint of this checkpoint."""
        total = 0
        for v in self.kv_cache.values() if isinstance(self.kv_cache, dict) else []:
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
        return total


# ── Recovery Plan ───────────────────────────────────────────────────────────────

@dataclass
class LayerRedistribution:
    """How a failed node's layers are reassigned to survivors."""
    surviving_node_id: str
    added_start_layer: int
    added_end_layer: int
    new_start_layer: int  # surviving node's start after absorbing layers
    new_end_layer: int


@dataclass
class NodeRecoveryPlan:
    """Full recovery plan generated when a node is declared dead."""
    failed_node_id: str
    redistributions: list[LayerRedistribution] = field(default_factory=list)
    recovered_sequences: list[str] = field(default_factory=list)
    total_sequences_lost: int = 0
    recovery_time_ms: float = 0.0
    drain_time_ms: float = 0.0


# ── Recovery Manager ────────────────────────────────────────────────────────────

class NodeRecoveryManager:
    """Orchestrates self-healing when a node fails.

    Usage (set callbacks, then call from health probe loop)::

        mgr = NodeRecoveryManager()
        mgr.set_drain_callback(lambda nid: ...)
        mgr.set_recover_sequences_callback(lambda nid, seqs: ...)
        mgr.set_redistribute_layers_callback(lambda nid, plan: ...)
        mgr.on_node_failure("node-3")
    """

    def __init__(self, node_id: str = "coordinator"):
        self._node_id = node_id
        self._draining: set[str] = set()
        self._dead_nodes: set[str] = set()
        self._checkpoints: dict[str, SequenceCheckpoint] = {}
        self._seq_to_node: dict[str, str] = {}  # request_id → node_id
        self._recovered_requests: set[str] = set()
        self._lock = threading.Lock()

        # Callbacks (set by coordinator)
        self._on_drain: Callable[[str], None] | None = None
        self._on_redistribute: Callable[[str, NodeRecoveryPlan], None] | None = None
        self._on_recover: Callable[[str, list[str]], list[Any]] | None = None
        self._on_mark_dead: Callable[[str], None] | None = None

        self._metrics = {
            "recoveries": 0,
            "failed_nodes": 0,
            "sequences_recovered": 0,
            "sequences_lost": 0,
            "checkpoint_count": 0,
            "total_recovery_time_ms": 0,
        }

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_drain_callback(self, cb: Callable[[str], None]) -> None:
        self._on_drain = cb

    def set_redistribute_layers_callback(
        self, cb: Callable[[str, NodeRecoveryPlan], None]
    ) -> None:
        self._on_redistribute = cb

    def set_recover_sequences_callback(
        self, cb: Callable[[str, list[str]], list[Any]]
    ) -> None:
        """Callback ``(failed_node_id, [request_ids]) → [kv_cache_snapshots]``."""
        self._on_recover = cb

    def set_mark_dead_callback(self, cb: Callable[[str], None]) -> None:
        """Callback to remove node from pipeline."""
        self._on_mark_dead = cb

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, request_id: str, kv_cache: Any,
        prompt_tokens: list[int], generated_tokens: list[int],
        node_id: str,
    ) -> None:
        """Periodically checkpoint an in-flight sequence.

        Should be called every ``N`` decode steps by the coordinator.
        Overwrites the previous checkpoint for the same *request_id*.
        """
        ckpt = SequenceCheckpoint(
            request_id=request_id,
            kv_cache=copy.deepcopy(kv_cache) if hasattr(kv_cache, '__copy__') else kv_cache,
            prompt_tokens=list(prompt_tokens),
            generated_tokens=list(generated_tokens),
            node_id=node_id,
        )
        with self._lock:
            old = self._checkpoints.get(request_id)
            self._checkpoints[request_id] = ckpt
            self._seq_to_node[request_id] = node_id
            if old is None:
                self._metrics["checkpoint_count"] += 1

    def get_checkpoint(self, request_id: str) -> SequenceCheckpoint | None:
        with self._lock:
            return self._checkpoints.get(request_id)

    def drop_checkpoint(self, request_id: str) -> None:
        with self._lock:
            self._checkpoints.pop(request_id, None)
            self._seq_to_node.pop(request_id, None)

    def get_checkpoints_for_node(self, node_id: str) -> dict[str, SequenceCheckpoint]:
        with self._lock:
            return {
                rid: ckpt for rid, ckpt in self._checkpoints.items()
                if ckpt.node_id == node_id
            }

    # ------------------------------------------------------------------
    # Drain & node state
    # ------------------------------------------------------------------

    def is_draining(self, node_id: str) -> bool:
        return node_id in self._draining

    def is_dead(self, node_id: str) -> bool:
        return node_id in self._dead_nodes

    @property
    def draining_nodes(self) -> list[str]:
        return list(self._draining)

    @property
    def dead_nodes(self) -> list[str]:
        return list(self._dead_nodes)

    def mark_alive(self, node_id: str) -> None:
        """Re-mark a previously dead/draining node as alive (after rejoin)."""
        with self._lock:
            self._draining.discard(node_id)
            self._dead_nodes.discard(node_id)

    # ------------------------------------------------------------------
    # Recovery protocol
    # ------------------------------------------------------------------

    def on_node_failure(self, failed_node_id: str) -> NodeRecoveryPlan:
        """Full recovery protocol when a node is declared dead.

        1. Mark node as draining → stops new requests
        2. Compute layer redistribution among survivors
        3. Recover in-flight sequences from checkpoints
        4. Reinitialize pipeline without the dead node
        """
        logger.warning(f"Starting recovery for failed node: {failed_node_id}")
        start_time = time.monotonic()
        plan = NodeRecoveryPlan(failed_node_id=failed_node_id)

        # ── Step 1: Drain ──
        with self._lock:
            self._draining.add(failed_node_id)
            self._dead_nodes.add(failed_node_id)
        if self._on_drain:
            self._on_drain(failed_node_id)
        plan.drain_time_ms = (time.monotonic() - start_time) * 1000

        # ── Step 2: Recover in-flight sequences ──
        t0 = time.monotonic()
        seqs = self.get_checkpoints_for_node(failed_node_id)
        recovered_ids = list(seqs.keys())
        if self._on_recover and recovered_ids:
            snapshots = self._on_recover(failed_node_id, recovered_ids)
            with self._lock:
                for rid in recovered_ids:
                    self._recovered_requests.add(rid)
                    self._checkpoints.pop(rid, None)
                    self._seq_to_node.pop(rid, None)
            plan.recovered_sequences = recovered_ids
        else:
            plan.total_sequences_lost = len(recovered_ids)

        # Clear checkpoints for the dead node
        with self._lock:
            for rid in recovered_ids:
                self._checkpoints.pop(rid, None)

        plan.recovery_time_ms = (time.monotonic() - t0) * 1000

        # ── Step 3: Redistribute layers ──
        if self._on_redistribute:
            self._on_redistribute(failed_node_id, plan)

        total_time = (time.monotonic() - start_time) * 1000
        plan.recovery_time_ms = total_time

        with self._lock:
            self._metrics["recoveries"] += 1
            self._metrics["failed_nodes"] += 1
            self._metrics["sequences_recovered"] += len(recovered_ids)
            self._metrics["sequences_lost"] += plan.total_sequences_lost
            self._metrics["total_recovery_time_ms"] += total_time

        logger.info(
            f"Recovery complete for {failed_node_id}: "
            f"{len(recovered_ids)} sequences recovered, "
            f"{len(plan.redistributions)} layer redistributions, "
            f"{total_time:.0f}ms total"
        )
        return plan

    def is_recovered_request(self, request_id: str) -> bool:
        """Check if a request was recovered from a node failure.

        Used to inject the ``x-distllm-recovered`` header.
        """
        return request_id in self._recovered_requests

    def consume_recovered_flag(self, request_id: str) -> bool:
        """Atomically check and clear the recovered flag.

        Returns True once, then clears the flag so subsequent
        responses don't carry the header.
        """
        with self._lock:
            if request_id in self._recovered_requests:
                self._recovered_requests.discard(request_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        with self._lock:
            m = dict(self._metrics)
            m["draining_nodes"] = len(self._draining)
            m["dead_nodes"] = len(self._dead_nodes)
            m["active_checkpoints"] = len(self._checkpoints)
            return m
