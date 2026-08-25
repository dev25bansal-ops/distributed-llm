"""Self-healing cluster: node failure detection, layer redistribution, in-flight recovery.

When a node dies mid-generation:
1. Detect failure (via health probes)
2. Mark the node as draining (stop new requests)
3. Redistribute the dead node's layers to remaining nodes
4. Recompute in-flight sequences from the last checkpoint
5. Flag responses with ``x-distllm-recovered``
"""

from __future__ import annotations
import asyncio
import enum
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

@dataclass
class SequenceCheckpoint:
    """KV cache + token state for a single in-flight sequence.

    Created periodically so that a dead node's sequences can be
    recovered on surviving nodes.
    """

    request_id: str
    kv_cache: Any
    prompt_tokens: list[int]
    generated_tokens: list[int]
    node_id: str
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        """Recursively sum tensor bytes in :attr:`kv_cache`.

        Handles nested dict/list/tuple combinations at any depth,
        including ``torch.nn.Parameter`` wrappers.
        """

        return _tensor_size_bytes(self.kv_cache)

def _tensor_size_bytes(obj: Any) -> int:
    """Recursively accumulate total byte size of all tensors in *obj*."""

    if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_size_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_size_bytes(item) for item in obj)
    return 0

@dataclass
class LayerRedistribution:
    """How a failed node's layers are reassigned to survivors.

    ``new_start_layer``/``new_end_layer`` describe the survivor's final
    (contiguous, non-overlapping) range after absorbing
    ``added_start_layer``..``added_end_layer``.

    Weight-transfer honesty: this planner is *metadata-only*.  It cannot
    move model weights over the network; the operator/coordinator must
    pull the added layers onto each survivor before traffic is routed to
    them (e.g. via ``TransferWeightsStream``).  Ownership-changing
    entries carry ``requires_weight_load=True`` so downstream consumers
    never mistake relabeled nodes for loaded ones.
    """

    surviving_node_id: str
    added_start_layer: int
    added_end_layer: int
    new_start_layer: int
    new_end_layer: int
    requires_weight_load: bool = True

@dataclass
class NodeRecoveryPlan:
    """Full recovery plan generated when a node is declared dead."""

    failed_node_id: str
    redistributions: list[LayerRedistribution] = field(default_factory=list)
    recovered_sequences: list[str] = field(default_factory=list)
    total_sequences_lost: int = 0
    recovery_time_ms: float = 0.0
    drain_time_ms: float = 0.0
    # Honest limitation marker: computing a plan does NOT transfer model
    # weights.  Consumers must check per-redistribution
    # ``LayerRedistribution.requires_weight_load`` and drive an actual
    # weight pull before routing traffic to reassigned nodes.
    weights_transferred: bool = False

# ── Partition helpers ──────────────────────────────────────────────────


def _sort_key(nid: str) -> str:
    """Deterministic ordering for node ids (numeric-aware when possible)."""

    try:
        return f"{int(nid):020d}"
    except ValueError:
        return nid


def _largest_remainder_split(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Apportion ``total`` units across keys proportionally to weight.

    Largest-remainder method: leftovers go to the largest fractional
    remainders (deterministic tie-break on numeric-aware key order).
    Returns counts summing exactly to ``total`` (keys may receive 0).
    """

    if not weights:
        return {}
    total_weight = sum(max(0.0, w) for w in weights.values())
    if total_weight <= 0:
        # Equal split.
        base = total // len(weights)
        extra = total % len(weights)
        return {
            nid: base + (1 if i < extra else 0)
            for i, nid in enumerate(sorted(weights))
        }

    exact = {nid: total * max(0.0, w) / total_weight for nid, w in weights.items()}
    floors = {nid: int(v) for nid, v in exact.items()}
    deficit = total - sum(floors.values())
    # Hand out the remaining units to the biggest fractional parts.
    order = sorted(
        weights,
        key=lambda nid: (exact[nid] - floors[nid], _sort_key(nid)),
        reverse=True,
    )
    for i in range(deficit):
        floors[order[i % len(order)]] += 1
    return floors


def _is_clean_tiling(
    ordered: list[tuple[str, tuple[int, int]]],
    failed_start_layer: int,
    failed_end_layer: int,
) -> bool:
    """True iff survivor ranges are pairwise disjoint AND none of them
    intersects the orphaned range — the precondition for neighbor
    absorption to preserve disjointness."""

    for i, (_nid, (s, e)) in enumerate(ordered):
        # Overlaps the orphan?
        if not (e < failed_start_layer or s > failed_end_layer):
            return False
        # Disjoint from the next range?
        if i + 1 < len(ordered):
            ns, _ne = ordered[i + 1][1]
            if e >= ns:
                return False
    return True


def _retiling_redistributions(
    survivors: list[tuple[str, tuple[int, int]]],
    span_start: int,
    span_end: int,
    weights: dict[str, float] | None = None,
) -> list[LayerRedistribution]:
    """Recompute a fresh clean partition of [span_start, span_end].

    Used when neighbor absorption cannot produce a valid result
    (overlapping survivors, survivors inside the orphaned range, or no
    adjacent neighbor on either side).  Layers are apportioned across
    survivors by ``weights`` (equal if None), laid out contiguously in
    survivor order, and emitted as LayerRedistribution entries whose
    ``new_*`` range is the survivor's full final range.  Survivors that
    receive zero layers are skipped entirely.
    """

    total = span_end - span_start + 1
    w = (
        {nid: float(e - s + 1) for nid, (s, e) in survivors}
        if weights is None
        else {nid: max(1.0, float(weights.get(nid, 1.0))) for nid, _ in survivors}
    )
    counts = _largest_remainder_split(total, w)

    rds: list[LayerRedistribution] = []
    cursor = span_start
    for nid, (_cur_s, _cur_e) in survivors:
        c = counts.get(nid, 0)
        if c <= 0:
            continue
        new_s, new_e = cursor, cursor + c - 1
        cursor = new_e + 1
        rds.append(LayerRedistribution(
            surviving_node_id=nid,
            added_start_layer=new_s,
            added_end_layer=new_e,
            new_start_layer=new_s,
            new_end_layer=new_e,
            requires_weight_load=True,
        ))
    return rds

def compute_redistributions(
    failed_start_layer: int,
    failed_end_layer: int,
    surviving_nodes: dict[str, tuple[int, int]],
) -> list[LayerRedistribution]:
    """Compute how to redistribute a failed node's layers across survivors.

    The orphaned range ``[failed_start_layer, failed_end_layer]`` is split
    between its *adjacent* neighbors: the closest survivor ending below
    the orphan absorbs the left portion (extending rightward), the
    closest survivor starting above absorbs the rest (extending
    leftward).  Survivors not adjacent to the orphan are untouched, so
    every output range stays contiguous and no layer index is claimed by
    two nodes.

    If the input is not a clean layout (survivor ranges overlap each
    other or intersect the orphaned range), neighbor absorption cannot
    produce a valid partition; the function then falls back to a full
    re-partition of the whole span across survivors.

    Note — metadata only: the returned plan relabels layer ownership but
    does NOT transfer model weights.  Entries that change ownership have
    ``requires_weight_load=True``; the caller must pull the added
    layers' weights onto each survivor (e.g. via
    ``TransferWeightsStream``) before routing traffic there.  See also
    :attr:`NodeRecoveryPlan.weights_transferred`.

    Args:
        failed_start_layer: First layer of the failed node.
        failed_end_layer: Last layer (inclusive) of the failed node.
        surviving_nodes: ``{node_id: (start_layer, end_layer)}`` for alive nodes.

    Returns:
        List of :class:`LayerRedistribution` describing the new assignments.
    """

    if not surviving_nodes:
        return []

    dead_count = failed_end_layer - failed_start_layer + 1
    if dead_count <= 0:
        return []

    ordered = sorted(surviving_nodes.items(), key=lambda x: (x[1][0], x[1][1]))

    if not _is_clean_tiling(ordered, failed_start_layer, failed_end_layer):
        span_start = min(ordered[0][1][0], failed_start_layer)
        span_end = max(ordered[-1][1][1], failed_end_layer)
        logger.warning(
            "Survivor topology is not a clean tiling around failed layers "
            f"[{failed_start_layer}, {failed_end_layer}] — falling back to "
            "full re-partition"
        )
        return _retiling_redistributions(ordered, span_start, span_end)

    # Left absorbent neighbor: survivor ending just below the orphan.
    left = None
    for nid, (_s, e) in ordered:
        if e < failed_start_layer:
            left = nid
    # Right absorbent neighbor: survivor starting just above the orphan.
    right = None
    for nid, (s, _e) in reversed(ordered):
        if s > failed_end_layer:
            right = nid

    if left is None and right is None:
        # No adjacent survivor on either side — re-tile the whole span so
        # every orphan layer still lands on exactly one live node.
        span_start = min(ordered[0][1][0], failed_start_layer)
        span_end = max(ordered[-1][1][1], failed_end_layer)
        return _retiling_redistributions(ordered, span_start, span_end)

    # Split the orphan between the two absorbing neighbors (either may be
    # absent).  Give the larger share to whichever neighbor holds more
    # layers already (cheap load balance); tie goes to the left.
    left_size = 0
    right_size = 0
    if left is not None:
        ls, le = surviving_nodes[left]
        left_size = le - ls + 1
    if right is not None:
        rs, re_ = surviving_nodes[right]
        right_size = re_ - rs + 1

    if right is None:
        left_share = dead_count
    elif left is None:
        left_share = 0
    else:
        left_share = dead_count * left_size // (left_size + right_size)

    redistributions: list[LayerRedistribution] = []
    offset = 0

    if left is not None:
        count = left_share
        cur_s, cur_e = surviving_nodes[left]
        added_start = failed_start_layer + offset
        added_end = added_start + count - 1
        offset += count
        redistributions.append(LayerRedistribution(
            surviving_node_id=left,
            added_start_layer=added_start,
            added_end_layer=added_end,
            new_start_layer=cur_s,
            new_end_layer=max(cur_e, added_end),
            requires_weight_load=count > 0,
        ))

    if right is not None:
        count = dead_count - offset
        cur_s, cur_e = surviving_nodes[right]
        added_start = failed_start_layer + offset
        added_end = added_start + count - 1
        redistributions.append(LayerRedistribution(
            surviving_node_id=right,
            added_start_layer=added_start,
            added_end_layer=added_end,
            new_start_layer=min(cur_s, added_start),
            new_end_layer=cur_e,
            requires_weight_load=count > 0,
        ))

    return redistributions

def compute_redistributions_capacity_aware(
    failed_start_layer: int,
    failed_end_layer: int,
    surviving_nodes: dict[str, tuple[int, int]],
    survivor_memory_gb: dict[str, float] | None = None,
    min_memory_per_layer_gb: float = 0.5,
) -> list[LayerRedistribution]:
    """Compute layer redistributions respecting survivor GPU capacity.

    Like :func:`compute_redistributions`, but splits the failed node's
    orphaned layers among *adjacent eligible* survivors proportionally to
    free GPU memory (largest-remainder apportionment, so counts always
    sum exactly to the orphan size).  Adjacency is computed over ALL
    survivors — not just eligible ones — because an absorber farther out
    than the nearest neighbor would necessarily swallow the intermediate
    survivor's layers.  Survivors without enough free memory are
    excluded; if neither adjacent neighbor qualifies, falls back to even
    distribution.

    Output guarantees match :func:`compute_redistributions`: contiguous,
    non-overlapping ranges covering every orphaned layer.

    Note — metadata only: ownership-changing entries carry
    ``requires_weight_load=True``; weights must be pulled onto each
    survivor before it serves the added layers.

    Args:
        failed_start_layer: First layer of the failed node.
        failed_end_layer: Last layer (inclusive) of the failed node.
        surviving_nodes: ``{node_id: (start_layer, end_layer)}`` for alive nodes.
        survivor_memory_gb: ``{node_id: free_memory_gb}`` for each survivor.
            If None, falls back to even distribution.
        min_memory_per_layer_gb: Minimum free GB required per additional layer.

    Returns:
        List of :class:`LayerRedistribution` describing the new assignments.
    """

    if not surviving_nodes:
        return []

    dead_count = failed_end_layer - failed_start_layer + 1
    if dead_count <= 0:
        return []

    # If no memory info, fall back to even distribution
    if survivor_memory_gb is None:
        return compute_redistributions(
            failed_start_layer, failed_end_layer, surviving_nodes
        )

    # Filter survivors with enough memory and compute capacity weights
    eligible: dict[str, tuple[int, int, float]] = {}  # nid -> (start, end, weight)
    for nid, (cur_start, cur_end) in surviving_nodes.items():
        free_gb = survivor_memory_gb.get(nid, 0.0)
        current_layers = cur_end - cur_start + 1
        # Estimate available capacity: free memory / memory per layer - current layers
        max_additional = max(0, int(free_gb / min_memory_per_layer_gb) - current_layers)
        if max_additional > 0:
            eligible[nid] = (cur_start, cur_end, float(max_additional))

    # If no survivor has capacity, fall back to even distribution (best effort)
    if not eligible:
        logger.warning(
            "No survivor has enough GPU capacity for redistribution, "
            "falling back to even distribution"
        )
        return compute_redistributions(
            failed_start_layer, failed_end_layer, surviving_nodes
        )

    ordered_eligible = sorted(
        eligible.items(), key=lambda x: (x[1][0], x[1][1])
    )

    # If the layout is not clean (a survivor overlaps another or
    # intersects the orphan itself), adjacency absorption cannot produce
    # a valid partition — re-tile the whole span with capacity weights.
    all_ordered_pre = sorted(
        ((nid, rng) for nid, rng in surviving_nodes.items()),
        key=lambda x: (x[1][0], x[1][1]),
    )
    if not _is_clean_tiling(all_ordered_pre, failed_start_layer, failed_end_layer):
        span_start = min(all_ordered_pre[0][1][0], failed_start_layer)
        span_end = max(all_ordered_pre[-1][1][1], failed_end_layer)
        weights = {nid: w for nid, (_s, _e, w) in ordered_eligible}
        logger.warning(
            "Survivor topology is not a clean tiling around failed layers "
            f"[{failed_start_layer}, {failed_end_layer}] — falling back to "
            "capacity-weighted full re-partition"
        )
        return _retiling_redistributions(
            all_ordered_pre, span_start, span_end, weights,
        )

    # Adjacency slots computed over ALL live survivors; keep only those
    # that are ALSO capacity-eligible, normalized to
    # (nid, (start, end), weight).
    all_ordered = all_ordered_pre
    left_entry = None
    for entry in all_ordered:
        if entry[1][1] < failed_start_layer:
            left_entry = entry
    right_entry = None
    for entry in reversed(all_ordered):
        if entry[1][0] > failed_end_layer:
            right_entry = entry

    eligible_by_id = {nid: w for nid, (_s, _e, w) in ordered_eligible}
    absorbers = [
        (e[0], e[1], eligible_by_id[e[0]])
        for e in (left_entry, right_entry)
        if e is not None and e[0] in eligible_by_id
    ]

    if not absorbers:
        if left_entry is None and right_entry is None:
            # No live node adjacent to the orphan on either side — re-tile
            # the whole span with capacity weights so coverage stays complete.
            span_start = min(all_ordered[0][1][0], failed_start_layer)
            span_end = max(all_ordered[-1][1][1], failed_end_layer)
            weights = {nid: w for nid, (_s, _e, w) in ordered_eligible}
            return _retiling_redistributions(
                all_ordered, span_start, span_end, weights,
            )
        # Adjacent neighbors exist but lack capacity — best-effort even
        # split (documented fallback), which preserves disjointness.
        logger.warning(
            "Adjacent survivors lack GPU capacity for redistribution — "
            "falling back to even distribution"
        )
        return compute_redistributions(
            failed_start_layer, failed_end_layer, surviving_nodes
        )

    if len(absorbers) == 2:
        cap_weights = {nid: w for nid, (_s, _e), w in absorbers}
        counts = _largest_remainder_split(dead_count, cap_weights)
    elif left_entry is not None and absorbers[0][0] == left_entry[0]:
        counts = {left_entry[0]: dead_count}
    else:
        counts = {right_entry[0]: dead_count}

    redistributions: list[LayerRedistribution] = []
    offset = 0
    for nid, (cur_start, cur_end), _w in absorbers:
        added_count = min(counts.get(nid, 0), dead_count - offset)
        if added_count <= 0:
            continue
        added_start = failed_start_layer + offset
        added_end = added_start + added_count - 1
        offset += added_count
        redistributions.append(LayerRedistribution(
            surviving_node_id=nid,
            added_start_layer=added_start,
            added_end_layer=added_end,
            new_start_layer=min(cur_start, added_start),
            new_end_layer=max(cur_end, added_end),
            requires_weight_load=True,
        ))

    # Safety net: exact apportionment should always cover the orphan, but
    # never emit an uncovered tail if rounding ever misbehaves.
    if offset < dead_count:
        nid, (cur_start, cur_end), _w = absorbers[0]
        existing = next(
            (r for r in redistributions if r.surviving_node_id == nid), None
        )
        if existing is not None:
            existing.added_end_layer = failed_end_layer
            existing.new_end_layer = max(existing.new_end_layer, cur_end)
        else:
            redistributions.append(LayerRedistribution(
                surviving_node_id=nid,
                added_start_layer=failed_start_layer + offset,
                added_end_layer=failed_end_layer,
                new_start_layer=min(cur_start, failed_start_layer + offset),
                new_end_layer=max(cur_end, failed_end_layer),
                requires_weight_load=True,
            ))

    return redistributions

class RecoveryState(str, enum.Enum):
    """States in the node recovery lifecycle."""

    IDLE = "idle"                      # No active recovery
    DETECTING = "detecting"            # Health probe failure detected
    DRAINING = "draining"              # Draining in-flight requests from failed node
    REDISTRIBUTING = "redistributing"  # Reassigning layers to survivors
    RECOVERING_SEQUENCES = "recovering_sequences"  # Restoring in-flight sequences
    VERIFYING = "verifying"            # Verifying recovery succeeded
    FAILED = "failed"                  # Recovery failed, manual intervention needed

class NodeRecoveryManager:
    """Orchestrates self-healing when a node fails.

    Usage (set callbacks, then call from health probe loop)::

        mgr = NodeRecoveryManager()
        mgr.set_drain_callback(lambda nid: ...)
        mgr.set_recover_sequences_callback(lambda nid, seqs: ...)
        mgr.set_redistribute_layers_callback(lambda nid, plan: ...)
        mgr.on_node_failure("node-3")
    """

    _recovery_serializer = (
        threading.Lock()
    )  # Serializes top-level on_node_failure calls to prevent state-machine races.

    def __init__(
        self,
        node_id: str = "coordinator",
        checkpoint_ttl_s: float = 300.0,
        persist_path: str | None = None,
        dry_run: bool = False,
        async_checkpoint_executor: ThreadPoolExecutor | None = None,
    ):
        self._node_id = node_id
        self._state = RecoveryState.IDLE
        self._draining: set[str] = set()
        self._dead_nodes: set[str] = set()
        self._checkpoints: dict[str, SequenceCheckpoint] = {}
        self._seq_to_node: dict[str, str] = {}
        self._recovered_requests: set[str] = set()
        # Survivors whose layer ranges were relabeled by redistribution but
        # whose model weights have NOT yet been pulled onto the node (the
        # planner is metadata-only).  Cleared via mark_weights_loaded().
        self._pending_weight_reload: set[str] = set()
        self._lock = threading.Lock()
        self._dry_run = dry_run

        # Thread pool for async checkpoint persistence (disk I/O off the
        # inference path).  Uses a shared pool if provided, otherwise creates
        # a single-thread executor.
        self._async_ckpt_executor = async_checkpoint_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ckpt-writer",
        )

        self._on_drain: Callable[[str], None] | None = None
        self._on_redistribute: Callable[[str, NodeRecoveryPlan], None] | None = None
        # {node_id: (start_layer, end_layer)} — topology used to plan
        # layer redistributions on node failure.
        self._layer_assignments: dict[str, tuple[int, int]] = {}
        self._on_recover: Callable[[str, list[str]], list[Any]] | None = None
        self._on_mark_dead: Callable[[str], None] | None = None

        self._checkpoint_ttl = checkpoint_ttl_s
        self._persist_path = persist_path
        self._recovery_history: list[dict] = []

        self._metrics = {
            "recoveries": 0,
            "failed_nodes": 0,
            "sequences_recovered": 0,
            "sequences_lost": 0,
            "checkpoint_count": 0,
            "total_recovery_time_ms": 0,
            "pending_weight_reloads": 0,
        }

        self._init_prometheus()

    # ── Prometheus metrics ──────────────────────────────────────────────────

    def _init_prometheus(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
            self._registry = CollectorRegistry()
            self._prom_recovery_counter = Counter(
                "distllm_recovery_total", "Total recoveries",
                ["result"], registry=self._registry,
            )
            self._prom_recovery_duration = Histogram(
                "distllm_recovery_duration_ms", "Recovery duration (ms)",
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
                registry=self._registry,
            )
            self._prom_recovery_duration_seconds = Histogram(
                "distllm_recovery_duration_seconds", "Recovery duration (seconds)",
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self._registry,
            )
            self._prom_checkpoints = Gauge(
                "distllm_checkpoints_active", "Active checkpoints",
                registry=self._registry,
            )
            self._prom_sequences_lost = Counter(
                "distllm_sequences_lost_total", "Sequences lost",
                registry=self._registry,
            )
            self._prom_async_checkpoints = Counter(
                "distllm_async_checkpoints_total",
                "Async checkpoint disk writes",
                registry=self._registry,
            )
            self._prometheus_enabled = True
        except ImportError:
            self._prometheus_enabled = False

    # ── Callbacks ───────────────────────────────────────────────────────────

    def set_drain_callback(self, cb: Callable[[str], None]) -> None:
        self._on_drain = cb

    def set_redistribute_layers_callback(
        self, cb: Callable[[str, NodeRecoveryPlan], None]
    ) -> None:
        self._on_redistribute = cb

    def set_layer_assignments(
        self, assignments: dict[str, tuple[int, int]]
    ) -> None:
        """Register the pipeline topology as ``{node_id: (start, end)}``.

        Needed for layer-redistribution planning: without it a failed
        node has no survivors to reassign layers to and the redistribute
        callback can never fire.
        """
        self._layer_assignments = dict(assignments)

    def set_recover_sequences_callback(
        self, cb: Callable[[str, list[str]], list[Any]]
    ) -> None:
        self._on_recover = cb

    def set_mark_dead_callback(self, cb: Callable[[str], None]) -> None:
        self._on_mark_dead = cb

    # ── Checkpoint CRUD ─────────────────────────────────────────────────────

    def save_checkpoint(
        self, request_id: str, kv_cache: Any,
        prompt_tokens: list[int], generated_tokens: list[int],
        node_id: str,
        async_persist: bool = False,
    ) -> None:
        """Save a checkpoint for an in-flight sequence.

        When *async_persist* is ``True``, the checkpoint is written to
        the in-memory store synchronously (fast, non-blocking), but the
        optional disk flush (:meth:`save_to_disk`) is offloaded to a
        background thread so the inference path is not blocked by file I/O.
        """

        ckpt = SequenceCheckpoint(
            request_id=request_id,
            kv_cache=list(kv_cache) if kv_cache is not None else kv_cache,
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
            if self._prometheus_enabled:
                self._prom_checkpoints.set(len(self._checkpoints))

        # Async disk persistence — offload to background thread pool so
        # the inference caller does not wait on file I/O.
        if async_persist and self._persist_path:
            self._async_ckpt_executor.submit(self._persist_bg, request_id)

    def _persist_bg(self, request_id: str) -> None:
        """Background task: flush a single checkpoint to disk."""

        try:
            self.save_to_disk(include_kv_cache=False)
            if self._prometheus_enabled:
                self._prom_async_checkpoints.inc()
            logger.debug(f"Async checkpoint persisted for {request_id}")
        except Exception as e:
            logger.warning(f"Async checkpoint persistence failed for {request_id}: {e}")

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

    def evict_stale_checkpoints(self) -> int:
        """Remove checkpoints older than TTL. Returns count evicted."""

        now = time.time()
        evicted = 0
        with self._lock:
            stale = [
                rid for rid, ckpt in self._checkpoints.items()
                if now - ckpt.timestamp > self._checkpoint_ttl
            ]
            for rid in stale:
                self._checkpoints.pop(rid, None)
                self._seq_to_node.pop(rid, None)
                evicted += 1
            if evicted:
                logger.debug(f"Evicted {evicted} stale checkpoints (TTL={self._checkpoint_ttl}s)")
        return evicted

    # ── Checkpoint disk persistence ─────────────────────────────────────────

    def save_to_disk(self, path: str | None = None, include_kv_cache: bool = False) -> bool:
        """Flush checkpoints to disk for crash recovery.

        Args:
            path: File path to save to. Uses persist_path if None.
            include_kv_cache: If True, serialize KV cache tensors to a
                companion .pt file.  The JSON manifest stores metadata
                only; the tensors are in the .pt file.

        Returns:
            True on success.
        """

        path = path or self._persist_path
        if not path:
            return False
        try:
            with self._lock:
                data = {
                    "version": 2,
                    "timestamp": time.time(),
                    "checkpoints": {},
                    "metrics": dict(self._metrics),
                }
                kv_caches = {}
                for rid, ckpt in self._checkpoints.items():
                    data["checkpoints"][rid] = {
                        "request_id": ckpt.request_id,
                        "prompt_tokens": ckpt.prompt_tokens,
                        "generated_tokens": ckpt.generated_tokens,
                        "node_id": ckpt.node_id,
                        "timestamp": ckpt.timestamp,
                        "has_kv_cache": ckpt.kv_cache is not None,
                    }
                    if include_kv_cache and ckpt.kv_cache is not None:
                        kv_caches[rid] = ckpt.kv_cache

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Save KV cache tensors to companion file
            if include_kv_cache and kv_caches:
                kv_path = path + ".kv.pt"
                torch.save(kv_caches, kv_path)
                logger.info(f"KV cache saved to {kv_path} ({len(kv_caches)} entries)")

            logger.info(f"Checkpoints saved to {path} ({len(data['checkpoints'])} entries)")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}")
            return False

    def load_from_disk(self, path: str | None = None) -> bool:
        """Restore checkpoints after coordinator restart.

        Loads the JSON manifest and, if present, the companion .kv.pt
        file containing serialized KV cache tensors.

        Returns:
            True on success.
        """

        path = path or self._persist_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            version = data.get("version", 1)
            if version not in (1, 2):
                logger.warning(f"Unsupported checkpoint file version: {version}")
                return False

            # Load KV cache tensors if companion file exists
            kv_path = path + ".kv.pt"
            kv_caches: dict[str, Any] = {}
            if os.path.exists(kv_path):
                try:
                    kv_caches = torch.load(kv_path, map_location="cpu", weights_only=True)
                    logger.info(f"KV cache loaded from {kv_path} ({len(kv_caches)} entries)")
                except Exception as e:
                    logger.warning(f"Failed to load KV cache from {kv_path}: {e}")

            with self._lock:
                for rid, ckpt_data in data.get("checkpoints", {}).items():
                    kv_cache = kv_caches.get(rid) if rid in kv_caches else None
                    self._checkpoints[rid] = SequenceCheckpoint(
                        request_id=ckpt_data["request_id"],
                        kv_cache=kv_cache,
                        prompt_tokens=ckpt_data["prompt_tokens"],
                        generated_tokens=ckpt_data["generated_tokens"],
                        node_id=ckpt_data["node_id"],
                        timestamp=ckpt_data.get("timestamp", time.time()),
                    )
                    self._seq_to_node[rid] = ckpt_data["node_id"]
                self._metrics["checkpoint_count"] = len(self._checkpoints)
            n_loaded = len(data.get("checkpoints", {}))
            n_kv = len(kv_caches)
            logger.info(f"Checkpoints loaded from {path} ({n_loaded} entries, {n_kv} with KV cache)")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoints: {e}")
            return False

    # ── Node state ──────────────────────────────────────────────────────────

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
        with self._lock:
            self._draining.discard(node_id)
            self._dead_nodes.discard(node_id)

    # ── Main recovery flow ──────────────────────────────────────────────────

    async def on_node_failure_async(self, failed_node_id: str) -> NodeRecoveryPlan:
        """Async version of on_node_failure for non-blocking recovery.

        Runs callbacks in a thread pool to avoid blocking the event loop.
        Returns the same NodeRecoveryPlan as the synchronous version.
        """

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.on_node_failure, failed_node_id)

    def dry_run_recovery(self, failed_node_id: str) -> NodeRecoveryPlan:
        """Test the recovery flow without destructive side effects.

        Simulates the full recovery lifecycle — detection, draining,
        redistribution, sequence recovery — but **skips** the callbacks
        that would actually terminate nodes, reassign layers, or restore
        sequences.  Returns the same :class:`NodeRecoveryPlan` so the
        caller can inspect redistributions, sequence count, and expected
        recovery time.

        Use this to validate that the cluster *would* survive a node
        failure before triggering actual recovery.
        """

        original = self._dry_run
        self._dry_run = True
        try:
            return self.on_node_failure(failed_node_id)
        finally:
            self._dry_run = original

    def on_node_failure(self, failed_node_id: str) -> NodeRecoveryPlan:
        """Handle a node failure event.

        M-03: Thread-safe against concurrent failures. Uses _recovery_serializer
        to serialize the *top-level* recovery flow so that overlapping calls for
        different nodes do not race on the state machine. The per-instance
        ``_lock`` guards individual data-structure operations (short-lived,
        released before callbacks).

        Callbacks (drain, recover, mark_dead, redistribute) are invoked
        *outside* ``_lock`` to avoid re-entrancy / deadlock — they are free
        to acquire other locks or call back into this manager.
        """

        with self._recovery_serializer:
            return self._on_node_failure_impl(failed_node_id)

    def _on_node_failure_impl(self, failed_node_id: str) -> NodeRecoveryPlan:
        """Synchronized implementation of :meth:`on_node_failure`.

        Split out so the serializer lock is held for the entire recovery
        sequence, preventing overlapping state transitions while keeping
        ``_lock`` short-lived (released before callbacks).
        """

        with self._lock:
            # Don't recover a node already being recovered
            if failed_node_id in self._dead_nodes or failed_node_id in self._draining:
                logger.warning(f"Node {failed_node_id} is already being recovered")
                return NodeRecoveryPlan(failed_node_id=failed_node_id)
            self._state = RecoveryState.DETECTING

        logger.warning(f"Starting recovery for failed node: {failed_node_id}")
        start_time = time.monotonic()
        plan = NodeRecoveryPlan(failed_node_id=failed_node_id)

        # Step 1: Mark draining + dead
        with self._lock:
            self._state = RecoveryState.DRAINING
            self._draining.add(failed_node_id)
            self._dead_nodes.add(failed_node_id)
        if self._on_drain and not self._dry_run:
            self._on_drain(failed_node_id)
        plan.drain_time_ms = (time.monotonic() - start_time) * 1000

        # Step 2: Retrieve checkpoints for failed node
        with self._lock:
            self._state = RecoveryState.REDISTRIBUTING
        seqs = self.get_checkpoints_for_node(failed_node_id)
        recovered_ids = list(seqs.keys())

        # Step 3: Attempt sequence recovery
        with self._lock:
            self._state = RecoveryState.RECOVERING_SEQUENCES
        t0 = time.monotonic()
        if self._on_recover and recovered_ids:
            try:
                if not self._dry_run:
                    self._on_recover(failed_node_id, recovered_ids)
                with self._lock:
                    for rid in recovered_ids:
                        self._recovered_requests.add(rid)
                        if not self._dry_run:
                            self._checkpoints.pop(rid, None)
                            self._seq_to_node.pop(rid, None)
                plan.recovered_sequences = recovered_ids
            except Exception as e:
                logger.error(f"Sequence recovery failed: {e}")
                plan.total_sequences_lost = len(recovered_ids)
        else:
            plan.total_sequences_lost = len(recovered_ids)
        plan.recovery_time_ms = (time.monotonic() - t0) * 1000

        # Step 4: Fire mark-dead callback (removes from pipeline.nodes)
        if self._on_mark_dead and not self._dry_run:
            try:
                self._on_mark_dead(failed_node_id)
            except Exception as e:
                logger.error(f"Mark-dead callback failed: {e}")

        # Step 5: Layer redistribution (parallel across survivors)
        # Planning also happens during dry runs (recovery drills need the
        # redistribution count); only dispatch/side effects are skipped.
        assignments = self._layer_assignments
        if assignments and failed_node_id in assignments:
            survivors = {
                nid: rng for nid, rng in assignments.items()
                if nid != failed_node_id
            }
            start_l, end_l = assignments[failed_node_id]
            plan.redistributions = compute_redistributions(
                start_l, end_l, survivors,
            )
            if plan.redistributions and not self._dry_run:
                self._flag_weight_reloads(plan)
                logger.warning(
                    f"Layer redistribution for {failed_node_id} is "
                    "METADATA-ONLY: no model weights were transferred. "
                    "Survivors "
                    f"{sorted(rd.surviving_node_id for rd in plan.redistributions if rd.requires_weight_load)} "
                    "must pull their new layers (e.g. via TransferWeightsStream) "
                    "before serving them — see nodes_needing_weight_reload(). "
                    "plan.weights_transferred remains False."
                )
                # Refresh the cached topology so a subsequent failure
                # plans against the post-recovery ranges.
                refreshed = dict(assignments)
                del refreshed[failed_node_id]
                for rd in plan.redistributions:
                    refreshed[rd.surviving_node_id] = (
                        rd.new_start_layer, rd.new_end_layer,
                    )
                self.set_layer_assignments(refreshed)
        if self._on_redistribute and not self._dry_run:
            self._redistribute_parallel(failed_node_id, plan)

        # Step 6: Record metrics
        total_time = (time.monotonic() - start_time) * 1000
        plan.recovery_time_ms = total_time

        with self._lock:
            self._metrics["recoveries"] += 1
            self._metrics["failed_nodes"] += 1
            self._metrics["sequences_recovered"] += len(plan.recovered_sequences)
            self._metrics["sequences_lost"] += plan.total_sequences_lost
            self._metrics["total_recovery_time_ms"] += total_time

        if self._prometheus_enabled:
            self._prom_recovery_counter.labels(
                result="success" if plan.recovered_sequences else "loss"
            ).inc()
            self._prom_recovery_duration.observe(total_time)
            self._prom_recovery_duration_seconds.observe(total_time / 1000.0)
            if plan.total_sequences_lost > 0:
                self._prom_sequences_lost.inc(plan.total_sequences_lost)

        # Step 7: Record in audit history
        self._recovery_history.append({
            "timestamp": time.time(),
            "event": "recovery",
            "node_id": failed_node_id,
            "sequences_recovered": len(plan.recovered_sequences),
            "sequences_lost": plan.total_sequences_lost,
            "redistributions": len(plan.redistributions),
            "duration_ms": total_time,
            "weights_transferred": plan.weights_transferred,
        })

        logger.info(
            f"Recovery complete for {failed_node_id}: "
            f"{len(plan.recovered_sequences)} sequences recovered, "
            f"{plan.total_sequences_lost} sequences lost, "
            f"{len(plan.redistributions)} layer redistributions, "
            f"{total_time:.0f}ms total"
        )
        with self._lock:
            self._state = RecoveryState.IDLE
        return plan

        # (end of _on_node_failure_impl — the serializer context manager
        #  in on_node_failure releases after return)

    @property
    def recovery_state(self) -> RecoveryState:
        """Return the current recovery state."""

        return self._state

    # ── Parallel redistribution ─────────────────────────────────────────────

    def _flag_weight_reloads(self, plan: NodeRecoveryPlan) -> None:
        """Record survivors whose new layers still need a weight pull."""

        with self._lock:
            for rd in plan.redistributions:
                if rd.requires_weight_load and rd.added_end_layer >= rd.added_start_layer:
                    self._pending_weight_reload.add(rd.surviving_node_id)
            self._metrics["pending_weight_reloads"] = len(self._pending_weight_reload)

    def nodes_needing_weight_reload(self) -> list[str]:
        """Survivors relabeled by redistribution whose weights are not loaded.

        The redistribution planner is metadata-only — it cannot move model
        weights.  Until the coordinator pulls each survivor's added layers
        (e.g. via ``TransferWeightsStream``) and calls
        :meth:`mark_weights_loaded`, that node must not serve its added
        layers.
        """

        with self._lock:
            return sorted(self._pending_weight_reload)

    def mark_weights_loaded(self, node_id: str) -> None:
        """Clear the pending-weight-reload flag after a successful pull."""

        with self._lock:
            self._pending_weight_reload.discard(node_id)
            self._metrics["pending_weight_reloads"] = len(self._pending_weight_reload)

    def _redistribute_parallel(self, failed_node_id: str, plan: NodeRecoveryPlan) -> None:
        """Redistribute layers across surviving nodes in parallel.

        Each surviving node's layer load operation is submitted to a
        thread pool so that redistributions happen concurrently rather
        than sequentially, reducing total recovery time proportional to
        the number of survivors.

        C3 fix: each callback invocation receives a single-redistribution
        plan scoped to ONE survivor.  The previous code passed the full
        multi-survivor plan on every call, so consumers applying the whole
        plan (e.g. ``core/coordinator.py:_on_node_redistribute``) applied
        every redistribution N times.
        """

        redistributions = plan.redistributions
        if not redistributions:
            logger.info("No layer redistributions needed — no survivors")
            return

        def _redistribute_one(rd: LayerRedistribution) -> bool:
            """Dispatch one redistribution to its survivor node."""

            try:
                if self._on_redistribute:
                    # Single-redistribution plan: this callback applies
                    # exactly this survivor's new range, nothing else.
                    per_node_plan = NodeRecoveryPlan(
                        failed_node_id=failed_node_id,
                        redistributions=[rd],
                    )
                    self._on_redistribute(failed_node_id, per_node_plan)
                return True
            except Exception as e:
                logger.error(
                    f"Parallel redistribution failed for node "
                    f"{rd.surviving_node_id}: {e}"
                )
                return False

        with ThreadPoolExecutor(
            max_workers=len(redistributions),
            thread_name_prefix="redistribute",
        ) as pool:
            futures = [pool.submit(_redistribute_one, rd) for rd in redistributions]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r)
        logger.info(
            f"Parallel redistribution: {success_count}/{len(redistributions)} "
            f"redistributions completed"
        )

    # ── Recovered request tracking ──────────────────────────────────────────

    def is_recovered_request(self, request_id: str) -> bool:
        return request_id in self._recovered_requests

    def consume_recovered_flag(self, request_id: str) -> bool:
        with self._lock:
            if request_id in self._recovered_requests:
                self._recovered_requests.discard(request_id)
                return True
            return False

    # ── Metrics & audit ─────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        with self._lock:
            m = dict(self._metrics)
            m["draining_nodes"] = len(self._draining)
            m["dead_nodes"] = len(self._dead_nodes)
            m["active_checkpoints"] = len(self._checkpoints)
            return m

    def get_recovery_history(self) -> list[dict]:
        return list(self._recovery_history)
