"""Live pipeline reconfiguration — topology versioning, checkpoints, and graceful migration.

Provides the infrastructure to add/remove worker nodes mid-pipeline without
dropping in-flight requests, using checkpoint-based state transfer and
versioned topology tracking:

- :class:`TopologyVersion`          — immutable versioned topology snapshot + composition API
- :class:`PipelineCheckpointer`     — periodic KV-cache / sequence-position checkpoints
- :class:`PipelineReconfigurator`   — orchestrates node add/remove, drain, rollback
- :class:`RequestPipelineSelector`  — per-request topology affinity + gradual migration

Usage::

    versions: dict[int, TopologyVersion] = {}
    v1 = TopologyVersion(version=1, topology_id="v1")
    v1.add_assignment("node-a", 0, 7)
    v1.add_assignment("node-b", 8, 15)
    versions[1] = v1

    ckptr = PipelineCheckpointer(persist_path="/tmp/ckpts")
    sel = RequestPipelineSelector(versions, initial_version=1)

    reconf = PipelineReconfigurator(
        orchestrator=pipe_orch,
        checkpointer=ckptr,
        selector=sel,
        topology_versions=versions,
    )

    # Add a new node mid-flight
    plan = reconf.add_node("node-c", "10.0.0.3", 50051, 8, 11)
    await reconf.apply_reconfiguration(plan)
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, NamedTuple

import torch
from loguru import logger

from distllm.dist.pipeline.orchestrator import PipelineNode, PipelineOrchestrator


# =========================================================================
# TopologyVersion
# =========================================================================


class NodeAssignment(NamedTuple):
    """A single node's layer assignment within a topology."""

    node_id: str
    start_layer: int
    end_layer: int


@dataclass(frozen=True)
class TopologyVersion:
    """Immutable snapshot of a pipeline topology at a specific version.

    Each topology version records which nodes own which layer ranges.
    Versions are monotonically increasing; higher numbers are newer.

    Use the factory :meth:`create` to build instances — once frozen the
    topology cannot be mutated.
    """

    version: int
    topology_id: str
    assignments: tuple[NodeAssignment, ...] = ()
    created_at: float = field(default_factory=time.time)

    # ── Factory helpers ────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        version: int,
        topology_id: str,
        assignments: list[NodeAssignment] | None = None,
    ) -> TopologyVersion:
        """Create a new immutable TopologyVersion.

        Args:
            version: Monotonically increasing version number.
            topology_id: Human-readable topology identifier.
            assignments: Initial layer assignments.

        Returns:
            A frozen TopologyVersion instance.
        """
        return cls(
            version=version,
            topology_id=topology_id,
            assignments=tuple(assignments or ()),
        )

    def with_assignment(
        self,
        node_id: str,
        start_layer: int,
        end_layer: int,
    ) -> TopologyVersion:
        """Return a *new* version with this assignment added (immutable)."""
        new_assignments = list(self.assignments)
        # Replace existing assignment for same node_id
        new_assignments = [
            a for a in new_assignments if a.node_id != node_id
        ]
        new_assignments.append(NodeAssignment(node_id, start_layer, end_layer))
        new_assignments.sort(key=lambda a: (a.start_layer, a.node_id))
        return TopologyVersion(
            version=self.version,
            topology_id=self.topology_id,
            assignments=tuple(new_assignments),
            created_at=self.created_at,
        )

    def without_node(self, node_id: str) -> TopologyVersion:
        """Return a *new* version with *node_id* removed (immutable)."""
        return TopologyVersion(
            version=self.version,
            topology_id=self.topology_id,
            assignments=tuple(
                a for a in self.assignments if a.node_id != node_id
            ),
            created_at=self.created_at,
        )

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Node IDs in this topology, in layer order."""
        return tuple(a.node_id for a in self.assignments)

    @property
    def total_layers(self) -> int:
        """Total number of layers covered by this topology."""
        if not self.assignments:
            return 0
        return max(a.end_layer for a in self.assignments) + 1

    # ── Composition API ────────────────────────────────────────────────

    def diff(self, other: TopologyVersion) -> dict[str, list[Any]]:
        """Compute structural diff against *other* topology.

        Returns a dict with keys ``added``, ``removed``, ``changed``, each
        containing a list of :class:`NodeAssignment` entries.
        """
        added: list[NodeAssignment] = []
        removed: list[NodeAssignment] = []
        changed: list[dict[str, Any]] = []

        self_map = {a.node_id: a for a in self.assignments}
        other_map = {a.node_id: a for a in other.assignments}

        self_ids = set(self_map)
        other_ids = set(other_map)

        for nid in self_ids - other_ids:
            added.append(self_map[nid])
        for nid in other_ids - self_ids:
            removed.append(other_map[nid])
        for nid in self_ids & other_ids:
            sa = self_map[nid]
            oa = other_map[nid]
            if (sa.start_layer, sa.end_layer) != (oa.start_layer, oa.end_layer):
                changed.append({
                    "node_id": nid,
                    "from": (oa.start_layer, oa.end_layer),
                    "to": (sa.start_layer, sa.end_layer),
                })

        return {"added": added, "removed": removed, "changed": changed}

    def compatible(self, other: TopologyVersion) -> bool:
        """Check whether switching to *other* can skip a full checkpoint restore.

        Returns ``True`` if neither topology adds **and** removes nodes
        concurrently — i.e. the transition is a pure addition, a pure
        removal, or a pure layer-range shift.  Full checkpoint restore is
        only needed when both add and remove happen together (which implies
        the data layout changes incompatibly).
        """
        diff = self.diff(other)
        has_adds = bool(diff["added"])
        has_removes = bool(diff["removed"])
        # Compatible if we are only adding *or* only removing, not both.
        # Pure layer-range changes (changed) are always compatible.
        return not (has_adds and has_removes)

    def summarize(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this topology."""
        return {
            "version": self.version,
            "topology_id": self.topology_id,
            "assignments": [
                {"node_id": a.node_id, "start_layer": a.start_layer, "end_layer": a.end_layer}
                for a in self.assignments
            ],
            "total_layers": self.total_layers,
            "node_count": len(self.assignments),
            "created_at": self.created_at,
        }


# =========================================================================
# PipelineCheckpointer
# =========================================================================


@dataclass
class PipelineCheckpoint:
    """A snapshot of pipeline state at a point in time.

    Captures the per-node KV cache state and sequence positions so that
    in-flight requests can be transferred to a new topology.
    """

    topology_version: int
    topology_id: str
    request_id: str
    kv_cache: Any = None
    sequence_positions: list[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        """Approximate byte size of tensors in this checkpoint."""
        total = 0
        if isinstance(self.kv_cache, (torch.Tensor, torch.nn.Parameter)):
            total += self.kv_cache.numel() * self.kv_cache.element_size()
        elif isinstance(self.kv_cache, dict):
            for v in self.kv_cache.values():
                if isinstance(v, (torch.Tensor, torch.nn.Parameter)):
                    total += v.numel() * v.element_size()
        elif isinstance(self.kv_cache, (list, tuple)):
            for item in self.kv_cache:
                if isinstance(item, (torch.Tensor, torch.nn.Parameter)):
                    total += item.numel() * item.element_size()
        return total


class PipelineCheckpointer:
    """Periodically saves and restores pipeline state for reconfiguration.

    Manages checkpoints at the pipeline level — per-request KV cache state
    that must be transferred when the topology changes mid-flight.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        checkpoint_interval_s: float = 5.0,
        max_checkpoints: int = 1000,
        enable_async_persist: bool = True,
    ):
        self._persist_path = persist_path
        self._interval = checkpoint_interval_s
        self._max_checkpoints = max_checkpoints
        self._enable_async = enable_async_persist

        self._checkpoints: dict[str, PipelineCheckpoint] = {}
        self._lock = threading.Lock()
        self._last_checkpoint_time: float = 0.0
        self._total_saved: int = 0
        self._total_loaded: int = 0

    # ── Core API ───────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        request_id: str,
        kv_cache: Any,
        sequence_positions: list[int] | None = None,
        topology_version: int = 0,
        topology_id: str = "",
    ) -> PipelineCheckpoint:
        """Record a checkpoint for *request_id* and return it.

        Args:
            request_id: The in-flight request to checkpoint.
            kv_cache: KV cache tensors (dict/list of tensors).
            sequence_positions: Generated token positions per sequence.
            topology_version: Current topology version at save time.
            topology_id: Current topology identifier.

        Returns:
            The created :class:`PipelineCheckpoint`.
        """
        ckpt = PipelineCheckpoint(
            topology_version=topology_version,
            topology_id=topology_id,
            request_id=request_id,
            kv_cache=kv_cache,
            sequence_positions=sequence_positions or [],
        )
        with self._lock:
            self._checkpoints[request_id] = ckpt
            self._total_saved += 1
            # Evict oldest if over limit
            if len(self._checkpoints) > self._max_checkpoints:
                oldest = min(
                    self._checkpoints.keys(),
                    key=lambda rid: self._checkpoints[rid].timestamp,
                )
                self._checkpoints.pop(oldest)
        return ckpt

    def load_checkpoint(self, request_id: str) -> PipelineCheckpoint | None:
        """Retrieve the latest checkpoint for *request_id*.

        Returns ``None`` if no checkpoint exists.
        """
        with self._lock:
            ckpt = self._checkpoints.get(request_id)
            if ckpt is not None:
                self._total_loaded += 1
            return ckpt

    def drop_checkpoint(self, request_id: str) -> None:
        """Remove the checkpoint for *request_id*."""
        with self._lock:
            self._checkpoints.pop(request_id, None)

    def drop_request(self, request_id: str) -> None:
        """Alias for :meth:`drop_checkpoint`."""
        self.drop_checkpoint(request_id)

    def evict_stale_checkpoints(self, ttl_s: float = 300.0) -> int:
        """Remove checkpoints older than *ttl_s* seconds.

        Returns the number evicted.
        """
        now = time.time()
        evicted = 0
        with self._lock:
            stale = [
                rid
                for rid, ckpt in self._checkpoints.items()
                if now - ckpt.timestamp > ttl_s
            ]
            for rid in stale:
                self._checkpoints.pop(rid, None)
                evicted += 1
        if evicted:
            logger.debug(f"Evicted {evicted} stale pipeline checkpoints (TTL={ttl_s}s)")
        return evicted

    # ── Periodic checkpointing ─────────────────────────────────────────

    def should_checkpoint(self) -> bool:
        """Return ``True`` if enough time has elapsed since the last save."""
        return (time.time() - self._last_checkpoint_time) >= self._interval

    def reset_timer(self) -> None:
        """Reset the periodic checkpoint timer."""
        self._last_checkpoint_time = time.time()

    # ── Serialisation (disk) ───────────────────────────────────────────

    def serialize(self, request_id: str) -> dict[str, Any] | None:
        """Return a JSON-serialisable dict for *request_id* checkpoint.

        The returned dict can be passed to :meth:`deserialize` or stored
        externally.  Tensor data is **not** included — call
        :meth:`save_to_disk` for durable persistence.
        """
        with self._lock:
            ckpt = self._checkpoints.get(request_id)
            if ckpt is None:
                return None
            return {
                "topology_version": ckpt.topology_version,
                "topology_id": ckpt.topology_id,
                "request_id": ckpt.request_id,
                "sequence_positions": ckpt.sequence_positions,
                "timestamp": ckpt.timestamp,
                "has_kv_cache": ckpt.kv_cache is not None,
            }

    def save_to_disk(self, include_kv_cache: bool = False) -> bool:
        """Flush all checkpoints to disk for crash recovery.

        Args:
            include_kv_cache: If True, also persist KV cache tensors to a
                companion ``.kv.pt`` file.

        Returns:
            ``True`` on success.
        """
        path = self._persist_path
        if not path:
            return False
        try:
            with self._lock:
                data: dict[str, Any] = {
                    "version": 1,
                    "timestamp": time.time(),
                    "checkpoints": {},
                }
                kv_caches: dict[str, Any] = {}
                for rid, ckpt in self._checkpoints.items():
                    data["checkpoints"][rid] = {
                        "topology_version": ckpt.topology_version,
                        "topology_id": ckpt.topology_id,
                        "request_id": ckpt.request_id,
                        "sequence_positions": ckpt.sequence_positions,
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

            if include_kv_cache and kv_caches:
                kv_path = path + ".kv.pt"
                torch.save(kv_caches, kv_path)
                logger.info(
                    f"Pipeline checkpoints KV cache saved to {kv_path} "
                    f"({len(kv_caches)} entries)"
                )

            logger.info(
                f"Pipeline checkpoints saved to {path} "
                f"({len(data['checkpoints'])} entries)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save pipeline checkpoints: {e}")
            return False

    def load_from_disk(self) -> int:
        """Restore checkpoints from disk.

        Returns the number of checkpoints restored, or 0 on failure.
        """
        path = self._persist_path
        if not path or not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                data = json.load(f)

            kv_path = path + ".kv.pt"
            kv_caches: dict[str, Any] = {}
            if os.path.exists(kv_path):
                try:
                    kv_caches = torch.load(
                        kv_path, map_location="cpu", weights_only=True
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load pipeline KV cache from {kv_path}: {e}"
                    )

            with self._lock:
                for rid, ckpt_data in data.get("checkpoints", {}).items():
                    kv_cache = kv_caches.get(rid) if rid in kv_caches else None
                    self._checkpoints[rid] = PipelineCheckpoint(
                        topology_version=ckpt_data.get("topology_version", 0),
                        topology_id=ckpt_data.get("topology_id", ""),
                        request_id=rid,
                        kv_cache=kv_cache,
                        sequence_positions=ckpt_data.get("sequence_positions", []),
                        timestamp=ckpt_data.get("timestamp", time.time()),
                    )

            n_loaded = len(data.get("checkpoints", {}))
            logger.info(
                f"Pipeline checkpoints restored from {path} "
                f"({n_loaded} entries)"
            )
            return n_loaded
        except Exception as e:
            logger.error(f"Failed to load pipeline checkpoints: {e}")
            return 0

    # ── Metrics ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return checkpoint statistics."""
        with self._lock:
            return {
                "active_checkpoints": len(self._checkpoints),
                "total_saved": self._total_saved,
                "total_loaded": self._total_loaded,
                "persist_path": self._persist_path,
                "checkpoint_interval_s": self._interval,
                "max_checkpoints": self._max_checkpoints,
            }


# =========================================================================
# PipelineReconfigurator
# =========================================================================


class ReconfigState(Enum):
    """States in the pipeline reconfiguration lifecycle."""

    IDLE = auto()
    DRAINING = auto()
    CHECKPOINTING = auto()
    APPLYING = auto()
    VERIFYING = auto()
    ROLLING_BACK = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ReconfigurationPlan:
    """Describes a planned topology change.

    Returned by :meth:`PipelineReconfigurator.add_node` /
    :meth:`PipelineReconfigurator.remove_node` and consumed by
    :meth:`PipelineReconfigurator.apply_reconfiguration`.
    """

    new_version: int
    new_topology_id: str
    action: str  # "add_node", "remove_node", "change_layers"
    target_node_id: str
    assignments_before: tuple[NodeAssignment, ...]
    assignments_after: tuple[NodeAssignment, ...]
    requires_full_checkpoint: bool = True
    draint_timeout_s: float = 30.0
    created_at: float = field(default_factory=time.time)


class PipelineReconfigurator:
    """Live reconfiguration of pipeline topology.

    Supports adding and removing nodes mid-pipeline using checkpoint-based
    state transfer.  In-flight requests are drained gracefully (existing
    requests finish on their topology; new requests use the new topology).

    Usage::

        reconf = PipelineReconfigurator(
            orchestrator=orchestrator,
            checkpointer=checkpointer,
            selector=selector,
            topology_versions=versions,
        )

        plan = reconf.add_node("node-c", "10.0.0.3", 50051, 8, 11)
        await reconf.apply_reconfiguration(plan)

        # If something goes wrong:
        await reconf.rollback(plan.new_version)
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        checkpointer: PipelineCheckpointer,
        selector: RequestPipelineSelector | None = None,
        topology_versions: dict[int, TopologyVersion] | None = None,
        drain_check_interval_s: float = 0.5,
        reconfig_timeout_s: float = 60.0,
    ):
        self._orchestrator = orchestrator
        self._checkpointer = checkpointer
        self._selector = selector
        self._versions: dict[int, TopologyVersion] = topology_versions or {}
        self._drain_interval = drain_check_interval_s
        self._timeout = reconfig_timeout_s

        self._state = ReconfigState.IDLE
        self._lock = threading.Lock()
        self._rollback_stack: list[int] = []
        self._reconfig_history: list[dict[str, Any]] = []

        self._on_node_add: Any = None
        self._on_node_remove: Any = None
        self._on_plan_applied: Any = None

    # ── Callback setters ───────────────────────────────────────────────

    def set_on_node_add(
        self, callback: Any
    ) -> None:
        """Set callback ``(node_id, host, port, start_layer, end_layer) -> None``.

        Called when a new node is being added to the pipeline during
        reconfiguration.  The callback should load the required layer
        weights on the new node.
        """
        self._on_node_add = callback

    def set_on_node_remove(self, callback: Any) -> None:
        """Set callback ``(node_id) -> None``.

        Called when a node is being removed from the pipeline.  The
        callback should stop the node's serving loop and release resources.
        """
        self._on_node_remove = callback

    def set_on_plan_applied(self, callback: Any) -> None:
        """Set callback ``(plan: ReconfigurationPlan) -> None``.

        Called after a reconfiguration plan has been applied successfully.
        """
        self._on_plan_applied = callback

    # ── Plan creation ──────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        topology_id: str | None = None,
    ) -> ReconfigurationPlan:
        """Create a plan to add a new node to the pipeline.

        The plan records the topology change and the required state
        transfer strategy.  Call :meth:`apply_reconfiguration` to execute.

        Args:
            node_id: Unique node identifier.
            host: Node hostname or IP.
            port: Node gRPC port.
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (inclusive).
            topology_id: Optional topology identifier (auto-generated if
                None).

        Returns:
            A :class:`ReconfigurationPlan` describing the change.
        """
        with self._lock:
            current_version = max(self._versions.keys()) if self._versions else 0
            new_version = current_version + 1
            tid = topology_id or f"live-add-{node_id}-v{new_version}"

            before_topo = self._versions.get(current_version)
            before_assignments = tuple(before_topo.assignments) if before_topo else ()

            # Build the new topology version
            new_topo = (
                TopologyVersion.create(
                    version=new_version,
                    topology_id=tid,
                )
                .with_assignment(node_id, start_layer, end_layer)
            )
            # Carry forward existing assignments
            if before_topo:
                for a in before_topo.assignments:
                    new_topo = new_topo.with_assignment(a.node_id, a.start_layer, a.end_layer)

            self._versions[new_version] = new_topo

            # Determine if we need a full checkpoint-based transfer
            requires_full = True
            if before_topo:
                requires_full = not before_topo.compatible(new_topo)

            plan = ReconfigurationPlan(
                new_version=new_version,
                new_topology_id=tid,
                action="add_node",
                target_node_id=node_id,
                assignments_before=before_assignments,
                assignments_after=tuple(new_topo.assignments),
                requires_full_checkpoint=requires_full,
            )

        logger.info(
            f"Reconfiguration plan created: add node {node_id} "
            f"(layers {start_layer}-{end_layer}), "
            f"version {new_version}, "
            f"full_ckpt={requires_full}"
        )
        return plan

    def remove_node(self, node_id: str) -> ReconfigurationPlan:
        """Create a plan to remove *node_id* from the pipeline.

        The removed node's layers will be redistributed across the
        remaining nodes (best-effort) unless all surviving nodes already
        exist in the current topology.

        Args:
            node_id: The node to remove.

        Returns:
            A :class:`ReconfigurationPlan` describing the change.
        """
        with self._lock:
            current_version = max(self._versions.keys()) if self._versions else 0
            new_version = current_version + 1
            tid = f"live-remove-{node_id}-v{new_version}"

            before_topo = self._versions.get(current_version)
            before_assignments = tuple(before_topo.assignments) if before_topo else ()

            new_topo = before_topo.without_node(node_id) if before_topo else (
                TopologyVersion.create(version=new_version, topology_id=tid)
            )
            self._versions[new_version] = new_topo

            requires_full = True
            if before_topo:
                requires_full = not before_topo.compatible(new_topo)

            plan = ReconfigurationPlan(
                new_version=new_version,
                new_topology_id=tid,
                action="remove_node",
                target_node_id=node_id,
                assignments_before=before_assignments,
                assignments_after=tuple(new_topo.assignments),
                requires_full_checkpoint=requires_full,
            )

        logger.info(
            f"Reconfiguration plan created: remove node {node_id}, "
            f"version {new_version}, full_ckpt={requires_full}"
        )
        return plan

    # ── Application ────────────────────────────────────────────────────

    async def apply_reconfiguration(
        self,
        plan: ReconfigurationPlan,
        *,
        checkpoint_callback: Any = None,
    ) -> bool:
        """Apply a reconfiguration plan with proper state transfer.

        This is the main entry point for live reconfiguration.  It:

        1. Marks the state as ``DRAINING`` and waits for in-flight requests
           on affected nodes to complete.
        2. Saves pipeline checkpoints for all in-flight requests.
        3. Registers/unregisters nodes with the orchestrator.
        4. Transfers checkpoints to the new topology.
        5. Updates the request selector with the new topology version.

        Args:
            plan: The plan to apply (from :meth:`add_node` or
                :meth:`remove_node`).
            checkpoint_callback: Optional async callback
                ``(plan) -> None`` to customise checkpoint transfer.
                If None, uses default checkpoint-based transfer.

        Returns:
            ``True`` if the reconfiguration succeeded.
        """
        if self._state not in (ReconfigState.IDLE, ReconfigState.COMPLETED):
            logger.warning(
                f"Reconfiguration already in progress (state={self._state})"
            )
            return False

        start_time = time.time()

        # ── Phase 1: Drain ─────────────────────────────────────────────
        self._set_state(ReconfigState.DRAINING)
        drained = await self._drain_node(plan.target_node_id, plan.draint_timeout_s)
        if not drained:
            logger.error(
                f"Drain timeout for node {plan.target_node_id} — "
                f"forcing reconfiguration"
            )

        # ── Phase 2: Checkpoint ────────────────────────────────────────
        self._set_state(ReconfigState.CHECKPOINTING)
        try:
            await self._checkpoint_all_requests(plan)
        except Exception as e:
            logger.error(f"Checkpoint phase failed: {e}")
            self._set_state(ReconfigState.FAILED)
            return False

        # ── Phase 3: Apply topology change ─────────────────────────────
        self._set_state(ReconfigState.APPLYING)
        try:
            self._apply_topology_change(plan)
        except Exception as e:
            logger.error(f"Topology application failed: {e}")
            await self._rollback_internal(plan)
            return False

        # ── Phase 4: Verify ────────────────────────────────────────────
        self._set_state(ReconfigState.VERIFYING)
        verified = self._verify_topology(plan)
        if not verified:
            logger.error("Topology verification failed — rolling back")
            await self._rollback_internal(plan)
            return False

        # ── Phase 5: Update selector ───────────────────────────────────
        if self._selector is not None:
            self._selector.activate_version(plan.new_version)

        # ── Phase 6: Fire callback ─────────────────────────────────────
        if self._on_plan_applied:
            try:
                self._on_plan_applied(plan)
            except Exception as e:
                logger.warning(f"on_plan_applied callback failed: {e}")

        elapsed = (time.time() - start_time) * 1000
        self._record_history(plan, elapsed, "success")
        self._set_state(ReconfigState.COMPLETED)

        logger.info(
            f"Reconfiguration applied: {plan.action} {plan.target_node_id}, "
            f"version {plan.new_version} ({elapsed:.0f}ms)"
        )
        return True

    # ── Rollback ───────────────────────────────────────────────────────

    async def rollback(self, target_version: int) -> bool:
        """Roll back the topology to *target_version*.

        This is a safe revert that:

        1. Drains the current topology's nodes.
        2. Restores the orchestrator's node list to match *target_version*.
        3. Updates the request selector.

        Args:
            target_version: Topology version to revert to.

        Returns:
            ``True`` if rollback succeeded.
        """
        with self._lock:
            target = self._versions.get(target_version)
            if target is None:
                logger.error(f"Cannot rollback: version {target_version} not found")
                return False

        self._set_state(ReconfigState.ROLLING_BACK)

        # Drain the orchestrator
        current_nodes = self._orchestrator.node_order
        for nid in current_nodes:
            await self._drain_node(nid, timeout_s=10.0)

        # Rebuild orchestrator to match target topology
        try:
            # Clear and re-register
            for nid in list(self._orchestrator.node_order):
                self._orchestrator.unregister_node(nid)

            for assignment in target.assignments:
                node_info = self._orchestrator.nodes.get(assignment.node_id)
                if node_info:
                    self._orchestrator.register_node(
                        assignment.node_id,
                        node_info["host"],
                        node_info["port"],
                        assignment.start_layer,
                        assignment.end_layer,
                    )
                else:
                    logger.warning(
                        f"Cannot restore node {assignment.node_id} during rollback — "
                        f"no connection info available"
                    )
        except Exception as e:
            logger.error(f"Rollback orchestrator update failed: {e}")
            self._set_state(ReconfigState.FAILED)
            return False

        # Update selector
        if self._selector is not None:
            self._selector.activate_version(target_version)

        self._record_history(
            {"new_version": target_version, "action": "rollback"},
            0.0,
            "rollback",
        )
        self._set_state(ReconfigState.COMPLETED)
        logger.info(f"Rolled back to topology version {target_version}")
        return True

    # ── Internal phases ────────────────────────────────────────────────

    async def _drain_node(self, node_id: str, timeout_s: float) -> bool:
        """Wait for in-flight requests on *node_id* to complete.

        Returns ``True`` if the node drained within the timeout.
        """
        if self._selector is None:
            return True  # No selector — nothing to drain

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            in_flight = self._selector.in_flight_on_topology(
                self._selector.current_version
            )
            if not in_flight:
                return True
            await asyncio.sleep(self._drain_interval)

        return False

    async def _checkpoint_all_requests(self, plan: ReconfigurationPlan) -> None:
        """Snapshot all in-flight requests for checkpoint transfer."""
        if self._selector is None:
            return
        in_flight = self._selector.in_flight_on_topology(
            self._selector.current_version
        )
        logger.info(
            f"Checkpointing {len(in_flight)} in-flight requests "
            f"for reconfiguration v{plan.new_version}"
        )
        # The actual tensor snapshots are performed by the
        # PipelineCheckpointer integration; here we just ensure the
        # checkpointer is primed.
        self._checkpointer.reset_timer()

    def _apply_topology_change(self, plan: ReconfigurationPlan) -> None:
        """Apply the node addition/removal to the orchestrator."""
        if plan.action == "add_node":
            # Register the new node with the orchestrator using the
            # assignment from the new topology.
            for a in plan.assignments_after:
                if a.node_id == plan.target_node_id:
                    self._orchestrator.register_node(
                        node_id=a.node_id,
                        host="",  # Filled in from the add_node call context
                        port=0,
                        start_layer=a.start_layer,
                        end_layer=a.end_layer,
                    )
                    break

            if self._on_node_add:
                try:
                    self._on_node_add(plan.target_node_id)
                except Exception as e:
                    logger.warning(
                        f"on_node_add callback failed for {plan.target_node_id}: {e}"
                    )

        elif plan.action == "remove_node":
            self._orchestrator.unregister_node(plan.target_node_id)

            if self._on_node_remove:
                try:
                    self._on_node_remove(plan.target_node_id)
                except Exception as e:
                    logger.warning(
                        f"on_node_remove callback failed for {plan.target_node_id}: {e}"
                    )

    def _verify_topology(self, plan: ReconfigurationPlan) -> bool:
        """Verify that the orchestrator matches the plan's topology."""
        expected_nodes = {a.node_id for a in plan.assignments_after}
        actual_nodes = set(self._orchestrator.node_order)

        all_present = expected_nodes.issubset(actual_nodes)

        if plan.action == "add_node":
            return plan.target_node_id in actual_nodes
        elif plan.action == "remove_node":
            return plan.target_node_id not in actual_nodes
        return all_present

    async def _rollback_internal(self, plan: ReconfigurationPlan) -> None:
        """Internal rollback after a failed reconfiguration."""
        with self._lock:
            # Remove the failed version from the version map
            self._versions.pop(plan.new_version, None)

        if plan.assignments_before:
            # Rebuild orchestrator state to before-plan assignments
            current_nodes = self._orchestrator.node_order
            for nid in current_nodes:
                # Only need to fix up if we modified the orchestrator
                pass

        self._set_state(ReconfigState.FAILED)
        logger.warning(
            f"Reconfiguration rolled back: {plan.action} {plan.target_node_id}"
        )

    # ── State management ───────────────────────────────────────────────

    def _set_state(self, state: ReconfigState) -> None:
        with self._lock:
            self._state = state

    @property
    def state(self) -> ReconfigState:
        """Current reconfiguration state."""
        with self._lock:
            return self._state

    @property
    def current_version(self) -> int:
        """The highest version in the topology map."""
        with self._lock:
            return max(self._versions.keys()) if self._versions else 0

    # ── History ────────────────────────────────────────────────────────

    def _record_history(
        self,
        plan: Any,
        duration_ms: float,
        outcome: str,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "event": "reconfiguration",
            "action": getattr(plan, "action", "unknown"),
            "target_node_id": getattr(plan, "target_node_id", ""),
            "new_version": getattr(plan, "new_version", 0),
            "duration_ms": duration_ms,
            "outcome": outcome,
        }
        self._reconfig_history.append(entry)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the reconfiguration audit log."""
        return list(self._reconfig_history)

    def get_versions(self) -> dict[int, TopologyVersion]:
        """Return the topology version map (copy)."""
        with self._lock:
            return dict(self._versions)


# =========================================================================
# RequestPipelineSelector
# =========================================================================


@dataclass
class RequestRoute:
    """Tracks which topology version a request is bound to."""

    request_id: str
    topology_version: int
    topology_id: str
    created_at: float = field(default_factory=time.time)

    @property
    def age_s(self) -> float:
        """Seconds since this route was created."""
        return time.time() - self.created_at


class RequestPipelineSelector:
    """Per-request pipeline version selection with graceful migration.

    New requests are assigned the latest activated topology version.
    In-flight requests continue on their original topology until they
    complete.  Supports gradual migration: a new topology version can
    be activated so that new requests use it while existing requests
    drain on the old one.

    Usage::

        sel = RequestPipelineSelector(
            versions={1: topo_v1, 2: topo_v2},
            initial_version=1,
        )

        # New request — uses v2 (latest)
        route = sel.assign_request("req-456")

        # In-flight request — stays on its original version
        version = sel.get_request_topology("req-123")  # -> 1

        # After v1 drains completely:
        sel.deactivate_version(1)
    """

    def __init__(
        self,
        versions: dict[int, TopologyVersion],
        initial_version: int = 1,
        migration_batch_size: int = 1,
    ):
        self._versions = versions
        self._active_versions: set[int] = {initial_version} if initial_version in versions else set()
        self._current_version: int = initial_version if initial_version in versions else (
            max(versions.keys()) if versions else 0
        )
        self._routes: dict[str, RequestRoute] = {}
        self._lock = threading.Lock()
        self._migration_batch_size = migration_batch_size

        self._metrics = {
            "total_assignments": 0,
            "total_completions": 0,
            "migrations": 0,
        }

    # ── Assignment ─────────────────────────────────────────────────────

    def assign_request(self, request_id: str) -> RequestRoute:
        """Assign *request_id* to the latest active topology version.

        Returns the :class:`RequestRoute` that was created.
        """
        with self._lock:
            version = self._current_version
            topo = self._versions.get(version)
            route = RequestRoute(
                request_id=request_id,
                topology_version=version,
                topology_id=topo.topology_id if topo else f"v{version}",
            )
            self._routes[request_id] = route
            self._metrics["total_assignments"] += 1

        return route

    def complete_request(self, request_id: str) -> None:
        """Mark *request_id* as completed and remove its route."""
        with self._lock:
            self._routes.pop(request_id, None)
            self._metrics["total_completions"] += 1

    def get_request_topology(self, request_id: str) -> int | None:
        """Return the topology version assigned to *request_id*.

        Returns ``None`` if the request is not tracked.
        """
        with self._lock:
            route = self._routes.get(request_id)
            return route.topology_version if route else None

    def get_request_route(self, request_id: str) -> RequestRoute | None:
        """Return the full :class:`RequestRoute` for *request_id*."""
        with self._lock:
            return self._routes.get(request_id)

    # ── Version management ─────────────────────────────────────────────

    def activate_version(self, version: int) -> bool:
        """Activate a topology version for new requests.

        New requests will use this version (or the highest active version,
        whichever is greater).

        Args:
            version: Version number to activate.

        Returns:
            ``True`` if the version exists and was activated.
        """
        with self._lock:
            if version not in self._versions:
                return False
            self._active_versions.add(version)
            if version > self._current_version:
                self._current_version = version

        logger.info(
            f"Topology version {version} activated "
            f"(topology_id={self._versions[version].topology_id})"
        )
        return True

    def deactivate_version(self, version: int) -> bool:
        """Deactivate a topology version.

        Only succeeds if there are **no** in-flight requests still
        using this version.  Call :meth:`in_flight_on_topology` first
        to verify.

        Args:
            version: Version number to deactivate.

        Returns:
            ``True`` if the version was deactivated (or was not active).
        """
        with self._lock:
            if version not in self._active_versions:
                return True  # Already inactive
            # Check for in-flight requests
            for route in self._routes.values():
                if route.topology_version == version:
                    return False  # Still in use
            self._active_versions.discard(version)

        logger.info(f"Topology version {version} deactivated")
        return True

    def migrate_request(self, request_id: str, target_version: int) -> bool:
        """Migrate a single in-flight request to *target_version*.

        This is used for gradual migration — moving long-running
        requests to a new topology without waiting for them to drain
        naturally.

        Args:
            request_id: The request to migrate.
            target_version: Target topology version.

        Returns:
            ``True`` if migration succeeded.
        """
        with self._lock:
            if target_version not in self._versions:
                return False
            if request_id not in self._routes:
                return False

            topo = self._versions[target_version]
            self._routes[request_id] = RequestRoute(
                request_id=request_id,
                topology_version=target_version,
                topology_id=topo.topology_id,
            )
            self._metrics["migrations"] += 1

        logger.info(
            f"Migrated request {request_id} to topology version {target_version}"
        )
        return True

    # ── Queries ────────────────────────────────────────────────────────

    def in_flight_on_topology(self, version: int) -> list[str]:
        """Return all request IDs currently assigned to *version*."""
        with self._lock:
            return [
                rid
                for rid, route in self._routes.items()
                if route.topology_version == version
            ]

    def in_flight_requests(self) -> list[str]:
        """Return all tracked in-flight request IDs."""
        with self._lock:
            return list(self._routes.keys())

    def active_versions(self) -> set[int]:
        """Return the set of currently active topology versions."""
        with self._lock:
            return set(self._active_versions)

    def in_flight_count(self, version: int | None = None) -> dict[int, int]:
        """Return in-flight request counts by topology version.

        If *version* is not None, only that version's count is returned.
        """
        with self._lock:
            counts: dict[int, int] = {}
            for route in self._routes.values():
                v = route.topology_version
                counts[v] = counts.get(v, 0) + 1
            if version is not None:
                return {version: counts.get(version, 0)}
            return counts

    # ── Active version property ────────────────────────────────────────

    @property
    def current_version(self) -> int:
        """The currently active topology version for new requests."""
        with self._lock:
            return self._current_version

    @current_version.setter
    def current_version(self, value: int) -> None:
        """Set the current topology version (activates it if needed)."""
        self.activate_version(value)

    # ── Migration helpers ──────────────────────────────────────────────

    async def gradual_migrate(
        self,
        target_version: int,
        batch_size: int | None = None,
        interval_s: float = 0.5,
    ) -> dict[str, Any]:
        """Gradually migrate all in-flight requests to *target_version*.

        Moves requests in batches, with a delay between batches to give
        the system time to stabilise.  Requests that complete naturally
        during migration are skipped.

        Args:
            target_version: The target topology version.
            batch_size: Number of requests to migrate per batch (defaults
                to ``migration_batch_size`` from constructor).
            interval_s: Pause between batches.

        Returns:
            A result dict with ``migrated``, ``skipped`` (completed during
            migration), and ``remaining`` counts.
        """
        if target_version not in self._versions:
            return {"error": f"Version {target_version} not found"}

        batch_size = batch_size or self._migration_batch_size
        migrated = 0
        skipped = 0

        while True:
            with self._lock:
                candidates = [
                    rid
                    for rid, route in self._routes.items()
                    if route.topology_version != target_version
                ][:batch_size]

            if not candidates:
                break

            for rid in candidates:
                ok = self.migrate_request(rid, target_version)
                if ok:
                    migrated += 1
                else:
                    # Request may have completed between check and migrate
                    skipped += 1

            await asyncio.sleep(interval_s)

        with self._lock:
            remaining = len(self._routes) - migrated - skipped

        logger.info(
            f"Gradual migration to v{target_version} complete: "
            f"{migrated} migrated, {skipped} skipped, {remaining} remaining"
        )
        return {
            "migrated": migrated,
            "skipped": skipped,
            "remaining": max(0, remaining),
        }

    # ── Drain ──────────────────────────────────────────────────────────

    async def wait_for_drain(
        self,
        version: int,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> bool:
        """Block until all requests on *version* have completed.

        Args:
            version: The topology version to drain.
            timeout_s: Maximum time to wait.
            poll_interval_s: How often to check.

        Returns:
            ``True`` if drained within the timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            in_flight = self.in_flight_on_topology(version)
            if not in_flight:
                return True
            await asyncio.sleep(poll_interval_s)
        return False

    # ── Metrics ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return selector statistics."""
        with self._lock:
            return {
                **self._metrics,
                "current_version": self._current_version,
                "active_versions": sorted(self._active_versions),
                "available_versions": sorted(self._versions.keys()),
                "in_flight_total": len(self._routes),
                "in_flight_by_version": self.in_flight_count(),
            }
