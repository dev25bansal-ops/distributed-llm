"""Tenant-aware load balancing for DaaS node selection.

Routes requests to nodes considering tenant affinity — prefers nodes where
a tenant's models are already loaded to avoid cold-start overhead, and
falls back to the least-loaded node within tenant constraints.

Usage::

    lb = TenantAwareLoadBalancer()

    # Register nodes with their capacities.
    lb.register_node("node-1", capacity=100, loaded_tenants={"tenant-a"})
    lb.register_node("node-2", capacity=200, loaded_tenants={"tenant-b"})

    # Select the best node for a tenant.
    best = lb.select_node("tenant-a", candidates=["node-1", "node-2"])
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class NodeInfo:
    """Information about a candidate node."""

    node_id: str
    capacity: float = 100.0        # max load units (e.g., request slots)
    current_load: float = 0.0      # current load units
    loaded_tenants: set[str] = field(default_factory=set)
    affinity_bonus: float = 0.0    # dynamic bonus from recent assignments


@dataclass
class NodeScore:
    """Scoring result for a single node candidate."""

    node_id: str
    raw_score: float
    adjusted_score: float
    has_affinity: bool
    reasons: list[str] = field(default_factory=list)


class LoadSource(Protocol):
    """Protocol for querying current node load."""

    def get_load(self, node_id: str) -> float: ...


class TenantAwareLoadBalancer:
    """Routes requests to nodes considering tenant affinity and current load.

    Scoring (higher is better):
    1. Tenant affinity: nodes that already have the tenant's models loaded
       get a significant score boost (avoids cold-start).
    2. Least-loaded: among nodes with equal affinity, the least-loaded node
       is preferred.
    3. Deterministic tie-breaking via node_id.

    Thread-safe.
    """

    def __init__(self, affinity_weight: float = 100.0, load_weight: float = 1.0) -> None:
        """
        Args:
            affinity_weight:  Score bonus added when a node has tenant affinity.
            load_weight:      Weight applied to ``(1 - utilization)`` when scoring.
        """
        self._nodes: dict[str, NodeInfo] = {}
        self._affinity_weight = affinity_weight
        self._load_weight = load_weight
        self._lock = threading.Lock()
        self._load_source: LoadSource | None = None

    def set_load_source(self, source: LoadSource | None) -> None:
        """Set an external load query source.

        When set, ``select_node()`` uses this to get per-node load instead
        of the internal counter.
        """
        self._load_source = source

    # ── Node management ───────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        capacity: float = 100.0,
        loaded_tenants: set[str] | None = None,
    ) -> None:
        """Register or update a candidate node.

        Args:
            node_id:        Unique node identifier.
            capacity:       Maximum load the node can handle.
            loaded_tenants: Set of tenant IDs whose models are already
                            loaded on this node.
        """
        with self._lock:
            info = self._nodes.setdefault(node_id, NodeInfo(node_id=node_id))
            info.capacity = capacity
            if loaded_tenants is not None:
                info.loaded_tenants = loaded_tenants

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from consideration."""
        with self._lock:
            self._nodes.pop(node_id, None)

    def record_assignment(self, tenant_id: str, node_id: str) -> None:
        """Record that *tenant_id* was assigned to *node_id*.

        This bumps the node's load and adds the tenant to its loaded set
        (so future requests from the same tenant benefit from affinity).
        """
        with self._lock:
            info = self._nodes.get(node_id)
            if info is None:
                return
            info.loaded_tenants.add(tenant_id)
            info.current_load += 1.0
            info.affinity_bonus += self._affinity_weight * 0.1  # decayed bonus

    def release_assignment(self, tenant_id: str, node_id: str) -> None:
        """Release a previous assignment, decrementing node load."""
        with self._lock:
            info = self._nodes.get(node_id)
            if info is None:
                return
            info.current_load = max(0.0, info.current_load - 1.0)

    # ── Scoring ───────────────────────────────────────────────────────

    def _score_node(
        self,
        node_id: str,
        info: NodeInfo,
        tenant_id: str,
    ) -> NodeScore:
        """Compute a score for *node_id* for *tenant_id*.

        Score components:
        - ``affinity_component``: ``affinity_weight`` if the node has the
          tenant's models loaded, plus any accumulated assignment bonus.
        - ``load_component``: ``load_weight * (1 - utilization)`` where
          utilization is ``current_load / capacity``.

        Returns a ``NodeScore``.
        """
        has_affinity = tenant_id in info.loaded_tenants
        affinity_component = self._affinity_weight if has_affinity else 0.0
        affinity_component += info.affinity_bonus

        utilization = info.current_load / max(info.capacity, 1.0)
        load_component = self._load_weight * (1.0 - min(utilization, 1.0))

        reasons: list[str] = []
        if has_affinity:
            reasons.append(f"tenant affinity (+{affinity_component:.1f})")
        reasons.append(f"load {info.current_load:.0f}/{info.capacity:.0f} "
                       f"(load_component={load_component:.2f})")

        raw_score = affinity_component + load_component
        adjusted_score = raw_score

        return NodeScore(
            node_id=node_id,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            has_affinity=has_affinity,
            reasons=reasons,
        )

    def select_node(
        self,
        tenant_id: str,
        candidates: list[str] | None = None,
    ) -> str | None:
        """Select the best node for *tenant_id* from *candidates*.

        If *candidates* is ``None``, all registered nodes are considered.
        Returns the node ID with the highest score, or ``None`` if no
        nodes are available.

        Args:
            tenant_id:  The tenant to route.
            candidates: Optional subset of node IDs to consider.

        Returns:
            The selected node ID, or ``None``.
        """
        with self._lock:
            if candidates is None:
                candidates = list(self._nodes.keys())

            if not candidates:
                return None

            # Optionally refresh load from external source.
            if self._load_source is not None:
                for node_id in candidates:
                    info = self._nodes.get(node_id)
                    if info is not None:
                        try:
                            info.current_load = self._load_source.get_load(node_id)
                        except Exception:
                            pass  # fall back to internal counter

            best_score: float = -1.0
            best_node: str | None = None

            for node_id in candidates:
                info = self._nodes.get(node_id)
                if info is None:
                    continue
                score = self._score_node(node_id, info, tenant_id)
                if score.adjusted_score > best_score:
                    best_score = score.adjusted_score
                    best_node = node_id

            return best_node

    # ── Inspection ────────────────────────────────────────────────────

    def get_node_info(self, node_id: str) -> NodeInfo | None:
        """Return info for a registered node, or ``None``."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_scores(
        self,
        tenant_id: str,
        candidates: list[str] | None = None,
    ) -> list[NodeScore]:
        """Score all (or given) nodes for *tenant_id* without selecting.

        Useful for observability and debugging.
        """
        with self._lock:
            if candidates is None:
                candidates = list(self._nodes.keys())

            scores: list[NodeScore] = []
            for node_id in candidates:
                info = self._nodes.get(node_id)
                if info is None:
                    continue
                scores.append(self._score_node(node_id, info, tenant_id))

            scores.sort(key=lambda s: s.adjusted_score, reverse=True)
            return scores

    def stats(self) -> dict[str, Any]:
        """Return statistics about registered nodes."""
        with self._lock:
            per_node = {}
            for node_id, info in self._nodes.items():
                utilization = info.current_load / max(info.capacity, 1.0)
                per_node[node_id] = {
                    "capacity": info.capacity,
                    "current_load": info.current_load,
                    "utilization_pct": round(utilization * 100.0, 1),
                    "loaded_tenants": len(info.loaded_tenants),
                    "affinity_bonus": round(info.affinity_bonus, 2),
                }
            return {
                "nodes": len(self._nodes),
                "affinity_weight": self._affinity_weight,
                "load_weight": self._load_weight,
                "per_node": per_node,
            }
