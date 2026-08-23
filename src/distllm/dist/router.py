"""Intelligent request routing engine with pluggable policies.

Routes each request to the optimal node based on:
- Model size loaded on each node
- Current load / queue depth
- KV cache locality (has this prefix been cached?)
- Network latency to the node
- GPU memory availability
- Historical performance (reputation)

Architecture::

    Request
       │
       ▼
    ┌─────────────────────┐
    │  RoutingPolicy      │  ← pluggable, composite
    │  ├─ LoadPolicy      │
    │  ├─ LocalityPolicy  │
    │  ├─ LatencyPolicy   │
    │  ├─ MemoryPolicy    │
    │  └─ ReputationPolicy│
    └─────────┬───────────┘
              │
              ▼
        selected_node_id
              │
              ▼
        PipelineOrchestrator

Usage::

    router = RequestRouter()
    router.add_policy(LoadPolicy(weight=0.4))
    router.add_policy(LocalityPolicy(weight=0.3))
    router.add_policy(LatencyPolicy(weight=0.2))
    router.add_policy(MemoryPolicy(weight=0.1))

    node = router.route("req-1", candidates=["node-0","node-1","node-2"],
                         context={"prefix_hash": "abc...", "model": "llama-8b"})
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class NodeScore:
    """Scored routing candidate."""
    node_id: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class RouteContext:
    """Context passed to each policy for scoring."""
    request_id: str = ""
    model_name: str = ""
    prefix_hash: str = ""          # KV cache prefix hash for locality
    input_length: int = 0
    max_tokens: int = 0
    tenant_id: str = ""
    adapter_id: str | None = None


class RoutingPolicy(abc.ABC):
    """Abstract base for a routing policy.

    Each policy scores candidate nodes from 0.0 (worst) to 1.0 (best)
    for a specific dimension.
    """

    def __init__(self, weight: float = 1.0, name: str = ""):
        self.weight = weight
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        """Score a single candidate node.  Higher = better match."""
        ...


class LoadPolicy(RoutingPolicy):
    """Score by current load fraction.  Lower load = higher score.

    Uses ``gpu_utilization`` and ``queue_depth`` from node_info.
    """

    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        gpu_util = node_info.get("gpu_utilization", 0.0)  # 0-100
        queue = node_info.get("queue_depth", 0)
        load = min(gpu_util / 100.0 + queue * 0.1, 1.0)
        return max(0.0, 1.0 - load)


class LocalityPolicy(RoutingPolicy):
    """Score by KV cache locality.  Cache hit = higher score.

    Uses ``cache_affinity`` (0.0-1.0) from node_info, set by the
    CacheDigestManager when a node has cached prefixes.
    """

    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        if not context.prefix_hash:
            return 0.5  # neutral — no way to know
        affinity = node_info.get("cache_affinity", 0.0)
        matched = node_info.get("matched_length", 0)
        if matched > 0 and affinity > 0.3:
            return min(affinity * 1.2, 1.0)
        return 0.3  # cold cache penalty


class LatencyPolicy(RoutingPolicy):
    """Score by historical latency.  Faster = higher score.

    Uses ``avg_latency_ms`` from node_info.  Normalises relative
    to the best candidate in the set.
    """

    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        latency = node_info.get("avg_latency_ms", 50.0)
        if latency <= 0:
            return 0.5
        # 10ms = 1.0, 100ms = 0.5, 1000ms = 0.09
        return max(0.0, min(1.0, 100.0 / (latency + 10.0)))


class MemoryPolicy(RoutingPolicy):
    """Score by available GPU memory.  More free = higher score.

    Uses ``free_memory_bytes`` and ``total_memory_bytes`` from node_info.
    """

    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        free = node_info.get("free_memory_bytes", 0)
        total = node_info.get("total_memory_bytes", 1)
        if total <= 0:
            return 0.5
        ratio = free / total
        return max(0.0, min(1.0, ratio * 2.0))  # 50% free = 1.0


class ReputationPolicy(RoutingPolicy):
    """Score by historical reliability.  Higher reputation = higher score.

    Uses ``reputation`` (0.0-1.0) from node_info, sourced from
    ``ReputationSystem.get_score()``.
    """

    def score_node(
        self,
        node_id: str,
        context: RouteContext,
        node_info: dict[str, Any],
    ) -> float:
        rep = node_info.get("reputation", 0.5)
        return max(0.0, min(1.0, rep))


class RequestRouter:
    """Intelligent request router with pluggable policies.

    Maintains a node capability registry with live metrics and routes
    requests using weighted policy scoring.
    """

    def __init__(self) -> None:
        self._policies: list[RoutingPolicy] = []
        self._nodes: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Default policies with sensible weights
        self._default_policies = [
            LoadPolicy(weight=0.35),
            LocalityPolicy(weight=0.25),
            LatencyPolicy(weight=0.20),
            MemoryPolicy(weight=0.10),
            ReputationPolicy(weight=0.10),
        ]

    # ── Policy management ──────────────────────────────────────────────

    def add_policy(self, policy: RoutingPolicy) -> None:
        """Add a routing policy."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.weight, reverse=True)

    def set_policies(self, policies: list[RoutingPolicy]) -> None:
        """Replace all policies."""
        self._policies = list(policies)
        self._policies.sort(key=lambda p: p.weight, reverse=True)

    def reset_policies(self) -> None:
        """Restore default policies."""
        self._policies = list(self._default_policies)

    def list_policies(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "weight": p.weight}
            for p in (self._policies or self._default_policies)
        ]

    # ── Node capability registry ───────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        *,
        model_name: str = "",
        gpu_name: str = "",
        total_memory_bytes: int = 0,
        free_memory_bytes: int = 0,
        gpu_utilization: float = 0.0,
        queue_depth: int = 0,
        avg_latency_ms: float = 50.0,
        reputation: float = 0.5,
        cache_affinity: float = 0.0,
        matched_length: int = 0,
        **extra: Any,
    ) -> None:
        """Register or update a node's capability info."""
        with self._lock:
            self._nodes.setdefault(node_id, {})
            self._nodes[node_id].update(
                model_name=model_name,
                gpu_name=gpu_name,
                total_memory_bytes=total_memory_bytes,
                free_memory_bytes=free_memory_bytes,
                gpu_utilization=gpu_utilization,
                queue_depth=queue_depth,
                avg_latency_ms=avg_latency_ms,
                reputation=reputation,
                cache_affinity=cache_affinity,
                matched_length=matched_length,
                last_updated=time.time(),
                **extra,
            )

    def update_node_metric(self, node_id: str, **metrics: Any) -> None:
        """Update live metrics for a node without full re-registration."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].update(**metrics)
                self._nodes[node_id]["last_updated"] = time.time()

    def get_node_info(self, node_id: str) -> dict[str, Any]:
        """Get capability info for a node."""
        with self._lock:
            return dict(self._nodes.get(node_id, {}))

    def list_nodes(self) -> list[dict[str, Any]]:
        """List all registered nodes with their info."""
        with self._lock:
            return [
                {"node_id": nid, **info}
                for nid, info in self._nodes.items()
            ]

    # ── Routing ────────────────────────────────────────────────────────

    def route(
        self,
        request_id: str,
        candidate_nodes: list[str] | None = None,
        context: RouteContext | None = None,
    ) -> tuple[str, NodeScore] | None:
        """Score all candidate nodes and return the best one.

        Args:
            request_id: Request identifier (for tracing).
            candidate_nodes: Nodes to consider.  ``None`` = all registered.
            context: Routing context (model, prefix hash, etc.).

        Returns:
            ``(best_node_id, NodeScore)`` or ``None`` if no candidates.
        """
        policies = self._policies or self._default_policies
        ctx = context or RouteContext(request_id=request_id)

        with self._lock:
            candidates = candidate_nodes or list(self._nodes.keys())
            if not candidates:
                return None

            scores: list[NodeScore] = []
            for nid in candidates:
                info = self._nodes.get(nid, {})
                total = 0.0
                reasons: list[str] = []
                for policy in policies:
                    try:
                        s = policy.score_node(nid, ctx, info)
                        total += s * policy.weight
                        reasons.append(f"{policy.name}={s:.2f}")
                    except Exception:
                        logger.debug(f"Policy {policy.name} failed for {nid}")

                scores.append(NodeScore(
                    node_id=nid,
                    score=total,
                    reasons=reasons,
                ))

        # Return the highest-scoring node
        scores.sort(key=lambda s: s.score, reverse=True)
        best = scores[0]
        logger.debug(
            f"Routed {request_id} to {best.node_id} "
            f"(score={best.score:.3f}, {', '.join(best.reasons[:3])})"
        )
        return best.node_id, best

    def route_with_fallback(
        self,
        request_id: str,
        candidate_nodes: list[str] | None = None,
        context: RouteContext | None = None,
        fallback: str | None = None,
    ) -> str:
        """Route like :meth:`route` but return *fallback* on failure."""
        result = self.route(request_id, candidate_nodes, context)
        if result is None:
            return fallback or (candidate_nodes[0] if candidate_nodes else "unknown")
        return result[0]

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policies": self.list_policies(),
                "registered_nodes": len(self._nodes),
                "nodes": {
                    nid: {
                        "model": info.get("model_name", ""),
                        "gpu": info.get("gpu_name", ""),
                        "gpu_util": info.get("gpu_utilization", 0),
                        "free_mem_mb": (info.get("free_memory_bytes", 0) // (1024 * 1024)),
                        "queue": info.get("queue_depth", 0),
                        "latency_ms": info.get("avg_latency_ms", 0),
                        "rep": info.get("reputation", 0),
                    }
                    for nid, info in self._nodes.items()
                },
            }
