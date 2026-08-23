"""Geo-aware cross-cluster routing for federated LLM inference.

Supports A/B model deployments via traffic-split weights so that a
fraction of requests can be routed to a new model version for gradual
rollouts, canary testing, or blue/green deployments.

Usage::

    router = GeoRouter(federation, latency_monitor)

    # Route 10% of requests to model-v2, 90% to model-v1
    router.set_traffic_split("model-v2", 0.10, deployment_id="canary-042")

    # select_target_cluster automatically accounts for traffic splits.
    target, reason, meta = router.select_target_cluster(
        request="...", source_cluster="us-east", model_version="model-v2",
    )
    # meta["deployment"]["version"] → "model-v2"
    # meta["deployment"]["rollout_pct"] → 10
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ClusterLoad:
    cluster_id: str
    active_requests: int = 0
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def is_overloaded(self) -> bool:
        return self.gpu_utilization > 0.85 or self.queue_depth > 50

    def is_overloaded_at(self, threshold: float) -> bool:
        queue_threshold = int(threshold * 100)
        return self.gpu_utilization > threshold or self.queue_depth > queue_threshold

    @property
    def available_capacity(self) -> float:
        load = max(self.gpu_utilization, self.queue_depth / 100.0)
        return max(0.0, 1.0 - min(load, 1.0))


class LoadReporter:
    def __init__(self, stale_threshold_s: float = 30.0):
        self._loads: dict[str, ClusterLoad] = {}
        self._stale_threshold = stale_threshold_s
        self._lock = threading.Lock()

    def report(self, cluster_id: str, active: int = 0, pending: int = 0,
               gpu_util: float = 0.0, queue_depth: int = 0) -> None:
        with self._lock:
            load = self._loads.get(cluster_id)
            if load is None:
                load = ClusterLoad(cluster_id=cluster_id)
                self._loads[cluster_id] = load
            load.active_requests = active
            load.pending_requests = pending
            load.gpu_utilization = gpu_util
            load.queue_depth = queue_depth
            load.last_updated = time.time()

    def get_load(self, cluster_id: str) -> ClusterLoad | None:
        with self._lock:
            load = self._loads.get(cluster_id)
            if load is None:
                return None
            if time.time() - load.last_updated > self._stale_threshold:
                logger.warning(f"Stale load data for {cluster_id} ({time.time() - load.last_updated:.0f}s old)")
                return None
            return load

    def get_all_loads(self) -> dict[str, ClusterLoad]:
        with self._lock:
            return dict(self._loads)

    def remove_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._loads.pop(cluster_id, None)


@dataclass
class TrafficSplit:
    """A single A/B traffic-split rule.

    Attributes:
        version: Model version identifier (e.g. ``"model-v2"``).
        weight: Fraction of traffic to route to this version (0.0 – 1.0).
        deployment_id: Human-readable label for observability
            (e.g. ``"canary-042"``, ``"blue"``).
        created_at: Timestamp when this split was created.
    """
    version: str
    weight: float
    deployment_id: str = ""
    created_at: float = field(default_factory=time.time)


class GeoRouter:
    def __init__(
        self,
        federation: Any,
        latency_monitor: Any,
        load_reporter: LoadReporter | None = None,
        local_latency_threshold_ms: float = 50.0,
        spill_threshold: float = 0.85,
    ):
        self._federation = federation
        self._latency_monitor = latency_monitor
        self._load_reporter = load_reporter or LoadReporter()
        self._local_latency_threshold = local_latency_threshold_ms
        self._spill_threshold = spill_threshold
        self._lock = threading.Lock()

        # A/B deployment state: model_version → TrafficSplit
        self._traffic_splits: dict[str, TrafficSplit] = {}
        # Per-version routing destinations: model_version → cluster_id
        self._version_routes: dict[str, str] = {}
        # A/B metrics
        self._ab_metrics: dict[str, dict[str, float]] = {}

    # ── A/B traffic split API ─────────────────────────────────────────

    def set_traffic_split(
        self,
        version: str,
        weight: float,
        deployment_id: str = "",
    ) -> None:
        """Route *weight* fraction of requests to *version*.

        The remaining ``1 - weight`` stays on the default (existing)
        model version.  The weight must be in ``[0.0, 1.0]``.

        A weight of ``0.0`` removes the split (routes 100 % to default).
        A weight of ``1.0`` fully migrates to the new version.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Traffic split weight must be in [0, 1], got {weight}")

        with self._lock:
            if weight == 0.0:
                self._traffic_splits.pop(version, None)
                self._version_routes.pop(version, None)
                logger.info(f"Traffic split removed for {version}")
            else:
                self._traffic_splits[version] = TrafficSplit(
                    version=version,
                    weight=weight,
                    deployment_id=deployment_id,
                )
                logger.info(
                    f"Traffic split set: {version} = {weight:.0%} "
                    f"(deployment={deployment_id or 'unnamed'})"
                )

    def set_version_route(self, version: str, cluster_id: str) -> None:
        """Pin *version* to a specific cluster (for blue/green deployments).

        When set, :meth:`select_target_cluster` with *model_version*
        routes to this cluster directly, bypassing geo-latency scoring.
        """
        with self._lock:
            self._version_routes[version] = cluster_id
            logger.info(f"Version route: {version} → cluster {cluster_id}")

    def get_traffic_splits(self) -> dict[str, dict[str, Any]]:
        """Return active traffic splits for observability."""
        with self._lock:
            return {
                v: {
                    "weight": s.weight,
                    "deployment_id": s.deployment_id,
                    "created_at": s.created_at,
                    "pinned_cluster": self._version_routes.get(v),
                }
                for v, s in self._traffic_splits.items()
            }

    def record_ab_metric(
        self,
        version: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a metric for A/B comparison (latency, error rate, etc.)."""
        with self._lock:
            if version not in self._ab_metrics:
                self._ab_metrics[version] = {}
            old = self._ab_metrics[version].get(metric_name, 0.0)
            self._ab_metrics[version][metric_name] = old * 0.9 + value * 0.1

    def get_ab_metrics(self) -> dict[str, dict[str, float]]:
        """Return A/B comparison metrics (EMA-smoothed)."""
        with self._lock:
            return {k: dict(v) for k, v in self._ab_metrics.items()}

    def _select_version(self, model_version: str | None) -> str | None:
        """Determine which model version to use based on traffic splits.

        Returns the selected version identifier, or ``None`` for the
        default (existing) version.
        """
        if model_version is None:
            return None

        with self._lock:
            split = self._traffic_splits.get(model_version)
            if split is None:
                return model_version  # no split configured — use as-is

            # Probabilistic routing: roll the dice, compare to weight.
            if random.random() < split.weight:
                return split.version
            return None  # default version

    def select_target_cluster(
        self,
        request: str = "",
        source_cluster: str = "default",
        model_version: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Select the best target cluster, optionally accounting for A/B.

        Args:
            request: Opaque request identifier (for future content-based routing).
            source_cluster: Cluster making the request.
            model_version: Model version to route.  When set, traffic
                splits are applied.

        Returns:
            ``(cluster_id, reason, metadata)`` where metadata includes
            ``deployment`` info when A/B routing is active.
        """
        selected_version = self._select_version(model_version)
        meta: dict[str, Any] = {
            "model_version": selected_version,
            "deployment": None,
        }

        # When A/B is active, tag the response with deployment metadata.
        if selected_version is not None and model_version != selected_version:
            with self._lock:
                split = self._traffic_splits.get(selected_version)
                if split:
                    meta["deployment"] = {
                        "version": split.version,
                        "rollout_pct": round(split.weight * 100),
                        "deployment_id": split.deployment_id,
                    }

        # If the version is pinned to a specific cluster, route there directly.
        if selected_version is not None:
            with self._lock:
                pinned = self._version_routes.get(selected_version)
                if pinned:
                    return pinned, f"version_pinned:{selected_version}", meta

        # Normal geo-routing (existing logic, unchanged).
        local_load = self._load_reporter.get_load(source_cluster)
        if local_load is not None and not local_load.is_overloaded:
            return source_cluster, "local_capacity", meta

        all_clusters = self._federation.list_clusters()
        candidate_clusters = [c for c in all_clusters if c != source_cluster]

        if not candidate_clusters:
            return source_cluster, "no_alternative", meta

        best_cluster = None
        best_score = float('inf')
        best_reason = "latency_fallback"

        for cluster_id in candidate_clusters:
            load = self._load_reporter.get_load(cluster_id)
            latency = self._latency_monitor.get_latency(source_cluster, cluster_id)

            if load is not None and load.is_overloaded:
                continue

            capacity = load.available_capacity if load is not None else 0.5
            score = latency * (1.0 / max(capacity, 0.01))

            if score < best_score:
                best_score = score
                best_cluster = cluster_id
                best_reason = f"nearest_with_capacity (latency={latency:.0f}ms, capacity={capacity:.0%})"

        if best_cluster is None:
            return source_cluster, "all_remote_overloaded", meta

        return best_cluster, best_reason, meta

    def select_edge_node(
        self,
        target_cluster: str,
        source_cluster: str = "default",
    ) -> str | None:
        edge_nodes = self._federation.get_edge_nodes(target_cluster)
        if not edge_nodes:
            edge_nodes = self._federation.get_nodes_in_cluster(target_cluster)

        if not edge_nodes:
            return None

        if len(edge_nodes) == 1:
            return next(iter(edge_nodes))

        loads = []
        for node_id in edge_nodes:
            load = self._load_reporter.get_load(target_cluster)
            if load is not None:
                loads.append((hash(node_id) % 100, node_id))
            else:
                loads.append((0, node_id))

        loads.sort(key=lambda x: x[0])
        return loads[0][1]

    def get_cluster_load(self, cluster_id: str) -> ClusterLoad | None:
        return self._load_reporter.get_load(cluster_id)

    def report_cluster_load(self, cluster_id: str, active: int = 0,
                            pending: int = 0, gpu_util: float = 0.0,
                            queue_depth: int = 0) -> None:
        self._load_reporter.report(
            cluster_id=cluster_id,
            active=active,
            pending=pending,
            gpu_util=gpu_util,
            queue_depth=queue_depth,
        )

    def stats(self) -> dict:
        loads = self._load_reporter.get_all_loads()
        return {
            "cluster_loads": {
                cid: {
                    "active": l.active_requests,
                    "pending": l.pending_requests,
                    "gpu_util": l.gpu_utilization,
                    "queue_depth": l.queue_depth,
                    "overloaded": l.is_overloaded,
                }
                for cid, l in loads.items()
            },
            "latency_matrix": getattr(self._latency_monitor, 'stats', lambda: {})(),
            "federation": getattr(self._federation, 'stats', lambda: {})(),
        }
