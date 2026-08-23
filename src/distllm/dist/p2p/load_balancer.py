"""Federation load balancer for cross-cluster load reporting.

Collects load metrics from remote clusters via heartbeat and
integrates with the existing LoadReporter in geo_router.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class RemoteClusterLoad:
    cluster_id: str
    active_requests: int = 0
    pending_requests: int = 0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    tokens_per_sec: float = 0.0
    last_report: float = 0.0
    stale: bool = False

    @property
    def is_overloaded(self) -> bool:
        return self.gpu_utilization > 85.0 or self.queue_depth > 50

    @property
    def available_capacity(self) -> float:
        if self.is_overloaded:
            return 0.0
        return max(0.0, 100.0 - self.gpu_utilization)

    def is_stale(self, threshold_s: float = 30.0) -> bool:
        return (time.time() - self.last_report) > threshold_s


class FederationLoadBalancer:
    def __init__(
        self,
        stale_threshold_s: float = 30.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.stale_threshold_s = stale_threshold_s
        self.ema_alpha = ema_alpha
        self._loads: dict[str, RemoteClusterLoad] = {}

    def report_load(
        self,
        cluster_id: str,
        active_requests: int,
        pending_requests: int,
        gpu_utilization: float,
        queue_depth: int,
        tokens_per_sec: float = 0.0,
    ) -> None:
        now = time.time()

        if cluster_id not in self._loads:
            self._loads[cluster_id] = RemoteClusterLoad(
                cluster_id=cluster_id,
                active_requests=active_requests,
                pending_requests=pending_requests,
                gpu_utilization=gpu_utilization,
                queue_depth=queue_depth,
                tokens_per_sec=tokens_per_sec,
                last_report=now,
            )
            return

        load = self._loads[cluster_id]
        alpha = self.ema_alpha

        load.active_requests = alpha * active_requests + (1 - alpha) * load.active_requests
        load.pending_requests = alpha * pending_requests + (1 - alpha) * load.pending_requests
        load.gpu_utilization = alpha * gpu_utilization + (1 - alpha) * load.gpu_utilization
        load.queue_depth = alpha * queue_depth + (1 - alpha) * load.queue_depth
        load.tokens_per_sec = alpha * tokens_per_sec + (1 - alpha) * load.tokens_per_sec
        load.last_report = now
        load.stale = False

    def get_remote_load(self, cluster_id: str) -> RemoteClusterLoad | None:
        load = self._loads.get(cluster_id)
        if load and load.is_stale(self.stale_threshold_s):
            load.stale = True
            logger.debug(f"Load report for cluster {cluster_id} is stale")
        return load

    def get_all_loads(self) -> dict[str, RemoteClusterLoad]:
        for load in self._loads.values():
            load.stale = load.is_stale(self.stale_threshold_s)
        return dict(self._loads)

    def get_best_cluster(self, cluster_ids: list[str]) -> str | None:
        candidates = []
        for cid in cluster_ids:
            load = self._loads.get(cid)
            if load is None:
                candidates.append((cid, 0.0))
            elif load.is_stale(self.stale_threshold_s):
                continue
            elif load.is_overloaded:
                continue
            else:
                score = load.gpu_utilization + load.queue_depth
                candidates.append((cid, score))

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1])[0]

    def remove_cluster(self, cluster_id: str) -> None:
        self._loads.pop(cluster_id, None)
        logger.info(f"Removed cluster {cluster_id} from load balancer")

    def to_dict(self) -> dict[str, Any]:
        return {
            cid: {
                "active": load.active_requests,
                "pending": load.pending_requests,
                "gpu_util": load.gpu_utilization,
                "queue_depth": load.queue_depth,
                "tokens_per_sec": load.tokens_per_sec,
                "stale": load.stale,
            }
            for cid, load in self._loads.items()
        }
