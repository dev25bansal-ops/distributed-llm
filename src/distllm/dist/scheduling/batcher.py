"""Latency-aware batching for cross-cluster inference."""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger

@dataclass
class BatchGroup:
    group_id: str
    cluster_id: str
    request_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class LatencyAwareBatcher:
    def __init__(self, node_to_cluster: Optional[Dict[str, str]] = None):
        self.node_to_cluster: Dict[str, str] = node_to_cluster or {}
        self._groups: Dict[str, BatchGroup] = {}
        self._lock = threading.Lock()
        self._group_counter = 0

    def set_cluster_map(self, node_to_cluster: Dict[str, str]) -> None:
        with self._lock:
            self.node_to_cluster = node_to_cluster

    def group_requests(
        self,
        requests: List[dict],
        max_group_size: int = 8,
    ) -> List[BatchGroup]:
        by_cluster: Dict[str, List[str]] = {}
        for req in requests:
            req_id = req.get("request_id", "")
            cluster = req.get("cluster_id")
            if cluster is None:
                node_id = req.get("node_id", "")
                cluster = self.node_to_cluster.get(node_id, "default")

            if cluster not in by_cluster:
                by_cluster[cluster] = []
            by_cluster[cluster].append(req_id)

        groups = []
        with self._lock:
            for cluster_id, req_ids in by_cluster.items():
                for i in range(0, len(req_ids), max_group_size):
                    chunk = req_ids[i:i + max_group_size]
                    self._group_counter += 1
                    group = BatchGroup(
                        group_id=f"batch-{self._group_counter}",
                        cluster_id=cluster_id,
                        request_ids=chunk,
                    )
                    groups.append(group)
                    self._groups[group.group_id] = group

        for g in groups:
            logger.debug(f"Batch group {g.group_id}: cluster={g.cluster_id}, "
                        f"requests={len(g.request_ids)}")

        return groups

    def get_group(self, group_id: str) -> Optional[BatchGroup]:
        with self._lock:
            return self._groups.get(group_id)

    def complete_group(self, group_id: str) -> None:
        with self._lock:
            self._groups.pop(group_id, None)

    def pending_groups(self) -> List[BatchGroup]:
        with self._lock:
            return list(self._groups.values())

    def prioritize_execution_order(self, groups: List[BatchGroup]) -> List[BatchGroup]:
        local = "default"
        return sorted(
            groups,
            key=lambda g: 0 if g.cluster_id == local else 1,
        )

    def stats(self) -> dict:
        with self._lock:
            clusters = {}
            for g in self._groups.values():
                if g.cluster_id not in clusters:
                    clusters[g.cluster_id] = 0
                clusters[g.cluster_id] += len(g.request_ids)
            return {
                "pending_groups": len(self._groups),
                "total_pending_requests": sum(len(g.request_ids) for g in self._groups.values()),
                "by_cluster": clusters,
            }
