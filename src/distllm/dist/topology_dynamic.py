"""Dynamic cluster topology discovery and live updates.

Maintains a live TopologyGraph that updates automatically when
nodes join or leave the cluster. Integrates with PipelineOrchestrator
register/unregister to probe links and maintain topology state.

Usage:
    topo = DynamicClusterTopology()
    topo.on_node_join("node-0", host="192.168.1.10", gpus=4)
    topo.on_node_leave("node-0")
    graph = topo.get_graph()
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from distllm.dist.partition.topology import (
    LinkProfile,
    TopologyGraph,
    TopologyProber,
)


@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int = 50051
    gpu_count: int = 1
    healthy: bool = True
    joined_at: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)


TopologyChangeCallback = Callable[[str, str, TopologyGraph], None]


class DynamicClusterTopology:
    def __init__(
        self,
        prober: TopologyProber | None = None,
        probe_on_join: bool = True,
        default_bandwidth: float = 12.5,
        default_latency_us: float = 500.0,
    ):
        self._prober = prober or TopologyProber()
        self._probe_on_join = probe_on_join
        self._default_bandwidth = default_bandwidth
        self._default_latency_us = default_latency_us

        self._nodes: dict[str, NodeInfo] = {}
        self._graph: TopologyGraph = TopologyGraph()
        self._callbacks: list[TopologyChangeCallback] = []
        self._lock = threading.RLock()

    def on_node_join(
        self,
        node_id: str,
        host: str = "",
        port: int = 50051,
        gpu_count: int = 1,
        tags: dict[str, str] | None = None,
    ) -> TopologyGraph:
        with self._lock:
            now = time.time()
            info = NodeInfo(
                node_id=node_id,
                host=host or node_id,
                port=port,
                gpu_count=gpu_count,
                joined_at=now,
                tags=tags or {},
            )

            is_new = node_id not in self._nodes
            self._nodes[node_id] = info

            if is_new:
                self._add_node_to_graph(info)
                if len(self._nodes) > 1:
                    self._add_default_links(info)

            graph = self._rebuild_graph()

        logger.info(
            f"Topology node joined: {node_id} "
            f"(host={host}, gpus={gpu_count}, total_nodes={len(self._nodes)})"
        )
        self._fire_callbacks(node_id, "join", graph)
        return graph

    def on_node_leave(self, node_id: str) -> TopologyGraph:
        with self._lock:
            if node_id not in self._nodes:
                logger.warning(f"Topology node leave ignored (unknown): {node_id}")
                return self._graph

            del self._nodes[node_id]
            self._graph.links = [
                link for link in self._graph.links
                if link.source != node_id and link.target != node_id
            ]
            self._graph.node_ids = [
                nid for nid in self._graph.node_ids if nid != node_id
            ]
            self._graph.gpu_counts.pop(node_id, None)
            self._graph.node_hostnames.pop(node_id, None)

            graph = TopologyGraph(
                node_ids=list(self._graph.node_ids),
                gpu_counts=dict(self._graph.gpu_counts),
                links=list(self._graph.links),
                node_hostnames=dict(self._graph.node_hostnames),
            )
            self._graph = graph

        logger.info(
            f"Topology node left: {node_id} "
            f"(total_nodes={len(self._nodes)})"
        )
        self._fire_callbacks(node_id, "leave", self._graph)
        return self._graph

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].healthy = False

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].healthy = True

    def get_graph(self) -> TopologyGraph:
        with self._lock:
            return TopologyGraph(
                node_ids=list(self._graph.node_ids),
                gpu_counts=dict(self._graph.gpu_counts),
                gpu_profiles=dict(self._graph.gpu_profiles),
                links=list(self._graph.links),
                node_hostnames=dict(self._graph.node_hostnames),
            )

    def get_node(self, node_id: str) -> NodeInfo | None:
        with self._lock:
            return self._nodes.get(node_id)

    def get_nodes(self) -> dict[str, NodeInfo]:
        with self._lock:
            return dict(self._nodes)

    def get_healthy_nodes(self) -> list[str]:
        with self._lock:
            return [
                nid for nid, info in self._nodes.items()
                if info.healthy
            ]

    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def total_gpus(self) -> int:
        with self._lock:
            return sum(n.gpu_count for n in self._nodes.values())

    def get_bandwidth(self, src: str, dst: str) -> float:
        return self._graph.get_bandwidth(src, dst)

    def get_latency(self, src: str, dst: str) -> float:
        return self._graph.get_latency(src, dst)

    def on_change(self, callback: TopologyChangeCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def _fire_callbacks(self, node_id: str, event: str, graph: TopologyGraph) -> None:
        cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(node_id, event, graph)
            except Exception as e:
                logger.error(f"Topology callback failed: {e}")

    def _add_node_to_graph(self, info: NodeInfo) -> None:
        self._graph.node_ids.append(info.node_id)
        self._graph.gpu_counts[info.node_id] = info.gpu_count
        self._graph.node_hostnames[info.node_id] = info.host

    def _add_default_links(self, info: NodeInfo) -> None:
        src = info.node_id
        for existing_id, existing_info in self._nodes.items():
            if existing_id == src:
                continue
            link = self._build_link(src, existing_id, info, existing_info)
            self._graph.links.append(link)

        if self._probe_on_join:
            self._probe_active_links(info)

    def _probe_active_links(self, info: NodeInfo) -> None:
        pass

    def _build_link(
        self,
        src: str,
        dst: str,
        src_info: NodeInfo,
        dst_info: NodeInfo,
    ) -> LinkProfile:
        is_same_host = src_info.host == dst_info.host
        if is_same_host:
            bw = 600.0
            lat = 5.0
            is_nvlink = True
            is_ib = False
        else:
            bw = self._default_bandwidth
            lat = self._default_latency_us
            is_nvlink = False
            is_ib = bw > 25.0

        return LinkProfile(
            source=src, target=dst,
            bandwidth_gbps=bw,
            latency_us=lat,
            is_nvlink=is_nvlink,
            is_infiniband=is_ib,
        )

    def _rebuild_graph(self) -> TopologyGraph:
        self._graph = TopologyGraph(
            node_ids=list(self._graph.node_ids),
            gpu_counts=dict(self._graph.gpu_counts),
            links=list(self._graph.links),
            node_hostnames=dict(self._graph.node_hostnames),
        )
        return self._graph

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "total_gpus": self.total_gpus(),
                "link_count": len(self._graph.links),
                "healthy_nodes": len(self.get_healthy_nodes()),
                "unhealthy_nodes": sum(
                    1 for n in self._nodes.values() if not n.healthy
                ),
                "nodes": {
                    nid: {
                        "host": info.host,
                        "gpus": info.gpu_count,
                        "healthy": info.healthy,
                        "uptime_s": round(time.time() - info.joined_at, 1),
                    }
                    for nid, info in self._nodes.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._graph = TopologyGraph()
