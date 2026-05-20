"""Multi-cluster discovery: discovers clusters and their coordinator endpoints.

Supports three discovery modes:
- **static**: explicit list of clusters with coordinator URLs
- **dns**: resolve cluster SRV records to discover coordinators
- **k8s**: watch Kubernetes API for cluster namespaces/labels

Each cluster tracks its coordinators, health, load, and measured latency.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger


@dataclass
class ClusterCoordinator:
    """A single coordinator endpoint within a cluster."""
    node_id: str
    url: str
    healthy: bool = True
    latency_ms: float = 0.0
    active_requests: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ClusterInfo:
    """Represents a known cluster in the federation."""
    cluster_id: str
    region: str = "default"
    coordinators: dict[str, ClusterCoordinator] = field(default_factory=dict)
    healthy: bool = True
    measured_latency_ms: float = 0.0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_overloaded(self) -> bool:
        return self.gpu_utilization > 0.85 or self.queue_depth > 50

    @property
    def available_capacity(self) -> float:
        load = max(self.gpu_utilization, self.queue_depth / 100.0)
        return max(0.0, 1.0 - min(load, 1.0))

    def get_healthy_coordinator(self, preferred: str | None = None) -> ClusterCoordinator | None:
        """Get the healthiest coordinator in this cluster.

        Prefers *preferred* if healthy, otherwise picks the one with
        the lowest active request count.
        """
        if preferred and preferred in self.coordinators:
            coord = self.coordinators[preferred]
            if coord.healthy:
                return coord

        healthy = [c for c in self.coordinators.values() if c.healthy]
        if not healthy:
            return None
        return min(healthy, key=lambda c: c.active_requests)


class ClusterDiscovery:
    """Discovers clusters and their coordinator endpoints.

    Usage::

        discovery = ClusterDiscovery()
        await discovery.start(mode="static", config={
            "clusters": [
                {"cluster_id": "us-east-1", "region": "us-east-1",
                 "coordinators": ["http://coord-0.us-east:8000"]},
            ],
        })
        clusters = discovery.get_clusters()
        coord = discovery.get_coordinator("us-east-1")
    """

    def __init__(self):
        self._clusters: dict[str, ClusterInfo] = {}
        self._on_cluster_add: list[Callable[[str], None]] = []
        self._on_cluster_remove: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self._watch_task: asyncio.Task | None = None

    def on_cluster_add(self, callback: Callable[[str], None]) -> None:
        self._on_cluster_add.append(callback)

    def on_cluster_remove(self, callback: Callable[[str], None]) -> None:
        self._on_cluster_remove.append(callback)

    def add_cluster(self, cluster_id: str, region: str = "default",
                    coordinator_urls: list[str] | None = None,
                    tags: dict[str, str] | None = None) -> ClusterInfo:
        """Register or update a cluster.

        Args:
            cluster_id: Unique cluster identifier.
            region: Geographic region name.
            coordinator_urls: URLs of coordinator endpoints in the cluster.
            tags: Optional metadata tags (cloud provider, instance type, etc.).

        Returns:
            The ``ClusterInfo`` for the registered cluster.
        """
        with self._lock:
            existing = self._clusters.get(cluster_id)
            if existing is None:
                info = ClusterInfo(
                    cluster_id=cluster_id,
                    region=region,
                    tags=tags or {},
                )
                self._clusters[cluster_id] = info
                added = True
            else:
                info = existing
                info.region = region
                if tags:
                    info.tags.update(tags)
                added = False

            if coordinator_urls:
                for i, url in enumerate(coordinator_urls):
                    node_id = f"{cluster_id}-coord-{i}"
                    if node_id not in info.coordinators:
                        info.coordinators[node_id] = ClusterCoordinator(
                            node_id=node_id, url=url,
                        )

            if added:
                for cb in self._on_cluster_add:
                    try:
                        cb(cluster_id)
                    except Exception:
                        pass
                logger.info(f"Cluster discovered: {cluster_id} ({region})")

        return info

    def remove_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._clusters.pop(cluster_id, None)
        for cb in self._on_cluster_remove:
            try:
                cb(cluster_id)
            except Exception:
                pass
        logger.info(f"Cluster removed: {cluster_id}")

    def add_coordinator(self, cluster_id: str, url: str, node_id: str | None = None) -> None:
        """Register a coordinator endpoint in a cluster."""
        with self._lock:
            info = self._clusters.get(cluster_id)
            if info is None:
                info = self.add_cluster(cluster_id)
            nid = node_id or f"{cluster_id}-coord-{len(info.coordinators)}"
            if nid not in info.coordinators:
                info.coordinators[nid] = ClusterCoordinator(node_id=nid, url=url)

    def remove_coordinator(self, cluster_id: str, node_id: str) -> None:
        with self._lock:
            info = self._clusters.get(cluster_id)
            if info:
                info.coordinators.pop(node_id, None)

    def get_cluster(self, cluster_id: str) -> ClusterInfo | None:
        with self._lock:
            return self._clusters.get(cluster_id)

    def get_clusters(self) -> dict[str, ClusterInfo]:
        with self._lock:
            return dict(self._clusters)

    def get_healthy_clusters(self) -> list[ClusterInfo]:
        with self._lock:
            return [c for c in self._clusters.values() if c.healthy]

    def get_cluster_ids(self) -> list[str]:
        with self._lock:
            return list(self._clusters.keys())

    def get_coordinator(self, cluster_id: str,
                        preferred: str | None = None) -> ClusterCoordinator | None:
        """Get the best coordinator for a cluster."""
        info = self.get_cluster(cluster_id)
        if info is None:
            return None
        return info.get_healthy_coordinator(preferred)

    def update_cluster_load(self, cluster_id: str, *,
                            gpu_util: float = 0.0, queue_depth: int = 0,
                            healthy: bool | None = None) -> None:
        """Update load metrics for a cluster."""
        with self._lock:
            info = self._clusters.get(cluster_id)
            if info:
                info.gpu_utilization = gpu_util
                info.queue_depth = queue_depth
                if healthy is not None:
                    info.healthy = healthy

    def update_coordinator_latency(self, cluster_id: str, node_id: str,
                                   latency_ms: float) -> None:
        with self._lock:
            info = self._clusters.get(cluster_id)
            if info and node_id in info.coordinators:
                info.coordinators[node_id].latency_ms = latency_ms

    def update_coordinator_health(self, cluster_id: str, node_id: str,
                                  healthy: bool) -> None:
        with self._lock:
            info = self._clusters.get(cluster_id)
            if info and node_id in info.coordinators:
                info.coordinators[node_id].healthy = healthy

    async def start(self, mode: str = "static", config: dict | None = None) -> None:
        """Start cluster discovery.

        Args:
            mode: ``"static"``, ``"dns"``, or ``"k8s"``.
            config: Discovery configuration dict.
        """
        cfg = config or {}
        if mode == "static":
            self._start_static(cfg)
        elif mode == "dns":
            self._watch_task = asyncio.create_task(self._dns_watch(cfg))
        elif mode == "k8s":
            self._watch_task = asyncio.create_task(self._k8s_watch(cfg))
        else:
            raise ValueError(f"Unknown discovery mode: {mode}")

    async def stop(self) -> None:
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    def _start_static(self, config: dict) -> None:
        clusters = config.get("clusters", [])
        for entry in clusters:
            cid = entry.get("cluster_id")
            if not cid:
                continue
            self.add_cluster(
                cluster_id=cid,
                region=entry.get("region", "default"),
                coordinator_urls=entry.get("coordinators", []),
                tags=entry.get("tags", {}),
            )

    async def _dns_watch(self, config: dict) -> None:
        import socket as _socket
        service_name = config.get("service", "_distllm._tcp.cluster.local")
        interval = config.get("interval", 30)

        while True:
            try:
                results = _socket.getaddrinfo(service_name, 8000, type=_socket.SOCK_STREAM)
                seen_clusters: set[str] = set()
                for i, (_, _, _, _, addr) in enumerate(results):
                    cluster_id = f"dns-cluster-{i // 3}"
                    url = f"http://{addr[0]}:8000"
                    self.add_coordinator(cluster_id, url)
                    seen_clusters.add(cluster_id)

                for cid in list(self._clusters.keys()):
                    if cid.startswith("dns-cluster-") and cid not in seen_clusters:
                        self.remove_cluster(cid)
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _k8s_watch(self, config: dict) -> None:
        try:
            from kubernetes import client, config as k8s_config, watch
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            v1 = client.CoreV1Api()
            namespace = config.get("namespace", "default")
            label_selector = config.get("label_selector", "app=distllm-cluster")
            port = config.get("port", 8000)

            w = watch.Watch()
            while True:
                for event in w.stream(
                    v1.list_namespaced_pod,
                    namespace=namespace,
                    label_selector=label_selector,
                    timeout_seconds=60,
                ):
                    pod = event["object"]
                    event_type = event["type"]
                    cluster_id = pod.metadata.labels.get("cluster-id", "k8s-cluster")
                    node_id = pod.metadata.name
                    pod_ip = pod.status.pod_ip

                    if event_type in ("ADDED", "MODIFIED") and pod_ip:
                        self.add_coordinator(cluster_id, f"http://{pod_ip}:{port}", node_id)
                    elif event_type == "DELETED":
                        self.remove_coordinator(cluster_id, node_id)
        except ImportError:
            logger.warning("kubernetes package not available — K8s discovery disabled")
        except Exception:
            pass
        await asyncio.sleep(10)
