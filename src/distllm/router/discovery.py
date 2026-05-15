"""Coordinator discovery for the distributed-llm router.

Supports three discovery modes:
- static: list of coordinator URLs from config
- dns: resolve coordinator service DNS records
- k8s: watch Kubernetes API for coordinator pods
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class CoordinatorInfo:
    node_id: str
    url: str  # Base URL e.g. "http://coord-0:8000"
    healthy: bool = True


class CoordinatorDiscovery:
    """Manages coordinator endpoint discovery."""

    def __init__(self, mode: str = "static"):
        self.mode = mode
        self._coordinators: Dict[str, CoordinatorInfo] = {}
        self._callbacks: List[Callable[[str, bool], None]] = []
        self._watch_task: Optional[asyncio.Task] = None

    def on_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for coordinator add/remove events."""
        self._callbacks.append(callback)

    def add_coordinator(self, node_id: str, url: str) -> None:
        self._coordinators[node_id] = CoordinatorInfo(node_id=node_id, url=url)
        for cb in self._callbacks:
            cb(node_id, True)

    def remove_coordinator(self, node_id: str) -> None:
        self._coordinators.pop(node_id, None)
        for cb in self._callbacks:
            cb(node_id, False)

    def get_all(self) -> Dict[str, CoordinatorInfo]:
        return dict(self._coordinators)

    def get_healthy(self) -> List[CoordinatorInfo]:
        return [c for c in self._coordinators.values() if c.healthy]

    def set_health(self, node_id: str, healthy: bool) -> None:
        if node_id in self._coordinators:
            self._coordinators[node_id].healthy = healthy

    async def start(self, config: dict) -> None:
        """Start discovery based on mode and config."""
        if self.mode == "static":
            urls = config.get("coordinators", [])
            for i, url in enumerate(urls):
                node_id = config.get("ids", [])[i] if "ids" in config else f"coord-{i}"
                self.add_coordinator(node_id, url)
        elif self.mode == "dns":
            self._watch_task = asyncio.create_task(self._dns_watch(config))
        elif self.mode == "k8s":
            self._watch_task = asyncio.create_task(self._k8s_watch(config))

    async def stop(self) -> None:
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    async def _dns_watch(self, config: dict) -> None:
        """Watch DNS for coordinator service changes."""
        service_name = config.get("service", "distllm-coordinator")
        port = config.get("port", 8000)
        while True:
            try:
                import socket
                results = socket.getaddrinfo(service_name, port, type=socket.SOCK_STREAM)
                seen = set()
                for i, (_, _, _, _, addr) in enumerate(results):
                    node_id = f"coord-{i}"
                    url = f"http://{addr[0]}:{port}"
                    if node_id not in self._coordinators:
                        self.add_coordinator(node_id, url)
                    seen.add(node_id)
                # Remove coordinators no longer in DNS
                for nid in list(self._coordinators.keys()):
                    if nid not in seen:
                        self.remove_coordinator(nid)
            except Exception:
                pass
            await asyncio.sleep(10)

    async def _k8s_watch(self, config: dict) -> None:
        """Watch K8s API for coordinator pod changes."""
        try:
            from kubernetes import client, config as k8s_config, watch

            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            v1 = client.CoreV1Api()
            namespace = config.get("namespace", "default")
            label_selector = config.get("label_selector", "app=distllm-coordinator")
            port = config.get("port", 8000)

            w = watch.Watch()
            for event in w.stream(
                v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                timeout_seconds=60,
            ):
                pod = event["object"]
                event_type = event["type"]
                node_id = pod.metadata.name
                pod_ip = pod.status.pod_ip

                if event_type in ("ADDED", "MODIFIED") and pod_ip:
                    url = f"http://{pod_ip}:{port}"
                    self.add_coordinator(node_id, url)
                elif event_type == "DELETED":
                    self.remove_coordinator(node_id)
        except ImportError:
            pass  # kubernetes package not available
        except Exception:
            pass
        await asyncio.sleep(1)
