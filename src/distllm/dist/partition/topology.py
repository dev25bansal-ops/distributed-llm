from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LinkProfile:
    source: str
    target: str
    bandwidth_gbps: float = 12.5
    latency_us: float = 100.0
    is_nvlink: bool = False
    is_infiniband: bool = False


@dataclass
class TopologyGraph:
    node_ids: list[str] = field(default_factory=list)
    gpu_counts: dict[str, int] = field(default_factory=dict)
    gpu_profiles: dict[str, list] = field(default_factory=dict)
    links: list[LinkProfile] = field(default_factory=list)
    node_hostnames: dict[str, str] = field(default_factory=dict)

    def get_bandwidth(self, source: str, target: str) -> float:
        for link in self.links:
            if (link.source == source and link.target == target) or \
               (link.source == target and link.target == source):
                return link.bandwidth_gbps
        return 1.0

    def get_latency(self, source: str, target: str) -> float:
        for link in self.links:
            if (link.source == source and link.target == target) or \
               (link.source == target and link.target == source):
                return link.latency_us
        return 1000.0

    def total_gpus(self) -> int:
        return sum(self.gpu_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": nid,
                    "gpu_count": self.gpu_counts.get(nid, 0),
                    "hostname": self.node_hostnames.get(nid, nid),
                }
                for nid in self.node_ids
            ],
            "total_gpus": self.total_gpus(),
            "links": [
                {
                    "source": l.source,
                    "target": l.target,
                    "bandwidth_gbps": l.bandwidth_gbps,
                    "latency_us": l.latency_us,
                    "is_nvlink": l.is_nvlink,
                    "is_infiniband": l.is_infiniband,
                }
                for l in self.links
            ],
        }


class TopologyProber:
    def __init__(
        self,
        default_intra_node_bandwidth: float = 600.0,
        default_inter_node_bandwidth: float = 12.5,
        ping_timeout_seconds: float = 2.0,
        bandwidth_test_bytes: int = 8 * 1024 * 1024,
    ):
        self._default_intra = default_intra_node_bandwidth
        self._default_inter = default_inter_node_bandwidth
        self._ping_timeout = ping_timeout_seconds
        self._bandwidth_test_bytes = bandwidth_test_bytes

    async def probe(
        self,
        node_ids: list[str],
        hostnames: dict[str, str] | None = None,
        gpu_counts: dict[str, int] | None = None,
        gpu_profiles: dict[str, list] | None = None,
    ) -> TopologyGraph:
        hosts = hostnames or {nid: nid for nid in node_ids}
        gpu_cnt = gpu_counts or {nid: 1 for nid in node_ids}

        graph = TopologyGraph(
            node_ids=list(node_ids),
            gpu_counts=dict(gpu_cnt),
            gpu_profiles=dict(gpu_profiles or {}),
            node_hostnames=dict(hosts),
        )

        links: list[LinkProfile] = []

        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i >= j:
                    continue

                is_same_node = hosts.get(src) == hosts.get(dst) or (
                    src == dst
                )

                if is_same_node:
                    bw = self._detect_intra_node_bandwidth(src, dst)
                    links.append(LinkProfile(
                        source=src, target=dst,
                        bandwidth_gbps=bw,
                        latency_us=self._measure_latency(src, dst),
                        is_nvlink=True,
                    ))
                else:
                    latency = await self._measure_latency_async(
                        hosts.get(src, src), hosts.get(dst, dst),
                    )
                    bw = self._detect_inter_node_bandwidth(src, dst)
                    links.append(LinkProfile(
                        source=src, target=dst,
                        bandwidth_gbps=bw,
                        latency_us=latency,
                        is_infiniband=bw > 25.0,
                    ))

        graph.links = links
        return graph

    def probe_local_topology(self, num_gpus: int) -> list[LinkProfile]:
        links: list[LinkProfile] = []
        has_nvlink = self._detect_nvlink(num_gpus)

        bw = 600.0 if has_nvlink else self._default_intra

        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                links.append(LinkProfile(
                    source=f"gpu-{i}",
                    target=f"gpu-{j}",
                    bandwidth_gbps=bw,
                    latency_us=5.0 if has_nvlink else 20.0,
                    is_nvlink=has_nvlink,
                ))

        return links

    def _detect_intra_node_bandwidth(self, src: str, dst: str) -> float:
        try:
            from distllm.dist.parallel import HardwareProber
            prober = HardwareProber()
            bw = prober._measure_p2p_bw(0, 1)
            return max(bw, 1.0)
        except Exception:
            return self._default_intra

    def _detect_inter_node_bandwidth(self, src: str, dst: str) -> float:
        try:
            from distllm.dist.parallel import HardwareProber
            prober = HardwareProber()
            return prober._measure_network_bw()
        except Exception:
            return self._default_inter

    def _detect_nvlink(self, num_gpus: int) -> bool:
        try:
            from distllm.dist.parallel import HardwareProber
            prober = HardwareProber()
            return prober._detect_nvlink(num_gpus)
        except Exception:
            return False

    def _measure_latency(self, src: str, dst: str) -> float:
        return 5.0

    async def _measure_latency_async(
        self, host_a: str, host_b: str
    ) -> float:
        if host_a == host_b:
            return 5.0
        try:
            import asyncio

            t0 = time.time()
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host_b, 50050),
                timeout=self._ping_timeout,
            )
            elapsed_us = (time.time() - t0) * 1_000_000
            writer.close()
            await writer.wait_closed()
            return elapsed_us
        except Exception:
            return 500.0

    @staticmethod
    def make_fallback_topology(
        num_nodes: int, gpus_per_node: int = 1
    ) -> TopologyGraph:
        node_ids = [f"node-{i}" for i in range(num_nodes)]
        links = [
            LinkProfile(
                source=node_ids[i],
                target=node_ids[j],
                bandwidth_gbps=12.5 if i != j else 600.0,
                latency_us=500.0 if i != j else 5.0,
            )
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
        ]
        return TopologyGraph(
            node_ids=node_ids,
            gpu_counts={nid: gpus_per_node for nid in node_ids},
            links=links,
            node_hostnames={nid: f"{nid}.local" for nid in node_ids},
        )
