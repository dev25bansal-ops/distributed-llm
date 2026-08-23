"""Federation peer discovery for cross-datacenter inference.

Supports seed-node bootstrap and DNS SRV record-based discovery
for automatic federation peer registration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.security import safe_urlopen


@dataclass
class PeerInfo:
    cluster_id: str
    host: str
    port: int
    is_edge: bool = False
    region: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: float = 0.0
    last_seen: float = 0.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class FederationPeerDiscovery:
    def __init__(
        self,
        own_cluster_id: str,
        own_host: str,
        own_port: int,
        discovery_interval_s: float = 30.0,
    ) -> None:
        self.own_cluster_id = own_cluster_id
        self.own_host = own_host
        self.own_port = own_port
        self.discovery_interval_s = discovery_interval_s

        self._peers: dict[str, PeerInfo] = {}
        self._seed_nodes: list[str] = []

    def add_seed_nodes(self, seed_nodes: list[str]) -> None:
        self._seed_nodes.extend(seed_nodes)
        logger.info(f"Added {len(seed_nodes)} seed nodes for federation discovery")

    def discover_peers(self, seed_nodes: list[str] | None = None) -> list[PeerInfo]:
        seeds = seed_nodes or self._seed_nodes
        discovered: list[PeerInfo] = []

        for seed_url in seeds:
            try:
                peers = self._fetch_peer_list(seed_url)
                for peer in peers:
                    if peer.cluster_id != self.own_cluster_id:
                        self._register_peer(peer)
                        discovered.append(peer)
                logger.info(f"Discovered {len(peers)} peers from {seed_url}")
            except Exception as e:
                logger.warning(f"Failed to discover peers from {seed_url}: {e}")

        return discovered

    def register_self(self, peer_url: str) -> bool:
        import urllib.request

        payload = json.dumps({
            "cluster_id": self.own_cluster_id,
            "host": self.own_host,
            "port": self.own_port,
            "is_edge": True,
            "region": "",
        }).encode()

        try:
            url = f"{peer_url.rstrip('/')}/api/v1/federation/register"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with safe_urlopen(req, timeout=10, allow_private_hosts=True) as resp:
                if resp.status == 200:
                    logger.info(f"Registered with federation peer: {peer_url}")
                    return True
        except Exception as e:
            logger.debug(f"Failed to register with {peer_url}: {e}")

        return False

    def get_peers(self) -> list[PeerInfo]:
        return list(self._peers.values())

    def get_peer(self, cluster_id: str) -> PeerInfo | None:
        return self._peers.get(cluster_id)

    def _register_peer(self, peer: PeerInfo) -> None:
        peer.last_seen = time.time()
        if peer.discovered_at == 0.0:
            peer.discovered_at = peer.last_seen
        self._peers[peer.cluster_id] = peer

    @staticmethod
    def _fetch_peer_list(seed_url: str) -> list[PeerInfo]:
        import urllib.request

        url = f"{seed_url.rstrip('/')}/api/v1/federation/peers"
        try:
            with safe_urlopen(url, timeout=10, allow_private_hosts=True) as resp:
                data = json.loads(resp.read())
                return [
                    PeerInfo(
                        cluster_id=p["cluster_id"],
                        host=p["host"],
                        port=p["port"],
                        is_edge=p.get("is_edge", False),
                        region=p.get("region", ""),
                        metadata=p.get("metadata", {}),
                    )
                    for p in data.get("peers", [])
                ]
        except Exception:
            return []
