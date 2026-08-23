"""Federated Draft Banks — community-operated draft model pools.

Enables organizations to share draft model inference capacity across
federations. Each participant contributes spare CPU/edge capacity as
a draft model endpoint, and the federation coordinates routing.

Privacy-preserving: draft tokens are token IDs only — no raw text
leaves the target model's trust boundary.

Usage::

    bank = FederatedDraftBank(
        own_cluster_id="cluster-a",
        own_host="10.0.0.1",
        own_port=9000,
    )

    # Discover peers and register our draft capacity
    bank.discover_and_register(seed_nodes=["http://coordinator:8000"])

    # Get the best draft endpoint from the federation
    endpoint = bank.get_best_draft_endpoint(
        workload_type="code",
        max_latency_ms=100.0,
    )
"""


from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class DraftBankEntry:
    """A draft model endpoint in the federated bank."""

    cluster_id: str
    endpoint_url: str
    model_name: str = ""
    hardware: str = "cpu"
    cost_per_hour: float = 0.0
    avg_latency_ms: float = 0.0
    avg_acceptance_rate: float = 0.0
    max_concurrent: int = 10
    current_load: int = 0
    total_served: int = 0
    total_errors: int = 0
    last_seen: float = 0.0
    region: str = ""
    reputation_score: float = 1.0  # 0.0-1.0, higher is better

    @property
    def is_stale(self, threshold_s: float = 60.0) -> bool:
        return (time.time() - self.last_seen) > threshold_s

    @property
    def is_overloaded(self) -> bool:
        return self.current_load >= self.max_concurrent

    @property
    def availability_score(self) -> float:
        """Score based on availability (0.0-1.0)."""

        if self.is_overloaded or self.is_stale:
            return 0.0
        load_ratio = self.current_load / max(self.max_concurrent, 1)
        return max(0.0, 1.0 - load_ratio)


@dataclass
class FederationDraftConfig:
    """Configuration for federated draft bank participation."""

    own_cluster_id: str = ""
    own_host: str = ""
    own_port: int = 9000
    own_model_name: str = ""
    own_hardware: str = "cpu"
    own_cost_per_hour: float = 0.05
    own_max_concurrent: int = 10
    discovery_interval_s: float = 30.0
    stale_threshold_s: float = 60.0
    reputation_decay: float = 0.95  # Per-interval decay factor
    min_reputation: float = 0.1


class FederatedDraftBank:
    """Community-operated draft model pool with P2P discovery.


    Each participant:
    1. Advertises its draft model capacity to the federation
    2. Discovers other participants' endpoints
    3. Routes draft requests to the best available endpoint
    4. Tracks reputation based on success/failure

    Integrates with the existing FederationPeerDiscovery system.
    """


    def __init__(
        self,
        own_cluster_id: str,
        own_host: str,
        own_port: int,
        config: FederationDraftConfig | None = None,
    ) -> None:
        self._config = config or FederationDraftConfig(
            own_cluster_id=own_cluster_id,
            own_host=own_host,
            own_port=own_port,
        )
        self._entries: dict[str, DraftBankEntry] = {}
        self._lock = threading.Lock()
        self._discovery: Any = None

    def register_local_capacity(
        self,
        model_name: str,
        hardware: str = "cpu",
        cost_per_hour: float = 0.05,
        max_concurrent: int = 10,
    ) -> None:
        """Register our own draft model capacity in the bank."""

        entry = DraftBankEntry(
            cluster_id=self._config.own_cluster_id,
            endpoint_url=f"http://{self._config.own_host}:{self._config.own_port}",
            model_name=model_name,
            hardware=hardware,
            cost_per_hour=cost_per_hour,
            max_concurrent=max_concurrent,
            last_seen=time.time(),
            reputation_score=1.0,
        )
        with self._lock:
            self._entries[self._config.own_cluster_id] = entry
        logger.info(
            f"Registered local draft capacity: {model_name} on {hardware} "
            f"(${cost_per_hour:.2f}/hr, max {max_concurrent} concurrent)"
        )

    def discover_and_register(
        self,
        seed_nodes: list[str] | None = None,
    ) -> list[DraftBankEntry]:
        """Discover federation peers and register our draft capacity.


        Uses the existing FederationPeerDiscovery system.
        """

        try:
            from distllm.dist.p2p.discovery import FederationPeerDiscovery

            self._discovery = FederationPeerDiscovery(
                own_cluster_id=self._config.own_cluster_id,
                own_host=self._config.own_host,
                own_port=self._config.own_port,
                discovery_interval_s=self._config.discovery_interval_s,
            )

            if seed_nodes:
                self._discovery.add_seed_nodes(seed_nodes)

            # Register self with seed nodes
            for seed in seed_nodes or []:
                self._discovery.register_self(seed)

            # Discover peers
            peers = self._discovery.discover_peers()

            # Convert peers to draft bank entries
            discovered: list[DraftBankEntry] = []
            for peer in peers:
                entry = DraftBankEntry(
                    cluster_id=peer.cluster_id,
                    endpoint_url=f"{peer.url}/v1/completions",
                    model_name=peer.metadata.get("draft_model", ""),
                    hardware=peer.metadata.get("hardware", "cpu"),
                    cost_per_hour=float(peer.metadata.get("cost_per_hour", 0.0)),
                    max_concurrent=int(peer.metadata.get("max_concurrent", 10)),
                    last_seen=time.time(),
                    region=peer.region,
                )
                with self._lock:
                    self._entries[peer.cluster_id] = entry
                discovered.append(entry)

            logger.info(f"Discovered {len(discovered)} draft endpoints in federation")
            return discovered

        except ImportError:
            logger.warning("FederationPeerDiscovery not available, running standalone")
            return []
        except Exception as e:
            logger.warning(f"Federation discovery failed: {e}")
            return []

    def get_best_draft_endpoint(
        self,
        workload_type: str = "unknown",
        max_latency_ms: float = 200.0,
        max_cost_per_hour: float = 10.0,
        prefer_local: bool = True,
    ) -> DraftBankEntry | None:
        """Select the best draft endpoint from the federation.


        Selection criteria:
        1. Must be healthy (not stale, not overloaded)
        2. Must meet latency and cost constraints
        3. Scored by: reputation * availability * (1 / latency)
        4. Local endpoints get a preference bonus
        """

        with self._lock:
            candidates: list[tuple[float, DraftBankEntry]] = []

            for entry in self._entries.values():
                if entry.is_stale(self._config.stale_threshold_s):
                    continue
                if entry.is_overloaded:
                    continue
                if entry.avg_latency_ms > max_latency_ms:
                    continue
                if entry.cost_per_hour > max_cost_per_hour:
                    continue

                score = (
                    entry.reputation_score
                    * entry.availability_score
                    / max(entry.avg_latency_ms, 1.0)
                )

                # Local preference bonus
                if prefer_local and entry.cluster_id == self._config.own_cluster_id:
                    score *= 1.5

                candidates.append((score, entry))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

    def record_success(
        self,
        cluster_id: str,
        latency_s: float,
        tokens_generated: int,
        acceptance_rate: float = 0.0,
    ) -> None:
        """Record a successful draft call to a federation endpoint."""

        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry is None:
                return
            entry.total_served += 1
            entry.last_seen = time.time()
            entry.avg_latency_ms = (
                entry.avg_latency_ms * 0.8 + (latency_s * 1000) * 0.2
            )
            entry.avg_acceptance_rate = (
                entry.avg_acceptance_rate * 0.8 + acceptance_rate * 0.2
            )
            # Boost reputation on success
            entry.reputation_score = min(1.0, entry.reputation_score * 1.01)

    def record_error(self, cluster_id: str, error: str) -> None:
        """Record a failed draft call to a federation endpoint."""

        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry is None:
                return
            entry.total_errors += 1
            entry.last_seen = time.time()
            # Penalize reputation on error
            entry.reputation_score = max(
                self._config.min_reputation,
                entry.reputation_score * 0.9,
            )

    def record_request_start(self, cluster_id: str) -> None:
        """Track active concurrent requests."""

        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry:
                entry.current_load += 1

    def record_request_end(self, cluster_id: str) -> None:
        """Track request completion."""

        with self._lock:
            entry = self._entries.get(cluster_id)
            if entry:
                entry.current_load = max(0, entry.current_load - 1)

    def decay_reputation(self) -> None:
        """Periodically decay reputation scores (prevents stale high scores)."""

        with self._lock:
            for entry in self._entries.values():
                entry.reputation_score = max(
                    self._config.min_reputation,
                    entry.reputation_score * self._config.reputation_decay,
                )

    def get_federation_status(self) -> dict[str, Any]:
        """Get status of all federation draft endpoints."""

        with self._lock:
            return {
                "total_endpoints": len(self._entries),
                "healthy_endpoints": sum(
                    1 for e in self._entries.values()
                    if not e.is_stale(self._config.stale_threshold_s)
                ),
                "endpoints": [
                    {
                        "cluster_id": e.cluster_id,
                        "endpoint_url": e.endpoint_url,
                        "model": e.model_name,
                        "hardware": e.hardware,
                        "load": f"{e.current_load}/{e.max_concurrent}",
                        "latency_ms": round(e.avg_latency_ms, 1),
                        "acceptance_rate": round(e.avg_acceptance_rate, 3),
                        "reputation": round(e.reputation_score, 3),
                        "total_served": e.total_served,
                        "total_errors": e.total_errors,
                        "stale": e.is_stale(self._config.stale_threshold_s),
                    }
                    for e in self._entries.values()
                ],
            }
