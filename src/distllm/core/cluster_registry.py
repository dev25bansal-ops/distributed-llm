"""Public cluster registry — opt-in directory of public clusters.

Allows cluster operators to list their clusters publicly so users
can discover and join them. Includes contribution credits for
tracking GPU hours contributed.

Usage::

    registry = ClusterRegistry()
    registry.register_cluster(
        cluster_id="my-cluster",
        host="gpu.example.com",
        port=50050,
        gpus=4,
        model="llama-3-70b",
    )
    clusters = registry.discover(model="llama-3-70b")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ClusterInfo:
    """Public information about a cluster."""
    cluster_id: str
    host: str
    port: int
    gpus: int = 1
    gpu_model: str = ""
    model: str = ""
    region: str = ""
    reputation: float = 0.5
    uptime_pct: float = 100.0
    avg_latency_ms: float = 0.0
    is_public: bool = True
    owner_id: str = ""
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


@dataclass
class ContributionCredit:
    """Credits earned by contributing GPU time."""
    user_id: str
    cluster_id: str
    gpu_hours: float = 0.0
    tokens_served: int = 0
    earned_credits: float = 0.0
    spent_credits: float = 0.0
    tier: str = "bronze"  # bronze, silver, gold, platinum


class ClusterRegistry:
    """Public directory of DistLLM clusters."""

    def __init__(self, heartbeat_timeout: float = 300.0):
        self._clusters: dict[str, ClusterInfo] = {}
        self._credits: dict[str, ContributionCredit] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._lock = threading.Lock()

    def register_cluster(self, **kwargs: Any) -> ClusterInfo:
        """Register a cluster in the public directory."""
        info = ClusterInfo(**kwargs)
        with self._lock:
            self._clusters[info.cluster_id] = info
        logger.info(f"Cluster registered: {info.cluster_id} ({info.gpus} GPUs)")
        return info

    def unregister_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from the directory."""
        with self._lock:
            return self._clusters.pop(cluster_id, None) is not None

    def heartbeat(self, cluster_id: str, **metrics: Any) -> None:
        """Update cluster heartbeat."""
        with self._lock:
            cluster = self._clusters.get(cluster_id)
            if cluster:
                cluster.last_heartbeat = time.time()
                for k, v in metrics.items():
                    if hasattr(cluster, k):
                        setattr(cluster, k, v)

    def discover(
        self,
        model: str = "",
        region: str = "",
        min_gpus: int = 0,
        min_reputation: float = 0.0,
        limit: int = 20,
    ) -> list[ClusterInfo]:
        """Discover public clusters matching criteria."""
        with self._lock:
            now = time.time()
            candidates = [
                c for c in self._clusters.values()
                if c.is_public
                and (now - c.last_heartbeat) < self._heartbeat_timeout
            ]

        if model:
            candidates = [c for c in candidates if model in c.model]
        if region:
            candidates = [c for c in candidates if c.region == region]
        if min_gpus > 0:
            candidates = [c for c in candidates if c.gpus >= min_gpus]
        if min_reputation > 0:
            candidates = [c for c in candidates if c.reputation >= min_reputation]

        candidates.sort(key=lambda c: c.reputation, reverse=True)
        return candidates[:limit]

    def get_cluster(self, cluster_id: str) -> ClusterInfo | None:
        """Get info about a specific cluster."""
        with self._lock:
            return self._clusters.get(cluster_id)

    def list_clusters(self) -> list[dict]:
        """List all registered clusters."""
        with self._lock:
            return [
                {
                    "cluster_id": c.cluster_id,
                    "host": c.host,
                    "port": c.port,
                    "gpus": c.gpus,
                    "model": c.model,
                    "reputation": c.reputation,
                    "region": c.region,
                }
                for c in self._clusters.values()
            ]

    # ── Contribution Credits ──────────────────────────────────────────

    def record_contribution(
        self,
        user_id: str,
        cluster_id: str,
        gpu_hours: float = 0.0,
        tokens_served: int = 0,
    ) -> ContributionCredit:
        """Record GPU time contributed by a user."""
        key = f"{user_id}:{cluster_id}"
        with self._lock:
            if key not in self._credits:
                self._credits[key] = ContributionCredit(
                    user_id=user_id,
                    cluster_id=cluster_id,
                )
            credit = self._credits[key]
            credit.gpu_hours += gpu_hours
            credit.tokens_served += tokens_served
            credit.earned_credits += gpu_hours * 10  # 10 credits per GPU-hour

            # Update tier
            if credit.gpu_hours >= 1000:
                credit.tier = "platinum"
            elif credit.gpu_hours >= 100:
                credit.tier = "gold"
            elif credit.gpu_hours >= 10:
                credit.tier = "silver"
            else:
                credit.tier = "bronze"

            return credit

    def spend_credits(self, user_id: str, amount: float) -> bool:
        """Spend credits earned from contributions."""
        with self._lock:
            for key, credit in self._credits.items():
                if credit.user_id == user_id and credit.earned_credits - credit.spent_credits >= amount:
                    credit.spent_credits += amount
                    return True
            return False

    def get_user_credits(self, user_id: str) -> dict:
        """Get credit summary for a user."""
        with self._lock:
            total_earned = 0
            total_spent = 0
            total_gpu_hours = 0
            tier = "bronze"

            for credit in self._credits.values():
                if credit.user_id == user_id:
                    total_earned += credit.earned_credits
                    total_spent += credit.spent_credits
                    total_gpu_hours += credit.gpu_hours
                    if credit.tier == "platinum":
                        tier = "platinum"
                    elif credit.tier == "gold" and tier != "platinum":
                        tier = "gold"
                    elif credit.tier == "silver" and tier not in ("gold", "platinum"):
                        tier = "silver"

            return {
                "user_id": user_id,
                "total_earned": round(total_earned, 2),
                "total_spent": round(total_spent, 2),
                "balance": round(total_earned - total_spent, 2),
                "total_gpu_hours": round(total_gpu_hours, 2),
                "tier": tier,
            }

    def get_leaderboard(self, limit: int = 10) -> list[dict]:
        """Get top contributors by GPU hours."""
        with self._lock:
            user_hours: dict[str, float] = {}
            for credit in self._credits.values():
                user_hours[credit.user_id] = user_hours.get(credit.user_id, 0) + credit.gpu_hours

            sorted_users = sorted(user_hours.items(), key=lambda x: x[1], reverse=True)
            return [
                {"user_id": uid, "gpu_hours": round(hours, 2)}
                for uid, hours in sorted_users[:limit]
            ]
