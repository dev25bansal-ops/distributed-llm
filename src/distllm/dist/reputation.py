"""GPU Reputation System for peer quality tracking.

Each peer in the cluster gets a reputation score (0.0–1.0) based on:
  - Reliability: ratio of successful requests to total requests
  - Speed: average tokens/sec compared to cluster median
  - Uptime: ratio of healthy health checks to total health checks
  - Liveness: how recently the node was seen (avoids stale entries)

Higher reputation → more requests routed to this peer.
Requesters can filter by minimum reputation threshold.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ReputationRecord:
    """Per-node reputation metrics and contribution credits."""
    node_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    health_check_passes: int = 0
    health_check_fails: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_failure: float = 0.0
    # Contribution credits / token economy
    tokens_contributed: int = 0  # Total tokens computed for others
    credits_earned: float = 0.0  # Credits earned from contributions
    credits_spent: float = 0.0  # Credits spent on using others' compute

    @property
    def reliability(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def health_ratio(self) -> float:
        total = self.health_check_passes + self.health_check_fails
        if total == 0:
            return 1.0
        return self.health_check_passes / total

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.first_seen) / 3600.0


class ReputationSystem:
    """Tracks and scores peer node reputations.

    Scores range from 0.0 (untrusted) to 1.0 (highly reliable).
    The score is a weighted combination of:
      - 40%: Reliability (success / total requests)
      - 25%: Health check pass ratio
      - 20%: Speed relative to cluster median
      - 15%: Uptime bonus (longer uptime = slightly higher trust)

    Usage:
        system = ReputationSystem()
        system.record_success("node_a", latency_ms=45.0, tokens=128)
        system.record_failure("node_b")
        system.record_health("node_a", healthy=True)
        score = system.get_score("node_a")  # 0.0–1.0
    """

    def __init__(self, min_reputation: float = 0.0):
        self._records: dict[str, ReputationRecord] = {}
        self._min_reputation = min_reputation
        self._lock = threading.RLock()
        self._weights = {
            "reliability": 0.40,
            "health": 0.25,
            "speed": 0.20,
            "uptime": 0.15,
        }

    def record_success(self, node_id: str, latency_ms: float = 0.0,
                       tokens: int = 0) -> None:
        """Record a successful request completion."""
        rec = self._get_or_create(node_id)
        rec.total_requests += 1
        rec.successful_requests += 1
        rec.total_latency_ms += latency_ms
        rec.total_tokens += tokens
        rec.last_seen = time.time()

    def record_failure(self, node_id: str) -> None:
        """Record a failed request."""
        rec = self._get_or_create(node_id)
        rec.total_requests += 1
        rec.failed_requests += 1
        rec.last_seen = time.time()
        rec.last_failure = time.time()

    def record_health(self, node_id: str, healthy: bool) -> None:
        """Record a health check result."""
        rec = self._get_or_create(node_id)
        rec.last_seen = time.time()
        if healthy:
            rec.health_check_passes += 1
        else:
            rec.health_check_fails += 1

    def get_score(self, node_id: str) -> float:
        """Get the reputation score (0.0–1.0) for a node."""
        rec = self._records.get(node_id)
        if rec is None or rec.total_requests < 1:
            return 0.5  # neutral score for unknown nodes

        if rec.failed_requests >= 5 and rec.failed_requests > rec.successful_requests:
            return 0.1  # persistently failing node

        reliability = rec.reliability * self._weights["reliability"]
        health = rec.health_ratio * self._weights["health"]

        speed_score = self._compute_speed_score(rec)
        speed = speed_score * self._weights["speed"]

        uptime_bonus = min(rec.uptime_hours / 24.0, 1.0) * self._weights["uptime"]

        score = reliability + health + speed + uptime_bonus
        return max(0.0, min(1.0, score))

    def get_scores(self) -> dict[str, float]:
        """Return reputation scores for all nodes."""
        with self._lock:
            return {nid: self.get_score(nid) for nid in self._records}

    # ── Contribution credits / token economy ──

    def record_contribution(self, node_id: str, tokens_computed: int, credit_rate: float = 1.0) -> None:
        """Record that *node_id* contributed compute for *tokens_computed* tokens.

        *credit_rate* determines how many credits per token (can vary by
        GPU type — A100s earn more than RTX 4060s).
        """
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.tokens_contributed += tokens_computed
            rec.credits_earned += tokens_computed * credit_rate

    def spend_credits(self, node_id: str, tokens_consumed: float) -> bool:
        """Spend credits for consuming compute. Returns True if sufficient."""
        with self._lock:
            rec = self._get_or_create(node_id)
            balance = rec.credits_earned - rec.credits_spent
            if balance >= tokens_consumed:
                rec.credits_spent += tokens_consumed
                return True
            return False

    def get_credit_balance(self, node_id: str) -> float:
        with self._lock:
            rec = self._get_or_create(node_id)
            return rec.credits_earned - rec.credits_spent

    def get_credit_summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                nid: {
                    "tokens_contributed": rec.tokens_contributed,
                    "credits_earned": round(rec.credits_earned, 1),
                    "credits_spent": round(rec.credits_spent, 1),
                    "credit_balance": round(rec.credits_earned - rec.credits_spent, 1),
                }
                for nid, rec in self._records.items()
            }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked reputations."""
        summaries = {}
        for nid, rec in self._records.items():
            summaries[nid] = {
                "score": round(self.get_score(nid), 3),
                "reliability": round(rec.reliability, 3),
                "health_ratio": round(rec.health_ratio, 3),
                "total_requests": rec.total_requests,
                "failed_requests": rec.failed_requests,
                "uptime_hours": round(rec.uptime_hours, 2),
            }
        return {
            "min_reputation": self._min_reputation,
            "nodes": summaries,
            "weights": self._weights,
        }

    def is_qualified(self, node_id: str) -> bool:
        """Check if a node meets the minimum reputation threshold."""
        return self.get_score(node_id) >= self._min_reputation

    def set_min_reputation(self, threshold: float) -> None:
        self._min_reputation = max(0.0, min(1.0, threshold))

    def _get_or_create(self, node_id: str) -> ReputationRecord:
        if node_id not in self._records:
            self._records[node_id] = ReputationRecord(node_id=node_id)
        return self._records[node_id]

    def _compute_speed_score(self, rec: ReputationRecord) -> float:
        """Score speed relative to cluster median (0.0–1.0)."""
        with self._lock:
            all_latencies = [
                r.avg_latency_ms for r in self._records.values()
                if r.successful_requests > 0
            ]
            if len(all_latencies) < 2:
                return 0.5
            median = statistics.median(all_latencies)
            if median <= 0 or rec.avg_latency_ms <= 0:
                return 0.5
            ratio = median / rec.avg_latency_ms
            return max(0.0, min(1.0, ratio))
