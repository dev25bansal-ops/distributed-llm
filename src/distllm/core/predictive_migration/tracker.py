from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrefixRecord:
    """A tracked prefix pattern with frequency and recency."""
    prefix_hash: str
    prefix_tokens: tuple[int, ...]
    frequency: float = 1.0
    last_seen: float = field(default_factory=time.time)
    hit_count: int = 1
    cluster: str = "default"


class PrefixFrequencyTracker:
    """Tracks prompt prefix frequency per cluster with time decay.

    Observes token sequences, extracts prefixes of configurable minimum
    length, and maintains frequency counts that decay exponentially over
    time. Supports grouping by cluster for multi-cluster deployments.

    Usage:
        tracker = PrefixFrequencyTracker(min_prefix_len=8)
        tracker.observe([101, 205, 309, ...], cluster="us-east")
        top = tracker.top_prefixes(k=10)
    """

    def __init__(
        self,
        min_prefix_len: int = 8,
        max_prefixes: int = 10000,
        decay_hours: float = 24.0,
    ):
        self._min_prefix_len = min_prefix_len
        self._max_prefixes = max_prefixes
        self._decay_seconds = decay_hours * 3600
        self._records: dict[str, PrefixRecord] = {}
        self._last_decay_time: float = time.time()

    def observe(self, token_ids: list[int], cluster: str = "default") -> str | None:
        self._maybe_decay()

        if len(token_ids) < self._min_prefix_len:
            return None

        prefix = tuple(token_ids[: self._min_prefix_len])
        prefix_hash = self._hash_prefix(prefix)

        if prefix_hash in self._records:
            record = self._records[prefix_hash]
            record.frequency += 1.0
            record.hit_count += 1
            record.last_seen = time.time()
        else:
            if len(self._records) >= self._max_prefixes:
                self._evict_lowest_score()
            self._records[prefix_hash] = PrefixRecord(
                prefix_hash=prefix_hash,
                prefix_tokens=prefix,
                cluster=cluster,
            )

        return prefix_hash

    def observe_batch(
        self, token_ids_list: list[list[int]], cluster: str = "default"
    ) -> list[str | None]:
        return [self.observe(tokens, cluster) for tokens in token_ids_list]

    def top_prefixes(self, k: int = 20) -> list[PrefixRecord]:
        scored = [(rec, self._score(rec)) for rec in self._records.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in scored[:k]]

    def get_frequency(self, token_ids: list[int]) -> float:
        if len(token_ids) < self._min_prefix_len:
            return 0.0
        prefix = tuple(token_ids[: self._min_prefix_len])
        prefix_hash = self._hash_prefix(prefix)
        record = self._records.get(prefix_hash)
        if record is None:
            return 0.0
        return self._score(record)

    def get_by_cluster(self, cluster: str) -> list[PrefixRecord]:
        return [r for r in self._records.values() if r.cluster == cluster]

    def _maybe_decay(self) -> None:
        now = time.time()
        elapsed = now - self._last_decay_time
        if elapsed < self._decay_seconds * 0.1:
            return

        factor = math.exp(-elapsed / self._decay_seconds)
        for record in self._records.values():
            record.frequency *= factor
        self._last_decay_time = now

    def _score(self, record: PrefixRecord) -> float:
        recency = math.exp(
            -(time.time() - record.last_seen) / self._decay_seconds
        )
        freq_norm = math.log1p(record.frequency)
        return 0.6 * recency + 0.4 * freq_norm

    def _evict_lowest_score(self) -> None:
        if not self._records:
            return
        scored = [(h, self._score(r)) for h, r in self._records.items()]
        scored.sort(key=lambda x: x[1])
        to_evict = max(1, len(self._records) // 10)
        for h, _ in scored[:to_evict]:
            self._records.pop(h, None)

    def _hash_prefix(self, prefix: tuple[int, ...]) -> str:
        raw = ",".join(str(t) for t in prefix)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def total_prefixes(self) -> int:
        return len(self._records)

    def stats(self) -> dict[str, Any]:
        top = self.top_prefixes(5)
        return {
            "total_prefixes": self.total_prefixes,
            "min_prefix_len": self._min_prefix_len,
            "max_prefixes": self._max_prefixes,
            "decay_hours": self._decay_seconds / 3600,
            "top_prefixes": [
                {
                    "hash": r.prefix_hash,
                    "frequency": round(r.frequency, 1),
                    "hit_count": r.hit_count,
                    "cluster": r.cluster,
                    "last_seen_ago": round(time.time() - r.last_seen, 1),
                }
                for r in top
            ],
        }

    def reset(self) -> None:
        self._records.clear()
        self._last_decay_time = time.time()
