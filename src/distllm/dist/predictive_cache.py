"""Predictive KV Cache Management: predicts prefix reuse patterns and pre-warms cache."""


from __future__ import annotations
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger


@dataclass
class CacheTier:
    name: str
    capacity_bytes: int = 0
    used_bytes: int = 0
    read_latency_ms: float = 0.1
    write_latency_ms: float = 0.1


@dataclass
class PrefixPattern:
    prefix_tokens: tuple[int, ...]
    frequency: int = 0
    last_seen: float = 0.0
    avg_match_length: float = 0.0
    hit_count: int = 0
    score: float = 0.0


@dataclass
class CachePrediction:
    prefix_tokens: tuple[int, ...]
    predicted_matches: int = 0
    confidence: float = 0.0
    should_prefetch: bool = False
    target_tier: str = "gpu"


class PatternLearner:
    def __init__(self, max_patterns: int = 10000, min_prefix_len: int = 8, decay_hours: float = 24.0):
        self.max_patterns = max_patterns
        self.min_prefix_len = min_prefix_len
        self.decay_seconds = decay_hours * 3600
        self._patterns: dict[tuple[int, ...], PrefixPattern] = {}
        self._recent_prefixes: deque = deque(maxlen=1000)
        self._token_frequencies: dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

        # E9: Adaptive scoring weights (learned from hit/miss feedback)
        self._recency_weight = 0.6
        self._frequency_weight = 0.4
        self._feedback_hits = 0
        self._feedback_misses = 0

    def observe(self, token_ids: list[int]) -> None:
        if len(token_ids) < self.min_prefix_len:
            return
        prefix = tuple(token_ids[:self.min_prefix_len])
        now = time.time()
        with self._lock:
            if prefix not in self._patterns:
                self._patterns[prefix] = PrefixPattern(
                    prefix_tokens=prefix,
                    last_seen=now,
                )
                if len(self._patterns) > self.max_patterns:
                    self._evict_lowest_score()
            else:
                pattern = self._patterns[prefix]
                elapsed = now - pattern.last_seen
                if elapsed > 0:
                    decay = max(0.5, 1.0 - elapsed / self.decay_seconds)
                    pattern.frequency = int(pattern.frequency * decay + 1)
                else:
                    pattern.frequency += 1
                pattern.last_seen = now
                pattern.hit_count += 1
            for tok in token_ids:
                self._token_frequencies[tok] += 1
            self._recent_prefixes.append(prefix)

    def _evict_lowest_score(self) -> None:
        if not self._patterns:
            return
        self._score_all()
        oldest = min(self._patterns.items(), key=lambda x: x[1].score)
        del self._patterns[oldest[0]]

    def _score_all(self) -> None:
        now = time.time()
        max_freq = max((p.frequency for p in self._patterns.values()), default=1)
        max_freq = max(max_freq, 1)

        # E9: Adapt weights based on hit/miss feedback
        total_feedback = self._feedback_hits + self._feedback_misses
        if total_feedback > 100:
            hit_rate = self._feedback_hits / total_feedback
            # If hit rate is low, favor recency (patterns change fast)
            # If hit rate is high, favor frequency (patterns are stable)
            self._recency_weight = max(0.3, min(0.8, 0.5 + (0.5 - hit_rate)))
            self._frequency_weight = 1.0 - self._recency_weight

        for pattern in self._patterns.values():
            recency = max(0.0, 1.0 - (now - pattern.last_seen) / self.decay_seconds)
            freq_norm = min(1.0, pattern.frequency / max_freq)
            pattern.score = self._recency_weight * recency + self._frequency_weight * freq_norm

    def record_feedback(self, was_hit: bool) -> None:
        """Record hit/miss feedback for adaptive scoring (E9)."""

        with self._lock:
            if was_hit:
                self._feedback_hits += 1
            else:
                self._feedback_misses += 1

    def predict(self, token_ids: list[int]) -> list[CachePrediction]:
        predictions = []
        with self._lock:
            self._score_all()
            for prefix, pattern in sorted(self._patterns.items(), key=lambda x: x[1].score, reverse=True)[:50]:
                match_len = self._compute_match_len(prefix, token_ids)
                if match_len >= self.min_prefix_len:
                    predictions.append(CachePrediction(
                        prefix_tokens=prefix,
                        predicted_matches=pattern.hit_count,
                        confidence=pattern.score,
                        should_prefetch=pattern.score > 0.3 and match_len >= self.min_prefix_len,
                        target_tier="gpu" if pattern.score > 0.6 else "cpu",
                    ))
        return predictions

    def _compute_match_len(self, prefix: tuple[int, ...], tokens: list[int]) -> int:
        if len(tokens) < self.min_prefix_len:
            return 0
        for i in range(min(len(prefix), len(tokens))):
            if prefix[i] != tokens[i]:
                return i
        return min(len(prefix), len(tokens))

    def top_patterns(self, n: int = 20) -> list[PrefixPattern]:
        with self._lock:
            self._score_all()
            return [p for _, p in sorted(self._patterns.items(), key=lambda x: x[1].score, reverse=True)[:n]]

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def save_patterns(self, path: str) -> None:
        """E11: Save patterns to disk for persistence across restarts."""

        import json
        data = {
            "patterns": {
                str(list(k)): {
                    "frequency": p.frequency,
                    "last_seen": p.last_seen,
                    "hit_count": p.hit_count,
                    "score": p.score,
                }
                for k, p in self._patterns.items()
            },
            "recency_weight": self._recency_weight,
            "frequency_weight": self._frequency_weight,
            "feedback_hits": self._feedback_hits,
            "feedback_misses": self._feedback_misses,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_patterns(self, path: str) -> None:
        """E11: Load patterns from disk."""

        import json
        from pathlib import Path
        if not Path(path).exists():
            return
        with open(path) as f:
            data = json.load(f)
        with self._lock:
            for key_str, pdata in data.get("patterns", {}).items():
                prefix = tuple(int(x) for x in key_str.strip("[]").split(","))
                self._patterns[prefix] = PrefixPattern(
                    prefix_tokens=prefix,
                    frequency=pdata.get("frequency", 0),
                    last_seen=pdata.get("last_seen", 0),
                    hit_count=pdata.get("hit_count", 0),
                    score=pdata.get("score", 0),
                )
            self._recency_weight = data.get("recency_weight", 0.6)
            self._frequency_weight = data.get("frequency_weight", 0.4)
            self._feedback_hits = data.get("feedback_hits", 0)
            self._feedback_misses = data.get("feedback_misses", 0)


class PredictiveCacheManager:
    def __init__(
        self,
        gpu_cache: Any = None,
        cpu_cache: Any = None,
        disk_cache: Any = None,
        gpu_memory_bytes: int = 512 * 1024 * 1024,
        cpu_memory_bytes: int = 4 * 1024 * 1024 * 1024,
        disk_path: str = "",
    ):
        self.learner = PatternLearner()
        self._gpu_cache = gpu_cache
        self._cpu_cache = cpu_cache
        self._disk_cache = disk_cache
        self._tiers = {
            "gpu": CacheTier(name="gpu", capacity_bytes=gpu_memory_bytes, read_latency_ms=0.1, write_latency_ms=0.1),
            "cpu": CacheTier(name="cpu", capacity_bytes=cpu_memory_bytes, read_latency_ms=10.0, write_latency_ms=10.0),
            "disk": CacheTier(name="disk", capacity_bytes=100 * 1024 * 1024 * 1024, read_latency_ms=100.0, write_latency_ms=100.0),
        }
        self._disk_path = Path(disk_path) if disk_path else None
        if self._disk_path:
            self._disk_path.mkdir(parents=True, exist_ok=True)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {
            "prefetches": 0,
            "prefetch_hits": 0,
            "gpu_hits": 0,
            "cpu_hits": 0,
            "disk_hits": 0,
            "misses": 0,
        }
        self._lock = threading.Lock()

    def observe_request(self, token_ids: list[int]) -> list[CachePrediction]:
        predictions = self.learner.predict(token_ids)
        self.learner.observe(token_ids)
        to_prefetch = [p for p in predictions if p.should_prefetch]
        if to_prefetch:
            self._start_prefetch(to_prefetch)
        return predictions

    def _start_prefetch(self, predictions: list[CachePrediction]) -> None:
        import queue
        if not hasattr(self, '_prefetch_queue') or self._prefetch_queue is None:
            self._prefetch_queue: queue.Queue[list[CachePrediction]] = queue.Queue(maxsize=1000)

        if not self._running:
            self._running = True
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self._prefetch_thread.start()

        # E10: Batch related prefixes by target tier
        gpu_batch = [p for p in predictions if p.target_tier == "gpu"]
        cpu_batch = [p for p in predictions if p.target_tier == "cpu"]
        for batch in [gpu_batch, cpu_batch]:
            if batch:
                self._prefetch_queue.put(batch)

    def _prefetch_worker(self) -> None:
        import queue

        while self._running:
            try:
                batch: list[CachePrediction] = self._prefetch_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # E10: Process batch of related prefixes together
            for pred in batch:
                try:
                    kv_data = None
                    if self._cpu_cache is not None and hasattr(self._cpu_cache, 'lookup'):
                        match_len, kv_data = self._cpu_cache.lookup(list(pred.prefix_tokens))
                        if match_len > 0 and kv_data is not None:
                            if self._gpu_cache is not None and hasattr(self._gpu_cache, 'store'):
                                self._gpu_cache.store(list(pred.prefix_tokens)[:match_len], kv_data)
                                with self._lock:
                                    self._stats["prefetch_hits"] += 1
                                logger.debug(f"Prefetched prefix from CPU->GPU ({match_len} tokens)")
                                continue

                    if kv_data is None and self._disk_cache is not None and hasattr(self._disk_cache, 'lookup'):
                        match_len, kv_data = self._disk_cache.lookup(list(pred.prefix_tokens))
                        if match_len > 0 and kv_data is not None:
                            if self._gpu_cache is not None and hasattr(self._gpu_cache, 'store'):
                                self._gpu_cache.store(list(pred.prefix_tokens)[:match_len], kv_data)
                                with self._lock:
                                    self._stats["prefetch_hits"] += 1
                                logger.debug(f"Prefetched prefix from Disk->GPU ({match_len} tokens)")
                except Exception as e:
                    logger.debug(f"Prefetch failed for prefix: {e}")

    def start_prefetch_service(self) -> None:
        import queue
        self._prefetch_queue: queue.Queue[CachePrediction] = queue.Queue(maxsize=1000)
        self._running = True
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self._prefetch_thread.start()
        logger.info("PredictiveCache prefetch service started")

    def stop_prefetch_service(self) -> None:
        self._running = False
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5)
            self._prefetch_thread = None
        logger.info("PredictiveCache prefetch service stopped")

    def lookup(self, token_ids: list[int]) -> tuple[int, Any]:
        tiers = [("gpu", self._gpu_cache), ("cpu", self._cpu_cache), ("disk", self._disk_cache)]
        for tier_name, cache in tiers:
            if cache is None:
                continue
            try:
                result = cache.lookup(token_ids) if hasattr(cache, 'lookup') else (0, None)
                if isinstance(result, tuple) and len(result) == 2 and result[0] > 0:
                    with self._lock:
                        self._stats[f"{tier_name}_hits"] += 1
                        if tier_name != "gpu":
                            self._promote_to_gpu(result[0], result[1], token_ids)
                    return result
            except Exception:
                pass
        with self._lock:
            self._stats["misses"] += 1
        return (0, None)

    def _promote_to_gpu(self, match_len: int, kv_data: Any, tokens: list[int] | None = None) -> None:
        if self._gpu_cache is not None and hasattr(self._gpu_cache, 'store'):
            try:
                actual_tokens = tokens[:match_len] if tokens is not None else list(range(match_len))
                self._gpu_cache.store(actual_tokens, kv_data)
            except Exception:
                pass

    def store(self, token_ids: list[int], kv_data: Any, tier: str = "gpu") -> bool:
        cache = {"gpu": self._gpu_cache, "cpu": self._cpu_cache, "disk": self._disk_cache}.get(tier)
        if cache is None:
            return False
        try:
            if hasattr(cache, 'store'):
                cache.store(token_ids, kv_data)
                return True
        except Exception as e:
            logger.debug(f"Failed to store in {tier} cache: {e}")
        return False

    def get_cold_prefixes(self) -> list[tuple[int, ...]]:
        cold = []
        for pattern in self.learner.top_patterns(100):
            if pattern.score < 0.2:
                cold.append(pattern.prefix_tokens)
        return cold

    def compress_to_disk(self) -> int:
        cold = self.get_cold_prefixes()
        count = 0
        for prefix in cold:
            if self._gpu_cache is not None and hasattr(self._gpu_cache, 'lookup'):
                try:
                    _, kv_data = self._gpu_cache.lookup(list(prefix))
                    if kv_data is not None and self._disk_cache is not None:
                        self._disk_cache.store(list(prefix), kv_data) if hasattr(self._disk_cache, 'store') else None
                        if hasattr(self._gpu_cache, 'evict'):
                            self._gpu_cache.evict(list(prefix))
                        count += 1
                except Exception:
                    pass
        if count > 0:
            logger.info(f"Compressed {count} cold prefixes to disk")
        return count

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def hit_rate(self) -> float:
        total = sum(self._stats.get(k, 0) for k in ("gpu_hits", "cpu_hits", "disk_hits", "misses"))
        hits = sum(self._stats.get(k, 0) for k in ("gpu_hits", "cpu_hits", "disk_hits"))
        return hits / max(total, 1)

    def start_background_compression(self, interval_s: float = 300.0) -> None:
        self._running = True
        def _loop():
            while self._running:
                time.sleep(interval_s)
                try:
                    self.compress_to_disk()
                except Exception as e:
                    logger.warning(f"Background compression failed: {e}")
        self._prefetch_thread = threading.Thread(target=_loop, daemon=True)
        self._prefetch_thread.start()

    def stop(self) -> None:
        self._running = False
