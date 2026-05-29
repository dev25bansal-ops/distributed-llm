"""N1: Cache benchmarking suite.

Measures per-tier hit rate, latency, throughput, and memory efficiency
with synthetic workloads (zipfian, bursty, periodic patterns).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class BenchmarkResult:
    """Results from a cache benchmark run."""
    workload: str
    duration_s: float
    total_requests: int
    cache_hits: int
    cache_misses: float
    hit_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    tier_stats: dict[str, dict[str, int]] = field(default_factory=dict)


class WorkloadGenerator:
    """Generates synthetic token sequences with different distribution patterns."""

    @staticmethod
    def zipfian(num_requests: int, vocab_size: int = 32000, prefix_len: int = 64) -> list[list[int]]:
        """Generate requests following Zipf's law (power-law distribution).

        Most requests reuse a small set of popular prefixes.
        """
        # Create a pool of popular prefixes
        num_unique = max(num_requests // 10, 10)
        prefix_pool = []
        for _ in range(num_unique):
            prefix = [random.randint(0, vocab_size - 1) for _ in range(prefix_len)]
            prefix_pool.append(prefix)

        # Sample with Zipfian weights
        requests = []
        for _ in range(num_requests):
            # Power-law: rank 1 is most popular
            rank = int(random.paretovariate(1.5))
            rank = min(rank, num_unique - 1)
            prefix = prefix_pool[rank]
            # Add some variation at the end
            suffix = [random.randint(0, vocab_size - 1) for _ in range(random.randint(0, 32))]
            requests.append(prefix + suffix)

        return requests

    @staticmethod
    def bursty(num_requests: int, vocab_size: int = 32000, prefix_len: int = 64) -> list[list[int]]:
        """Generate bursty requests — groups of similar requests.

        Simulates traffic spikes with repeated prefixes.
        """
        requests = []
        burst_size = max(num_requests // 20, 5)
        current_prefix = [random.randint(0, vocab_size - 1) for _ in range(prefix_len)]

        for i in range(num_requests):
            if i % burst_size == 0:
                # New burst — new prefix
                current_prefix = [random.randint(0, vocab_size - 1) for _ in range(prefix_len)]

            suffix = [random.randint(0, vocab_size - 1) for _ in range(random.randint(0, 16))]
            requests.append(current_prefix + suffix)

        return requests

    @staticmethod
    def periodic(num_requests: int, vocab_size: int = 32000, prefix_len: int = 64, period: int = 100) -> list[list[int]]:
        """Generate periodic requests — cycles through a fixed set of prefixes.

        Simulates cron jobs or scheduled tasks.
        """
        num_templates = min(period, 20)
        templates = [
            [random.randint(0, vocab_size - 1) for _ in range(prefix_len)]
            for _ in range(num_templates)
        ]

        requests = []
        for i in range(num_requests):
            template = templates[i % num_templates]
            suffix = [random.randint(0, vocab_size - 1) for _ in range(random.randint(0, 8))]
            requests.append(template + suffix)

        return requests

    @staticmethod
    def uniform(num_requests: int, vocab_size: int = 32000, prefix_len: int = 64) -> list[list[int]]:
        """Generate uniformly random requests (no caching benefit)."""
        requests = []
        for _ in range(num_requests):
            tokens = [random.randint(0, vocab_size - 1) for _ in range(prefix_len + random.randint(0, 32))]
            requests.append(tokens)
        return requests


class CacheBench:
    """Cache benchmarking suite.

    Measures hit rate, latency, throughput across different workloads
    and tier configurations.
    """

    def __init__(self, cache_manager: Any = None):
        self._cache_manager = cache_manager
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        workload: str = "zipfian",
        num_requests: int = 1000,
        rate_limit: int = 0,
        tiers: list[str] | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark with the specified workload.

        Args:
            workload: Workload type ("zipfian", "bursty", "periodic", "uniform").
            num_requests: Number of requests to generate.
            rate_limit: Max requests per second (0 = unlimited).
            tiers: Tiers to enable (None = all).

        Returns:
            BenchmarkResult with metrics.
        """
        # Generate workload
        gen = WorkloadGenerator()
        if workload == "zipfian":
            requests = gen.zipfian(num_requests)
        elif workload == "bursty":
            requests = gen.bursty(num_requests)
        elif workload == "periodic":
            requests = gen.periodic(num_requests)
        elif workload == "uniform":
            requests = gen.uniform(num_requests)
        else:
            raise ValueError(f"Unknown workload: {workload}")

        # Run benchmark
        latencies: list[float] = []
        hits = 0
        misses = 0
        start_time = time.time()

        for i, tokens in enumerate(requests):
            req_start = time.time()

            if self._cache_manager is not None:
                match_len, _ = self._cache_manager.lookup_prefix(tokens)
                if match_len > 0:
                    hits += 1
                else:
                    misses += 1
                    # Store for future hits
                    self._cache_manager.store_prefix(tokens, {"benchmark": i})

            latency_ms = (time.time() - req_start) * 1000
            latencies.append(latency_ms)

            # Rate limiting
            if rate_limit > 0:
                expected_time = (i + 1) / rate_limit
                actual_time = time.time() - start_time
                if actual_time < expected_time:
                    time.sleep(expected_time - actual_time)

        duration = time.time() - start_time

        # Compute percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        result = BenchmarkResult(
            workload=workload,
            duration_s=duration,
            total_requests=num_requests,
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hits / max(hits + misses, 1),
            avg_latency_ms=sum(latencies) / max(n, 1),
            p50_latency_ms=sorted_latencies[n // 2] if n > 0 else 0,
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            throughput_rps=num_requests / max(duration, 0.001),
        )

        # Get tier stats if available
        if self._cache_manager and hasattr(self._cache_manager, 'get_tier_stats'):
            result.tier_stats = self._cache_manager.get_tier_stats()

        self._results.append(result)
        return result

    def run_all_workloads(self, num_requests: int = 1000) -> list[BenchmarkResult]:
        """Run benchmarks for all workload types."""
        results = []
        for workload in ["zipfian", "bursty", "periodic", "uniform"]:
            result = self.run(workload=workload, num_requests=num_requests)
            results.append(result)
            logger.info(
                f"{workload}: hit_rate={result.hit_rate:.2%}, "
                f"avg_latency={result.avg_latency_ms:.2f}ms, "
                f"throughput={result.throughput_rps:.0f} rps"
            )
        return results

    def get_results(self) -> list[BenchmarkResult]:
        """Return all benchmark results."""
        return list(self._results)

    def compare(self, baseline: BenchmarkResult, current: BenchmarkResult) -> dict:
        """Compare two benchmark results.

        Returns:
            Dict with comparison metrics.
        """
        return {
            "hit_rate_change": current.hit_rate - baseline.hit_rate,
            "latency_change_ms": current.avg_latency_ms - baseline.avg_latency_ms,
            "throughput_change_rps": current.throughput_rps - baseline.throughput_rps,
            "p95_latency_change_ms": current.p95_latency_ms - baseline.p95_latency_ms,
        }
