from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class TrialResult:
    """Outcome of running a single trial configuration."""

    config: dict[str, Any]
    throughput_tok_s: float
    avg_latency_ms: float
    p99_latency_ms: float
    duration_seconds: float
    num_requests: int
    error_count: int = 0
    extra_metrics: dict[str, float] = field(default_factory=dict)


class TrialRunner:
    """Applies a configuration to the system and collects performance metrics.

    The runner:
    1. Takes a suggested config (param dict)
    2. Applies it via a callback (e.g., updates coordinator settings)
    3. Runs a warmup phase
    4. Executes a benchmark workload
    5. Collects and returns metrics
    """

    def __init__(
        self,
        apply_config: Callable[[dict[str, Any]], None],
        run_benchmark: Callable[[], TrialResult] | None = None,
        warmup_seconds: float = 5.0,
        cooldown_seconds: float = 2.0,
    ):
        self._apply_config = apply_config
        self._run_benchmark = run_benchmark
        self._warmup_seconds = warmup_seconds
        self._cooldown_seconds = cooldown_seconds
        self._results: list[TrialResult] = []

    def set_benchmark(self, benchmark_fn: Callable[[], TrialResult]) -> None:
        self._run_benchmark = benchmark_fn

    def run(self, config: dict[str, Any]) -> TrialResult | None:
        """Apply a config, warm up, run benchmark, return result."""
        logger.info(f"TrialRunner: applying config {config}")

        try:
            self._apply_config(config)
        except Exception as e:
            logger.error(f"TrialRunner: failed to apply config: {e}")
            return None

        if self._warmup_seconds > 0:
            logger.info(
                f"TrialRunner: warming up for {self._warmup_seconds}s..."
            )
            time.sleep(self._warmup_seconds)

        if self._run_benchmark is None:
            logger.warning("TrialRunner: no benchmark function set")
            return None

        try:
            result = self._run_benchmark()
            self._results.append(result)
            logger.info(
                f"TrialRunner: result — throughput={result.throughput_tok_s:.1f} tok/s, "
                f"avg_latency={result.avg_latency_ms:.1f}ms, "
                f"p99={result.p99_latency_ms:.1f}ms"
            )
            return result
        except Exception as e:
            logger.error(f"TrialRunner: benchmark failed: {e}")
            return None
        finally:
            if self._cooldown_seconds > 0 and self._results:
                time.sleep(self._cooldown_seconds)

    @property
    def results(self) -> list[TrialResult]:
        return list(self._results)

    def last_result(self) -> TrialResult | None:
        return self._results[-1] if self._results else None

    def best_result(self) -> TrialResult | None:
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.throughput_tok_s)
