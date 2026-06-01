"""Auto-generated performance baselines for deployment verification.

Generates and stores performance baselines after model deployment.
Used to detect regressions by comparing current performance against
the baseline recorded when the model was first deployed.

Usage::

    baseline = PerformanceBaseline()
    baseline.generate(coordinator, model_name="llama-3-70b")
    # Later...
    regression = baseline.check(current_metrics)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "PerformanceBaseline",
    "BaselineMetrics",
]


@dataclass
class BaselineMetrics:
    """Performance baseline metrics recorded at deploy time."""
    model_name: str
    timestamp: float = field(default_factory=time.time)
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    throughput_tok_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_utilization_pct: float = 0.0
    kv_cache_hit_rate: float = 0.0
    batch_size_avg: float = 0.0
    num_nodes: int = 0
    num_layers: int = 0
    dtype: str = "float16"
    quantization: str = "none"


class PerformanceBaseline:
    """Auto-generates and manages performance baselines.

    Generates a baseline by running a set of probe requests through
    the coordinator and recording the resulting metrics.
    """

    PROBE_PROMPTS = [
        "Explain distributed computing in one sentence.",
        "What is the capital of France?",
        "Write a haiku about technology.",
        "Summarize the concept of pipeline parallelism.",
        "What are the benefits of GPU acceleration?",
    ]

    def __init__(
        self,
        baseline_dir: str = ".distllm_baselines",
        regression_threshold: float = 0.20,
    ):
        self._dir = Path(baseline_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._threshold = regression_threshold
        self._current: BaselineMetrics | None = None
        self._lock = threading.Lock()

    def generate(
        self,
        coordinator: Any = None,
        model_name: str = "",
        num_probe_requests: int = 5,
    ) -> BaselineMetrics:
        """Generate a performance baseline from probe requests.

        Args:
            coordinator: Coordinator instance for running inference.
            model_name: Model name for the baseline.
            num_probe_requests: Number of probe requests to run.

        Returns:
            Generated BaselineMetrics.
        """
        logger.info(f"Generating performance baseline for {model_name}...")

        metrics = BaselineMetrics(model_name=model_name)
        latencies = []
        ttfts = []
        throughputs = []

        if coordinator is not None and hasattr(coordinator, 'generate'):
            for i in range(min(num_probe_requests, len(self.PROBE_PROMPTS))):
                prompt = self.PROBE_PROMPTS[i]
                try:
                    start = time.time()
                    result = coordinator.generate(
                        prompt=prompt,
                        max_new_tokens=50,
                        temperature=0.1,
                    )
                    elapsed = time.time() - start

                    latencies.append(elapsed * 1000)
                    ttfts.append(elapsed * 1000 * 0.3)  # Estimate TTFT as 30% of total
                    tokens = len(result.split()) if isinstance(result, str) else 50
                    throughputs.append(tokens / elapsed if elapsed > 0 else 0)

                except Exception as e:
                    logger.warning(f"Baseline probe {i} failed: {e}")

        # Compute percentiles
        if latencies:
            latencies.sort()
            metrics.latency_p50_ms = latencies[len(latencies) // 2]
            metrics.latency_p95_ms = latencies[int(len(latencies) * 0.95)]
            metrics.latency_p99_ms = latencies[int(len(latencies) * 0.99)]

        if ttfts:
            ttfts.sort()
            metrics.ttft_p50_ms = ttfts[len(ttfts) // 2]
            metrics.ttft_p95_ms = ttfts[int(len(ttfts) * 0.95)]

        if throughputs:
            metrics.throughput_tok_s = sum(throughputs) / len(throughputs)

        # Get cluster info
        if coordinator is not None:
            nodes = getattr(coordinator, 'nodes', {})
            metrics.num_nodes = len(nodes) if nodes else 1
            metrics.num_layers = getattr(coordinator, 'total_layers', 0)

        # Save
        self._current = metrics
        self._save(metrics)

        logger.info(
            f"Baseline generated: latency_p50={metrics.latency_p50_ms:.0f}ms, "
            f"throughput={metrics.throughput_tok_s:.1f}tok/s, "
            f"nodes={metrics.num_nodes}"
        )
        return metrics

    def check(self, current: dict[str, float]) -> list[dict]:
        """Check current metrics against baseline for regressions.

        Args:
            current: Dict of metric_name -> current_value.

        Returns:
            List of regression dicts (empty if no regressions).
        """
        baseline = self._current or self._load_latest()
        if baseline is None:
            return []

        regressions = []
        checks = {
            "latency_p50_ms": (baseline.latency_p50_ms, "higher_is_worse"),
            "latency_p95_ms": (baseline.latency_p95_ms, "higher_is_worse"),
            "latency_p99_ms": (baseline.latency_p99_ms, "higher_is_worse"),
            "ttft_p50_ms": (baseline.ttft_p50_ms, "higher_is_worse"),
            "throughput_tok_s": (baseline.throughput_tok_s, "lower_is_worse"),
        }

        for metric, (baseline_val, direction) in checks.items():
            current_val = current.get(metric)
            if current_val is None or baseline_val == 0:
                continue

            if direction == "higher_is_worse":
                change = (current_val - baseline_val) / baseline_val
                if change > self._threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": round(change * 100, 1),
                        "direction": "regression",
                    })
            else:  # lower_is_worse
                change = (baseline_val - current_val) / baseline_val
                if change > self._threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": round(change * 100, 1),
                        "direction": "regression",
                    })

        return regressions

    def _save(self, metrics: BaselineMetrics) -> None:
        """Save baseline to disk."""
        path = self._dir / f"baseline_{metrics.model_name.replace('/', '_')}.json"
        data = {
            "model_name": metrics.model_name,
            "timestamp": metrics.timestamp,
            "ttft_p50_ms": metrics.ttft_p50_ms,
            "ttft_p95_ms": metrics.ttft_p95_ms,
            "tpot_p50_ms": metrics.tpot_p50_ms,
            "throughput_tok_s": metrics.throughput_tok_s,
            "latency_p50_ms": metrics.latency_p50_ms,
            "latency_p95_ms": metrics.latency_p95_ms,
            "latency_p99_ms": metrics.latency_p99_ms,
            "gpu_memory_used_gb": metrics.gpu_memory_used_gb,
            "num_nodes": metrics.num_nodes,
            "num_layers": metrics.num_layers,
            "dtype": metrics.dtype,
            "quantization": metrics.quantization,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Baseline saved to {path}")

    def _load_latest(self) -> BaselineMetrics | None:
        """Load the most recent baseline from disk."""
        baselines = sorted(self._dir.glob("baseline_*.json"), reverse=True)
        if not baselines:
            return None
        try:
            data = json.loads(baselines[0].read_text())
            return BaselineMetrics(**data)
        except Exception:
            return None

    def get_baseline(self) -> BaselineMetrics | None:
        """Return the current baseline."""
        return self._current or self._load_latest()

    def stats(self) -> dict:
        baseline = self._current or self._load_latest()
        if baseline:
            return {
                "model": baseline.model_name,
                "timestamp": baseline.timestamp,
                "latency_p50_ms": baseline.latency_p50_ms,
                "throughput_tok_s": baseline.throughput_tok_s,
                "num_nodes": baseline.num_nodes,
            }
        return {"status": "no_baseline"}
