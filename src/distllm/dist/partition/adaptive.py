"""Online adaptive re-partitioning with straggler feedback.

Continuously monitors pipeline performance and triggers re-partitioning
when a straggler node is detected.  Integrates with the DP solver to
find a new partition that accounts for the degraded node.

Flow::

    StragglerDetector → AdaptiveRepartitioner → PartitionOptimizer
         (real-time)       (decision logic)       (DP re-solve)
                                ↓
                         Live partition migration

Typical usage::

    repartitioner = AdaptiveRepartitioner(
        cost_model=cost_model,
        node_ids=["gpu-0", "gpu-1", "gpu-2"],
        straggler_threshold=1.5,
    )
    repartitioner.start_monitoring()
    # ... during inference ...
    new_solution = repartitioner.check_and_repartition(observed_latencies)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution


class RepartitionTrigger(str, Enum):
    NONE = "none"
    STRAGGLER = "straggler"
    OOM = "oom"
    THROUGHPUT_DROP = "throughput_drop"
    MANUAL = "manual"


@dataclass
class LatencySample:
    """A single latency measurement from a node."""
    node_id: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    batch_size: int = 1
    seq_len: int = 4096


@dataclass
class StragglerReport:
    """Report of a detected straggler."""
    node_id: str
    observed_latency_ms: float
    expected_latency_ms: float
    ratio: float
    severity: float
    trigger: RepartitionTrigger
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepartitionEvent:
    """Record of a re-partitioning event."""
    trigger: RepartitionTrigger
    old_solution: PartitionSolution | None
    new_solution: PartitionSolution
    straggler_report: StragglerReport | None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


class StragglerDetector:
    """Detects slow nodes using sliding window statistics.

    Uses a combination of:
    - Absolute threshold: node latency exceeds a fixed multiple of expected
    - Relative threshold: node latency exceeds median by a factor
    - Trend detection: node latency increasing over time

    Args:
        window_size: Number of recent samples to consider.
        abs_threshold: Absolute latency multiplier threshold.
        rel_threshold: Relative to median threshold.
        trend_window: Samples for trend detection.
        trend_slope_threshold: Minimum slope to trigger trend alert.
    """

    def __init__(
        self,
        window_size: int = 100,
        abs_threshold: float = 1.5,
        rel_threshold: float = 2.0,
        trend_window: int = 20,
        trend_slope_threshold: float = 0.5,
    ):
        self._window_size = window_size
        self._abs_threshold = abs_threshold
        self._rel_threshold = rel_threshold
        self._trend_window = trend_window
        self._trend_slope = trend_slope_threshold
        self._samples: dict[str, deque[LatencySample]] = {}
        self._expected: dict[str, float] = {}

    def set_expected(self, node_latencies: dict[str, float]) -> None:
        """Set expected latency for each node (from partition solution)."""
        self._expected = dict(node_latencies)

    def record(self, sample: LatencySample) -> StragglerReport | None:
        """Record a latency sample and check for straggler."""
        if sample.node_id not in self._samples:
            self._samples[sample.node_id] = deque(maxlen=self._window_size)
        self._samples[sample.node_id].append(sample)

        return self._check_node(sample.node_id)

    def _check_node(self, node_id: str) -> StragglerReport | None:
        samples = self._samples.get(node_id)
        if not samples or len(samples) < 3:
            return None

        recent = list(samples)[-min(len(samples), 20):]
        avg_latency = sum(s.latency_ms for s in recent) / len(recent)
        expected = self._expected.get(node_id, avg_latency)

        if expected <= 0:
            return None

        ratio = avg_latency / expected

        if ratio >= self._abs_threshold:
            severity = min((ratio - 1.0) / (self._abs_threshold - 1.0), 1.0)
            return StragglerReport(
                node_id=node_id,
                observed_latency_ms=avg_latency,
                expected_latency_ms=expected,
                ratio=ratio,
                severity=severity,
                trigger=RepartitionTrigger.STRAGGLER,
            )

        all_nodes = list(self._samples.keys())
        medians: list[float] = []
        for nid in all_nodes:
            ns = list(self._samples[nid])[-min(len(self._samples[nid]), 20):]
            if ns:
                sorted_lat = sorted(s.latency_ms for s in ns)
                medians.append(sorted_lat[len(sorted_lat) // 2])

        if medians:
            median_latency = sorted(medians)[len(medians) // 2]
            rel_ratio = avg_latency / max(median_latency, 0.001)
            if rel_ratio >= self._rel_threshold:
                severity = min((rel_ratio - 1.0) / (self._rel_threshold - 1.0), 1.0)
                return StragglerReport(
                    node_id=node_id,
                    observed_latency_ms=avg_latency,
                    expected_latency_ms=expected,
                    ratio=rel_ratio,
                    severity=severity,
                    trigger=RepartitionTrigger.STRAGGLER,
                )

        if len(recent) >= self._trend_window:
            trend_samples = recent[-self._trend_window:]
            slope = self._compute_slope(trend_samples)
            if slope > self._trend_slope:
                severity = min(slope / (self._trend_slope * 3), 1.0)
                return StragglerReport(
                    node_id=node_id,
                    observed_latency_ms=avg_latency,
                    expected_latency_ms=expected,
                    ratio=ratio,
                    severity=severity,
                    trigger=RepartitionTrigger.STRAGGLER,
                )

        return None

    def _compute_slope(self, samples: list[LatencySample]) -> float:
        n = len(samples)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(s.latency_ms for s in samples) / n
        num = sum((i - x_mean) * (samples[i].latency_ms - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / max(den, 1e-10)

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get current statistics for all nodes."""
        stats: dict[str, dict[str, float]] = {}
        for node_id, samples in self._samples.items():
            if not samples:
                continue
            recent = list(samples)[-min(len(samples), 20):]
            latencies = [s.latency_ms for s in recent]
            sorted_lat = sorted(latencies)
            stats[node_id] = {
                "mean_ms": sum(latencies) / len(latencies),
                "median_ms": sorted_lat[len(sorted_lat) // 2],
                "p99_ms": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[-1],
                "min_ms": sorted_lat[0],
                "max_ms": sorted_lat[-1],
                "samples": len(samples),
            }
        return stats


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive re-partitioning."""
    enabled: bool = True
    straggler_threshold: float = 1.5
    min_repartition_interval_s: float = 30.0
    cooldown_after_repartition_s: float = 60.0
    max_repartitions_per_hour: int = 10
    require_quorum: bool = True
    quorum_fraction: float = 0.5


class AdaptiveRepartitioner:
    """Monitors pipeline and triggers re-partitioning when needed.

    Integrates with the DP solver to re-run partition optimization
    with adjusted cost estimates when a straggler is detected.

    Args:
        cost_model: Cost model for partition optimization.
        node_ids: List of node identifiers.
        config: Adaptive re-partitioning configuration.
        on_repartition: Callback when re-partitioning occurs.
    """

    def __init__(
        self,
        cost_model: PartitionCostModel,
        node_ids: list[str],
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        config: AdaptiveConfig | None = None,
        on_repartition: Callable[[RepartitionEvent], None] | None = None,
    ):
        self._cost_model = cost_model
        self._node_ids = list(node_ids)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom
        self._config = config or AdaptiveConfig()
        self._on_repartition = on_repartition

        self._detector = StragglerDetector(
            abs_threshold=self._config.straggler_threshold,
        )
        self._current_solution: PartitionSolution | None = None
        self._current_num_layers: int = 0
        self._last_repartition_time: float = 0.0
        self._repartition_count: int = 0
        self._repartition_history: list[RepartitionEvent] = []
        self._straggler_cost_overrides: dict[str, float] = {}

    def set_initial_partition(
        self, solution: PartitionSolution, num_layers: int,
    ) -> None:
        """Set the initial partition solution for monitoring."""
        self._current_solution = solution
        self._current_num_layers = num_layers

        expected: dict[str, float] = {}
        for pt in solution.points:
            expected[pt.node_id] = pt.estimated_time_ms
        self._detector.set_expected(expected)

    def record_latency(
        self, node_id: str, latency_ms: float,
        batch_size: int = 1, seq_len: int = 4096,
    ) -> StragglerReport | None:
        """Record an observed latency and check for straggler."""
        sample = LatencySample(
            node_id=node_id,
            latency_ms=latency_ms,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        return self._detector.record(sample)

    def check_and_repartition(
        self, observed_latencies: dict[str, float],
    ) -> PartitionSolution | None:
        """Check observed latencies and trigger re-partition if needed.

        Args:
            observed_latencies: Per-node observed latency in ms.

        Returns:
            New PartitionSolution if re-partitioned, None otherwise.
        """
        if not self._config.enabled:
            return None

        if not self._can_repartition():
            return None

        straggler = self._find_straggler(observed_latencies)
        if straggler is None:
            return None

        logger.warning(
            f"Straggler detected: {straggler.node_id} "
            f"(observed={straggler.observed_latency_ms:.1f}ms, "
            f"expected={straggler.expected_latency_ms:.1f}ms, "
            f"ratio={straggler.ratio:.2f})"
        )

        return self._do_repartition(straggler)

    def force_repartition(
        self, reason: str = "manual",
    ) -> PartitionSolution | None:
        """Force a re-partition regardless of straggler status."""
        if not self._current_solution:
            return None

        report = StragglerReport(
            node_id="",
            observed_latency_ms=0,
            expected_latency_ms=0,
            ratio=0,
            severity=0,
            trigger=RepartitionTrigger.MANUAL,
        )
        return self._do_repartition(report)

    @property
    def current_solution(self) -> PartitionSolution | None:
        return self._current_solution

    @property
    def repartition_history(self) -> list[RepartitionEvent]:
        return list(self._repartition_history)

    @property
    def straggler_stats(self) -> dict[str, dict[str, float]]:
        return self._detector.get_all_stats()

    def _can_repartition(self) -> bool:
        now = time.time()
        if now - self._last_repartition_time < self._config.min_repartition_interval_s:
            return False

        hour_ago = now - 3600
        recent_count = sum(
            1 for e in self._repartition_history
            if e.timestamp > hour_ago
        )
        if recent_count >= self._config.max_repartitions_per_hour:
            return False

        return True

    def _find_straggler(
        self, observed: dict[str, float],
    ) -> StragglerReport | None:
        if not self._current_solution:
            return None

        expected: dict[str, float] = {}
        for pt in self._current_solution.points:
            expected[pt.node_id] = pt.estimated_time_ms

        reports: list[StragglerReport] = []
        for node_id, obs_ms in observed.items():
            exp_ms = expected.get(node_id, obs_ms)
            if exp_ms <= 0:
                continue
            ratio = obs_ms / exp_ms
            if ratio >= self._config.straggler_threshold:
                severity = min(
                    (ratio - 1.0) / (self._config.straggler_threshold - 1.0), 1.0
                )
                reports.append(StragglerReport(
                    node_id=node_id,
                    observed_latency_ms=obs_ms,
                    expected_latency_ms=exp_ms,
                    ratio=ratio,
                    severity=severity,
                    trigger=RepartitionTrigger.STRAGGLER,
                ))

        if not reports:
            return None

        if self._config.require_quorum:
            straggler_fraction = len(reports) / max(len(observed), 1)
            if straggler_fraction < self._config.quorum_fraction:
                logger.debug(
                    f"Straggler quorum not met: {len(reports)}/{len(observed)} "
                    f"< {self._config.quorum_fraction:.0%}"
                )
                return None

        return max(reports, key=lambda r: r.severity)

    def _do_repartition(
        self, report: StragglerReport,
    ) -> PartitionSolution:
        t0 = time.time()

        if report.node_id:
            self._straggler_cost_overrides[report.node_id] = report.ratio

        override_cost_model = self._create_override_cost_model()

        optimizer = PartitionOptimizer(
            cost_model=override_cost_model,
            node_ids=self._node_ids,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
            allow_oom=self._allow_oom,
        )

        new_solution = optimizer.solve(self._current_num_layers)

        old_solution = self._current_solution
        self._current_solution = new_solution
        self._last_repartition_time = time.time()

        expected: dict[str, float] = {}
        for pt in new_solution.points:
            expected[pt.node_id] = pt.estimated_time_ms
        self._detector.set_expected(expected)

        self._straggler_cost_overrides.pop(report.node_id, None)

        duration_ms = (time.time() - t0) * 1000
        event = RepartitionEvent(
            trigger=report.trigger,
            old_solution=old_solution,
            new_solution=new_solution,
            straggler_report=report,
            duration_ms=round(duration_ms, 2),
        )
        self._repartition_history.append(event)
        self._repartition_count += 1

        logger.info(
            f"Re-partitioned ({report.trigger.value}): "
            f"{new_solution.num_nodes} nodes, "
            f"max_latency={new_solution.max_node_time_ms:.1f}ms, "
            f"duration={duration_ms:.0f}ms"
        )

        if self._on_repartition:
            try:
                self._on_repartition(event)
            except Exception as e:
                logger.debug(f"Repartition callback failed: {e}")

        return new_solution

    def _create_override_cost_model(self) -> PartitionCostModel:
        """Create a cost model that inflates straggler node costs."""
        return _OverriddenCostModel(
            base=self._cost_model,
            overrides=dict(self._straggler_cost_overrides),
        )


class _OverriddenCostModel:
    """Wraps a PartitionCostModel with per-node cost multipliers."""

    def __init__(
        self,
        base: PartitionCostModel,
        overrides: dict[str, float],
    ):
        self._base = base
        self._overrides = overrides
        self._layer_weights = base._layer_weights
        self._topology = base._topology

    def evaluate(
        self, node_id: str, start_layer_id: int, end_layer_id: int,
        batch_size: int = 1, seq_len: int = 4096,
    ) -> Any:
        cost = self._base.evaluate(
            node_id, start_layer_id, end_layer_id, batch_size, seq_len,
        )
        multiplier = self._overrides.get(node_id, 1.0)
        if multiplier > 1.0:
            cost.total_time_ms *= multiplier
            cost.compute_time_ms *= multiplier
        return cost

    def evaluate_partition(
        self, partition: list[tuple[str, int, int]],
        batch_size: int = 1, seq_len: int = 4096,
    ) -> list[Any]:
        return [
            self.evaluate(nid, s, e, batch_size, seq_len)
            for nid, s, e in partition
        ]

    def pipeline_latency(
        self, partition: list[tuple[str, int, int]],
        batch_size: int = 1, seq_len: int = 4096,
        num_pipeline_stages: int | None = None,
    ) -> float:
        return self._base.pipeline_latency(
            partition, batch_size, seq_len, num_pipeline_stages,
        )

    def combined_throughput(
        self, partition: list[tuple[str, int, int]],
        batch_size: int = 1, seq_len: int = 4096,
    ) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0
        bottleneck = max(c.total_time_ms for c in costs)
        if bottleneck <= 0:
            return 0.0
        return (batch_size * seq_len) / (bottleneck / 1000.0)
