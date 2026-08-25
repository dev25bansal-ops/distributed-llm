"""Detect and mitigate slow nodes in the pipeline.

Monitors per-node latency, detects stragglers using statistical analysis,
and triggers mitigation actions. Standalone module that integrates with
the existing Rebalancer for layer reassignment.

Detection methods:
- Threshold-based: node exceeds p95 latency of others
- MAD-based: median absolute deviation outlier detection (calibrated,
  see ``StragglerDetector.__init__`` for threshold rationale)
- Trend-based: sustained latency increase over time
- Throughput-based: tokens/second below expected throughput
- Ensemble: weighted voting across all methods

Advanced features:
- Rolling baseline via exponential moving average (EMA)
- Stale node detection via last_seen timeout
- Predictive detection via Holt-Winters forecasting
- Root cause attribution (GPU temp, memory, network)
- Adaptive thresholds via Welford's online algorithm
- Straggler event history for analytics
- Configurable multipliers for all thresholds
- Callback throttling per node
"""

from __future__ import annotations

import math
import statistics
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class DetectionMethod(Enum):
    THRESHOLD = "threshold"
    MAD = "mad"
    TREND = "trend"
    THROUGHPUT = "throughput"
    ENSEMBLE = "ensemble"


class StragglerSeverity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class RootCauseAttribution:
    """System metrics correlated with a straggler event."""
    node_id: str = ""
    gpu_temp_c: float = 0.0
    gpu_memory_used_pct: float = 0.0
    network_bandwidth_mbps: float = 0.0
    cpu_utilization_pct: float = 0.0
    io_wait_pct: float = 0.0
    probable_cause: str = "unknown"  # "thermal", "memory", "network", "cpu", "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "gpu_temp_c": self.gpu_temp_c,
            "gpu_memory_used_pct": self.gpu_memory_used_pct,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "cpu_utilization_pct": self.cpu_utilization_pct,
            "io_wait_pct": self.io_wait_pct,
            "probable_cause": self.probable_cause,
        }


@dataclass
class StragglerEvent:
    """A historical straggler detection event."""
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    severity: StragglerSeverity = StragglerSeverity.NONE
    latency_ms: float = 0.0
    baseline_ms: float = 0.0
    action_taken: str = ""
    detection_method: str = ""
    root_cause: RootCauseAttribution | None = None
    resolved: bool = False
    resolution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "severity": self.severity.value,
            "latency_ms": self.latency_ms,
            "baseline_ms": self.baseline_ms,
            "action_taken": self.action_taken,
            "detection_method": self.detection_method,
            "root_cause": self.root_cause.to_dict() if self.root_cause else None,
            "resolved": self.resolved,
            "resolution_time_s": self.resolution_time_s,
        }


class AdaptiveThreshold:
    """Learns normal latency distribution using Welford's online algorithm.

    Automatically sets outlier thresholds based on observed data.
    """

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self._mean = 0.0
        self._m2 = 0.0  # Welford's variance accumulator
        self._count = 0

    def update(self, value: float) -> None:
        """Update with a new observation."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._count - 1))

    def is_outlier(self, value: float) -> bool:
        """Check if value is an outlier (> sensitivity * std from mean).

        With a zero-variance baseline (e.g. 20 identical samples) any
        deviation beyond a small relative floor is still an outlier —
        ``std == 0`` must not silently mask spikes.
        """
        if self._count < 10:
            return False  # Not enough data
        std = self.std
        if std <= 0:
            # Zero variance: flag deviations beyond a quarter of the mean
            # (scaled by sensitivity). A change of this magnitude cannot be
            # explained by noise on an otherwise stable distribution.
            floor = self.sensitivity * max(abs(self._mean) * 0.25, 1e-9)
            return abs(value - self._mean) > floor
        return abs(value - self._mean) > self.sensitivity * std

    def percentile_rank(self, value: float) -> float:
        """Return approximate percentile rank of value (0-100)."""
        if self._count < 2 or self.std <= 0:
            return 50.0
        z = (value - self._mean) / self.std
        # Approximate CDF using logistic function
        return 100.0 / (1.0 + math.exp(-1.7 * z))


@dataclass
class NodeTiming:
    node_id: str
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    throughputs: deque = field(default_factory=lambda: deque(maxlen=100))
    last_seen: float = 0.0
    consecutive_slow: int = 0
    is_straggler: bool = False
    severity: StragglerSeverity = StragglerSeverity.NONE
    baseline_latency: float = 0.0
    baseline_throughput: float = 0.0
    baseline_alpha: float = 0.1  # EMA smoothing factor
    # Root cause data
    last_root_cause: RootCauseAttribution | None = None
    # Adaptive threshold per node
    adaptive: AdaptiveThreshold = field(default_factory=AdaptiveThreshold)
    # Predictive model state
    _hw_level: float = 0.0
    _hw_trend: float = 0.0
    _hw_seasonal: list[float] = field(default_factory=list)
    _hw_initialized: bool = False

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def avg_throughput(self) -> float:
        return sum(self.throughputs) / max(len(self.throughputs), 1)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 2:
            return self.avg_latency
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    def update_baseline(self, value: float) -> None:
        """Update baseline using exponential moving average."""
        if self.baseline_latency <= 0:
            self.baseline_latency = value
        else:
            self.baseline_latency = (
                (1 - self.baseline_alpha) * self.baseline_latency
                + self.baseline_alpha * value
            )

    def update_throughput_baseline(self, value: float) -> None:
        """Update throughput baseline using EMA."""
        if self.baseline_throughput <= 0:
            self.baseline_throughput = value
        else:
            self.baseline_throughput = (
                (1 - self.baseline_alpha) * self.baseline_throughput
                + self.baseline_alpha * value
            )

    def predict_latency(self, horizon: int = 5) -> float | None:
        """Predict future latency using Holt-Winters triple exponential smoothing.

        Returns None if insufficient data (< 24 samples).
        """
        if len(self.latencies) < 24:
            return None

        prices = list(self.latencies)
        n = len(prices)
        season_len = min(12, n // 2)

        if not self._hw_initialized:
            # Initialize level, trend, seasonal
            self._hw_level = sum(prices[:season_len]) / season_len
            if n >= 2 * season_len:
                self._hw_trend = (
                    (sum(prices[season_len:2 * season_len]) - sum(prices[:season_len]))
                    / (season_len * season_len)
                )
            else:
                self._hw_trend = 0
            self._hw_seasonal = [1.0] * season_len
            if self._hw_level > 0:
                for i in range(min(season_len, n)):
                    self._hw_seasonal[i] = prices[i] / self._hw_level
            self._hw_initialized = True

        alpha = 0.3
        beta = 0.1
        gamma = 0.2

        for t in range(season_len, n):
            s_idx = t % season_len
            new_level = alpha * (prices[t] / max(self._hw_seasonal[s_idx], 0.01)) + (1 - alpha) * (self._hw_level + self._hw_trend)
            new_trend = beta * (new_level - self._hw_level) + (1 - beta) * self._hw_trend
            self._hw_seasonal[s_idx] = gamma * (prices[t] / max(new_level, 0.01)) + (1 - gamma) * self._hw_seasonal[s_idx]
            self._hw_level = new_level
            self._hw_trend = new_trend

        forecast = (self._hw_level + self._hw_trend * horizon) * self._hw_seasonal[(n + horizon) % season_len]
        return max(0, forecast)


@dataclass
class StragglerReport:
    node_id: str
    severity: StragglerSeverity
    avg_latency: float
    p95_latency: float
    baseline_latency: float
    slowdown_factor: float
    detection_method: DetectionMethod
    consecutive_detections: int
    recommended_action: str
    root_cause: RootCauseAttribution | None = None
    predicted_latency: float | None = None
    adaptive_percentile: float = 0.0


class StragglerDetector:
    """Detects and mitigates slow nodes in the pipeline.

    Features:
    - 5 detection methods: Threshold, MAD, Trend, Throughput, Ensemble
    - Rolling baseline via EMA (never stale)
    - Stale node detection (nodes that stop reporting)
    - Configurable multipliers for all thresholds
    - Callback throttling per node
    - Predictive detection via Holt-Winters
    - Root cause attribution
    - Adaptive thresholds via Welford's algorithm
    - Straggler event history

    MAD threshold calibration (``mad_threshold``, default 2.0)
    ----------------------------------------------------------
    The MAD method computes a robust z-score over peer p95 latencies::

        z_i = |p95_i - median(p95)| / (MAD / 0.6745)

    where MAD is the raw median absolute deviation across peers.  Dividing
    by the consistency constant 0.6745 (Iglewicz-Hoban) scales the score to
    be comparable to a standard deviation under normality; without it,
    ``mad_threshold=2.0`` would correspond to only ~1.35 sigma and fire on
    routine jitter.

    A second guard, ``min_relative_deviation`` (default 0.25), requires the
    absolute deviation from the peer median to also exceed 25% of that
    median.  This exists because with only a handful of peers whose p95s
    cluster within a few percent of each other, the raw MAD collapses toward
    zero and pure sampling noise systematically exceeds any fixed multiple
    of it — measured at ~40% false-positive check rounds on ±20% uniform
    jitter before this guard was added.  A genuine straggler running at 3x
    its peers deviates by +200% and is unaffected.

    Chosen operating point (verified against synthetic latency sequences in
    ``tests/dist/test_straggler.py::TestMADCharacterization``):

    - ±20% normal jitter across all nodes: no detections (was ~40% of checks)
    - sudden spike to 3x peers: detected within 3 consecutive checks
    - slow drift: detected once latency exceeds peers by ~25-50%

    Usage::

        detector = StragglerDetector(
            on_straggler_cb=lambda report: print(report),
            detection_method=DetectionMethod.MAD,
        )

        # During inference:
        detector.record_latency("node_1", 45.0)
        detector.record_throughput("node_1", 120.0)

        if detector.check():
            for report in detector.get_reports():
                print(report)
    """

    def __init__(
        self,
        on_straggler_cb: Callable[[StragglerReport], None] | None = None,
        detection_method: DetectionMethod = DetectionMethod.MAD,
        slow_threshold_ms: float = 100.0,
        consecutive_threshold: int = 3,
        window_size: int = 50,
        mad_threshold: float = 2.0,
        min_relative_deviation: float = 0.25,
        check_interval_s: float = 10.0,
        # Configurable multipliers (fixes #7, #8, #9)
        threshold_multiplier: float = 1.5,
        trend_multiplier: float = 1.5,
        throughput_floor: float = 0.5,
        severe_multiplier: float = 3.0,
        moderate_multiplier: float = 2.0,
        mild_multiplier: float = 1.5,
        # Stale node detection (fixes #4)
        stale_timeout_s: float = 60.0,
        # Callback throttling (fixes #13)
        callback_cooldown_s: float = 60.0,
        # Baseline EMA alpha (fixes #1)
        baseline_alpha: float = 0.1,
        # Network RTT filtering (B-19)
        network_rtt_threshold: float = 0.0,
        network_rtt_fn: Callable[[str], float] | None = None,
        # Root cause attribution
        gpu_health_fn: Callable[[str], RootCauseAttribution] | None = None,
    ):
        self._on_straggler = on_straggler_cb
        self._detection_method = detection_method
        self._slow_threshold = slow_threshold_ms
        self._consecutive_threshold = consecutive_threshold
        self._window_size = window_size
        self._mad_threshold = mad_threshold
        self._min_relative_deviation = min_relative_deviation
        self._check_interval = check_interval_s

        # Configurable multipliers
        self._threshold_multiplier = threshold_multiplier
        self._trend_multiplier = trend_multiplier
        self._throughput_floor = throughput_floor
        self._severe_multiplier = severe_multiplier
        self._moderate_multiplier = moderate_multiplier
        self._mild_multiplier = mild_multiplier

        # Stale node detection
        self._stale_timeout_s = stale_timeout_s

        # Callback throttling
        self._callback_cooldown_s = callback_cooldown_s
        self._callback_cooldowns: dict[str, float] = {}

        # Baseline alpha
        self._baseline_alpha = baseline_alpha

        # Network RTT filtering (B-19)
        self._network_rtt_threshold = network_rtt_threshold
        self._network_rtt_fn = network_rtt_fn

        # Root cause
        self._gpu_health_fn = gpu_health_fn

        self._nodes: dict[str, NodeTiming] = {}
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._total_checks = 0
        self._total_detections = 0

        # Straggler event history
        self._events: list[StragglerEvent] = []
        self._max_events = 1000

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        with self._lock:
            if node_id not in self._nodes:
                node = NodeTiming(node_id=node_id)
                node.baseline_alpha = self._baseline_alpha
                self._nodes[node_id] = node
            node = self._nodes[node_id]
            node.latencies.append(latency_ms)
            node.last_seen = time.time()

            # Rolling EMA baseline (fixes #1: baseline never updates)
            node.update_baseline(latency_ms)

            # Update adaptive threshold
            node.adaptive.update(latency_ms)

    def record_throughput(self, node_id: str, tokens_per_second: float) -> None:
        with self._lock:
            if node_id not in self._nodes:
                node = NodeTiming(node_id=node_id)
                node.baseline_alpha = self._baseline_alpha
                self._nodes[node_id] = node
            node = self._nodes[node_id]
            node.throughputs.append(tokens_per_second)
            node.update_throughput_baseline(tokens_per_second)

    def record_batch(
        self,
        node_id: str,
        latency_ms: float,
        tokens_generated: int = 0,
        batch_size: int = 0,
    ) -> None:
        self.record_latency(node_id, latency_ms)
        if tokens_generated > 0 and latency_ms > 0:
            self.record_throughput(node_id, tokens_generated / (latency_ms / 1000.0))

    def record_root_cause(self, node_id: str, cause: RootCauseAttribution) -> None:
        """Record root cause attribution for a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_root_cause = cause

    def check(self) -> list[StragglerReport]:
        now = time.time()
        if now - self._last_check < self._check_interval:
            return []

        self._last_check = now
        self._total_checks += 1
        reports: list[StragglerReport] = []

        with self._lock:
            # Check for stale nodes (fixes #4) — independent of peer count:
            # a single silent node is still a failure.
            stale_reports = self._check_stale_nodes(now)

            if len(self._nodes) < 2:
                return stale_reports

            eligible_nodes = {
                nid: n for nid, n in self._nodes.items()
                if len(n.latencies) >= 5
                or (self._detection_method == DetectionMethod.THROUGHPUT
                    and len(n.throughputs) >= 5)
            }
            if len(eligible_nodes) < 2:
                return stale_reports

            node_p95s = {nid: n.p95_latency for nid, n in eligible_nodes.items()}
            node_avgs = {nid: n.avg_latency for nid, n in eligible_nodes.items()}
            all_p95_vals = list(node_p95s.values())
            all_avg_vals = list(node_avgs.values())
            median_p95 = statistics.median(all_p95_vals)
            median_avg = statistics.median(all_avg_vals)

            for node_id, node in eligible_nodes.items():
                is_slow = False
                method = self._detection_method
                severity = StragglerSeverity.NONE

                if method == DetectionMethod.ENSEMBLE:
                    is_slow = self._check_ensemble(node, median_p95, all_p95_vals)
                elif method == DetectionMethod.THRESHOLD:
                    is_slow = node.p95_latency > median_p95 * self._threshold_multiplier
                elif method == DetectionMethod.MAD:
                    devs = [abs(p - median_p95) for p in all_p95_vals]
                    mad = statistics.median(devs) if devs else 0
                    # Calibrated robust z-score: scale raw MAD by its
                    # consistency constant (0.6745) so mad_threshold is
                    # comparable to a sigma-multiple under normality.
                    scaled_mad = max(mad / 0.6745, 1e-9)
                    deviation = abs(node.p95_latency - median_p95)
                    relative = deviation / median_p95 if median_p95 > 0 else float("inf")
                    if (
                        deviation / scaled_mad > self._mad_threshold
                        and relative > self._min_relative_deviation
                    ):
                        is_slow = True
                    elif node.p95_latency > median_p95 * self._threshold_multiplier:
                        # Fixes #2: MAD≈0 fallback.  With two nodes both
                        # deviate from the median equally, so MAD alone cannot
                        # discriminate; fall back to the threshold rule so the
                        # slower node is still caught.
                        is_slow = True
                elif method == DetectionMethod.TREND:
                    # Windowed trend: compare the recent half of the window
                    # against the earlier half.  Using the rolling EMA baseline
                    # here is self-defeating — a sustained slowdown drags its
                    # own baseline up and eventually masks the very signal it
                    # is meant to capture.
                    lats = list(node.latencies)
                    n = len(lats)
                    if n >= 4:
                        half = n // 2
                        older_avg = sum(lats[:half]) / half
                        recent_avg = sum(lats[half:]) / (n - half)
                        is_slow = recent_avg > older_avg * self._trend_multiplier
                    elif node.baseline_latency > 0:
                        is_slow = node.avg_latency > node.baseline_latency * self._trend_multiplier
                elif method == DetectionMethod.THROUGHPUT:
                    # Same windowed reasoning as TREND, inverted (lower is worse).
                    tps = list(node.throughputs)
                    n = len(tps)
                    if n >= 4:
                        half = n // 2
                        older_avg = sum(tps[:half]) / half
                        recent_avg = sum(tps[half:]) / (n - half)
                        is_slow = recent_avg < older_avg * self._throughput_floor
                    elif node.baseline_throughput > 0:
                        is_slow = node.avg_throughput < node.baseline_throughput * self._throughput_floor

                if is_slow:
                    # Compute slowdown relative to this node's own baseline
                    # (which captures sustained degradation) and relative to
                    # the peer median excluding this node (which captures
                    # disparity without the node skewing its own reference);
                    # severity follows the stronger signal.  For detection,
                    # the median still includes this node (peer-relative
                    # methods like THRESHOLD/MAD are defined over all nodes).
                    base = node.baseline_latency
                    if base > 0:
                        slowdown = node.p95_latency / base
                    else:
                        slowdown = float("nan")
                    peer_vals = [p for p in all_p95_vals if p != node.p95_latency]
                    if peer_vals:
                        peers_median = statistics.median(peer_vals)
                        if peers_median > 0:
                            peer_slowdown = node.p95_latency / peers_median
                            slowdown = max(slowdown, peer_slowdown)
                    if slowdown > self._severe_multiplier:
                        severity = StragglerSeverity.SEVERE
                    elif slowdown > self._moderate_multiplier:
                        severity = StragglerSeverity.MODERATE
                    elif slowdown > self._mild_multiplier:
                        severity = StragglerSeverity.MILD
                    # Non-baseline methods (THRESHOLD/MAD) have no meaningful
                    # ratio; the method signal itself is the severity evidence.
                    if severity == StragglerSeverity.NONE:
                        severity = StragglerSeverity.MILD

                    node.consecutive_slow += 1
                else:
                    node.consecutive_slow = 0

                node.is_straggler = node.consecutive_slow >= self._consecutive_threshold
                node.is_straggler = node.is_straggler and self._rtt_allows(node_id)
                node.severity = severity if node.is_straggler else StragglerSeverity.NONE

                if node.is_straggler:
                    action = self._recommend_action(severity)

                    # Root cause attribution
                    root_cause = node.last_root_cause
                    if self._gpu_health_fn and root_cause is None:
                        try:
                            root_cause = self._gpu_health_fn(node_id)
                        except Exception:
                            pass

                    # Predictive latency
                    predicted = node.predict_latency(horizon=5)

                    # Adaptive percentile
                    percentile = node.adaptive.percentile_rank(node.p95_latency)

                    report = StragglerReport(
                        node_id=node_id,
                        severity=severity,
                        avg_latency=round(node.avg_latency, 2),
                        p95_latency=round(node.p95_latency, 2),
                        baseline_latency=round(node.baseline_latency, 2),
                        slowdown_factor=round(node.avg_latency / max(node.baseline_latency, 1), 2),
                        detection_method=method,
                        consecutive_detections=node.consecutive_slow,
                        recommended_action=action,
                        root_cause=root_cause,
                        predicted_latency=round(predicted, 2) if predicted else None,
                        adaptive_percentile=round(percentile, 1),
                    )
                    reports.append(report)
                    self._total_detections += 1

        # Fire callbacks with throttling (fixes #13)
        for report in reports:
            logger.warning(
                f"Straggler detected: {report.node_id} "
                f"({report.severity.value}, {report.slowdown_factor}x slower, "
                f"action: {report.recommended_action})"
            )
            if self._on_straggler and self._should_fire_callback(report.node_id, now):
                try:
                    self._on_straggler(report)
                except Exception as e:
                    logger.error(f"Straggler callback failed: {e}")

            # Record event history
            self._record_event(report)

        return reports

    def _check_stale_nodes(self, now: float) -> list[StragglerReport]:
        """Check for nodes that stopped reporting (fixes #4)."""
        reports = []
        for node_id, node in self._nodes.items():
            # A node that never reported has nothing to go stale.
            if node.last_seen <= 0 and not node.latencies:
                continue
            if now - node.last_seen > self._stale_timeout_s and not node.is_straggler:
                node.is_straggler = True
                node.severity = StragglerSeverity.SEVERE
                node.consecutive_slow = self._consecutive_threshold
                reports.append(StragglerReport(
                    node_id=node_id,
                    severity=StragglerSeverity.SEVERE,
                    avg_latency=0,
                    p95_latency=0,
                    baseline_latency=round(node.baseline_latency, 2),
                    slowdown_factor=float("inf"),
                    detection_method=self._detection_method,
                    consecutive_detections=0,
                    recommended_action="reassign_layers",
                ))
        return reports

    def _check_ensemble(
        self,
        node: NodeTiming,
        median_p95: float,
        all_p95_vals: list[float],
    ) -> bool:
        """Ensemble detection: flag if ≥2 methods agree."""
        votes = 0
        # Threshold
        if node.p95_latency > median_p95 * self._threshold_multiplier:
            votes += 1
        # MAD (calibrated robust z-score + relative-deviation guard; see
        # class docstring for threshold rationale)
        devs = [abs(p - median_p95) for p in all_p95_vals]
        mad = statistics.median(devs) if devs else 0
        scaled_mad = max(mad / 0.6745, 1e-9)
        deviation = abs(node.p95_latency - median_p95)
        relative = deviation / median_p95 if median_p95 > 0 else float("inf")
        if (
            deviation / scaled_mad > self._mad_threshold
            and relative > self._min_relative_deviation
        ):
            votes += 1
        elif node.p95_latency > median_p95 * self._threshold_multiplier:
            votes += 1
        # Trend
        if node.baseline_latency > 0 and node.avg_latency > node.baseline_latency * self._trend_multiplier:
            votes += 1
        # Throughput
        if node.baseline_throughput > 0 and node.avg_throughput < node.baseline_throughput * self._throughput_floor:
            votes += 1
        return votes >= 2

    def _should_fire_callback(self, node_id: str, now: float) -> bool:
        """Throttle callbacks per node (fixes #13)."""
        last = self._callback_cooldowns.get(node_id, 0)
        if now - last >= self._callback_cooldown_s:
            self._callback_cooldowns[node_id] = now
            return True
        return False

    def _rtt_allows(self, node_id: str) -> bool:
        """Apply the network RTT filter (B-19).

        A node whose RTT exceeds the threshold is experiencing a network
        problem rather than a compute problem — suppress straggler detection
        for it.  When no RTT callback is configured every node passes.
        """
        if self._network_rtt_threshold <= 0 or self._network_rtt_fn is None:
            return True
        try:
            return self._network_rtt_fn(node_id) < self._network_rtt_threshold
        except Exception:
            return True

    def _recommend_action(self, severity: StragglerSeverity) -> str:
        if severity == StragglerSeverity.SEVERE:
            return "reassign_layers"
        elif severity == StragglerSeverity.MODERATE:
            return "reduce_batch"
        return "monitor_only"

    def _record_event(self, report: StragglerReport) -> None:
        """Record a straggler event to history."""
        event = StragglerEvent(
            node_id=report.node_id,
            severity=report.severity,
            latency_ms=report.p95_latency,
            baseline_ms=report.baseline_latency,
            action_taken=report.recommended_action,
            detection_method=report.detection_method.value,
            root_cause=report.root_cause,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def get_reports(self) -> list[StragglerReport]:
        """Get current straggler reports (fixes #3: includes recommended_action)."""
        self.check()
        reports = []
        with self._lock:
            for node in self._nodes.values():
                if node.is_straggler:
                    predicted = node.predict_latency(horizon=5)
                    percentile = node.adaptive.percentile_rank(node.p95_latency)
                    reports.append(StragglerReport(
                        node_id=node.node_id,
                        severity=node.severity,
                        avg_latency=round(node.avg_latency, 2),
                        p95_latency=round(node.p95_latency, 2),
                        baseline_latency=round(node.baseline_latency, 2),
                        slowdown_factor=round(node.avg_latency / max(node.baseline_latency, 1), 2),
                        detection_method=self._detection_method,
                        consecutive_detections=node.consecutive_slow,
                        recommended_action=self._recommend_action(node.severity),  # Fixed!
                        root_cause=node.last_root_cause,
                        predicted_latency=round(predicted, 2) if predicted else None,
                        adaptive_percentile=round(percentile, 1),
                    ))
        return reports

    def predict_stragglers(self, horizon: int = 5) -> list[dict[str, Any]]:
        """Predict which nodes will become stragglers soon.

        Uses Holt-Winters forecasting on latency time series.
        Returns list of nodes with predicted latency above threshold.
        """
        predictions = []
        with self._lock:
            for node_id, node in self._nodes.items():
                predicted = node.predict_latency(horizon)
                if predicted is None:
                    continue
                median_baseline = node.baseline_latency or node.avg_latency
                if median_baseline > 0 and predicted > median_baseline * self._threshold_multiplier:
                    predictions.append({
                        "node_id": node_id,
                        "predicted_latency": round(predicted, 2),
                        "current_latency": round(node.avg_latency, 2),
                        "baseline_latency": round(node.baseline_latency, 2),
                        "predicted_slowdown": round(predicted / max(median_baseline, 1), 2),
                    })
        return sorted(predictions, key=lambda p: p["predicted_slowdown"], reverse=True)

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get straggler event history."""
        with self._lock:
            return [e.to_dict() for e in self._events[-limit:]]

    def get_analytics(self) -> dict[str, Any]:
        """Get straggler analytics summary."""
        with self._lock:
            if not self._events:
                return {"total_events": 0}
            by_severity = {}
            by_method = {}
            by_node = {}
            resolution_times = []
            for e in self._events:
                by_severity[e.severity.value] = by_severity.get(e.severity.value, 0) + 1
                by_method[e.detection_method] = by_method.get(e.detection_method, 0) + 1
                by_node[e.node_id] = by_node.get(e.node_id, 0) + 1
                if e.resolved and e.resolution_time_s > 0:
                    resolution_times.append(e.resolution_time_s)
            return {
                "total_events": len(self._events),
                "by_severity": by_severity,
                "by_method": by_method,
                "by_node": by_node,
                "avg_resolution_time_s": (
                    round(sum(resolution_times) / len(resolution_times), 1)
                    if resolution_times else 0
                ),
            }

    def clear_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def reset_baseline(self, node_id: str) -> None:
        """Reset baseline for a node (call after restart/recovery)."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.baseline_latency = 0.0
                node.baseline_throughput = 0.0
                node.consecutive_slow = 0
                node.is_straggler = False
                node.severity = StragglerSeverity.NONE
                logger.info(f"Straggler baseline reset for {node_id}")

    def reset_all(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._total_checks = 0
            self._total_detections = 0
            self._callback_cooldowns.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = len(self._nodes)
            stragglers = sum(1 for n in self._nodes.values() if n.is_straggler)
            return {
                "active_nodes": active,
                "straggler_nodes": stragglers,
                "detection_method": self._detection_method.value,
                "total_checks": self._total_checks,
                "total_detections": self._total_detections,
                "total_events": len(self._events),
                "nodes": {
                    nid: {
                        "avg_latency": round(node.avg_latency, 2),
                        "p95_latency": round(node.p95_latency, 2),
                        "avg_throughput": round(node.avg_throughput, 1),
                        "baseline_latency": round(node.baseline_latency, 2),
                        "slowdown": round(node.avg_latency / max(node.baseline_latency, 1), 2),
                        "consecutive_slow": node.consecutive_slow,
                        "is_straggler": node.is_straggler,
                        "severity": node.severity.value,
                        "adaptive_percentile": round(node.adaptive.percentile_rank(node.p95_latency), 1),
                    }
                    for nid, node in self._nodes.items()
                },
            }
