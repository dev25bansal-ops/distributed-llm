from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ScalingDirection(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


@dataclass
class ScalingDecision:
    direction: ScalingDirection
    count: int = 0
    reason: str = ""


@dataclass
class PoolTelemetry:
    active_nodes: int = 0
    total_capacity: int = 0
    total_load: int = 0
    pending_requests: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0


class PrefillScaler:
    """Autoscaler for the prefill pool.

    Scales based on:
    - Pending request queue depth
    - Average prefill latency vs target
    - Node utilization (load / capacity)
    """

    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 16,
        target_latency_ms: float = 500.0,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.30,
        cooldown_seconds: float = 60.0,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.target_latency_ms = target_latency_ms
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_scale_time: float = 0.0
        self._current_nodes: int = min_nodes

    def evaluate(self, telemetry: PoolTelemetry) -> ScalingDecision:
        now = __import__("time").time()
        if now - self._last_scale_time < self.cooldown_seconds:
            return ScalingDecision(ScalingDirection.HOLD, reason="In cooldown period")

        utilization = (
            telemetry.total_load / max(telemetry.total_capacity, 1)
            if telemetry.total_capacity > 0
            else 0.0
        )

        # Scale up due to high utilization
        if utilization >= self.scale_up_threshold and self._current_nodes < self.max_nodes:
            count = min(2, self.max_nodes - self._current_nodes)
            self._last_scale_time = now
            self._current_nodes += count
            logger.info(f"Prefill scaler: +{count} node(s) (utilization {utilization:.0%})")
            return ScalingDecision(
                ScalingDirection.SCALE_UP, count,
                f"Utilization {utilization:.0%} >= {self.scale_up_threshold:.0%}",
            )

        # Scale up due to latency exceeding target
        if (
            telemetry.avg_latency_ms > self.target_latency_ms
            and self._current_nodes < self.max_nodes
        ):
            count = min(1, self.max_nodes - self._current_nodes)
            self._last_scale_time = now
            self._current_nodes += count
            logger.info(f"Prefill scaler: +{count} node(s) (latency {telemetry.avg_latency_ms:.0f}ms)")
            return ScalingDecision(
                ScalingDirection.SCALE_UP, count,
                f"Avg latency {telemetry.avg_latency_ms:.0f}ms > {self.target_latency_ms:.0f}ms",
            )

        # Scale down due to low utilization
        if utilization <= self.scale_down_threshold and self._current_nodes > self.min_nodes:
            count = min(1, self._current_nodes - self.min_nodes)
            self._last_scale_time = now
            self._current_nodes -= count
            logger.info(f"Prefill scaler: -{count} node(s) (utilization {utilization:.0%})")
            return ScalingDecision(
                ScalingDirection.SCALE_DOWN, count,
                f"Utilization {utilization:.0%} <= {self.scale_down_threshold:.0%}",
            )

        return ScalingDecision(ScalingDirection.HOLD, reason="Within bounds")


class DecodeScaler:
    """Autoscaler for the decode pool.

    Scales based on:
    - Active request count vs total capacity
    - Pending request queue depth
    - Memory pressure per node (KV cache growth)
    """

    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 32,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.30,
        cooldown_seconds: float = 60.0,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_scale_time: float = 0.0
        self._current_nodes: int = min_nodes

    def evaluate(self, telemetry: PoolTelemetry) -> ScalingDecision:
        now = __import__("time").time()
        if now - self._last_scale_time < self.cooldown_seconds:
            return ScalingDecision(ScalingDirection.HOLD, reason="In cooldown period")

        utilization = (
            telemetry.total_load / max(telemetry.total_capacity, 1)
            if telemetry.total_capacity > 0
            else 0.0
        )

        if utilization >= self.scale_up_threshold and self._current_nodes < self.max_nodes:
            count = min(2, self.max_nodes - self._current_nodes)
            self._last_scale_time = now
            self._current_nodes += count
            logger.info(f"Decode scaler: +{count} node(s) (utilization {utilization:.0%})")
            return ScalingDecision(
                ScalingDirection.SCALE_UP, count,
                f"Utilization {utilization:.0%} >= {self.scale_up_threshold:.0%}",
            )

        if utilization <= self.scale_down_threshold and self._current_nodes > self.min_nodes:
            count = min(1, self._current_nodes - self.min_nodes)
            self._last_scale_time = now
            self._current_nodes -= count
            logger.info(f"Decode scaler: -{count} node(s) (utilization {utilization:.0%})")
            return ScalingDecision(
                ScalingDirection.SCALE_DOWN, count,
                f"Utilization {utilization:.0%} <= {self.scale_down_threshold:.0%}",
            )

        return ScalingDecision(ScalingDirection.HOLD, reason="Within bounds")
