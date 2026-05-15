"""Failover decision engine for distributed-llm health service."""

from typing import Callable

from distllm.health.state import NodeState, HealthRecord


class FailoverEngine:
    """Manages state transitions and triggers callbacks.

    Transitions:
        HEALTHY -> DEGRADED: latency exceeds threshold
        DEGRADED -> UNHEALTHY: consecutive failures reach threshold
        UNHEALTHY -> OFFLINE: prolonged failure
        OFFLINE -> DEGRADED: recovery probe succeeds
        DEGRADED -> HEALTHY: consecutive successes reach threshold
        Any -> DRAINING: manual drain request
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        degraded_latency_ms: float = 2000.0,
        recovery_threshold: int = 2,
    ):
        self._failure_threshold = failure_threshold
        self._degraded_latency_ms = degraded_latency_ms
        self._recovery_threshold = recovery_threshold
        self._callbacks: list[Callable[[str, NodeState, NodeState], None]] = []

    def on_state_change(
        self, callback: Callable[[str, NodeState, NodeState], None]
    ) -> None:
        """Register a callback for state transitions."""
        self._callbacks.append(callback)

    def evaluate(self, record: HealthRecord, success: bool, latency_ms: float) -> NodeState:
        """Evaluate health record and return new state.

        Applies transition rules based on probe results.
        """
        old_state = record.state

        if success:
            record.consecutive_failures = 0
            record.consecutive_successes += 1
            record.record_latency(latency_ms)

            if old_state == NodeState.OFFLINE:
                record.state = NodeState.DEGRADED
            elif old_state == NodeState.DEGRADED and record.consecutive_successes >= self._recovery_threshold:
                record.state = NodeState.HEALTHY
            elif old_state in (NodeState.HEALTHY, NodeState.DEGRADED):
                if latency_ms > self._degraded_latency_ms:
                    record.state = NodeState.DEGRADED
                else:
                    record.state = NodeState.HEALTHY
        else:
            record.consecutive_successes = 0
            record.consecutive_failures += 1

            if record.consecutive_failures >= self._failure_threshold:
                record.state = NodeState.UNHEALTHY
            else:
                record.state = NodeState.DEGRADED

        if old_state != record.state:
            for cb in self._callbacks:
                cb(record.node_id, old_state, record.state)

        return record.state
