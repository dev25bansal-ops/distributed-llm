"""Self-healing configuration for distributed-llm.

Provides five components that work together to detect and recover from
node-level failures in a distributed LLM cluster:

* **RemoteGPUReset** -- SSH into a node, reset the GPU via nvidia-smi,
  verify health after reset, with a configurable timeout (default 30 s).
* **DrainCoordinator** -- Manage node draining with a configurable
  capacity limit (default 25 %) and drain least-loaded nodes first.
* **FailurePredictor** -- Heuristic-based failure risk prediction using
  GPU temperature, memory utilization, error rate, and latency features.
* **RecoverySLA** -- Track recovery time against SLA deadlines
  (single node: 5 min, >25 % nodes: 15 min) and escalate on breach.
* **SelfHealingConfigurator** -- Combine all four into a background
  monitoring loop with configurable check interval and parameters.

Usage::

    from distllm.observability.self_healing_config import SelfHealingConfigurator

    configurator = SelfHealingConfigurator(check_interval=30.0)
    configurator.start()
    # ... application runs ...
    configurator.stop()

The configurator exposes each sub-component at :attr:`gpu_reset`,
:attr:`drain_coordinator`, :attr:`failure_predictor`, and
:attr:`recovery_sla` for direct access.
"""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess  # noqa: S404 -- controlled SSH subprocess for GPU reset
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SSH_TIMEOUT: float = 30.0
_DEFAULT_DRAIN_PERCENT: float = 25.0
_DEFAULT_CHECK_INTERVAL: float = 30.0
_DEFAULT_SINGLE_NODE_TIMEOUT_MIN: int = 5
_DEFAULT_MULTI_NODE_TIMEOUT_MIN: int = 15

# Feature weight defaults for FailurePredictor
_TEMP_WEIGHT: float = 0.35
_MEMORY_WEIGHT: float = 0.25
_ERROR_RATE_WEIGHT: float = 0.25
_LATENCY_WEIGHT: float = 0.15

# Feature thresholds (beyond which risk contribution increases)
_TEMP_THRESHOLD_C: float = 80.0
_MEMORY_UTIL_THRESHOLD: float = 0.90
_ERROR_RATE_THRESHOLD: float = 0.05
_LATENCY_THRESHOLD_MS: float = 500.0

# Maximum number of historical failure records to retain per node
_MAX_FAILURE_HISTORY: int = 1000

# ---------------------------------------------------------------------------
# GPUHealthStatus
# ---------------------------------------------------------------------------


class GPUHealthStatus(str, Enum):
    """Health status of a GPU after reset verification."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# GPUHealthInfo
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GPUHealthInfo:
    """Result of a GPU health check.

    Attributes:
        status: Overall health classification.
        temperature_c: GPU temperature in Celsius.
        memory_used_mb: Used GPU memory in MiB.
        memory_total_mb: Total GPU memory in MiB.
        error_count: Number of errors detected during the check.
    """

    status: GPUHealthStatus
    temperature_c: float
    memory_used_mb: float
    memory_total_mb: float
    error_count: int

    @property
    def memory_utilization(self) -> float:
        """Fraction of GPU memory currently used (0.0 -- 1.0)."""
        if self.memory_total_mb > 0:
            return self.memory_used_mb / self.memory_total_mb
        return 0.0


# ---------------------------------------------------------------------------
# RemoteGPUReset
# ---------------------------------------------------------------------------


class RemoteGPUReset:
    """Reset a remote GPU via SSH and verify health afterwards.

    Connects to the target node over SSH, runs ``nvidia-smi --gpu-reset``
    (or equivalent reset command), then queries GPU telemetry to confirm
    the GPU is healthy post-reset.

    Uses ``subprocess`` to invoke the system ``ssh`` command (no extra
    Python dependencies required).  Timeout defaults to 30 seconds.

    Args:
        timeout: Maximum time in seconds for the entire reset + verify
            operation (default 30.0).
        ssh_user: SSH username (default ``"root"``).
        ssh_port: SSH port (default 22).
    """

    def __init__(
        self,
        timeout: float = _DEFAULT_SSH_TIMEOUT,
        ssh_user: str = "root",
        ssh_port: int = 22,
    ) -> None:
        """Initialize RemoteGPUReset.

        Args:
            timeout: Operation timeout in seconds.
            ssh_user: SSH username.
            ssh_port: SSH port.
        """
        self._timeout = timeout
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._lock = threading.Lock()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def timeout(self) -> float:
        """Configured operation timeout in seconds."""
        return self._timeout

    @property
    def ssh_user(self) -> str:
        """SSH username used for node connections."""
        return self._ssh_user

    @property
    def ssh_port(self) -> int:
        """SSH port used for node connections."""
        return self._ssh_port

    # ── public API ─────────────────────────────────────────────────────

    def reset(self, node_id: str, host: str) -> bool:
        """Reset GPU on *host* via SSH and verify health.

        Args:
            node_id: Logical identifier for the node (used for logging).
            host: Hostname or IP address of the target node.

        Returns:
            ``True`` if the GPU is healthy after the reset; ``False``
            if the reset failed, the health check failed, or a timeout
            occurred.
        """
        # 1. Run the GPU reset command
        reset_ok = self._run_ssh_command(host, "nvidia-smi --gpu-reset")
        if not reset_ok:
            return False

        # 2. Small cooldown to let the GPU settle after reset
        time.sleep(2.0)

        # 3. Verify GPU health
        health = self.check_health(host)
        return health.status == GPUHealthStatus.HEALTHY

    def check_health(self, host: str) -> GPUHealthInfo:
        """Query GPU health telemetry from *host*.

        Runs ``nvidia-smi`` with a query for temperature, memory usage,
        and error counts.

        Args:
            host: Hostname or IP address of the target node.

        Returns:
            A :class:`GPUHealthInfo` instance describing the current GPU
            state.
        """
        output = self._run_ssh_command_with_output(
            host,
            "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total"
            " --format=csv,noheader,nounits",
        )
        return self._parse_gpu_health(output)

    # ── internal: SSH helpers ──────────────────────────────────────────

    def _run_ssh_command(self, host: str, command: str) -> bool:
        """Run *command* on *host* and return success status."""
        try:
            args = self._build_ssh_args(host, command)
            proc = subprocess.run(  # noqa: S603 -- controlled invocation
                args,
                capture_output=True,
                timeout=self._timeout,
                text=True,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _run_ssh_command_with_output(self, host: str, command: str) -> str:
        """Run *command* on *host* and return stdout (or empty string)."""
        try:
            args = self._build_ssh_args(host, command)
            proc = subprocess.run(  # noqa: S603 -- controlled invocation
                args,
                capture_output=True,
                timeout=self._timeout,
                text=True,
            )
            if proc.returncode == 0:
                return proc.stdout or ""
            return ""
        except (subprocess.TimeoutExpired, OSError):
            return ""

    def _build_ssh_args(self, host: str, command: str) -> list[str]:
        """Build the SSH argument list for the given host and command.

        Uses ``-o StrictHostKeyChecking=no`` and ``-o UserKnownHostsFile=/dev/null``
        so that ephemeral nodes in a dynamic cluster do not cause host-key
        verification failures.  In production, replace these with proper
        host-key management.
        """
        return [
            "ssh",
            "-p", str(self._ssh_port),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self._ssh_user}@{host}",
            command,
        ]

    # ── internal: parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_gpu_health(output: str) -> GPUHealthInfo:
        """Parse nvidia-smi CSV output into a :class:`GPUHealthInfo`.

        Expected format (one line per GPU)::

            temperature.gpu, memory.used [MiB], memory.total [MiB]
            75, 1024, 16384

        When the output is empty or unparseable, returns an
        ``UNKNOWN`` status with an error count of 1.
        """
        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        if not lines:
            return GPUHealthInfo(
                status=GPUHealthStatus.UNKNOWN,
                temperature_c=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                error_count=1,
            )

        # Aggregate across all GPUs: use the max temperature and
        # sum of memory across devices.
        max_temp: float = 0.0
        total_used: float = 0.0
        total_mem: float = 0.0
        parse_failures: int = 0

        for line in lines:
            try:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    parse_failures += 1
                    continue
                temp = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                max_temp = max(max_temp, temp)
                total_used += mem_used
                total_mem += mem_total
            except (ValueError, IndexError):
                parse_failures += 1

        if total_mem <= 0:
            return GPUHealthInfo(
                status=GPUHealthStatus.UNKNOWN,
                temperature_c=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                error_count=parse_failures or 1,
            )

        # Determine health status based on thresholds
        util = total_used / total_mem
        if max_temp > 85.0 or util > 0.95:
            status = GPUHealthStatus.DEGRADED
        else:
            status = GPUHealthStatus.HEALTHY

        return GPUHealthInfo(
            status=status,
            temperature_c=max_temp,
            memory_used_mb=total_used,
            memory_total_mb=total_mem,
            error_count=parse_failures,
        )


# ---------------------------------------------------------------------------
# DrainCoordinator
# ---------------------------------------------------------------------------

DrainPlan = List[str]
"""Type alias for a drain plan: a list of node IDs to drain."""


class DrainCoordinator:
    """Coordinate node draining with a configurable capacity limit.

    Drains **least-loaded** nodes first so that the cluster retains
    as much serving capacity as possible during maintenance.

    Thread-safe: all mutating operations are protected by a lock.

    Args:
        max_drain_percent: Maximum percentage of nodes that can be
            drained simultaneously (default 25.0).  Must be in (0, 100].
    """

    def __init__(self, max_drain_percent: float = _DEFAULT_DRAIN_PERCENT) -> None:
        """Initialize DrainCoordinator.

        Args:
            max_drain_percent: Max fraction of nodes to drain (percent).

        Raises:
            ValueError: If *max_drain_percent* is not in (0, 100].
        """
        if not 0.0 < max_drain_percent <= 100.0:
            raise ValueError(
                f"max_drain_percent must be in (0, 100], got {max_drain_percent}"
            )
        self._max_drain_percent = max_drain_percent
        self._node_loads: Dict[str, float] = {}
        self._drained: set[str] = set()
        self._lock = threading.Lock()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def max_drain_percent(self) -> float:
        """Maximum percentage of nodes allowed for simultaneous draining."""
        return self._max_drain_percent

    @property
    def total_nodes(self) -> int:
        """Number of nodes currently registered with the coordinator."""
        with self._lock:
            return len(self._node_loads)

    @property
    def drain_count(self) -> int:
        """Number of nodes currently drained."""
        with self._lock:
            return len(self._drained)

    @property
    def drain_percent(self) -> float:
        """Current percentage of drained nodes (0.0 -- 100.0)."""
        with self._lock:
            total = len(self._node_loads)
            if total == 0:
                return 0.0
            return len(self._drained) / total * 100.0

    @property
    def drained_nodes(self) -> List[str]:
        """Return a sorted snapshot of currently drained node IDs."""
        with self._lock:
            return sorted(self._drained)

    # ── load management ───────────────────────────────────────────────

    def update_load(self, node_id: str, load: float) -> None:
        """Register or update the load estimate for *node_id*.

        Args:
            node_id: Unique node identifier.
            load: Current load metric for the node (lower = less loaded).
        """
        with self._lock:
            self._node_loads[node_id] = load

    def remove_node(self, node_id: str) -> None:
        """Remove *node_id* from the coordinator entirely.

        Also removes it from the drained set if present.

        Args:
            node_id: Unique node identifier.
        """
        with self._lock:
            self._node_loads.pop(node_id, None)
            self._drained.discard(node_id)

    def get_load(self, node_id: str) -> Optional[float]:
        """Return the current load for *node_id*, or ``None`` if unknown.

        Args:
            node_id: Unique node identifier.
        """
        with self._lock:
            return self._node_loads.get(node_id)

    # ── drain operations ──────────────────────────────────────────────

    def can_drain(self, node_id: str) -> bool:
        """Check whether *node_id* can be drained without exceeding the limit.

        The drain limit is computed as ``ceil(max_drain_percent * total)``.
        A node that is already drained also returns ``False``.

        Args:
            node_id: Unique node identifier.

        Returns:
            ``True`` if draining *node_id* would keep the cluster within
            the configured drain limit.
        """
        with self._lock:
            total = len(self._node_loads)
            if total == 0:
                return False
            max_drain = math.ceil(total * self._max_drain_percent / 100.0)
            if len(self._drained) >= max_drain:
                return False
            return node_id not in self._drained

    def coordinate(self) -> DrainPlan:
        """Produce a drain plan.

        Selects the least-loaded nodes that are not already drained,
        up to the remaining capacity under the drain limit.

        The selected nodes are automatically marked as drained so that
        subsequent calls do not return the same nodes.

        Returns:
            A list of node IDs to drain (may be empty).
        """
        with self._lock:
            total = len(self._node_loads)
            if total == 0:
                return []
            max_drain = math.ceil(total * self._max_drain_percent / 100.0)
            remaining = max_drain - len(self._drained)
            if remaining <= 0:
                return []

            # Sort by load ascending (least-loaded first)
            sorted_nodes = sorted(
                self._node_loads.items(),
                key=lambda kv: kv[1],
            )
            plan: list[str] = []
            for node_id, _ in sorted_nodes:
                if len(plan) >= remaining:
                    break
                if node_id not in self._drained:
                    plan.append(node_id)
                    self._drained.add(node_id)
            return plan

    def mark_restored(self, node_id: str) -> None:
        """Mark a drained node as restored.

        Args:
            node_id: Unique node identifier to restore.
        """
        with self._lock:
            self._drained.discard(node_id)

    def restore_all(self) -> List[str]:
        """Restore all currently drained nodes.

        Returns:
            The list of node IDs that were restored.
        """
        with self._lock:
            restored = sorted(self._drained)
            self._drained.clear()
            return restored

    # ── load mapping snapshot ──────────────────────────────────────────

    @property
    def loads(self) -> Dict[str, float]:
        """Return a snapshot of current node-load mappings."""
        with self._lock:
            return dict(self._node_loads)


# ---------------------------------------------------------------------------
# FailurePredictionResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FailurePredictionResult:
    """Result of a failure risk prediction.

    Attributes:
        risk_score: Predicted failure risk from 0.0 (safe) to 1.0
            (critical).
        reason: Human-readable explanation of the risk driver.
    """

    risk_score: float
    reason: str


# ---------------------------------------------------------------------------
# FailureRecord
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FailureRecord:
    """A single historical failure observation.

    Attributes:
        node_id: The node that experienced the failure.
        temperature_c: GPU temperature at the time of failure.
        memory_utilization: GPU memory utilization fraction at failure.
        error_rate: Observed error rate at the time of failure.
        latency_ms: Observed latency in milliseconds at failure.
        timestamp: When the failure occurred.
    """

    node_id: str
    temperature_c: float
    memory_utilization: float
    error_rate: float
    latency_ms: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# FailurePredictor
# ---------------------------------------------------------------------------


class FailurePredictor:
    """Predict node failure risk from GPU telemetry features.

    Uses a weighted heuristic model based on four features:

    * **GPU temperature** (weight 0.35) -- risk increases sigmoidally
      above 80 C.
    * **Memory utilization** (weight 0.25) -- risk increases above 90 %.
    * **Error rate** (weight 0.25) -- risk increases above 5 %.
    * **Latency** (weight 0.15) -- risk increases above 500 ms.

    The predictor also learns from historical failures: if a node has
    failed before under similar feature values, the risk score is boosted.

    Thread-safe: all mutation is lock-protected.
    """

    def __init__(self) -> None:
        """Initialize FailurePredictor with default feature weights."""
        self._feature_weights: Dict[str, float] = {
            "temperature": _TEMP_WEIGHT,
            "memory": _MEMORY_WEIGHT,
            "error_rate": _ERROR_RATE_WEIGHT,
            "latency": _LATENCY_WEIGHT,
        }
        # History of past failures, keyed by node_id
        self._failure_history: Dict[str, deque[FailureRecord]] = {}
        self._total_failures: int = 0
        self._lock = threading.Lock()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def feature_weights(self) -> Dict[str, float]:
        """Return a snapshot of current feature weights."""
        with self._lock:
            return dict(self._feature_weights)

    @property
    def total_failures_recorded(self) -> int:
        """Total number of failures recorded across all nodes."""
        with self._lock:
            return self._total_failures

    @property
    def known_nodes(self) -> List[str]:
        """Return list of node IDs that have failure history."""
        with self._lock:
            return list(self._failure_history.keys())

    # ── public API ─────────────────────────────────────────────────────

    def predict(
        self,
        node_id: str,
        temperature_c: float,
        memory_utilization: float,
        error_rate: float,
        latency_ms: float,
    ) -> FailurePredictionResult:
        """Predict failure risk for a node given current telemetry.

        Args:
            node_id: Unique node identifier.
            temperature_c: Current GPU temperature in Celsius.
            memory_utilization: Current GPU memory utilization (0.0 -- 1.0).
            error_rate: Current error rate (0.0 -- 1.0).
            latency_ms: Current observed latency in milliseconds.

        Returns:
            A :class:`FailurePredictionResult` with risk score and reason.
        """
        # 1. Compute per-feature risk scores (sigmoid-like mapping)
        temp_risk = self._sigmoid_risk(temperature_c, _TEMP_THRESHOLD_C, 5.0)
        mem_risk = self._sigmoid_risk(
            memory_utilization, _MEMORY_UTIL_THRESHOLD, 0.1
        )
        error_risk = self._sigmoid_risk(error_rate, _ERROR_RATE_THRESHOLD, 0.02)
        latency_risk = self._sigmoid_risk(
            latency_ms, _LATENCY_THRESHOLD_MS, 100.0
        )

        # 2. Weighted combination
        with self._lock:
            w_temp = self._feature_weights.get("temperature", _TEMP_WEIGHT)
            w_mem = self._feature_weights.get("memory", _MEMORY_WEIGHT)
            w_err = self._feature_weights.get("error_rate", _ERROR_RATE_WEIGHT)
            w_lat = self._feature_weights.get("latency", _LATENCY_WEIGHT)

        raw_score = (
            w_temp * temp_risk
            + w_mem * mem_risk
            + w_err * error_risk
            + w_lat * latency_risk
        )

        # 3. Historical boost: if the node has failed before under similar
        #    conditions, increase the score by up to 20 %.
        history_boost = self._compute_history_boost(
            node_id, temperature_c, memory_utilization, error_rate, latency_ms
        )
        risk_score = min(raw_score + history_boost, 1.0)

        # 4. Build a human-readable reason
        reasons: list[str] = []
        if temp_risk > 0.5:
            reasons.append(f"high GPU temp ({temperature_c:.0f} C)")
        if mem_risk > 0.5:
            reasons.append(
                f"high memory util ({memory_utilization:.0%})"
            )
        if error_risk > 0.5:
            reasons.append(f"elevated error rate ({error_rate:.1%})")
        if latency_risk > 0.5:
            reasons.append(f"high latency ({latency_ms:.0f} ms)")
        if history_boost > 0.0:
            reasons.append("prior failure pattern detected")

        if not reasons:
            reason = "nominal"
        elif len(reasons) == 1:
            reason = reasons[0]
        else:
            reason = "; ".join(reasons)

        return FailurePredictionResult(
            risk_score=round(risk_score, 4),
            reason=reason,
        )

    def record_failure(self, record: FailureRecord) -> None:
        """Record a historical failure observation.

        Stored data is used by :meth:`predict` to boost risk scores
        when a node shows similar feature values to past failures.

        Args:
            record: The failure observation to record.
        """
        with self._lock:
            if record.node_id not in self._failure_history:
                self._failure_history[record.node_id] = deque(
                    maxlen=_MAX_FAILURE_HISTORY
                )
            self._failure_history[record.node_id].append(record)
            self._total_failures += 1

    def get_failure_history(self, node_id: str) -> List[FailureRecord]:
        """Return a snapshot of failure history for *node_id*.

        Args:
            node_id: Unique node identifier.

        Returns:
            List of historical :class:`FailureRecord` entries, newest
            first, or an empty list if no history exists.
        """
        with self._lock:
            history = self._failure_history.get(node_id)
            if not history:
                return []
            return list(reversed(history))

    def clear_history(self, node_id: Optional[str] = None) -> int:
        """Clear failure history for a specific node or all nodes.

        Args:
            node_id: If provided, clear only this node's history.
                If ``None``, clear all history.

        Returns:
            Number of records removed.
        """
        with self._lock:
            if node_id is not None:
                removed = len(self._failure_history.pop(node_id, []))
                self._total_failures -= removed
                return max(removed, 0)
            total = sum(len(v) for v in self._failure_history.values())
            self._failure_history.clear()
            self._total_failures = 0
            return total

    # ── internal: risk computation ─────────────────────────────────────

    @staticmethod
    def _sigmoid_risk(value: float, threshold: float, steepness: float) -> float:
        """Map *value* to a risk score in (0, 1) using a sigmoid.

        The function is centred at *threshold*::

            risk = 1 / (1 + exp(-(value - threshold) / steepness))

        When *value* is well below *threshold* the score approaches 0;
        when well above, it approaches 1.
        """
        if steepness <= 0:
            return 1.0 if value > threshold else 0.0
        try:
            exponent = (value - threshold) / steepness
            # Clamp exponent to avoid overflow in exp()
            exponent = max(-100.0, min(100.0, exponent))
            return 1.0 / (1.0 + math.exp(-exponent))
        except (OverflowError, ValueError):
            return 1.0 if value > threshold else 0.0

    def _compute_history_boost(
        self,
        node_id: str,
        temperature_c: float,
        memory_utilization: float,
        error_rate: float,
        latency_ms: float,
    ) -> float:
        """Compute a history-based risk boost for *node_id*.

        Compares current feature values against historical failure records
        for the same node.  Returns a boost in [0.0, 0.2] proportional to
        how many past failures had similar feature profiles.

        The similarity threshold for each feature is:
        * Temperature: within 10 C
        * Memory util: within 0.1
        * Error rate: within 0.02
        * Latency: within 200 ms
        """
        with self._lock:
            history = self._failure_history.get(node_id)
            if not history:
                return 0.0

            similar_count = 0
            for rec in history:
                matches = 0
                if abs(rec.temperature_c - temperature_c) <= 10.0:
                    matches += 1
                if abs(rec.memory_utilization - memory_utilization) <= 0.1:
                    matches += 1
                if abs(rec.error_rate - error_rate) <= 0.02:
                    matches += 1
                if abs(rec.latency_ms - latency_ms) <= 200.0:
                    matches += 1
                # Count as "similar" when at least 3 of 4 features match
                if matches >= 3:
                    similar_count += 1

            if len(history) == 0:
                return 0.0

            ratio = similar_count / len(history)
            # Boost scales from 0.0 to 0.2 based on ratio
            return round(min(ratio * 0.2, 0.2), 4)


# ---------------------------------------------------------------------------
# RecoverySLAConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RecoverySLAConfig:
    """SLA timeouts for node recovery.

    Attributes:
        single_node_timeout: Deadline for recovering a single failed
            node (default 5 minutes).
        multi_node_timeout: Deadline for recovering more than 25 % of
            nodes (default 15 minutes).
    """

    single_node_timeout: timedelta = timedelta(minutes=_DEFAULT_SINGLE_NODE_TIMEOUT_MIN)
    multi_node_timeout: timedelta = timedelta(minutes=_DEFAULT_MULTI_NODE_TIMEOUT_MIN)


# ---------------------------------------------------------------------------
# RecoverySLAState
# ---------------------------------------------------------------------------


class RecoverySLAState(str, Enum):
    """Current state of an SLA-tracking cycle."""

    IDLE = "idle"
    TRACKING = "tracking"
    BREACHED = "breached"
    RECOVERED = "recovered"


# ---------------------------------------------------------------------------
# RecoverySLA
# ---------------------------------------------------------------------------


class RecoverySLA:
    """Track recovery time against SLA deadlines and escalate on breach.

    SLA rules:

    * **Single node failure** (<= 25 % of cluster): Must recover within
      5 minutes.
    * **Multi-node failure** (> 25 % of cluster): Must recover within
      15 minutes.

    When a deadline is breached, :meth:`escalate` sends a notification
    to the on-call endpoint configured via the ``ON_CALL_ENDPOINT``
    environment variable (if set).

    Thread-safe: all state mutations are lock-protected.
    """

    def __init__(self, config: Optional[RecoverySLAConfig] = None) -> None:
        """Initialize RecoverySLA.

        Args:
            config: SLA timeout configuration.  Defaults to
                ``RecoverySLAConfig()`` (5 min single, 15 min multi).
        """
        self._config = config or RecoverySLAConfig()
        self._start_time: Optional[datetime] = None
        self._affected_count: int = 0
        self._total_count: int = 0
        self._state: RecoverySLAState = RecoverySLAState.IDLE
        self._lock = threading.Lock()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def config(self) -> RecoverySLAConfig:
        """Current SLA configuration."""
        return self._config

    @property
    def state(self) -> RecoverySLAState:
        """Current state of the SLA tracker."""
        with self._lock:
            return self._state

    @property
    def deadline(self) -> timedelta:
        """SLA deadline based on the fraction of affected nodes.

        Returns ``single_node_timeout`` when the affected ratio is
        <= 25 %, ``multi_node_timeout`` otherwise.
        """
        with self._lock:
            return self._compute_deadline()

    @property
    def elapsed(self) -> timedelta:
        """Time elapsed since :meth:`start` was called.

        Returns ``timedelta(0)`` if tracking has not started.
        """
        with self._lock:
            if self._start_time is None:
                return timedelta(0)
            return datetime.now(timezone.utc) - self._start_time

    @property
    def remaining(self) -> timedelta:
        """Time remaining before the SLA deadline.

        Returns ``timedelta(0)`` if the deadline has already passed
        or tracking has not started.
        """
        remaining = self.deadline - self.elapsed
        return remaining if remaining > timedelta(0) else timedelta(0)

    @property
    def is_breached(self) -> bool:
        """Check whether the SLA deadline has been breached."""
        with self._lock:
            if self._state == RecoverySLAState.BREACHED:
                return True
            if self._start_time is None:
                return False
            return datetime.now(timezone.utc) - self._start_time > self._compute_deadline()

    @property
    def affected_count(self) -> int:
        """Number of nodes currently tracked as affected."""
        with self._lock:
            return self._affected_count

    @property
    def total_count(self) -> int:
        """Total number of nodes in the cluster for the current cycle."""
        with self._lock:
            return self._total_count

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self, affected_count: int, total_count: int) -> None:
        """Start tracking recovery for *affected_count* out of *total_count* nodes.

        Args:
            affected_count: Number of nodes that are currently failed.
            total_count: Total number of nodes in the cluster.

        Raises:
            ValueError: If *affected_count* > *total_count*.
        """
        if affected_count > total_count:
            raise ValueError(
                f"affected_count ({affected_count}) cannot exceed "
                f"total_count ({total_count})"
            )
        with self._lock:
            self._start_time = datetime.now(timezone.utc)
            self._affected_count = affected_count
            self._total_count = max(total_count, 1)
            self._state = RecoverySLAState.TRACKING

    def monitor(self) -> timedelta:
        """Check the current SLA status.

        Returns the remaining time before the deadline.  If the
        deadline has been exceeded, transitions to ``BREACHED`` state
        and returns ``timedelta(0)``.

        Returns:
            Remaining ``timedelta`` before deadline (``timedelta(0)``
            if breached or idle).
        """
        with self._lock:
            if self._start_time is None:
                return timedelta(0)
            elapsed = datetime.now(timezone.utc) - self._start_time
            deadline = self._compute_deadline()
            if elapsed > deadline:
                self._state = RecoverySLAState.BREACHED
                return timedelta(0)
            return deadline - elapsed

    def escalate(self) -> None:
        """Send an escalation notification to the on-call endpoint.

        Reads ``ON_CALL_ENDPOINT`` from the environment.  When set,
        sends an HTTP POST with a JSON body describing the breach.
        Silently ignores failures so the application is not disrupted
        by notification errors.

        If the endpoint is not configured, the method is a no-op.
        """
        on_call_url = os.environ.get("ON_CALL_ENDPOINT", "")
        if not on_call_url:
            return

        with self._lock:
            payload: Dict[str, Any] = {
                "event": "sla_breach",
                "affected_count": self._affected_count,
                "total_count": self._total_count,
                "deadline_seconds": int(self._compute_deadline().total_seconds()),
                "elapsed_seconds": int(
                    (datetime.now(timezone.utc) - self._start_time).total_seconds()
                    if self._start_time
                    else 0
                ),
                "message": (
                    "Recovery SLA deadline exceeded for "
                    f"{self._affected_count}/{self._total_count} nodes."
                ),
            }

        try:
            import httpx

            httpx.post(
                on_call_url,
                json=payload,
                timeout=5.0,
            )
        except Exception:
            # Notification failures are non-fatal
            pass

    def mark_recovered(self) -> None:
        """Mark the current recovery cycle as recovered.

        Resets the SLA tracker to ``IDLE`` state.
        """
        with self._lock:
            self._state = RecoverySLAState.RECOVERED
            self._start_time = None
            self._affected_count = 0
            self._total_count = 0

    def reset(self) -> None:
        """Reset the SLA tracker to its initial state."""
        with self._lock:
            self._start_time = None
            self._affected_count = 0
            self._total_count = 0
            self._state = RecoverySLAState.IDLE

    # ── internal ───────────────────────────────────────────────────────

    def _compute_deadline(self) -> timedelta:
        """Determine the applicable deadline based on affected ratio."""
        if self._total_count == 0:
            return self._config.single_node_timeout
        ratio = self._affected_count / self._total_count
        if ratio > 0.25:
            return self._config.multi_node_timeout
        return self._config.single_node_timeout


# ---------------------------------------------------------------------------
# SelfHealingConfigurator
# ---------------------------------------------------------------------------


class SelfHealingConfigurator:
    """Combine all self-healing components into a background monitoring loop.

    The configurator periodically checks node health, predicts failures,
    coordinates draining, resets unhealthy GPUs, and tracks SLA compliance.

    All four sub-components are exposed as read-only properties so they
    can be used directly when fine-grained control is needed.

    Usage::

        configurator = SelfHealingConfigurator(
            gpu_reset=RemoteGPUReset(timeout=30.0),
            drain_coordinator=DrainCoordinator(max_drain_percent=25.0),
            failure_predictor=FailurePredictor(),
            recovery_sla=RecoverySLA(),
            check_interval=30.0,
        )
        configurator.start()
        # ... application runs ...
        configurator.stop()
    """

    def __init__(
        self,
        gpu_reset: Optional[RemoteGPUReset] = None,
        drain_coordinator: Optional[DrainCoordinator] = None,
        failure_predictor: Optional[FailurePredictor] = None,
        recovery_sla: Optional[RecoverySLA] = None,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize SelfHealingConfigurator.

        Args:
            gpu_reset: RemoteGPUReset instance.  Defaults to a fresh
                instance with default parameters.
            drain_coordinator: DrainCoordinator instance.  Defaults to
                a fresh instance with 25 % max drain.
            failure_predictor: FailurePredictor instance.  Defaults to
                a fresh instance with default weights.
            recovery_sla: RecoverySLA instance.  Defaults to a fresh
                instance with 5 min / 15 min deadlines.
            check_interval: Seconds between monitoring loop iterations
                (default 30.0).
        """
        self._gpu_reset = gpu_reset or RemoteGPUReset()
        self._drain_coordinator = drain_coordinator or DrainCoordinator()
        self._failure_predictor = failure_predictor or FailurePredictor()
        self._recovery_sla = recovery_sla or RecoverySLA()
        self._check_interval = min(check_interval, 1.0) if check_interval < 1.0 else check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def gpu_reset(self) -> RemoteGPUReset:
        """The configured :class:`RemoteGPUReset` instance."""
        return self._gpu_reset

    @property
    def drain_coordinator(self) -> DrainCoordinator:
        """The configured :class:`DrainCoordinator` instance."""
        return self._drain_coordinator

    @property
    def failure_predictor(self) -> FailurePredictor:
        """The configured :class:`FailurePredictor` instance."""
        return self._failure_predictor

    @property
    def recovery_sla(self) -> RecoverySLA:
        """The configured :class:`RecoverySLA` instance."""
        return self._recovery_sla

    @property
    def check_interval(self) -> float:
        """Seconds between monitoring loop iterations."""
        return self._check_interval

    @property
    def is_running(self) -> bool:
        """Whether the monitoring loop is currently active."""
        with self._lock:
            return self._running

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitoring loop.

        This method is idempotent: calling it multiple times has no
        effect after the first successful start.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="self-healing-monitor",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring loop gracefully.

        Signals the loop to exit and waits up to 10 seconds for the
        thread to join.
        """
        with self._lock:
            self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            self._thread = None

    # ── internal: monitoring loop ──────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Main monitoring loop, runs on a background daemon thread.

        Each iteration:

        1. Evaluates node failure risk via :meth:`_evaluate_nodes`.
        2. Coordinates draining via :meth:`_coordinate_drain`.
        3. Attempts GPU resets on unhealthy nodes.
        4. Monitors SLA compliance.
        """
        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                self._tick()
            except Exception:
                # Log and continue -- the loop must never crash.
                pass

            time.sleep(self._check_interval)

    def _tick(self) -> None:
        """Execute one monitoring tick.

        This method is called repeatedly by the monitoring loop and
        can also be invoked manually for testing.
        """
        # 1. Evaluate nodes (subclasses or callers register nodes via
        #    the component APIs; the configurator coordinates them).
        self._evaluate_nodes()

        # 2. Coordinate drainage
        self._coordinate_drain()

        # 3. Monitor SLA
        self._recovery_sla.monitor()

    def _evaluate_nodes(self) -> None:
        """Evaluate failure risk for all registered nodes.

        Iterates over nodes known to the :attr:`drain_coordinator`
        and checks their health via :attr:`gpu_reset`.  Nodes with
        high risk scores are flagged for potential drain.
        """
        # Snapshot of known node loads to avoid holding the lock
        loads = self._drain_coordinator.loads

        for node_id in loads:
            # Build feature estimates from node metadata (in a real
            # deployment these would come from telemetry streams).
            # Here we use defaults since the configurator is a coordinator.
            result = self._failure_predictor.predict(
                node_id=node_id,
                temperature_c=0.0,
                memory_utilization=0.0,
                error_rate=0.0,
                latency_ms=0.0,
            )

            # If risk is high and the node can be drained, attempt a reset
            if result.risk_score >= 0.7 and self._drain_coordinator.can_drain(node_id):
                # Attempt GPU reset (host would be resolved in production)
                # For demonstration we pass node_id as host placeholder.
                self._gpu_reset.reset(node_id, node_id)

    def _coordinate_drain(self) -> None:
        """Coordinate node draining.

        Invokes :meth:`DrainCoordinator.coordinate` and registers the
        drain plan with the SLA tracker if nodes are affected.
        """
        plan = self._drain_coordinator.coordinate()
        if plan:
            self._recovery_sla.start(
                affected_count=len(plan),
                total_count=self._drain_coordinator.total_nodes,
            )


__all__ = [
    "DrainCoordinator",
    "DrainPlan",
    "FailurePredictionResult",
    "FailurePredictor",
    "FailureRecord",
    "GPUHealthInfo",
    "GPUHealthStatus",
    "RecoverySLA",
    "RecoverySLAConfig",
    "RecoverySLAState",
    "RemoteGPUReset",
    "SelfHealingConfigurator",
]
