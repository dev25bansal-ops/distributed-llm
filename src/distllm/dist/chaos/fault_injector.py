"""Fault injector for chaos engineering in distributed LLM inference.

Provides gRPC-level fault injection (latency, errors, message drops),
predefined chaos scenarios with steady-state hypotheses, blast-radius
controls, and template generators for Litmus and Chaos Mesh experiments.

Usage::

    injector = FaultInjector()
    injector.inject_latency("ForwardPass", prob=0.5, delay_ms=200)
    injector.inject_error("HealthCheck", prob=0.1, code=grpc.StatusCode.UNAVAILABLE)

    scenario = ChaosScenario(
        name="network-delay-forward",
        scenario_type=ChaosScenarioType.LATENCY_SPIKE,
        duration=30.0,
        faults=[
            LatencyFault(method="ForwardPass", probability=1.0, delay_ms=500),
        ],
    )
    scenario.run(injector)
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable

from loguru import logger

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ── Supporting Data Types ───────────────────────────────────────────────────


class ChaosScenarioType(Enum):
    """Well-known chaos scenario types for distributed LLM inference."""

    NODE_FAILURE = auto()
    NETWORK_PARTITION = auto()
    LATENCY_SPIKE = auto()
    PACKET_LOSS = auto()
    CPU_THROTTLE = auto()


class FaultType(Enum):
    """Types of faults that can be injected."""

    LATENCY = auto()
    ERROR = auto()
    MESSAGE_DROP = auto()


@dataclass(frozen=True)
class LatencyFault:
    """A latency-injection fault specification.

    Attributes:
        method: gRPC method name (e.g. "ForwardPass", "HealthCheck").
        probability: Probability (0.0-1.0) of injecting this fault per call.
        delay_ms: Artificial delay in milliseconds.
    """

    method: str
    probability: float = 1.0
    delay_ms: float = 100.0


@dataclass(frozen=True)
class ErrorFault:
    """An error-injection fault specification.

    Attributes:
        method: gRPC method name.
        probability: Probability (0.0-1.0) of injecting this fault per call.
        code: gRPC status code to return (int or grpc.StatusCode).
        message: Optional error detail message.
    """

    method: str
    probability: float = 1.0
    code: int = 14  # grpc.StatusCode.UNAVAILABLE
    message: str = "Injected fault"


@dataclass(frozen=True)
class MessageDropFault:
    """A message-drop fault specification.

    Attributes:
        method: gRPC method name.
        probability: Probability (0.0-1.0) of dropping the message.
    """

    method: str
    probability: float = 1.0


# ── Fault Configuration ─────────────────────────────────────────────────────


@dataclass
class _LatencyRule:
    """Internal latency injection rule."""

    method: str
    prob: float
    delay_ms: float


@dataclass
class _ErrorRule:
    """Internal error injection rule."""

    method: str
    prob: float
    code: int
    message: str


@dataclass
class _DropRule:
    """Internal message drop rule."""

    method: str
    prob: float


# ── FaultInjector ───────────────────────────────────────────────────────────


class FaultInjector:
    """gRPC-interceptor-compatible fault injector for chaos engineering.

    Maintains independent rule sets for latency, error, and message-drop
    faults.  Each rule is matched by gRPC method name and applied
    probabilistically.

    The injector provides ``unary_interceptor()`` and ``streaming_interceptor()``
    callables that follow the gRPC Python interceptor convention so they can
    be wired into ``grpc.intercept_channel()`` or a server interceptor stack.

    Thread-safe for concurrent rule registration and interceptor invocation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latency_rules: list[_LatencyRule] = []
        self._error_rules: list[_ErrorRule] = []
        self._drop_rules: list[_DropRule] = []

    # ------------------------------------------------------------------
    # Public rule-registration API
    # ------------------------------------------------------------------

    def inject_latency(
        self,
        method: str,
        prob: float = 1.0,
        delay_ms: float = 100.0,
    ) -> None:
        """Register a latency-injection rule.

        Args:
            method: gRPC method name (e.g. ``"ForwardPass"``).
            prob: Probability (0.0-1.0) of injecting delay per call.
            delay_ms: Artificial delay in milliseconds.
        """
        with self._lock:
            self._latency_rules.append(
                _LatencyRule(method=method, prob=prob, delay_ms=delay_ms)
            )
        logger.info(
            "Injected latency rule: method={} prob={} delay_ms={}",
            method,
            prob,
            delay_ms,
        )

    def inject_error(
        self,
        method: str,
        prob: float = 1.0,
        code: int = 14,
        message: str = "Injected fault",
    ) -> None:
        """Register an error-injection rule.

        Args:
            method: gRPC method name.
            prob: Probability (0.0-1.0) of returning an error per call.
            code: gRPC status code (default ``14`` = UNAVAILABLE).
            message: Optional error detail message.
        """
        with self._lock:
            self._error_rules.append(
                _ErrorRule(method=method, prob=prob, code=code, message=message)
            )
        logger.info(
            "Injected error rule: method={} prob={} code={}",
            method,
            prob,
            code,
        )

    def inject_message_drop(self, method: str, prob: float = 1.0) -> None:
        """Register a message-drop rule.

        Drops the request entirely — the interceptor returns ``None``
        which the gRPC runtime treats as a transport-level failure.

        Args:
            method: gRPC method name.
            prob: Probability (0.0-1.0) of dropping the message.
        """
        with self._lock:
            self._drop_rules.append(_DropRule(method=method, prob=prob))
        logger.info(
            "Injected message drop rule: method={} prob={}",
            method,
            prob,
        )

    def reset(self) -> None:
        """Clear all fault injection rules."""
        with self._lock:
            self._latency_rules.clear()
            self._error_rules.clear()
            self._drop_rules.clear()
        logger.info("All fault injection rules cleared")

    @property
    def rule_count(self) -> int:
        """Total number of active fault injection rules."""
        with self._lock:
            return (
                len(self._latency_rules)
                + len(self._error_rules)
                + len(self._drop_rules)
            )

    @property
    def is_active(self) -> bool:
        """True when at least one fault injection rule is registered."""
        return self.rule_count > 0

    # ------------------------------------------------------------------
    # Interceptors
    # ------------------------------------------------------------------

    def _matches_method(self, rules: list, method: str) -> list:
        """Return rules whose method pattern matches *method*."""
        return [r for r in rules if r.method == method]

    def _should_apply(self, prob: float) -> bool:
        """Return True with probability *prob*."""
        return prob >= 1.0 or (prob > 0.0 and random.random() < prob)

    def unary_interceptor(
        self,
    ) -> Callable[[Callable, Any, Any], Any]:
        """Build a unary gRPC interceptor callable.

        The returned callable follows the same signature as
        ``grpc.UnaryUnaryClientInterceptor.intercept_unary_unary``:

            ``intercept(continuation, client_call_details, request) -> response``

        It checks latency, error, and drop rules in order.  If a drop rule
        matches, ``None`` is returned (simulating a transport failure).  If an
        error rule matches, a :class:`grpc.RpcError` is raised.  If a latency
        rule matches, an artificial sleep is applied before forwarding.

        Usage::

            channel = grpc.intercept_channel(
                channel,
                FaultInjectorUnaryInterceptor(fault_injector),
            )
        """
        injector = self

        def _intercept(
            continuation: Callable,
            client_call_details: Any,
            request: Any,
        ) -> Any:
            # Extract the method name from the client_call_details.
            method = _get_method_name(client_call_details)

            with injector._lock:
                drops = injector._matches_method(injector._drop_rules, method)
                errors = injector._matches_method(injector._error_rules, method)
                latencies = injector._matches_method(
                    injector._latency_rules, method
                )

            # Check drop rules first (short-circuit).
            for rule in drops:
                if injector._should_apply(rule.prob):
                    logger.debug("Dropping unary call to {}", method)
                    return None

            # Check error rules.
            for rule in errors:
                if injector._should_apply(rule.prob):
                    logger.debug("Injecting error on unary call to {}", method)
                    _raise_grpc_error(rule.code, rule.message)

            # Check latency rules (apply the *largest* matching delay).
            max_delay = 0.0
            for rule in latencies:
                if injector._should_apply(rule.prob):
                    max_delay = max(max_delay, rule.delay_ms)
            if max_delay > 0.0:
                logger.debug("Injecting {} ms latency on unary call to {}", max_delay, method)
                time.sleep(max_delay / 1000.0)

            return continuation(client_call_details, request)

        return _intercept

    def streaming_interceptor(
        self,
    ) -> Callable[[Callable, Any, Any], Any]:
        """Build a streaming gRPC interceptor callable.

        The returned callable follows the same signature as
        ``grpc.UnaryStreamClientInterceptor.intercept_unary_stream``.

        Behaves identically to :meth:`unary_interceptor` but handles
        streaming (response-stream) RPCs.
        """
        injector = self

        def _intercept(
            continuation: Callable,
            client_call_details: Any,
            request: Any,
        ) -> Any:
            method = _get_method_name(client_call_details)

            with injector._lock:
                drops = injector._matches_method(injector._drop_rules, method)
                errors = injector._matches_method(injector._error_rules, method)
                latencies = injector._matches_method(
                    injector._latency_rules, method
                )

            for rule in drops:
                if injector._should_apply(rule.prob):
                    logger.debug("Dropping streaming call to {}", method)
                    return None

            for rule in errors:
                if injector._should_apply(rule.prob):
                    logger.debug("Injecting error on streaming call to {}", method)
                    _raise_grpc_error(rule.code, rule.message)

            max_delay = 0.0
            for rule in latencies:
                if injector._should_apply(rule.prob):
                    max_delay = max(max_delay, rule.delay_ms)
            if max_delay > 0.0:
                logger.debug(
                    "Injecting {} ms latency on streaming call to {}", max_delay, method
                )
                time.sleep(max_delay / 1000.0)

            return continuation(client_call_details, request)

        return _intercept


# ── SteadyStateChecker ──────────────────────────────────────────────────────


@dataclass
class SteadyStateMetrics:
    """Snapshot of key observability metrics at a point in time.

    Attributes:
        timestamp: When the snapshot was taken.
        p99_latency_ms: P99 latency in milliseconds.
        throughput_req_per_sec: Requests per second.
        error_rate: Fraction of requests that resulted in errors (0.0-1.0).
    """

    timestamp: datetime
    p99_latency_ms: float
    throughput_req_per_sec: float
    error_rate: float

    def degraded_relative_to(
        self,
        baseline: SteadyStateMetrics,
        latency_threshold: float = 2.0,
        throughput_threshold: float = 0.5,
        error_rate_threshold: float = 0.05,
    ) -> bool:
        """Check whether this snapshot is degraded relative to a baseline.

        Args:
            baseline: The baseline (steady-state) metrics snapshot.
            latency_threshold: Max acceptable latency multiplier over baseline.
            throughput_threshold: Min acceptable throughput fraction of baseline.
            error_rate_threshold: Max acceptable absolute error rate.

        Returns:
            True if any metric exceeds its degradation threshold.
        """
        if baseline.p99_latency_ms > 0 and self.p99_latency_ms > 0:
            if self.p99_latency_ms / baseline.p99_latency_ms > latency_threshold:
                return True
        if baseline.throughput_req_per_sec > 0 and self.throughput_req_per_sec > 0:
            if self.throughput_req_per_sec / baseline.throughput_req_per_sec < throughput_threshold:
                return True
        if self.error_rate > error_rate_threshold:
            return True
        return False


class SteadyStateChecker:
    """Captures and compares steady-state metrics for chaos experiments.

    Used before, during, and after a fault injection to evaluate whether
    the system remained healthy, was temporarily degraded, or failed.
    """

    def __init__(
        self,
        metric_collector: Callable[[], SteadyStateMetrics] | None = None,
    ) -> None:
        """Initialize the checker.

        Args:
            metric_collector: Optional callable that returns the current
                metrics snapshot.  If omitted, metrics will be zeros
                (useful when metrics are collected externally).
        """
        self._collector = metric_collector
        self._baseline: SteadyStateMetrics | None = None
        self._during: SteadyStateMetrics | None = None
        self._after: SteadyStateMetrics | None = None

    def capture_baseline(self) -> SteadyStateMetrics:
        """Capture a baseline metrics snapshot before fault injection."""
        metrics = self._collect_metrics()
        self._baseline = metrics
        logger.info(
            "Steady-state baseline: p99={:.1f}ms throughput={:.1f} req/s error_rate={:.3f}",
            metrics.p99_latency_ms,
            metrics.throughput_req_per_sec,
            metrics.error_rate,
        )
        return metrics

    def capture_during(self) -> SteadyStateMetrics:
        """Capture a metrics snapshot during fault injection."""
        metrics = self._collect_metrics()
        self._during = metrics
        return metrics

    def capture_after(self) -> SteadyStateMetrics:
        """Capture a metrics snapshot after fault injection stops."""
        metrics = self._collect_metrics()
        self._after = metrics
        return metrics

    def check(self) -> tuple[bool, SteadyStateMetrics | None, SteadyStateMetrics | None, SteadyStateMetrics | None]:
        """Compare all three snapshots and return the verdict.

        Returns:
            A tuple of ``(passed, baseline, during, after)`` where *passed*
            is True if the system returned to steady state (the *after*
            snapshot is not degraded relative to *baseline*).
        """
        if self._baseline is None or self._after is None:
            logger.warning("SteadyStateChecker.check() called without complete snapshots")
            return (False, self._baseline, self._during, self._after)

        degraded = self._after.degraded_relative_to(self._baseline)
        passed = not degraded

        if passed:
            logger.info("Steady-state hypothesis PASSED — system recovered fully")
        else:
            logger.warning("Steady-state hypothesis FAILED — system still degraded")

        return (passed, self._baseline, self._during, self._after)

    def _collect_metrics(self) -> SteadyStateMetrics:
        if self._collector is not None:
            return self._collector()
        return SteadyStateMetrics(
            timestamp=datetime.now(timezone.utc),
            p99_latency_ms=0.0,
            throughput_req_per_sec=0.0,
            error_rate=0.0,
        )


# ── BlastRadiusControl ──────────────────────────────────────────────────────


@dataclass
class BlastRadiusControl:
    """Limits the blast radius of chaos experiments to prevent cascading failure.

    Attributes:
        max_nodes: Maximum number of nodes that can be targeted (default 1).
        allowed_nodes: Set of node addresses allowed for fault injection.
            Empty set means no node-level restriction.
        allowed_services: Set of service names allowed for fault injection.
            Empty set means no service-level restriction.
        allowed_methods: Set of gRPC method names allowed for fault injection.
            Empty set means no method-level restriction.
        maintenance_mode: When True, all fault injection is suppressed.
        maintenance_reason: Optional reason string for maintenance mode.
    """

    max_nodes: int = 1
    allowed_nodes: set[str] = field(default_factory=set)
    allowed_services: set[str] = field(default_factory=set)
    allowed_methods: set[str] = field(default_factory=set)
    maintenance_mode: bool = False
    maintenance_reason: str | None = None

    def can_inject(
        self,
        node: str | None = None,
        service: str | None = None,
        method: str | None = None,
    ) -> bool:
        """Check whether fault injection is permitted given the current limits.

        Args:
            node: Optional node identifier to check.
            service: Optional service name to check.
            method: Optional gRPC method name to check.

        Returns:
            True if injection is permitted, False otherwise.
        """
        # Maintenance mode blocks everything.
        if self.maintenance_mode:
            logger.debug(
                "Fault injection suppressed: maintenance mode{}",
                f" ({self.maintenance_reason})" if self.maintenance_reason else "",
            )
            return False

        # Node-level restriction.
        if node is not None and self.allowed_nodes:
            if node not in self.allowed_nodes:
                logger.debug("Fault injection blocked: node {} not in allowed set", node)
                return False

        # Service-level restriction.
        if service is not None and self.allowed_services:
            if service not in self.allowed_services:
                logger.debug("Fault injection blocked: service {} not in allowed set", service)
                return False

        # Method-level restriction.
        if method is not None and self.allowed_methods:
            if method not in self.allowed_methods:
                logger.debug("Fault injection blocked: method {} not in allowed set", method)
                return False

        return True

    def __post_init__(self) -> None:
        """Validate constraints after initialization."""
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")

    def with_maintenance(self, reason: str = "") -> BlastRadiusControl:
        """Return a copy of this control with maintenance mode enabled."""
        return BlastRadiusControl(
            max_nodes=self.max_nodes,
            allowed_nodes=self.allowed_nodes.copy(),
            allowed_services=self.allowed_services.copy(),
            allowed_methods=self.allowed_methods.copy(),
            maintenance_mode=True,
            maintenance_reason=reason or self.maintenance_reason,
        )

    def without_maintenance(self) -> BlastRadiusControl:
        """Return a copy of this control with maintenance mode disabled."""
        return BlastRadiusControl(
            max_nodes=self.max_nodes,
            allowed_nodes=self.allowed_nodes.copy(),
            allowed_services=self.allowed_services.copy(),
            allowed_methods=self.allowed_methods.copy(),
            maintenance_mode=False,
            maintenance_reason=None,
        )


# ── ChaosScenario ───────────────────────────────────────────────────────────


@dataclass
class ChaosScenario:
    """A named, bounded chaos experiment with fault specifications.

    Attributes:
        name: Human-readable name for the scenario.
        scenario_type: The category of chaos being introduced.
        description: Optional longer description.
        duration: Duration in seconds for the scenario.
        faults: List of fault specifications to apply.
        blast_radius: Optional blast-radius controls.
        steady_state_checker: Optional steady-state checker to run
            before, during, and after the experiment.
    """

    name: str
    scenario_type: ChaosScenarioType
    description: str = ""
    duration: float = 10.0
    faults: list[LatencyFault | ErrorFault | MessageDropFault] = field(default_factory=list)
    blast_radius: BlastRadiusControl | None = None
    steady_state_checker: SteadyStateChecker | None = None

    def run(self, injector: FaultInjector) -> dict[str, Any]:
        """Execute the chaos scenario.

        1. Captures steady-state baseline (if a checker is configured).
        2. Registers all faults on the injector.
        3. Waits for the scenario duration.
        4. Removes all injected faults.
        5. Captures the post-experiment snapshot and returns the result.

        Args:
            injector: The :class:`FaultInjector` to register faults on.

        Returns:
            A dict with keys ``"name"``, ``"duration"``, ``"steady_state_passed"``,
            ``"baseline"``, ``"during"``, ``"after"``, and ``"timestamp"``.
        """
        logger.info("Starting chaos scenario: {} ({})", self.name, self.scenario_type.name)

        result: dict[str, Any] = {
            "name": self.name,
            "scenario_type": self.scenario_type.name,
            "duration": self.duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steady_state_passed": None,
            "baseline": None,
            "during": None,
            "after": None,
        }

        # Steady-state baseline.
        checker = self.steady_state_checker
        if checker is not None:
            result["baseline"] = checker.capture_baseline()

        # Register all faults.
        for fault in self.faults:
            if isinstance(fault, LatencyFault):
                injector.inject_latency(
                    method=fault.method,
                    prob=fault.probability,
                    delay_ms=fault.delay_ms,
                )
            elif isinstance(fault, ErrorFault):
                injector.inject_error(
                    method=fault.method,
                    prob=fault.probability,
                    code=fault.code,
                    message=fault.message,
                )
            elif isinstance(fault, MessageDropFault):
                injector.inject_message_drop(
                    method=fault.method,
                    prob=fault.probability,
                )

        # Capture mid-experiment snapshot (after a short settling delay).
        if checker is not None:
            time.sleep(1.0)
            result["during"] = checker.capture_during()

        # Wait for the scenario duration.
        logger.info("Scenario {} running for {:.1f}s ...", self.name, self.duration)
        time.sleep(max(0.0, self.duration - 1.0))

        # Clean up all injected faults.
        injector.reset()
        logger.info("Scenario {} faults cleared", self.name)

        # Post-experiment verification.
        if checker is not None:
            time.sleep(2.0)  # Allow system to stabilise.
            result["after"] = checker.capture_after()
            passed, baseline, during, after = checker.check()
            result["steady_state_passed"] = passed
            result["baseline"] = baseline
            result["during"] = during
            result["after"] = after

        return result

    # ── Built-in scenario factories ──────────────────────────────────────

    @classmethod
    def node_failure(
        cls,
        node: str = "node-0",
        duration: float = 30.0,
        method: str = "HealthCheck",
    ) -> ChaosScenario:
        """Create a node-failure scenario that drops health-check messages."""
        return cls(
            name=f"node-failure-{node}",
            scenario_type=ChaosScenarioType.NODE_FAILURE,
            description=f"Simulates failure of node {node} by dropping {method} messages",
            duration=duration,
            faults=[MessageDropFault(method=method, probability=1.0)],
            blast_radius=BlastRadiusControl(max_nodes=1),
        )

    @classmethod
    def network_partition(
        cls,
        nodes: tuple[str, str] = ("node-0", "node-1"),
        duration: float = 60.0,
        method: str = "ForwardPass",
    ) -> ChaosScenario:
        """Create a network-partition scenario between two nodes."""
        return cls(
            name=f"network-partition-{nodes[0]}-{nodes[1]}",
            scenario_type=ChaosScenarioType.NETWORK_PARTITION,
            description=f"Simulates network partition between {nodes[0]} and {nodes[1]}",
            duration=duration,
            faults=[
                MessageDropFault(method=method, probability=1.0),
                LatencyFault(method=method, probability=1.0, delay_ms=30000.0),
            ],
        )

    @classmethod
    def latency_spike(
        cls,
        method: str = "ForwardPass",
        delay_ms: float = 500.0,
        duration: float = 30.0,
    ) -> ChaosScenario:
        """Create a latency-spike scenario on a specific gRPC method."""
        return cls(
            name=f"latency-spike-{method}",
            scenario_type=ChaosScenarioType.LATENCY_SPIKE,
            description=f"Injects {delay_ms}ms latency on {method} calls",
            duration=duration,
            faults=[LatencyFault(method=method, probability=1.0, delay_ms=delay_ms)],
        )

    @classmethod
    def packet_loss(
        cls,
        method: str = "ForwardPass",
        loss_probability: float = 0.3,
        duration: float = 30.0,
    ) -> ChaosScenario:
        """Create a packet-loss scenario with probabilistic message drops."""
        return cls(
            name=f"packet-loss-{method}",
            scenario_type=ChaosScenarioType.PACKET_LOSS,
            description=f"Injects {loss_probability:.0%} message drop rate on {method}",
            duration=duration,
            faults=[MessageDropFault(method=method, probability=loss_probability)],
        )

    @classmethod
    def cpu_throttle(
        cls,
        method: str = "ForwardPass",
        delay_ms: float = 1000.0,
        duration: float = 60.0,
    ) -> ChaosScenario:
        """Create a CPU-throttle scenario simulating resource contention."""
        return cls(
            name=f"cpu-throttle-{method}",
            scenario_type=ChaosScenarioType.CPU_THROTTLE,
            description=f"Simulates CPU throttling with {delay_ms}ms latency on {method}",
            duration=duration,
            faults=[
                LatencyFault(method=method, probability=0.7, delay_ms=delay_ms),
            ],
        )


# ── ChaosTemplate ───────────────────────────────────────────────────────────


class ChaosTemplate:
    """Generates experiment YAML for third-party chaos engineering platforms.

    Produces Litmus and Chaos Mesh experiment definitions from internal
    :class:`ChaosScenario` objects so the same scenarios can be executed
    against real Kubernetes clusters.
    """

    # ── Litmus Experiment Templates ──────────────────────────────────────

    @staticmethod
    def generate_litmus_experiment(scenario: ChaosScenario) -> str:
        """Generate a Litmus chaos experiment YAML from a scenario.

        Args:
            scenario: The chaos scenario to translate.

        Returns:
            A YAML string representing a Litmus ChaosEngine or
            ChaosExperiment resource.

        Raises:
            RuntimeError: If PyYAML is not installed.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is required to generate Litmus templates")

        manifest = _build_litmus_base(scenario)

        # Determine which experiment template to use.
        if scenario.scenario_type == ChaosScenarioType.NODE_FAILURE:
            _apply_litmus_pod_kill(scenario, manifest)
        elif scenario.scenario_type in (
            ChaosScenarioType.LATENCY_SPIKE,
            ChaosScenarioType.NETWORK_PARTITION,
            ChaosScenarioType.PACKET_LOSS,
        ):
            _apply_litmus_network_delay(scenario, manifest)
        elif scenario.scenario_type == ChaosScenarioType.CPU_THROTTLE:
            _apply_litmus_cpu_stress(scenario, manifest)
        else:
            _apply_litmus_pod_kill(scenario, manifest)

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    # ── Chaos Mesh Experiment Templates ──────────────────────────────────

    @staticmethod
    def generate_chaos_mesh_experiment(scenario: ChaosScenario) -> str:
        """Generate a Chaos Mesh experiment YAML from a scenario.

        Args:
            scenario: The chaos scenario to translate.

        Returns:
            A YAML string representing a Chaos Mesh ``Schedule`` or
            ``*Chaos`` resource.

        Raises:
            RuntimeError: If PyYAML is not installed.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is required to generate Chaos Mesh templates")

        manifest = _build_chaos_mesh_base(scenario)

        if scenario.scenario_type == ChaosScenarioType.NODE_FAILURE:
            _apply_chaos_mesh_pod_kill(scenario, manifest)
        elif scenario.scenario_type in (
            ChaosScenarioType.LATENCY_SPIKE,
            ChaosScenarioType.NETWORK_PARTITION,
            ChaosScenarioType.PACKET_LOSS,
        ):
            _apply_chaos_mesh_network_delay(scenario, manifest)
        elif scenario.scenario_type == ChaosScenarioType.CPU_THROTTLE:
            _apply_chaos_mesh_cpu_stress(scenario, manifest)
        else:
            _apply_chaos_mesh_pod_kill(scenario, manifest)

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    # ── Built-in Template Factories ──────────────────────────────────────

    @staticmethod
    def pod_kill_template(
        namespace: str = "distllm",
        pod_label: str = "app=distllm-node",
        duration: str = "30s",
        target_node: str = "node-0",
    ) -> str:
        """Generate a Litmus pod-kill experiment YAML directly.

        Args:
            namespace: Kubernetes namespace.
            pod_label: Label selector for the target pod.
            duration: Experiment duration (e.g. ``"30s"``).
            target_node: Node name for annotation.

        Returns:
            A Litmus experiment YAML string.
        """
        scenario = ChaosScenario.node_failure(node=target_node, duration=30.0)
        manifest = _build_litmus_base(scenario)
        manifest["spec"]["engineState"] = "active"
        manifest["spec"]["annotationCheck"] = "false"
        manifest["spec"]["chaosServiceAccount"] = "litmus-admin"
        manifest["spec"]["experiments"] = [
            {
                "name": "pod-kill",
                "spec": {
                    "components": {
                        "env": [
                            {"name": "TOTAL_CHAOS_DURATION", "value": duration},
                            {"name": "CHAOS_INTERVAL", "value": "10"},
                            {"name": "FORCE", "value": "true"},
                            {"name": "TARGET_PODS", "value": pod_label},
                        ],
                    },
                    "probe": [
                        {
                            "name": "check-node-health",
                            "type": "httpProbe",
                            "httpProbe": {
                                "url": f"http://{target_node}:50051/healthz",
                                "insecureSkipVerify": True,
                            },
                        },
                    ],
                },
            },
        ]
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    @staticmethod
    def network_delay_template(
        namespace: str = "distllm",
        pod_label: str = "app=distllm-node",
        delay_ms: str = "500",
        duration: str = "30s",
    ) -> str:
        """Generate a Litmus network-delay experiment YAML directly.

        Args:
            namespace: Kubernetes namespace.
            pod_label: Label selector for the target pod.
            delay_ms: Network delay in milliseconds.
            duration: Experiment duration.

        Returns:
            A Litmus experiment YAML string.
        """
        scenario = ChaosScenario.latency_spike(delay_ms=float(delay_ms), duration=30.0)
        manifest = _build_litmus_base(scenario)
        manifest["spec"]["engineState"] = "active"
        manifest["spec"]["annotationCheck"] = "false"
        manifest["spec"]["chaosServiceAccount"] = "litmus-admin"
        manifest["spec"]["experiments"] = [
            {
                "name": "pod-network-latency",
                "spec": {
                    "components": {
                        "env": [
                            {"name": "TOTAL_CHAOS_DURATION", "value": duration},
                            {"name": "CHAOS_INTERVAL", "value": "10"},
                            {"name": "NETWORK_LATENCY", "value": delay_ms},
                            {"name": "TARGET_PODS", "value": pod_label},
                        ],
                    },
                },
            },
        ]
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    @staticmethod
    def cpu_stress_template(
        namespace: str = "distllm",
        pod_label: str = "app=distllm-node",
        cpu_cores: str = "1",
        duration: str = "60s",
    ) -> str:
        """Generate a Litmus CPU-stress experiment YAML directly.

        Args:
            namespace: Kubernetes namespace.
            pod_label: Label selector for the target pod.
            cpu_cores: Number of CPU cores to stress.
            duration: Experiment duration.

        Returns:
            A Litmus experiment YAML string.
        """
        scenario = ChaosScenario.cpu_throttle(duration=60.0)
        manifest = _build_litmus_base(scenario)
        manifest["spec"]["engineState"] = "active"
        manifest["spec"]["annotationCheck"] = "false"
        manifest["spec"]["chaosServiceAccount"] = "litmus-admin"
        manifest["spec"]["experiments"] = [
            {
                "name": "pod-cpu-stress",
                "spec": {
                    "components": {
                        "env": [
                            {"name": "TOTAL_CHAOS_DURATION", "value": duration},
                            {"name": "CPU_CORES", "value": cpu_cores},
                            {"name": "CPU_LOAD", "value": "100"},
                            {"name": "TARGET_PODS", "value": pod_label},
                        ],
                    },
                },
            },
        ]
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def _build_litmus_base(scenario: ChaosScenario) -> dict[str, Any]:
    """Build the skeleton Litmus ChaosEngine manifest."""
    return {
        "apiVersion": "litmuschaos.io/v1alpha1",
        "kind": "ChaosEngine",
        "metadata": {
            "name": f"{scenario.name}-engine",
            "namespace": "distllm",
            "labels": {
                "chaos": scenario.name,
                "scenario-type": scenario.scenario_type.name.lower(),
            },
        },
        "spec": {
            "engineState": "active",
            "annotationCheck": "false",
            "chaosServiceAccount": "litmus-admin",
            "monitoring": {
                "labels": {"app": "distllm-monitor"},
            },
            "jobCleanUpPolicy": "retain",
            "experiments": [],
        },
    }


def _apply_litmus_pod_kill(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a pod-kill experiment to a Litmus manifest."""
    faults = _find_faults_by_type(scenario, MessageDropFault)
    drop_rate = faults[0].probability if faults else 1.0
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["experiments"].append(
        {
            "name": "pod-kill",
            "spec": {
                "components": {
                    "env": [
                        {"name": "TOTAL_CHAOS_DURATION", "value": duration_str},
                        {"name": "CHAOS_INTERVAL", "value": "10"},
                        {"name": "FORCE", "value": "true"},
                        {"name": "TARGET_PODS", "value": f"app=distllm-node-{scenario.name}"},
                        {"name": "PODS_AFFECTED_PERC", "value": str(int(drop_rate * 100))},
                    ],
                },
            },
        },
    )


def _apply_litmus_network_delay(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a network-latency experiment to a Litmus manifest."""
    faults = _find_faults_by_type(scenario, LatencyFault)
    delay_ms = str(int(faults[0].delay_ms)) if faults else "100"
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["experiments"].append(
        {
            "name": "pod-network-latency",
            "spec": {
                "components": {
                    "env": [
                        {"name": "TOTAL_CHAOS_DURATION", "value": duration_str},
                        {"name": "NETWORK_LATENCY", "value": delay_ms},
                        {"name": "TARGET_PODS", "value": "app=distllm-node"},
                        {"name": "JITTER", "value": "0"},
                    ],
                },
            },
        },
    )


def _apply_litmus_cpu_stress(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a CPU-stress experiment to a Litmus manifest."""
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["experiments"].append(
        {
            "name": "pod-cpu-stress",
            "spec": {
                "components": {
                    "env": [
                        {"name": "TOTAL_CHAOS_DURATION", "value": duration_str},
                        {"name": "CPU_CORES", "value": "1"},
                        {"name": "CPU_LOAD", "value": "100"},
                        {"name": "TARGET_PODS", "value": "app=distllm-node"},
                    ],
                },
            },
        },
    )


def _build_chaos_mesh_base(scenario: ChaosScenario) -> dict[str, Any]:
    """Build the skeleton Chaos Mesh Schedule manifest."""
    return {
        "apiVersion": "chaos-mesh.org/v1alpha1",
        "kind": "Schedule",
        "metadata": {
            "name": f"{scenario.name}-schedule",
            "namespace": "distllm",
            "labels": {
                "chaos": scenario.name,
                "scenario-type": scenario.scenario_type.name.lower(),
            },
        },
        "spec": {
            "schedule": "0 0 * * *",
            "historyLimit": 1,
            "concurrencyPolicy": "Allow",
            "type": "",
            "experiment": {
                "target": None,
            },
        },
    }


def _apply_chaos_mesh_pod_kill(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a PodKill chaos to a Chaos Mesh manifest."""
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["type"] = "PodChaos"
    manifest["spec"]["experiment"]["target"] = {
        "action": "pod-kill",
        "mode": "one",
        "selector": {
            "namespaces": ["distllm"],
            "labelSelectors": {"app": "distllm-node"},
        },
        "duration": duration_str,
        "scheduler": {
            "cron": "@once",
        },
    }


def _apply_chaos_mesh_network_delay(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a NetworkChaos to a Chaos Mesh manifest."""
    faults = _find_faults_by_type(scenario, LatencyFault)
    delay_ms = str(int(faults[0].delay_ms)) if faults else "500"
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["type"] = "NetworkChaos"
    manifest["spec"]["experiment"]["target"] = {
        "action": "delay",
        "mode": "all",
        "selector": {
            "namespaces": ["distllm"],
            "labelSelectors": {"app": "distllm-node"},
        },
        "delay": {
            "latency": f"{delay_ms}ms",
            "correlation": "50",
            "jitter": "10ms",
        },
        "duration": duration_str,
        "scheduler": {
            "cron": "@once",
        },
    }


def _apply_chaos_mesh_cpu_stress(scenario: ChaosScenario, manifest: dict[str, Any]) -> None:
    """Add a StressChaos to a Chaos Mesh manifest."""
    duration_str = f"{int(scenario.duration)}s"

    manifest["spec"]["type"] = "StressChaos"
    manifest["spec"]["experiment"]["target"] = {
        "action": "cpu-stress",
        "mode": "one",
        "selector": {
            "namespaces": ["distllm"],
            "labelSelectors": {"app": "distllm-node"},
        },
        "stressors": {
            "cpu": {
                "workers": 1,
                "load": 100,
            },
        },
        "duration": duration_str,
        "scheduler": {
            "cron": "@once",
        },
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_method_name(client_call_details: Any) -> str:
    """Extract the gRPC method name from client_call_details.

    Handles both named-tuple style details (common with the standard
    gRPC Python interceptor API) and arbitrary objects with a ``method``
    attribute.
    """
    if hasattr(client_call_details, "method"):
        method: str = client_call_details.method
        # Strip the leading "/service/" prefix to get just the method name.
        if method.startswith("/"):
            parts = method.split("/")
            if len(parts) >= 3:
                return parts[2]
        return method
    return str(client_call_details)


def _raise_grpc_error(code: int, message: str) -> None:
    """Raise a gRPC-style RpcError with the given status code and message.

    When the ``grpc`` package is available, a real :class:`grpc.RpcError`
    is raised.  Otherwise a plain :class:`RuntimeError` with equivalent
    information is raised (for offline testing).
    """
    try:
        import grpc
    except ImportError:
        raise RuntimeError(f"gRPC error code={code}: {message}")

    # Build a minimal RpcError via the standard gRPC exception mechanism.
    context = grpc.ServicerContext.__new__(grpc.ServicerContext)  # type: ignore[call-overload]
    context._state = grpc._server._ServerRpcState()  # type: ignore[attr-defined]
    context._state.code = grpc.StatusCode(code)
    context._state.details = message
    context.abort(grpc.StatusCode(code), message)


def _find_faults_by_type(
    scenario: ChaosScenario,
    fault_type: type,
) -> list:
    """Return all faults in *scenario* that are instances of *fault_type*."""
    return [f for f in scenario.faults if isinstance(f, fault_type)]


__all__ = [
    "BlastRadiusControl",
    "ChaosScenario",
    "ChaosScenarioType",
    "ChaosTemplate",
    "ErrorFault",
    "FaultInjector",
    "FaultType",
    "LatencyFault",
    "MessageDropFault",
    "SteadyStateChecker",
    "SteadyStateMetrics",
]
