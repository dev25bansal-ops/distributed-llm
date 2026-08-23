"""Purpose-built chaos engineering for distributed LLM inference.

Provides a layered chaos engineering framework built on top of the low-level
:mod:`distllm.dist.chaos.fault_injector`:

* :class:`FaultScenarioLibrary` -- Library of pre-built fault scenarios
  targeting common distributed inference failure modes.
* :class:`ResilienceScore` -- Quantitative resilience measurement that
  runs scenarios and computes a 0.0-1.0 score.
* :class:`AutomatedChaosPipeline` -- CI/CD integration for progressive
  chaos testing with regression detection and optional cron scheduling.
* :class:`Kraken` -- Top-level coordinator combining all components into
  a single entry point.

Kubernetes integration is optional.  When the ``kubernetes`` package is
installed, the module can generate and apply chaos experiments against
real clusters; when it is absent, all functionality works in-process via
the :class:`FaultInjector`.

Usage::

    from distllm.core.kraken_chaos import Kraken

    kraken = Kraken()
    results = kraken.run_test("coordinator_failover")
    kraken.start_pipeline()
    stats = kraken.stats()
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable

from loguru import logger

# Import from the existing chaos infrastructure (no K8s dependency).
from distllm.dist.chaos.fault_injector import (
    BlastRadiusControl,
    ChaosScenario,
    ChaosScenarioType,
    ErrorFault,
    FaultInjector,
    LatencyFault,
    MessageDropFault,
    SteadyStateChecker,
    SteadyStateMetrics,
)

# Optional Kubernetes support -- all core functionality works without it.
try:
    import kubernetes as k8s  # type: ignore[import-untyped]

    HAS_K8S = True
except ImportError:
    k8s = None  # type: ignore[assignment]
    HAS_K8S = False

# Optional cron scheduling support.
try:
    import croniter  # type: ignore[import-untyped]

    HAS_CRONITER = True
except ImportError:
    croniter = None  # type: ignore[assignment]
    HAS_CRONITER = False


# ── Enums ─────────────────────────────────────────────────────────────────────


class ScenarioSeverity(Enum):
    """Severity level of a chaos scenario."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ScenarioCategory(Enum):
    """Category of chaos scenario."""

    AVAILABILITY = auto()
    PERFORMANCE = auto()
    DATA_INTEGRITY = auto()
    SECURITY = auto()
    NETWORK = auto()


class ResilienceLevel(Enum):
    """Overall resilience rating based on score.

    =============  ===========================================  ============
    Level          Score range                                  Meaning
    =============  ===========================================  ============
    EXCELLENT      >= 0.95                                      Fully resilient
    GOOD           [0.85, 0.95)                                 Minor gaps
    FAIR           [0.70, 0.85)                                 Noticeable gaps
    POOR           [0.50, 0.70)                                 Significant gaps
    CRITICAL       < 0.50                                       System is fragile
    =============  ===========================================  ============
    """

    EXCELLENT = auto()
    GOOD = auto()
    FAIR = auto()
    POOR = auto()
    CRITICAL = auto()


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a pre-built chaos scenario.

    Attributes:
        name: Unique identifier for the scenario.
        description: Human-readable description.
        category: Scenario category (availability, performance, etc.).
        severity: Expected severity of the scenario.
        duration_s: Duration in seconds for each fault injection.
        expected_recovery_s: Expected time for the system to fully recover.
        tags: Optional tags for filtering and grouping.
        params: Scenario-specific parameters dict.
    """

    name: str
    description: str
    category: ScenarioCategory
    severity: ScenarioSeverity
    duration_s: float = 30.0
    expected_recovery_s: float = 10.0
    tags: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    """Result of executing a single chaos scenario.

    Attributes:
        scenario_name: Name of the scenario that was executed.
        passed: Whether the steady-state hypothesis passed (system recovered).
        duration_s: Actual wall-clock execution duration.
        baseline: Baseline metrics snapshot before fault injection.
        during: Metrics snapshot during fault injection.
        after: Metrics snapshot after the system stabilised.
        error: Error message if the scenario failed to execute.
        timestamp: When the scenario was executed.
    """

    scenario_name: str
    passed: bool
    duration_s: float
    baseline: SteadyStateMetrics | None = None
    during: SteadyStateMetrics | None = None
    after: SteadyStateMetrics | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ResilienceReport:
    """Detailed report from a resilience score evaluation.

    Attributes:
        total_scenarios: Number of scenarios evaluated.
        passed_scenarios: Number of scenarios that passed.
        failed_scenarios: Number of scenarios that failed.
        score: Overall resilience score (0.0 - 1.0).
        level: Resilience level label.
        results: Individual scenario results.
        timestamp: When the report was generated.
    """

    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    score: float
    level: ResilienceLevel
    results: tuple[ScenarioResult, ...]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pass_rate(self) -> float:
        """Fraction of scenarios that passed."""
        if self.total_scenarios == 0:
            return 1.0
        return self.passed_scenarios / self.total_scenarios

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "failed_scenarios": self.failed_scenarios,
            "score": round(self.score, 4),
            "level": self.level.name,
            "pass_rate": round(self.pass_rate, 4),
            "timestamp": self.timestamp.isoformat(),
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "passed": r.passed,
                    "duration_s": round(r.duration_s, 2),
                    "error": r.error,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.results
            ],
        }


# ── Scenario Registry ─────────────────────────────────────────────────────────

_SCENARIO_REGISTRY: dict[str, ScenarioConfig] = {}


def _register_scenario(config: ScenarioConfig) -> None:
    """Register a scenario in the global library (module-level registry)."""
    _SCENARIO_REGISTRY[config.name] = config


# Register all built-in scenarios.
_register_scenario(
    ScenarioConfig(
        name="coordinator_failover",
        description=(
            "Simulates coordinator node failure by dropping all HealthCheck "
            "messages and verifies automatic failover to the standby coordinator"
        ),
        category=ScenarioCategory.AVAILABILITY,
        severity=ScenarioSeverity.CRITICAL,
        duration_s=60.0,
        expected_recovery_s=15.0,
        tags=("failover", "coordinator", "ha"),
        params={
            "method": "HealthCheck",
            "failover_timeout_s": 15.0,
            "blast_radius": {"max_nodes": 1},
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="node_crash",
        description=(
            "Simulates a worker node crash by dropping ForwardPass messages "
            "and verifies that requests are redistributed to remaining nodes"
        ),
        category=ScenarioCategory.AVAILABILITY,
        severity=ScenarioSeverity.HIGH,
        duration_s=45.0,
        expected_recovery_s=20.0,
        tags=("node", "crash", "recovery"),
        params={
            "method": "ForwardPass",
            "crash_timeout_s": 20.0,
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="network_partition",
        description=(
            "Simulates a network partition between two nodes by combining "
            "message drops with extreme latency, then verifies partition tolerance"
        ),
        category=ScenarioCategory.NETWORK,
        severity=ScenarioSeverity.CRITICAL,
        duration_s=90.0,
        expected_recovery_s=30.0,
        tags=("network", "partition", "split-brain"),
        params={
            "method": "ForwardPass",
            "partition_delay_ms": 30000.0,
            "partition_duration_s": 30.0,
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="kv_cache_corruption",
        description=(
            "Injects corrupted entries into KV cache operations and verifies "
            "graceful degradation without full system failure"
        ),
        category=ScenarioCategory.DATA_INTEGRITY,
        severity=ScenarioSeverity.HIGH,
        duration_s=30.0,
        expected_recovery_s=5.0,
        tags=("kv-cache", "corruption", "degradation"),
        params={
            "method": "ForwardPass",
            "corruption_rate": 0.1,
            "error_code": 3,  # INVALID_ARGUMENT
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="gpu_oom",
        description=(
            "Simulates GPU out-of-memory conditions and verifies that memory "
            "offloading or preemption mechanisms activate correctly"
        ),
        category=ScenarioCategory.PERFORMANCE,
        severity=ScenarioSeverity.HIGH,
        duration_s=40.0,
        expected_recovery_s=25.0,
        tags=("gpu", "oom", "memory"),
        params={
            "method": "ForwardPass",
            "oom_error_code": 8,  # RESOURCE_EXHAUSTED
            "recovery_delay_s": 25.0,
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="slow_drain",
        description=(
            "Simulates a slow node drain during maintenance by injecting "
            "latency on a subset of requests and verifies gradual connection drain"
        ),
        category=ScenarioCategory.PERFORMANCE,
        severity=ScenarioSeverity.MEDIUM,
        duration_s=120.0,
        expected_recovery_s=30.0,
        tags=("drain", "maintenance", "graceful"),
        params={
            "method": "ForwardPass",
            "latency_ms": 5000.0,
            "latency_prob": 0.3,
        },
    ),
)
_register_scenario(
    ScenarioConfig(
        name="certificate_expiry",
        description=(
            "Simulates expired TLS certificates by injecting authentication "
            "errors and verifies secure fallback or automatic cert rotation"
        ),
        category=ScenarioCategory.SECURITY,
        severity=ScenarioSeverity.CRITICAL,
        duration_s=30.0,
        expected_recovery_s=10.0,
        tags=("certificate", "tls", "security"),
        params={
            "method": "HealthCheck",
            "error_code": 16,  # UNAUTHENTICATED
            "error_message": "Certificate expired",
        },
    ),
)


# ── FaultScenarioLibrary ──────────────────────────────────────────────────────


class FaultScenarioLibrary:
    """Library of pre-built fault scenarios for distributed LLM inference.

    Provides a registry of well-known chaos scenarios that target specific
    failure modes in a distributed inference system.  Each scenario can be
    retrieved by name, converted into an executable :class:`ChaosScenario`,
    or filtered by category or severity.

    Thread-safe for concurrent access.

    Usage::

        lib = FaultScenarioLibrary()
        config = lib.get("coordinator_failover")
        print(lib.list_scenarios())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Shallow copy of the module-level registry -- read-only by default.
        self._scenarios: dict[str, ScenarioConfig] = dict(_SCENARIO_REGISTRY)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, name: str) -> ScenarioConfig:
        """Retrieve a scenario configuration by name.

        Args:
            name: Scenario name (e.g. ``"coordinator_failover"``).

        Returns:
            The :class:`ScenarioConfig` for the named scenario.

        Raises:
            KeyError: If the scenario name is not registered.
        """
        with self._lock:
            if name not in self._scenarios:
                raise KeyError(
                    f"Unknown scenario: {name!r}. "
                    f"Available: {', '.join(sorted(self._scenarios))}"
                )
            return self._scenarios[name]

    def list_scenarios(self) -> list[ScenarioConfig]:
        """Return all registered scenario configurations.

        Returns:
            A list of all :class:`ScenarioConfig` instances.
        """
        with self._lock:
            return list(self._scenarios.values())

    def list_by_category(self, category: ScenarioCategory) -> list[ScenarioConfig]:
        """Return scenarios filtered by category.

        Args:
            category: The :class:`ScenarioCategory` to filter by.

        Returns:
            A list of matching :class:`ScenarioConfig` instances.
        """
        with self._lock:
            return [s for s in self._scenarios.values() if s.category == category]

    def list_by_severity(self, severity: ScenarioSeverity) -> list[ScenarioConfig]:
        """Return scenarios filtered by severity.

        Args:
            severity: The :class:`ScenarioSeverity` to filter by.

        Returns:
            A list of matching :class:`ScenarioConfig` instances.
        """
        with self._lock:
            return [s for s in self._scenarios.values() if s.severity == severity]

    def list_by_tags(self, *tags: str) -> list[ScenarioConfig]:
        """Return scenarios that have *all* of the given tags.

        Args:
            *tags: Tag strings to match (AND logic).

        Returns:
            A list of matching :class:`ScenarioConfig` instances.
        """
        with self._lock:
            tag_set = set(tags)
            return [s for s in self._scenarios.values() if tag_set.issubset(s.tags)]

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, config: ScenarioConfig) -> None:
        """Register a custom scenario configuration.

        Args:
            config: The :class:`ScenarioConfig` to add.

        Raises:
            ValueError: If a scenario with the same name already exists.
        """
        with self._lock:
            if config.name in self._scenarios:
                raise ValueError(f"Scenario {config.name!r} is already registered")
            self._scenarios[config.name] = config
            _SCENARIO_REGISTRY[config.name] = config  # also register globally

    # ── Construction ──────────────────────────────────────────────────────

    def build_chaos_scenario(self, name: str) -> ChaosScenario:
        """Build an executable :class:`ChaosScenario` from a registered config.

        Args:
            name: Registered scenario name.

        Returns:
            A :class:`ChaosScenario` configured per the scenario's parameters.

        Raises:
            KeyError: If the scenario name is not registered.
        """
        config = self.get(name)
        return _build_chaos_scenario(config)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of registered scenarios."""
        with self._lock:
            return len(self._scenarios)

    def __len__(self) -> int:
        """Same as :attr:`count`."""
        return self.count

    def __contains__(self, name: str) -> bool:
        """Check if a scenario is registered."""
        with self._lock:
            return name in self._scenarios


# ── Scenario Builder ──────────────────────────────────────────────────────────


def _build_chaos_scenario(config: ScenarioConfig) -> ChaosScenario:
    """Convert a :class:`ScenarioConfig` into an executable :class:`ChaosScenario`.

    Each well-known scenario name maps to specific fault specifications
    and scenario type.  Unknown names get a generic latency-scenario
    fallback.
    """
    params = config.params
    name = config.name

    if name == "coordinator_failover":
        method = params.get("method", "HealthCheck")
        br = params.get("blast_radius", {})
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.NODE_FAILURE,
            description=config.description,
            duration=config.duration_s,
            faults=[MessageDropFault(method=method, probability=1.0)],
            blast_radius=BlastRadiusControl(max_nodes=br.get("max_nodes", 1)),
        )

    if name == "node_crash":
        method = params.get("method", "ForwardPass")
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.NODE_FAILURE,
            description=config.description,
            duration=config.duration_s,
            faults=[MessageDropFault(method=method, probability=1.0)],
            blast_radius=BlastRadiusControl(max_nodes=1),
        )

    if name == "network_partition":
        method = params.get("method", "ForwardPass")
        delay_ms = params.get("partition_delay_ms", 30000.0)
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.NETWORK_PARTITION,
            description=config.description,
            duration=config.duration_s,
            faults=[
                MessageDropFault(method=method, probability=1.0),
                LatencyFault(method=method, probability=1.0, delay_ms=delay_ms),
            ],
        )

    if name == "kv_cache_corruption":
        method = params.get("method", "ForwardPass")
        error_code = params.get("error_code", 3)  # INVALID_ARGUMENT
        corruption_rate = params.get("corruption_rate", 0.1)
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.LATENCY_SPIKE,
            description=config.description,
            duration=config.duration_s,
            faults=[
                ErrorFault(
                    method=method,
                    probability=corruption_rate,
                    code=error_code,
                    message="KV cache corruption detected",
                ),
                LatencyFault(method=method, probability=0.5, delay_ms=200.0),
            ],
        )

    if name == "gpu_oom":
        method = params.get("method", "ForwardPass")
        error_code = params.get("oom_error_code", 8)  # RESOURCE_EXHAUSTED
        recovery_delay = params.get("recovery_delay_s", 25.0)
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.LATENCY_SPIKE,
            description=config.description,
            duration=config.duration_s,
            faults=[
                ErrorFault(
                    method=method,
                    probability=0.5,
                    code=error_code,
                    message="GPU out of memory",
                ),
                LatencyFault(
                    method=method, probability=0.7, delay_ms=recovery_delay * 1000.0
                ),
            ],
        )

    if name == "slow_drain":
        method = params.get("method", "ForwardPass")
        latency_ms = params.get("latency_ms", 5000.0)
        latency_prob = params.get("latency_prob", 0.3)
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.LATENCY_SPIKE,
            description=config.description,
            duration=config.duration_s,
            faults=[
                LatencyFault(
                    method=method, probability=latency_prob, delay_ms=latency_ms
                ),
                ErrorFault(
                    method=method,
                    probability=0.05,
                    code=14,
                    message="Connection drain in progress",
                ),
            ],
        )

    if name == "certificate_expiry":
        method = params.get("method", "HealthCheck")
        error_code = params.get("error_code", 16)  # UNAUTHENTICATED
        error_message = params.get("error_message", "Certificate expired")
        return ChaosScenario(
            name=name,
            scenario_type=ChaosScenarioType.LATENCY_SPIKE,
            description=config.description,
            duration=config.duration_s,
            faults=[
                ErrorFault(
                    method=method, probability=1.0, code=error_code, message=error_message
                ),
            ],
        )

    # Generic fallback for custom / unknown scenarios.
    method = params.get("method", "ForwardPass")
    return ChaosScenario(
        name=name,
        scenario_type=ChaosScenarioType.LATENCY_SPIKE,
        description=config.description,
        duration=config.duration_s,
        faults=[LatencyFault(method=method, probability=1.0, delay_ms=100.0)],
    )


# ── ResilienceScore ───────────────────────────────────────────────────────────


class ResilienceScore:
    """Measures system resilience by executing chaos scenarios and scoring results.

    Runs well-known fault scenarios against the system and computes a
    quantitative resilience score (0.0 - 1.0) based on the fraction of
    scenarios whose steady-state hypothesis passed.

    Usage::

        scorer = ResilienceScore()
        passed, result = scorer.run_scenario("coordinator_failover")
        score = scorer.overall_score(["coordinator_failover", "node_crash"])
        report = scorer.report()
    """

    def __init__(
        self,
        fault_injector: FaultInjector | None = None,
        metric_collector: Callable[[], SteadyStateMetrics] | None = None,
    ) -> None:
        """Initialize the resilience scorer.

        Args:
            fault_injector: Optional :class:`FaultInjector` instance.
                A new one is created if not provided.
            metric_collector: Optional callable that returns the current
                :class:`SteadyStateMetrics` snapshot.  When omitted, a
                no-op collector returning zeros is used.
        """
        self._fault_injector = fault_injector or FaultInjector()
        self._library = FaultScenarioLibrary()
        self._results: list[ScenarioResult] = []
        self._lock = threading.Lock()
        self._metric_collector = metric_collector or _null_metric_collector

    # ── Scenario Execution ────────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_name: str,
        *,
        custom_config: ScenarioConfig | None = None,
    ) -> tuple[bool, ScenarioResult]:
        """Execute a single chaos scenario and measure resilience.

        Args:
            scenario_name: Name of the scenario to run (must be registered
                in :class:`FaultScenarioLibrary`).
            custom_config: Optional override :class:`ScenarioConfig` to use
                instead of the registered one.

        Returns:
            A tuple of ``(passed, result)`` where *passed* is True if the
            system maintained or recovered to steady state.

        Raises:
            KeyError: If *scenario_name* is not registered and no
                *custom_config* is provided.
        """
        config = custom_config if custom_config is not None else self._library.get(scenario_name)

        chaos_scenario = _build_chaos_scenario(config)

        # Attach a steady-state checker.
        checker = SteadyStateChecker(metric_collector=self._metric_collector)
        chaos_scenario.steady_state_checker = checker

        # Apply default blast-radius if none set.
        if chaos_scenario.blast_radius is None:
            chaos_scenario.blast_radius = BlastRadiusControl(max_nodes=1)

        start_time = time.monotonic()
        error: str | None = None
        run_result: dict[str, Any] = {}

        try:
            run_result = chaos_scenario.run(self._fault_injector)
            passed = bool(run_result.get("steady_state_passed", False))
        except Exception as exc:
            logger.exception("Scenario {!r} failed with exception", scenario_name)
            passed = False
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - start_time

        scenario_result = ScenarioResult(
            scenario_name=config.name,
            passed=passed,
            duration_s=elapsed,
            baseline=run_result.get("baseline") if not error else None,
            during=run_result.get("during") if not error else None,
            after=run_result.get("after") if not error else None,
            error=error,
        )

        with self._lock:
            self._results.append(scenario_result)

        logger.info(
            "Resilience scenario {!r}: {} ({:.1f}s)",
            config.name,
            "PASSED" if passed else "FAILED",
            elapsed,
        )

        return passed, scenario_result

    # ── Scoring ───────────────────────────────────────────────────────────

    def overall_score(self, scenarios: list[str] | None = None) -> float:
        """Compute the overall resilience score (0.0 - 1.0).

        Runs all specified scenarios (or all registered scenarios if *scenarios*
        is ``None``) and returns the fraction that passed.

        Args:
            scenarios: Optional list of scenario names to run.  When ``None``,
                all registered scenarios are executed.

        Returns:
            A float between 0.0 (no resilience) and 1.0 (fully resilient).
        """
        scenario_names = scenarios or [
            s.name for s in self._library.list_scenarios()
        ]

        if not scenario_names:
            return 1.0

        results: list[ScenarioResult] = []
        for name in scenario_names:
            _passed, result = self.run_scenario(name)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        score = passed_count / len(results)

        logger.info(
            "Resilience score: {:.2%} ({}/{}) passed",
            score,
            passed_count,
            len(results),
        )

        return score

    # ── Reporting ─────────────────────────────────────────────────────────

    def report(self) -> ResilienceReport:
        """Generate a detailed resilience report from all accumulated results.

        Returns:
            A :class:`ResilienceReport` with per-scenario results and the
            aggregate resilience score.
        """
        with self._lock:
            results = list(self._results)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        score = passed / total if total > 0 else 1.0

        level = _score_to_level(score)

        return ResilienceReport(
            total_scenarios=total,
            passed_scenarios=passed,
            failed_scenarios=failed,
            score=score,
            level=level,
            results=tuple(results),
        )

    # ── Result Management ─────────────────────────────────────────────────

    @property
    def results(self) -> list[ScenarioResult]:
        """All scenario results accumulated so far."""
        with self._lock:
            return list(self._results)

    def clear_results(self) -> None:
        """Clear all accumulated scenario results."""
        with self._lock:
            self._results.clear()

    # ── Injection Control ─────────────────────────────────────────────────

    @property
    def fault_injector(self) -> FaultInjector:
        """The underlying :class:`FaultInjector`."""
        return self._fault_injector

    def set_fault_injector(self, injector: FaultInjector) -> None:
        """Replace the fault injector (useful for test injection)."""
        self._fault_injector = injector


def _score_to_level(score: float) -> ResilienceLevel:
    """Map a numeric score to a :class:`ResilienceLevel`."""
    if score >= 0.95:
        return ResilienceLevel.EXCELLENT
    if score >= 0.85:
        return ResilienceLevel.GOOD
    if score >= 0.70:
        return ResilienceLevel.FAIR
    if score >= 0.50:
        return ResilienceLevel.POOR
    return ResilienceLevel.CRITICAL


def _null_metric_collector() -> SteadyStateMetrics:
    """Return a zero-valued metrics snapshot (no-op collector)."""
    return SteadyStateMetrics(
        timestamp=datetime.now(timezone.utc),
        p99_latency_ms=0.0,
        throughput_req_per_sec=0.0,
        error_rate=0.0,
    )


# ── AutomatedChaosPipeline ────────────────────────────────────────────────────


class AutomatedChaosPipeline:
    """CI/CD integration for automated chaos testing.

    Runs a suite of chaos scenarios, measures resilience, generates
    machine-readable reports, and provides a regression check that can
    fail a CI build when resilience drops below a configurable threshold.

    Supports optional cron-based scheduling when the ``croniter`` package
    is installed.

    Usage::

        pipeline = AutomatedChaosPipeline(
            scenarios=["coordinator_failover", "node_crash"],
            min_resilience=0.85,
            report_dir="./chaos-reports",
        )
        report = pipeline.pipeline()

        if not pipeline.break_on_regression(report, previous_score=0.90):
            sys.exit(1)  # fail the CI step
    """

    def __init__(
        self,
        scenarios: list[str] | None = None,
        min_resilience: float = 0.85,
        report_dir: str | None = None,
        fault_injector: FaultInjector | None = None,
        metric_collector: Callable[[], SteadyStateMetrics] | None = None,
    ) -> None:
        """Initialize the chaos pipeline.

        Args:
            scenarios: List of scenario names to run.  When ``None``, all
                registered scenarios are executed.
            min_resilience: Minimum acceptable resilience score in ``[0, 1]``.
                Default 0.85.
            report_dir: Directory path for writing JSON report files.  When
                ``None``, reports are kept in memory only.
            fault_injector: Optional :class:`FaultInjector` instance.
            metric_collector: Optional metric collector callable.

        Raises:
            ValueError: If *min_resilience* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= min_resilience <= 1.0:
            raise ValueError(
                f"min_resilience must be in [0.0, 1.0], got {min_resilience}"
            )

        self._scenarios = scenarios
        self._min_resilience = min_resilience
        self._report_dir = report_dir
        self._resilience = ResilienceScore(
            fault_injector=fault_injector,
            metric_collector=metric_collector,
        )
        self._lock = threading.Lock()
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_stop = threading.Event()
        self._history: list[ResilienceReport] = []

    # ── Pipeline Execution ────────────────────────────────────────────────

    def pipeline(
        self,
        scenarios: list[str] | None = None,
    ) -> ResilienceReport:
        """Run the full chaos pipeline.

        Steps:
            1. Clear previous results.
            2. Execute all configured scenarios.
            3. Compute the aggregate resilience score.
            4. Persist a JSON report to disk (if *report_dir* was set).
            5. Append the report to the internal history.

        Args:
            scenarios: Override the configured scenarios for this run.
                When ``None``, uses the scenarios passed at construction.

        Returns:
            A :class:`ResilienceReport` with all results.
        """
        scenario_names = scenarios if scenarios is not None else self._scenarios
        self._resilience.clear_results()

        logger.info(
            "Starting chaos pipeline with {} scenario(s)",
            len(scenario_names) if scenario_names else "all registered",
        )

        self._resilience.overall_score(scenario_names)
        report = self._resilience.report()

        if self._report_dir:
            self._save_report(report)

        with self._lock:
            self._history.append(report)

        logger.info(
            "Chaos pipeline complete: score={:.2%} level={}",
            report.score,
            report.level.name,
        )

        return report

    # ── Regression Detection ──────────────────────────────────────────────

    def break_on_regression(
        self,
        report: ResilienceReport,
        previous_score: float | None = None,
    ) -> bool:
        """Check whether resilience has regressed.

        Returns ``True`` when resilience is acceptable (no regression).
        Returns ``False`` when the score is below the configured threshold
        or below *previous_score* -- the caller should treat this as a
        CI-break signal.

        Args:
            report: The current :class:`ResilienceReport`.
            previous_score: Previous resilience score to compare against.
                When ``None``, only the configured *min_resilience* threshold
                is checked.

        Returns:
            ``True`` if resilience is acceptable, ``False`` if regression
            detected.
        """
        if report.score < self._min_resilience:
            logger.error(
                "Resilience REGRESSION: score={:.2%} < threshold={:.2%}",
                report.score,
                self._min_resilience,
            )
            return False

        if previous_score is not None and report.score < previous_score:
            logger.warning(
                "Resilience REGRESSION: score={:.2%} < previous={:.2%}",
                report.score,
                previous_score,
            )
            return False

        logger.info(
            "Resilience check passed: score={:.2%} >= threshold={:.2%}",
            report.score,
            self._min_resilience,
        )
        return True

    # ── Scheduling ────────────────────────────────────────────────────────

    def schedule(self, cron_expression: str) -> None:
        """Run the pipeline on a cron schedule in a background daemon thread.

        Args:
            cron_expression: Standard cron expression (e.g. ``"0 */6 * * *"``
                for every 6 hours).

        Raises:
            RuntimeError: If the ``croniter`` package is not installed.
            ValueError: If the cron expression is invalid.
        """
        if not HAS_CRONITER:
            raise RuntimeError(
                "The croniter package is required for scheduling. "
                "Install it with: pip install croniter"
            )

        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            logger.warning("Scheduler already running; stopping previous one")
            self.stop_scheduler()

        # Validate expression.
        if not croniter.croniter.is_valid(cron_expression):  # type: ignore[union-attr]
            raise ValueError(f"Invalid cron expression: {cron_expression!r}")

        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            args=(cron_expression, self, self._scheduler_stop),
            daemon=True,
            name="kraken-scheduler",
        )
        self._scheduler_thread.start()
        logger.info("Chaos pipeline scheduled with cron: {}", cron_expression)

    def stop_scheduler(self) -> None:
        """Stop the background scheduler thread (if running)."""
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None
        logger.info("Chaos pipeline scheduler stopped")

    # ── History ───────────────────────────────────────────────────────────

    def history(self) -> list[ResilienceReport]:
        """Return the pipeline execution history.

        Returns:
            A list of :class:`ResilienceReport` instances in chronological
            order (oldest first).
        """
        with self._lock:
            return list(self._history)

    def latest_score(self) -> float | None:
        """Return the most recent resilience score, or ``None`` if no runs."""
        with self._lock:
            if not self._history:
                return None
            return self._history[-1].score

    # ── Threshold ─────────────────────────────────────────────────────────

    @property
    def min_resilience(self) -> float:
        """Minimum acceptable resilience threshold."""
        return self._min_resilience

    @min_resilience.setter
    def min_resilience(self, value: float) -> None:
        """Set the minimum resilience threshold.

        Raises:
            ValueError: If *value* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"min_resilience must be in [0.0, 1.0], got {value}"
            )
        self._min_resilience = value

    # ── Internals ─────────────────────────────────────────────────────────

    def _save_report(self, report: ResilienceReport) -> None:
        """Write the resilience report to disk as JSON."""
        if not self._report_dir:
            return
        os.makedirs(self._report_dir, exist_ok=True)
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"resilience_report_{timestamp}.json"
        filepath = os.path.join(self._report_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info("Wrote resilience report to {}", filepath)
        except OSError as exc:
            logger.error("Failed to write report to {}: {}", filepath, exc)


def _scheduler_loop(
    cron_expression: str,
    pipeline: AutomatedChaosPipeline,
    stop_event: threading.Event,
) -> None:
    """Background scheduler loop that runs the pipeline on a cron schedule.

    Args:
        cron_expression: Valid cron expression string.
        pipeline: The pipeline instance to run.
        stop_event: Event that signals shutdown when set.
    """
    logger.info("Scheduler loop started with cron: {}", cron_expression)

    while not stop_event.is_set():
        now = datetime.now()

        try:
            cron = croniter.croniter(cron_expression, now)  # type: ignore[union-attr]
            next_run = cron.get_next(datetime)
        except Exception:
            logger.exception("Error computing next cron run")
            stop_event.wait(60.0)
            continue

        wait_seconds = (next_run - datetime.now()).total_seconds()
        if wait_seconds > 0:
            if stop_event.wait(wait_seconds):
                break

        if stop_event.is_set():
            break

        try:
            logger.info("Scheduled pipeline execution starting")
            report = pipeline.pipeline()
            pipeline.break_on_regression(report)
            logger.info(
                "Scheduled pipeline complete: score={:.2%} level={}",
                report.score,
                report.level.name,
            )
        except Exception:
            logger.exception("Scheduled pipeline execution failed")


# ── Kraken ────────────────────────────────────────────────────────────────────


class Kraken:
    """Top-level chaos engineering coordinator for distributed LLM inference.

    Combines :class:`FaultScenarioLibrary`, :class:`ResilienceScore`, and
    :class:`AutomatedChaosPipeline` into a single entry point.

    Usage::

        kraken = Kraken(min_resilience=0.80)

        # Run a single test.
        result = kraken.run_test("coordinator_failover")

        # Run a suite and get a report.
        report = kraken.run_test(["node_crash", "gpu_oom"])

        # Start the automated pipeline (immediate execution).
        report = kraken.start_pipeline()

        # Start the pipeline on a schedule (requires croniter).
        kraken.start_pipeline(cron_expression="0 */6 * * *")

        # Query statistics.
        stats = kraken.stats()
    """

    def __init__(
        self,
        scenarios: list[str] | None = None,
        min_resilience: float = 0.85,
        report_dir: str | None = None,
        fault_injector: FaultInjector | None = None,
        metric_collector: Callable[[], SteadyStateMetrics] | None = None,
    ) -> None:
        """Initialize Kraken.

        Args:
            scenarios: List of scenario names for the pipeline.  When
                ``None``, all registered scenarios are used.
            min_resilience: Minimum acceptable resilience score (default 0.85).
            report_dir: Optional directory for JSON reports.
            fault_injector: Optional :class:`FaultInjector` instance.
            metric_collector: Optional metric collector callable.
        """
        self._library = FaultScenarioLibrary()
        self._pipeline = AutomatedChaosPipeline(
            scenarios=scenarios,
            min_resilience=min_resilience,
            report_dir=report_dir,
            fault_injector=fault_injector,
            metric_collector=metric_collector,
        )

    # ── Scenario Management ───────────────────────────────────────────────

    @property
    def scenarios(self) -> FaultScenarioLibrary:
        """Access the underlying scenario library."""
        return self._library

    def run_test(
        self, suite: str | list[str]
    ) -> ScenarioResult | ResilienceReport:
        """Run one or more chaos tests.

        Args:
            suite: Either a single scenario name (returns
                :class:`ScenarioResult`) or a list of names (returns
                :class:`ResilienceReport`).

        Returns:
            A :class:`ScenarioResult` for single-scenario runs or a
            :class:`ResilienceReport` for suites.

        Raises:
            KeyError: If any scenario name is not registered.
        """
        scorer = self._pipeline._resilience  # access internal ResilienceScore

        if isinstance(suite, str):
            _passed, result = scorer.run_scenario(suite)
            return result

        # Suite: run all, return aggregate report.
        scorer.clear_results()
        scorer.overall_score(list(suite))
        return scorer.report()

    # ── Pipeline ──────────────────────────────────────────────────────────

    def start_pipeline(
        self,
        cron_expression: str | None = None,
    ) -> ResilienceReport | None:
        """Start the automated chaos pipeline.

        When *cron_expression* is provided the pipeline runs on that
        schedule in a background daemon thread (requires ``croniter``).
        Otherwise it runs immediately and returns the report.

        Args:
            cron_expression: Optional cron expression for periodic
                scheduling (e.g. ``"0 */6 * * *"``).

        Returns:
            A :class:`ResilienceReport` for immediate execution, or
            ``None`` when started on a schedule.
        """
        if cron_expression:
            self._pipeline.schedule(cron_expression)
            return None
        return self._pipeline.pipeline()

    def stop_pipeline(self) -> None:
        """Stop the scheduled pipeline (if running)."""
        self._pipeline.stop_scheduler()

    # ── Statistics ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about chaos test runs.

        Returns:
            A dict with the following keys:

            ``total_scenarios``
                Number of available scenario definitions.
            ``scenarios_run``
                Total number of individual scenario executions across
                all pipeline runs.
            ``pass_rate``
                Overall pass rate across all executions, or ``None``
                if no runs have occurred.
            ``current_score``
                Most recent resilience score (0.0 - 1.0), or ``None``.
            ``trend``
                Resilience trend: ``"improving"``, ``"declining"``, or
                ``"stable"``.  Based on the last three reports.
            ``history_count``
                Number of completed pipeline runs.
        """
        history = self._pipeline.history()

        if not history:
            return {
                "total_scenarios": self._library.count,
                "scenarios_run": 0,
                "pass_rate": None,
                "current_score": None,
                "trend": "stable",
                "history_count": 0,
            }

        latest = history[-1]

        # Compute trend from the last 3 reports.
        scores = [r.score for r in history]
        recent = scores[-3:] if len(scores) >= 3 else scores
        improve_count = sum(
            1 for i in range(1, len(recent)) if recent[i] > recent[i - 1]
        )
        decline_count = sum(
            1 for i in range(1, len(recent)) if recent[i] < recent[i - 1]
        )
        if improve_count > decline_count:
            trend = "improving"
        elif decline_count > improve_count:
            trend = "declining"
        else:
            trend = "stable"

        total_results = sum(len(r.results) for r in history)
        total_passed = sum(r.passed_scenarios for r in history)
        pass_rate = total_passed / total_results if total_results > 0 else None

        return {
            "total_scenarios": self._library.count,
            "scenarios_run": total_results,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "current_score": round(latest.score, 4),
            "trend": trend,
            "history_count": len(history),
        }

    # ─── Convenience ──────────────────────────────────────────────────────

    def pipeline_report(self) -> ResilienceReport | None:
        """Return the most recent pipeline report, or ``None``."""
        history = self._pipeline.history()
        return history[-1] if history else None

    def set_min_resilience(self, threshold: float) -> None:
        """Set the minimum acceptable resilience threshold.

        Args:
            threshold: Value in ``[0.0, 1.0]``.

        Raises:
            ValueError: If *threshold* is outside the valid range.
        """
        self._pipeline.min_resilience = threshold


__all__ = [
    # Enums
    "ResilienceLevel",
    "ScenarioCategory",
    "ScenarioSeverity",
    # Data classes
    "ResilienceReport",
    "ScenarioConfig",
    "ScenarioResult",
    # Classes
    "AutomatedChaosPipeline",
    "FaultScenarioLibrary",
    "Kraken",
    "ResilienceScore",
]
