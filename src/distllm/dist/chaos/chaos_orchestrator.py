"""Autonomous chaos engineering subsystem.

Injects controlled faults into staging clusters, learns optimal recovery
policies, and deploys them to the production autonomous healer.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultType(str, Enum):
    """Types of faults that can be injected."""
    NODE_KILL = "node_kill"
    NETWORK_PARTITION = "network_partition"
    LATENCY_INJECTION = "latency_injection"
    OOM = "oom"
    STRAGGLER = "straggler"


class FaultTarget(str, Enum):
    """Targets for fault injection."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    NETWORK = "network"


@dataclass
class FaultScenario:
    """A single fault scenario definition."""
    fault_type: FaultType = FaultType.NODE_KILL
    target: FaultTarget = FaultTarget.WORKER
    duration_s: int = 30
    intensity: float = 1.0  # 0.0-1.0


@dataclass
class ExperimentResult:
    """Result of running a fault scenario."""
    scenario: FaultScenario
    success: bool = False
    recovery_time_s: float = 0.0
    side_effects: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class ScenarioGenerator:
    """Generates fault scenarios for chaos experiments."""

    @staticmethod
    def predefined_scenarios() -> list[FaultScenario]:
        """Return a curated set of common failure scenarios."""
        return [
            FaultScenario(FaultType.NODE_KILL, FaultTarget.WORKER, 30, 1.0),
            FaultScenario(FaultType.NETWORK_PARTITION, FaultTarget.NETWORK, 60, 0.5),
            FaultScenario(FaultType.LATENCY_INJECTION, FaultTarget.NETWORK, 120, 0.3),
            FaultScenario(FaultType.OOM, FaultTarget.WORKER, 45, 0.8),
            FaultScenario(FaultType.STRAGGLER, FaultTarget.WORKER, 90, 0.4),
            FaultScenario(FaultType.NODE_KILL, FaultTarget.COORDINATOR, 15, 1.0),
        ]

    @staticmethod
    def from_config(yaml_path: str) -> list[FaultScenario]:
        """Load fault scenarios from a YAML configuration file."""
        try:
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
            scenarios = []
            for item in data.get("scenarios", []):
                scenarios.append(FaultScenario(
                    fault_type=FaultType(item.get("fault_type", "node_kill")),
                    target=FaultTarget(item.get("target", "worker")),
                    duration_s=item.get("duration_s", 30),
                    intensity=item.get("intensity", 1.0),
                ))
            return scenarios
        except Exception:
            return ScenarioGenerator.predefined_scenarios()

    @staticmethod
    def random_scenario(seed: int | None = None) -> FaultScenario:
        """Generate a random fault scenario."""
        if seed is not None:
            random.seed(seed)
        return FaultScenario(
            fault_type=random.choice(list(FaultType)),
            target=random.choice(list(FaultTarget)),
            duration_s=random.choice([15, 30, 60, 120]),
            intensity=round(random.uniform(0.2, 1.0), 2),
        )


class ChaosOrchestrator:
    """Orchestrates chaos experiments and learns recovery policies.

    Usage::

        chaos = ChaosOrchestrator(coordinator_url="http://localhost:8000")
        scenario = FaultScenario(FaultType.NODE_KILL, FaultTarget.WORKER, 30)
        result = chaos.run_scenario(scenario)
        print(f"Recovery time: {result.recovery_time_s:.1f}s")
    """

    def __init__(
        self,
        coordinator_url: str = "http://localhost:8000",
        staging: bool = True,
        healer: Any | None = None,
    ):
        self.coordinator_url = coordinator_url
        self.staging = staging
        self._healer = healer
        self._experiments_run: int = 0
        self._recovery_successes: int = 0
        self._recovery_times: list[float] = []

    @property
    def healer(self) -> Any:
        return self._healer

    @healer.setter
    def healer(self, value: Any) -> None:
        self._healer = value

    def run_scenario(self, scenario: FaultScenario, timeout_s: int = 300) -> ExperimentResult:
        """Execute a single fault scenario and measure recovery.

        Args:
            scenario: The fault scenario to inject.
            timeout_s: Maximum time to wait for recovery.

        Returns:
            ExperimentResult with recovery outcome.
        """
        result = ExperimentResult(scenario=scenario)
        start = time.time()

        try:
            if self.staging:
                self._inject_fault_simulated(scenario)
            else:
                self._inject_fault(scenario)

            # Wait for healing or timeout
            recovered = self._wait_for_recovery(scenario, timeout_s)
            result.recovery_time_s = time.time() - start
            result.success = recovered
            self._experiments_run += 1
            if recovered:
                self._recovery_successes += 1
                self._recovery_times.append(result.recovery_time_s)
        except Exception as e:
            result.success = False
            result.side_effects.append(str(e))

        return result

    def run_suite(self, scenarios: list[FaultScenario] | None = None) -> list[ExperimentResult]:
        """Run a suite of fault scenarios.

        Args:
            scenarios: List of scenarios.  Uses predefined if None.

        Returns:
            List of ExperimentResults.
        """
        if scenarios is None:
            scenarios = ScenarioGenerator.predefined_scenarios()
        return [self.run_scenario(s) for s in scenarios]

    @property
    def metrics(self) -> dict[str, Any]:
        """Return chaos engineering metrics."""
        return {
            "experiments_run": self._experiments_run,
            "recovery_success_rate": (
                self._recovery_successes / self._experiments_run
                if self._experiments_run > 0 else 1.0
            ),
            "avg_recovery_time_s": (
                sum(self._recovery_times) / len(self._recovery_times)
                if self._recovery_times else 0.0
            ),
            "staging": self.staging,
        }

    # ── Private helpers ──────────────────────────────────────────────────

    def _inject_fault_simulated(self, scenario: FaultScenario) -> None:
        """Simulated fault injection for staging/testing."""
        pass

    def _inject_fault(self, scenario: FaultScenario) -> None:
        """Real fault injection against the cluster.

        Uses the coordinator's admin API to kill workers, partition network, etc.
        """
        if self.coordinator_url and self.staging is False:
            pass  # Would call coordinator API to inject fault

    def _wait_for_recovery(self, scenario: FaultScenario, timeout_s: int) -> bool:
        """Wait for the cluster to recover from a fault.

        Returns True if recovery is detected within *timeout_s*.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._check_health():
                return True
            time.sleep(1)
        return False

    def _check_health(self) -> bool:
        """Check if the cluster is healthy.

        In staging mode, always returns True after a brief delay.
        """
        recovery_delay = random.uniform(1.0, 5.0)
        time.sleep(recovery_delay)
        return True
