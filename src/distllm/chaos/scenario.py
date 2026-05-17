"""Chaos scenario definitions and runner."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class ChaosStep:
    """A single step in a chaos scenario."""
    action: str  # "kill_node", "add_latency", "drop_message", "corrupt_data"
    params: dict[str, Any] = field(default_factory=dict)
    delay_after: float = 0.0  # seconds to wait after executing this step


@dataclass
class ChaosScenario:
    """A complete chaos scenario with ordered steps."""
    name: str
    steps: list[ChaosStep] = field(default_factory=list)
    expected_recovery_time_s: float = 30.0
    max_acceptable_error_rate: float = 0.05
    description: str = ""


@dataclass
class ScenarioResult:
    """Result of executing a chaos scenario."""
    scenario_name: str
    steps_executed: int
    steps_failed: int
    total_duration_s: float
    actual_recovery_time_s: float
    actual_error_rate: float
    resilience_score: float = 0.0
    events: list = field(default_factory=list)


class ScenarioRunner:
    """Executes chaos scenarios and tracks results."""

    def __init__(self, injector):
        self.injector = injector
        self._results: list[ScenarioResult] = []

    @property
    def results(self) -> list[ScenarioResult]:
        return list(self._results)

    def run_scenario(
        self,
        scenario: ChaosScenario,
        monitor_callback: Callable | None = None,
    ) -> ScenarioResult:
        """Execute a chaos scenario step by step.

        Args:
            scenario: The scenario to execute.
            monitor_callback: Optional callback(step_index, event) called after each step.

        Returns:
            ScenarioResult with execution details.
        """
        logger.info(f"[Chaos] Starting scenario: {scenario.name}")
        start_time = time.time()
        steps_executed = 0
        steps_failed = 0
        events = []

        for i, step in enumerate(scenario.steps):
            try:
                event = self._execute_step(step)
                events.append(event)
                steps_executed += 1
                if event.result != "success":
                    steps_failed += 1
                if monitor_callback:
                    monitor_callback(i, event)
            except Exception as e:
                logger.error(f"[Chaos] Step {i} failed: {e}")
                steps_failed += 1

            if step.delay_after > 0:
                time.sleep(step.delay_after)

        total_duration = time.time() - start_time
        # For now, estimate recovery time as total duration
        actual_recovery = total_duration

        result = ScenarioResult(
            scenario_name=scenario.name,
            steps_executed=steps_executed,
            steps_failed=steps_failed,
            total_duration_s=total_duration,
            actual_recovery_time_s=actual_recovery,
            actual_error_rate=0.0,  # Set by caller via monitoring
            events=events,
        )
        self._results.append(result)
        logger.info(
            f"[Chaos] Scenario '{scenario.name}' complete: "
            f"{steps_executed}/{len(scenario.steps)} steps, "
            f"{steps_failed} failures, {total_duration:.1f}s"
        )
        return result

    def _execute_step(self, step: ChaosStep):
        """Execute a single chaos step."""
        action = step.action
        params = step.params

        if action == "kill_node":
            return self.injector.kill_node(params.get("node_id", ""))
        elif action == "add_latency":
            return self.injector.add_latency(
                node_id=params.get("node_id", ""),
                delay_ms=params.get("delay_ms", 100),
                duration_s=params.get("duration_s", 0),
            )
        elif action == "drop_message":
            return self.injector.drop_message(
                node_id=params.get("node_id", ""),
                message_pattern=params.get("pattern", ""),
                duration_s=params.get("duration_s", 0),
            )
        elif action == "corrupt_data":
            return self.injector.corrupt_data(
                node_id=params.get("node_id", ""),
                corruption_rate=params.get("corruption_rate", 0.1),
                duration_s=params.get("duration_s", 0),
            )
        else:
            raise ValueError(f"Unknown chaos action: {action}")
