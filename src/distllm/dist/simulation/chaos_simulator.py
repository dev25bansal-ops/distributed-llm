"""Chaos simulator for failure scenario testing and resilience measurement.

Models node failures, network partitions, latency spikes, and packet
loss to evaluate how a distributed inference cluster degrades under
adverse conditions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from distllm.dist.simulation.cluster_simulator import (
    ClusterSimulator,
    ModelConfig,
    NodeSpec,
    SimulatedPipelineResult,
    get_model_preset,
)


class ScenarioType(str, Enum):
    """Types of failure scenarios supported by ChaosSimulator."""

    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    LATENCY_SPIKE = "latency_spike"
    PACKET_LOSS = "packet_loss"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class ScenarioResult:
    """Result of a single chaos scenario simulation."""

    scenario_type: ScenarioType
    params: dict[str, Any]
    survived: bool  # Whether the cluster sustained service
    degraded_requests: float  # Fraction of requests degraded (0.0-1.0)
    recovery_time_ms: float  # Estimated time to recover
    latency_before_ms: float = 0.0
    latency_during_ms: float = 0.0
    throughput_before_tok_s: float = 0.0
    throughput_during_tok_s: float = 0.0
    detail: str = ""


@dataclass
class ResilienceReport:
    """Overall resilience assessment across multiple scenarios."""

    score: float  # 0.0 (brittle) to 1.0 (fully resilient)
    scenarios_evaluated: int = 0
    scenarios_survived: int = 0
    avg_degradation: float = 0.0
    avg_recovery_ms: float = 0.0
    weakest_scenario: str = ""
    strongest_scenario: str = ""
    per_scenario: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


# ── Simulation constants ──────────────────────────────────────────────

_DEFAULT_NODE_MTBF_HOURS = 5000.0  # Mean time between failures
_DEFAULT_RECOVERY_BASE_MS = 5000.0  # Base recovery time for a single node
_DEFAULT_NETWORK_RECOVERY_MS = 2000.0


class ChaosSimulator:
    """Simulate failure scenarios and measure cluster resilience.

    Supports node failures, network partitions, latency spikes, and
    packet loss.  Each scenario evaluates whether the cluster survives,
    the fraction of degraded requests, and estimated recovery time.
    """

    def __init__(
        self,
        model: str | ModelConfig = "LLaMA-7B",
        batch_size: int = 1,
        seq_len: int = 2048,
        seed: int | None = None,
    ) -> None:
        self._simulator = ClusterSimulator()
        self._model = get_model_preset(model) if isinstance(model, str) else model
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._rng = random.Random(seed)
        self._scenario_history: list[ScenarioResult] = []

    # ── Cluster setup ────────────────────────────────────────────────────

    def add_node(self, spec: NodeSpec) -> None:
        """Register a node in the simulation cluster."""
        self._simulator.add_node(spec)

    def add_nodes(self, *specs: NodeSpec) -> None:
        """Register multiple nodes."""
        for spec in specs:
            self._simulator.add_node(spec)

    def clear_nodes(self) -> None:
        """Remove all registered nodes."""
        self._simulator.clear_nodes()

    @property
    def num_nodes(self) -> int:
        return self._simulator.num_nodes

    # ── Scenario runners ─────────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_type: ScenarioType | str,
        params: dict[str, Any] | None = None,
    ) -> ScenarioResult:
        """Run a single chaos scenario and return the result.

        Args:
            scenario_type: The type of failure to simulate.
            params: Scenario-specific parameters.  See individual
                    ``_simulate_*`` methods for details.

        Returns:
            ScenarioResult with survival status, degradation, and recovery.
        """
        if isinstance(scenario_type, str):
            scenario_type = ScenarioType(scenario_type)

        params = params or {}
        runner = self._get_runner(scenario_type)
        result = runner(params)
        self._scenario_history.append(result)
        return result

    def _get_runner(
        self,
        scenario_type: ScenarioType,
    ) -> Any:
        """Return the runner method for a given scenario type."""
        runners = {
            ScenarioType.NODE_FAILURE: self._simulate_node_failure,
            ScenarioType.NETWORK_PARTITION: self._simulate_network_partition,
            ScenarioType.LATENCY_SPIKE: self._simulate_latency_spike,
            ScenarioType.PACKET_LOSS: self._simulate_packet_loss,
            ScenarioType.GRACEFUL_DEGRADATION: self._simulate_graceful_degradation,
            ScenarioType.CASCADING_FAILURE: self._simulate_cascading_failure,
        }
        runner = runners.get(scenario_type)
        if runner is None:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        return runner

    # ── Node failure ─────────────────────────────────────────────────────

    def _simulate_node_failure(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate one or more nodes failing.

        Params:
            failed_nodes (int | list[str]): Number of nodes to fail, or
                specific node IDs.  Default: 1 random node.
            recovery_time_ms (float | None): Override recovery time.
        """
        failed_nodes_spec = params.get("failed_nodes", 1)

        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]
        if not nodes:
            return ScenarioResult(
                scenario_type=ScenarioType.NODE_FAILURE,
                params=params,
                survived=False,
                degraded_requests=1.0,
                recovery_time_ms=0.0,
                detail="No nodes in cluster",
            )

        # Determine which nodes fail
        if isinstance(failed_nodes_spec, int):
            num_to_fail = min(failed_nodes_spec, len(nodes))
            failed = self._rng.sample(nodes, num_to_fail)
        elif isinstance(failed_nodes_spec, list):
            failed = [n for n in nodes if n.node_id in failed_nodes_spec]
        else:
            failed = [self._rng.choice(nodes)]

        failed_ids = {n.node_id for n in failed}

        # Baseline (healthy)
        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # Simulate failure: remove failed nodes
        surviving = [n for n in nodes if n.node_id not in failed_ids]

        degraded_requests = len(failed) / max(len(nodes), 1)
        survived = len(surviving) > 0

        if survived:
            degraded = self._simulator.run_pipeline(
                model=self._model,
                nodes=surviving,
                batch_size=self._batch_size,
                seq_len=self._seq_len,
            )
            latency_during = degraded.latency_ms
            throughput_during = degraded.throughput_tok_s
        else:
            latency_during = float("inf")
            throughput_during = 0.0

        recovery_ms = params.get(
            "recovery_time_ms",
            _DEFAULT_RECOVERY_BASE_MS * len(failed),
        )

        return ScenarioResult(
            scenario_type=ScenarioType.NODE_FAILURE,
            params=params,
            survived=survived,
            degraded_requests=round(degraded_requests, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=(
                latency_during if math.isfinite(latency_during) else -1.0
            ),
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=throughput_during,
            detail=(
                f"{len(failed)} node(s) failed: "
                f"{', '.join(sorted(failed_ids))}"
                + (
                    f"; {len(surviving)} surviving node(s)"
                    if survived
                    else "; cluster fully down"
                )
            ),
        )

    # ── Network partition ────────────────────────────────────────────────

    def _simulate_network_partition(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate a network partition isolating some nodes.

        Params:
            partition_size (int): Number of nodes in the isolated partition.
                Default: half the cluster.
            is_full_partition (bool): If True, the partition is complete
                (no cross-partition communication).  Default: True.
        """
        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]
        if not nodes:
            return ScenarioResult(
                scenario_type=ScenarioType.NETWORK_PARTITION,
                params=params,
                survived=False,
                degraded_requests=1.0,
                recovery_time_ms=0.0,
                detail="No nodes in cluster",
            )

        partition_size = params.get(
            "partition_size", max(1, len(nodes) // 2),
        )
        partition_size = min(partition_size, len(nodes) - 1)

        self._rng.shuffle(nodes)  # type: ignore[arg-type]
        isolated = nodes[:partition_size]
        main = nodes[partition_size:]

        if not main:
            main, isolated = isolated, main

        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # During partition: only nodes in the main partition can serve
        degraded_requests = len(isolated) / max(len(nodes), 1)
        survived = len(main) > 0

        if survived:
            degraded = self._simulator.run_pipeline(
                model=self._model,
                nodes=main,
                batch_size=self._batch_size,
                seq_len=self._seq_len,
            )
            latency_during = degraded.latency_ms
            throughput_during = degraded.throughput_tok_s
        else:
            latency_during = float("inf")
            throughput_during = 0.0

        recovery_ms = params.get(
            "recovery_time_ms", _DEFAULT_NETWORK_RECOVERY_MS,
        )

        return ScenarioResult(
            scenario_type=ScenarioType.NETWORK_PARTITION,
            params=params,
            survived=survived,
            degraded_requests=round(degraded_requests, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=(
                latency_during if math.isfinite(latency_during) else -1.0
            ),
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=throughput_during,
            detail=(
                f"Network partition: {partition_size} node(s) isolated "
                f"(main partition: {len(main)} node(s))"
            ),
        )

    # ── Latency spike ────────────────────────────────────────────────────

    def _simulate_latency_spike(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate a sudden latency spike on inter-node communication.

        Params:
            multiplier (float): How much to multiply normal latency by.
                Default: 10.0.
            affected_nodes (int | list[str] | None): Nodes affected.
                Default: all nodes.
        """
        multiplier = params.get("multiplier", 10.0)
        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]
        if not nodes:
            return ScenarioResult(
                scenario_type=ScenarioType.LATENCY_SPIKE,
                params=params,
                survived=False,
                degraded_requests=1.0,
                recovery_time_ms=0.0,
                detail="No nodes in cluster",
            )

        affected_spec = params.get("affected_nodes")
        if isinstance(affected_spec, int):
            affected = self._rng.sample(nodes, min(affected_spec, len(nodes)))
        elif isinstance(affected_spec, list):
            affected = [n for n in nodes if n.node_id in affected_spec]
        else:
            affected = list(nodes)

        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # Simulate latency spike: degrade interconnect bandwidth
        degraded_nodes = [
            NodeSpec(
                node_id=n.node_id,
                gpu_name=n.gpu_name,
                gpu_count=n.gpu_count,
                compute_tflops=n.compute_tflops,
                memory_gb=n.memory_gb,
                memory_bandwidth_gbps=n.memory_bandwidth_gbps,
                interconnect_gbps=n.interconnect_gbps / multiplier,
                intra_node_bw_gbps=n.intra_node_bw_gbps,
            )
            if n.node_id in {a.node_id for a in affected}
            else n
            for n in nodes
        ]

        degraded = self._simulator.run_pipeline(
            model=self._model,
            nodes=degraded_nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        degraded_fraction = len(affected) / max(len(nodes), 1)

        recovery_ms = params.get("recovery_time_ms", 3000.0)

        return ScenarioResult(
            scenario_type=ScenarioType.LATENCY_SPIKE,
            params=params,
            survived=True,
            degraded_requests=round(degraded_fraction, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=degraded.latency_ms,
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=degraded.throughput_tok_s,
            detail=(
                f"Latency spike x{multiplier} on "
                f"{len(affected)} node(s)"
            ),
        )

    # ── Packet loss ──────────────────────────────────────────────────────

    def _simulate_packet_loss(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate packet loss affecting inter-node communication.

        Params:
            loss_rate (float): Fraction of packets lost (0.0-1.0).
                Default: 0.05.
            affected_nodes (int | list[str] | None): Nodes affected.
                Default: all nodes.
        """
        loss_rate = max(0.0, min(1.0, params.get("loss_rate", 0.05)))
        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]

        # Packet loss causes effective bandwidth to drop by loss_rate
        # and adds retransmission latency
        bw_multiplier = 1.0 - loss_rate
        latency_multiplier = 1.0 + loss_rate * 20  # retransmission penalty

        affected_spec = params.get("affected_nodes")
        if isinstance(affected_spec, int):
            affected = self._rng.sample(nodes, min(affected_spec, len(nodes)))
        elif isinstance(affected_spec, list):
            affected = [n for n in nodes if n.node_id in affected_spec]
        else:
            affected = list(nodes)

        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        affected_ids = {a.node_id for a in affected}
        degraded_nodes = [
            NodeSpec(
                node_id=n.node_id,
                gpu_name=n.gpu_name,
                gpu_count=n.gpu_count,
                compute_tflops=n.compute_tflops,
                memory_gb=n.memory_gb,
                memory_bandwidth_gbps=n.memory_bandwidth_gbps,
                interconnect_gbps=(
                    n.interconnect_gbps * bw_multiplier
                    if n.node_id in affected_ids
                    else n.interconnect_gbps
                ),
                intra_node_bw_gbps=n.intra_node_bw_gbps,
            )
            for n in nodes
        ]

        degraded = self._simulator.run_pipeline(
            model=self._model,
            nodes=degraded_nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        degraded_fraction = len(affected) / max(len(nodes), 1)
        recovery_ms = params.get("recovery_time_ms", 1000.0)

        return ScenarioResult(
            scenario_type=ScenarioType.PACKET_LOSS,
            params=params,
            survived=True,
            degraded_requests=round(degraded_fraction, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=degraded.latency_ms,
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=degraded.throughput_tok_s,
            detail=(
                f"Packet loss at {loss_rate:.0%} on "
                f"{len(affected)} node(s); effective BW at "
                f"{bw_multiplier:.0%}"
            ),
        )

    # ── Graceful degradation ─────────────────────────────────────────────

    def _simulate_graceful_degradation(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate graceful performance degradation (e.g. thermal throttling).

        Params:
            performance_factor (float): Fraction of original performance
                retained (0.0-1.0).  Default: 0.5.
            affected_nodes (int | list[str] | None): Nodes affected.
                Default: all nodes.
        """
        perf_factor = max(0.05, min(1.0, params.get("performance_factor", 0.5)))
        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]

        affected_spec = params.get("affected_nodes")
        if isinstance(affected_spec, int):
            affected = self._rng.sample(nodes, min(affected_spec, len(nodes)))
        elif isinstance(affected_spec, list):
            affected = [n for n in nodes if n.node_id in affected_spec]
        else:
            affected = list(nodes)

        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        affected_ids = {a.node_id for a in affected}
        throttled_nodes = [
            NodeSpec(
                node_id=n.node_id,
                gpu_name=n.gpu_name,
                gpu_count=n.gpu_count,
                compute_tflops=(
                    n.compute_tflops * perf_factor
                    if n.node_id in affected_ids
                    else n.compute_tflops
                ),
                memory_gb=n.memory_gb,
                memory_bandwidth_gbps=(
                    n.memory_bandwidth_gbps * perf_factor
                    if n.node_id in affected_ids
                    else n.memory_bandwidth_gbps
                ),
                interconnect_gbps=n.interconnect_gbps,
                intra_node_bw_gbps=n.intra_node_bw_gbps,
            )
            for n in nodes
        ]

        degraded = self._simulator.run_pipeline(
            model=self._model,
            nodes=throttled_nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        degraded_fraction = len(affected) / max(len(nodes), 1)
        recovery_ms = params.get("recovery_time_ms", 30000.0)

        return ScenarioResult(
            scenario_type=ScenarioType.GRACEFUL_DEGRADATION,
            params=params,
            survived=True,
            degraded_requests=round(degraded_fraction, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=degraded.latency_ms,
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=degraded.throughput_tok_s,
            detail=(
                f"Graceful degradation at {perf_factor:.0%} performance "
                f"on {len(affected)} node(s)"
            ),
        )

    # ── Cascading failure ────────────────────────────────────────────────

    def _simulate_cascading_failure(
        self,
        params: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate a cascading failure where one node's failure overloads others.

        Params:
            initial_failures (int): Nodes that fail initially.  Default: 1.
            overload_factor (float): How much extra load hits surviving nodes,
                as a fraction of the failed node's share.  Default: 1.2.
            cascade_rounds (int): How many rounds of cascading to simulate.
                Default: 3.
        """
        initial = params.get("initial_failures", 1)
        overload = params.get("overload_factor", 1.2)
        cascade_rounds = params.get("cascade_rounds", 3)

        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]
        if not nodes:
            return ScenarioResult(
                scenario_type=ScenarioType.CASCADING_FAILURE,
                params=params,
                survived=False,
                degraded_requests=1.0,
                recovery_time_ms=0.0,
                detail="No nodes in cluster",
            )

        baseline = self._simulator.run_pipeline(
            model=self._model,
            nodes=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        failed_ids: set[str] = set()
        current_nodes = list(nodes)

        for round_idx in range(cascade_rounds):
            if not current_nodes:
                break

            # First round: fail initial nodes
            if round_idx == 0:
                to_fail = self._rng.sample(
                    current_nodes, min(initial, len(current_nodes)),
                )
            else:
                # Subsequent rounds: fail overloaded nodes
                # Simulate overload by failing a fraction of survivors
                overloaded_count = max(
                    1, int(len(current_nodes) * 0.3 * overload),
                )
                to_fail = self._rng.sample(
                    current_nodes,
                    min(overloaded_count, len(current_nodes)),
                )

            for node in to_fail:
                failed_ids.add(node.node_id)
            current_nodes = [
                n for n in current_nodes if n.node_id not in failed_ids
            ]

        survived = len(current_nodes) > 0

        if survived:
            degraded = self._simulator.run_pipeline(
                model=self._model,
                nodes=current_nodes,
                batch_size=self._batch_size,
                seq_len=self._seq_len,
            )
            latency_during = degraded.latency_ms
            throughput_during = degraded.throughput_tok_s
        else:
            latency_during = float("inf")
            throughput_during = 0.0

        degraded_requests = len(failed_ids) / max(len(nodes), 1)
        recovery_ms = params.get(
            "recovery_time_ms",
            _DEFAULT_RECOVERY_BASE_MS * len(failed_ids) * 2,
        )

        return ScenarioResult(
            scenario_type=ScenarioType.CASCADING_FAILURE,
            params=params,
            survived=survived,
            degraded_requests=round(degraded_requests, 4),
            recovery_time_ms=recovery_ms,
            latency_before_ms=baseline.latency_ms,
            latency_during_ms=(
                latency_during if math.isfinite(latency_during) else -1.0
            ),
            throughput_before_tok_s=baseline.throughput_tok_s,
            throughput_during_tok_s=throughput_during,
            detail=(
                f"Cascading failure: {len(failed_ids)} total node(s) "
                f"lost over {cascade_rounds} round(s) "
                f"(initial={initial}, overload={overload:.1f}x)"
            ),
        )

    # ── Resilience measurement ───────────────────────────────────────────

    def measure_resilience(
        self,
        num_scenarios: int = 10,
        seed: int | None = None,
    ) -> ResilienceReport:
        """Evaluate cluster resilience across a battery of scenarios.

        Runs ``num_scenarios`` random scenarios and computes a resilience
        score in [0.0, 1.0].

        Args:
            num_scenarios: Number of random scenarios to run.
            seed: Random seed for reproducibility.

        Returns:
            ResilienceReport with aggregate score and per-scenario breakdown.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        nodes = list(self._simulator._nodes.values())  # type: ignore[attr-defined]
        if not nodes:
            return ResilienceReport(
                score=0.0,
                recommendations=["No nodes in cluster to evaluate"],
            )

        scenario_types = list(ScenarioType)

        scenario_results: list[ScenarioResult] = []
        for _ in range(num_scenarios):
            st = self._rng.choice(scenario_types)
            params: dict[str, Any] = {}

            if st == ScenarioType.NODE_FAILURE:
                params["failed_nodes"] = self._rng.randint(1, max(1, len(nodes) - 1))
            elif st == ScenarioType.NETWORK_PARTITION:
                params["partition_size"] = self._rng.randint(
                    1, max(1, len(nodes) - 1),
                )
            elif st == ScenarioType.LATENCY_SPIKE:
                params["multiplier"] = self._rng.choice([2.0, 5.0, 10.0, 20.0])
            elif st == ScenarioType.PACKET_LOSS:
                params["loss_rate"] = self._rng.uniform(0.01, 0.15)
            elif st == ScenarioType.GRACEFUL_DEGRADATION:
                params["performance_factor"] = self._rng.uniform(0.3, 0.8)
            elif st == ScenarioType.CASCADING_FAILURE:
                params["initial_failures"] = self._rng.randint(
                    1, max(1, len(nodes) // 3),
                )
                params["overload_factor"] = self._rng.uniform(1.0, 1.5)
                params["cascade_rounds"] = self._rng.randint(2, 4)

            try:
                result = self.run_scenario(st, params)
                scenario_results.append(result)
            except Exception:
                continue

        if not scenario_results:
            return ResilienceReport(
                score=0.0,
                recommendations=["All scenarios failed to execute"],
            )

        # Compute aggregate metrics
        survived_count = sum(1 for r in scenario_results if r.survived)
        avg_degradation = sum(
            r.degraded_requests for r in scenario_results
        ) / len(scenario_results)
        avg_recovery = sum(
            r.recovery_time_ms for r in scenario_results
        ) / len(scenario_results)

        # Survival contributes 50%, low degradation contributes 30%,
        # fast recovery contributes 20%
        survival_score = survived_count / len(scenario_results)
        degradation_score = 1.0 - avg_degradation
        recovery_score = max(
            0.0,
            1.0 - avg_recovery / 120000.0,  # Scale: 2 min recovery = 0
        )
        total_score = (
            survival_score * 0.5
            + degradation_score * 0.3
            + recovery_score * 0.2
        )
        total_score = max(0.0, min(1.0, total_score))

        # Per-scenario-type score
        per_type: dict[str, float] = {}
        for st in scenario_types:
            relevant = [r for r in scenario_results if r.scenario_type == st]
            if not relevant:
                continue
            s_survived = sum(1 for r in relevant if r.survived)
            s_degradation = sum(r.degraded_requests for r in relevant)
            per_type[st.value] = round(
                (s_survived / len(relevant)) * 0.6
                + (1.0 - s_degradation / len(relevant)) * 0.4,
                4,
            )

        # Weakest / strongest
        if per_type:
            weakest = min(per_type, key=per_type.get)  # type: ignore[arg-type]
            strongest = max(per_type, key=per_type.get)  # type: ignore[arg-type]
        else:
            weakest = ""
            strongest = ""

        recommendations = self._generate_recommendations(
            per_type, total_score,
        )

        return ResilienceReport(
            score=round(total_score, 4),
            scenarios_evaluated=len(scenario_results),
            scenarios_survived=survived_count,
            avg_degradation=round(avg_degradation, 4),
            avg_recovery_ms=round(avg_recovery, 2),
            weakest_scenario=weakest,
            strongest_scenario=strongest,
            per_scenario=per_type,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        per_type: dict[str, float],
        total_score: float,
    ) -> list[str]:
        """Generate human-readable recommendations from resilience scores."""
        recs: list[str] = []

        if total_score < 0.3:
            recs.append(
                "CRITICAL: Cluster has very low resilience. "
                "Consider adding redundant nodes and network paths."
            )
        elif total_score < 0.6:
            recs.append(
                "WARNING: Cluster resilience is moderate. "
                "Review the weakest scenario types."
            )

        for scenario_type, score in per_type.items():
            if score < 0.4:
                if "node_failure" in scenario_type:
                    recs.append(
                        f"Low resilience to node failures (score={score:.2f}). "
                        "Consider N+1 redundancy."
                    )
                elif "network_partition" in scenario_type:
                    recs.append(
                        f"Low resilience to network partitions (score={score:.2f}). "
                        "Consider multi-homing or redundant fabrics."
                    )
                elif "latency_spike" in scenario_type:
                    recs.append(
                        f"Low resilience to latency spikes (score={score:.2f}). "
                        "Consider request timeouts and circuit breakers."
                    )
                elif "packet_loss" in scenario_type:
                    recs.append(
                        f"Low resilience to packet loss (score={score:.2f}). "
                        "Consider reliable transport or FEC."
                    )
                elif "graceful" in scenario_type:
                    recs.append(
                        f"Low resilience to graceful degradation (score={score:.2f}). "
                        "Consider dynamic batch sizing under throttling."
                    )
                elif "cascading" in scenario_type:
                    recs.append(
                        f"Low resilience to cascading failures (score={score:.2f}). "
                        "Consider load shedding and per-node capacity limits."
                    )

        return recs

    # ── Utilities ────────────────────────────────────────────────────────

    def history(self) -> list[ScenarioResult]:
        """Return the full history of scenario results."""
        return list(self._scenario_history)

    def clear_history(self) -> None:
        """Clear the scenario result history."""
        self._scenario_history.clear()

    def summary(self) -> str:
        """Human-readable summary of the chaos simulator state."""
        lines: list[str] = [
            f"ChaosSimulator: {self._simulator.num_nodes} nodes, "
            f"{len(self._scenario_history)} scenarios run"
        ]
        if self._scenario_history:
            survived = sum(1 for r in self._scenario_history if r.survived)
            avg_degradation = (
                sum(r.degraded_requests for r in self._scenario_history)
                / len(self._scenario_history)
            )
            lines.append(
                f"  Survival rate: {survived}/{len(self._scenario_history)} "
                f"({survived / max(len(self._scenario_history), 1):.0%})"
            )
            lines.append(
                f"  Avg degradation: {avg_degradation:.1%}"
            )
        return "\n".join(lines)
