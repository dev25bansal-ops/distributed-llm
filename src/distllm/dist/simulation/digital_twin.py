"""Digital Twin Simulator for infrastructure what-if planning.

Provides a digital twin of the distributed LLM inference cluster that can be
snapshotted from a production coordinator, modified via what-if mutations,
and simulated to estimate throughput, latency, cost, and failure rates under
different topologies and load conditions.

Usage::

    from distllm.dist.simulation.digital_twin import DigitalTwin, WhatIfEngine

    twin = DigitalTwin()
    twin.add_nodes(count=4, gpu_type="H100", region="us-east-1")

    result = twin.run_simulation(duration_s=1800)
    print(result.throughput, result.latency_p99)

    engine = WhatIfEngine(twin)
    deltas = engine.compare_with_baseline({"gpu_type": "A100", "count": 8})
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimClusterNode:
    """A single node in the simulated cluster.

    Attributes:
        node_id: Unique identifier for the node.
        gpu_type: GPU hardware identifier (e.g. ``"H100"``, ``"A100"``).
        gpu_count: Number of GPUs on this node.
        region: Cloud region or data-centre label.
        hourly_cost: Cost per hour in USD.
        layers: Optional range of transformer layers assigned to this node
            as ``(first_layer_index, last_layer_index)``.
    """

    node_id: str
    gpu_type: str
    gpu_count: int
    region: str
    hourly_cost: float
    layers: tuple[int, int] | None = None


@dataclass(frozen=True)
class SimRequest:
    """A single inference request submitted during simulation.

    Attributes:
        prompt: The input prompt text.
        prompt_length: Number of tokens in the prompt.
        max_tokens: Maximum number of tokens to generate.
        model: Model identifier (e.g. ``"llama-70b"``).
        arrival_time: Simulated arrival timestamp in seconds.
        request_id: Auto-generated unique identifier.
    """

    prompt: str
    prompt_length: int
    max_tokens: int
    model: str
    arrival_time: float
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated metrics produced by a single simulation run.

    Attributes:
        total_requests: Number of requests submitted during the run.
        completed_requests: Number of requests that finished successfully.
        throughput: Completed requests per second.
        latency_p50: Median end-to-end latency in milliseconds.
        latency_p95: 95th percentile latency in milliseconds.
        latency_p99: 99th percentile latency in milliseconds.
        failures: Number of requests that failed or timed out.
        total_cost: Aggregate cost across all active nodes in USD.
        duration_s: Simulation wall-clock duration in seconds.
        node_hours: Total node-hours consumed.
    """

    total_requests: int
    completed_requests: int
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    failures: int
    total_cost: float
    duration_s: float
    node_hours: float


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------


class DigitalTwin:
    """Digital twin of the distributed inference cluster for what-if planning.

    The twin maintains a mutable view of cluster topology (nodes, GPU types,
    regions, costs) and a load multiplier that scales request arrival rates.
    Call :meth:`run_simulation` to obtain performance projections under the
    current configuration.

    Example::

        twin = DigitalTwin()
        twin.snapshot_from_production("http://coordinator:8080")
        twin.add_nodes(2, "H100", "eu-west-1")
        twin.set_load_multiplier(1.5)
        result = twin.run_simulation(duration_s=3600)
        print(result.throughput)
    """

    # Per-GPU throughput lookup tokens/s (arbitrary reference values used
    # when the model is not explicitly profiled).
    _GPU_TOKENS_PER_SECOND: dict[str, float] = {
        "H100": 850.0,
        "H200": 1100.0,
        "A100": 600.0,
        "A100-80GB": 650.0,
        "L40S": 450.0,
        "V100": 350.0,
        "T4": 200.0,
    }

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise the digital twin.

        Args:
            config_path: Optional path to a JSON or YAML configuration file
                that pre-populates nodes and settings.  If ``None`` the twin
                starts empty and must be populated via :meth:`add_nodes` or
                :meth:`snapshot_from_production`.
        """
        self._nodes: dict[str, SimClusterNode] = {}
        self._load_multiplier: float = 1.0
        self._base_arrival_rate: float = 1.0  # requests / second at 1x load

        if config_path is not None:
            self._load_config(config_path)

    # -- topology mutation ---------------------------------------------------

    def snapshot_from_production(self, coordinator_url: str) -> int:
        """Capture the current cluster topology from a live coordinator.

        Args:
            coordinator_url: Base URL of the production coordinator (e.g.
                ``"http://coordinator:8080"``).

        Returns:
            Number of nodes captured.

        Note:
            This is a stub that returns simulated data when the coordinator
            is unreachable.  In a real deployment it would call the
            coordinator's topology API.
        """
        # In production this would be:
        #   import requests
        #   resp = requests.get(f"{coordinator_url}/api/v1/topology", timeout=10)
        #   for node_data in resp.json()["nodes"]: ...
        node_count = 0
        # Stub: generate 4 default nodes when no config has been loaded.
        if not self._nodes:
            for i in range(4):
                node = SimClusterNode(
                    node_id=f"snap-{uuid.uuid4().hex[:8]}",
                    gpu_type="H100",
                    gpu_count=8,
                    region="us-east-1",
                    hourly_cost=45.92,
                )
                self._nodes[node.node_id] = node
                node_count += 1
        return node_count

    def add_nodes(
        self,
        count: int,
        gpu_type: str,
        region: str,
        gpu_count: int = 8,
        hourly_cost: float | None = None,
    ) -> list[str]:
        """Add simulated nodes to the cluster.

        This is the primary method for "what-if" topology changes.

        Args:
            count: Number of nodes to add.
            gpu_type: GPU hardware type (e.g. ``"H100"``, ``"A100"``).
            region: Cloud region or data-centre label.
            gpu_count: Number of GPUs per node. Defaults to 8.
            hourly_cost: Per-node hourly cost in USD.  If ``None``, a sensible
                default is looked up based on the GPU type.

        Returns:
            List of newly created node IDs.
        """
        if hourly_cost is None:
            hourly_cost = self._default_hourly_cost(gpu_type, gpu_count)

        node_ids: list[str] = []
        for _ in range(count):
            node_id = f"sim-{uuid.uuid4().hex[:12]}"
            node = SimClusterNode(
                node_id=node_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                region=region,
                hourly_cost=hourly_cost,
            )
            self._nodes[node_id] = node
            node_ids.append(node_id)
        return node_ids

    def remove_nodes(self, node_ids: list[str]) -> int:
        """Remove nodes from the simulated cluster.

        Silently skips any node IDs that do not exist.

        Args:
            node_ids: List of node identifiers to remove.

        Returns:
            Number of nodes actually removed.
        """
        removed = 0
        for node_id in node_ids:
            if node_id in self._nodes:
                del self._nodes[node_id]
                removed += 1
        return removed

    def set_load_multiplier(self, multiplier: float) -> None:
        """Scale the request arrival rate.

        A multiplier of ``2.0`` doubles the number of requests per second.

        Args:
            multiplier: Scaling factor applied to the base arrival rate.
                Must be non-negative.
        """
        self._load_multiplier = max(0.0, multiplier)

    # -- simulation ----------------------------------------------------------

    def run_simulation(
        self,
        duration_s: float = 3600.0,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run the digital twin simulation and produce performance metrics.

        The simulator models a Poisson arrival process, assigns requests to
        nodes in a round-robin fashion, estimates per-request latency from
        GPU throughput tables, and tracks failures resulting from capacity
        exhaustion or timeouts.

        Args:
            duration_s: Simulated wall-clock time in seconds.
                Defaults to 3600 (1 hour).
            seed: Optional random seed for reproducibility.

        Returns:
            A :class:`SimulationResult` with aggregated metrics.

        Raises:
            RuntimeError: If the cluster has no nodes.
        """
        if not self._nodes:
            raise RuntimeError("Cannot run simulation with zero nodes")

        rng = random.Random(seed)
        node_list = list(self._nodes.values())
        arrival_rate = self._base_arrival_rate * self._load_multiplier

        # -- generate request timeline (Poisson process) --
        requests: list[SimRequest] = []
        t = 0.0
        while t < duration_s:
            inter_arrival = rng.expovariate(arrival_rate) if arrival_rate > 0 else duration_s
            t += inter_arrival
            if t >= duration_s:
                break

            prompt_length = int(rng.gauss(1024, 256))
            prompt_length = max(16, min(4096, prompt_length))
            max_tokens = int(rng.gauss(256, 64))
            max_tokens = max(16, min(2048, max_tokens))

            requests.append(SimRequest(
                prompt="",
                prompt_length=prompt_length,
                max_tokens=max_tokens,
                model="llama-70b",
                arrival_time=t,
            ))

        # -- process requests --
        latencies: list[float] = []
        failures = 0
        total_node_time = 0.0

        # Track per-node busy-until timestamps for simple capacity modelling.
        node_busy_until: dict[str, float] = {n.node_id: 0.0 for n in node_list}

        for req in requests:
            # Pick the earliest-available node (round-robin tie-break).
            candidate: tuple[str, float] | None = None
            for node in node_list:
                busy_until = node_busy_until[node.node_id]
                if candidate is None or busy_until < candidate[1]:
                    candidate = (node.node_id, busy_until)

            if candidate is None:
                failures += 1
                continue

            node_id, ready_at = candidate
            node = self._nodes[node_id]
            start_time = max(req.arrival_time, ready_at)

            # Estimate latency: (prompt + generation) / tokens-per-second
            total_tokens = req.prompt_length + req.max_tokens
            gpu_tps = self._GPU_TOKENS_PER_SECOND.get(node.gpu_type, 400.0)
            effective_tps = gpu_tps * node.gpu_count * 0.85  # 85% scaling efficiency
            processing_time_s = total_tokens / effective_tps if effective_tps > 0 else 999.0

            # Add a small latency jitter
            processing_time_s *= 1.0 + rng.gauss(0, 0.05)
            processing_time_s = max(0.001, processing_time_s)

            end_time = start_time + processing_time_s
            node_busy_until[node_id] = end_time
            total_node_time += processing_time_s

            # Timeout check (requests exceeding 300 s are failed)
            if processing_time_s > 300.0:
                failures += 1
                continue

            latencies.append((end_time - req.arrival_time) * 1000.0)  # ms

        # -- aggregate results --
        completed = len(latencies)
        total_requests = len(requests)
        failures += max(0, total_requests - completed - failures)

        if completed > 0:
            sorted_lat = sorted(latencies)
            throughput = completed / max(duration_s, 0.001)
            latency_p50 = _percentile(sorted_lat, 50)
            latency_p95 = _percentile(sorted_lat, 95)
            latency_p99 = _percentile(sorted_lat, 99)
        else:
            throughput = 0.0
            latency_p50 = 0.0
            latency_p95 = 0.0
            latency_p99 = 0.0

        node_hours = total_node_time / 3600.0
        total_cost = sum(
            n.hourly_cost * (total_node_time / 3600.0) / len(self._nodes)
            for n in self._nodes.values()
        )

        return SimulationResult(
            total_requests=total_requests,
            completed_requests=completed,
            throughput=round(throughput, 2),
            latency_p50=round(latency_p50, 1),
            latency_p95=round(latency_p95, 1),
            latency_p99=round(latency_p99, 1),
            failures=failures,
            total_cost=round(total_cost, 2),
            duration_s=duration_s,
            node_hours=round(node_hours, 3),
        )

    # -- helpers -------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load cluster configuration from a file.

        Supports JSON with keys ``"nodes"`` (list of node dicts) and
        ``"base_arrival_rate"`` (float).  Each node dict must contain at
        least ``"gpu_type"`` and ``"region"``.

        Args:
            config_path: Path to the configuration file.
        """

        with open(config_path) as f:
            config = json.load(f)

        self._base_arrival_rate = config.get("base_arrival_rate", self._base_arrival_rate)

        for entry in config.get("nodes", []):
            node = SimClusterNode(
                node_id=entry.get("node_id", f"cfg-{uuid.uuid4().hex[:12]}"),
                gpu_type=entry["gpu_type"],
                gpu_count=entry.get("gpu_count", 8),
                region=entry["region"],
                hourly_cost=entry.get("hourly_cost", self._default_hourly_cost(entry["gpu_type"])),
                layers=tuple(entry["layers"]) if entry.get("layers") else None,
            )
            self._nodes[node.node_id] = node

    @staticmethod
    def _default_hourly_cost(gpu_type: str, gpu_count: int = 8) -> float:
        """Return a sensible default hourly cost for a given GPU type.

        Values are rough on-demand estimates for an 8-GPU instance.
        """
        _costs: dict[str, float] = {
            "H200": 55.00,
            "H100": 45.92,
            "A100-80GB": 40.00,
            "A100": 32.77,
            "L40S": 18.00,
            "V100": 12.00,
            "T4": 4.50,
        }
        base = _costs.get(gpu_type, 20.0)
        return round(base * (gpu_count / 8), 2)


# ---------------------------------------------------------------------------
# What-If Engine
# ---------------------------------------------------------------------------


class WhatIfEngine:
    """Run what-if queries against a :class:`DigitalTwin` and compare results.

    The engine clones the twin's current state, applies parameter overrides,
    runs a simulation, and optionally computes deltas against a baseline.

    Example::

        twin = DigitalTwin()
        twin.add_nodes(4, "H100", "us-east-1")
        engine = WhatIfEngine(twin)

        # Single query
        result = engine.query({"gpu_type": "A100", "count": 8})

        # Compare with baseline
        deltas = engine.compare_with_baseline({"gpu_type": "H200", "count": 2})
    """

    def __init__(self, twin: DigitalTwin) -> None:
        """Initialise the what-if engine.

        Args:
            twin: The :class:`DigitalTwin` instance to base queries on.
                The original twin is not mutated by :meth:`query` or
                :meth:`compare_with_baseline`.
        """
        self._twin = twin

    def query(self, params: dict[str, Any], seed: int | None = None) -> SimulationResult:
        """Run a what-if simulation with topology overrides.

        The ``params`` dict can contain any of the following keys:

        * ``count`` (int) -- number of nodes to add (after removing existing
          nodes if ``replace`` is true).
        * ``gpu_type`` (str) -- GPU hardware type.
        * ``region`` (str) -- cloud region.
        * ``gpu_count`` (int) -- GPUs per node.
        * ``replace`` (bool) -- if ``True``, remove all existing nodes first.
        * ``load_multiplier`` (float) -- load scaling factor.
        * ``duration_s`` (float) -- simulation duration in seconds.

        Args:
            params: Dictionary of parameter overrides.

        Returns:
            A :class:`SimulationResult` for the what-if topology.
        """
        twin = self._make_twin(params)
        duration_s = float(params.get("duration_s", 3600.0))
        return twin.run_simulation(duration_s=duration_s, seed=seed)

    def compare_with_baseline(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run a what-if query and return delta metrics vs the baseline twin.

        The baseline is the twin as it was when the engine was constructed.
        The ``params`` dict follows the same schema as :meth:`query`.

        Args:
            params: Dictionary of parameter overrides (same as :meth:`query`).

        Returns:
            A dictionary containing the baseline result, what-if result, and
            per-metric deltas::

                {
                    "baseline": SimulationResult,
                    "what_if": SimulationResult,
                    "delta": {
                        "throughput_pct": 25.3,
                        "latency_p50_pct": -12.1,
                        "latency_p95_pct": -8.4,
                        "latency_p99_pct": -5.2,
                        "failures_delta": -10,
                        "total_cost_pct": 18.7,
                    },
                }

            Positive delta values indicate an increase; negative values
            indicate a decrease.  Percentage changes use ``((new - old) /
            abs(old)) * 100``.
        """
        baseline = self._twin.run_simulation(duration_s=3600.0)
        what_if = self.query(params)

        def pct(old: float, new: float) -> float:
            if abs(old) < 1e-9:
                return 0.0
            return round(((new - old) / abs(old)) * 100, 1)

        delta = {
            "throughput_pct": pct(baseline.throughput, what_if.throughput),
            "latency_p50_pct": pct(baseline.latency_p50, what_if.latency_p50),
            "latency_p95_pct": pct(baseline.latency_p95, what_if.latency_p95),
            "latency_p99_pct": pct(baseline.latency_p99, what_if.latency_p99),
            "failures_delta": what_if.failures - baseline.failures,
            "total_cost_pct": pct(baseline.total_cost, what_if.total_cost),
        }

        return {
            "baseline": baseline,
            "what_if": what_if,
            "delta": delta,
        }

    # -- helpers -------------------------------------------------------------

    def _make_twin(self, params: dict[str, Any]) -> DigitalTwin:
        """Create a shallow copy of the baseline twin with overrides applied."""
        twin = DigitalTwin()

        # Clone existing nodes.
        for node in self._twin._nodes.values():
            twin._nodes[node.node_id] = node

        # Copy load multiplier.
        twin._load_multiplier = self._twin._load_multiplier

        # Apply overrides.
        if params.get("replace"):
            twin._nodes.clear()

        count = params.get("count", 0)
        if count > 0:
            twin.add_nodes(
                count=count,
                gpu_type=params.get("gpu_type", "H100"),
                region=params.get("region", "us-east-1"),
                gpu_count=params.get("gpu_count", 8),
                # Cost-aware what-if: when a real cloud PriceQuote is fed in,
                # ``hourly_cost`` carries the quoted per-node price into the
                # simulated topology so the scenario's cost reflects live
                # pricing instead of the built-in GPU default table.
                hourly_cost=params.get("hourly_cost"),
            )

        if "load_multiplier" in params:
            twin.set_load_multiplier(float(params["load_multiplier"]))

        return twin


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_data: list[float], p: int) -> float:
    """Return the *p*-th percentile of a sorted list.

    Uses linear interpolation between adjacent values (same behaviour as
    ``numpy.percentile`` with ``interpolation="linear"``).
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1
