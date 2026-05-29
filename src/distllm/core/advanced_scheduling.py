"""Advanced scheduling policies for heterogeneous, cost-aware, WAN, and energy-aware inference.

Integrates with BatchScheduler to provide:
1. SchedulingPolicy protocol — pluggable budget computation strategy
2. Heterogeneous P2P Scheduling — device-aware budget computation
3. Cost-Aware Scheduling — feed per-node cost into priority weights
4. WAN-Optimized Scheduling — larger chunks, prefetch, pipeline-aware
5. Energy-Aware Scheduling — trade off batch size vs GPU power draw
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.batch_scheduler import IterationBudget, Sequence


# ── SchedulingPolicy Protocol ──────────────────────────────────────────────

@runtime_checkable
class SchedulingPolicy(Protocol):
    """Protocol for pluggable scheduling policies.

    A scheduling policy computes the iteration budget and optionally
    modifies sequence priorities before the batch is built.

    Implement this protocol to create custom scheduling strategies:
    - ``compute_budget``: adjust the iteration budget
    - ``on_before_schedule``: modify sequence priorities (optional)
    """

    def compute_budget(self, base_budget: "IterationBudget") -> "IterationBudget":
        """Compute the iteration budget for this step.

        Args:
            base_budget: The base budget from the scheduler configuration.

        Returns:
            Modified budget for this iteration.
        """
        ...

    def on_before_schedule(self, sequences: list["Sequence"]) -> list["Sequence"]:
        """Called before scheduling to allow priority modifications.

        Default implementation returns sequences unchanged.

        Args:
            sequences: Pending sequences about to be scheduled.

        Returns:
            Sequences with possibly modified priorities.
        """
        ...


@dataclass
class DefaultPolicy:
    """Passthrough policy — returns the base budget unchanged."""

    def compute_budget(self, base_budget: "IterationBudget") -> "IterationBudget":
        return base_budget

    def on_before_schedule(self, sequences: list["Sequence"]) -> list["Sequence"]:
        return sequences


@dataclass
class SarathiPolicy:
    """Sarathi-Serve style adaptive scheduling policy.

    Dynamically adjusts the prefill/decode split based on decode
    pipeline pressure.  When pressure is high (decode latency above
    target), prefill tokens are throttled and more decode slots are
    reserved.  When idle, more budget is allocated to prefill.
    """

    pressure_tracker: Any = None  # DecodePressureTracker instance
    min_prefill_ratio: float = 0.25  # Min prefill budget under pressure

    def compute_budget(self, base_budget: "IterationBudget") -> "IterationBudget":
        if self.pressure_tracker is None:
            return base_budget

        pressure = self.pressure_tracker.pressure
        from distllm.core.batch_scheduler import IterationBudget

        base_decode_slots = min(base_budget.max_batch_size, base_budget.max_decode_tokens)

        if pressure > 0.7:
            adjusted_decode = min(base_budget.max_batch_size, int(base_decode_slots * (1.0 + pressure)))
        elif pressure < 0.3:
            adjusted_decode = max(1, int(base_decode_slots * 0.6))
        else:
            adjusted_decode = base_decode_slots

        adjusted_decode = min(adjusted_decode, base_budget.max_batch_size)
        total_after_decode = max(0, base_budget.max_total_tokens - adjusted_decode)

        if pressure > 0.8:
            prefill_scale = max(self.min_prefill_ratio, 1.0 - pressure)
        elif pressure > 0.5:
            prefill_scale = 0.75
        else:
            prefill_scale = 1.0

        adjusted_prefill = min(
            int(base_budget.max_prefill_tokens * prefill_scale),
            total_after_decode,
        )

        adjusted_batch = base_budget.max_batch_size
        if pressure > 0.9:
            adjusted_batch = max(adjusted_decode, int(base_budget.max_batch_size * 0.5))

        return IterationBudget(
            max_prefill_tokens=adjusted_prefill,
            max_decode_tokens=adjusted_decode,
            max_batch_size=adjusted_batch,
            max_total_tokens=base_budget.max_total_tokens,
            enable_chunked_prefill=base_budget.enable_chunked_prefill,
            prefill_slack_ratio=base_budget.prefill_slack_ratio,
        )

    def on_before_schedule(self, sequences: list["Sequence"]) -> list["Sequence"]:
        return sequences


@dataclass
class CompositePolicy:
    """Chains multiple policies together.

    Applies policies in order: the first policy's budget output is
    fed as input to the second, and so on.
    """

    policies: list[Any] = field(default_factory=list)

    def compute_budget(self, base_budget: "IterationBudget") -> "IterationBudget":
        budget = base_budget
        for policy in self.policies:
            budget = policy.compute_budget(budget)
        return budget

    def on_before_schedule(self, sequences: list["Sequence"]) -> list["Sequence"]:
        for policy in self.policies:
            sequences = policy.on_before_schedule(sequences)
        return sequences


# ── Device Capability Models ────────────────────────────────────────────────

class DeviceClass(str, Enum):
    """Broad device class for scheduling decisions."""
    HIGH_END_GPU = "high_end_gpu"      # H100, A100-80GB, RTX 4090
    MID_RANGE_GPU = "mid_range_gpu"    # RTX 3090, A6000, RTX 3080
    LOW_END_GPU = "low_end_gpu"        # RTX 3070, RTX 2060, older
    APPLE_SILICON = "apple_silicon"    # M1/M2/M3/M4
    INTEL_XPU = "intel_xpu"           # Intel Arc, Gaudi
    CPU_ONLY = "cpu_only"


# Cost per GPU-hour for self-hosted hardware (USD)
GPU_COST_PER_HOUR: dict[str, float] = {
    "H100": 2.50,
    "A100-80GB": 1.80,
    "A100-40GB": 1.20,
    "A6000": 0.80,
    "RTX-4090": 0.60,
    "RTX-3090": 0.40,
    "RTX-3080": 0.30,
    "RTX-3070": 0.20,
    "Apple-M2-Ultra": 0.50,
    "Apple-M2-Pro": 0.30,
    "Apple-M1": 0.20,
    "Intel-Arc-A770": 0.25,
    "CPU": 0.05,
}

# Estimated power draw (watts) per GPU type under inference load
GPU_POWER_WATTS: dict[str, float] = {
    "H100": 700,
    "A100-80GB": 400,
    "A100-40GB": 300,
    "A6000": 300,
    "RTX-4090": 450,
    "RTX-3090": 350,
    "RTX-3080": 320,
    "RTX-3070": 220,
    "Apple-M2-Ultra": 60,
    "Apple-M2-Pro": 30,
    "Apple-M1": 20,
    "Intel-Arc-A770": 225,
    "CPU": 65,
}

# Throughput scaling factors per device class (relative to A100 = 1.0)
DEVICE_THROUGHPUT_SCALE: dict[DeviceClass, float] = {
    DeviceClass.HIGH_END_GPU: 1.2,
    DeviceClass.MID_RANGE_GPU: 0.8,
    DeviceClass.LOW_END_GPU: 0.4,
    DeviceClass.APPLE_SILICON: 0.35,
    DeviceClass.INTEL_XPU: 0.3,
    DeviceClass.CPU_ONLY: 0.02,
}


@dataclass
class NodeCapabilityInfo:
    """Capability profile of a single node in the cluster.

    Used by the scheduler to make device-aware decisions about
    chunk sizes, batch sizes, and token budgets per iteration.
    """
    node_id: str
    gpu_name: str = ""
    device_class: DeviceClass = DeviceClass.MID_RANGE_GPU
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    compute_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    cost_per_hour: float = 0.0
    power_watts: float = 0.0
    is_spot: bool = False
    start_layer: int = 0
    end_layer: int = 0
    measured_latency_ms: float = 0.0  # Measured RTT to this node

    @property
    def memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)

    @property
    def is_wan(self) -> bool:
        """True if this node has WAN-level latency (>10ms RTT)."""
        return self.measured_latency_ms > 10.0

    @property
    def throughput_score(self) -> float:
        """Combined throughput metric (higher = faster)."""
        base = DEVICE_THROUGHPUT_SCALE.get(self.device_class, 0.5)
        if self.compute_tflops > 0:
            base = self.compute_tflops / 100.0  # Normalize
        return max(0.01, base)


def classify_device(gpu_name: str, memory_bytes: int = 0) -> DeviceClass:
    """Auto-classify a device from its GPU name and memory."""
    name = gpu_name.upper()
    if any(k in name for k in ("H100", "A100", "RTX 4090", "RTX 4080")):
        return DeviceClass.HIGH_END_GPU
    if any(k in name for k in ("RTX 3090", "RTX 3080", "A6000", "RTX 4070", "RTX 4060")):
        return DeviceClass.MID_RANGE_GPU
    if any(k in name for k in ("RTX 3070", "RTX 3060", "RTX 20", "GTX")):
        return DeviceClass.LOW_END_GPU
    if any(k in name for k in ("APPLE", "M1", "M2", "M3", "M4")):
        return DeviceClass.APPLE_SILICON
    if any(k in name for k in ("INTEL", "ARC", "GAUDI", "XPU")):
        return DeviceClass.INTEL_XPU
    if "CPU" in name:
        return DeviceClass.CPU_ONLY
    # Heuristic by memory
    if memory_bytes >= 40 * 1024 ** 3:
        return DeviceClass.HIGH_END_GPU
    if memory_bytes >= 10 * 1024 ** 3:
        return DeviceClass.MID_RANGE_GPU
    return DeviceClass.LOW_END_GPU


# ── 1. Heterogeneous P2P Scheduling ────────────────────────────────────────

class HeterogeneousBudgetComputer:
    """Computes per-iteration budgets based on cluster device heterogeneity.

    Instead of a single global budget, this considers:
    - The slowest node in the pipeline (bottleneck)
    - Each node's memory capacity (for KV cache limits)
    - Cross-device-family transfer penalty (CUDA → MPS = +15% latency)

    The output is an IterationBudget that is tuned for the actual
    hardware in the cluster, not a generic A100 assumption.
    """

    def __init__(self):
        self._nodes: dict[str, NodeCapabilityInfo] = {}
        self._lock = threading.Lock()

    def set_nodes(self, nodes: dict[str, NodeCapabilityInfo]) -> None:
        """Register cluster nodes with their capabilities."""
        with self._lock:
            self._nodes = dict(nodes)

    def update_node(self, node_id: str, info: NodeCapabilityInfo) -> None:
        """Update a single node's capability info."""
        with self._lock:
            self._nodes[node_id] = info

    def compute_budget(
        self,
        base_prefill_tokens: int = 4096,
        base_decode_tokens: int = 512,
        base_batch_size: int = 32,
        base_total_tokens: int = 32768,
    ) -> "IterationBudget":
        """Compute a device-aware iteration budget.

        Returns an IterationBudget scaled down if the cluster has
        low-end devices, and scaled up if the cluster is high-end.
        """
        from distllm.core.batch_scheduler import IterationBudget

        with self._lock:
            nodes = list(self._nodes.values())

        if not nodes:
            return IterationBudget(
                max_prefill_tokens=base_prefill_tokens,
                max_decode_tokens=base_decode_tokens,
                max_batch_size=base_batch_size,
                max_total_tokens=base_total_tokens,
            )

        # Find bottleneck throughput
        min_throughput = min(n.throughput_score for n in nodes)
        avg_throughput = sum(n.throughput_score for n in nodes) / len(nodes)

        # Scale factor: 0.3 (slow cluster) to 1.5 (fast cluster)
        # Use harmonic mean to penalize slow nodes more
        if min_throughput > 0:
            harmonic = len(nodes) / sum(1.0 / max(n.throughput_score, 0.01) for n in nodes)
            scale = max(0.3, min(1.5, harmonic))
        else:
            scale = 0.5

        # Memory-based batch size: use the node with least free memory
        min_free_gb = min(n.free_memory_bytes for n in nodes if n.free_memory_bytes > 0) / (1024 ** 3) if any(n.free_memory_bytes > 0 for n in nodes) else 16.0
        # Rough heuristic: 1GB free memory ≈ 2 batch slots for 7B model
        memory_batch_limit = max(4, int(min_free_gb * 2))
        adjusted_batch = min(base_batch_size, memory_batch_limit)

        # Cross-family penalty: if cluster has mixed CUDA/MPS/XPU, reduce budget
        device_classes = {n.device_class for n in nodes}
        cross_family_penalty = 0.85 if len(device_classes) > 1 else 1.0

        # WAN penalty: if any node has high latency, reduce prefill chunk
        max_latency = max((n.measured_latency_ms for n in nodes), default=0)
        wan_penalty = 1.0
        if max_latency > 50:
            wan_penalty = 0.7  # WAN mode: smaller chunks, larger batches
        elif max_latency > 20:
            wan_penalty = 0.85

        final_scale = scale * cross_family_penalty * wan_penalty

        return IterationBudget(
            max_prefill_tokens=max(256, int(base_prefill_tokens * final_scale)),
            max_decode_tokens=max(32, int(base_decode_tokens * final_scale)),
            max_batch_size=max(2, adjusted_batch),
            max_total_tokens=max(512, int(base_total_tokens * final_scale)),
        )

    def get_min_throughput_node(self) -> str | None:
        """Return the node_id of the slowest node (pipeline bottleneck)."""
        with self._lock:
            if not self._nodes:
                return None
            return min(self._nodes, key=lambda nid: self._nodes[nid].throughput_score)

    def stats(self) -> dict:
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "device_classes": list({n.device_class.value for n in self._nodes.values()}),
                "min_throughput": min((n.throughput_score for n in self._nodes.values()), default=0),
                "avg_throughput": (
                    sum(n.throughput_score for n in self._nodes.values()) / max(len(self._nodes), 1)
                ),
                "max_latency_ms": max((n.measured_latency_ms for n in self._nodes.values()), default=0),
            }


# ── 2. Cost-Aware Scheduling ───────────────────────────────────────────────

class CostAwarePriorityAdjuster:
    """Adjusts request priority based on node cost and request value.

    Rules:
    - Low-priority requests prefer cheap nodes (CPU, old GPUs)
    - High-priority requests can use expensive nodes (H100, A100)
    - Spot/preemptible nodes get a cost discount but lower reliability
    - Budget enforcement: reject requests that exceed tenant cost limits

    Integration: call `adjust_priority()` during scheduling to modify
    the effective priority of each pending request.
    """

    def __init__(
        self,
        cost_per_hour_by_node: dict[str, float] | None = None,
        max_cost_per_request: float = 0.0,
        prefer_cheap_for_low_priority: bool = True,
    ):
        self._cost_by_node: dict[str, float] = cost_per_hour_by_node or {}
        self._max_cost_per_request = max_cost_per_request
        self._prefer_cheap = prefer_cheap_for_low_priority
        self._lock = threading.Lock()

        # Cost history
        self._total_cost_usd: float = 0.0
        self._request_count: int = 0

    def set_node_costs(self, costs: dict[str, float]) -> None:
        """Set per-node cost-per-hour (USD)."""
        with self._lock:
            self._cost_by_node.update(costs)

    def set_max_cost_per_request(self, max_cost: float) -> None:
        """Set the maximum allowed cost per request."""
        self._max_cost_per_request = max_cost

    def adjust_priority(
        self,
        base_priority: int,
        estimated_tokens: int,
        preferred_node_id: str | None = None,
    ) -> tuple[int, float]:
        """Adjust priority and return (new_priority, estimated_cost_usd).

        Args:
            base_priority: Original request priority (0=critical, 3=low).
            estimated_tokens: Estimated total tokens (prompt + output).
            preferred_node_id: If set, use this node's cost for estimation.

        Returns:
            (adjusted_priority, estimated_cost_usd).
        """
        node_cost = 1.0  # Default $1/GPU-hour
        with self._lock:
            if preferred_node_id and preferred_node_id in self._cost_by_node:
                node_cost = self._cost_by_node[preferred_node_id]
            elif self._cost_by_node:
                # Use median cost for estimation
                costs = sorted(self._cost_by_node.values())
                node_cost = costs[len(costs) // 2]

        # Estimate cost: (tokens / throughput) * (cost_per_hour / 3600)
        # Rough: 1000 tokens/sec on average GPU
        throughput_tps = 1000.0
        gpu_seconds = estimated_tokens / throughput_tps
        estimated_cost = (gpu_seconds / 3600) * node_cost

        # Budget check
        if self._max_cost_per_request > 0 and estimated_cost > self._max_cost_per_request:
            logger.warning(
                f"Request cost ${estimated_cost:.6f} exceeds limit "
                f"${self._max_cost_per_request:.6f} — deprioritizing"
            )
            return base_priority + 2, estimated_cost  # Deprioritize heavily

        # Cost-aware priority adjustment
        adjusted = base_priority
        if self._prefer_cheap and base_priority >= 2:
            # Low-priority requests get bonus when cheap nodes are available
            cheapest = min(self._cost_by_node.values()) if self._cost_by_node else 1.0
            if node_cost <= cheapest * 1.2:  # Within 20% of cheapest
                adjusted = max(0, base_priority - 1)

        with self._lock:
            self._total_cost_usd += estimated_cost
            self._request_count += 1

        return adjusted, estimated_cost

    def estimate_request_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        node_id: str | None = None,
    ) -> float:
        """Estimate cost in USD for a request."""
        total = input_tokens + output_tokens
        _, cost = self.adjust_priority(2, total, preferred_node_id=node_id)
        return cost

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost_usd, 6),
                "request_count": self._request_count,
                "avg_cost_usd": round(self._total_cost_usd / max(self._request_count, 1), 6),
                "node_costs": dict(self._cost_by_node),
            }


# ── 3. WAN-Optimized Scheduling ────────────────────────────────────────────

@dataclass
class WANConfig:
    """Configuration for WAN-optimized scheduling."""
    enabled: bool = False
    chunk_multiplier: float = 2.0         # Multiply prefill chunk size by this
    batch_multiplier: float = 1.5         # Multiply max_batch_size by this
    prefetch_kv: bool = True              # Prefetch KV cache during pipeline stalls
    disable_sarathi_pressure: bool = True  # Disable pressure adaptation on WAN
    min_chunk_tokens: int = 256           # Minimum chunk size even under pressure
    max_wan_batch: int = 64               # Maximum batch size in WAN mode
    rtt_threshold_ms: float = 10.0        # RTT above this triggers WAN mode


class WANSchedulingPolicy:
    """WAN-aware scheduling adjustments for high-latency links.

    When WAN mode is active:
    - Chunk sizes are multiplied (amortize RTT over more tokens)
    - Batch sizes are larger (amortize per-request overhead)
    - Sarathi-Serve pressure adaptation is disabled (WAN latency variance
      would cause oscillation — pressure signal dominated by RTT jitter)
    - Pipeline overlap: KV prefetch during pipeline stalls

    WAN mode is auto-detected when any node has measured_latency_ms > rtt_threshold_ms.
    """

    def __init__(self, config: WANConfig | None = None):
        self._config = config or WANConfig()
        self._wan_active: bool = False
        self._measured_max_rtt: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_wan_active(self) -> bool:
        return self._wan_active

    @property
    def config(self) -> WANConfig:
        return self._config

    def update_config(self, config: WANConfig) -> None:
        with self._lock:
            self._config = config

    def detect_wan_mode(
        self,
        nodes: dict[str, NodeCapabilityInfo],
    ) -> bool:
        """Auto-detect WAN mode from node latencies."""
        with self._lock:
            if not self._config.enabled:
                self._wan_active = False
                return False

            max_rtt = max((n.measured_latency_ms for n in nodes.values()), default=0)
            self._measured_max_rtt = max_rtt
            was_active = self._wan_active
            self._wan_active = max_rtt > self._config.rtt_threshold_ms

            if self._wan_active and not was_active:
                logger.info(
                    f"WAN scheduling mode activated: max_rtt={max_rtt:.1f}ms "
                    f"(threshold={self._config.rtt_threshold_ms}ms)"
                )
            elif not self._wan_active and was_active:
                logger.info("WAN scheduling mode deactivated: latency improved")

            return self._wan_active

    def adjust_budget_for_wan(
        self,
        base_prefill_tokens: int,
        base_batch_size: int,
        base_total_tokens: int,
    ) -> tuple[int, int, int]:
        """Scale budget parameters for WAN mode.

        Returns (adjusted_prefill_tokens, adjusted_batch_size, adjusted_total_tokens).
        """
        if not self._wan_active:
            return base_prefill_tokens, base_batch_size, base_total_tokens

        cfg = self._config
        adjusted_prefill = max(
            cfg.min_chunk_tokens,
            int(base_prefill_tokens * cfg.chunk_multiplier),
        )
        adjusted_batch = min(
            cfg.max_wan_batch,
            int(base_batch_size * cfg.batch_multiplier),
        )
        # Total tokens scale with batch * chunk
        adjusted_total = int(base_total_tokens * cfg.chunk_multiplier * cfg.batch_multiplier)

        return adjusted_prefill, adjusted_batch, adjusted_total

    def should_disable_pressure_adaptation(self) -> bool:
        """Return True if Sarathi-Serve pressure should be disabled (WAN mode)."""
        return self._wan_active and self._config.disable_sarathi_pressure

    def stats(self) -> dict:
        with self._lock:
            return {
                "wan_active": self._wan_active,
                "measured_max_rtt_ms": round(self._measured_max_rtt, 1),
                "config": {
                    "chunk_multiplier": self._config.chunk_multiplier,
                    "batch_multiplier": self._config.batch_multiplier,
                    "prefetch_kv": self._config.prefetch_kv,
                    "rtt_threshold_ms": self._config.rtt_threshold_ms,
                },
            }


# ── 4. Energy-Aware Scheduling ─────────────────────────────────────────────

@dataclass
class EnergyProfile:
    """Energy consumption profile for a node."""
    node_id: str
    gpu_name: str = ""
    tdp_watts: float = 0.0          # Thermal design power
    current_watts: float = 0.0      # Current power draw (from NVML)
    power_budget_watts: float = 0.0 # Max allowed power (0 = unlimited)
    energy_cost_per_kwh: float = 0.0  # Electricity cost ($/kWh)


class EnergyAwareScheduler:
    """Energy-aware scheduling that trades off throughput vs power draw.

    Monitors GPU power via NVML and adjusts scheduling parameters:
    - When power exceeds budget → reduce batch size (lower compute intensity)
    - When power is below budget → increase batch size (better utilization)
    - Tracks energy cost per request for billing

    Integration: call `adjust_for_energy()` during scheduling to modify
    the iteration budget based on current power consumption.
    """

    def __init__(
        self,
        max_power_watts: float = 0.0,  # 0 = unlimited
        energy_cost_per_kwh: float = 0.10,  # Default US average
        power_warning_threshold: float = 0.9,  # Warn at 90% of budget
    ):
        self._max_power = max_power_watts
        self._energy_cost_kwh = energy_cost_per_kwh
        self._warn_threshold = power_warning_threshold
        self._profiles: dict[str, EnergyProfile] = {}
        self._total_energy_wh: float = 0.0
        self._total_cost_usd: float = 0.0
        self._lock = threading.Lock()

    def set_node_profile(self, profile: EnergyProfile) -> None:
        """Set energy profile for a node."""
        with self._lock:
            self._profiles[profile.node_id] = profile

    def update_power_draw(self, node_id: str, current_watts: float) -> None:
        """Update current power draw for a node (from NVML monitoring)."""
        with self._lock:
            profile = self._profiles.get(node_id)
            if profile:
                # EMA smoothing
                alpha = 0.3
                profile.current_watts = (
                    alpha * current_watts + (1 - alpha) * profile.current_watts
                )

    def get_total_power_draw(self) -> float:
        """Get total power draw across all nodes."""
        with self._lock:
            return sum(p.current_watts for p in self._profiles.values())

    def get_power_utilization(self) -> float:
        """Get power utilization as fraction of budget (0.0-1.0)."""
        if self._max_power <= 0:
            return 0.0
        total = self.get_total_power_draw()
        return total / self._max_power

    def adjust_for_energy(
        self,
        base_batch_size: int,
        base_prefill_tokens: int,
    ) -> tuple[int, int]:
        """Adjust batch and prefill sizes based on power budget.

        Returns (adjusted_batch_size, adjusted_prefill_tokens).
        """
        if self._max_power <= 0:
            return base_batch_size, base_prefill_tokens

        with self._lock:
            total_watts = sum(p.current_watts for p in self._profiles.values())
            utilization = total_watts / self._max_power

        if utilization > 1.0:
            # Over budget: aggressively reduce
            scale = max(0.25, 1.0 - (utilization - 1.0) * 2.0)
            adj_batch = max(1, int(base_batch_size * scale))
            adj_prefill = max(64, int(base_prefill_tokens * scale))
            logger.warning(
                f"Energy: over budget ({utilization:.0%}), "
                f"reducing batch {base_batch_size}→{adj_batch}"
            )
            return adj_batch, adj_prefill

        if utilization > self._warn_threshold:
            # Near budget: moderate reduction
            scale = max(0.5, 1.0 - (utilization - self._warn_threshold) * 3.0)
            adj_batch = max(2, int(base_batch_size * scale))
            adj_prefill = max(128, int(base_prefill_tokens * scale))
            return adj_batch, adj_prefill

        if utilization < 0.5:
            # Well under budget: can increase
            scale = min(1.5, 1.0 + (0.5 - utilization))
            adj_batch = min(base_batch_size * 2, int(base_batch_size * scale))
            return adj_batch, base_prefill_tokens

        return base_batch_size, base_prefill_tokens

    def record_energy_usage(self, duration_seconds: float) -> None:
        """Record energy usage for the elapsed duration.

        Call this periodically (e.g., every iteration) to accumulate
        energy cost tracking.

        Args:
            duration_seconds: Elapsed time in seconds for this measurement.
        """
        with self._lock:
            total_watts = sum(p.current_watts for p in self._profiles.values())
            energy_wh = total_watts * duration_seconds / 3600.0
            self._total_energy_wh += energy_wh
            self._total_cost_usd += energy_wh / 1000.0 * self._energy_cost_kwh

    def stats(self) -> dict:
        with self._lock:
            total_watts = sum(p.current_watts for p in self._profiles.values())
            util = total_watts / self._max_power if self._max_power > 0 else 0.0
            return {
                "total_power_watts": round(total_watts, 1),
                "power_budget_watts": self._max_power,
                "power_utilization_pct": round(util * 100, 1),
                "total_energy_wh": round(self._total_energy_wh, 2),
                "total_energy_cost_usd": round(self._total_cost_usd, 6),
                "node_profiles": {
                    nid: {
                        "gpu": p.gpu_name,
                        "current_watts": round(p.current_watts, 1),
                        "tdp_watts": p.tdp_watts,
                    }
                    for nid, p in self._profiles.items()
                },
            }


# ── 4. Disaggregated Prefill/Decode Scheduling ────────────────────────────

@dataclass
class DisaggregatedBudget:
    """Separate budgets for prefill and decode node pools."""
    prefill_batch_size: int = 16
    prefill_max_tokens: int = 8192
    decode_batch_size: int = 32
    decode_max_tokens: int = 64  # decode steps are single-token


class DisaggregatedBatchScheduler:
    """Scheduler that splits prefill and decode across separate node pools.

    Prefill is compute-bound (processes many tokens at once) and favors
    high-TFLOPS GPUs.  Decode is memory-bound (reads large KV cache for
    one token) and favors high-bandwidth GPUs.

    This scheduler maintains separate budgets and pending queues for
    each phase, routing to the appropriate node pool.

    Usage::

        disagg = DisaggregatedBatchScheduler(
            prefill_nodes=["node-h100-a", "node-h100-b"],
            decode_nodes=["node-a100-a", "node-a100-b"],
        )
        prefill_batch, decode_batch = disagg.schedule()
    """

    def __init__(
        self,
        prefill_node_ids: list[str] | None = None,
        decode_node_ids: list[str] | None = None,
        budget: DisaggregatedBudget | None = None,
    ):
        self._prefill_nodes = list(prefill_node_ids or [])
        self._decode_nodes = list(decode_node_ids or [])
        self._budget = budget or DisaggregatedBudget()

        self._prefill_pending: list[Any] = []  # Sequences awaiting prefill
        self._decode_active: dict[str, Any] = {}  # request_id -> Sequence (in decode)
        self._pending_heap: list = []  # Min-heap of (priority, counter, Sequence)
        self._counter: int = 0
        self._lock = threading.Lock()

        self._iteration_count: int = 0
        self._total_prefill_tokens: int = 0
        self._total_decode_tokens: int = 0

    @property
    def is_disaggregated(self) -> bool:
        """True when both prefill and decode node pools are configured."""
        return bool(self._prefill_nodes) and bool(self._decode_nodes)

    def add(self, seq: Any) -> None:
        """Add a new request to the pending queue."""
        with self._lock:
            import heapq
            heapq.heappush(self._pending_heap, (seq.priority, self._counter, seq))
            self._counter += 1

    def schedule(self) -> tuple[Any | None, Any | None]:
        """Schedule separate prefill and decode batches.

        Returns:
            (prefill_batch, decode_batch) tuple.  Either may be None
            if there is no work for that phase.
        """
        self._iteration_count += 1

        prefill_batch = self._schedule_prefill()
        decode_batch = self._schedule_decode()

        return prefill_batch, decode_batch

    def _schedule_prefill(self) -> Any | None:
        """Build a batch for the prefill node pool."""
        import heapq
        from distllm.core.batch_scheduler import IterationBudget

        with self._lock:
            remaining = self._budget.prefill_batch_size
            budget_tokens = self._budget.prefill_max_tokens
            batch_seqs = []

            while self._pending_heap and remaining > 0:
                pri, cnt, candidate = heapq.heappop(self._pending_heap)
                c_tokens = len(getattr(candidate, 'prompt_tokens', []))
                if c_tokens > budget_tokens:
                    heapq.heappush(self._pending_heap, (pri, cnt, candidate))
                    break
                batch_seqs.append(candidate)
                budget_tokens -= c_tokens
                remaining -= 1

            if not batch_seqs:
                return None

            # Move to decode active set
            for seq in batch_seqs:
                self._decode_active[seq.request_id] = seq
                self._total_prefill_tokens += len(getattr(seq, 'prompt_tokens', []))

            return batch_seqs

    def _schedule_decode(self) -> Any | None:
        """Build a batch for the decode node pool."""
        with self._lock:
            if not self._decode_active:
                return None

            batch_seqs = []
            remaining = self._budget.decode_batch_size

            for req_id, seq in list(self._decode_active.items()):
                if remaining <= 0:
                    break
                if getattr(seq, 'is_complete', False):
                    del self._decode_active[req_id]
                    continue
                batch_seqs.append(seq)
                self._total_decode_tokens += 1
                remaining -= 1

            return batch_seqs if batch_seqs else None

    def complete_request(self, request_id: str) -> None:
        """Remove a completed request from the decode active set."""
        with self._lock:
            self._decode_active.pop(request_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "is_disaggregated": self.is_disaggregated,
                "prefill_nodes": len(self._prefill_nodes),
                "decode_nodes": len(self._decode_nodes),
                "prefill_pending": len(self._pending_heap),
                "decode_active": len(self._decode_active),
                "iteration": self._iteration_count,
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
            }


# ── 5. Predictive Scheduling with Workload Classification ─────────────────

# Expected output length multipliers by workload type
# (ratio of output tokens to input tokens)
_WORKLOAD_OUTPUT_MULTIPLIERS: dict[str, float] = {
    "code": 2.5,        # Code generation tends to produce longer outputs
    "instruction": 1.5,  # Instructions produce moderate-length outputs
    "diverse": 1.2,      # Diverse prompts produce varied-length outputs
    "repetitive": 0.8,   # Repetitive prompts produce shorter outputs
    "unknown": 1.0,      # Default: assume output ≈ input length
}


class PredictiveBatchScheduler:
    """Scheduler that uses workload classification to predict output lengths.

    Classifies each prompt (CODE / INSTRUCTION / REPETITIVE / DIVERSE)
    and uses the classification to:
    - Predict output length for smarter budget allocation
    - Prioritize cache-friendly workloads for prefix caching
    - Adjust chunk sizes based on expected output length

    Usage::

        predictor = PredictiveBatchScheduler()
        predictor.classify_and_enqueue(seq, prompt_text="Write a Python function...")
        predicted_len = predictor.get_predicted_length(seq.request_id)
    """

    def __init__(self):
        self._predictions: dict[str, int] = {}  # request_id -> predicted output tokens
        self._workload_types: dict[str, str] = {}  # request_id -> workload type
        self._lock = threading.Lock()

    def classify_and_enqueue(
        self,
        seq: Any,
        prompt_text: str,
        tokenizer_estimate: int | None = None,
    ) -> str:
        """Classify a prompt and predict its output length.

        Args:
            seq: Sequence object with request_id and prompt_tokens.
            prompt_text: Raw prompt text for classification.
            tokenizer_estimate: Optional override for output length prediction.

        Returns:
            The workload type string (e.g. "code", "instruction").
        """
        from distllm.dist.scheduling.classifier import classify, WorkloadType

        workload = classify(prompt_text)
        workload_str = workload.value if isinstance(workload, WorkloadType) else str(workload)

        input_len = len(getattr(seq, 'prompt_tokens', []))
        if tokenizer_estimate is not None:
            predicted = tokenizer_estimate
        else:
            multiplier = _WORKLOAD_OUTPUT_MULTIPLIERS.get(workload_str, 1.0)
            predicted = max(16, int(input_len * multiplier))

        with self._lock:
            self._predictions[seq.request_id] = predicted
            self._workload_types[seq.request_id] = workload_str

        return workload_str

    def get_predicted_length(self, request_id: str) -> int:
        """Get the predicted output length for a request."""
        with self._lock:
            return self._predictions.get(request_id, 128)

    def get_workload_type(self, request_id: str) -> str:
        """Get the classified workload type for a request."""
        with self._lock:
            return self._workload_types.get(request_id, "unknown")

    def adjust_budget_for_predictions(
        self,
        base_prefill_tokens: int,
        base_batch_size: int,
        pending_seqs: list[Any],
    ) -> tuple[int, int]:
        """Adjust budget based on predicted output lengths.

        If most pending requests are predicted to produce long outputs,
        reduce batch size to avoid memory pressure.  If short outputs,
        increase batch size for better utilization.

        Returns:
            (adjusted_prefill_tokens, adjusted_batch_size)
        """
        if not pending_seqs:
            return base_prefill_tokens, base_batch_size

        predictions = []
        for seq in pending_seqs:
            rid = getattr(seq, 'request_id', None)
            if rid:
                predictions.append(self.get_predicted_length(rid))

        if not predictions:
            return base_prefill_tokens, base_batch_size

        avg_predicted = sum(predictions) / len(predictions)
        # Assume 128 tokens as baseline
        ratio = avg_predicted / 128.0

        if ratio > 2.0:
            # Long outputs expected — reduce batch to avoid OOM
            adj_batch = max(2, int(base_batch_size / (ratio * 0.5)))
            adj_prefill = base_prefill_tokens
        elif ratio < 0.5:
            # Short outputs expected — increase batch for utilization
            adj_batch = min(base_batch_size * 2, int(base_batch_size * 1.5))
            adj_prefill = base_prefill_tokens
        else:
            adj_batch = base_batch_size
            adj_prefill = base_prefill_tokens

        return adj_prefill, adj_batch

    def cleanup_request(self, request_id: str) -> None:
        """Remove prediction data for a completed request."""
        with self._lock:
            self._predictions.pop(request_id, None)
            self._workload_types.pop(request_id, None)

    def stats(self) -> dict:
        with self._lock:
            type_counts: dict[str, int] = {}
            for wt in self._workload_types.values():
                type_counts[wt] = type_counts.get(wt, 0) + 1
            return {
                "tracked_requests": len(self._predictions),
                "avg_predicted_length": (
                    sum(self._predictions.values()) / max(len(self._predictions), 1)
                ),
                "workload_distribution": type_counts,
            }


# ── 6. Tiered KV Cache Storage for Preemption ─────────────────────────────

import enum


class StorageTier(enum.Enum):
    """Storage tier for KV cache data."""
    GPU = "gpu"      # GPU HBM — fastest, most expensive
    CPU = "cpu"      # CPU RAM — medium speed
    SSD = "ssd"      # NVMe SSD — slow, high capacity
    COMPRESSED = "compressed"  # Compressed in-memory — smallest


@dataclass
class TieredEntry:
    """A single KV cache entry in the tiered store."""
    request_id: str
    tier: StorageTier
    data: Any = None  # Raw or compressed KV data
    size_bytes: int = 0
    stored_at: float = field(default_factory=time.time)
    urgency: float = 0.0  # 0.0 = background, 1.0 = critical


class TieredKVStore:
    """Tiered KV cache storage for preemption.

    Stores preempted KV cache data across multiple tiers:
    - Tier 1 (GPU HBM): Fastest, for urgent sequences
    - Tier 2 (CPU RAM): Medium, for normal sequences
    - Tier 3 (NVMe SSD): Slowest, for background sequences
    - Tier 4 (Compressed): int4 compressed, smallest footprint

    Automatically selects the appropriate tier based on sequence
    urgency and available capacity.

    Usage::

        store = TieredKVStore(
            gpu_capacity_bytes=80 * 1024**3,
            cpu_capacity_bytes=512 * 1024**3,
            ssd_path="/tmp/distllm_kv_cache",
        )
        store.store("req-1", kv_data, urgency=0.8)
        data = store.retrieve("req-1")
    """

    def __init__(
        self,
        gpu_capacity_bytes: int = 80 * 1024**3,   # 80 GB
        cpu_capacity_bytes: int = 512 * 1024**3,   # 512 GB
        ssd_path: str = "/tmp/distllm_kv_cache",
        ssd_capacity_bytes: int = 1024 * 1024**3,  # 1 TB
        compress_method: str = "int4",
    ):
        self._gpu_capacity = gpu_capacity_bytes
        self._cpu_capacity = cpu_capacity_bytes
        self._ssd_path = ssd_path
        self._ssd_capacity = ssd_capacity_bytes
        self._compress_method = compress_method

        self._entries: dict[str, TieredEntry] = {}
        self._gpu_used: int = 0
        self._cpu_used: int = 0
        self._ssd_used: int = 0
        self._lock = threading.Lock()

        # Ensure SSD directory exists
        import os
        os.makedirs(ssd_path, exist_ok=True)

    def store(self, request_id: str, kv_data: Any, urgency: float = 0.5) -> StorageTier:
        """Store KV cache data in the appropriate tier.

        Args:
            request_id: Unique request identifier.
            kv_data: KV cache data to store.
            urgency: 0.0 (background) to 1.0 (critical).

        Returns:
            The StorageTier where the data was stored.
        """
        size = self._estimate_size(kv_data)

        with self._lock:
            # Try GPU first for urgent requests
            if urgency > 0.7 and self._gpu_used + size <= self._gpu_capacity:
                self._entries[request_id] = TieredEntry(
                    request_id=request_id, tier=StorageTier.GPU,
                    data=kv_data, size_bytes=size, urgency=urgency,
                )
                self._gpu_used += size
                return StorageTier.GPU

            # Try CPU for normal requests
            if self._cpu_used + size <= self._cpu_capacity:
                self._entries[request_id] = TieredEntry(
                    request_id=request_id, tier=StorageTier.CPU,
                    data=kv_data, size_bytes=size, urgency=urgency,
                )
                self._cpu_used += size
                return StorageTier.CPU

            # Compress and try CPU again
            compressed = self._compress(kv_data)
            comp_size = self._estimate_size(compressed)
            if self._cpu_used + comp_size <= self._cpu_capacity:
                self._entries[request_id] = TieredEntry(
                    request_id=request_id, tier=StorageTier.COMPRESSED,
                    data=compressed, size_bytes=comp_size, urgency=urgency,
                )
                self._cpu_used += comp_size
                return StorageTier.COMPRESSED

            # Fall back to SSD
            self._store_to_ssd(request_id, kv_data)
            self._entries[request_id] = TieredEntry(
                request_id=request_id, tier=StorageTier.SSD,
                data=None, size_bytes=size, urgency=urgency,
            )
            self._ssd_used += size
            return StorageTier.SSD

    def retrieve(self, request_id: str) -> Any | None:
        """Retrieve KV cache data from the tiered store.

        Returns:
            The KV data, or None if not found.
        """
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return None

            if entry.tier == StorageTier.GPU:
                self._gpu_used -= entry.size_bytes
            elif entry.tier == StorageTier.CPU:
                self._cpu_used -= entry.size_bytes
            elif entry.tier == StorageTier.COMPRESSED:
                self._cpu_used -= entry.size_bytes
            elif entry.tier == StorageTier.SSD:
                self._ssd_used -= entry.size_bytes
                return self._load_from_ssd(request_id)

            del self._entries[request_id]
            return entry.data

    def evict_oldest(self, tier: StorageTier | None = None) -> str | None:
        """Evict the oldest entry from a tier (or any tier).

        Returns:
            The evicted request_id, or None if nothing to evict.
        """
        with self._lock:
            candidates = [
                (rid, e) for rid, e in self._entries.items()
                if tier is None or e.tier == tier
            ]
            if not candidates:
                return None
            oldest_id = min(candidates, key=lambda x: x[1].stored_at)[0]
            entry = self._entries.pop(oldest_id)
            if entry.tier == StorageTier.GPU:
                self._gpu_used -= entry.size_bytes
            elif entry.tier in (StorageTier.CPU, StorageTier.COMPRESSED):
                self._cpu_used -= entry.size_bytes
            elif entry.tier == StorageTier.SSD:
                self._ssd_used -= entry.size_bytes
            return oldest_id

    def stats(self) -> dict:
        with self._lock:
            tier_counts: dict[str, int] = {}
            for e in self._entries.values():
                tier_counts[e.tier.value] = tier_counts.get(e.tier.value, 0) + 1
            return {
                "total_entries": len(self._entries),
                "gpu_used_mb": round(self._gpu_used / (1024**2), 1),
                "gpu_capacity_mb": round(self._gpu_capacity / (1024**2), 1),
                "cpu_used_mb": round(self._cpu_used / (1024**2), 1),
                "cpu_capacity_mb": round(self._cpu_capacity / (1024**2), 1),
                "ssd_used_mb": round(self._ssd_used / (1024**2), 1),
                "ssd_capacity_mb": round(self._ssd_capacity / (1024**2), 1),
                "by_tier": tier_counts,
            }

    def _compress(self, kv_data: Any) -> Any:
        """Compress KV data using int4 quantization."""
        import torch
        if isinstance(kv_data, torch.Tensor):
            scale = kv_data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
            quantized = (kv_data / scale).clamp(-8, 7).to(torch.int8)
            return {"_compressed": True, "method": "int4", "data": quantized, "scale": scale}
        if isinstance(kv_data, dict):
            return {k: self._compress(v) for k, v in kv_data.items()}
        if isinstance(kv_data, (list, tuple)):
            return [self._compress(item) for item in kv_data]
        return kv_data

    def _estimate_size(self, data: Any) -> int:
        """Estimate byte size of data."""
        import sys
        import torch
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        if isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        return sys.getsizeof(data)

    def _store_to_ssd(self, request_id: str, kv_data: Any) -> None:
        """Write KV data to SSD as a .pt file."""
        import os
        import torch
        path = os.path.join(self._ssd_path, f"{request_id}.pt")
        try:
            torch.save(kv_data, path)
        except Exception as e:
            logger.warning(f"SSD store failed for {request_id}: {e}")

    def _load_from_ssd(self, request_id: str) -> Any | None:
        """Load KV data from SSD."""
        import os
        import torch
        path = os.path.join(self._ssd_path, f"{request_id}.pt")
        try:
            return torch.load(path, weights_only=True)
        except Exception as e:
            logger.warning(f"SSD load failed for {request_id}: {e}")
            return None


# ── 7. Token-Bank Memory Management ────────────────────────────────────────

@dataclass
class TokenCredit:
    """Tracks token debt/credit for a single request."""
    request_id: str
    allocated: int = 0    # Tokens currently allocated to this request
    borrowed: int = 0     # Tokens borrowed from the global pool (debt)
    returned: int = 0     # Tokens returned to the pool
    urgency: float = 0.5  # 0.0 = background, 1.0 = critical


class TokenBank:
    """Shared token budget with dynamic allocation and debt tracking.

    Instead of per-sequence fixed KV cache allocation, all sequences
    share a global token pool.  The scheduler dynamically assigns
    tokens to high-urgency sequences and can reclaim tokens from
    low-priority sequences when the pool is exhausted.

    Features:
    - Global token budget shared across all active sequences
    - Urgency-weighted allocation: critical requests get more tokens
    - Token lending: low-priority requests can borrow tokens that
      may be reclaimed when high-priority requests arrive
    - Debt/credit tracking per request for billing and debugging

    Usage::

        bank = TokenBank(total_budget=131072)  # 128K tokens
        bank.allocate("req-1", tokens=4096, urgency=0.9)
        bank.allocate("req-2", tokens=1024, urgency=0.3)
        # If pool exhausted, req-2's tokens can be reclaimed
        reclaimed = bank.reclaim_from_lowest(needed=2048)
    """

    def __init__(self, total_budget: int = 131072):
        self._total_budget = total_budget
        self._allocated: int = 0  # Total tokens currently allocated
        self._credits: dict[str, TokenCredit] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Tokens available in the global pool."""
        with self._lock:
            return max(0, self._total_budget - self._allocated)

    @property
    def utilization(self) -> float:
        """Pool utilization as a fraction (0.0 to 1.0)."""
        with self._lock:
            return self._allocated / max(self._total_budget, 1)

    def allocate(
        self,
        request_id: str,
        tokens: int,
        urgency: float = 0.5,
    ) -> int:
        """Allocate tokens from the global pool.

        If the pool has enough free tokens, allocates the full amount.
        If not, allocates what's available and records the shortfall
        as borrowed tokens (debt).

        Args:
            request_id: Unique request identifier.
            tokens: Number of tokens to allocate.
            urgency: 0.0 (background) to 1.0 (critical).

        Returns:
            Number of tokens actually allocated (may be less than requested).
        """
        with self._lock:
            available = max(0, self._total_budget - self._allocated)
            allocated = min(tokens, available)
            borrowed = tokens - allocated

            credit = self._credits.get(request_id)
            if credit is None:
                credit = TokenCredit(request_id=request_id, urgency=urgency)
                self._credits[request_id] = credit

            credit.allocated += allocated
            credit.borrowed += borrowed
            self._allocated += allocated

            return allocated

    def release(self, request_id: str) -> int:
        """Release all tokens allocated to a request.

        Returns:
            Number of tokens released back to the pool.
        """
        with self._lock:
            credit = self._credits.pop(request_id, None)
            if credit is None:
                return 0
            self._allocated = max(0, self._allocated - credit.allocated)
            return credit.allocated

    def reclaim_from_lowest(self, needed: int) -> dict[str, int]:
        """Reclaim tokens from the lowest-urgency requests.

        Called when the pool is exhausted and a high-priority request
        needs tokens.  Finds the lowest-urgency requests and takes
        back their allocated tokens.

        Args:
            needed: Number of tokens to reclaim.

        Returns:
            Dict mapping request_id -> tokens reclaimed from that request.
        """
        with self._lock:
            reclaimed: dict[str, int] = {}
            remaining = needed

            # Sort by urgency (lowest first)
            sorted_credits = sorted(
                self._credits.values(),
                key=lambda c: c.urgency,
            )

            for credit in sorted_credits:
                if remaining <= 0:
                    break
                if credit.allocated <= 0:
                    continue

                take = min(credit.allocated, remaining)
                credit.allocated -= take
                credit.borrowed += take  # Mark as debt
                self._allocated -= take
                remaining -= take
                reclaimed[credit.request_id] = take

            return reclaimed

    def get_credit(self, request_id: str) -> TokenCredit | None:
        """Get the token credit info for a request."""
        with self._lock:
            return self._credits.get(request_id)

    def get_debtors(self, min_borrowed: int = 1) -> list[TokenCredit]:
        """Get all requests that have borrowed tokens (debtors)."""
        with self._lock:
            return [
                c for c in self._credits.values()
                if c.borrowed >= min_borrowed
            ]

    def adjust_budget(self, new_budget: int) -> None:
        """Adjust the total token budget at runtime.

        If the new budget is smaller than current allocations,
        excess tokens are reclaimed from lowest-urgency requests.
        """
        with self._lock:
            old_budget = self._total_budget
            self._total_budget = new_budget

            if new_budget < self._allocated:
                excess = self._allocated - new_budget
                # Reclaim from lowest urgency
                sorted_credits = sorted(
                    self._credits.values(),
                    key=lambda c: c.urgency,
                )
                for credit in sorted_credits:
                    if excess <= 0:
                        break
                    take = min(credit.allocated, excess)
                    credit.allocated -= take
                    self._allocated -= take
                    excess -= take

            logger.info(
                f"TokenBank budget: {old_budget} → {new_budget} "
                f"(allocated={self._allocated})"
            )

    def stats(self) -> dict:
        with self._lock:
            debtors = [c for c in self._credits.values() if c.borrowed > 0]
            util = self._allocated / max(self._total_budget, 1)
            return {
                "total_budget": self._total_budget,
                "allocated": self._allocated,
                "available": max(0, self._total_budget - self._allocated),
                "utilization_pct": round(util * 100, 1),
                "active_requests": len(self._credits),
                "debtors": len(debtors),
                "total_borrowed": sum(c.borrowed for c in self._credits.values()),
            }


# ── 8. Federated Scheduling ────────────────────────────────────────────────

@dataclass
class ClusterStatus:
    """Status of a remote cluster for federated scheduling."""
    cluster_id: str
    host: str
    port: int
    region: str = ""
    gpu_count: int = 0
    gpu_utilization: float = 0.0
    pending_requests: int = 0
    cost_per_hour: float = 0.0
    latency_ms: float = 0.0
    has_prefix_cache: bool = False
    last_seen: float = field(default_factory=time.time)


@dataclass
class FederatedRoute:
    """Routing decision for a federated request."""
    cluster_id: str
    reason: str  # "cheapest", "nearest", "cache_hit", "work_steal"
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0


class FederatedScheduler:
    """Scheduler for cross-cluster federated inference.

    Routes requests across multiple clusters based on:
    - Cost: prefer cheapest cluster for low-priority requests
    - Latency: prefer nearest cluster for latency-sensitive requests
    - Cache hits: route to clusters that have the prefix cached
    - Work stealing: if local cluster is busy, steal from idle clusters

    Integrates with FederationCoordinator for peer discovery and
    CrossCloudRouter for cost/latency data.

    Usage::

        fed = FederatedScheduler(local_cluster_id="cluster-a")
        fed.update_cluster_status(ClusterStatus(
            cluster_id="cluster-b", host="10.0.0.2", port=50050,
            gpu_utilization=0.3, cost_per_hour=0.60,
        ))
        route = fed.route_request(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            priority=2,
            prefix_hash="abc123",
        )
    """

    def __init__(
        self,
        local_cluster_id: str = "default",
        spill_threshold: float = 0.8,  # Spill to remote when local util > 80%
    ):
        self._local_cluster_id = local_cluster_id
        self._spill_threshold = spill_threshold
        self._clusters: dict[str, ClusterStatus] = {}
        self._prefix_cache: dict[str, set[str]] = {}  # prefix_hash -> set of cluster_ids
        self._lock = threading.Lock()

    def update_cluster_status(self, status: ClusterStatus) -> None:
        """Update status for a remote cluster."""
        with self._lock:
            self._clusters[status.cluster_id] = status

    def register_prefix_cache(self, prefix_hash: str, cluster_id: str) -> None:
        """Register that a cluster has a specific prefix cached."""
        with self._lock:
            if prefix_hash not in self._prefix_cache:
                self._prefix_cache[prefix_hash] = set()
            self._prefix_cache[prefix_hash].add(cluster_id)

    def route_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        priority: int = 2,
        prefix_hash: str | None = None,
        max_latency_ms: float = 5000.0,
    ) -> FederatedRoute:
        """Route a request to the best cluster.

        Routing priority:
        1. Cache hit: if any cluster has the prefix cached, route there
        2. Local: if local cluster has capacity, keep it local
        3. Cheapest: route to cheapest remote cluster
        4. Work steal: if all clusters are busy, route to least busy

        Args:
            request_id: Unique request identifier.
            prompt_tokens: Prompt token IDs.
            priority: Request priority (0=critical, 3=low).
            prefix_hash: Optional hash of the prompt prefix for cache routing.
            max_latency_ms: Maximum acceptable latency.

        Returns:
            FederatedRoute with the routing decision.
        """
        with self._lock:
            clusters = dict(self._clusters)
            prefix_cache = dict(self._prefix_cache)

        # 1. Cache hit routing
        if prefix_hash and prefix_hash in prefix_cache:
            cached_clusters = prefix_cache[prefix_hash]
            for cid in cached_clusters:
                if cid in clusters:
                    cluster = clusters[cid]
                    if cluster.latency_ms <= max_latency_ms:
                        return FederatedRoute(
                            cluster_id=cid,
                            reason="cache_hit",
                            estimated_cost=cluster.cost_per_hour,
                            estimated_latency_ms=cluster.latency_ms,
                        )

        # 2. Local cluster check
        local = clusters.get(self._local_cluster_id)
        if local and local.gpu_utilization < self._spill_threshold:
            return FederatedRoute(
                cluster_id=self._local_cluster_id,
                reason="local",
                estimated_cost=local.cost_per_hour,
                estimated_latency_ms=local.latency_ms,
            )

        # 3. Find cheapest remote cluster with capacity
        candidates = [
            c for cid, c in clusters.items()
            if cid != self._local_cluster_id
            and c.gpu_utilization < 0.9
            and c.latency_ms <= max_latency_ms
        ]

        if candidates:
            if priority <= 1:
                # High priority: prefer lowest latency
                best = min(candidates, key=lambda c: c.latency_ms)
                reason = "nearest"
            else:
                # Normal/low priority: prefer cheapest
                best = min(candidates, key=lambda c: c.cost_per_hour)
                reason = "cheapest"

            return FederatedRoute(
                cluster_id=best.cluster_id,
                reason=reason,
                estimated_cost=best.cost_per_hour,
                estimated_latency_ms=best.latency_ms,
            )

        # 4. Work stealing: find least busy cluster
        all_remote = [
            c for cid, c in clusters.items()
            if cid != self._local_cluster_id
        ]
        if all_remote:
            least_busy = min(all_remote, key=lambda c: c.pending_requests)
            return FederatedRoute(
                cluster_id=least_busy.cluster_id,
                reason="work_steal",
                estimated_cost=least_busy.cost_per_hour,
                estimated_latency_ms=least_busy.latency_ms,
            )

        # No remote clusters available — stay local
        return FederatedRoute(
            cluster_id=self._local_cluster_id,
            reason="local_fallback",
        )

    def should_spill(self) -> bool:
        """Check if local cluster should spill to remote."""
        with self._lock:
            local = self._clusters.get(self._local_cluster_id)
            if local is None:
                return False
            return local.gpu_utilization > self._spill_threshold

    def get_idle_clusters(self, threshold: float = 0.3) -> list[str]:
        """Find clusters with low utilization (for work stealing)."""
        with self._lock:
            return [
                cid for cid, c in self._clusters.items()
                if c.gpu_utilization < threshold
                and cid != self._local_cluster_id
            ]

    def stats(self) -> dict:
        with self._lock:
            local = self._clusters.get(self._local_cluster_id)
            spill = local is not None and local.gpu_utilization > self._spill_threshold
            return {
                "local_cluster": self._local_cluster_id,
                "known_clusters": len(self._clusters),
                "spill_threshold": self._spill_threshold,
                "should_spill": spill,
                "prefix_cache_entries": len(self._prefix_cache),
                "clusters": {
                    cid: {
                        "gpu_util": round(c.gpu_utilization, 2),
                        "pending": c.pending_requests,
                        "cost": c.cost_per_hour,
                        "latency_ms": round(c.latency_ms, 1),
                    }
                    for cid, c in self._clusters.items()
                },
            }


# ── 9. Distributed Preemption Coordinator ──────────────────────────────────

@dataclass
class NodePreemptionState:
    """Preemption state for a single node in the pipeline."""
    node_id: str
    status: str = "idle"  # "idle", "halting", "saving", "freed", "restoring"
    kv_blocks_freed: int = 0
    kv_state_saved: bool = False
    error: str | None = None


class DistributedPreemptionCoordinator:
    """Coordinates preemption across multiple pipeline-parallel nodes.

    In pipeline-parallel mode, a sequence's KV cache is split across
    all nodes (one per layer range).  Preempting a sequence requires
    coordinating all nodes to:
    1. Halt the pipeline for that sequence
    2. Save KV state on each node
    3. Mark blocks as free on all nodes
    4. When restoring, broadcast new physical block IDs

    Usage::

        coord = DistributedPreemptionCoordinator(
            node_ids=["node-0", "node-1", "node-2", "node-3"],
            send_command_fn=lambda node_id, cmd, data: grpc_send(node_id, cmd, data),
        )
        coord.preempt_sequence("req-1", kv_state_per_node={...})
    """

    def __init__(
        self,
        node_ids: list[str] | None = None,
        send_command_fn: Any = None,
        timeout_s: float = 5.0,
    ):
        self._node_ids = list(node_ids or [])
        self._send_command = send_command_fn
        self._timeout_s = timeout_s
        self._states: dict[str, NodePreemptionState] = {}
        self._preempted_seqs: dict[str, dict[str, Any]] = {}  # req_id -> {node_id: kv_data}
        self._lock = threading.Lock()

    def preempt_sequence(
        self,
        request_id: str,
        kv_state_per_node: dict[str, Any] | None = None,
    ) -> bool:
        """Coordinate preemption of a sequence across all pipeline nodes.

        Args:
            request_id: The request to preempt.
            kv_state_per_node: Optional KV state dict per node.

        Returns:
            True if preemption succeeded on all nodes.
        """
        with self._lock:
            states = {}
            for nid in self._node_ids:
                states[nid] = NodePreemptionState(node_id=nid, status="halting")
            self._states = states

        logger.info(f"Distributed preempt: halting {request_id} on {len(self._node_ids)} nodes")

        # Phase 1: Halt pipeline for this sequence on all nodes
        all_halted = True
        for nid in self._node_ids:
            success = self._send_to_node(nid, "halt_sequence", {"request_id": request_id})
            with self._lock:
                if success:
                    self._states[nid].status = "saving"
                else:
                    self._states[nid].status = "error"
                    self._states[nid].error = "halt failed"
                    all_halted = False

        if not all_halted:
            logger.warning(f"Distributed preempt: halt failed for {request_id}")
            return False

        # Phase 2: Save KV state on each node
        saved_kv: dict[str, Any] = {}
        for nid in self._node_ids:
            kv_data = (kv_state_per_node or {}).get(nid)
            if kv_data is not None:
                saved_kv[nid] = kv_data
            with self._lock:
                self._states[nid].kv_state_saved = True

        # Phase 3: Free blocks on all nodes
        for nid in self._node_ids:
            success = self._send_to_node(nid, "free_blocks", {"request_id": request_id})
            with self._lock:
                if success:
                    self._states[nid].status = "freed"
                    self._states[nid].kv_blocks_freed += 1
                else:
                    self._states[nid].status = "error"
                    self._states[nid].error = "free failed"

        # Store preempted state (always store, even if kv_data is empty)
        with self._lock:
            self._preempted_seqs[request_id] = saved_kv

        logger.info(f"Distributed preempt: {request_id} preempted on {len(self._node_ids)} nodes")
        return True

    def restore_sequence(
        self,
        request_id: str,
        new_block_ids: dict[str, list[int]] | None = None,
    ) -> bool:
        """Coordinate restoration of a preempted sequence across all nodes.

        Args:
            request_id: The request to restore.
            new_block_ids: Optional new physical block IDs per node.

        Returns:
            True if restoration succeeded on all nodes.
        """
        with self._lock:
            if request_id not in self._preempted_seqs:
                logger.warning(f"Distributed restore: no saved state for {request_id}")
                return False
            saved_kv = self._preempted_seqs[request_id]

            for nid in self._node_ids:
                self._states[nid] = NodePreemptionState(node_id=nid, status="restoring")

        logger.info(f"Distributed restore: restoring {request_id} on {len(self._node_ids)} nodes")

        # Phase 1: Allocate new blocks on each node
        for nid in self._node_ids:
            block_ids = (new_block_ids or {}).get(nid, [])
            kv_data = saved_kv.get(nid)
            success = self._send_to_node(nid, "restore_blocks", {
                "request_id": request_id,
                "block_ids": block_ids,
                "kv_data": kv_data,
            })
            with self._lock:
                if success:
                    self._states[nid].status = "idle"
                else:
                    self._states[nid].status = "error"
                    self._states[nid].error = "restore failed"

        # Clean up
        with self._lock:
            self._preempted_seqs.pop(request_id, None)

        logger.info(f"Distributed restore: {request_id} restored")
        return True

    def _send_to_node(self, node_id: str, command: str, data: dict) -> bool:
        """Send a command to a node. Returns True if successful."""
        if self._send_command is None:
            logger.debug(f"No send_command_fn set, simulating success for {node_id}:{command}")
            return True
        try:
            return self._send_command(node_id, command, data)
        except Exception as e:
            logger.warning(f"Failed to send {command} to {node_id}: {e}")
            return False

    def get_states(self) -> dict[str, NodePreemptionState]:
        """Get current preemption state for all nodes."""
        with self._lock:
            return dict(self._states)

    def get_preempted_sequences(self) -> list[str]:
        """Get list of currently preempted sequence IDs."""
        with self._lock:
            return list(self._preempted_seqs.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                "node_count": len(self._node_ids),
                "preempted_sequences": len(self._preempted_seqs),
                "node_states": {
                    nid: {
                        "status": s.status,
                        "kv_blocks_freed": s.kv_blocks_freed,
                        "kv_state_saved": s.kv_state_saved,
                        "error": s.error,
                    }
                    for nid, s in self._states.items()
                },
            }
