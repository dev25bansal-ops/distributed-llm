"""Dynamic pipeline rebalancer with straggler detection and auto-mitigation.

V2 adds:
- Auto-migration: dynamically reassigns layers from stragglers to faster nodes
- Batch size adjustment: reduces batch size for slow nodes
- Grace period: tolerates transient slowdowns before triggering rebalance
"""


from __future__ import annotations
import statistics
import time
import threading
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from distllm.dist.latency import LatencyTracker
from distllm.config.settings import RebalancerSettings


@dataclass
class PartitionRecommendation:
    """Recommended layer partition for a node."""

    node_id: str
    start_layer: int
    end_layer: int


@dataclass
class StragglerAction:
    """Recommended mitigation action for a straggler node."""

    node_id: str
    action: str
    layer_count_change: int = 0
    batch_size_reduction: int = 0
    reason: str = ""


class Rebalancer:
    """Detects straggler nodes and recommends or auto-executes layer redistribution.


    Supports two modes:
    - Advisory: logs recommendations only (on_reassign=None)
    - Auto: executes redistribution via on_reassign callback (on_reassign set)

    Auto-rebalancing triggers when:
    1. A node's latency exceeds straggler_threshold * average
    2. The node has been a straggler for grace_period_steps consecutive checks
    3. auto_mitigate is enabled in settings
    """


    def __init__(
        self,
        latency_tracker: LatencyTracker,
        settings: RebalancerSettings,
        on_reassign: Callable | None = None,
    ):
        self._tracker = latency_tracker
        self._settings = settings
        self._last_rebalance_time: float = 0.0
        self._current_partition: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()
        self._on_reassign = on_reassign

        self._straggler_history: dict[str, int] = {}
        self._grace_period_steps: int = settings.grace_period_steps

        self._auto_mitigate_enabled: bool = settings.auto_mitigate
        self._batch_size_adjustments: dict[str, int] = {}

    def detect_stragglers(self, threshold: float | None = None) -> list[str]:
        threshold = threshold or self._settings.straggler_threshold
        all_avg = self._tracker.get_all_avg()
        if len(all_avg) < 2:
            return []
        latencies = list(all_avg.values())
        median = statistics.median(latencies)
        if median == 0:
            return []
        stragglers = [
            node_id for node_id, avg in all_avg.items()
            if avg > threshold * median
        ]

        # M-06: Protect _straggler_history with lock
        with self._lock:
            result = []
            for node_id in stragglers:
                self._straggler_history[node_id] = self._straggler_history.get(node_id, 0) + 1
                if self._straggler_history[node_id] >= self._grace_period_steps:
                    result.append(node_id)
            for node_id in list(self._straggler_history.keys()):
                if node_id not in stragglers:
                    del self._straggler_history[node_id]
        return result

    def compute_new_partition(
        self,
        total_layers: int,
        node_latencies: dict[str, float],
    ) -> list[PartitionRecommendation]:
        if not node_latencies or total_layers <= 0:
            return []

        inverse = {nid: 1.0 / lat for nid, lat in node_latencies.items() if lat > 0}
        total_inverse = sum(inverse.values())
        if total_inverse == 0:
            return []

        recommendations = []
        remaining_layers = total_layers
        sorted_nodes = sorted(inverse.keys(), key=lambda n: inverse[n], reverse=True)
        start = 0

        for i, node_id in enumerate(sorted_nodes):
            if i == len(sorted_nodes) - 1:
                layers = remaining_layers
            else:
                share = inverse[node_id] / total_inverse
                layers = max(1, int(share * total_layers))
                remaining_layers -= layers

            end = start + layers - 1
            recommendations.append(PartitionRecommendation(node_id, start, end))
            start = end + 1

        return recommendations

    def compute_mitigation_actions(self, stragglers: list[str]) -> list[StragglerAction]:
        if not stragglers:
            return []

        all_avg = self._tracker.get_all_avg()
        if len(all_avg) < 2:
            return []

        actions = []
        latencies = list(all_avg.values())
        median = statistics.median(latencies)

        for node_id in stragglers:
            avg_lat = all_avg.get(node_id, 0)
            if avg_lat <= median:
                continue

            slowdown_ratio = avg_lat / max(median, 0.001)

            if slowdown_ratio > 2.0 and self._auto_mitigate_enabled:
                actions.append(StragglerAction(
                    node_id=node_id,
                    action="reassign",
                    layer_count_change=-1,
                    reason=f"slowdown {slowdown_ratio:.1f}x > 2.0x, reassigning 1 layer",
                ))
            elif slowdown_ratio > 1.5:
                current_reduction = self._batch_size_adjustments.get(node_id, 0)
                new_reduction = min(current_reduction + 1, 4)
                self._batch_size_adjustments[node_id] = new_reduction
                actions.append(StragglerAction(
                    node_id=node_id,
                    action="reduce_batch",
                    batch_size_reduction=new_reduction,
                    reason=f"slowdown {slowdown_ratio:.1f}x, reducing batch by {new_reduction * 25}%",
                ))
            else:
                actions.append(StragglerAction(
                    node_id=node_id,
                    action="none",
                    reason=f"slowdown {slowdown_ratio:.1f}x within tolerance",
                ))

        return actions

    def apply_mitigation_actions(self, actions: list[StragglerAction]) -> None:
        for action in actions:
            if action.action == "reassign" and self._on_reassign is not None:
                try:
                    self._on_reassign(action.node_id, action.layer_count_change)
                    logger.info(f"Auto-mitigation: reassigned layers for {action.node_id}")
                except Exception as e:
                    logger.error(f"Auto-migration failed for {action.node_id}: {e}")

            elif action.action == "reduce_batch":
                logger.info(f"Auto-mitigation: reducing batch for {action.node_id} "
                           f"by {action.batch_size_reduction * 25}%")

    def get_batch_size_adjustment(self, node_id: str) -> float:
        reduction = self._batch_size_adjustments.get(node_id, 0)
        return max(0.25, 1.0 - reduction * 0.25)

    def clear_batch_adjustments(self, node_id: str) -> None:
        self._batch_size_adjustments.pop(node_id, None)

    def should_rebalance(self) -> tuple[bool, str]:
        if not self._settings.enabled:
            return False, "rebalancer disabled"

        now = time.time()
        cooldown = self._settings.cooldown_seconds
        if now - self._last_rebalance_time < cooldown:
            remaining = cooldown - (now - self._last_rebalance_time)
            return False, f"cooldown active ({remaining:.0f}s remaining)"

        stragglers = self.detect_stragglers()
        if not stragglers:
            return False, "no stragglers detected"

        all_avg = self._tracker.get_all_avg()
        if len(all_avg) < 2:
            return False, "insufficient latency data"

        latencies = list(all_avg.values())
        median = statistics.median(latencies)
        max_latency = max(latencies)
        improvement_pct = ((max_latency - median) / median) * 100 if median > 0 else 0

        if improvement_pct < self._settings.min_improvement_pct:
            return False, f"improvement {improvement_pct:.1f}% below threshold {self._settings.min_improvement_pct}%"

        return True, f"stragglers={stragglers}, improvement={improvement_pct:.1f}%"

    def record_rebalance(self) -> None:
        with self._lock:
            self._last_rebalance_time = time.time()

    def set_current_partition(self, partition: list[tuple[str, int, int]]) -> None:
        with self._lock:
            self._current_partition = partition
