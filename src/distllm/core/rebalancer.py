"""Dynamic pipeline rebalancer with straggler detection and auto-mitigation.

V2 adds:
- Auto-migration: dynamically reassigns layers from stragglers to faster nodes
- Batch size adjustment: reduces batch size for slow nodes
- Grace period: tolerates transient slowdowns before triggering rebalance
"""

import statistics
import time
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from distllm.core.latency_tracker import LatencyTracker
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
    action: str  # "reassign", "reduce_batch", "none"
    layer_count_change: int = 0
    batch_size_reduction: int = 0
    reason: str = ""


class Rebalancer:
    """Detects straggler nodes and recommends layer redistribution.

    V1: advisory only — logs recommendations, does not auto-migrate.
    V2: supports auto-migration via on_reassign callback.
    """

    def __init__(
        self,
        latency_tracker: LatencyTracker,
        settings: RebalancerSettings,
        on_reassign: Optional[Callable] = None,
    ):
        self._tracker = latency_tracker
        self._settings = settings
        self._last_rebalance_time: float = 0.0
        self._current_partition: List[Tuple[str, int, int]] = []
        self._lock = threading.Lock()
        self._on_reassign = on_reassign

        # Grace period tracking
        self._straggler_history: Dict[str, int] = {}  # node_id -> consecutive detections
        self._grace_period_steps: int = getattr(settings, 'grace_period_steps', 3)

        # Auto-mitigation state
        self._auto_mitigate_enabled: bool = getattr(settings, 'auto_mitigate', False)
        self._batch_size_adjustments: Dict[str, int] = {}

    def detect_stragglers(self, threshold: Optional[float] = None) -> List[str]:
        """Return nodes with avg latency > threshold * median latency."""
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

        # Grace period: require consecutive detections
        result = []
        for node_id in stragglers:
            self._straggler_history[node_id] = self._straggler_history.get(node_id, 0) + 1
            if self._straggler_history[node_id] >= self._grace_period_steps:
                result.append(node_id)
        for node_id in list(self._straggler_history.keys()):
            if node_id not in stragglers:
                # Fast decay: reset to 0 immediately on clean cycle
                del self._straggler_history[node_id]

        return result

    def compute_new_partition(
        self,
        total_layers: int,
        node_latencies: Dict[str, float],
    ) -> List[PartitionRecommendation]:
        """Redistribute layers proportional to inverse latency.

        Faster nodes (lower latency) get more layers.
        """
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

    def compute_mitigation_actions(self, stragglers: List[str]) -> List[StragglerAction]:
        """Compute mitigation actions for detected stragglers.

        Each straggler gets one of:
        - reassign: move 1 layer to a faster node
        - reduce_batch: cut batch size by 25%
        - none: no action needed

        Returns:
            List of StragglerAction for each straggler.
        """
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

    def apply_mitigation_actions(self, actions: List[StragglerAction]) -> None:
        """Apply mitigation actions to the pipeline.

        For auto-migration: calls the on_reassign callback if configured.
        For batch reduction: records the adjustment (applied by pipeline).
        """
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
        """Get batch size multiplier for a node (1.0 = full, 0.5 = half, etc.)."""
        reduction = self._batch_size_adjustments.get(node_id, 0)
        return max(0.25, 1.0 - reduction * 0.25)

    def clear_batch_adjustments(self, node_id: str) -> None:
        """Reset batch size adjustment for a node (node recovered)."""
        self._batch_size_adjustments.pop(node_id, None)

    def should_rebalance(self) -> Tuple[bool, str]:
        """Check if rebalancing is warranted.

        Returns (should_rebalance, reason).
        """
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
        """Mark that a rebalance occurred (updates cooldown timer)."""
        with self._lock:
            self._last_rebalance_time = time.time()

    def set_current_partition(self, partition: List[Tuple[str, int, int]]) -> None:
        """Store the current partition for reference."""
        with self._lock:
            self._current_partition = partition
