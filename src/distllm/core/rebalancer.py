"""Dynamic pipeline rebalancer with straggler detection."""

import statistics
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from distllm.core.latency_tracker import LatencyTracker
from distllm.config.settings import RebalancerSettings


@dataclass
class PartitionRecommendation:
    """Recommended layer partition for a node."""
    node_id: str
    start_layer: int
    end_layer: int


class Rebalancer:
    """Detects straggler nodes and recommends layer redistribution.

    V1: advisory only — logs recommendations, does not auto-migrate.
    """

    def __init__(
        self,
        latency_tracker: LatencyTracker,
        settings: RebalancerSettings,
    ):
        self._tracker = latency_tracker
        self._settings = settings
        self._last_rebalance_time: float = 0.0
        self._current_partition: List[Tuple[str, int, int]] = []
        self._lock = threading.Lock()

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
        return [
            node_id for node_id, avg in all_avg.items()
            if avg > threshold * median
        ]

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

        # Inverse latency weighting
        inverse = {nid: 1.0 / lat for nid, lat in node_latencies.items() if lat > 0}
        total_inverse = sum(inverse.values())
        if total_inverse == 0:
            return []

        recommendations = []
        remaining_layers = total_layers
        sorted_nodes = sorted(inverse.keys())
        start = 0

        for i, node_id in enumerate(sorted_nodes):
            if i == len(sorted_nodes) - 1:
                # Last node gets remaining layers
                layers = remaining_layers
            else:
                share = inverse[node_id] / total_inverse
                layers = max(1, int(share * total_layers))
                remaining_layers -= layers

            end = start + layers - 1
            recommendations.append(PartitionRecommendation(node_id, start, end))
            start = end + 1

        return recommendations

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
