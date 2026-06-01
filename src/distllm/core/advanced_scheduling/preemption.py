"""Distributed preemption coordination."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodePreemptionState:
    """Preemption state for a single node."""
    node_id: str
    is_preempted: bool = False
    preempted_at: float = 0.0
    reason: str = ""
    estimated_resume_time: float = 0.0


class DistributedPreemptionCoordinator:
    """Coordinates preemption decisions across multiple nodes."""

    def __init__(self, max_preempted_fraction: float = 0.3):
        self._max_preempted_fraction = max_preempted_fraction
        self._states: dict[str, NodePreemptionState] = {}
        self._lock = threading.Lock()

    def update_state(self, state: NodePreemptionState) -> None:
        with self._lock:
            self._states[state.node_id] = state

    def should_preempt(self, node_id: str) -> bool:
        with self._lock:
            total = len(self._states)
            preempted = sum(1 for s in self._states.values() if s.is_preempted)
            if total == 0:
                return False
            return (preempted / total) < self._max_preempted_fraction

    def get_preempted_nodes(self) -> list[str]:
        with self._lock:
            return [nid for nid, s in self._states.items() if s.is_preempted]
