"""Health check state tracking for distributed-llm."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class NodeState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class HealthRecord:
    node_id: str
    state: NodeState = NodeState.OFFLINE
    last_probe_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    gpu_utilization: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    layer_range: str = ""
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        self.latency_p50_ms = sorted_latencies[n // 2] if n else 0.0
        self.latency_p99_ms = sorted_latencies[int(n * 0.99)] if n else 0.0


class HealthStateStore:
    """Thread-safe storage for node health records."""

    def __init__(self):
        self._records: Dict[str, HealthRecord] = {}
        self._lock = threading.Lock()

    def get(self, node_id: str) -> Optional[HealthRecord]:
        with self._lock:
            return self._records.get(node_id)

    def get_all(self) -> Dict[str, HealthRecord]:
        with self._lock:
            return dict(self._records)

    def set(self, node_id: str, record: HealthRecord) -> None:
        with self._lock:
            self._records[node_id] = record

    def update_state(self, node_id: str, state: NodeState) -> Optional[HealthRecord]:
        with self._lock:
            record = self._records.get(node_id)
            if record:
                record.state = state
            return record

    def remove(self, node_id: str) -> None:
        with self._lock:
            self._records.pop(node_id, None)

    def healthy_nodes(self) -> list[str]:
        with self._lock:
            return [
                r.node_id for r in self._records.values()
                if r.state in (NodeState.HEALTHY, NodeState.DEGRADED)
            ]
