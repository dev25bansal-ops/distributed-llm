"""Per-request latency tracking for batch scheduling.

Tracks TTFT, TPOT, and SLA compliance for individual requests
during continuous batching. Separate from the node-level LatencyTracker
used by the rebalancer for straggler detection.
"""

import time
import threading
from dataclasses import dataclass


@dataclass
class RequestLatencyInfo:
    request_id: str
    enqueued_at: float
    first_token_at: float | None = None
    last_token_at: float | None = None
    tokens_generated: int = 0
    sla_target_ms: float = 5000.0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.enqueued_at) * 1000

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.enqueued_at) * 1000

    @property
    def tpot_ms(self) -> float | None:
        if self.first_token_at is None or self.last_token_at is None or self.tokens_generated <= 1:
            return None
        return (self.last_token_at - self.first_token_at) * 1000 / (self.tokens_generated - 1)

    @property
    def is_overdue(self) -> bool:
        return self.elapsed_ms > self.sla_target_ms


class RequestLatencyTracker:
    def __init__(self, default_sla_ms: float = 5000.0):
        self._default_sla_ms = default_sla_ms
        self._requests: dict[str, RequestLatencyInfo] = {}
        self._completed: list[RequestLatencyInfo] = []
        self._lock = threading.Lock()

    def register(self, request_id: str, sla_ms: float | None = None) -> None:
        with self._lock:
            self._requests[request_id] = RequestLatencyInfo(
                request_id=request_id,
                enqueued_at=time.time(),
                sla_target_ms=sla_ms or self._default_sla_ms,
            )

    def record_first_token(self, request_id: str) -> None:
        with self._lock:
            info = self._requests.get(request_id)
            if info:
                info.first_token_at = time.time()

    def record_token(self, request_id: str) -> None:
        with self._lock:
            info = self._requests.get(request_id)
            if info:
                info.last_token_at = time.time()
                info.tokens_generated += 1

    def complete(self, request_id: str) -> None:
        with self._lock:
            info = self._requests.pop(request_id, None)
            if info:
                if info.last_token_at is None:
                    info.last_token_at = time.time()
                self._completed.append(info)

    def get_latency_boost(self, request_id: str, base_priority: int) -> int:
        with self._lock:
            info = self._requests.get(request_id)
            if not info:
                return base_priority
            if info.is_overdue:
                if info.elapsed_ms > info.sla_target_ms * 2:
                    return min(base_priority, 0)
                return max(0, base_priority - 1)
            return base_priority

    def get_metrics(self, request_id: str) -> dict:
        with self._lock:
            info = self._requests.get(request_id)
            if not info:
                for c in reversed(self._completed):
                    if c.request_id == request_id:
                        info = c
                        break
            if not info:
                return {}
            return {
                "request_id": request_id,
                "ttft_ms": info.ttft_ms,
                "tpot_ms": info.tpot_ms,
                "elapsed_ms": info.elapsed_ms,
                "tokens_generated": info.tokens_generated,
                "is_overdue": info.is_overdue,
                "sla_target_ms": info.sla_target_ms,
            }

    def get_recent_metrics(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [
                {
                    "request_id": c.request_id,
                    "ttft_ms": c.ttft_ms,
                    "tpot_ms": c.tpot_ms,
                    "elapsed_ms": c.elapsed_ms,
                    "tokens_generated": c.tokens_generated,
                    "is_overdue": c.is_overdue,
                }
                for c in self._completed[-limit:]
            ]

    @property
    def tracked_count(self) -> int:
        with self._lock:
            return len(self._requests)
