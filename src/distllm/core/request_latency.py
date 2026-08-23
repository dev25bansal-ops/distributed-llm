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
    completed_at: float | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.enqueued_at) * 1000

    @property
    def elapsed_ms(self) -> float:
        # For COMPLETED requests, freeze elapsed at completion — the live clock
        # would otherwise make finished-fast requests look overdue as wall time
        # advances, corrupting SLA/compliance percentiles (F-022).
        end = self.completed_at or time.time()
        return (end - self.enqueued_at) * 1000

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
                info.completed_at = time.time()
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

    def get_urgency_score(self, request_id: str) -> float:
        """Return a normalized urgency score (0=not urgent, 1=critical).

        Used by the scheduler to reorder active decode sequences:
        sequences closer to their SLO deadline get higher urgency.
        """
        with self._lock:
            info = self._requests.get(request_id)
            if not info:
                return 0.0
            ratio = info.elapsed_ms / max(info.sla_target_ms, 1)
            return min(1.0, ratio)

    def get_overdue_request_ids(self) -> list[str]:
        """Return all request IDs that have exceeded their SLA."""
        with self._lock:
            return [rid for rid, info in self._requests.items() if info.is_overdue]

    def get_requests_sorted_by_deadline(self) -> list[tuple[str, float]]:
        """Return (request_id, elapsed_ratio) pairs sorted by urgency (most urgent first).

        Used by the scheduler to prioritize active decode sequences
        that are closest to their SLO deadline.
        """
        with self._lock:
            items = []
            for rid, info in self._requests.items():
                if info.first_token_at is not None:
                    ratio = info.elapsed_ms / max(info.sla_target_ms, 1)
                    items.append((rid, ratio))
            items.sort(key=lambda x: -x[1])  # Most urgent first
            return items

    @property
    def tracked_count(self) -> int:
        with self._lock:
            return len(self._requests)

    def get_sla_percentiles(self, window_size: int = 100) -> dict:
        """Compute SLA compliance percentiles from recent completed requests.

        Returns:
            Dict with P50, P95, P99 latency, SLA compliance rate,
            and TTFT/TPOT percentiles.
        """
        with self._lock:
            recent = self._completed[-window_size:]
            if not recent:
                return {
                    "sample_size": 0,
                    "ttft_p50_ms": 0, "ttft_p95_ms": 0, "ttft_p99_ms": 0,
                    "tpot_p50_ms": 0, "tpot_p95_ms": 0, "tpot_p99_ms": 0,
                    "elapsed_p50_ms": 0, "elapsed_p95_ms": 0, "elapsed_p99_ms": 0,
                    "sla_compliance_pct": 100.0,
                    "overdue_count": 0,
                }

            ttfts = sorted([r.ttft_ms for r in recent if r.ttft_ms is not None])
            tpots = sorted([r.tpot_ms for r in recent if r.tpot_ms is not None])
            elapsed = sorted([r.elapsed_ms for r in recent])
            overdue = sum(1 for r in recent if r.is_overdue)

            def percentile(sorted_list, pct):
                if not sorted_list:
                    return 0.0
                idx = int(len(sorted_list) * pct / 100)
                return round(sorted_list[min(idx, len(sorted_list) - 1)], 2)

            return {
                "sample_size": len(recent),
                "ttft_p50_ms": percentile(ttfts, 50),
                "ttft_p95_ms": percentile(ttfts, 95),
                "ttft_p99_ms": percentile(ttfts, 99),
                "tpot_p50_ms": percentile(tpots, 50),
                "tpot_p95_ms": percentile(tpots, 95),
                "tpot_p99_ms": percentile(tpots, 99),
                "elapsed_p50_ms": percentile(elapsed, 50),
                "elapsed_p95_ms": percentile(elapsed, 95),
                "elapsed_p99_ms": percentile(elapsed, 99),
                "sla_compliance_pct": round((1 - overdue / len(recent)) * 100, 1),
                "overdue_count": overdue,
            }
