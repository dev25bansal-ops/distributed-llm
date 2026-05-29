"""Route Audit Log — persists every routing decision to JSONL.

Records provider, region, pricing, carbon, latency, and outcome
for every routing decision. Enables replay analysis, cost auditing,
and debugging of routing behavior.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RouteAuditEntry:
    """A single routing decision audit record."""
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""

    # Decision
    provider: str = ""
    instance_type: str = ""
    region: str = ""
    gpu_type: str = ""
    scoring_method: str = ""

    # Pricing at decision time
    price_per_hour: float = 0.0
    spot_used: bool = False
    estimated_cost: float = 0.0

    # Carbon at decision time
    carbon_intensity: float = 0.0

    # Latency at decision time
    latency_at_decision: float = 0.0

    # Outcome (filled after request completes)
    result: str = ""  # "success", "error", "timeout"
    actual_cost: float = 0.0
    actual_latency: float = 0.0
    tokens_generated: int = 0
    error_message: str = ""

    # Alternatives
    alternatives_considered: int = 0
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteAuditLog:
    """Persists routing decisions to a JSONL file.

    Usage::

        audit = RouteAuditLog("/var/log/distllm/route_audit.jsonl")
        entry = RouteAuditEntry(provider="aws", instance_type="p4d.24xlarge", ...)
        audit.log(entry)
        audit.update_outcome(request_id="req-123", result="success", actual_cost=14.40)
    """

    def __init__(
        self,
        log_path: str = "route_audit.jsonl",
        max_entries_memory: int = 10000,
        flush_interval_s: float = 10.0,
    ):
        self._log_path = log_path
        self._max_entries = max_entries_memory
        self._flush_interval = flush_interval_s
        self._buffer: list[str] = []
        self._recent: list[RouteAuditEntry] = []
        self._lock = threading.Lock()
        self._pending_outcomes: dict[str, RouteAuditEntry] = {}

        # Ensure directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, entry: RouteAuditEntry) -> None:
        """Log a routing decision."""
        line = json.dumps(entry.to_dict(), default=str)
        with self._lock:
            self._buffer.append(line)
            self._recent.append(entry)
            if len(self._recent) > self._max_entries:
                self._recent = self._recent[-self._max_entries:]
            if entry.request_id:
                self._pending_outcomes[entry.request_id] = entry
        # Flush if buffer is large
        if len(self._buffer) >= 100:
            self.flush()

    def update_outcome(
        self,
        request_id: str,
        result: str = "success",
        actual_cost: float = 0.0,
        actual_latency: float = 0.0,
        tokens_generated: int = 0,
        error_message: str = "",
    ) -> None:
        """Update the outcome of a previously logged routing decision."""
        with self._lock:
            entry = self._pending_outcomes.pop(request_id, None)
        if not entry:
            return
        entry.result = result
        entry.actual_cost = actual_cost
        entry.actual_latency = actual_latency
        entry.tokens_generated = tokens_generated
        entry.error_message = error_message
        line = json.dumps(entry.to_dict(), default=str)
        with self._lock:
            self._buffer.append(line)

    def flush(self) -> None:
        """Flush buffered entries to disk."""
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()
        if not lines:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except OSError as e:
            logger.warning(f"Failed to flush route audit log: {e}")

    def get_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit entries from memory."""
        with self._lock:
            return [e.to_dict() for e in self._recent[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        with self._lock:
            total = len(self._recent)
            if not total:
                return {"total": 0}
            providers = {}
            successes = 0
            for e in self._recent:
                providers[e.provider] = providers.get(e.provider, 0) + 1
                if e.result == "success":
                    successes += 1
            return {
                "total": total,
                "successes": successes,
                "success_rate": successes / total if total else 0,
                "by_provider": providers,
                "buffered": len(self._buffer),
            }

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
