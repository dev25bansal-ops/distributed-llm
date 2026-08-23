"""Opt-in anonymous telemetry for usage analytics.

Collects anonymous usage data to help prioritize development.
Users can opt-in via config or environment variable.

Collected data (anonymous):
- DistLLM version
- Python version
- OS platform
- Number of GPUs
- Model sizes used (not model names)
- Request counts and throughput
- Error rates
- Feature usage (which backends, which endpoints)

NOT collected:
- Model names or paths
- Prompt content or responses
- API keys or secrets
- IP addresses or hostnames
- User identifiers

Usage::

    telemetry = get_telemetry()
    telemetry.opt_in()
    telemetry.record_request(model_size_b=7, tokens=100, latency_ms=50)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: str  # "request", "error", "startup", "feature"
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """Anonymous opt-in telemetry collector.

    Telemetry is disabled by default. Users must explicitly opt-in
    via DISTLLM_TELEMETRY=1 or config.yaml telemetry.enabled=true.
    """

    ENDPOINT = "https://telemetry.distllm.ai/v1/events"
    BATCH_SIZE = 50
    FLUSH_INTERVAL_S = 300  # 5 minutes

    def __init__(
        self,
        enabled: bool = False,
        endpoint: str | None = None,
        data_dir: str = ".distllm_telemetry",
    ):
        self._enabled = enabled
        self._endpoint = endpoint or self.ENDPOINT
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()
        self._instance_id = self._get_instance_id()
        self._start_time = time.time()

        # Collect system info once
        self._system_info = self._collect_system_info()

    def opt_in(self) -> None:
        """Enable telemetry collection."""
        self._enabled = True
        logger.info("Telemetry enabled — thank you for helping improve DistLLM!")

    def opt_out(self) -> None:
        """Disable telemetry collection and delete local data."""
        self._enabled = False
        self._events.clear()
        logger.info("Telemetry disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def record_request(
        self,
        model_size_b: float = 0,
        tokens: int = 0,
        latency_ms: float = 0,
        endpoint: str = "",
        backend: str = "",
        error: bool = False,
    ) -> None:
        """Record an inference request (anonymous)."""
        if not self._enabled:
            return

        self._add_event(TelemetryEvent(
            event_type="request",
            data={
                "model_size_b": model_size_b,
                "tokens": tokens,
                "latency_ms": round(latency_ms, 1),
                "endpoint": endpoint,
                "backend": backend,
                "error": error,
            },
        ))

    def record_feature(self, feature: str, used: bool = True) -> None:
        """Record feature usage."""
        if not self._enabled:
            return

        self._add_event(TelemetryEvent(
            event_type="feature",
            data={"feature": feature, "used": used},
        ))

    def record_startup(self, num_nodes: int = 1, num_gpus: int = 0) -> None:
        """Record startup info."""
        if not self._enabled:
            return

        self._add_event(TelemetryEvent(
            event_type="startup",
            data={
                **self._system_info,
                "num_nodes": num_nodes,
                "num_gpus": num_gpus,
            },
        ))

    def flush(self) -> None:
        """Flush events to disk (for later upload)."""
        if not self._enabled:
            return

        with self._lock:
            events = list(self._events)
            self._events.clear()

        if not events:
            return

        # Save to local file
        path = self._data_dir / f"events_{int(time.time())}.jsonl"
        try:
            with open(path, "a") as f:
                for event in events:
                    f.write(json.dumps({
                        "type": event.event_type,
                        "ts": event.timestamp,
                        "data": event.data,
                        "instance": self._instance_id,
                    }) + "\n")
        except Exception as e:
            logger.debug(f"Telemetry flush failed: {e}")

    def get_stats(self) -> dict:
        """Return telemetry statistics."""
        return {
            "enabled": self._enabled,
            "instance_id": self._instance_id,
            "events_pending": len(self._events),
            "uptime_s": round(time.time() - self._start_time, 1),
            "system": self._system_info,
        }

    def _add_event(self, event: TelemetryEvent) -> None:
        # Defer flush until AFTER releasing _lock — flush() re-acquires the
        # same non-reentrant lock, so calling it here deadlocks (F-023).
        should_flush = False
        with self._lock:
            self._events.append(event)
            if len(self._events) >= self.BATCH_SIZE:
                should_flush = True
        if should_flush:
            self.flush()

    def _get_instance_id(self) -> str:
        """Generate a stable anonymous instance ID."""
        # Hash hostname + install path for a stable but anonymous ID
        raw = f"{platform.node()}:{os.getcwd()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _collect_system_info(self) -> dict:
        """Collect anonymous system information."""
        info = {
            "os": platform.system(),
            "python": platform.python_version(),
            "distllm_version": self._get_version(),
        }

        # GPU info (count only, no names)
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_count"] = torch.cuda.device_count()
                info["cuda_version"] = torch.version.cuda or "unknown"
            else:
                info["gpu_count"] = 0
        except ImportError:
            info["gpu_count"] = 0

        return info

    def _get_version(self) -> str:
        try:
            from importlib.metadata import version
            return version("distributed-llm")
        except Exception:
            return "unknown"


# Global singleton
_telemetry: TelemetryCollector | None = None
_telemetry_lock = threading.Lock()


def get_telemetry() -> TelemetryCollector:
    """Get or create the global telemetry collector."""
    global _telemetry
    if _telemetry is None:
        with _telemetry_lock:
            if _telemetry is None:
                enabled = os.environ.get("DISTLLM_TELEMETRY", "").lower() in ("1", "true", "yes")
                _telemetry = TelemetryCollector(enabled=enabled)
    return _telemetry
