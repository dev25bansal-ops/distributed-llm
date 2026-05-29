"""Built-in plugins shipped with the distributed LLM distribution.

Each plugin subclasses ``PluginBase`` and registers its hooks:

* ``RateLimitPlugin`` — configurable per-tenant and per-model rate limiting
* ``AuditLogPlugin`` — structured JSON audit logging to file or stderr
* ``MetricsPlugin`` — plugin health and hook-invocation counters
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.core.plugin_system import PluginBase


# ── RateLimitPlugin ──────────────────────────────────────────────────────────

class _SlidingWindowCounter:
    """Thread-safe sliding-window counter for rate limiting."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._prune(now)
            if len(self._timestamps) >= self.max_requests:
                return False
            self._timestamps.append(now)
            return True

    def remaining(self) -> int:
        now = time.time()
        with self._lock:
            self._prune(now)
            return max(0, self.max_requests - len(self._timestamps))


class RateLimitPlugin(PluginBase):
    """Per-tenant and per-model request rate limiting.

    Activated when ``DISTLLM_PLUGIN_RATELIMIT_ENABLED=1`` is set.
    Thresholds are configured via environment variables:

    * ``DISTLLM_PLUGIN_RATELIMIT_DEFAULT`` — default max requests per
       window (default: 1000)
    * ``DISTLLM_PLUGIN_RATELIMIT_WINDOW`` — window in seconds (default: 60)
    * ``DISTLLM_PLUGIN_RATELIMIT_TENANT_<NAME>`` — per-tenant override
    * ``DISTLLM_PLUGIN_RATELIMIT_MODEL_<NAME>`` — per-model override

    The ``on_request`` hook returns ``{"_reject": {"reason": ..., "retry_after": ...}}``
    when the limit is exceeded, which the ``PluginHookMiddleware`` converts
    to a 429 response.
    """

    def name(self) -> str:
        return "rate-limit"

    def version(self) -> str:
        return "1.0.0"

    def on_init(self, context: dict[str, Any]) -> None:
        self._enabled = os.environ.get("DISTLLM_PLUGIN_RATELIMIT_ENABLED", "0") == "1"
        if not self._enabled:
            return
        try:
            default = int(os.environ.get("DISTLLM_PLUGIN_RATELIMIT_DEFAULT", "1000"))
        except (ValueError, TypeError):
            default = 1000
        try:
            window = int(os.environ.get("DISTLLM_PLUGIN_RATELIMIT_WINDOW", "60"))
        except (ValueError, TypeError):
            window = 60
        self._default_counter = _SlidingWindowCounter(default, window)
        self._tenant_limiters: dict[str, _SlidingWindowCounter] = {}
        self._model_limiters: dict[str, _SlidingWindowCounter] = {}
        # Parse per-tenant overrides from env vars
        for key, value in os.environ.items():
            if key.startswith("DISTLLM_PLUGIN_RATELIMIT_TENANT_"):
                tenant = key[len("DISTLLM_PLUGIN_RATELIMIT_TENANT_"):].lower()
                try:
                    self._tenant_limiters[tenant] = _SlidingWindowCounter(int(value), window)
                except (ValueError, TypeError):
                    pass
            elif key.startswith("DISTLLM_PLUGIN_RATELIMIT_MODEL_"):
                model = key[len("DISTLLM_PLUGIN_RATELIMIT_MODEL_"):].lower()
                try:
                    self._model_limiters[model] = _SlidingWindowCounter(int(value), window)
                except (ValueError, TypeError):
                    pass
        logger.info(
            f"RateLimitPlugin: {default} req/{window}s default, "
            f"{len(self._tenant_limiters)} tenant overrides, "
            f"{len(self._model_limiters)} model overrides"
        )

    def on_request(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        tenant = (context.get("tenant") or "default").lower()
        model = (context.get("model") or "unknown").lower()

        # Check tenant override first, then model override, then default
        limiter = self._tenant_limiters.get(tenant)
        if limiter is None:
            limiter = self._model_limiters.get(model)
        if limiter is None:
            limiter = self._default_counter

        if not limiter.allow():
            return {
                "_reject": {
                    "reason": "rate_limit_exceeded",
                    "retry_after": int(limiter.window_seconds),
                }
            }
        return None


# ── AuditLogPlugin ───────────────────────────────────────────────────────────

class AuditLogPlugin(PluginBase):
    """Structured audit logging of all API requests.

    Writes one JSON line per request to the path specified by
    ``DISTLLM_AUDIT_LOG`` (default: stderr via loguru at INFO level).

    Each audit entry contains: request_id, timestamp, method, path, status,
    tenant, model, duration_ms, client_ip, and user_agent.
    """

    def name(self) -> str:
        return "audit-log"

    def version(self) -> str:
        return "1.0.0"

    def on_init(self, context: dict[str, Any]) -> None:
        audit_path = os.environ.get("DISTLLM_AUDIT_LOG", "")
        self._file: Any = None
        if audit_path:
            p = Path(audit_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._file = p.open("a", encoding="utf-8")
            logger.info(f"AuditLogPlugin writing to {audit_path}")
        else:
            logger.info("AuditLogPlugin writing to loguru (stderr)")

    def on_response(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        entry = {
            "request_id": request.get("request_id", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.get("method", ""),
            "path": request.get("path", ""),
            "status": response.get("status_code", 0),
            "tenant": request.get("tenant", ""),
            "model": request.get("model", ""),
            "duration_ms": response.get("duration_ms", 0),
            "client_ip": request.get("client_ip", ""),
            "user_agent": request.get("user_agent", ""),
        }
        if self._file is not None:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()
        else:
            logger.info(f"AUDIT {json.dumps(entry, default=str)}")

    def on_error(self, request: dict[str, Any], error: Exception) -> None:
        entry = {
            "request_id": request.get("request_id", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.get("method", ""),
            "path": request.get("path", ""),
            "error": str(error),
            "tenant": request.get("tenant", ""),
            "model": request.get("model", ""),
            "client_ip": request.get("client_ip", ""),
        }
        if self._file is not None:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()
        else:
            logger.error(f"AUDIT {json.dumps(entry, default=str)}")

    def on_stop(self, context: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ── MetricsPlugin ────────────────────────────────────────────────────────────

class MetricsPlugin(PluginBase):
    """Tracks plugin-system health and hook-invocation counters.

    Counts every hook dispatch and accumulates error/failure rates per
    plugin.  State is exposed via ``get_counts()`` for Prometheus export
    or dashboard display.
    """

    def name(self) -> str:
        return "metrics"

    def version(self) -> str:
        return "1.0.0"

    def on_init(self, context: dict[str, Any]) -> None:
        self._lock = threading.Lock()
        self._hook_counts: dict[str, int] = defaultdict(int)
        self._error_counts: dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def on_start(self, context: dict[str, Any]) -> None:
        self._incr("on_start")

    def on_request(self, context: dict[str, Any]) -> dict[str, Any] | None:
        self._incr("on_request")
        return None

    def on_response(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        self._incr("on_response")
        if response.get("status_code", 200) >= 500:
            self._incr("server_error")

    def on_error(self, request: dict[str, Any], error: Exception) -> None:
        self._incr("on_error")

    def on_model_load(self, model_name: str, config: dict[str, Any]) -> None:
        self._incr("on_model_load")

    def on_model_unload(self, model_name: str) -> None:
        self._incr("on_model_unload")

    def _incr(self, hook: str) -> None:
        with self._lock:
            self._hook_counts[hook] += 1

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._hook_counts)

    def get_error_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._error_counts)

    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
