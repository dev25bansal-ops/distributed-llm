"""Webhook System — event-driven webhooks for cluster events.

Dispatches HTTP callbacks when cluster events occur (node join/leave,
model load/unload, errors, threshold breaches). Supports configurable
event types, retry with exponential backoff, and HMAC signing.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from urllib.parse import urlparse

# Type alias for a formatter: (event, payload) -> (body_dict, extra_headers_dict)
WebhookFormatter = Callable[[str, dict[str, Any]], tuple[dict[str, Any], dict[str, str]]]

from loguru import logger


def is_safe_webhook_url(url: str, allowlist: set[str] | None = None) -> bool:
    """Return True only for http(s) URLs that cannot hit metadata/private targets.

    Server-Side Request Forgery guard: rejects non-http(s) schemes, cloud
    metadata addresses (169.254.169.254), loopback, private/link-local ranges,
    and any host that resolves to a private address.  When
    ``DISTLLM_WEBHOOK_ALLOWLIST`` (or the *allowlist* arg) is set, ONLY
    allowlisted hosts are permitted (deny-by-default).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    if allowlist is None:
        allowlist = set(
            h for h in os.environ.get("DISTLLM_WEBHOOK_ALLOWLIST", "").split(",") if h.strip()
        )
    if allowlist:
        return host in allowlist
    if host == "169.254.169.254":
        return False
    if host in ("localhost", "127.0.0.1", "::1"):
        return False
    # A literal IP literal: reject private/loopback/link-local directly.
    try:
        ip = ipaddress.ip_address(host)
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
        return True  # literal public IP is safe
    except ValueError:
        pass  # not a literal IP — resolve below
    # Resolve the host; reject only if it demonstrably maps to a private
    # address.  Unresolvable hosts are accepted (delivery will fail naturally);
    # we do NOT hard-fail on DNS lookups for non-literal hosts.
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, OSError):
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


# Backward-compatible alias (original private name).
_is_safe_webhook_url = is_safe_webhook_url


def _is_private_webhook_url(url: str) -> bool:
    """Return True if the URL targets loopback or a private/link-local host.

    Used to gate the explicit ``allow_private`` opt-in on webhook registration.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        if host in ("localhost", "127.0.0.1", "::1"):
            return True
        try:
            ip = ipaddress.ip_address(host)
            return (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast)
        except ValueError:
            return False
    except ValueError:
        return False


class WebhookEvent(Enum):
    NODE_JOINED = "node.joined"
    NODE_LEFT = "node.left"
    NODE_FAILED = "node.failed"
    NODE_DRAINING = "node.draining"
    NODE_RECOVERED = "node.recovered"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_ERROR = "model.error"
    INFERENCE_STARTED = "inference.started"
    INFERENCE_COMPLETED = "inference.completed"
    INFERENCE_ERROR = "inference.error"
    THRESHOLD_BREACHED = "threshold.breached"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker.opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker.closed"
    HIGH_LATENCY = "high_latency"
    CLUSTER_HEALTH_CHANGED = "cluster.health_changed"
    STRAAGLER_DETECTED = "straggler.detected"
    RECOVERY_STARTED = "recovery.started"
    RECOVERY_COMPLETED = "recovery.completed"
    BACKUP_CREATED = "backup.created"
    BACKUP_FAILED = "backup.failed"
    CERTIFICATE_EXPIRING = "certificate.expiring"
    LEADER_ELECTED = "leader.elected"
    LEADER_LOST = "leader.lost"
    COORDINATOR_FAILOVER = "coordinator.failover"


@dataclass
class WebhookTarget:
    """A registered webhook endpoint."""
    url: str
    events: list[str]
    secret: str = ""
    retry_count: int = 3
    timeout_s: float = 5.0
    active: bool = True
    created_at: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    label: str = ""
    formatter: WebhookFormatter | None = None
    # Explicit registration-time opt-in for loopback/private targets;
    # honored by the delivery-time SSRF re-validation below.
    allow_private: bool = False


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    event: str
    url: str
    status_code: int
    success: bool
    duration_ms: float
    attempt: int
    error: str = ""


class WebhookManager:
    """Event-driven webhook dispatcher.

    Usage:
        wh = WebhookManager()
        wh.register("https://hooks.slack.com/...",
                     events=["node.joined", "node.left"])
        wh.dispatch(WebhookEvent.NODE_JOINED, {"node_id": "worker-1"})
    """

    MAX_RETRIES: int = 3
    BACKOFF_BASE: float = 1.0
    MAX_TARGETS: int = 50
    QUEUE_SIZE: int = 1000

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_targets: int = 50,
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_targets = max_targets
        self._lock = threading.RLock()
        self._targets: list[WebhookTarget] = []
        self._delivery_log: list[WebhookDelivery] = []
        self._delivery_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._queue: list[tuple[WebhookEvent, dict[str, Any]]] = []
        self._running = False

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        url: str,
        events: list[str] | None = None,
        secret: str = "",
        retry_count: int = 3,
        timeout_s: float = 5.0,
        label: str = "",
        formatter: WebhookFormatter | None = None,
        allow_private: bool = False,
    ) -> bool:
        """Register a webhook target URL.

        Args:
            url: HTTP endpoint to deliver webhooks.
            events: List of event types to subscribe to (all if empty/None).
            secret: HMAC signing secret.
            retry_count: Number of retry attempts on failure.
            timeout_s: HTTP request timeout.
            label: Human-readable label for this target.
            formatter: Optional callable that transforms (event, payload) into
                       (body_dict, extra_headers_dict) for platform-specific formatting.
            allow_private: Allow explicitly private/localhost targets.  Defaults
                to False (SSRF-safe); operators registering a local-only webhook
                must opt in knowingly.

        Returns:
            True if registered successfully.
        """
        if not is_safe_webhook_url(url) and not (allow_private and _is_private_webhook_url(url)):
            logger.warning(f"Invalid webhook URL (unsafe/unsupported): {url}")
            return False

        with self._lock:
            if len(self._targets) >= self._max_targets:
                logger.warning("Max webhook targets reached")
                return False

            for t in self._targets:
                if t.url == url:
                    logger.debug(f"Webhook {url} already registered")
                    t.events = events or list(WebhookEvent.__members__.values())
                    return True

            target = WebhookTarget(
                url=url,
                events=events or [e.value for e in WebhookEvent],
                secret=secret,
                retry_count=min(retry_count, self._max_retries),
                timeout_s=timeout_s,
                created_at=time.time(),
                label=label or url,
                formatter=formatter,
                allow_private=bool(allow_private and _is_private_webhook_url(url)),
            )
            self._targets.append(target)
            logger.info(f"Registered webhook: {target.label} ({len(target.events)} events)")
            return True

    def unregister(self, url: str) -> bool:
        """Remove a webhook target."""
        with self._lock:
            before = len(self._targets)
            self._targets = [t for t in self._targets if t.url != url]
            return len(self._targets) < before

    def list_targets(self) -> list[WebhookTarget]:
        with self._lock:
            return list(self._targets)

    # ── Dispatch ────────────────────────────────────────────────────────

    def dispatch(
        self, event: WebhookEvent, payload: dict[str, Any] | None = None,
    ) -> int:
        """Dispatch an event to all matching webhook targets.

        Returns the number of targets that will receive this event.
        """
        payload = payload or {}
        payload["event"] = event.value
        payload["timestamp"] = time.time()

        with self._lock:
            targets = list(self._targets)

        count = 0
        for target in targets:
            if not target.active:
                continue
            if event.value not in target.events and "*" not in target.events:
                continue
            self._enqueue(target, event, payload)
            count += 1

        if count > 0:
            logger.debug(f"Dispatched {event.value} to {count} target(s)")
        return count

    def dispatch_async(
        self, event: WebhookEvent, payload: dict[str, Any] | None = None,
    ) -> int:
        """Non-blocking dispatch via background worker thread."""
        return self.dispatch(event, payload)

    # ── Delivery log ────────────────────────────────────────────────────

    def delivery_log(
        self, limit: int = 100, event_type: str | None = None,
    ) -> list[WebhookDelivery]:
        with self._delivery_lock:
            log = list(self._delivery_log)
        if event_type:
            log = [d for d in log if d.event == event_type]
        return log[-limit:]

    def success_rate(self, event_type: str | None = None) -> float:
        with self._delivery_lock:
            relevant = [
                d for d in self._delivery_log
                if event_type is None or d.event == event_type
            ]
        if not relevant:
            return 1.0
        return sum(1 for d in relevant if d.success) / len(relevant)

    # ── Background worker ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the background webhook delivery worker."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=_webhook_worker, args=(self,), daemon=True)
        self._worker_thread.start()
        logger.debug("Webhook worker started")

    def stop(self) -> None:
        self._running = False

    # ── Internal ────────────────────────────────────────────────────────

    def _enqueue(
        self, target: WebhookTarget, event: WebhookEvent,
        payload: dict[str, Any],
    ) -> None:
        if len(self._queue) >= self.QUEUE_SIZE:
            self._queue.pop(0)
        self._queue.append((target, event, payload))

    def _deliver(
        self, target: WebhookTarget, event: WebhookEvent,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver a single webhook with retry logic."""
        import httpx

        # Re-validate at delivery time: a URL that was safe at registration
        # could have been mutated afterwards (store tampering, update path
        # bypassing the guard).  Fail closed — deactivate the target.
        # Targets that explicitly opted in to a private URL at registration
        # (allow_private=True) keep that allowance here.
        if not is_safe_webhook_url(target.url) and not (
            target.allow_private and _is_private_webhook_url(target.url)
        ):
            logger.warning(
                "Webhook %s URL failed delivery-time SSRF re-validation; "
                "deactivating: %s", target.label, target.url,
            )
            target.active = False
            delivery = WebhookDelivery(
                event=event.value, url=target.url,
                status_code=0, success=False,
                duration_ms=0.0, attempt=1,
                error="URL blocked by delivery-time SSRF re-validation",
            )
            self._log_delivery(delivery)
            return delivery

        if target.formatter is not None:
            formatted_body, extra_headers = target.formatter(event.value, payload)
            body = json.dumps(formatted_body, default=str).encode("utf-8")
            headers = {"Content-Type": "application/json", **extra_headers}
        else:
            body = json.dumps(payload, default=str).encode("utf-8")
            headers = {"Content-Type": "application/json"}

        if target.secret:
            signature = hmac.new(
                target.secret.encode(), body, hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature

        last_error = ""
        duration = 0.0
        final_status = 0

        for attempt in range(1, target.retry_count + 2):
            t0 = time.monotonic()
            try:
                resp = httpx.post(
                    target.url, content=body, headers=headers,
                    timeout=target.timeout_s,
                )
                duration = (time.monotonic() - t0) * 1000
                final_status = resp.status_code
                if resp.status_code < 500:
                    target.last_success = time.time()
                    target.consecutive_failures = 0
                    delivery = WebhookDelivery(
                        event=event.value, url=target.url,
                        status_code=resp.status_code, success=True,
                        duration_ms=round(duration, 2), attempt=attempt,
                    )
                    self._log_delivery(delivery)
                    return delivery
                last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                duration = (time.monotonic() - t0) * 1000
                last_error = str(e)

            if attempt <= target.retry_count:
                delay = self._backoff_base * (2 ** (attempt - 1))
                time.sleep(min(delay, 30.0))

        target.last_failure = time.time()
        target.consecutive_failures += 1
        if target.consecutive_failures >= 5:
            target.active = False
            logger.warning(f"Webhook {target.url} deactivated after {target.consecutive_failures} failures")

        delivery = WebhookDelivery(
            event=event.value, url=target.url,
            status_code=final_status, success=False,
            duration_ms=round(duration, 2),
            attempt=target.retry_count + 1, error=last_error,
        )
        self._log_delivery(delivery)
        return delivery

    def _log_delivery(self, delivery: WebhookDelivery) -> None:
        with self._delivery_lock:
            self._delivery_log.append(delivery)
            if len(self._delivery_log) > 10000:
                self._delivery_log = self._delivery_log[-5000:]


def _webhook_worker(mgr: WebhookManager) -> None:
    """Background loop that drains the webhook queue."""
    while mgr._running:
        try:
            if not mgr._queue:
                time.sleep(0.1)
                continue
            target, event, payload = mgr._queue.pop(0)
            mgr._deliver(target, event, payload)
        except Exception:
            time.sleep(0.1)
