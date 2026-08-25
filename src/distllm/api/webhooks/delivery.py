"""Webhook delivery engine with registration, HMAC-signing, retry, and DLQ.

Supports registering webhooks, dispatching events to matching subscribers,
exponential backoff retry (1s, 2s, 4s, 8s, 16s, max 5 attempts), dead-letter
queuing, and thread-safe in-memory storage.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

import httpx

from distllm.core.webhook_manager import is_safe_webhook_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class WebhookEvent(str, enum.Enum):
    """Standard event types the webhook system can emit."""

    JOB_COMPLETED = "job.completed"
    BATCH_COMPLETED = "batch.completed"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    ERROR_THRESHOLD = "error.threshold"
    NODE_ONLINE = "node.online"
    NODE_OFFLINE = "node.offline"
    RATE_LIMIT_WARNING = "rate_limit.warning"
    QUOTA_EXCEEDED = "quota.exceeded"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker.open"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker.closed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WebhookRegistration:
    """A registered webhook subscriber."""

    id: str
    url: str
    events: set[str]
    secret: str
    created_at: str  # ISO-8601
    is_active: bool = True

    def matches(self, event: str) -> bool:
        """Return True if this registration subscribes to *event*."""
        return event in self.events


@dataclass
class WebhookDelivery:
    """Record of a single delivery attempt."""

    id: str
    webhook_id: str
    event: str
    payload: dict[str, Any]
    attempt: int
    status: str  # "pending" | "success" | "failed" | "dead"
    error: str | None = None
    next_retry: float | None = None  # Unix timestamp
    created_at: str = field(default_factory=lambda: _now_iso())


class WebhookDeliveryStatus:
    """Constants for delivery status values."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD = "dead"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return time.time()


def _compute_signature(secret: str, body: bytes) -> str:
    """Return hex-encoded HMAC-SHA256 of *body* using *secret*."""
    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


def _build_payload(event: str, data: dict[str, Any]) -> dict[str, Any]:
    """Build the standard webhook payload envelope."""
    return {
        "event": event,
        "timestamp": _now_iso(),
        "data": data,
    }


# Retry schedule: 1s, 2s, 4s, 8s, 16s (5 retries total)
_RETRY_DELAYS = [1.0, 2.0, 4.0, 8.0, 16.0]
_MAX_RETRIES = len(_RETRY_DELAYS)


class UnsafeWebhookURLError(ValueError):
    """Raised when a webhook URL fails the SSRF safety check.

    Rejected targets include non-http(s) schemes, cloud metadata endpoints
    (169.254.169.254), loopback, private/link-local/reserved ranges, and
    hostnames that resolve to those — unless explicitly listed in
    ``DISTLLM_WEBHOOK_ALLOWLIST`` (deny-by-default when set).
    """


# ---------------------------------------------------------------------------
# WebhookManager
# ---------------------------------------------------------------------------

class WebhookManager:
    """Thread-safe manager for webhook registration, dispatch, and DLQ.

    Typical usage::

        mgr = WebhookManager()
        wid = mgr.register("https://example.com/hook", {"job.completed"}, "sekret")
        mgr.dispatch("job.completed", {"job_id": "abc"})
        mgr.retry_failed()
        failed = mgr.get_dlq()
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._lock = RLock()
        self._registrations: dict[str, WebhookRegistration] = {}
        self._delivery_log: list[WebhookDelivery] = []
        self._dead_letter_queue: list[WebhookDelivery] = []
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._clock = clock or _now_ts

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        url: str,
        events: set[str] | list[str],
        secret: str,
    ) -> str:
        """Register a new webhook subscriber.

        Args:
            url: Target URL that will receive POST requests.  Must pass the
                SSRF guard (:func:`distllm.core.webhook_manager.is_safe_webhook_url`)
                — cloud metadata, loopback, private/link-local hosts are
                rejected unless present in ``DISTLLM_WEBHOOK_ALLOWLIST``.
            events: Event types to subscribe to (e.g. ``{"job.completed"}``).
            secret: Shared secret used for HMAC signing.

        Returns:
            The unique webhook ID.

        Raises:
            UnsafeWebhookURLError: If *url* fails the SSRF safety check.
        """
        if not is_safe_webhook_url(url):
            logger.warning("Rejected unsafe webhook URL registration: %s", url)
            raise UnsafeWebhookURLError(
                f"Webhook URL rejected by SSRF guard (private/loopback/metadata "
                f"hosts are not allowed unless DISTLLM_WEBHOOK_ALLOWLIST includes them): {url}"
            )
        events_set = set(events)
        webhook_id = uuid.uuid4().hex
        reg = WebhookRegistration(
            id=webhook_id,
            url=url,
            events=events_set,
            secret=secret,
            created_at=_now_iso(),
        )
        with self._lock:
            self._registrations[webhook_id] = reg
        logger.info("Registered webhook %s for %d event(s)", webhook_id, len(events_set))
        return webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Remove a webhook registration.

        Returns:
            True if the registration was found and removed.
        """
        with self._lock:
            if webhook_id not in self._registrations:
                return False
            del self._registrations[webhook_id]
        logger.info("Unregistered webhook %s", webhook_id)
        return True

    def list(self) -> list[WebhookRegistration]:
        """Return a snapshot of all registered webhooks."""
        with self._lock:
            return list(self._registrations.values())

    def get(self, webhook_id: str) -> WebhookRegistration | None:
        """Return a single registration by ID, or None."""
        with self._lock:
            return self._registrations.get(webhook_id)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        event: str,
        data: dict[str, Any],
    ) -> list[str]:
        """Dispatch *event* with *data* to all matching webhooks.

        This method is *fire-and-forget* for each matched webhook -- it
        spawns the delivery attempt and does not await the response.
        Callers that need guaranteed ordering should call this from an
        async context that awaits the returned coroutines.

        Returns:
            List of webhook IDs that matched (for informational purposes).
        """
        payload = _build_payload(event, data)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        with self._lock:
            matched = [
                reg
                for reg in self._registrations.values()
                if reg.is_active and reg.matches(event)
            ]

        for reg in matched:
            delivery = WebhookDelivery(
                id=uuid.uuid4().hex,
                webhook_id=reg.id,
                event=event,
                payload=payload,
                attempt=1,
                status=WebhookDeliveryStatus.PENDING,
            )
            self._submit_delivery(delivery, reg, body)

        return [reg.id for reg in matched]

    def _submit_delivery(
        self,
        delivery: WebhookDelivery,
        reg: WebhookRegistration,
        body: bytes,
    ) -> None:
        """Attempt one delivery and schedule retry on failure."""
        # Re-validate at delivery time: a URL that was safe at registration
        # could have been mutated afterwards (persistent-store tampering, an
        # update path that skipped the guard).  Fail closed — no request is
        # made; the delivery is dead-lettered.
        if not is_safe_webhook_url(reg.url):
            delivery.error = "URL blocked by delivery-time SSRF re-validation"
            delivery.status = WebhookDeliveryStatus.DEAD
            with self._lock:
                self._dead_letter_queue.append(delivery)
                self._delivery_log.append(delivery)
            logger.warning(
                "Webhook %s URL failed delivery-time SSRF re-validation: %s",
                delivery.id, reg.url,
            )
            return

        signature = _compute_signature(reg.secret, body)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Id": reg.id,
            "X-Webhook-Delivery": delivery.id,
        }

        try:
            resp = httpx.post(
                reg.url,
                content=body,
                headers=headers,
                timeout=30.0,
                follow_redirects=False,
            )
            if 200 <= resp.status_code < 300:
                delivery.status = WebhookDeliveryStatus.SUCCESS
                with self._lock:
                    self._delivery_log.append(delivery)
                logger.debug("Delivered webhook %s -> %s (200)", delivery.id, reg.url)
                return
            # Non-2xx response
            delivery.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except httpx.RequestError as exc:
            delivery.error = str(exc)

        # Failed -- schedule retry or move to DLQ
        if delivery.attempt < _MAX_RETRIES:
            delay = _RETRY_DELAYS[delivery.attempt]
            delivery.status = WebhookDeliveryStatus.FAILED
            delivery.next_retry = self._clock() + delay
            with self._lock:
                self._delivery_log.append(delivery)
            logger.info(
                "Webhook %s attempt %d/%d failed, retry in %.0fs: %s",
                delivery.id,
                delivery.attempt,
                _MAX_RETRIES,
                delay,
                delivery.error,
            )
        else:
            delivery.status = WebhookDeliveryStatus.DEAD
            with self._lock:
                self._dead_letter_queue.append(delivery)
                self._delivery_log.append(delivery)
            logger.warning(
                "Webhook %s max retries reached, moved to DLQ: %s",
                delivery.id,
                delivery.error,
            )

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def retry_failed(self) -> int:
        """Retry all failed deliveries whose ``next_retry`` time has passed.

        Returns:
            Number of deliveries that were retried.
        """
        now = self._clock()
        to_retry: list[WebhookDelivery] = []
        remaining: list[WebhookDelivery] = []

        with self._lock:
            # Collect failed deliveries whose retry timer has expired.
            for d in self._delivery_log:
                if (
                    d.status == WebhookDeliveryStatus.FAILED
                    and d.next_retry is not None
                    and now >= d.next_retry
                ):
                    to_retry.append(d)
                else:
                    remaining.append(d)

            if not to_retry:
                return 0

            self._delivery_log = remaining

        for d in to_retry:
            reg = self.get(d.webhook_id)
            if reg is None or not reg.is_active:
                # Registration gone -- mark dead immediately.
                d.status = WebhookDeliveryStatus.DEAD
                d.error = "Registration removed or deactivated"
                with self._lock:
                    self._dead_letter_queue.append(d)
                continue

            next_attempt = d.attempt + 1
            new_delivery = WebhookDelivery(
                id=d.id,
                webhook_id=d.webhook_id,
                event=d.event,
                payload=d.payload,
                attempt=next_attempt,
                status=WebhookDeliveryStatus.PENDING,
            )
            body = json.dumps(d.payload, separators=(",", ":")).encode("utf-8")
            self._submit_delivery(new_delivery, reg, body)

        return len(to_retry)

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def get_dlq(self) -> list[WebhookDelivery]:
        """Return a snapshot of all deliveries in the dead-letter queue."""
        with self._lock:
            return list(self._dead_letter_queue)

    def replay_dlq(self) -> int:
        """Re-queue all dead-letter deliveries for fresh delivery attempts.

        Returns:
            Number of deliveries re-queued.
        """
        with self._lock:
            requeued = list(self._dead_letter_queue)
            self._dead_letter_queue.clear()

        for d in requeued:
            reg = self.get(d.webhook_id)
            if reg is None or not reg.is_active:
                continue
            new_delivery = WebhookDelivery(
                id=d.id,
                webhook_id=d.webhook_id,
                event=d.event,
                payload=d.payload,
                attempt=1,
                status=WebhookDeliveryStatus.PENDING,
            )
            body = json.dumps(d.payload, separators=(",", ":")).encode("utf-8")
            self._submit_delivery(new_delivery, reg, body)

        return len(requeued)

    def clear_dlq(self) -> int:
        """Remove all entries from the dead-letter queue.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._dead_letter_queue)
            self._dead_letter_queue.clear()
        return count

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_delivery_log(
        self,
        *,
        limit: int = 100,
        webhook_id: str | None = None,
    ) -> list[WebhookDelivery]:
        """Return recent delivery log entries, newest first."""
        with self._lock:
            log = list(self._delivery_log)
        log.reverse()
        if webhook_id is not None:
            log = [d for d in log if d.webhook_id == webhook_id]
        return log[:limit]

    @property
    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the webhook system."""
        with self._lock:
            total = len(self._delivery_log)
            succeeded = sum(
                1 for d in self._delivery_log if d.status == WebhookDeliveryStatus.SUCCESS
            )
            failed = sum(
                1 for d in self._delivery_log if d.status == WebhookDeliveryStatus.FAILED
            )
            dead = len(self._dead_letter_queue)
            registrations = len(self._registrations)
        return {
            "registrations": registrations,
            "total_deliveries": total,
            "succeeded": succeeded,
            "failed": failed,
            "dead_letter": dead,
        }


# ---------------------------------------------------------------------------
# One-shot dispatch helper
# ---------------------------------------------------------------------------

async def dispatch_webhook(
    url: str,
    secret: str,
    event: str,
    data: dict[str, Any],
    timeout: float = 30.0,
) -> bool:
    """Deliver a single webhook event to *url* (HMAC-signed).

    The URL is re-validated by the SSRF guard immediately before the POST —
    this is the last line of defense even if registration-time checks were
    bypassed or the stored URL was mutated afterwards.  Redirects are NOT
    followed (a 3xx response counts as failure).

    Returns:
        True if the endpoint responded 2xx, False otherwise (including when
        *url* fails the SSRF guard).
    """
    if not is_safe_webhook_url(url):
        logger.warning("Blocked webhook dispatch to unsafe URL: %s", url)
        return False

    payload = _build_payload(event, data)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": _compute_signature(secret, body),
    }

    try:
        # httpx.AsyncClient follows redirects by default; disable explicitly.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.post(url, content=body, headers=headers)
        ok = 200 <= resp.status_code < 300
        if not ok:
            logger.debug("Webhook %s -> %s returned HTTP %d", event, url, resp.status_code)
        return ok
    except httpx.RequestError as exc:
        logger.info("Webhook %s -> %s delivery failed: %s", event, url, exc)
        return False
