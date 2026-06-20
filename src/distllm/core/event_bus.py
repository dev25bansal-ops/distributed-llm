"""Event Bus for marketplace job lifecycle events.

Provides publish/subscribe semantics with:
- Typed event categories (job.matched, job.started, etc.)
- Webhook delivery via existing WebhookManager
- Exponential backoff retry (max 5 attempts)
- Event persistence for replay
- Thread-safe with asyncio support

Usage:
    bus = EventBus(webhook_manager=wh)
    bus.subscribe("job.completed", my_handler)
    bus.publish("job.completed", {"job_id": "job-abc", "tokens": 1024})

    # Replay missed events
    for event in bus.replay(since=time.time() - 3600):
        process(event)
"""

from __future__ import annotations

import asyncio
import enum
import json
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger


# ── Event Types ──────────────────────────────────────────────────────────────

class MarketplaceEventType(enum.Enum):
    """Marketplace lifecycle event types."""
    JOB_MATCHED = "job.matched"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    LISTING_STATUS_CHANGED = "listing.status_changed"


# ── Event Data ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarketplaceEvent:
    """An immutable marketplace event record.

    Attributes:
        event_id: Unique identifier for deduplication and replay ordering.
        event_type: The event category (e.g. "job.matched").
        payload: Arbitrary event data. Must be JSON-serializable.
        timestamp: Unix epoch seconds when the event was created.
        source: Originating component (e.g. "marketplace").
    """
    event_id: str
    event_type: str
    payload: dict[str, Any]
    timestamp: float
    source: str = "marketplace"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


# ── Type Aliases ─────────────────────────────────────────────────────────────

# Synchronous handler: (event) -> None
SyncHandler = Callable[[MarketplaceEvent], None]
# Async handler: (event) -> None (coroutine)
AsyncHandler = Callable[[MarketplaceEvent], Coroutine[Any, Any, None]]


# ── Event Bus ────────────────────────────────────────────────────────────────

class EventBus:
    """Publish/subscribe event bus for marketplace lifecycle events.

    Thread-safe. Supports both synchronous and async subscribers.
    Integrates with WebhookManager for HTTP webhook delivery.

    Args:
        webhook_manager: Optional WebhookManager for HTTP delivery.
        max_retries: Maximum webhook delivery retry attempts (default 5).
        backoff_base: Base seconds for exponential backoff (default 1.0).
        persistence_max: Maximum events kept in the replay buffer (default 10000).
    """

    MAX_RETRIES: int = 5
    BACKOFF_BASE: float = 1.0
    PERSISTENCE_MAX: int = 10_000

    def __init__(
        self,
        webhook_manager: Any | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        persistence_max: int = 10_000,
    ) -> None:
        self._webhook_manager = webhook_manager
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._persistence_max = persistence_max

        # Subscribers: event_type -> list of handlers
        self._sync_subscribers: dict[str, list[SyncHandler]] = defaultdict(list)
        self._async_subscribers: dict[str, list[AsyncHandler]] = defaultdict(list)

        # Event persistence ring buffer
        self._event_log: list[MarketplaceEvent] = []
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()

        # Async delivery loop state
        self._async_queue: asyncio.Queue[MarketplaceEvent] | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_task: asyncio.Task[None] | None = None
        self._async_running = False

    # ── Subscribe ────────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: SyncHandler | AsyncHandler,
    ) -> None:
        """Register a handler for a specific event type.

        Use "*" as event_type to subscribe to all events.

        Args:
            event_type: Event name (e.g. "job.matched") or "*" for all.
            handler: Callable that receives a MarketplaceEvent.
        """
        if asyncio.iscoroutinefunction(handler):
            with self._lock:
                self._async_subscribers[event_type].append(handler)
        else:
            with self._lock:
                self._sync_subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to '{event_type}': {handler.__qualname__}")

    def unsubscribe(
        self,
        event_type: str,
        handler: SyncHandler | AsyncHandler,
    ) -> bool:
        """Remove a handler from an event type. Returns True if removed."""
        with self._lock:
            if asyncio.iscoroutinefunction(handler):
                handlers = self._async_subscribers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)
                    return True
            else:
                handlers = self._sync_subscribers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)
                    return True
        return False

    def subscriber_count(self, event_type: str | None = None) -> int:
        """Count subscribers. If event_type is None, count all."""
        with self._lock:
            if event_type is not None:
                return (
                    len(self._sync_subscribers.get(event_type, []))
                    + len(self._async_subscribers.get(event_type, []))
                    + len(self._sync_subscribers.get("*", []))
                    + len(self._async_subscribers.get("*", []))
                )
            total = 0
            for handlers in self._sync_subscribers.values():
                total += len(handlers)
            for handlers in self._async_subscribers.values():
                total += len(handlers)
            return total

    # ── Publish ──────────────────────────────────────────────────────────────

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> MarketplaceEvent:
        """Publish an event synchronously.

        Dispatches to sync subscribers and webhooks immediately.
        Queues async subscribers for the background loop.

        Args:
            event_type: The event name (e.g. "job.matched").
            payload: JSON-serializable event data.

        Returns:
            The created MarketplaceEvent.
        """
        event = MarketplaceEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            payload=payload or {},
            timestamp=time.time(),
        )

        # Persist
        self._persist(event)

        # Dispatch sync subscribers (exact match + wildcard)
        with self._lock:
            sync_handlers = list(self._sync_subscribers.get(event_type, []))
            sync_handlers += list(self._sync_subscribers.get("*", []))

        for handler in sync_handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    f"Sync handler {handler.__qualname__} failed for {event_type}"
                )

        # Dispatch async subscribers via queue
        with self._lock:
            has_async = (
                bool(self._async_subscribers.get(event_type))
                or bool(self._async_subscribers.get("*"))
            )
        if has_async and self._async_queue is not None:
            try:
                self._async_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Async event queue full, dropping {event_type}")

        # Dispatch to webhooks
        self._dispatch_webhook(event)

        logger.debug(f"Published {event_type} (id={event.event_id[:8]})")
        return event

    async def publish_async(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> MarketplaceEvent:
        """Publish an event from an async context.

        Same as publish() but awaits async handlers directly.
        """
        event = MarketplaceEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            payload=payload or {},
            timestamp=time.time(),
        )

        self._persist(event)

        # Sync handlers
        with self._lock:
            sync_handlers = list(self._sync_subscribers.get(event_type, []))
            sync_handlers += list(self._sync_subscribers.get("*", []))

        for handler in sync_handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    f"Sync handler {handler.__qualname__} failed for {event_type}"
                )

        # Async handlers -- await directly
        with self._lock:
            async_handlers = list(self._async_subscribers.get(event_type, []))
            async_handlers += list(self._async_subscribers.get("*", []))

        for handler in async_handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    f"Async handler {handler.__qualname__} failed for {event_type}"
                )

        self._dispatch_webhook(event)

        logger.debug(f"Published async {event_type} (id={event.event_id[:8]})")
        return event

    # ── Webhook Delivery ─────────────────────────────────────────────────────

    def _dispatch_webhook(self, event: MarketplaceEvent) -> None:
        """Send event to WebhookManager if configured."""
        if self._webhook_manager is None:
            return
        try:
            self._webhook_manager.dispatch(
                _event_type_to_webhook_enum(event.event_type),
                event.to_dict(),
            )
        except Exception:
            logger.exception(f"Webhook dispatch failed for {event.event_type}")

    def deliver_webhook_with_retry(
        self,
        url: str,
        event: MarketplaceEvent,
        secret: str = "",
        timeout_s: float = 5.0,
    ) -> bool:
        """Deliver a single event to a URL with exponential backoff retry.

        Attempts up to self._max_retries times. Uses the same backoff
        formula as WebhookManager: delay = base * 2^(attempt-1).

        Args:
            url: HTTP endpoint.
            event: The event to deliver.
            secret: HMAC signing secret.
            timeout_s: HTTP timeout per attempt.

        Returns:
            True if delivery succeeded (HTTP < 400).
        """
        import hashlib
        import hmac as hmac_mod
        import json as json_mod

        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, webhook delivery skipped")
            return False

        body = json_mod.dumps(event.to_dict(), default=str).encode("utf-8")
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if secret:
            sig = hmac_mod.new(
                secret.encode(), body, hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = sig

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = httpx.post(
                    url, content=body, headers=headers, timeout=timeout_s,
                )
                if resp.status_code < 400:
                    logger.debug(
                        f"Webhook delivered to {url} on attempt {attempt}"
                    )
                    return True
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)

            if attempt < self._max_retries:
                delay = self._backoff_base * (2 ** (attempt - 1))
                time.sleep(min(delay, 60.0))

        logger.warning(
            f"Webhook delivery to {url} failed after "
            f"{self._max_retries} attempts: {last_error}"
        )
        return False

    # ── Async Background Loop ───────────────────────────────────────────────

    def start_async_loop(self) -> None:
        """Start a background thread running an asyncio event loop.

        The loop drains async subscriber queues.
        """
        if self._async_running:
            return

        self._async_running = True
        self._async_queue = asyncio.Queue(maxsize=10_000)

        thread = threading.Thread(
            target=self._run_async_loop, daemon=True, name="event-bus-async",
        )
        thread.start()
        logger.debug("EventBus async loop started")

    def stop_async_loop(self) -> None:
        """Signal the async loop to stop."""
        self._async_running = False
        if self._async_loop and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        logger.debug("EventBus async loop stopped")

    def _run_async_loop(self) -> None:
        """Target for the background thread."""
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        self._async_task = self._async_loop.create_task(self._async_drain())
        try:
            self._async_loop.run_forever()
        finally:
            self._async_loop.close()
            self._async_loop = None

    async def _async_drain(self) -> None:
        """Continuously drain the async event queue."""
        assert self._async_queue is not None
        while self._async_running:
            try:
                event = await asyncio.wait_for(
                    self._async_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            event_type = event.event_type
            with self._lock:
                handlers = list(self._async_subscribers.get(event_type, []))
                handlers += list(self._async_subscribers.get("*", []))

            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    logger.exception(
                        f"Async handler {handler.__qualname__} failed "
                        f"for {event_type}"
                    )

    # ── Persistence & Replay ─────────────────────────────────────────────────

    def _persist(self, event: MarketplaceEvent) -> None:
        """Append event to the in-memory ring buffer."""
        with self._log_lock:
            self._event_log.append(event)
            # Evict oldest when over capacity
            if len(self._event_log) > self._persistence_max:
                self._event_log = self._event_log[-self._persistence_max:]

    def replay(
        self,
        since: float = 0.0,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[MarketplaceEvent]:
        """Replay persisted events matching the filters.

        Args:
            since: Minimum timestamp (exclusive). 0 = no lower bound.
            event_type: Filter by event type. None = all types.
            limit: Maximum events to return.

        Returns:
            List of MarketplaceEvent ordered by timestamp ascending.
        """
        with self._log_lock:
            events = list(self._event_log)

        if since > 0:
            events = [e for e in events if e.timestamp > since]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events[:limit]

    def event_count(self) -> int:
        """Number of persisted events."""
        with self._log_lock:
            return len(self._event_log)

    def clear_log(self) -> int:
        """Clear the event log. Returns number of events cleared."""
        with self._log_lock:
            count = len(self._event_log)
            self._event_log.clear()
            return count


# ── Helpers ──────────────────────────────────────────────────────────────────

def _event_type_to_webhook_enum(event_type: str) -> Any:
    """Map a MarketplaceEventType string to a WebhookEvent enum member.

    Falls back to a generic event if no mapping exists.
    """
    # Import here to avoid circular dependency at module level
    from distllm.core.webhook_manager import WebhookEvent

    _MAP: dict[str, WebhookEvent] = {
        "job.matched": WebhookEvent.INFERENCE_STARTED,
        "job.started": WebhookEvent.INFERENCE_STARTED,
        "job.completed": WebhookEvent.INFERENCE_COMPLETED,
        "job.failed": WebhookEvent.INFERENCE_ERROR,
        "job.cancelled": WebhookEvent.INFERENCE_COMPLETED,
        "listing.status_changed": WebhookEvent.CLUSTER_HEALTH_CHANGED,
    }
    return _MAP.get(event_type, WebhookEvent.THRESHOLD_BREACHED)
