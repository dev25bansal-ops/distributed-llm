"""Tests for the EventBus publish/subscribe system.

Covers: subscribe, unsubscribe, publish, replay, persistence, thread safety.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

from distllm.core.event_bus import EventBus, MarketplaceEvent


def _evt(event_type: str, **payload: object) -> MarketplaceEvent:
    """Helper to create a MarketplaceEvent with auto-generated fields."""
    return MarketplaceEvent(
        event_id=f"evt-{time.time_ns()}",
        event_type=event_type,
        payload=payload,
        timestamp=time.time(),
    )


def _handler(name: str = "h"):
    h = Mock()
    h.__qualname__ = name
    return h


class TestSubscribe:
    def test_subscribe_and_dispatch(self):
        bus = EventBus()
        h = _handler()
        bus.subscribe("job.matched", h)
        bus.publish("job.matched", {"job_id": "j1"})
        h.assert_called_once()
        assert h.call_args[0][0].payload["job_id"] == "j1"

    def test_wildcard(self):
        bus = EventBus()
        h = _handler()
        bus.subscribe("*", h)
        bus.publish("a")
        bus.publish("b")
        assert h.call_count == 2

    def test_multi_handler(self):
        bus = EventBus()
        h1, h2 = _handler("h1"), _handler("h2")
        bus.subscribe("x", h1)
        bus.subscribe("x", h2)
        bus.publish("x")
        h1.assert_called_once()
        h2.assert_called_once()

    def test_unsubscribe(self):
        bus = EventBus()
        h = _handler()
        bus.subscribe("x", h)
        bus.unsubscribe("x", h)
        bus.publish("x")
        h.assert_not_called()

    def test_unsubscribe_nonexistent(self):
        EventBus().unsubscribe("x", lambda e: None)

    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe("a", _handler())
        bus.subscribe("a", _handler())
        bus.subscribe("b", _handler())
        assert bus.subscriber_count("a") == 2
        assert bus.subscriber_count("b") == 1
        assert bus.subscriber_count() == 3


class TestPublish:
    def test_no_subscribers(self):
        EventBus().publish("x")

    def test_to_dict(self):
        d = _evt("x", count=3).to_dict()
        assert d["event_type"] == "x"
        assert d["payload"]["count"] == 3
        assert "event_id" in d

    def test_sync_handler(self):
        bus = EventBus()
        got = []
        bus.subscribe("x", lambda e: got.append(e.event_type))
        bus.publish("x")
        assert got == ["x"]


class TestPersistence:
    def test_replay(self):
        bus = EventBus()
        bus.publish("a")
        bus.publish("b")
        events = bus.replay(limit=10)
        assert len(events) == 2

    def test_replay_limit(self):
        bus = EventBus()
        for i in range(20):
            bus.publish(f"e{i}")
        assert len(bus.replay(limit=5)) == 5

    def test_event_count(self):
        bus = EventBus()
        assert bus.event_count() == 0
        bus.publish("x")
        assert bus.event_count() == 1

    def test_clear_log(self):
        bus = EventBus()
        for _ in range(5):
            bus.publish("x")
        assert bus.clear_log() == 5
        assert bus.event_count() == 0


class TestThreadSafety:
    def test_concurrent_publish(self):
        bus = EventBus()
        results = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                results.append(event.event_type)

        bus.subscribe("*", handler)
        errors = []

        def pub():
            try:
                for _ in range(20):
                    bus.publish("t")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pub) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors, f"Errors: {errors}"
        assert len(results) == 100


class TestWebhook:
    def test_no_webhook_noop(self):
        EventBus().publish("x")

    def test_with_webhook(self):
        mgr = Mock()
        bus = EventBus(webhook_manager=mgr)
        bus.subscribe("x", lambda e: None)
        bus.publish("x")
