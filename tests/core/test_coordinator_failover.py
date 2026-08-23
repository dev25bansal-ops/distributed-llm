"""Tests for CoordinatorFailoverHandler -- TCP-based coordinator failover detection.

Covers:
- Construction and initial state
- start/stop lifecycle
- stats and property access
- update_peer_hosts
- Manual failover trigger via _trigger_failover (no sockets)

No MagicMock -- real threading primitives and counters.
"""

from __future__ import annotations

import threading
import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/coordinator_failover.py")
CoordinatorFailoverHandler = _mod.CoordinatorFailoverHandler


class TestFailoverConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        assert handler._coordinator_host == "10.0.0.1"
        assert handler._coordinator_port == 50050
        assert handler._peer_hosts == []
        assert handler._consecutive_failures == 0
        assert handler._failover_count == 0
        assert handler._running is False

    def test_current_coordinator_property(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        assert handler.current_coordinator == ("10.0.0.1", 50050)

    def test_failover_count_starts_zero(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        assert handler.failover_count == 0

    def test_stats_default_state(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        s = handler.stats()
        assert s["current_coordinator"] == "10.0.0.1:50050"
        assert s["consecutive_failures"] == 0
        assert s["failover_count"] == 0
        assert s["peer_count"] == 0
        assert s["running"] is False


class TestFailoverStartStop:
    """Start and stop lifecycle."""

    def test_start_sets_running_and_creates_thread(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        handler.start()
        assert handler._running is True
        assert handler._thread is not None
        assert handler._thread.is_alive()
        handler.stop()

    def test_start_is_idempotent(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        handler.start()
        thread_id = id(handler._thread)
        handler.start()  # second start should be no-op
        assert id(handler._thread) == thread_id
        handler.stop()

    def test_stop_clears_running(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        handler.start()
        handler.stop()
        assert handler._running is False

    def test_stop_joins_thread(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        handler.start()
        handler.stop()
        assert handler._thread is not None
        assert not handler._thread.is_alive()


class TestFailoverPeerManagement:
    """Peer host list management."""

    def test_update_peer_hosts(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )
        handler.update_peer_hosts([("10.0.0.2", 50050), ("10.0.0.3", 50050)])
        assert len(handler._peer_hosts) == 2
        assert ("10.0.0.2", 50050) in handler._peer_hosts

    def test_update_peer_hosts_replaces(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            peer_hosts=[("10.0.0.2", 50050)],
        )
        handler.update_peer_hosts([("10.0.0.4", 50050)])
        assert len(handler._peer_hosts) == 1
        assert handler._peer_hosts[0] == ("10.0.0.4", 50050)


class TestFailoverTrigger:
    """Failover triggering logic (no real sockets)."""

    def test_trigger_failover_no_peers(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            failure_threshold=1,
        )
        handler._consecutive_failures = 1
        handler._trigger_failover()
        assert handler.current_coordinator == ("10.0.0.1", 50050)
        assert handler.failover_count == 0

    def test_trigger_failover_skips_current_coordinator(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            peer_hosts=[("10.0.0.1", 50050)],
            failure_threshold=1,
        )
        handler._consecutive_failures = 1
        handler._trigger_failover()
        assert handler.current_coordinator == ("10.0.0.1", 50050)
        assert handler.failover_count == 0

    def test_consecutive_failure_tracking(self) -> None:
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            failure_threshold=3,
        )
        handler._consecutive_failures = 2
        assert handler.stats()["consecutive_failures"] == 2
