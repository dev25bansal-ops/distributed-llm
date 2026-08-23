"""Tests for distllm.dist.federation — zero mocks, real objects only."""

from __future__ import annotations

import time
import pytest
from pydantic import ValidationError

from distllm.dist.federation import (
    FederationConfig,
    BreakerState,
    PeerBreakerEntry,
    PeerCircuitBreaker,
    FederationCoordinator,
)


class _FakeCoordinatorConfig:
    """Minimal coordinator config stub for TLS / cluster-key tests.

    Not a mock — a real object with the attributes
    FederationCoordinator.__init__ inspects.
    """

    def __init__(self) -> None:
        self.cluster_key: str | None = None
        self.tls = _FakeTLSConfig()


class _FakeTLSConfig:
    enabled: bool = True


class _FakeCoordinatorRef:
    """Minimal coordinator reference stub.

    Only carries the ``config`` attribute that
    FederationCoordinator.__init__ checks.
    """

    def __init__(self, cluster_key: str | None = None) -> None:
        self.config = _FakeCoordinatorConfig()
        self.config.cluster_key = cluster_key


# ── FederationConfig ──────────────────────────────────────────────────────────


class TestFederationConfig:
    """Pydantic settings — env_prefix, constraints, defaults."""

    def test_defaults(self) -> None:
        cfg = FederationConfig()
        assert cfg.enabled is False
        assert cfg.cluster_id == "default"
        assert cfg.listen_host == "0.0.0.0"
        assert cfg.listen_port == 50060
        assert cfg.seed_nodes == []
        assert cfg.discovery_interval_s == 30.0
        assert cfg.heartbeat_interval_s == 15.0
        assert cfg.spillover_enabled is True
        assert cfg.spillover_threshold_gpu_util == 80.0
        assert cfg.circuit_breaker_threshold == 5
        assert cfg.circuit_breaker_reset_s == 60.0
        assert cfg.cache_digest_ttl_s == 300.0
        assert cfg.gossip_enabled is False
        assert cfg.gossip_fanout == 3

    def test_custom_values(self) -> None:
        cfg = FederationConfig(
            enabled=True,
            cluster_id="us-east",
            listen_port=55000,
            seed_nodes=["10.0.0.1:50050"],
            discovery_interval_s=10.0,
            heartbeat_interval_s=5.0,
            spillover_threshold_gpu_util=90.0,
            circuit_breaker_threshold=3,
            circuit_breaker_reset_s=30.0,
            cache_digest_ttl_s=120.0,
            gossip_enabled=True,
            gossip_fanout=5,
        )
        assert cfg.enabled is True
        assert cfg.cluster_id == "us-east"
        assert cfg.listen_port == 55000
        assert cfg.seed_nodes == ["10.0.0.1:50050"]

    def test_port_constraints(self) -> None:
        with pytest.raises(ValidationError):
            FederationConfig(listen_port=0)
        with pytest.raises(ValidationError):
            FederationConfig(listen_port=65536)

    def test_gt_zero_constraints(self) -> None:
        with pytest.raises(ValidationError):
            FederationConfig(discovery_interval_s=-1)
        with pytest.raises(ValidationError):
            FederationConfig(heartbeat_interval_s=0)

    def test_gpu_util_range(self) -> None:
        with pytest.raises(ValidationError):
            FederationConfig(spillover_threshold_gpu_util=-1)
        with pytest.raises(ValidationError):
            FederationConfig(spillover_threshold_gpu_util=101)

    def test_circuit_breaker_constraints(self) -> None:
        cfg = FederationConfig(circuit_breaker_threshold=1)
        assert cfg.circuit_breaker_threshold == 1
        with pytest.raises(ValidationError):
            FederationConfig(circuit_breaker_threshold=0)
        with pytest.raises(ValidationError):
            FederationConfig(circuit_breaker_reset_s=0)

    def test_frozen(self) -> None:
        cfg = FederationConfig()
        with pytest.raises(ValidationError):
            cfg.enabled = True  # type: ignore[misc]

    def test_env_prefix(self) -> None:
        """Simulate FEDERATION_* environment variable override."""
        import os

        os.environ["FEDERATION_CLUSTER_ID"] = "env-cluster"
        os.environ["FEDERATION_ENABLED"] = "true"
        os.environ["FEDERATION_HEARTBEAT_INTERVAL_S"] = "42"
        try:
            cfg = FederationConfig()
            assert cfg.cluster_id == "env-cluster"
            assert cfg.enabled is True
            assert cfg.heartbeat_interval_s == 42.0
        finally:
            del os.environ["FEDERATION_CLUSTER_ID"]
            del os.environ["FEDERATION_ENABLED"]
            del os.environ["FEDERATION_HEARTBEAT_INTERVAL_S"]


# ── BreakerState ──────────────────────────────────────────────────────────────


class TestBreakerState:
    def test_values(self) -> None:
        assert BreakerState.CLOSED.value == "closed"
        assert BreakerState.HALF_OPEN.value == "half_open"
        assert BreakerState.OPEN.value == "open"

    def test_membership(self) -> None:
        assert BreakerState("closed") is BreakerState.CLOSED
        assert BreakerState("half_open") is BreakerState.HALF_OPEN
        assert BreakerState("open") is BreakerState.OPEN


# ── PeerBreakerEntry ──────────────────────────────────────────────────────────


class TestPeerBreakerEntry:
    def test_defaults(self) -> None:
        entry = PeerBreakerEntry()
        assert entry.state is BreakerState.CLOSED
        assert entry.failure_count == 0
        assert entry.open_until == 0.0
        assert entry.half_open_probe_ok is False
        assert entry.consecutive_successes == 0

    def test_is_timed_out_not_set(self) -> None:
        entry = PeerBreakerEntry()
        assert entry.is_timed_out(time.time()) is False

    def test_is_timed_out_past(self) -> None:
        entry = PeerBreakerEntry(open_until=time.time() - 10)
        assert entry.is_timed_out(time.time()) is True

    def test_is_timed_out_future(self) -> None:
        entry = PeerBreakerEntry(open_until=time.time() + 60)
        assert entry.is_timed_out(time.time()) is False

    def test_is_timed_out_zero(self) -> None:
        """open_until == 0 means not set — never timed out."""
        entry = PeerBreakerEntry(open_until=0.0)
        assert entry.is_timed_out(time.time()) is False


# ── PeerCircuitBreaker ────────────────────────────────────────────────────────


class TestPeerCircuitBreaker:
    """State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    def test_initial_state(self) -> None:
        cb = PeerCircuitBreaker(threshold=3, reset_s=60.0)
        # is_open calls _get_entry which lazily creates the entry.
        assert cb.is_open("peer-1") is False
        state = cb.get_state()
        # The entry was created by the is_open call above.
        assert len(state["tracked_peers"]) == 1
        assert state["tracked_peers"][0]["peer_id"] == "peer-1"
        assert state["tracked_peers"][0]["state"] == "closed"
        assert state["tracked_peers"][0]["failures"] == 0

    def test_closed_normally(self) -> None:
        """No failures — circuit stays closed."""
        cb = PeerCircuitBreaker(threshold=3, reset_s=60.0)
        assert cb.is_open("peer-a") is False
        cb.record_success("peer-a")
        assert cb.is_open("peer-a") is False
        state = cb.get_state()
        assert state["open_breakers"] == []

    def test_closed_to_open_on_threshold(self) -> None:
        cb = PeerCircuitBreaker(threshold=3, reset_s=60.0)
        cb.record_failure("peer-a")
        assert cb.is_open("peer-a") is False
        cb.record_failure("peer-a")
        assert cb.is_open("peer-a") is False
        cb.record_failure("peer-a")  # 3rd failure -> OPEN
        assert cb.is_open("peer-a") is True
        state = cb.get_state()
        assert "peer-a" in state["open_breakers"]

    def test_open_to_half_open_on_timeout(self) -> None:
        cb = PeerCircuitBreaker(threshold=2, reset_s=0.01)
        cb.record_failure("peer-b")
        cb.record_failure("peer-b")
        assert cb.is_open("peer-b") is True
        time.sleep(0.02)
        # Now the reset window should have elapsed -> HALF_OPEN -> is_open returns False
        assert cb.is_open("peer-b") is False  # transitions to HALF_OPEN
        state = cb.get_state()
        assert "peer-b" in state["half_open_breakers"]

    def test_half_open_success_closes_breaker(self) -> None:
        cb = PeerCircuitBreaker(threshold=2, reset_s=0.01, half_open_max=2)
        cb.record_failure("peer-c")
        cb.record_failure("peer-c")
        assert cb.is_open("peer-c") is True
        time.sleep(0.02)
        # Transition to HALF_OPEN
        assert cb.is_open("peer-c") is False
        # Probe successes should close
        cb.record_success("peer-c")
        cb.record_success("peer-c")
        assert cb.is_open("peer-c") is False
        state = cb.get_state()
        assert "peer-c" not in state["open_breakers"]
        assert "peer-c" not in state["half_open_breakers"]

    def test_half_open_failure_reopens(self) -> None:
        cb = PeerCircuitBreaker(threshold=2, reset_s=60.0, half_open_max=3)
        cb.record_failure("peer-d")
        cb.record_failure("peer-d")
        assert cb.is_open("peer-d") is True
        # Force time-forward: manually set entry to simulate timed-out transition
        entry = cb._entries.setdefault("peer-d", __import__("distllm.dist.federation", fromlist=["PeerBreakerEntry"]).PeerBreakerEntry())
        entry.state = BreakerState.HALF_OPEN
        entry.open_until = 0.0
        entry.consecutive_successes = 0
        cb.record_failure("peer-d")
        # Should go back to OPEN
        assert cb.is_open("peer-d") is True

    def test_force_open(self) -> None:
        cb = PeerCircuitBreaker(threshold=5, reset_s=60.0)
        cb.force_open("peer-e")
        assert cb.is_open("peer-e") is True
        state = cb.get_state()
        assert "peer-e" in state["open_breakers"]

    def test_success_resets_failures_in_closed_state(self) -> None:
        cb = PeerCircuitBreaker(threshold=3, reset_s=60.0)
        cb.record_failure("peer-f")
        cb.record_failure("peer-f")
        cb.record_success("peer-f")
        # failures reset to 0
        assert cb._entries["peer-f"].failure_count == 0
        assert cb.is_open("peer-f") is False

    def test_get_state_consistency(self) -> None:
        cb = PeerCircuitBreaker(threshold=2, reset_s=60.0)
        cb.record_failure("p1")
        cb.record_failure("p1")
        cb.record_failure("p2")
        state = cb.get_state()
        assert "p1" in state["open_breakers"]
        assert "p2" not in state["open_breakers"]  # only 1 failure
        assert len(state["tracked_peers"]) == 2

    def test_is_open_returns_false_for_unknown_peer(self) -> None:
        cb = PeerCircuitBreaker()
        assert cb.is_open("unknown") is False

    def test_half_open_allows_probes(self) -> None:
        """In HALF_OPEN state, is_open returns False to allow probe requests."""
        cb = PeerCircuitBreaker(threshold=2, reset_s=60.0)
        entry = cb._get_entry("probe-peer")
        entry.state = BreakerState.HALF_OPEN
        assert cb.is_open("probe-peer") is False


# ── FederationCoordinator ─────────────────────────────────────────────────────


class TestFederationCoordinatorConstruction:
    """Construction, basic properties, and pure-method paths."""

    def test_minimal_construction(self) -> None:
        cfg = FederationConfig()
        coord = FederationCoordinator(cfg, "test-cluster", "127.0.0.1", 8080)
        assert coord._local_cluster_id == "test-cluster"
        assert coord._scheme() == "http"
        assert coord.config is cfg

    def test_construction_with_none_coordinator(self) -> None:
        cfg = FederationConfig()
        coord = FederationCoordinator(cfg, "c1", "localhost", 9090, coordinator_ref=None)
        assert coord._coordinator is None
        assert coord._use_tls is False

    def test_scheme_http_by_default(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "x", "h", 1)
        assert coord._scheme() == "http"

    def test_scheme_https(self) -> None:
        """Construct with coordinator_ref containing TLS config to trigger https."""
        coord = FederationCoordinator(
            FederationConfig(), "x", "h", 1,
            coordinator_ref=_FakeCoordinatorRef(),
        )
        assert coord._scheme() == "https"


class TestFederationCoordinatorPeerMgmt:
    """get_peers, get_status, get_peer_slo_status, select_peer_for_sla."""

    def test_get_peers_empty(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_peers() == []

    def test_get_status_enabled_false(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        status = coord.get_status()
        assert status["enabled"] is False
        assert status["cluster_id"] == "default"
        assert status["peers"] == []
        assert status["spillover_enabled"] is True

    def test_get_peer_slo_status_empty(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_peer_slo_status() == []

    def test_select_peer_for_sla_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.select_peer_for_sla() is None

    def test_select_peer_for_sla_custom_params(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.select_peer_for_sla("high", 1000.0, 20.0) is None


class TestFederationCoordinatorSpillover:
    """should_spillover — depends on config and load metrics."""

    def test_disabled_spillover(self) -> None:
        cfg = FederationConfig(spillover_enabled=False)
        coord = FederationCoordinator(cfg, "c", "h", 1)
        assert coord.should_spillover() is False

    def test_no_peers(self) -> None:
        cfg = FederationConfig(spillover_enabled=True)
        coord = FederationCoordinator(cfg, "c", "h", 1)
        assert coord.should_spillover() is False

    def test_coordinator_ref_none_returns_false(self) -> None:
        cfg = FederationConfig(spillover_enabled=True)
        coord = FederationCoordinator(cfg, "c", "h", 1, coordinator_ref=None)
        # Without coordinator_ref, _get_local_load returns zero load,
        # so gpu_util (0.0) < threshold (80.0) => False
        assert coord.should_spillover() is False


class TestFederationCoordinatorCache:
    """Cache-related methods (non-network)."""

    def test_update_cache_digest_none(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord.update_cache_digest(None)
        assert coord._local_cache_digest is None

    def test_update_cache_digest_with_tokens(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord.update_cache_digest([1, 2, 3])
        digest = coord._local_cache_digest
        assert digest is not None
        assert digest["length"] == 3
        assert digest["cluster_id"] == "default"
        assert "hash" in digest
        assert "prefix_hash" in digest

    def test_get_best_peer_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_best_peer() is None

    def test_get_best_peer_with_digest_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_best_peer(prompt_digest={"hash": "abc"}) is None

    def test_get_peers_with_cache_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_peers_with_cache({"hash": "abc"}) == []

    def test_get_peers_with_cache_min_affinity_default(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        assert coord.get_peers_with_cache({"hash": "abc"}, min_affinity=0.5) == []

    def test_evict_stale_cache_digests_empty(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord._evict_stale_cache_digests()  # should not raise


class TestFederationCoordinatorMetrics:
    """Metrics bookkeeping."""

    def test_get_metrics_defaults(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        metrics = coord.get_metrics()
        assert metrics["total_forwards"] == 0
        assert metrics["forward_successes"] == 0
        assert metrics["forward_failures"] == 0
        assert metrics["peer_count"] == 0
        assert metrics["cache_digest_count"] == 0

    def test_record_metric(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord._record_metric("total_forwards", 3.0)
        assert coord.get_metrics()["total_forwards"] == 3.0

    def test_record_metric_unknown_name(self) -> None:
        """Unknown metric names are silently ignored."""
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord._record_metric("nonexistent", 1.0)
        # Should not raise and should not add a key
        assert "nonexistent" not in coord.get_metrics()


class TestFederationCoordinatorAuthAndTLS:
    """_build_auth_headers, _require_tls."""

    def test_auth_headers_no_keys(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        headers = coord._build_auth_headers()
        # No env vars set => empty dict
        assert headers == {}

    def test_require_tls_raises_without_tls(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        with pytest.raises(RuntimeError, match="TLS is required"):
            coord._require_tls()

    def test_require_tls_passes_with_tls(self) -> None:
        coord = FederationCoordinator(
            FederationConfig(), "c", "h", 1,
            coordinator_ref=_FakeCoordinatorRef(),
        )
        coord._require_tls()  # should not raise


class TestFederationCoordinatorStartStop:
    """start/stop/close — non-network lifecycle."""

    def test_start_disabled_does_nothing(self) -> None:
        """When config.enabled=False, start() returns immediately."""
        cfg = FederationConfig(enabled=False)
        coord = FederationCoordinator(cfg, "c", "h", 1)
        coord.start()  # should not raise
        coord.stop()   # should not raise

    def test_start_enabled_starts_thread_then_stop(self) -> None:
        """Enabled start spins up heartbeat (and optionally gossip) threads."""
        cfg = FederationConfig(enabled=True, gossip_enabled=False)
        coord = FederationCoordinator(cfg, "c", "h", 1)
        coord.start()
        assert coord._running.is_set()
        assert coord._heartbeat_thread is not None
        assert coord._heartbeat_thread.is_alive()
        coord.stop()
        assert not coord._running.is_set()
        # Thread should have been joined
        assert coord._heartbeat_thread is None

    def test_start_with_gossip(self) -> None:
        cfg = FederationConfig(enabled=True, gossip_enabled=True, seed_nodes=["127.0.0.1:50050"])
        coord = FederationCoordinator(cfg, "c", "h", 1)
        coord.start()
        assert coord._gossip_thread is not None
        assert coord._gossip_thread.is_alive()
        coord.stop()
        assert coord._gossip_thread is None

    def test_close(self) -> None:
        """close() is the async variant — calls stop() then aclose()."""
        import asyncio
        cfg = FederationConfig(enabled=True)
        coord = FederationCoordinator(cfg, "c", "h", 1)
        coord.start()
        asyncio.run(coord.close())
        assert not coord._running.is_set()

    def test_double_stop_safe(self) -> None:
        cfg = FederationConfig()
        coord = FederationCoordinator(cfg, "c", "h", 1)
        coord.stop()
        coord.stop()  # second call should not raise


class TestFederationCoordinatorForwardErrors:
    """forward_request, forward_request_streaming,
    forward_with_cache_affinity — error paths that don't need network."""

    def test_forward_request_no_peers(self) -> None:
        """Circuit breaker OPEN for peer ID — RuntimeError raised before HTTP."""
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord._circuit_breaker.force_open("unknown")
        import asyncio
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            asyncio.run(coord.forward_request(
                {"cluster_id": "unknown", "host": "x", "port": 1},
                {"model": "test"},
            ))

    def test_forward_request_streaming_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        coord._circuit_breaker.force_open("unknown")
        import asyncio
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            asyncio.run(coord.forward_request_streaming(
                {"cluster_id": "unknown", "host": "x", "port": 1},
                {"model": "test"},
            ).__anext__())

    def test_forward_with_cache_affinity_no_peers(self) -> None:
        """When no peers are registered, get_best_peer returns None."""
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        import asyncio
        with pytest.raises(RuntimeError, match="No suitable peer"):
            asyncio.run(coord.forward_with_cache_affinity(
                {"model": "test"},
                prompt_token_ids=[1, 2, 3],
            ))


class TestFederationCoordinatorHealthCheckErrors:
    """check_peer_health — error paths that don't need network."""

    def test_check_peer_health_unknown(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        import asyncio
        with pytest.raises(ValueError, match="Unknown peer"):
            asyncio.run(coord.check_peer_health("nonexistent"))

    def test_get_all_peers_health_no_peers(self) -> None:
        coord = FederationCoordinator(FederationConfig(), "c", "h", 1)
        import asyncio
        result = asyncio.run(coord.get_all_peers_health())
        assert result == []


# ── PeerBreakerEntry edge cases ──────────────────────────────────────────────


class TestPeerBreakerEntryEdge:
    def test_negative_open_until(self) -> None:
        """A negative open_until is not set (guard is open_until > 0), so not timed out."""
        entry = PeerBreakerEntry(open_until=-1e6)
        # is_timed_out checks self.open_until > 0 first; negative values fail the guard.
        assert entry.is_timed_out(time.time()) is False

    def test_large_consecutive_successes(self) -> None:
        entry = PeerBreakerEntry(consecutive_successes=10**6)
        assert entry.consecutive_successes == 10**6
        entry.consecutive_successes += 1
        assert entry.consecutive_successes == 10**6 + 1

    def test_default_is_not_timed_out(self) -> None:
        entry = PeerBreakerEntry()
        assert entry.is_timed_out(time.time()) is False
