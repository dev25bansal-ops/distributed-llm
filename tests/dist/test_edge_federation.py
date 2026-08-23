"""Tests for edge_federation.py -- Edge Inference Federation.

Tests the public API of EdgeFederationManager, EdgeNodeProfile, and
the EDGE_MODEL_SIZE_LIMIT constant.  Zero mocks -- uses only real objects.
No GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.edge_federation import (
    EDGE_MODEL_SIZE_LIMIT,
    EdgeFederationManager,
    EdgeNodeProfile,
)


# ============================================================================
# EdgeNodeProfile dataclass
# ============================================================================


class TestEdgeNodeProfile:
    """EdgeNodeProfile dataclass construction and defaults."""

    def test_default_values(self) -> None:
        """Fields not passed to the constructor receive sensible defaults."""
        profile = EdgeNodeProfile(node_id="test-node")
        assert profile.node_id == "test-node"
        assert profile.device_type == "browser"
        assert profile.model_name == ""
        assert profile.max_tokens_per_request == 256
        assert profile.max_batch_size == 1
        assert profile.avg_latency_ms == 0.0
        assert profile.is_online is True
        assert profile.transport == "webrtc"
        assert profile.session_id == ""
        assert profile.total_requests_served == 0
        assert profile.total_errors == 0
        assert isinstance(profile.last_seen, float)
        assert profile.last_seen > 0

    def test_all_fields_explicit(self) -> None:
        """Every dataclass field can be set through the constructor."""
        profile = EdgeNodeProfile(
            node_id="node-42",
            device_type="mobile",
            model_name="llama-3b",
            max_tokens_per_request=512,
            max_batch_size=4,
            avg_latency_ms=150.0,
            last_seen=1000.0,
            is_online=False,
            transport="websocket",
            session_id="sess-abc",
            total_requests_served=10,
            total_errors=2,
        )
        assert profile.node_id == "node-42"
        assert profile.device_type == "mobile"
        assert profile.model_name == "llama-3b"
        assert profile.max_tokens_per_request == 512
        assert profile.max_batch_size == 4
        assert profile.avg_latency_ms == 150.0
        assert profile.last_seen == 1000.0
        assert profile.is_online is False
        assert profile.transport == "websocket"
        assert profile.session_id == "sess-abc"
        assert profile.total_requests_served == 10
        assert profile.total_errors == 2

    def test_node_id_required(self) -> None:
        """node_id is the only mandatory field; omitting it should raise."""
        with pytest.raises(TypeError):
            EdgeNodeProfile()  # type: ignore[call-arg]

    def test_accepts_all_device_types(self) -> None:
        """Any device-type string is accepted (no enum validation)."""
        for dtype in ("mobile", "browser", "iot", "raspberry-pi", ""):
            p = EdgeNodeProfile(node_id=dtype, device_type=dtype)
            assert p.device_type == dtype

    def test_accepts_all_transports(self) -> None:
        """Any transport string is accepted (no enum validation)."""
        for t in ("webrtc", "websocket", "mqtt", "bluetooth", ""):
            p = EdgeNodeProfile(node_id=t, transport=t)
            assert p.transport == t

    def test_mutable(self) -> None:
        """EdgeNodeProfile is a regular dataclass (not frozen)."""
        p = EdgeNodeProfile(node_id="n")
        p.avg_latency_ms = 200.0
        p.is_online = False
        assert p.avg_latency_ms == 200.0
        assert p.is_online is False


# ============================================================================
# EDGE_MODEL_SIZE_LIMIT constant
# ============================================================================


class TestEdgeModelSizeLimit:
    """EDGE_MODEL_SIZE_LIMIT constant checks."""

    def test_value(self) -> None:
        assert EDGE_MODEL_SIZE_LIMIT == 3_000_000_000

    def test_is_int(self) -> None:
        assert isinstance(EDGE_MODEL_SIZE_LIMIT, int)


# ============================================================================
# EdgeFederationManager -- node registration
# ============================================================================


class TestEdgeFederationManagerNodeRegistration:
    """register_node, unregister_node, get_node."""

    def test_register_new_node_returns_profile(self) -> None:
        mgr = EdgeFederationManager()
        profile = mgr.register_node("device-1")
        assert isinstance(profile, EdgeNodeProfile)
        assert profile.node_id == "device-1"
        assert profile.is_online is True
        # Auto-generated session ID
        assert profile.session_id.startswith("edge-")
        assert len(profile.session_id) == 13  # "edge-" + 8 hex chars

    def test_register_with_explicit_params(self) -> None:
        mgr = EdgeFederationManager()
        profile = mgr.register_node(
            node_id="mobile-1",
            device_type="mobile",
            model_name="tiny-llama",
            transport="websocket",
            session_id="my-session",
        )
        assert profile.device_type == "mobile"
        assert profile.model_name == "tiny-llama"
        assert profile.transport == "websocket"
        assert profile.session_id == "my-session"

    def test_register_existing_node_updates_fields(self) -> None:
        """device_type is NOT updated on re-registration (code only sets it on creation)."""
        mgr = EdgeFederationManager()
        mgr.register_node("n1", device_type="browser")
        updated = mgr.register_node("n1", device_type="mobile", model_name="llama")
        assert updated.node_id == "n1"
        # device_type is set at creation time only; re-registration skips it
        assert updated.device_type == "browser"
        assert updated.model_name == "llama"
        assert updated.is_online is True

    def test_register_reuses_session_id_when_not_given(self) -> None:
        mgr = EdgeFederationManager()
        first = mgr.register_node("n1")
        second = mgr.register_node("n1")
        assert second.session_id == first.session_id

    def test_register_updates_session_id_when_given(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1", session_id="old")
        updated = mgr.register_node("n1", session_id="new")
        assert updated.session_id == "new"

    def test_unregister_existing_node_returns_true(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        assert mgr.unregister_node("n1") is True

    def test_unregister_missing_node_returns_false(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.unregister_node("ghost") is False

    def test_get_node_returns_profile(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        profile = mgr.get_node("n1")
        assert profile is not None
        assert profile.node_id == "n1"

    def test_get_node_missing_returns_none(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.get_node("ghost") is None

    def test_get_node_after_unregister_returns_none(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        mgr.unregister_node("n1")
        assert mgr.get_node("n1") is None


# ============================================================================
# EdgeFederationManager -- online-node filtering
# ============================================================================


class TestEdgeFederationManagerOnlineNodes:
    """get_online_nodes behaviour."""

    def test_empty_initially(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.get_online_nodes() == []

    def test_newly_registered_is_online(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        online = mgr.get_online_nodes()
        assert len(online) == 1
        assert online[0].node_id == "n1"

    def test_offline_node_excluded(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        mgr.get_node("n1").is_online = False
        assert mgr.get_online_nodes() == []

    def test_stale_node_excluded(self) -> None:
        mgr = EdgeFederationManager(node_timeout_seconds=10.0)
        mgr.register_node("n1")
        mgr.get_node("n1").last_seen = 0.0
        assert mgr.get_online_nodes() == []

    def test_multiple_online_nodes(self) -> None:
        mgr = EdgeFederationManager()
        for i in range(5):
            mgr.register_node(f"n{i}")
        assert len(mgr.get_online_nodes()) == 5

    def test_online_within_timeout_boundary(self) -> None:
        mgr = EdgeFederationManager(node_timeout_seconds=30.0)
        mgr.register_node("n1")
        profile = mgr.get_node("n1")
        profile.last_seen = time.time() - 29.999
        assert len(mgr.get_online_nodes()) == 1

    def test_offline_past_timeout_boundary(self) -> None:
        mgr = EdgeFederationManager(node_timeout_seconds=30.0)
        mgr.register_node("n1")
        profile = mgr.get_node("n1")
        profile.last_seen = time.time() - 30.001
        assert len(mgr.get_online_nodes()) == 0


# ============================================================================
# EdgeFederationManager -- should_route_to_edge
# ============================================================================


class TestEdgeFederationManagerShouldRouteToEdge:
    """should_route_to_edge routing decisions."""

    def test_one_billion_routes_to_edge(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.should_route_to_edge(1) is True

    def test_exact_limit_routes_to_edge(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.should_route_to_edge(3) is True

    def test_exceeds_limit_does_not_route(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.should_route_to_edge(4) is False

    def test_zero_params_routes(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.should_route_to_edge(0) is True

    def test_custom_limit(self) -> None:
        mgr = EdgeFederationManager(model_size_limit=1_000_000_000)
        assert mgr.should_route_to_edge(1) is True
        assert mgr.should_route_to_edge(2) is False


# ============================================================================
# EdgeFederationManager -- route_inference
# ============================================================================


class TestEdgeFederationManagerRouteInference:
    """route_inference with real (non-networked) fallback path."""

    def test_no_nodes_no_fallback_returns_none(self) -> None:
        mgr = EdgeFederationManager()
        assert mgr.route_inference("hello", "model") is None

    def test_no_nodes_calls_fallback(self) -> None:
        mgr = EdgeFederationManager()
        calls: list[str] = []

        def fb(p: str, m: str, t: int) -> str:
            calls.append(p)
            return f"fb-{p}"

        result = mgr.route_inference("hi", "m", fallback_fn=fb)
        assert calls == ["hi"]
        assert result == "fb-hi"

    def test_online_node_fails_then_falls_back(self) -> None:
        """The edge attempt will fail (no actual server), so fallback runs."""
        mgr = EdgeFederationManager()
        mgr.register_node("edge-1", transport="websocket")
        calls: list[str] = []

        def fb(p: str, m: str, t: int) -> str:
            calls.append(p)
            return "fallback-ok"

        result = mgr.route_inference("hello", "m", fallback_fn=fb)
        assert calls == ["hello"]
        assert result == "fallback-ok"

    def test_failed_attempt_marks_node_offline(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("edge-1", transport="websocket")
        mgr.route_inference("hi", "m", fallback_fn=lambda p, m, t: "fb")
        profile = mgr.get_node("edge-1")
        assert profile is not None
        assert profile.is_online is False
        assert profile.total_errors == 1

    def test_error_counter_accumulates(self) -> None:
        """Total errors for the node should be 1 because after the first failure
        the node is immediately marked offline and not tried again."""
        mgr = EdgeFederationManager()
        mgr.register_node("edge-1", transport="websocket")
        fb = lambda p, m, t: "fb"
        mgr.route_inference("a", "m", fallback_fn=fb)
        mgr.route_inference("b", "m", fallback_fn=fb)
        assert mgr.get_node("edge-1").total_errors == 1

    def test_picks_lowest_latency_node(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("slow", transport="websocket")
        mgr.register_node("fast", transport="websocket")
        mgr.get_node("slow").avg_latency_ms = 500.0
        mgr.get_node("fast").avg_latency_ms = 50.0
        fb = lambda p, m, t: "fb"
        mgr.route_inference("hi", "m", fallback_fn=fb)
        # "fast" was tried first (lower latency), failed, marked offline.
        # "slow" should still be online and untouched.
        assert mgr.get_node("slow").total_errors == 0
        assert mgr.get_node("slow").is_online is True
        assert mgr.get_node("fast").total_errors == 1
        assert mgr.get_node("fast").is_online is False

    def test_fallback_returns_none(self) -> None:
        """When fallback_fn itself returns None, route_inference returns None."""
        mgr = EdgeFederationManager()
        result = mgr.route_inference("hi", "m", fallback_fn=lambda p, m, t: None)
        assert result is None


# ============================================================================
# EdgeFederationManager -- stats property
# ============================================================================


class TestEdgeFederationManagerStats:
    """stats property keys and values."""

    def test_defaults(self) -> None:
        mgr = EdgeFederationManager()
        s = mgr.stats
        assert s["connected_nodes"] == 0
        assert s["online_nodes"] == 0
        assert s["offline_nodes"] == 0
        assert s["requests_routed"] == 0
        assert s["requests_failed"] == 0
        assert s["success_rate"] == 1.0
        assert s["avg_edge_latency_ms"] == 0.0
        assert s["model_size_limit_b"] == 3.0
        assert s["devices"] == {"mobile": 0, "browser": 0, "iot": 0}

    def test_after_registration(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("a", device_type="mobile")
        mgr.register_node("b", device_type="browser")
        mgr.register_node("c", device_type="mobile")
        mgr.register_node("d", device_type="iot")
        s = mgr.stats
        assert s["connected_nodes"] == 4
        assert s["online_nodes"] == 4
        assert s["offline_nodes"] == 0
        assert s["devices"]["mobile"] == 2
        assert s["devices"]["browser"] == 1
        assert s["devices"]["iot"] == 1

    def test_after_failed_routes(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1", transport="websocket")
        fb = lambda p, m, t: "fb"
        mgr.route_inference("x", "m", fallback_fn=fb)
        s = mgr.stats
        assert s["requests_failed"] == 1
        assert s["requests_routed"] == 0
        # success_rate = routed / (routed + failed) = 0 / 1 = 0.0
        assert s["success_rate"] == 0.0

    def test_after_unregister(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1")
        mgr.unregister_node("n1")
        assert mgr.stats["connected_nodes"] == 0


# ============================================================================
# EdgeFederationManager -- lifecycle
# ============================================================================


class TestEdgeFederationManagerLifecycle:
    """start / stop."""

    def test_start_sets_running_and_starts_thread(self) -> None:
        mgr = EdgeFederationManager()
        mgr.start()
        assert mgr._running is True
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        assert mgr._thread.daemon is True
        assert mgr._thread.name == "edge-health"
        mgr.stop()

    def test_stop_without_start_does_not_raise(self) -> None:
        mgr = EdgeFederationManager()
        mgr.stop()

    def test_stop_clears_running_flag(self) -> None:
        mgr = EdgeFederationManager()
        mgr.start()
        mgr.stop()
        assert mgr._running is False

    def test_restart_works(self) -> None:
        mgr = EdgeFederationManager()
        mgr.start()
        t1 = mgr._thread
        mgr.stop()
        mgr.start()
        t2 = mgr._thread
        # A new thread object is created on restart
        assert t2 is not t1
        assert t2.is_alive()
        assert t2.daemon is True
        mgr.stop()


# ============================================================================
# EdgeFederationManager -- edge cases
# ============================================================================


class TestEdgeFederationManagerEdgeCases:
    """Boundary and unusual inputs."""

    def test_empty_node_id(self) -> None:
        mgr = EdgeFederationManager()
        profile = mgr.register_node("")
        assert profile.node_id == ""

    def test_long_node_id(self) -> None:
        mgr = EdgeFederationManager()
        long_id = "x" * 10000
        profile = mgr.register_node(long_id)
        assert profile.node_id == long_id

    def test_special_chars_in_node_id(self) -> None:
        mgr = EdgeFederationManager()
        profile = mgr.register_node("node-1_2:3@4#5!")
        assert profile.node_id == "node-1_2:3@4#5!"

    def test_register_unregister_register_cycle(self) -> None:
        mgr = EdgeFederationManager()
        mgr.register_node("n1", device_type="mobile")
        mgr.unregister_node("n1")
        profile = mgr.register_node("n1", device_type="iot")
        assert profile.device_type == "iot"
        # Re-registering after removal gets a brand-new profile
        assert profile.session_id.startswith("edge-")

    def test_zero_timeout_with_old_node(self) -> None:
        mgr = EdgeFederationManager(node_timeout_seconds=0.0)
        mgr.register_node("n1")
        mgr.get_node("n1").last_seen = time.time() - 0.001
        assert mgr.get_online_nodes() == []

    def test_route_empty_prompt(self) -> None:
        mgr = EdgeFederationManager()
        fb = lambda p, m, t: f"echo-{p}"
        assert mgr.route_inference("", "m", fallback_fn=fb) == "echo-"

    def test_route_explicit_none_fallback(self) -> None:
        mgr = EdgeFederationManager()
        # Passing no fallback_fn at all (default None)
        assert mgr.route_inference("hello", "m") is None

    def test_negative_model_size_limit(self) -> None:
        """A negative limit rejects every model including zero-sized ones."""
        mgr = EdgeFederationManager(model_size_limit=-1)
        assert mgr.should_route_to_edge(0) is False
        assert mgr.should_route_to_edge(1) is False
