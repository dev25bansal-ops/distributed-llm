"""F-015 regression: HA leader election must gate request serving.

A standby (follower) coordinator must NOT serve inference requests — only the
elected leader may.  This suite proves that both request-serving entry points
(:meth:`Coordinator.generate` and :meth:`Coordinator.generate_async`) reject a
standby while letting the leader serve, and that single-node (non-HA) serving
is unaffected.

Mocks only — no real Ray / election thread / GPU.  The ``RayFaultTolerance``
election is stubbed out so no heartbeat loop runs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from distllm.errors import NotLeaderError


def _make_ha_coordinator(is_leader: bool, leader_id: str = "coord-1"):
    """Build a Coordinator with HA enabled and a stubbed election.

    The ``RayFaultTolerance`` election is replaced by a mock whose
    ``is_leader()`` returns *is_leader*, so the coordinator's gate sees the
    desired leadership state without starting a real election thread.

    NOTE: Workaround for Coordinator.__init__ referencing _subsystem_mgr
    before it is assigned (same as tests/dist/test_coordinator_failover.py).
    """
    from distllm.core.coordinator import Coordinator
    from distllm.core.coordinator_config import CoordinatorConfig

    mock_election = MagicMock()
    mock_election.is_leader.return_value = is_leader
    mock_election.get_leader.return_value = leader_id

    had_attr = hasattr(Coordinator, "_subsystem_mgr")
    old = getattr(Coordinator, "_subsystem_mgr", None)
    Coordinator._subsystem_mgr = MagicMock()
    try:
        with patch(
            "distllm.core.ha_coordinator.RayFaultTolerance",
            return_value=mock_election,
        ):
            coord = Coordinator(
                config=CoordinatorConfig(
                    model_name="test-model",
                    port=50050,
                    ha_enabled=True,
                )
            )
            coord._election._ha_election = mock_election
        return coord
    finally:
        if had_attr:
            Coordinator._subsystem_mgr = old
        else:
            del Coordinator._subsystem_mgr


def _make_plain_coordinator():
    """Build a Coordinator with HA disabled (single-node operation)."""
    from distllm.core.coordinator import Coordinator
    from distllm.core.coordinator_config import CoordinatorConfig

    had_attr = hasattr(Coordinator, "_subsystem_mgr")
    old = getattr(Coordinator, "_subsystem_mgr", None)
    Coordinator._subsystem_mgr = MagicMock()
    try:
        coord = Coordinator(
            config=CoordinatorConfig(model_name="test-model", port=50050)
        )
        return coord
    finally:
        if had_attr:
            Coordinator._subsystem_mgr = old
        else:
            del Coordinator._subsystem_mgr


class TestStandbyRejectsRequests:
    """A standby coordinator must refuse to serve inference."""

    def test_generate_raises_not_leader_on_standby(self) -> None:
        coord = _make_ha_coordinator(is_leader=False)
        assert coord.is_leader is False
        with pytest.raises(NotLeaderError) as exc_info:
            coord.generate("hello", max_new_tokens=8)
        assert exc_info.value.code == "NOT_LEADER"
        # The standby must never reach the request handler.
        coord._request_handler.generate = MagicMock(
            side_effect=AssertionError("standby must not serve")
        )
        with pytest.raises(NotLeaderError):
            coord.generate("hello", max_new_tokens=8)
        coord._request_handler.generate.assert_not_called()

    def test_generate_async_raises_not_leader_on_standby(self) -> None:
        # The async/batch path bypasses generate() and must be gated too.
        coord = _make_ha_coordinator(is_leader=False)
        coord._request_handler.generate_async = MagicMock(
            side_effect=AssertionError("standby must not serve async")
        )
        with pytest.raises(NotLeaderError):
            coord.generate_async("hello", max_new_tokens=8)
        coord._request_handler.generate_async.assert_not_called()


class TestLeaderServesRequests:
    """The elected leader must keep serving inference."""

    def test_generate_serves_on_leader(self) -> None:
        coord = _make_ha_coordinator(is_leader=True)
        assert coord.is_leader is True
        coord._request_handler.generate = MagicMock(return_value="served")
        result = coord.generate("hello", max_new_tokens=8)
        assert result == "served"
        coord._request_handler.generate.assert_called_once()

    def test_generate_async_serves_on_leader(self) -> None:
        coord = _make_ha_coordinator(is_leader=True)
        coord._request_handler.generate_async = MagicMock(
            return_value="req-123"
        )
        result = coord.generate_async("hello", max_new_tokens=8)
        assert result == "req-123"
        coord._request_handler.generate_async.assert_called_once()


class TestNonHAServingUnaffected:
    """Single-node (HA disabled) operation must be unaffected."""

    def test_generate_serves_without_ha(self) -> None:
        coord = _make_plain_coordinator()
        assert coord.ha_status == {"enabled": False}
        assert coord.is_leader is True
        coord._request_handler.generate = MagicMock(return_value="ok")
        assert coord.generate("hello", max_new_tokens=8) == "ok"
        coord._request_handler.generate.assert_called_once()

    def test_generate_async_serves_without_ha(self) -> None:
        coord = _make_plain_coordinator()
        coord._request_handler.generate_async = MagicMock(
            return_value="req-456"
        )
        assert coord.generate_async("hello", max_new_tokens=8) == "req-456"
        coord._request_handler.generate_async.assert_called_once()
