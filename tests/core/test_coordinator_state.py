"""Tests for coordinator state machine -- CoordinatorStateMachine and CoordinatorRole.

Covers:
- CoordinatorRole enum values
- Valid and invalid state transitions
- Thread safety properties
- Callbacks and logging
- Force role override
- stats() output
- CoordinatorState legacy wrapper
"""

from __future__ import annotations

import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_state = load_module("distllm/core/coordinator_state.py")
CoordinatorRole = _state.CoordinatorRole
CoordinatorStateMachine = _state.CoordinatorStateMachine
CoordinatorState = _state.CoordinatorState


class TestCoordinatorRole:
    """CoordinatorRole enum values."""

    def test_values(self):
        assert CoordinatorRole.INIT.value == "init"
        assert CoordinatorRole.FOLLOWER.value == "follower"
        assert CoordinatorRole.CANDIDATE.value == "candidate"
        assert CoordinatorRole.LEADER.value == "leader"
        assert CoordinatorRole.RECOVERING.value == "recovering"
        assert CoordinatorRole.SHUTDOWN.value == "shutdown"

    def test_all_unique(self):
        values = [r.value for r in CoordinatorRole]
        assert len(values) == len(set(values))


class TestCoordinatorStateMachineInit:
    """Initial state of the state machine."""

    def test_initial_state_is_init(self):
        sm = CoordinatorStateMachine()
        assert sm.role == CoordinatorRole.INIT

    def test_initial_previous_role_is_none(self):
        sm = CoordinatorStateMachine()
        assert sm.previous_role is None

    def test_initial_is_leader_false(self):
        sm = CoordinatorStateMachine()
        assert sm.is_leader is False

    def test_initial_is_active_false(self):
        sm = CoordinatorStateMachine()
        assert sm.is_active is False

    def test_initial_is_shutdown_false(self):
        sm = CoordinatorStateMachine()
        assert sm.is_shutdown is False

    def test_initial_uptime_zero(self):
        sm = CoordinatorStateMachine()
        assert sm.uptime_s() == 0.0

    def test_initial_stats(self):
        sm = CoordinatorStateMachine()
        s = sm.stats()
        assert s["role"] == "init"
        assert s["previous_role"] is None
        assert s["transitions"] == 0


class TestCoordinatorStateMachineTransitions:
    """Valid and invalid state transitions."""

    def test_init_to_follower(self):
        sm = CoordinatorStateMachine()
        assert sm.transition_to(CoordinatorRole.FOLLOWER) is True
        assert sm.role == CoordinatorRole.FOLLOWER
        assert sm.previous_role == CoordinatorRole.INIT

    def test_init_to_leader(self):
        sm = CoordinatorStateMachine()
        assert sm.transition_to(CoordinatorRole.LEADER) is True
        assert sm.role == CoordinatorRole.LEADER

    def test_init_to_shutdown(self):
        sm = CoordinatorStateMachine()
        assert sm.transition_to(CoordinatorRole.SHUTDOWN) is True
        assert sm.role == CoordinatorRole.SHUTDOWN

    def test_follower_to_candidate(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        assert sm.transition_to(CoordinatorRole.CANDIDATE) is True

    def test_follower_to_leader(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        assert sm.transition_to(CoordinatorRole.LEADER) is True

    def test_candidate_to_leader(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.CANDIDATE)
        assert sm.transition_to(CoordinatorRole.LEADER) is True

    def test_candidate_to_follower(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.CANDIDATE)
        assert sm.transition_to(CoordinatorRole.FOLLOWER) is True

    def test_leader_to_follower(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.LEADER)
        assert sm.transition_to(CoordinatorRole.FOLLOWER) is True

    def test_leader_to_recovering(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.LEADER)
        assert sm.transition_to(CoordinatorRole.RECOVERING) is True

    def test_recovering_to_follower(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.RECOVERING)
        assert sm.transition_to(CoordinatorRole.FOLLOWER) is True

    def test_recovering_to_leader(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.RECOVERING)
        assert sm.transition_to(CoordinatorRole.LEADER) is True

    def test_shutdown_is_terminal(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.SHUTDOWN)
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(CoordinatorRole.LEADER)


class TestCoordinatorStateMachineInvalidTransitions:
    """Transitions that should raise ValueError."""

    def _assert_invalid(self, from_role: CoordinatorRole, to_role: CoordinatorRole):
        sm = CoordinatorStateMachine()
        if from_role != CoordinatorRole.INIT:
            # Walk through a valid path to reach from_role
            if from_role == CoordinatorRole.FOLLOWER:
                sm.transition_to(CoordinatorRole.FOLLOWER)
            elif from_role == CoordinatorRole.LEADER:
                sm.transition_to(CoordinatorRole.LEADER)
            elif from_role == CoordinatorRole.RECOVERING:
                sm.transition_to(CoordinatorRole.LEADER)
                sm.transition_to(CoordinatorRole.RECOVERING)
            elif from_role == CoordinatorRole.CANDIDATE:
                sm.transition_to(CoordinatorRole.FOLLOWER)
                sm.transition_to(CoordinatorRole.CANDIDATE)
            elif from_role == CoordinatorRole.SHUTDOWN:
                sm.transition_to(CoordinatorRole.SHUTDOWN)
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(to_role)

    def test_init_to_candidate(self):
        self._assert_invalid(CoordinatorRole.INIT, CoordinatorRole.CANDIDATE)

    def test_init_to_recovering(self):
        self._assert_invalid(CoordinatorRole.INIT, CoordinatorRole.RECOVERING)

    def test_follower_to_recovering(self):
        self._assert_invalid(CoordinatorRole.FOLLOWER, CoordinatorRole.RECOVERING)

    def test_candidate_to_recovering(self):
        self._assert_invalid(CoordinatorRole.CANDIDATE, CoordinatorRole.RECOVERING)

    def test_recovering_to_candidate(self):
        self._assert_invalid(CoordinatorRole.RECOVERING, CoordinatorRole.CANDIDATE)

    def test_shutdown_to_anything(self):
        self._assert_invalid(CoordinatorRole.SHUTDOWN, CoordinatorRole.LEADER)
        self._assert_invalid(CoordinatorRole.SHUTDOWN, CoordinatorRole.FOLLOWER)


class TestCoordinatorStateMachineProperties:
    """Property accessors after transitions."""

    def test_is_leader(self):
        sm = CoordinatorStateMachine()
        assert sm.is_leader is False
        sm.transition_to(CoordinatorRole.LEADER)
        assert sm.is_leader is True

    def test_is_active_after_follower(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        assert sm.is_active is True

    def test_is_active_after_shutdown(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.SHUTDOWN)
        assert sm.is_active is False

    def test_is_shutdown_true(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.SHUTDOWN)
        assert sm.is_shutdown is True

    def test_uptime_s_increases(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        u1 = sm.uptime_s()
        assert u1 > 0
        # uptime should increase (or at least not decrease)
        u2 = sm.uptime_s()
        assert u2 >= u1


class TestCoordinatorStateMachineCallbacks:
    """on_transition callback registration and invocation."""

    def test_callback_invoked(self):
        sm = CoordinatorStateMachine()
        calls = []
        sm.on_transition(lambda old, new: calls.append((old, new)))
        sm.transition_to(CoordinatorRole.FOLLOWER)
        assert len(calls) == 1
        assert calls[0][0] == CoordinatorRole.INIT
        assert calls[0][1] == CoordinatorRole.FOLLOWER

    def test_multiple_callbacks(self):
        sm = CoordinatorStateMachine()
        calls1 = []
        calls2 = []
        sm.on_transition(lambda old, new: calls1.append(1))
        sm.on_transition(lambda old, new: calls2.append(2))
        sm.transition_to(CoordinatorRole.LEADER)
        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_callback_error_does_not_block(self):
        sm = CoordinatorStateMachine()
        results = []

        def bad_cb(old, new):
            raise RuntimeError("cb error")

        def good_cb(old, new):
            results.append("ok")

        sm.on_transition(bad_cb)
        sm.on_transition(good_cb)
        sm.transition_to(CoordinatorRole.LEADER)
        assert results == ["ok"]


class TestCoordinatorStateMachineForceRole:
    """force_role sets state without validation."""

    def test_force_role_skips_validation(self):
        sm = CoordinatorStateMachine()
        # INIT -> RECOVERING is normally invalid
        sm.force_role(CoordinatorRole.RECOVERING)
        assert sm.role == CoordinatorRole.RECOVERING
        assert sm.previous_role == CoordinatorRole.INIT

    def test_force_role_records_transition(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.force_role(CoordinatorRole.LEADER)
        s = sm.stats()
        assert s["transitions"] == 2
        assert s["role"] == "leader"
        assert s["previous_role"] == "follower"


class TestCoordinatorStateMachineStats:
    """stats() output structure."""

    def test_stats_after_transitions(self):
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.CANDIDATE)
        sm.transition_to(CoordinatorRole.LEADER)
        s = sm.stats()
        assert s["role"] == "leader"
        assert s["previous_role"] == "candidate"
        assert s["transitions"] == 3
        assert "uptime_s" in s
        assert "time_in_role_s" in s


class TestCoordinatorStateLegacyWrapper:
    """Legacy CoordinatorState wrapper around state machine."""

    def test_initial_not_running(self):
        cs = CoordinatorState()
        assert cs.is_running is False

    def test_start_sets_running(self):
        cs = CoordinatorState()
        cs.start()
        assert cs.is_running is True

    def test_stop_clears_running(self):
        cs = CoordinatorState()
        cs.start()
        cs.stop()
        assert cs.is_running is False

    def test_uptime_s_after_start(self):
        cs = CoordinatorState()
        cs.start()
        assert cs.uptime_s() > 0

    def test_reset(self):
        cs = CoordinatorState()
        cs.start()
        cs.reset()
        assert cs.is_running is False
        assert cs.uptime_s() == 0.0
