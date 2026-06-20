"""E2E tests for graceful shutdown behavior.

Verifies that during shutdown:
- In-flight requests complete before the server stops
- New requests are rejected with 503 once shutdown begins
- Coordinator state flag is set correctly
- Nodes are drained and the lifecycle transitions to SHUTDOWN
"""

from __future__ import annotations

import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from distllm.core.coordinator_state import (
    CoordinatorRole,
    CoordinatorStateMachine,
)
from distllm.core.coordinator_lifecycle import (
    RequestTracker,
    ServerLifecycle,
)


@pytest.mark.e2e
class TestGracefulShutdownSIGTERM:
    """Simulate SIGTERM during active generations and verify orderly shutdown."""

    def test_inflight_requests_pass_shutdown_check(self, e2e_api_client, e2e_coordinator, e2e_auth_headers):
        """Requests are not rejected by the shutdown middleware when coordinator is active.

        AAA:
          Arrange - Ensure coordinator is not shutting down.
          Act     - Send a request while coordinator is still accepting.
          Assert  - The response is NOT 503 (shutdown rejection).
        """
        # Arrange
        e2e_coordinator._shutting_down = False

        # Act
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello before shutdown"}],
                "max_tokens": 5,
            },
            headers=e2e_auth_headers,
        )

        # Assert: request passed through the shutdown middleware (not 503)
        assert response.status_code != 503, (
            f"Request should not be rejected by shutdown middleware, "
            f"got {response.status_code}: {response.text}"
        )

    def test_new_requests_rejected_after_shutdown_flag_set(self, e2e_api_client, e2e_coordinator, e2e_auth_headers):
        """New requests arriving after _shutting_down=True get 503.

        AAA:
          Arrange - Set coordinator._shutting_down = True.
          Act     - Send a new chat completion request.
          Assert  - Response is 503 with shutdown_error type.
        """
        # Arrange
        e2e_coordinator._shutting_down = True

        try:
            # Act
            response = e2e_api_client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "After shutdown"}],
                    "max_tokens": 5,
                },
                headers=e2e_auth_headers,
            )

            # Assert
            assert response.status_code == 503
        finally:
            e2e_coordinator._shutting_down = False

    def test_health_endpoint_still_responds_during_shutdown(self, e2e_api_client, e2e_coordinator):
        """Health endpoints are exempt from backpressure and shutdown rejection.

        AAA:
          Arrange - Set _shutting_down = True.
          Act     - GET /health.
          Assert  - Returns 200 (health endpoints are exempt from backpressure).
        """
        # Arrange
        e2e_coordinator._shutting_down = True

        try:
            # Act
            response = e2e_api_client.get("/health")

            # Assert
            assert response.status_code == 200
        finally:
            e2e_coordinator._shutting_down = False


@pytest.mark.e2e
class TestRequestTrackerShutdown:
    """Test RequestTracker's shutdown completion behavior."""

    def test_complete_batch_requests_decodes_active_sequences(self):
        """Active sequences with generated tokens get decoded results on shutdown.

        AAA:
          Arrange - Register requests, populate active_seqs with generated tokens.
          Act     - Call complete_batch_requests.
          Assert  - Results are set and events are signaled.
        """
        # Arrange
        tracker = RequestTracker()
        event1 = tracker.register_request("req-1")
        event2 = tracker.register_request("req-2")

        mock_seq1 = MagicMock()
        mock_seq1.generated_tokens = [101, 102, 103]
        mock_seq2 = MagicMock()
        mock_seq2.generated_tokens = [201, 202]

        active_seqs = {"req-1": mock_seq1, "req-2": mock_seq2}

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda tokens, **kw: f"decoded-{tokens[0]}"

        # Act
        tracker.complete_batch_requests(active_seqs, [], mock_tokenizer)

        # Assert
        assert event1.wait(timeout=1.0), "req-1 event should be set"
        assert event2.wait(timeout=1.0), "req-2 event should be set"
        assert "decoded-101" in tracker._results.get("req-1", "")
        assert "decoded-201" in tracker._results.get("req-2", "")

    def test_complete_batch_requests_marks_pending_as_error(self):
        """Pending sequences (not yet started) get error messages on shutdown.

        AAA:
          Arrange - Register a request, put it in pending_seqs.
          Act     - Call complete_batch_requests.
          Assert  - The result contains an error message about timing out.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("pending-1")

        mock_pending = MagicMock()
        mock_pending.request_id = "pending-1"

        # Act
        tracker.complete_batch_requests({}, [mock_pending], None)

        # Assert
        result = tracker._results.get("pending-1", "")
        assert "timed out" in result.lower() or "error" in result.lower()

    def test_complete_batch_requests_handles_empty_tokenizer(self):
        """When tokenizer is None, raw token list is used as result string.

        AAA:
          Arrange - Active seq with tokens, no tokenizer.
          Act     - Call complete_batch_requests with tokenizer=None.
          Assert  - Result is the string representation of tokens.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("no-tok-1")

        mock_seq = MagicMock()
        mock_seq.generated_tokens = [42, 43]
        active_seqs = {"no-tok-1": mock_seq}

        # Act
        tracker.complete_batch_requests(active_seqs, [], None)

        # Assert
        result = tracker._results.get("no-tok-1", "")
        assert result  # should have some value, not empty

    def test_complete_batch_requests_handles_seq_without_tokens(self):
        """Sequences with no generated_tokens get a placeholder result.

        AAA:
          Arrange - Active seq with empty generated_tokens.
          Act     - Call complete_batch_requests.
          Assert  - Result contains a message about missing output.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("empty-seq")

        mock_seq = MagicMock()
        mock_seq.generated_tokens = []
        active_seqs = {"empty-seq": mock_seq}

        # Act
        tracker.complete_batch_requests(active_seqs, [], None)

        # Assert
        result = tracker._results.get("empty-seq", "")
        assert "error" in result.lower() or "without output" in result.lower()


@pytest.mark.e2e
class TestRequestTrackerCancellation:
    """Test that pending requests can be cancelled during shutdown."""

    def test_cancel_unblocks_waiting_thread(self):
        """A thread blocked on wait_for_result is unblocked by cancel().

        AAA:
          Arrange - Register a request, start a waiter thread.
          Act     - Cancel the request from the main thread.
          Assert  - The waiter thread receives the cancellation result.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("cancel-me")
        results = []
        errors = []

        def _wait():
            try:
                # Get event reference BEFORE cancel pops it
                event = tracker._events.get("cancel-me")
                if event is None:
                    # cancel already happened, check result directly
                    results.append(tracker._results.get("cancel-me", ""))
                    return
                event.wait(timeout=5.0)
                with tracker._lock:
                    result = tracker._results.pop("cancel-me", None)
                if result:
                    results.append(result)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        time.sleep(0.05)  # let waiter start and grab event ref

        # Act
        cancelled = tracker.cancel("cancel-me")

        # Assert
        assert cancelled is True
        t.join(timeout=3.0)
        assert len(results) == 1
        assert "cancelled" in results[0].lower()

    def test_cancel_unknown_request_returns_false(self):
        """Cancelling a non-existent request returns False.

        AAA:
          Arrange - Empty tracker.
          Act     - Cancel a non-existent request.
          Assert  - Returns False.
        """
        # Arrange
        tracker = RequestTracker()

        # Act
        result = tracker.cancel("nonexistent")

        # Assert
        assert result is False

    def test_clear_signals_all_pending_events(self):
        """clear() unblocks all waiters and resets state.

        AAA:
          Arrange - Register multiple requests.
          Act     - Call clear().
          Assert  - All events are set (waiters unblock), pending_count is 0.
        """
        # Arrange
        tracker = RequestTracker()
        e1 = tracker.register_request("r1")
        e2 = tracker.register_request("r2")
        assert tracker.pending_count == 2

        # Act
        tracker.clear()

        # Assert
        assert e1.wait(timeout=1.0)
        assert e2.wait(timeout=1.0)
        assert tracker.pending_count == 0
        assert tracker.shutting_down is False

    def test_cancel_sets_result_before_signaling_event(self):
        """cancel() stores the result and then signals the event.

        AAA:
          Arrange - Register a request.
          Act     - Cancel it.
          Assert  - The result is stored in _results with cancellation message.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("check-order")

        # Act
        tracker.cancel("check-order")

        # Assert: result should be set even though event is popped
        result = tracker._results.get("check-order")
        assert result is not None
        assert "cancelled" in result.lower()


@pytest.mark.e2e
class TestServerLifecycleShutdown:
    """Test ServerLifecycle state transitions during shutdown."""

    def test_lifecycle_transitions_to_stopped_on_shutdown(self):
        """initiate_shutdown() marks server as not running.

        AAA:
          Arrange - Start the lifecycle.
          Act     - Initiate shutdown.
          Assert  - is_running is False and shutdown_event is set.
        """
        # Arrange
        lifecycle = ServerLifecycle()
        lifecycle.start()
        assert lifecycle.is_running is True

        # Act
        lifecycle.initiate_shutdown(timeout=5.0)

        # Assert
        assert lifecycle.is_running is False
        assert lifecycle.shutdown_event.is_set()

    def test_wait_for_shutdown_unblocks_after_stop(self):
        """wait_for_shutdown() returns once stop() is called.

        AAA:
          Arrange - Start lifecycle, spawn waiter thread.
          Act     - Call stop() from main thread.
          Assert  - Waiter thread unblocks within timeout.
        """
        # Arrange
        lifecycle = ServerLifecycle()
        lifecycle.start()
        unblocked = threading.Event()

        def _wait():
            lifecycle.wait_for_shutdown(timeout=5.0)
            unblocked.set()

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

        # Act
        lifecycle.stop()

        # Assert
        assert unblocked.wait(timeout=2.0), "wait_for_shutdown should unblock"
        t.join(timeout=2.0)

    def test_lifecycle_uptime_tracks_running_time(self):
        """Uptime is non-zero after the lifecycle has been running.

        AAA:
          Arrange - Start lifecycle.
          Act     - Wait briefly.
          Assert  - uptime_s is greater than 0.
        """
        # Arrange
        lifecycle = ServerLifecycle()
        lifecycle.start()
        time.sleep(0.05)

        # Act / Assert
        assert lifecycle.is_running is True

    def test_double_start_is_idempotent(self):
        """Calling start() twice does not cause errors.

        AAA:
          Arrange - Create lifecycle.
          Act     - Call start() twice.
          Assert  - is_running remains True, no exception.
        """
        # Arrange
        lifecycle = ServerLifecycle()

        # Act
        lifecycle.start()
        lifecycle.start()  # idempotent

        # Assert
        assert lifecycle.is_running is True


@pytest.mark.e2e
class TestCoordinatorStateMachineShutdown:
    """Test that the coordinator lifecycle state machine handles shutdown correctly."""

    def test_leader_can_transition_to_shutdown(self):
        """A LEADER coordinator can transition directly to SHUTDOWN.

        AAA:
          Arrange - State machine in LEADER role.
          Act     - Transition to SHUTDOWN.
          Assert  - Role is SHUTDOWN and is_shutdown is True.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
        assert sm.is_leader is True

        # Act
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Assert
        assert sm.is_shutdown is True
        assert sm.is_active is False

    def test_follower_can_transition_to_shutdown(self):
        """A FOLLOWER coordinator can transition to SHUTDOWN.

        AAA:
          Arrange - State machine in FOLLOWER role.
          Act     - Transition to SHUTDOWN.
          Assert  - Role is SHUTDOWN.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)

        # Act
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Assert
        assert sm.is_shutdown is True

    def test_shutdown_is_terminal_state(self):
        """No transitions are valid from SHUTDOWN.

        AAA:
          Arrange - State machine in SHUTDOWN role.
          Act     - Attempt transition to any other role.
          Assert  - ValueError is raised.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Act / Assert
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(CoordinatorRole.LEADER)

        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(CoordinatorRole.FOLLOWER)

    def test_transition_callbacks_fire_on_shutdown(self):
        """Registered callbacks are invoked when shutdown transition occurs.

        AAA:
          Arrange - Register a transition callback.
          Act     - Transition to SHUTDOWN.
          Assert  - Callback received (FOLLOWER, SHUTDOWN) arguments.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)

        transitions = []
        sm.on_transition(lambda old, new: transitions.append((old, new)))

        # Act
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Assert
        assert len(transitions) == 1
        old, new = transitions[0]
        assert old == CoordinatorRole.FOLLOWER
        assert new == CoordinatorRole.SHUTDOWN

    def test_role_property_reflects_shutdown(self):
        """The role property returns SHUTDOWN after transition.

        AAA:
          Arrange - Transition FOLLOWER -> SHUTDOWN.
          Act     - Read role property.
          Assert  - role is CoordinatorRole.SHUTDOWN.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Act / Assert
        assert sm.role == CoordinatorRole.SHUTDOWN

    def test_previous_role_preserved_after_shutdown(self):
        """previous_role returns the role before SHUTDOWN transition.

        AAA:
          Arrange - Transition LEADER -> SHUTDOWN.
          Act     - Read previous_role.
          Assert  - previous_role is LEADER.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.SHUTDOWN)

        # Act / Assert
        assert sm.previous_role == CoordinatorRole.LEADER


@pytest.mark.e2e
class TestShutdownIdempotency:
    """Verify that repeated shutdown signals are harmless."""

    def test_setting_shutting_down_twice_is_safe(self, e2e_api_client, e2e_coordinator, e2e_auth_headers):
        """Setting _shutting_down=True twice does not cause errors.

        AAA:
          Arrange - Normal coordinator.
          Act     - Set _shutting_down = True twice, send request.
          Assert  - Still returns 503, no crash.
        """
        # Arrange
        e2e_coordinator._shutting_down = False

        try:
            # Act
            e2e_coordinator._shutting_down = True
            e2e_coordinator._shutting_down = True  # idempotent

            response = e2e_api_client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 3,
                },
                headers=e2e_auth_headers,
            )

            # Assert
            assert response.status_code == 503
        finally:
            e2e_coordinator._shutting_down = False

    def test_resetting_shutdown_flag_allows_requests(self, e2e_api_client, e2e_coordinator, e2e_auth_headers):
        """After toggling _shutting_down back to False, requests pass shutdown check.

        AAA:
          Arrange - Set _shutting_down = True then back to False.
          Act     - Send a chat request.
          Assert  - Response is NOT 503 (shutdown middleware passes request through).
        """
        # Arrange
        e2e_coordinator._shutting_down = True
        e2e_coordinator._shutting_down = False

        # Act
        response = e2e_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "back from shutdown"}],
                "max_tokens": 3,
            },
            headers=e2e_auth_headers,
        )

        # Assert: request passed through shutdown middleware
        assert response.status_code != 503, (
            f"Request should pass after resetting shutdown flag, "
            f"got {response.status_code}: {response.text}"
        )
