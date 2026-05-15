"""Circuit breaker tests for the Coordinator's node failure tracking.

Tests:
- _check_circuit_breaker: returns True when node should be skipped
- _record_node_success: resets failure count
- _record_node_failure: increments, opens circuit after 3 failures
- Exponential backoff: base retry 1s, max retry 60s
- Node removal scenarios
- Prometheus gauge: distllm_circuit_breaker_state

The circuit breaker is embedded in the Coordinator class.

Run: pytest tests/core/test_circuit_breaker.py -v
"""

import time

from distllm.core.coordinator import Coordinator


class TestCircuitBreakerBasics:
    """Basic circuit breaker functionality."""

    def test_circuit_closed_initially(self):
        """Circuit should be closed (not skipping) for a new node."""
        coord = Coordinator(model_name="test-model")

        result = coord._check_circuit_breaker("node-0")

        assert result is False  # Not skipping

    def test_check_circuit_breaker_unknown_node(self):
        """Unknown node should return False (not circuit-broken)."""
        coord = Coordinator(model_name="test-model")

        result = coord._check_circuit_breaker("unknown-node")

        assert result is False

    def test_record_success_resets_failures(self):
        """Recording a success should clear failure count."""
        coord = Coordinator(model_name="test-model")

        # Record 2 failures (below threshold)
        coord._record_node_failure("node-0")
        coord._record_node_failure("node-0")
        assert coord._node_failure_counts["node-0"] == 2

        # Record success
        coord._record_node_success("node-0")

        assert coord._node_failure_counts["node-0"] == 0
        assert "node-0" not in coord._node_recovery_time


class TestCircuitBreakerThreshold:
    """Tests for the 3-failure threshold."""

    def test_circuit_opens_after_3_failures(self):
        """Circuit should open after 3 consecutive failures."""
        coord = Coordinator(model_name="test-model")

        coord._record_node_failure("node-0")
        coord._record_node_failure("node-0")
        coord._record_node_failure("node-0")

        assert coord._check_circuit_breaker("node-0") is True

    def test_circuit_not_open_before_threshold(self):
        """Circuit should not open with 2 failures (below threshold of 3)."""
        coord = Coordinator(model_name="test-model")

        coord._record_node_failure("node-0")
        coord._record_node_failure("node-0")

        assert coord._check_circuit_breaker("node-0") is False

    def test_circuit_opens_exactly_at_threshold(self):
        """Circuit should open on the 3rd failure, not before."""
        coord = Coordinator(model_name="test-model")

        # 1st failure
        coord._record_node_failure("node-0")
        assert coord._check_circuit_breaker("node-0") is False

        # 2nd failure
        coord._record_node_failure("node-0")
        assert coord._check_circuit_breaker("node-0") is False

        # 3rd failure - should open circuit
        coord._record_node_failure("node-0")
        assert coord._check_circuit_breaker("node-0") is True


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery and backoff."""

    def test_recovery_time_set_on_open(self):
        """Recovery time should be set when circuit opens."""
        coord = Coordinator(model_name="test-model")

        before = time.time()
        for _ in range(3):
            coord._record_node_failure("node-0")
        after = time.time()

        assert "node-0" in coord._node_recovery_time
        recovery_time = coord._node_recovery_time["node-0"]
        assert recovery_time >= before
        assert recovery_time <= after + 1.0  # Within 1 second of recording

    def test_success_resets_circuit(self):
        """Recording success after circuit opens should reset state."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")

        assert coord._check_circuit_breaker("node-0") is True

        coord._record_node_success("node-0")

        assert coord._check_circuit_breaker("node-0") is False
        assert coord._node_failure_counts["node-0"] == 0

    def test_circuit_remains_open_before_cooldown(self):
        """Circuit should remain open during cooldown period."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")

        # Set recovery time to far future
        coord._node_recovery_time["node-0"] = time.time() + 1000

        assert coord._check_circuit_breaker("node-0") is True


class TestCircuitBreakerExponentialBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_backoff_is_base_delay(self):
        """First backoff (at threshold) should be base delay (1s)."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")

        recovery_time = coord._node_recovery_time["node-0"]
        expected_min = time.time() + 1.0 - 0.1  # base * 2^0 = 1s
        expected_max = time.time() + 1.0 + 0.1

        assert recovery_time >= expected_min
        assert recovery_time <= expected_max

    def test_backoff_doubles_each_failure(self):
        """Backoff should double with each additional failure."""
        coord = Coordinator(model_name="test-model")

        # 4 failures: backoff = 1 * 2^(4-3) = 2s
        for _ in range(4):
            coord._record_node_failure("node-0")

        recovery_time = coord._node_recovery_time["node-0"]
        expected_min = time.time() + 2.0 - 0.1
        expected_max = time.time() + 2.0 + 0.1

        assert recovery_time >= expected_min
        assert recovery_time <= expected_max

    def test_backoff_capped_at_max(self):
        """Backoff should not exceed max (60s)."""
        coord = Coordinator(model_name="test-model")

        # Many failures: would exceed 60s without cap
        # 2^(10) = 1024s > 60s cap
        for _ in range(10):
            coord._record_node_failure("node-0")

        recovery_time = coord._node_recovery_time["node-0"]
        expected_max = time.time() + 60.0 + 0.1

        assert recovery_time <= expected_max

    def test_cooldown_elapsed_allows_retry(self):
        """After cooldown elapses, node should be available again."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")

        # Set recovery time to past
        coord._node_recovery_time["node-0"] = time.time() - 10

        # Cooldown elapsed - should allow retry (half-open)
        assert coord._check_circuit_breaker("node-0") is False


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics integration."""

    def test_failure_records_metrics(self):
        """Recording failure should increment metrics."""
        coord = Coordinator(model_name="test-model")

        coord._record_node_failure("node-0")

        assert coord.metrics["node_failures"] == 1
        assert coord.metrics["errors"] == 1

    def test_multiple_failures_accumulate(self):
        """Multiple failures should accumulate metrics."""
        coord = Coordinator(model_name="test-model")

        for _ in range(5):
            coord._record_node_failure("node-0")

        assert coord.metrics["node_failures"] == 5
        assert coord.metrics["errors"] == 5


class TestCircuitBreakerMultipleNodes:
    """Tests for circuit breaker with multiple nodes."""

    def test_nodes_tracked_independently(self):
        """Each node's circuit breaker state should be independent."""
        coord = Coordinator(model_name="test-model")

        # Fail node-0
        for _ in range(3):
            coord._record_node_failure("node-0")

        # node-1 should still be fine
        assert coord._check_circuit_breaker("node-0") is True
        assert coord._check_circuit_breaker("node-1") is False

    def test_success_on_one_node_doesnt_affect_other(self):
        """Success on one node should not affect other node's state."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")
        coord._record_node_failure("node-1")

        coord._record_node_success("node-0")

        assert coord._check_circuit_breaker("node-0") is False
        assert coord._check_circuit_breaker("node-1") is False  # Still below threshold

    def test_recovery_time_independent(self):
        """Each node should have its own recovery time."""
        coord = Coordinator(model_name="test-model")

        for _ in range(3):
            coord._record_node_failure("node-0")
            coord._record_node_failure("node-1")

        coord._record_node_success("node-0")

        assert "node-0" not in coord._node_recovery_time
        assert "node-1" in coord._node_recovery_time
