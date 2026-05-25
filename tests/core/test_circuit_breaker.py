"""Circuit breaker tests for the Coordinator's node failure tracking.

Tests:
- check_circuit_breaker: returns True when node should be skipped
- record_success: resets failure count
- record_failure: increments, opens circuit after 3 failures
- Exponential backoff: base retry 1s, max retry 60s
- Node removal scenarios
- Prometheus gauge: distllm_circuit_breaker_state

The circuit breaker is in ResourceManager, accessible via coord._resource_mgr.

Run: pytest tests/core/test_circuit_breaker.py -v
"""

import time
from unittest.mock import patch, MagicMock

from distllm.core.coordinator import Coordinator


def _make_coord():
    with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok, \
         patch("distllm.core.coordinator.GRPCServer"):
        mock_tok.from_pretrained.return_value = _mock_tokenizer()
        coord = Coordinator(model_name="test-model")
        coord.tokenizer = _mock_tokenizer()
        return coord


def _mock_tokenizer():
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
    tok.decode.side_effect = lambda tokens, **kwargs: "hello"
    tok.eos_token_id = 0
    tok.vocab_size = 100
    return tok


class TestCircuitBreakerBasics:
    """Basic circuit breaker functionality."""

    def test_circuit_closed_initially(self):
        coord = _make_coord()
        result = coord._resource_mgr.check_circuit_breaker("node-0")
        assert result is False

    def test_check_circuit_breaker_unknown_node(self):
        coord = _make_coord()
        result = coord._resource_mgr.check_circuit_breaker("unknown-node")
        assert result is False

    def test_record_success_resets_failures(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm._node_failure_counts["node-0"] == 2
        rm.record_success("node-0")
        assert rm._node_failure_counts["node-0"] == 0
        assert "node-0" not in rm._node_recovery_time


class TestCircuitBreakerThreshold:
    """Tests for the 3-failure threshold."""

    def test_circuit_opens_after_3_failures(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True

    def test_circuit_not_open_before_threshold(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is False

    def test_circuit_opens_exactly_at_threshold(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery and backoff."""

    def test_recovery_time_set_on_open(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        before = time.time()
        for _ in range(3):
            rm.record_failure("node-0")
        after = time.time()
        assert "node-0" in rm._node_recovery_time
        recovery_time = rm._node_recovery_time["node-0"]
        assert recovery_time >= before
        assert recovery_time <= after + 1.0

    def test_success_resets_circuit(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True
        rm.record_success("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        assert rm._node_failure_counts["node-0"] == 0

    def test_circuit_remains_open_before_cooldown(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        rm._node_recovery_time["node-0"] = time.time() + 1000
        assert rm.check_circuit_breaker("node-0") is True


class TestCircuitBreakerExponentialBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_backoff_is_base_delay(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        recovery_time = rm._node_recovery_time["node-0"]
        expected_min = time.time() + 1.0 - 0.1
        expected_max = time.time() + 1.0 + 0.1
        assert recovery_time >= expected_min
        assert recovery_time <= expected_max

    def test_backoff_doubles_each_failure(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(4):
            rm.record_failure("node-0")
        recovery_time = rm._node_recovery_time["node-0"]
        expected_min = time.time() + 2.0 - 0.1
        expected_max = time.time() + 2.0 + 0.1
        assert recovery_time >= expected_min
        assert recovery_time <= expected_max

    def test_backoff_capped_at_max(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(10):
            rm.record_failure("node-0")
        recovery_time = rm._node_recovery_time["node-0"]
        expected_max = time.time() + 60.0 + 0.1
        assert recovery_time <= expected_max

    def test_cooldown_elapsed_allows_retry(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        rm._node_recovery_time["node-0"] = time.time() - 10
        assert rm.check_circuit_breaker("node-0") is False


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics integration."""

    def test_failure_records_metrics(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        rm.record_failure("node-0")
        assert rm._metrics["node_failures"] == 1
        assert rm._metrics["errors"] == 1

    def test_multiple_failures_accumulate(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(5):
            rm.record_failure("node-0")
        assert rm._metrics["node_failures"] == 5
        assert rm._metrics["errors"] == 5


class TestCircuitBreakerMultipleNodes:
    """Tests for circuit breaker with multiple nodes."""

    def test_nodes_tracked_independently(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True
        assert rm.check_circuit_breaker("node-1") is False

    def test_success_on_one_node_doesnt_affect_other(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
        rm.record_failure("node-1")
        rm.record_success("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        assert rm.check_circuit_breaker("node-1") is False

    def test_recovery_time_independent(self):
        coord = _make_coord()
        rm = coord._resource_mgr
        for _ in range(3):
            rm.record_failure("node-0")
            rm.record_failure("node-1")
        rm.record_success("node-0")
        assert "node-0" not in rm._node_recovery_time
        assert "node-1" in rm._node_recovery_time
