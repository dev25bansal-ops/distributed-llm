"""Unit tests for ReputationSystem.

Covers:
- Race condition in speed score
- Persistent failure penalty
- Credit spending insufficient balance
"""

import threading

import pytest

from distllm.dist.reputation import ReputationRecord, ReputationSystem


@pytest.fixture
def system():
    return ReputationSystem(min_reputation=0.3)


class TestSpeedScoreRaceCondition:
    """Test _compute_speed_score is thread-safe."""

    def test_concurrent_scoring_no_crash(self, system):
        # Seed some data
        for i in range(10):
            system.record_success(f"node_{i}", latency_ms=50.0 + i * 10, tokens=100)

        errors = []

        def score_all():
            try:
                for i in range(10):
                    system.get_score(f"node_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=score_all) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_record_and_score(self, system):
        errors = []

        def record_and_score(i):
            try:
                system.record_success(f"node_{i}", latency_ms=50.0, tokens=100)
                system.get_score(f"node_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_and_score, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestPersistentFailurePenalty:
    """Test persistent failure penalty logic."""

    def test_penalized_after_5_failures(self, system):
        for _ in range(6):
            system.record_failure("bad_node")
        score = system.get_score("bad_node")
        assert score == 0.1

    def test_not_penalized_below_5_failures(self, system):
        for _ in range(4):
            system.record_failure("ok_node")
        system.record_success("ok_node", latency_ms=50.0, tokens=100)
        score = system.get_score("ok_node")
        assert score > 0.1

    def test_mixed_requests_not_penalized(self, system):
        for _ in range(3):
            system.record_failure("mixed_node")
        for _ in range(10):
            system.record_success("mixed_node", latency_ms=50.0, tokens=100)
        score = system.get_score("mixed_node")
        assert score > 0.1


class TestCreditSpending:
    """Test credit spending logic."""

    def test_spend_credits_sufficient(self, system):
        system.record_contribution("node_1", tokens_computed=1000, credit_rate=1.0)
        assert system.get_credit_balance("node_1") == 1000.0
        assert system.spend_credits("node_1", 500) is True
        assert system.get_credit_balance("node_1") == 500.0

    def test_spend_credits_insufficient(self, system):
        system.record_contribution("node_1", tokens_computed=100, credit_rate=1.0)
        assert system.spend_credits("node_1", 200) is False
        assert system.get_credit_balance("node_1") == 100.0

    def test_credit_balance_unknown_node(self, system):
        assert system.get_credit_balance("unknown") == 0.0

    def test_credit_summary(self, system):
        system.record_contribution("node_1", tokens_computed=500, credit_rate=2.0)
        system.spend_credits("node_1", 200)
        summary = system.get_credit_summary()
        assert "node_1" in summary
        assert summary["node_1"]["tokens_contributed"] == 500
        assert summary["node_1"]["credits_earned"] == 1000.0
        assert summary["node_1"]["credits_spent"] == 200.0


class TestReputationScoring:
    """Test reputation score computation."""

    def test_unknown_node_gets_neutral_score(self, system):
        assert system.get_score("unknown") == 0.5

    def test_qualified_check(self, system):
        system._min_reputation = 0.3
        for _ in range(10):
            system.record_success("good_node", latency_ms=50.0, tokens=100)
        assert system.is_qualified("good_node") is True

    def test_unqualified_node(self, system):
        system._min_reputation = 0.9
        for _ in range(5):
            system.record_failure("bad_node")
        assert system.is_qualified("bad_node") is False

    def test_summary(self, system):
        system.record_success("node_1", latency_ms=50.0, tokens=100)
        system.record_health("node_1", healthy=True)
        summary = system.get_summary()
        assert "nodes" in summary
        assert "node_1" in summary["nodes"]
