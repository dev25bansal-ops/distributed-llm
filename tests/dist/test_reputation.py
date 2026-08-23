"""Tests for the GPU Reputation System (zero mocks)."""

from __future__ import annotations

import threading
import time

import pytest

from distllm.dist.reputation import ReputationRecord, ReputationSystem


class TestReputationRecord:
    """Unit tests for ReputationRecord dataclass."""

    def test_defaults(self) -> None:
        """All numeric fields start at zero / appropriate default."""
        rec = ReputationRecord(node_id="test_node")
        assert rec.node_id == "test_node"
        assert rec.total_requests == 0
        assert rec.successful_requests == 0
        assert rec.failed_requests == 0
        assert rec.total_tokens == 0
        assert rec.total_latency_ms == 0.0
        assert rec.health_check_passes == 0
        assert rec.health_check_fails == 0
        assert rec.tokens_contributed == 0
        assert rec.credits_earned == 0.0
        assert rec.credits_spent == 0.0
        # timestamps are wall-clock; just check they are positive
        assert rec.first_seen > 0
        assert rec.last_seen > 0
        assert rec.last_failure == 0.0

    def test_reliability_no_requests(self) -> None:
        """reliability is 0.0 when no requests have been made."""
        rec = ReputationRecord(node_id="n")
        assert rec.reliability == 0.0

    def test_reliability_all_success(self) -> None:
        """reliability is 1.0 when every request succeeded."""
        rec = ReputationRecord(node_id="n", total_requests=10, successful_requests=10)
        assert rec.reliability == 1.0

    def test_reliability_half(self) -> None:
        """reliability is 0.5 when half the requests succeeded."""
        rec = ReputationRecord(node_id="n", total_requests=10, successful_requests=5)
        assert rec.reliability == 0.5

    def test_reliability_zero_success(self) -> None:
        """reliability is 0.0 when no request succeeded."""
        rec = ReputationRecord(node_id="n", total_requests=10, successful_requests=0)
        assert rec.reliability == 0.0

    def test_avg_latency_no_successful(self) -> None:
        """avg_latency_ms is 0.0 when there are zero successful requests."""
        rec = ReputationRecord(node_id="n", total_requests=5, successful_requests=0)
        assert rec.avg_latency_ms == 0.0

    def test_avg_latency_some(self) -> None:
        """avg_latency_ms divides total latency by successful request count."""
        rec = ReputationRecord(
            node_id="n",
            successful_requests=4,
            total_latency_ms=200.0,
        )
        assert rec.avg_latency_ms == 50.0

    def test_avg_latency_zero_total(self) -> None:
        """avg_latency_ms is 0.0 when total_latency_ms is 0 and no successes."""
        rec = ReputationRecord(node_id="n")
        assert rec.avg_latency_ms == 0.0

    def test_health_ratio_no_checks(self) -> None:
        """health_ratio defaults to 1.0 when no health checks have run."""
        rec = ReputationRecord(node_id="n")
        assert rec.health_ratio == 1.0

    def test_health_ratio_all_pass(self) -> None:
        rec = ReputationRecord(node_id="n", health_check_passes=5, health_check_fails=0)
        assert rec.health_ratio == 1.0

    def test_health_ratio_half(self) -> None:
        rec = ReputationRecord(node_id="n", health_check_passes=3, health_check_fails=3)
        assert rec.health_ratio == 0.5

    def test_health_ratio_all_fail(self) -> None:
        rec = ReputationRecord(node_id="n", health_check_passes=0, health_check_fails=7)
        assert rec.health_ratio == 0.0

    def test_uptime_hours_positive(self) -> None:
        """uptime_hours returns a positive value based on wall clock."""
        rec = ReputationRecord(node_id="n")
        assert rec.uptime_hours >= 0.0

    def test_uptime_hours_increases(self) -> None:
        """uptime_hours grows as time passes."""
        rec = ReputationRecord(node_id="n")
        first = rec.uptime_hours
        time.sleep(0.01)
        second = rec.uptime_hours
        assert second >= first

    def test_explicit_field_values(self) -> None:
        """Manually assigned fields produce the expected computed properties."""
        now = time.time()
        rec = ReputationRecord(
            node_id="explicit",
            total_requests=20,
            successful_requests=15,
            failed_requests=5,
            total_tokens=5000,
            total_latency_ms=300.0,
            health_check_passes=8,
            health_check_fails=2,
            first_seen=now - 3600,       # 1 hour ago
            last_seen=now,
            last_failure=now - 1800,
            tokens_contributed=1000,
            credits_earned=50.0,
            credits_spent=10.0,
        )
        assert rec.reliability == pytest.approx(0.75)
        assert rec.avg_latency_ms == pytest.approx(20.0)
        assert rec.health_ratio == pytest.approx(0.8)
        assert rec.uptime_hours == pytest.approx(1.0, rel=0.1)


class TestReputationSystem:
    """Tests for the ReputationSystem class."""

    def test_empty_unknown_node_score(self) -> None:
        """get_score returns 0.5 for nodes that have never been seen."""
        system = ReputationSystem()
        assert system.get_score("nonexistent") == 0.5

    def test_empty_get_scores(self) -> None:
        """get_scores returns empty dict when no records exist."""
        system = ReputationSystem()
        assert system.get_scores() == {}

    def test_record_success_creates_record(self) -> None:
        system = ReputationSystem()
        system.record_success("node_a", latency_ms=50.0, tokens=128)
        assert system.get_score("node_a") > 0.0

    def test_record_failure_creates_record(self) -> None:
        system = ReputationSystem()
        system.record_failure("node_b")
        # one failure only → total_requests=1, failed=1 → reliability 0.0
        score = system.get_score("node_b")
        assert score < 0.5

    def test_record_success_fields(self) -> None:
        system = ReputationSystem()
        system.record_success("n", latency_ms=30.0, tokens=64)
        rec = system._records["n"]
        assert rec.total_requests == 1
        assert rec.successful_requests == 1
        assert rec.total_latency_ms == 30.0
        assert rec.total_tokens == 64

    def test_record_failure_fields(self) -> None:
        system = ReputationSystem()
        system.record_failure("n")
        rec = system._records["n"]
        assert rec.total_requests == 1
        assert rec.failed_requests == 1
        assert rec.last_failure > 0

    def test_record_health_pass(self) -> None:
        system = ReputationSystem()
        system.record_health("n", healthy=True)
        rec = system._records["n"]
        assert rec.health_check_passes == 1
        assert rec.health_check_fails == 0

    def test_record_health_fail(self) -> None:
        system = ReputationSystem()
        system.record_health("n", healthy=False)
        rec = system._records["n"]
        assert rec.health_check_passes == 0
        assert rec.health_check_fails == 1

    def test_record_health_updates_last_seen(self) -> None:
        system = ReputationSystem()
        system.record_success("n")
        before = system._records["n"].last_seen
        time.sleep(0.005)
        system.record_health("n", healthy=True)
        after = system._records["n"].last_seen
        assert after > before

    def test_record_success_updates_last_seen(self) -> None:
        system = ReputationSystem()
        system.record_health("n", healthy=True)
        before = system._records["n"].last_seen
        time.sleep(0.005)
        system.record_success("n")
        after = system._records["n"].last_seen
        assert after > before

    def test_record_failure_updates_last_seen(self) -> None:
        system = ReputationSystem()
        system.record_success("n")
        before = system._records["n"].last_seen
        time.sleep(0.005)
        system.record_failure("n")
        after = system._records["n"].last_seen
        assert after > before

    def test_get_scores_after_records(self) -> None:
        system = ReputationSystem()
        system.record_success("a")
        system.record_failure("b")
        scores = system.get_scores()
        assert "a" in scores
        assert "b" in scores
        assert len(scores) == 2

    def test_get_scores_immutable_copy(self) -> None:
        """get_scores returns a new dict each time (scores are approximately equal within same ms)."""
        system = ReputationSystem()
        system.record_success("n")
        s1 = system.get_scores()
        s2 = system.get_scores()
        # Scores may differ slightly due to uptime moving between calls
        assert s1.keys() == s2.keys()
        for k in s1:
            assert s1[k] == pytest.approx(s2[k], abs=1e-4)
        # modifying the returned dict does not affect the system
        s1.clear()
        assert system.get_scores()["n"] > 0.0

    # ── Score edge cases ──────────────────────────────────────────────

    def test_score_perfect_reliability(self) -> None:
        """Perfect reliability plus health yields a high score."""
        system = ReputationSystem()
        for _ in range(10):
            system.record_success("good_node", latency_ms=10.0)
        for _ in range(10):
            system.record_health("good_node", healthy=True)
        score = system.get_score("good_node")
        assert score > 0.5
        assert score <= 1.0

    def test_score_bad_node(self) -> None:
        """All failures and health fails yields a low score."""
        system = ReputationSystem()
        for _ in range(5):
            system.record_failure("bad_node")
        for _ in range(5):
            system.record_health("bad_node", healthy=False)
        score = system.get_score("bad_node")
        assert score < 0.3

    def test_persistent_failure_floor(self) -> None:
        """A node with 5+ failures and more failures than successes gets 0.1."""
        system = ReputationSystem()
        # 5 failures, 0 successes → failed >= 5 AND failed > success
        for _ in range(5):
            system.record_failure("n")
        assert system.get_score("n") == 0.1

    def test_persistent_failure_floor_not_applied_few_failures(self) -> None:
        """4 failures with 0 successes should NOT get the 0.1 floor."""
        system = ReputationSystem()
        for _ in range(4):
            system.record_failure("n")
        score = system.get_score("n")
        assert score != 0.1  # 0.1 is reserved for the 5-fail floor

    def test_persistent_failure_floor_not_applied_when_more_successes(self) -> None:
        """5 failures but 6 successes → floor NOT applied because successes > failures."""
        system = ReputationSystem()
        for _ in range(5):
            system.record_failure("n")
        for _ in range(6):
            system.record_success("n", latency_ms=20.0)
        score = system.get_score("n")
        assert score > 0.1

    def test_score_clamped_upper(self) -> None:
        """Score never exceeds 1.0."""
        system = ReputationSystem()
        # Many successes with low latency and perfect health
        for _ in range(100):
            system.record_success("n", latency_ms=1.0)
        for _ in range(100):
            system.record_health("n", healthy=True)
        score = system.get_score("n")
        assert score <= 1.0

    def test_score_clamped_lower(self) -> None:
        """Score never goes below 0.0."""
        system = ReputationSystem()
        # A node that only ever fails and fails health checks
        for _ in range(100):
            system.record_failure("n")
        for _ in range(100):
            system.record_health("n", healthy=False)
        score = system.get_score("n")
        assert score >= 0.0

    def test_speed_score_single_node(self) -> None:
        """With only one node, speed score is neutral (0.5)."""
        system = ReputationSystem()
        system.record_success("n", latency_ms=9999.0)
        rec = system._records["n"]
        ss = system._compute_speed_score(rec)
        assert ss == 0.5

    def test_speed_score_ratio(self) -> None:
        """With two nodes, the faster node gets a higher speed score."""
        system = ReputationSystem()
        system.record_success("slow", latency_ms=100.0)
        system.record_success("fast", latency_ms=10.0)
        slow_rec = system._records["slow"]
        fast_rec = system._records["fast"]
        slow_ss = system._compute_speed_score(slow_rec)
        fast_ss = system._compute_speed_score(fast_rec)
        assert fast_ss > slow_ss
        assert 0.0 <= fast_ss <= 1.0
        assert 0.0 <= slow_ss <= 1.0

    def test_speed_score_clamped(self) -> None:
        """Speed score is clamped to [0.0, 1.0]."""
        system = ReputationSystem()
        system.record_success("normal", latency_ms=50.0)
        system.record_success("extremely_slow", latency_ms=1e12)
        slow_rec = system._records["extremely_slow"]
        ss = system._compute_speed_score(slow_rec)
        assert 0.0 <= ss <= 1.0

    # ── Multiple nodes ────────────────────────────────────────────────

    def test_multiple_nodes_independent(self) -> None:
        """Scores of different nodes are independent."""
        system = ReputationSystem()
        system.record_success("good", latency_ms=10.0)
        system.record_success("good", latency_ms=10.0)
        system.record_failure("bad")
        system.record_failure("bad")
        system.record_failure("bad")
        good_score = system.get_score("good")
        bad_score = system.get_score("bad")
        assert good_score > bad_score

    def test_get_scores_reflects_all_nodes(self) -> None:
        system = ReputationSystem()
        for nid in ("a", "b", "c"):
            system.record_success(nid, latency_ms=20.0)
        scores = system.get_scores()
        assert set(scores.keys()) == {"a", "b", "c"}

    # ── Contribution credits / token economy ──────────────────────────

    def test_record_contribution(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=1000, credit_rate=2.0)
        balance = system.get_credit_balance("n")
        assert balance == 2000.0

    def test_record_contribution_default_rate(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=500)
        assert system.get_credit_balance("n") == 500.0

    def test_spend_credits_sufficient(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=100)
        assert system.spend_credits("n", tokens_consumed=50) is True
        assert system.get_credit_balance("n") == 50.0

    def test_spend_credits_insufficient(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=10)
        assert system.spend_credits("n", tokens_consumed=20) is False
        assert system.get_credit_balance("n") == 10.0  # unchanged

    def test_spend_credits_exact_balance(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=50)
        assert system.spend_credits("n", tokens_consumed=50) is True
        assert system.get_credit_balance("n") == 0.0

    def test_spend_credits_zero_consumed(self) -> None:
        system = ReputationSystem()
        system.record_contribution("n", tokens_computed=10)
        assert system.spend_credits("n", tokens_consumed=0) is True
        assert system.get_credit_balance("n") == 10.0

    def test_credit_balance_no_contributions(self) -> None:
        system = ReputationSystem()
        assert system.get_credit_balance("unknown") == 0.0

    def test_credit_summary(self) -> None:
        system = ReputationSystem()
        system.record_contribution("a", tokens_computed=200, credit_rate=1.5)
        system.record_contribution("a", tokens_computed=100)
        system.spend_credits("a", tokens_consumed=50)
        summary = system.get_credit_summary()
        assert "a" in summary
        entry = summary["a"]
        assert entry["tokens_contributed"] == 300
        assert entry["credits_earned"] == pytest.approx(400.0)  # 200*1.5 + 100*1.0
        assert entry["credits_spent"] == pytest.approx(50.0)
        assert entry["credit_balance"] == pytest.approx(350.0)

    def test_credit_summary_empty(self) -> None:
        system = ReputationSystem()
        assert system.get_credit_summary() == {}

    # ── get_summary ───────────────────────────────────────────────────

    def test_get_summary(self) -> None:
        system = ReputationSystem(min_reputation=0.3)
        system.record_success("n", latency_ms=40.0, tokens=256)
        system.record_health("n", healthy=True)
        summary = system.get_summary()
        assert summary["min_reputation"] == 0.3
        assert "n" in summary["nodes"]
        node = summary["nodes"]["n"]
        assert "score" in node
        assert "reliability" in node
        assert "health_ratio" in node
        assert "total_requests" in node
        assert "failed_requests" in node
        assert "uptime_hours" in node
        assert node["total_requests"] == 1
        assert node["failed_requests"] == 0

    def test_get_summary_empty(self) -> None:
        system = ReputationSystem()
        summary = system.get_summary()
        assert summary["nodes"] == {}
        assert summary["min_reputation"] == 0.0
        assert summary["weights"] == {
            "reliability": 0.40,
            "health": 0.25,
            "speed": 0.20,
            "uptime": 0.15,
        }

    # ── is_qualified ──────────────────────────────────────────────────

    def test_is_qualified_unknown_node(self) -> None:
        """Unknown nodes have score 0.5, which meets threshold 0.0."""
        system = ReputationSystem(min_reputation=0.0)
        assert system.is_qualified("unknown") is True

    def test_is_qualified_meets_threshold(self) -> None:
        """A good node meets a reasonable threshold."""
        system = ReputationSystem(min_reputation=0.6)
        for _ in range(20):
            system.record_success("n", latency_ms=5.0)
        # Also record health checks to boost the score
        for _ in range(5):
            system.record_health("n", healthy=True)
        assert system.is_qualified("n") is True

    def test_is_qualified_below_threshold(self) -> None:
        system = ReputationSystem(min_reputation=0.9)
        for _ in range(5):
            system.record_failure("n")
        assert system.is_qualified("n") is False

    # ── set_min_reputation ────────────────────────────────────────────

    def test_set_min_reputation(self) -> None:
        system = ReputationSystem()
        system.set_min_reputation(0.75)
        assert system._min_reputation == 0.75

    def test_set_min_reputation_clamps_below_zero(self) -> None:
        system = ReputationSystem()
        system.set_min_reputation(-0.5)
        assert system._min_reputation == 0.0

    def test_set_min_reputation_clamps_above_one(self) -> None:
        system = ReputationSystem()
        system.set_min_reputation(1.5)
        assert system._min_reputation == 1.0

    def test_set_min_reputation_edge(self) -> None:
        system = ReputationSystem()
        system.set_min_reputation(0.0)
        assert system._min_reputation == 0.0
        system.set_min_reputation(1.0)
        assert system._min_reputation == 1.0

    # ── Initial min_reputation ────────────────────────────────────────

    def test_default_min_reputation(self) -> None:
        system = ReputationSystem()
        assert system._min_reputation == 0.0

    def test_custom_min_reputation(self) -> None:
        system = ReputationSystem(min_reputation=0.5)
        assert system._min_reputation == 0.5

    # ── Re-entrant lock usage ─────────────────────────────────────────

    def test_lock_is_rlock(self) -> None:
        """The internal lock is a re-entrant lock so nested acquires work."""
        system = ReputationSystem()
        assert isinstance(system._lock, type(threading.RLock()))
        # nested acquire under the same thread must not deadlock
        with system._lock:
            with system._lock:
                system.record_success("n")
        assert system.get_score("n") > 0.0

    # ── is_qualified with dynamic threshold ───────────────────────────

    def test_is_qualified_changes_with_threshold(self) -> None:
        system = ReputationSystem(min_reputation=0.0)
        system.record_success("n")
        assert system.is_qualified("n") is True
        system.set_min_reputation(1.0)
        assert system.is_qualified("n") is False

    # ── Default credit rate ───────────────────────────────────────────

    def test_record_contribution_multi_node(self) -> None:
        system = ReputationSystem()
        system.record_contribution("a", tokens_computed=100, credit_rate=1.0)
        system.record_contribution("b", tokens_computed=100, credit_rate=3.0)
        assert system.get_credit_balance("a") == 100.0
        assert system.get_credit_balance("b") == 300.0

    # ── get_score after mixed records ─────────────────────────────────

    def test_score_mixed_records(self) -> None:
        system = ReputationSystem()
        # 3 successes, 2 failures
        for _ in range(3):
            system.record_success("n", latency_ms=30.0)
        for _ in range(2):
            system.record_failure("n")
        # health: 4 passes, 1 fail
        for _ in range(4):
            system.record_health("n", healthy=True)
        system.record_health("n", healthy=False)
        score = system.get_score("n")
        assert 0.0 <= score <= 1.0
        # reliability = 0.6 * 0.40 = 0.24
        # health = 0.8 * 0.25 = 0.20
        # speed = ... * 0.20
        # uptime = ... * 0.15
        # total >= 0.24 + 0.20
        assert score >= 0.44
