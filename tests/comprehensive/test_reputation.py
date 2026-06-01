"""Reputation score computation tests.

Covers weighted score computation (40% reliability, 25% health, 20% speed,
15% uptime), credit system, and property-based score bounds.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_reputation = _load_module("distllm/dist/reputation.py")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reputation Score Computation
# ═══════════════════════════════════════════════════════════════════════════

class TestReputationScore:
    """Weighted score: 40% reliability, 25% health, 20% speed, 15% uptime."""

    @pytest.fixture
    def sys(self):
        return _reputation.ReputationSystem()

    def test_new_node_gets_neutral_score(self, sys):
        score = sys.get_score("unknown")
        assert score == 0.5

    def test_perfect_node_scores_high(self, sys):
        nid = "perfect"
        for _ in range(10):
            sys.record_success(nid, latency_ms=10.0, tokens=50)
        for _ in range(10):
            sys.record_health(nid, healthy=True)
        score = sys.get_score(nid)
        # 40% reliability(1.0) + 25% health(1.0) + 20% speed(0.5 no comp) + 15% uptime(0)
        assert score == pytest.approx(0.75, abs=1e-3)

    def test_failing_node_scores_low(self, sys):
        nid = "failing"
        for _ in range(10):
            sys.record_failure(nid)
        sys.record_health(nid, healthy=False)
        score = sys.get_score(nid)
        assert score < 0.3

    def test_persistently_failing_gets_0_1(self, sys):
        nid = "bad"
        for _ in range(5):
            sys.record_failure(nid)
        sys.record_success(nid, latency_ms=100, tokens=1)
        score = sys.get_score(nid)
        assert score == 0.1

    def test_record_success_reliability(self, sys):
        sys.record_success("n1", latency_ms=42.0, tokens=100)
        rec = sys._records["n1"]
        assert rec.total_requests == 1
        assert rec.successful_requests == 1
        assert rec.total_latency_ms == 42.0
        assert rec.total_tokens == 100
        assert rec.reliability == 1.0

    def test_record_failure_reliability(self, sys):
        sys.record_failure("n1")
        rec = sys._records["n1"]
        assert rec.total_requests == 1
        assert rec.failed_requests == 1
        assert rec.reliability == 0.0

    def test_record_health(self, sys):
        sys.record_health("n1", healthy=True)
        sys.record_health("n1", healthy=False)
        rec = sys._records["n1"]
        assert rec.health_check_passes == 1
        assert rec.health_check_fails == 1
        assert rec.health_ratio == 0.5

    def test_reliability_property_zero_requests(self):
        rec = _reputation.ReputationRecord(node_id="n")
        assert rec.reliability == 0.0

    def test_avg_latency_property(self, sys):
        sys.record_success("n1", latency_ms=50.0)
        sys.record_success("n1", latency_ms=30.0)
        assert sys._records["n1"].avg_latency_ms == 40.0

    def test_avg_latency_zero_when_no_success(self):
        rec = _reputation.ReputationRecord(node_id="n")
        assert rec.avg_latency_ms == 0.0

    def test_is_qualified_below_threshold(self):
        sys = _reputation.ReputationSystem(min_reputation=0.8)
        nid = "unknown"
        assert sys.get_score(nid) == 0.5
        assert not sys.is_qualified(nid)

    def test_is_qualified_above_threshold(self, sys):
        nid = "good"
        for _ in range(10):
            sys.record_success(nid, latency_ms=5.0)
            sys.record_health(nid, healthy=True)
        assert sys.is_qualified(nid)

    def test_set_min_reputation(self, sys):
        sys.set_min_reputation(0.9)
        assert sys._min_reputation == 0.9
        sys.set_min_reputation(-0.1)
        assert sys._min_reputation == 0.0
        sys.set_min_reputation(1.5)
        assert sys._min_reputation == 1.0

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=100)
    @given(
        successes=st.integers(min_value=0, max_value=50),
        failures=st.integers(min_value=0, max_value=50),
        health_passes=st.integers(min_value=0, max_value=50),
        health_fails=st.integers(min_value=0, max_value=50),
    )
    def test_score_bounds_property(self, successes, failures, health_passes, health_fails):
        sys = _reputation.ReputationSystem()
        nid = "prop"
        for _ in range(successes):
            sys.record_success(nid, latency_ms=20.0)
        for _ in range(failures):
            sys.record_failure(nid)
        for _ in range(health_passes):
            sys.record_health(nid, healthy=True)
        for _ in range(health_fails):
            sys.record_health(nid, healthy=False)
        score = sys.get_score(nid)
        assert 0.0 <= score <= 1.0

    def test_get_summary_returns_expected_keys(self, sys):
        sys.record_success("n1", latency_ms=10.0, tokens=10)
        summary = sys.get_summary()
        assert "min_reputation" in summary
        assert "nodes" in summary
        assert "weights" in summary
        assert "n1" in summary["nodes"]

    def test_uptime_hours_increases(self):
        rec = _reputation.ReputationRecord(node_id="n")
        u1 = rec.uptime_hours
        time.sleep(0.001)
        u2 = rec.uptime_hours
        assert u2 >= u1

    def test_last_seen_updates_on_success(self, sys):
        t1 = time.time()
        sys.record_success("n1")
        rec = sys._records["n1"]
        assert rec.last_seen >= t1

    def test_last_failure_updates_on_failure(self, sys):
        sys.record_failure("n1")
        rec = sys._records["n1"]
        assert rec.last_failure > 0

    def test_get_scores_returns_neutral_for_unknown(self):
        rec = _reputation.ReputationRecord(node_id="n")
        assert rec.reliability == 0.0
        assert rec.health_ratio == 1.0

    def test_compute_speed_score_single_node_returns_05(self):
        sys = _reputation.ReputationSystem()
        sys.record_success("n1", latency_ms=50.0)
        rec = sys._records["n1"]
        speed = sys._compute_speed_score(rec)
        assert speed == 0.5

    def test_record_contribution_adds_credits(self):
        sys = _reputation.ReputationSystem()
        sys.record_contribution("n1", tokens_computed=1000, credit_rate=2.0)
        rec = sys._records["n1"]
        assert rec.tokens_contributed == 1000
        assert rec.credits_earned == 2000.0

    def test_spend_credits_allows_with_balance(self):
        sys = _reputation.ReputationSystem()
        sys.record_contribution("n1", tokens_computed=1000)
        result = sys.spend_credits("n1", tokens_consumed=500)
        assert result
        assert sys.get_credit_balance("n1") == 500.0

    def test_spend_credits_denies_without_balance(self):
        sys = _reputation.ReputationSystem()
        sys.record_contribution("n1", tokens_computed=100)
        result = sys.spend_credits("n1", tokens_consumed=500)
        assert not result
        assert sys.get_credit_balance("n1") == 100.0

    def test_credit_summary(self):
        sys = _reputation.ReputationSystem()
        sys.record_contribution("n1", tokens_computed=500, credit_rate=1.0)
        summary = sys.get_credit_summary()
        assert "n1" in summary
        assert summary["n1"]["credit_balance"] == 500.0
