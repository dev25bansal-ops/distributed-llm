"""Tests for LearningRouter — online RL-based model selection via contextual bandits.

Covers:
- _feature_hash: deterministic hashing with L2 normalization
- RewardSignal: individual components and to_reward() scalar
- _ArmStats: running statistics
- LearningRouter: construction, routing (cold start, explore, exploit),
  record_outcome, policy persistence (save/load), stats
"""

from __future__ import annotations

import json
import math
import tempfile

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_lr = load_module("distllm/core/learning_router.py")
LearningRouter = _lr.LearningRouter
RewardSignal = _lr.RewardSignal
_ArmStats = _lr._ArmStats
_feature_hash = _lr._feature_hash

# We need ModelRouter and RoutingContext for tests that call the base router.
# Import them normally since they're already bootstrapped.
from distllm.core.model_router import ModelRouter, RoutingContext, RouteMatch
from distllm.config.settings import ChatRouterSettings, RouteRuleSettings


# ── _feature_hash ─────────────────────────────────────────────────────────────


class TestFeatureHash:
    def test_returns_fixed_size_vector(self):
        vec = _feature_hash("hello", num_buckets=256)
        assert len(vec) == 256

    def test_values_in_range(self):
        vec = _feature_hash("test string", num_buckets=256)
        for v in vec:
            assert -1.0 <= v <= 1.0

    def test_deterministic(self):
        v1 = _feature_hash("hello world", 64)
        v2 = _feature_hash("hello world", 64)
        assert v1 == v2

    def test_different_inputs_different_vectors(self):
        v1 = _feature_hash("cat", 64)
        v2 = _feature_hash("dog", 64)
        assert v1 != v2

    def test_l2_normalized(self):
        vec = _feature_hash("hello world", 256)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_empty_string_returns_normalized_vector(self):
        vec = _feature_hash("", 256)
        assert len(vec) == 256
        norm = math.sqrt(sum(v * v for v in vec))
        # Empty string has no n-grams -> zero vector, kept as-is when norm is 0
        assert norm == 0.0

    def test_custom_bucket_count(self):
        vec = _feature_hash("hello", 128)
        assert len(vec) == 128


# ── _ArmStats ──────────────────────────────────────────────────────────────────


class TestArmStats:
    def test_defaults(self):
        a = _ArmStats()
        assert a.pulls == 0
        assert a.total_reward == 0.0
        assert a.mean_reward == 0.0

    def test_update(self):
        a = _ArmStats()
        a.update(1.0)
        a.update(0.5)
        assert a.pulls == 2
        assert a.total_reward == 1.5
        assert a.mean_reward == 0.75


# ── RewardSignal ──────────────────────────────────────────────────────────────


class TestRewardSignal:
    def test_default_reward(self):
        rs = RewardSignal()
        assert rs.to_reward() == 0.5

    def test_user_rating(self):
        rs = RewardSignal(user_rating=0.9)
        assert rs.to_reward() == 0.9

    def test_user_rating_clamped(self):
        rs = RewardSignal(user_rating=1.5)
        assert rs.to_reward() == 1.0
        rs = RewardSignal(user_rating=-0.5)
        assert rs.to_reward() == 0.0

    def test_latency_within_sla(self):
        rs = RewardSignal(latency_ms=100, latency_sla_ms=200)
        # 100/200=0.5, 1.0 - (0.5-1)/2 = 1.0
        assert rs.to_reward() >= 0.9

    def test_latency_exceeds_sla(self):
        rs = RewardSignal(latency_ms=500, latency_sla_ms=200)
        # 500/200 = 2.5, 1.0 - (2.5-1)/2 = 0.25
        reward = rs.to_reward()
        assert 0.2 <= reward <= 0.3

    def test_latency_three_times_sla(self):
        rs = RewardSignal(latency_ms=600, latency_sla_ms=200)
        # At 3x, reward hits 0
        assert rs.to_reward() == pytest.approx(0.0, abs=0.01)

    def test_cost_under_budget(self):
        rs = RewardSignal(cost_usd=0.5, cost_budget_usd=1.0)
        reward = rs.to_reward()
        assert reward >= 0.9

    def test_cost_over_budget(self):
        rs = RewardSignal(cost_usd=1.5, cost_budget_usd=1.0)
        # 1.5/1.0 = 1.5, 1.0 - (1.5-1) = 0.5
        reward = rs.to_reward()
        assert 0.4 <= reward <= 0.6

    def test_quality_score(self):
        rs = RewardSignal(quality_score=0.8)
        assert rs.to_reward() == 0.8

    def test_combined_signals_averaged(self):
        rs = RewardSignal(
            user_rating=1.0,
            quality_score=0.0,
            latency_ms=100,
            latency_sla_ms=200,
            cost_usd=0.5,
            cost_budget_usd=1.0,
        )
        reward = rs.to_reward()
        # All signals should be 1.0 (within budget), so avg = 1.0
        assert 0.9 <= reward <= 1.0


# ── Helper to create a base router ────────────────────────────────────────────


def _default_base_router() -> ModelRouter:
    settings = ChatRouterSettings(
        enabled=True,
        name="test",
        default_model="llama3",
        routes=[],
    )
    return ModelRouter(settings)


# ── LearningRouter ────────────────────────────────────────────────────────────


class TestLearningRouterConstruction:
    def test_default_construction(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3", "mathgpt"])
        assert lr._models == ["codellama", "llama3", "mathgpt"]
        assert lr._epsilon == 0.15
        assert lr._epsilon_decay == 0.999
        assert lr._epsilon_floor == 0.02
        assert lr._total_decisions == 0
        assert lr._explore_count == 0
        assert lr._exploit_count == 0

    def test_custom_params(self):
        base = _default_base_router()
        lr = LearningRouter(
            base, models=["a", "b"],
            epsilon=0.3, epsilon_decay=0.99, epsilon_floor=0.05,
            num_buckets=128, context_granularity=32,
        )
        assert lr._epsilon == 0.3
        assert lr._num_buckets == 128
        assert lr._context_granularity == 32


class TestLearningRouterColdStart:
    def test_route_returns_model_string(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        model = lr.route("hello", tenant_id="")
        assert isinstance(model, str)
        assert model in ("codellama", "llama3")

    def test_route_with_context_returns_route_match(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        match = lr.route_with_context(
            [{"role": "user", "content": "hello"}],
            available_models=["codellama", "llama3"],
        )
        assert isinstance(match, RouteMatch)
        assert match.model in ("codellama", "llama3")
        assert match.rule_name == "learning"

    def test_route_with_context_empty_available_falls_back(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        match = lr.route_with_context(
            [{"role": "user", "content": "hello"}],
            available_models=[],
        )
        assert isinstance(match, RouteMatch)
        assert match.model == "llama3"  # default

    def test_cold_start_all_arms_zero_pulls_uses_base(self):
        """When all arms have zero pulls, should use base router."""
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        model = lr.route("hello")
        assert isinstance(model, str)


class TestLearningRouterRecordOutcome:
    def test_record_outcome_updates_policy(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        lr.record_outcome(
            "codellama",
            RewardSignal(user_rating=0.9),
            text="hello",
            tenant_id="tenant1",
        )
        policy = lr._get_policy("tenant1")
        assert len(policy) == 1
        bucket = list(policy.keys())[0]
        assert "codellama" in policy[bucket]

    def test_record_outcome_decays_epsilon(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"], epsilon=0.5, epsilon_decay=0.5)
        lr.record_outcome("codellama", RewardSignal(user_rating=0.9), text="hello")
        assert lr._epsilon < 0.5
        assert lr._epsilon >= 0.02  # floor

    def test_outcome_updates_correct_context_bucket(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        lr.record_outcome("codellama", RewardSignal(user_rating=0.8), text="hello world")
        policy = lr._get_policy("")
        assert len(policy) > 0

    def test_route_after_learning_uses_exploit(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"], epsilon=0.0)
        # Record many outcomes so arms have data for both models
        for _ in range(10):
            lr.record_outcome("llama3", RewardSignal(user_rating=1.0), text="hello")
            lr.record_outcome("codellama", RewardSignal(user_rating=0.0), text="hello")
        model = lr.route("hello")
        assert model == "llama3"  # Best arm should win


class TestLearningRouterPolicyPersistence:
    def test_save_policy_no_error(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        lr.record_outcome("a", RewardSignal(user_rating=0.5), text="hello")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            lr.save_policy(path)
            with open(path) as f:
                data = json.load(f)
            assert data["version"] == 1
            assert data["epsilon"] >= 0
            assert data["total_decisions"] >= 0
        finally:
            import os
            os.unlink(path)

    def test_load_policy(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        data = {
            "version": 1,
            "epsilon": 0.05,
            "total_decisions": 10,
            "explore_count": 2,
            "exploit_count": 8,
            "policies": {
                "tenant1": {
                    "0": {"a": {"pulls": 5, "total_reward": 4.0}},
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            assert lr.load_policy(path) is True
            assert lr._epsilon == pytest.approx(0.05)
            assert lr._total_decisions == 10
            tenant_policy = lr._get_policy("tenant1")
            assert 0 in tenant_policy
            assert tenant_policy[0]["a"].pulls == 5
        finally:
            import os
            os.unlink(path)

    def test_load_policy_missing_file(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        assert lr.load_policy("/nonexistent/policy.json") is False

    def test_load_policy_wrong_version(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        data = {"version": 999, "epsilon": 0.1}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            assert lr.load_policy(path) is False
        finally:
            import os
            os.unlink(path)

    def test_load_policy_corrupt_json(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            f.write("not valid json")
            path = f.name
        try:
            assert lr.load_policy(path) is False
        finally:
            import os
            os.unlink(path)

    def test_save_load_roundtrip(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["a", "b"])
        lr.record_outcome("a", RewardSignal(user_rating=0.9), text="hello", tenant_id="t1")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            path = f.name
        try:
            lr.save_policy(path)

            lr2 = LearningRouter(base, models=["a", "b"])
            assert lr2.load_policy(path) is True
            assert lr2._epsilon == lr._epsilon
            assert lr2._total_decisions == lr._total_decisions
        finally:
            import os
            os.unlink(path)


class TestLearningRouterStats:
    def test_stats_empty(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        s = lr.stats
        assert s["total_decisions"] == 0
        assert s["explore_count"] == 0
        assert s["exploit_count"] == 0
        assert s["num_tenants"] == 0
        assert s["total_context_buckets"] == 0

    def test_stats_after_activity(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        lr.route("hello", tenant_id="t1")
        lr.record_outcome("llama3", RewardSignal(user_rating=0.9), text="hello", tenant_id="t1")
        s = lr.stats
        assert s["total_decisions"] >= 1
        assert s["num_tenants"] >= 1


class TestLearningRouterEdgeCases:
    def test_route_empty_text(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        model = lr.route("")
        assert isinstance(model, str)

    def test_route_with_null_context(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        model = lr.route("hello", ctx=None)
        assert isinstance(model, str)

    def test_record_outcome_empty_text(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        # Should not raise
        lr.record_outcome("llama3", RewardSignal(user_rating=0.5), text="")

    def test_get_policy_invalid_tenant(self):
        base = _default_base_router()
        lr = LearningRouter(base, models=["codellama", "llama3"])
        policy = lr._get_policy("nonexistent")
        assert isinstance(policy, dict)
