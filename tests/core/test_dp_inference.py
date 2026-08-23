"""Tests for DP inference modules: DifferentialPrivacyInference,
PrivacyBudgetManager, RDPAccounting, and standalone DP operations.

Covers:
- DPConfig: default values, sigma computation, rdp_orders
- BudgetEntry: defaults
- RDPAccounting: compute_rdp, add_query, get_epsilon, get_privacy_spent
- PrivacyBudgetManager: defaults, tenant budgets, check_budget, record_query,
  periodic reset, global privacy spent, all_tenants, reset_all
- clip_gradients: per-layer and global norm clipping
- dp_noise_injection: Gaussian noise addition
- gumbel_noise_mechanism: Gumbel noise perturbation
- DifferentialPrivacyInference: construction, config, noise injection,
  tensor clipping, logit DP, generate calls, budget enforcement, _dp_sample,
  _estimate_epsilon_cost
- wrap_with_dp: convenience wrapper
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_dpi = load_module("distllm/core/dp_inference.py")
DifferentialPrivacyInference = _dpi.DifferentialPrivacyInference
PrivacyBudgetManager = _dpi.PrivacyBudgetManager
RDPAccounting = _dpi.RDPAccounting
DPConfig = _dpi.DPConfig
BudgetEntry = _dpi.BudgetEntry
DPGenerationResult = _dpi.DPGenerationResult
clip_gradients = _dpi.clip_gradients
dp_noise_injection = _dpi.dp_noise_injection
gumbel_noise_mechanism = _dpi.gumbel_noise_mechanism
wrap_with_dp = _dpi.wrap_with_dp


try:
    import torch
except ImportError:
    torch = None

# Helper to skip if torch is missing
requires_torch = pytest.mark.skipif(torch is None, reason="torch not available")


class _FakeEngine:
    """Deterministic engine stub for DP inference tests.

    Tests configure behavior by setting attributes directly::

        engine = _FakeEngine()
        engine.generate_stream = lambda prompt, **kw: ["hello", "world"]
        engine.generate = lambda prompt, **kw: ["hello", "world"]
    """
    pass


# ── DPConfig ──────────────────────────────────────────────────────────────────


class TestDPConfig:
    def test_default_values(self):
        cfg = DPConfig()
        assert cfg.epsilon == 4.0
        assert cfg.delta == 1e-6
        assert cfg.max_grad_norm == 1.0
        assert cfg.noise_multiplier == 0.0
        assert cfg.target_mechanism == "dp-sgd"
        assert cfg.gumbel_noise_scale == 1.0
        assert cfg.clip_per_layer is True

    def test_rdp_orders_defaulted(self):
        cfg = DPConfig()
        assert len(cfg.rdp_orders) > 0
        assert 1.0 in cfg.rdp_orders
        assert 10.0 in cfg.rdp_orders

    def test_custom_rdp_orders(self):
        cfg = DPConfig(rdp_orders=[0.5, 1.0, 2.0])
        assert cfg.rdp_orders == [0.5, 1.0, 2.0]

    def test_sigma_noise_multiplier_zero(self):
        cfg = DPConfig(epsilon=4.0, delta=1e-6)
        expected = 1.0 * math.sqrt(2 * math.log(1.25 / 1e-6)) / 4.0
        assert cfg.sigma == pytest.approx(expected, rel=1e-4)

    def test_sigma_noise_multiplier_set(self):
        cfg = DPConfig(noise_multiplier=0.5, max_grad_norm=2.0)
        assert cfg.sigma == 1.0

    def test_sigma_scales_with_epsilon(self):
        low = DPConfig(epsilon=1.0).sigma
        high = DPConfig(epsilon=10.0).sigma
        assert low > high


# ── BudgetEntry ───────────────────────────────────────────────────────────────


class TestBudgetEntry:
    def test_defaults(self):
        be = BudgetEntry(epsilon=4.0, delta=1e-6)
        assert be.epsilon == 4.0
        assert be.delta == 1e-6
        assert be.epsilon_spent == 0.0
        assert be.delta_spent == 0.0
        assert be.num_queries == 0
        assert be.last_reset > 0


# ── RDPAccounting ─────────────────────────────────────────────────────────────


class TestRDPAccounting:
    def test_default_orders(self):
        acc = RDPAccounting()
        assert len(acc.orders) > 0
        assert 1.0 in acc.orders

    def test_compute_rdp_zero_sigma(self):
        acc = RDPAccounting()
        result = acc.compute_rdp(sigma=0.0)
        assert all(v == 0.0 for v in result)

    def test_compute_rdp_positive_sigma(self):
        acc = RDPAccounting(orders=[1.0, 2.0, 5.0])
        result = acc.compute_rdp(sigma=1.0)
        assert len(result) == 3
        # RDP(alpha) = alpha / (2 * sigma^2) = alpha / 2
        assert result[0] == pytest.approx(0.5)  # 1.0 / 2
        assert result[1] == pytest.approx(1.0)  # 2.0 / 2
        assert result[2] == pytest.approx(2.5)  # 5.0 / 2

    def test_compute_rdp_scales_with_num_queries(self):
        acc = RDPAccounting(orders=[2.0])
        r1 = acc.compute_rdp(sigma=2.0, num_queries=1)
        r5 = acc.compute_rdp(sigma=2.0, num_queries=5)
        assert r5[0] == pytest.approx(r1[0] * 5)

    def test_add_query_updates_total(self):
        acc = RDPAccounting(orders=[2.0])
        acc.add_query(sigma=2.0, query_id=1)
        acc.add_query(sigma=2.0, query_id=2)
        # Each: alpha / (2 * sigma^2) = 2.0 / (2 * 4) = 0.25
        expected_total = 0.25 * 2
        assert acc._total_rdp[0] == pytest.approx(expected_total)

    def test_add_query_tracks_per_query(self):
        acc = RDPAccounting(orders=[2.0])
        acc.add_query(sigma=2.0, query_id=42)
        assert 42 in acc._rdp_per_query

    def test_get_epsilon_empty(self):
        acc = RDPAccounting()
        assert acc.get_epsilon(delta=1e-6) == 0.0

    def test_get_epsilon_positive(self):
        acc = RDPAccounting(orders=[2.0])
        acc.add_query(sigma=2.0)
        eps = acc.get_epsilon(delta=1e-6)
        # eps = rdp - log(delta) / (alpha - 1)
        # rdp = 2.0 / (2 * 4) = 0.25
        # eps >= 0.25 - log(1e-6) / 1 = 0.25 + 13.816 = 14.066
        assert eps > 0

    def test_get_privacy_spent(self):
        acc = RDPAccounting(orders=[2.0])
        acc.add_query(sigma=2.0)
        spent = acc.get_privacy_spent(delta=1e-6)
        assert "epsilon" in spent
        assert "delta" in spent
        assert "orders_used" in spent
        assert spent["epsilon"] > 0
        assert spent["delta"] == 1e-6

    def test_get_privacy_spent_non_negative(self):
        acc = RDPAccounting(orders=[])
        spent = acc.get_privacy_spent(delta=1e-6)
        assert spent["epsilon"] == 0.0


# ── PrivacyBudgetManager ──────────────────────────────────────────────────────


class TestPrivacyBudgetManager:
    def test_defaults(self):
        mgr = PrivacyBudgetManager()
        assert mgr._reset_period == "daily"
        assert mgr._default_epsilon == 4.0
        assert mgr._default_delta == 1e-6
        assert mgr._budgets == {}

    def test_set_defaults(self):
        mgr = PrivacyBudgetManager()
        mgr.set_defaults(epsilon=8.0, delta=1e-5, period="weekly")
        assert mgr._default_epsilon == 8.0
        assert mgr._default_delta == 1e-5
        assert mgr._reset_period == "weekly"

    def test_set_defaults_invalid_period(self):
        mgr = PrivacyBudgetManager()
        mgr.set_defaults(epsilon=8.0, delta=1e-5, period="monthly")
        # Should stay at "daily"
        assert mgr._reset_period == "daily"

    def test_set_tenant_budget(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("tenant-1", epsilon=8.0, delta=1e-5)
        entry = mgr._budgets["tenant-1"]
        assert entry.epsilon == 8.0
        assert entry.delta == 1e-5

    def test_set_tenant_budget_defaults(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("tenant-1")
        entry = mgr._budgets["tenant-1"]
        assert entry.epsilon == 4.0  # default

    def test_set_tenant_budget_updates_existing(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("tenant-1", epsilon=8.0)
        mgr.set_tenant_budget("tenant-1", epsilon=16.0)
        assert mgr._budgets["tenant-1"].epsilon == 16.0

    def test_get_tenant_budget(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=8.0)
        entry = mgr.get_tenant_budget("t1")
        assert entry is not None
        assert entry.epsilon == 8.0

    def test_get_tenant_budget_missing(self):
        mgr = PrivacyBudgetManager()
        assert mgr.get_tenant_budget("nonexistent") is None

    def test_check_budget_auto_creates(self):
        mgr = PrivacyBudgetManager()
        status = mgr.check_budget("new-tenant")
        assert status["has_budget"] is True
        assert status["epsilon_remaining"] > 0.99

    def test_check_budget_after_spending(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=4.0)
        mgr._budgets["t1"].epsilon_spent = 2.0
        status = mgr.check_budget("t1")
        assert status["epsilon_remaining"] == pytest.approx(0.5)

    def test_check_budget_exhausted(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=1.0)
        mgr._budgets["t1"].epsilon_spent = 1.0
        status = mgr.check_budget("t1")
        assert status["has_budget"] is False

    def test_record_query(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=4.0)
        mgr.record_query("t1", sigma=2.0)
        entry = mgr._budgets["t1"]
        assert entry.num_queries == 1
        assert entry.epsilon_spent > 0

    def test_record_query_with_explicit_cost(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=4.0)
        mgr.record_query("t1", sigma=2.0, epsilon_cost=0.5)
        entry = mgr._budgets["t1"]
        assert entry.epsilon_spent == 0.5

    def test_record_query_auto_creates_tenant(self):
        mgr = PrivacyBudgetManager()
        mgr.record_query("auto-tenant", sigma=2.0)
        assert "auto-tenant" in mgr._budgets

    def test_global_privacy_spent(self):
        mgr = PrivacyBudgetManager()
        mgr.record_query("t1", sigma=2.0)
        spent = mgr.global_privacy_spent()
        assert "epsilon" in spent
        assert spent["epsilon"] > 0

    def test_global_privacy_spent_no_queries(self):
        mgr = PrivacyBudgetManager()
        spent = mgr.global_privacy_spent()
        assert spent["epsilon"] == 0.0

    def test_all_tenants(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1")
        mgr.set_tenant_budget("t2")
        assert set(mgr.all_tenants()) == {"t1", "t2"}

    def test_reset_all(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=4.0)
        mgr._budgets["t1"].epsilon_spent = 2.0
        mgr._budgets["t1"].num_queries = 5
        mgr.reset_all()
        entry = mgr._budgets["t1"]
        assert entry.epsilon_spent == 0.0
        assert entry.num_queries == 0

    def test_should_reset_daily(self):
        mgr = PrivacyBudgetManager()
        mgr.set_defaults(epsilon=4.0, delta=1e-6, period="daily")
        entry = BudgetEntry(epsilon=4.0, delta=1e-6, last_reset=0)  # Old timestamp
        assert mgr._should_reset(entry) is True

    def test_should_reset_recent(self):
        mgr = PrivacyBudgetManager()
        entry = BudgetEntry(epsilon=4.0, delta=1e-6)  # Recent last_reset
        assert mgr._should_reset(entry) is False

    def test_record_query_resets_if_needed(self):
        mgr = PrivacyBudgetManager()
        mgr.set_tenant_budget("t1", epsilon=4.0)
        mgr._budgets["t1"].last_reset = 0  # Force reset
        mgr._budgets["t1"].epsilon_spent = 3.0
        mgr.record_query("t1", sigma=2.0)
        # Should have been reset before recording
        assert mgr._budgets["t1"].epsilon_spent < 3.0


# ── clip_gradients ────────────────────────────────────────────────────────────


class TestClipGradients:
    @requires_torch
    def test_per_layer_clipping_under_norm(self):
        grads = [torch.ones(10) * 0.1]  # norm < 1.0
        clipped = clip_gradients(grads, max_norm=1.0, clip_per_layer=True)
        assert torch.allclose(clipped[0], grads[0])

    @requires_torch
    def test_per_layer_clipping_over_norm(self):
        grads = [torch.ones(10) * 10]  # norm > 1.0
        clipped = clip_gradients(grads, max_norm=1.0, clip_per_layer=True)
        assert clipped[0].norm().item() <= 1.0 + 1e-6

    @requires_torch
    def test_global_norm_clipping_under(self):
        grads = [torch.ones(10) * 0.1, torch.ones(5) * 0.1]
        clipped = clip_gradients(grads, max_norm=10.0, clip_per_layer=False)
        for c, g in zip(clipped, grads):
            assert torch.allclose(c, g)

    @requires_torch
    def test_global_norm_clipping_over(self):
        grads = [torch.ones(10) * 10, torch.ones(5) * 10]
        clipped = clip_gradients(grads, max_norm=1.0, clip_per_layer=False)
        # Global norm should be <= 1.0
        global_norm = math.sqrt(sum(g.norm().item() ** 2 for g in clipped))
        assert global_norm <= 1.0 + 1e-5

    @requires_torch
    def test_original_tensors_not_mutated(self):
        g = torch.ones(10) * 10
        original = g.clone()
        clip_gradients([g], max_norm=1.0)
        assert torch.allclose(g, original)

    @requires_torch
    def test_empty_grad_list(self):
        assert clip_gradients([], max_norm=1.0) == []


# ── dp_noise_injection ────────────────────────────────────────────────────────


class TestDpNoiseInjection:
    @requires_torch
    def test_adds_noise(self):
        grads = [torch.ones(10)]
        noisy = dp_noise_injection(grads, sigma=1.0)
        assert noisy[0].shape == (10,)
        assert not torch.allclose(noisy[0], grads[0], atol=0.5)

    @requires_torch
    def test_zero_sigma_no_noise(self):
        grads = [torch.ones(10)]
        noisy = dp_noise_injection(grads, sigma=0.0)
        assert torch.allclose(noisy[0], grads[0])

    @requires_torch
    def test_original_not_mutated(self):
        g = torch.ones(10)
        dp_noise_injection([g], sigma=1.0)
        assert torch.allclose(g, torch.ones(10))

    @requires_torch
    def test_seed_for_reproducibility(self):
        grads = [torch.zeros(10)]
        r1 = dp_noise_injection(grads, sigma=1.0, seed=42)
        r2 = dp_noise_injection(grads, sigma=1.0, seed=42)
        assert torch.allclose(r1[0], r2[0])

    @requires_torch
    def test_different_seeds_different_noise(self):
        grads = [torch.zeros(10)]
        r1 = dp_noise_injection(grads, sigma=1.0, seed=42)
        r2 = dp_noise_injection(grads, sigma=1.0, seed=99)
        assert not torch.allclose(r1[0], r2[0])

    @requires_torch
    def test_empty_grad_list(self):
        assert dp_noise_injection([], sigma=1.0) == []


# ── gumbel_noise_mechanism ────────────────────────────────────────────────────


class TestGumbelNoiseMechanism:
    @requires_torch
    def test_adds_noise_to_1d_logits(self):
        logits = torch.zeros(10)
        noisy = gumbel_noise_mechanism(logits, noise_scale=1.0)
        assert noisy.shape == (10,)
        assert not torch.allclose(noisy, logits)

    @requires_torch
    def test_adds_noise_to_2d_logits(self):
        logits = torch.zeros(2, 10)
        noisy = gumbel_noise_mechanism(logits, noise_scale=1.0)
        assert noisy.shape == (2, 10)

    @requires_torch
    def test_zero_noise_scale(self):
        logits = torch.randn(10)
        noisy = gumbel_noise_mechanism(logits, noise_scale=0.0)
        assert torch.allclose(noisy, logits)

    @requires_torch
    def test_seed_for_reproducibility(self):
        logits = torch.zeros(10)
        r1 = gumbel_noise_mechanism(logits, noise_scale=1.0, seed=42)
        r2 = gumbel_noise_mechanism(logits, noise_scale=1.0, seed=42)
        assert torch.allclose(r1, r2)

    @requires_torch
    def test_different_seed_different_noise(self):
        logits = torch.zeros(10)
        r1 = gumbel_noise_mechanism(logits, noise_scale=1.0, seed=42)
        r2 = gumbel_noise_mechanism(logits, noise_scale=1.0, seed=99)
        assert not torch.allclose(r1, r2)

    @requires_torch
    def test_original_not_mutated(self):
        logits = torch.zeros(10)
        original = logits.clone()
        gumbel_noise_mechanism(logits, noise_scale=1.0)
        assert torch.allclose(logits, original)


# ── DifferentialPrivacyInference ──────────────────────────────────────────────


class TestDPInferenceConstruction:
    def test_default_construction(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        assert dpi._config.epsilon == 4.0
        assert dpi._config.delta == 1e-6
        assert dpi._enforce_budget is True
        assert dpi._sigma > 0

    def test_custom_params(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(
            engine=engine,
            epsilon=8.0,
            delta=1e-5,
            max_grad_norm=0.5,
            noise_multiplier=0.3,
            mechanism="gumbel",
            gumbel_noise_scale=2.0,
            enforce_budget=False,
        )
        assert dpi._config.epsilon == 8.0
        assert dpi._config.target_mechanism == "gumbel"
        assert dpi._enforce_budget is False
        assert dpi._sigma == 0.5 * 0.3

    def test_budget_manager_property(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        assert isinstance(dpi.budget_manager, PrivacyBudgetManager)

    def test_sigma_property(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        assert dpi.sigma == dpi._config.sigma


class TestDPInferenceConfig:
    def test_set_epsilon(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        old_sigma = dpi.sigma
        dpi.set_epsilon(8.0)
        assert dpi._config.epsilon == 8.0
        assert dpi.sigma < old_sigma  # Larger epsilon -> less noise

    def test_set_delta(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        dpi.set_delta(1e-5)
        assert dpi._config.delta == 1e-5

    def test_set_noise_multiplier(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        dpi.set_noise_multiplier(0.5)
        assert dpi._config.noise_multiplier == 0.5
        assert dpi.sigma == 1.0 * 0.5

    def test_set_noise_multiplizer_zero(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, noise_multiplier=0.5)
        dpi.set_noise_multiplier(0.0)
        # When zero, sigma reverts to auto-computed (for the test we
        # just check it's set)
        assert dpi._config.noise_multiplier == 0.0


class TestDPInferenceTensorOps:
    @requires_torch
    def test_add_gaussian_noise(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=1.0)
        tensor = torch.ones(10)
        noisy = dpi.add_gaussian_noise(tensor)
        assert noisy.shape == (10,)
        assert not torch.allclose(noisy, tensor)
        # Original unchanged
        assert torch.allclose(tensor, torch.ones(10))

    @requires_torch
    def test_add_gaussian_noise_zero_sigma(self):
        """noise_multiplier=0.0 auto-derives sigma from (epsilon, delta).

        At epsilon=1000 this yields sigma ≈ 0.0053, so small noise IS added
        (not zero). The tensor is L2-clipped to max_grad_norm before noise,
        so compare against the clipped input at the sigma scale.
        """
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, noise_multiplier=0.0, epsilon=1000.0)
        tensor = torch.ones(10)
        noisy = dpi.add_gaussian_noise(tensor)
        # sigma auto-derived from epsilon=1000 is ≈0.0053: tiny noise IS
        # added on top of the unclipped input.
        assert torch.allclose(noisy, tensor, atol=0.05)

    @requires_torch
    def test_clip_tensor_under_norm(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, max_grad_norm=10.0)
        tensor = torch.ones(10)
        clipped = dpi.clip_tensor(tensor)
        assert torch.allclose(clipped, tensor)

    @requires_torch
    def test_clip_tensor_over_norm(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, max_grad_norm=1.0)
        tensor = torch.ones(10) * 5
        clipped = dpi.clip_tensor(tensor)
        assert clipped.norm().item() <= 1.0

    @requires_torch
    def test_apply_dp_to_logits_gaussian(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, mechanism="dp-sgd")
        logits = torch.randn(10)
        result = dpi.apply_dp_to_logits(logits)
        assert result.shape == (10,)

    @requires_torch
    def test_apply_dp_to_logits_gumbel(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, mechanism="gumbel")
        logits = torch.randn(10)
        result = dpi.apply_dp_to_logits(logits)
        assert result.shape == (10,)


class TestDPInferenceBudgetManagement:
    def test_set_tenant_budget(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        dpi.set_tenant_budget("acme", epsilon=8.0, delta=1e-5)
        entry = dpi._budget_manager.get_tenant_budget("acme")
        assert entry is not None
        assert entry.epsilon == 8.0

    def test_check_budget(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        status = dpi.check_budget("new-tenant")
        assert status["has_budget"] is True

    def test_global_privacy_spent(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine)
        spent = dpi.global_privacy_spent()
        assert "epsilon" in spent

    def test_budget_exhausted_raises(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=0.001)
        dpi._budget_manager.set_tenant_budget("t1", epsilon=0.001)
        # Spend the budget
        dpi._budget_manager._budgets["t1"].epsilon_spent = 0.001
        with pytest.raises(RuntimeError, match="budget exhausted"):
            dpi.generate("hello", user_id="t1")


class TestDPInferenceGenerate:
    """P0: DP generate must FAIL CLOSED (it cannot apply noise at this layer)
    rather than returning plaintext while charging the tenant's privacy budget."""

    def test_generate_raises_notimplemented_with_generate_stream(self):
        engine = _FakeEngine()
        engine.generate_stream = lambda prompt, **kw: iter(["Hello", " ", "world"])
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        with pytest.raises(NotImplementedError, match="privacy budget was charged"):
            dpi.generate("test prompt", user_id="default")

    def test_generate_raises_with_generate_only(self):
        engine = _FakeEngine()
        engine.generate = lambda prompt, **kw: iter(["a", "b"])
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        with pytest.raises(NotImplementedError, match="privacy budget was charged"):
            dpi.generate("test", user_id="default")

    def test_generate_raises_even_without_enforcement(self):
        engine = _FakeEngine()
        engine.generate_stream = lambda prompt, **kw: iter(["ok"])
        dpi = DifferentialPrivacyInference(
            engine, epsilon=0.001, enforce_budget=False,
        )
        dpi._budget_manager.set_tenant_budget("t1", epsilon=0.001)
        dpi._budget_manager._budgets["t1"].epsilon_spent = 0.001
        # DP cannot be applied regardless of budget enforcement — fail closed.
        with pytest.raises(NotImplementedError):
            dpi.generate("hello", user_id="t1")

    def test_generate_does_not_charge_budget_on_fail_closed(self):
        """The core guarantee: raising must NOT consume the tenant's budget."""
        engine = _FakeEngine()
        engine.generate_stream = lambda prompt, **kw: iter(["Hello", "world"])
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        dpi._budget_manager.set_tenant_budget("t1", epsilon=4.0)
        before = dpi._budget_manager.check_budget("t1")["epsilon_remaining"]
        with pytest.raises(NotImplementedError):
            dpi.generate("hello", user_id="t1")
        after = dpi._budget_manager.check_budget("t1")["epsilon_remaining"]
        assert after == before, "fail-closed raise must not spend the privacy budget"

    def test_generate_without_any_engine_method_raises_notimplemented(self):
        engine = _FakeEngine()  # No methods at all
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        with pytest.raises(NotImplementedError):
            dpi.generate("test", user_id="default")


class TestDPInferenceInternal:
    @requires_torch
    def test_dp_sample_with_temperature(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        logits = torch.randn(1, 100)
        token = dpi._dp_sample(logits, sigma=1.0, temperature=0.7)
        assert token.dim() == 1  # (1,) or scalar
        assert token.numel() == 1

    @requires_torch
    def test_dp_sample_zero_temperature(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        logits = torch.randn(1, 100)
        # With temp=0, should argmax
        token = dpi._dp_sample(logits, sigma=0.0, temperature=0.0)
        assert token.numel() == 1

    def test_estimate_epsilon_cost(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        cost = dpi._estimate_epsilon_cost(sigma=2.0, num_tokens=10)
        # per_token = 1 / (2 * 4) = 0.125, total = 1.25
        assert cost == pytest.approx(1.25)

    def test_estimate_epsilon_cost_zero_sigma(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        assert dpi._estimate_epsilon_cost(sigma=0.0, num_tokens=10) == 0.0

    def test_estimate_epsilon_cost_zero_tokens(self):
        engine = _FakeEngine()
        dpi = DifferentialPrivacyInference(engine, epsilon=4.0)
        assert dpi._estimate_epsilon_cost(sigma=2.0, num_tokens=0) == 0.0


# ── wrap_with_dp ──────────────────────────────────────────────────────────────


class TestWrapWithDP:
    def test_wraps_engine(self):
        engine = _FakeEngine()
        wrapped = wrap_with_dp(engine, epsilon=8.0, delta=1e-5)
        assert isinstance(wrapped, DifferentialPrivacyInference)
        assert wrapped._config.epsilon == 8.0
        assert wrapped._config.delta == 1e-5

    def test_wrap_with_custom_mechanism(self):
        engine = _FakeEngine()
        wrapped = wrap_with_dp(engine, mechanism="gumbel")
        assert wrapped._config.target_mechanism == "gumbel"
