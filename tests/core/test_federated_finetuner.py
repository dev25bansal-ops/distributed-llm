"""Tests for FederatedFineTuner -- distributed LoRA training across P2P nodes.

Covers:
- Construction with defaults
- Peer management (add/remove)
- Gradient clipping for differential privacy
- Noise addition for differential privacy
- Gradient averaging (FedAvg, FedProx)
- train_round execution
- Full run() loop
- Stats tracking
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_fft = load_module("distllm/core/federated_finetuner.py")
FederatedFineTuner = _fft.FederatedFineTuner


# We need torch for most tests; skip if unavailable
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class _FakeFn:
    """Deterministic callable stub that records every invocation.

    Parameters
    ----------
    return_value :
        Value to return on each call (or None).
    side_effect :
        Exception to raise on each call, or a callable invoked with args.
    """

    def __init__(self, return_value: Any = None, side_effect: Any = None):
        self._return_value = return_value
        self._side_effect = side_effect
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self._side_effect is not None:
            if isinstance(self._side_effect, Exception):
                raise self._side_effect
            return self._side_effect(*args, **kwargs)
        return self._return_value


def _make_tensor_grads(*shapes) -> list["torch.Tensor"]:
    """Create list of gradient tensors for testing."""
    return [torch.randn(*shape) for shape in shapes]


@pytest.fixture
def finetuner():
    """Basic FederatedFineTuner with no callbacks."""
    return FederatedFineTuner(node_id="node-0")


@pytest.fixture
def finetuner_with_callbacks():
    """FederatedFineTuner with stubbed broadcast/receive/apply."""
    return FederatedFineTuner(
        node_id="node-0",
        gossip_broadcast=_FakeFn(),
        gossip_receive=_FakeFn(),
        apply_gradients=_FakeFn(),
    )


# ======================================================================
# Construction and defaults
# ======================================================================


class TestConstruction:
    def test_defaults(self, finetuner):
        assert finetuner._node_id == "node-0"
        assert finetuner._local_steps == 100
        assert finetuner._num_rounds == 10
        assert finetuner._lr == 1e-4
        assert finetuner._algorithm == "fedavg"
        # A-C2: honest DP defaults.
        assert finetuner._dp_mode == "clip_only"
        assert finetuner._dp_epsilon == 1.0
        assert finetuner._dp_delta == 1e-5
        assert finetuner._dp_max_grad_norm == 1.0
        assert finetuner._dp_noise_multiplier == 0.0
        assert finetuner.stats["dp_mode"] == "clip_only"
        assert finetuner.stats["dp_cumulative_epsilon"] == 0.0
        assert finetuner.stats["dp_sigma"] is None
        assert finetuner._round == 0
        assert finetuner._peers == set()

    def test_fedprox_config(self):
        fft = FederatedFineTuner(
            node_id="n1", algorithm="fedprox", fedprox_mu=0.1,
        )
        assert fft._algorithm == "fedprox"
        assert fft._fedprox_mu == 0.1

    def test_custom_values(self):
        fft = FederatedFineTuner(
            node_id="n1",
            local_steps=50,
            num_rounds=5,
            learning_rate=1e-3,
            dp_epsilon=0.5,
            dp_noise_multiplier=1.2,
        )
        assert fft._local_steps == 50
        assert fft._num_rounds == 5
        assert fft._lr == 1e-3
        assert fft._dp_epsilon == 0.5
        assert fft._dp_noise_multiplier == 1.2


# ======================================================================
# Peer management
# ======================================================================


class TestPeerManagement:
    def test_add_peer(self, finetuner):
        finetuner.add_peer("node-1")
        assert "node-1" in finetuner._peers

    def test_remove_peer(self, finetuner):
        finetuner.add_peer("node-1")
        finetuner.remove_peer("node-1")
        assert "node-1" not in finetuner._peers

    def test_remove_nonexistent_no_error(self, finetuner):
        finetuner.remove_peer("nonexistent")  # should not raise


# ======================================================================
# Gradient clipping
# ======================================================================


class TestGradientClipping:
    def test_clip_below_threshold_noop(self, finetuner):
        grads = [torch.tensor([0.1, 0.2])]
        clipped = finetuner._clip_gradients(grads)
        assert torch.allclose(clipped[0], grads[0])

    def test_clip_above_threshold(self, finetuner):
        grads = [torch.tensor([10.0, 0.0])]
        finetuner._dp_max_grad_norm = 1.0
        clipped = finetuner._clip_gradients(grads)
        norm = torch.sqrt(sum(g.norm() ** 2 for g in clipped))
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_clip_counts_stats(self, finetuner):
        grads = [torch.tensor([100.0])]
        finetuner._dp_max_grad_norm = 1.0
        finetuner._clip_gradients(grads)
        assert finetuner._stats["dp_clips"] == 1

    def test_clip_not_incremented_if_no_clip(self, finetuner):
        grads = [torch.tensor([0.1, 0.2])]
        finetuner._clip_gradients(grads)
        assert finetuner._stats["dp_clips"] == 0


# ======================================================================
# Noise addition
# ======================================================================


class TestNoiseAddition:
    def test_noise_added_when_multiplier_set(self, finetuner):
        finetuner._dp_mode = "enabled"
        finetuner._dp_noise_multiplier = 0.5
        grads = [torch.zeros(10)]
        noisy = finetuner._add_dp_noise(grads)
        assert not torch.allclose(noisy[0], grads[0])
        assert finetuner._stats["dp_noise_added"] is True

    def test_no_noise_when_multiplier_zero_and_clip_only(self, finetuner):
        """Default mode is clip_only: no noise, no privacy claim (A-C2)."""
        finetuner._dp_mode = "clip_only"
        finetuner._dp_epsilon = float("inf")  # silence constructor warning path
        grads = [torch.zeros(10)]
        noisy = finetuner._add_dp_noise(grads)
        assert torch.allclose(noisy[0], grads[0])
        assert finetuner._stats["dp_noise_added"] is False

    def test_zero_multiplier_enabled_derives_sigma_from_budget(self, finetuner):
        """In enabled mode, multiplier 0 auto-derives calibrated sigma."""
        import math as _math

        finetuner._dp_mode = "enabled"
        finetuner._dp_epsilon = 1.0
        finetuner._dp_delta = 1e-5
        finetuner._num_rounds = 10
        finetuner._per_round_epsilon = 1.0 / 10
        finetuner._dp_max_grad_norm = 1.0

        grads = [torch.zeros(100)]
        noisy = finetuner._add_dp_noise(grads)
        # sigma = max_grad_norm * sqrt(2 ln(1.25/delta)) / eps_round,
        # with eps_round = total_eps / num_rounds = 0.1 here.
        expected_sigma = _math.sqrt(2 * _math.log(1.25 / 1e-5)) / 0.1
        # Empirical std of the added noise over a large tensor should be
        # close to sigma.
        empirical_std = (noisy[0] - grads[0]).std().item()
        assert abs(empirical_std - expected_sigma) < 0.2 * expected_sigma
        assert finetuner._stats["dp_noise_added"] is True
        assert finetuner._stats["dp_sigma"] == pytest.approx(expected_sigma)

    def test_disabled_when_epsilon_infinite_even_in_enabled_mode(
        self, finetuner
    ):
        finetuner._dp_mode = "enabled"
        finetuner._dp_epsilon = float("inf")
        grads = [torch.zeros(10)]
        noisy = finetuner._add_dp_noise(grads)
        assert torch.allclose(noisy[0], grads[0])
        assert finetuner._stats["dp_noise_added"] is False

    def test_enabled_output_differs_across_runs(self):
        """Statistical check: calibrated noise makes runs differ (A-C2)."""
        applied_sets = []
        for _ in range(3):
            seen = []
            fft = FederatedFineTuner(
                node_id="n1",
                local_steps=1,
                num_rounds=1,
                dp_mode="enabled",
                dp_epsilon=8.0,
                dp_delta=1e-5,
                dp_max_grad_norm=1.0,
                apply_gradients=lambda g, lr: seen.append(g[0].clone()),
            )
            fft.train_round(lambda steps: [torch.zeros(64)])
            applied_sets.append(seen[0])
        assert not torch.allclose(applied_sets[0], applied_sets[1])
        assert not torch.allclose(applied_sets[1], applied_sets[2])

    def test_noise_scale_reasonable(self, finetuner):
        finetuner._dp_mode = "enabled"
        finetuner._dp_noise_multiplier = 0.1
        grads = [torch.zeros(1000)]
        noisy = finetuner._add_dp_noise(grads)
        # Noise should have roughly zero mean
        assert abs(noisy[0].mean().item()) < 1.0


# ======================================================================
# Gradient averaging
# ======================================================================


class TestGradientAveraging:
    def test_single_source_returns_local(self, finetuner):
        grads = [torch.tensor([1.0, 2.0])]
        result = finetuner._average_gradients(grads)
        assert len(result) == 1
        assert torch.allclose(result[0], grads[0])

    def test_average_multiple_sources(self, finetuner):
        g1 = [torch.tensor([1.0, 2.0])]
        g2 = [torch.tensor([3.0, 4.0])]
        finetuner._received_grads["peer-1"] = g2
        result = finetuner._average_gradients(g1)
        expected = [(g1[0] + g2[0]) / 2]
        assert torch.allclose(result[0], expected[0])

    def test_average_ignores_mismatched_lengths(self, finetuner):
        g1 = [torch.tensor([1.0, 2.0])]
        g2 = [torch.tensor([3.0])]  # wrong length
        finetuner._received_grads["peer-bad"] = g2
        result = finetuner._average_gradients(g1)
        # Should ignore peer-bad and just return local grads
        assert torch.allclose(result[0], g1[0])


# ======================================================================
# FedProx
# ======================================================================


class TestFedProx:
    def test_fedprox_term_applied(self):
        global_params = [torch.tensor([0.5, 0.5])]
        local_params = [torch.tensor([1.0, 2.0])]
        fft = FederatedFineTuner(
            node_id="n1",
            algorithm="fedprox",
            fedprox_mu=1.0,
            lora_adapter=lambda: local_params,
            global_model_params=global_params,
        )
        grads = [torch.tensor([2.0, 3.0])]
        result = fft._apply_fedprox_term(grads)
        # proximal = mu * (w_local - w_global) = (0.5, 1.5)
        # result = grad + proximal = (2.5, 4.5)
        assert torch.allclose(result[0], torch.tensor([2.5, 4.5]))

    def test_fedprox_term_noop_when_mu_zero(self):
        fft = FederatedFineTuner(
            node_id="n1", algorithm="fedprox", fedprox_mu=0.0,
        )
        grads = [torch.tensor([1.0])]
        result = fft._apply_fedprox_term(grads)
        assert torch.allclose(result[0], grads[0])

    def test_fedprox_term_noop_when_no_global(self):
        fft = FederatedFineTuner(
            node_id="n1", algorithm="fedprox", fedprox_mu=1.0,
            global_model_params=None,
        )
        grads = [torch.tensor([1.0])]
        result = fft._apply_fedprox_term(grads)
        assert torch.allclose(result[0], grads[0])

    def test_fedprox_term_index_boundary(self):
        global_params = [torch.tensor([0.0]), torch.tensor([0.0])]
        fft = FederatedFineTuner(
            node_id="n1",
            algorithm="fedprox",
            fedprox_mu=1.0,
            global_model_params=global_params,
        )
        # More gradients than global params
        grads = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        result = fft._apply_fedprox_term(grads)
        # First two get proximal term, third passes through
        assert torch.allclose(result[2], grads[2])

    def test_set_global_model(self):
        params = [torch.tensor([1.0, 2.0])]
        fft = FederatedFineTuner(node_id="n1")
        fft.set_global_model(params)
        assert fft._global_model_params is not None
        assert len(fft._global_model_params) == 1
        assert torch.allclose(fft._global_model_params[0], params[0])


# ======================================================================
# train_round
# ======================================================================


class TestTrainRound:
    def test_round_increments(self, finetuner_with_callbacks):
        fft = finetuner_with_callbacks
        local_train_fn = _FakeFn(return_value=[torch.tensor([1.0, 2.0])])

        metrics = fft.train_round(local_train_fn)
        assert metrics["round"] == 1
        assert fft._round == 1

        metrics2 = fft.train_round(local_train_fn)
        assert metrics2["round"] == 2

    def test_round_calls_local_train_fn(self, finetuner_with_callbacks):
        fft = finetuner_with_callbacks
        local_train_fn = _FakeFn(return_value=[torch.tensor([1.0])])
        fft.train_round(local_train_fn)
        assert local_train_fn.call_count == 1

    def test_round_broadcasts_to_peers(self):
        broadcast = _FakeFn()
        fft = FederatedFineTuner(
            node_id="n1",
            gossip_broadcast=broadcast,
            apply_gradients=_FakeFn(),
        )
        fft.add_peer("peer-1")
        fft.add_peer("peer-2")
        local_fn = _FakeFn(return_value=[torch.tensor([1.0])])
        fft.train_round(local_fn)
        assert broadcast.call_count == 2

    def test_round_broadcast_failure_handled(self):
        broadcast = _FakeFn(side_effect=RuntimeError("network error"))
        fft = FederatedFineTuner(
            node_id="n1", gossip_broadcast=broadcast,
        )
        fft.add_peer("peer-1")
        local_fn = _FakeFn(return_value=[torch.tensor([1.0])])
        # Should not raise
        metrics = fft.train_round(local_fn)
        assert metrics["round"] == 1

    def test_round_applies_gradients(self):
        apply_fn = _FakeFn()
        fft = FederatedFineTuner(
            node_id="n1", apply_gradients=apply_fn,
        )
        local_fn = _FakeFn(return_value=[torch.tensor([2.0])])
        fft.train_round(local_fn)
        assert apply_fn.call_count == 1
        args, _ = apply_fn.calls[0]
        assert len(args) == 2  # merged grads, lr

    def test_round_applies_dp(self):
        fft = FederatedFineTuner(
            node_id="n1",
            dp_epsilon=0.5,
            dp_max_grad_norm=1.0,
            apply_gradients=_FakeFn(),
        )
        local_fn = _FakeFn(return_value=[torch.tensor([10.0])])  # will be clipped
        fft.train_round(local_fn)
        assert fft._stats["dp_clips"] >= 1

    def test_round_no_dp_when_epsilon_infinite(self):
        fft = FederatedFineTuner(
            node_id="n1",
            dp_epsilon=float("inf"),
            apply_gradients=_FakeFn(),
        )
        local_fn = _FakeFn(return_value=[torch.tensor([10.0])])
        fft.train_round(local_fn)
        assert fft._stats["dp_clips"] == 0


# ======================================================================
# Full run
# ======================================================================


class TestRun:
    def test_run_completes_all_rounds(self):
        fft = FederatedFineTuner(node_id="n1", num_rounds=3, apply_gradients=_FakeFn())
        local_fn = _FakeFn(return_value=[torch.tensor([1.0])])
        final_stats = fft.run(local_fn)
        assert final_stats["rounds_completed"] == 3
        assert local_fn.call_count == 3

    def test_run_with_zero_rounds(self):
        fft = FederatedFineTuner(node_id="n1", num_rounds=0, apply_gradients=_FakeFn())
        local_fn = _FakeFn(return_value=[torch.tensor([1.0])])
        final_stats = fft.run(local_fn)
        assert final_stats["rounds_completed"] == 0


# ======================================================================
# Stats
# ======================================================================


class TestStats:
    def test_initial_stats(self, finetuner):
        s = finetuner.stats
        assert s["rounds_completed"] == 0
        assert s["total_local_steps"] == 0
        assert s["peers_contacted"] == 0
        assert s["dp_clips"] == 0
        assert s["dp_noise_added"] is False
        assert s["algorithm"] == "fedavg"

    def test_stats_immutable(self, finetuner):
        s1 = finetuner.stats
        s1["rounds_completed"] = 99
        s2 = finetuner.stats
        assert s2["rounds_completed"] == 0  # not mutated


# ======================================================================
# A-C2: honest DP contract (mode gating, warnings, budget accounting)
# ======================================================================


class _WarningSink:
    """Loguru sink that captures WARNING+ messages."""

    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message):
        if message.record["level"].name in ("WARNING", "ERROR", "CRITICAL"):
            self.messages.append(message.record["message"])


class TestHonestDPContract:
    """dp_mode='clip_only' must never imply a privacy guarantee."""

    def test_clip_only_with_epsilon_warns_loudly(self):
        from loguru import logger

        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            FederatedFineTuner(node_id="n1", dp_epsilon=1.0)
        finally:
            logger.remove(handler_id)
        assert any(
            "no" in m.lower() and "guarantee" in m.lower() for m in sink.messages
        ), f"default config must warn that no DP guarantee holds, got: {sink.messages}"

    def test_clip_only_with_infinite_epsilon_no_warning(self):
        from loguru import logger

        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            FederatedFineTuner(
                node_id="n1", dp_epsilon=float("inf")
            )
        finally:
            logger.remove(handler_id)
        # Declared non-private: honest, no warning needed.
        assert sink.messages == []

    def test_enabled_mode_no_warning(self):
        from loguru import logger

        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            FederatedFineTuner(
                node_id="n1", dp_mode="enabled", dp_epsilon=1.0
            )
        finally:
            logger.remove(handler_id)
        assert sink.messages == []

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="dp_mode"):
            FederatedFineTuner(node_id="n1", dp_mode="on_please")

    def test_invalid_delta_rejected(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="dp_delta"):
                FederatedFineTuner(node_id="n1", dp_delta=bad)

    def test_negative_multiplier_rejected(self):
        with pytest.raises(ValueError, match="dp_noise_multiplier"):
            FederatedFineTuner(node_id="n1", dp_noise_multiplier=-1.0)

    def test_explicit_positive_multiplier_implies_enabled(self):
        fft = FederatedFineTuner(
            node_id="n1",
            dp_noise_multiplier=1.0,
            dp_epsilon=8.0,
        )
        assert fft._dp_mode == "enabled"

    def test_default_path_applies_clipping_but_no_noise(self):
        """The shipped default clips and does NOT claim to add noise."""
        fft = FederatedFineTuner(
            node_id="n1",
            local_steps=1,
            num_rounds=1,
            apply_gradients=_FakeFn(),
        )
        assert fft._dp_mode == "clip_only"
        local_fn = _FakeFn(return_value=[torch.tensor([10.0])])
        fft.train_round(local_fn)
        assert fft._stats["dp_clips"] >= 1
        assert fft._stats["dp_noise_added"] is False
        assert fft._stats["dp_cumulative_epsilon"] == 0.0

    def test_default_path_is_deterministic_across_runs(self):
        """Clip-only default adds no randomness (convergence preserved)."""
        results = []
        for _ in range(3):
            seen = []
            fft = FederatedFineTuner(
                node_id="n1",
                local_steps=1,
                num_rounds=1,
                apply_gradients=lambda g, lr: seen.append(g[0].clone()),
            )
            fft.train_round(lambda steps: [torch.tensor([4.0])])
            results.append(seen[0])
        assert torch.allclose(results[0], results[1])
        assert torch.allclose(results[1], results[2])


class TestBudgetAccounting:
    """Cumulative epsilon accounting reflects reality (A-C2/A-C3)."""

    def test_per_round_budget_split(self):
        fft = FederatedFineTuner(
            node_id="n1",
            num_rounds=5,
            dp_mode="enabled",
            dp_epsilon=2.0,
            dp_delta=1e-5,
            apply_gradients=_FakeFn(),
        )
        assert fft._per_round_epsilon == pytest.approx(0.4)

    def test_auto_derived_sigma_scales_with_round_count(self):
        sigmas = {}
        for rounds in (2, 10):
            fft = FederatedFineTuner(
                node_id=f"n{rounds}",
                num_rounds=rounds,
                dp_mode="enabled",
                dp_epsilon=1.0,
                dp_delta=1e-5,
                apply_gradients=_FakeFn(),
            )
            fft.train_round(lambda steps: [torch.zeros(32)])
            sigmas[rounds] = fft.stats["dp_sigma"]
        # Same total budget over more rounds -> larger per-round sigma.
        assert sigmas[10] > sigmas[2]

    def test_cumulative_epsilon_composition_within_budget(self):
        """N rounds of auto-derived noise charge at most the full budget."""
        import math as _math

        total_eps = 2.0
        rounds = 10
        ft = FederatedFineTuner(
            node_id="n1",
            local_steps=1,
            num_rounds=rounds,
            dp_mode="enabled",
            dp_epsilon=total_eps,
            dp_delta=1e-5,
            dp_max_grad_norm=1.0,
            apply_gradients=_FakeFn(),
        )

        def fake_train(steps):
            return [torch.zeros(2, 2) for _ in range(2)]

        stats = ft.run(fake_train)
        used = stats["dp_cumulative_epsilon"]
        assert used <= total_eps + 1e-9, (
            f"cumulative epsilon {used} exceeded budget {total_eps}"
        )
        assert used > 0.0
        assert _math.isclose(
            used, total_eps, rel_tol=1e-9
        ), f"expected naive composition {total_eps}, got {used}"
        assert stats["dp_noise_added"] is True
        assert stats["dp_sigma"] is not None

    def test_run_stops_when_budget_exhausted(self):
        """Explicit-multiplier cost above the budget share stops the loop."""
        from loguru import logger

        calls = {"train": 0}

        def train_fn(steps):
            calls["train"] += 1
            return [torch.zeros(4)]

        # Budget share is 1.0/4 = 0.25 per round, but sigma=1.0 Gaussian
        # queries cost ~5.3 each -- run() must stop after the first round.
        fft = FederatedFineTuner(
            node_id="n1",
            local_steps=1,
            num_rounds=6,
            dp_mode="enabled",
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_max_grad_norm=1.0,
            dp_noise_multiplier=1.0,
            apply_gradients=_FakeFn(),
        )
        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            stats = fft.run(train_fn)
        finally:
            logger.remove(handler_id)

        assert stats["rounds_completed"] == 1
        assert stats["dp_cumulative_epsilon"] > 1.0
        assert any("budget" in m.lower() for m in sink.messages), (
            f"early stop must warn loudly, got: {sink.messages}"
        )

    def test_train_round_raises_once_budget_spent(self):
        from distllm.core.federated_finetuner import DPBudgetExhausted

        fft = FederatedFineTuner(
            node_id="n1",
            local_steps=1,
            num_rounds=1,  # single round consumes the entire budget
            dp_mode="enabled",
            dp_epsilon=0.05,
            dp_delta=1e-5,
            apply_gradients=_FakeFn(),
        )
        fft.train_round(lambda steps: [torch.zeros(4)])
        assert fft.stats["dp_cumulative_epsilon"] == pytest.approx(0.05)
        with pytest.raises(DPBudgetExhausted, match="budget exhausted"):
            fft.train_round(lambda steps: [torch.zeros(4)])

    def test_budget_check_noop_when_disabled_or_infinite(self):
        fft = FederatedFineTuner(
            node_id="n1", dp_mode="enabled", dp_epsilon=float("inf")
        )
        fft._check_budget()  # must not raise
        fft2 = FederatedFineTuner(node_id="n1")  # clip_only
        fft2._stats["dp_cumulative_epsilon"] = 99.0
        fft2._check_budget()  # must not raise

    def test_rdp_accountant_used_for_explicit_multiplier(self):
        """Explicit multiplier path measures cost with RDP accountant."""
        fft2 = FederatedFineTuner(
            node_id="n1",
            local_steps=1,
            num_rounds=1,
            dp_mode="enabled",
            dp_epsilon=100.0,
            dp_delta=1e-5,
            dp_max_grad_norm=1.0,
            dp_noise_multiplier=1.0,
            apply_gradients=_FakeFn(),
        )
        fft2.train_round(lambda steps: [torch.zeros(8)])
        spent = fft2.stats["dp_cumulative_epsilon"]
        # sigma=1.0, delta=1e-5 single Gaussian query: known bound ~5.3
        # (matches the demo output's first-round figure).
        assert 4.0 < spent < 7.0, f"unexpected epsilon {spent}"
