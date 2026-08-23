"""Tests for adaptive speculative decoding.

Covers:
- AcceptanceProfile statistics and EMA updates
- AdaptiveSpeculatorConfig defaults and customization
- AdaptiveSpeculator public API (init, getters, stats, recording, adaptation, generate)
- Edge cases: zero tokens, empty input, None prompt, batch > 1

Zero mocks: uses only real objects from the distllm.dist.adaptive_speculator module
and its dependencies.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from distllm.dist.adaptive_speculator import (
    AcceptanceProfile,
    AdaptiveSpeculator,
    AdaptiveSpeculatorConfig,
)
from distllm.dist.scheduling.classifier import WorkloadType


# ── Module-level helpers (real functions, never MagicMock) ────────────


async def _target_forward(input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    """Real async target forward: logits with token 0 always most likely.

    This is a deterministic function used across all AdaptiveSpeculator
    tests that need a forward pass.  It is never mocked.
    """
    batch, seq_len = input_ids.shape
    logits = torch.zeros(batch, seq_len, 3)
    logits[..., 0] = 1.0
    return logits


# ══════════════════════════════════════════════════════════════════════
# AcceptanceProfile
# ══════════════════════════════════════════════════════════════════════


class TestAcceptanceProfile:
    """AcceptanceProfile dataclass -- per-class acceptance statistics."""

    def test_default_initial_state(self) -> None:
        profile = AcceptanceProfile()
        assert profile.ema_acceptance == 0.0
        assert profile.observation_count == 0
        assert profile.total_speculated == 0
        assert profile.total_accepted == 0
        assert profile.acceptance_rate == 0.0

    def test_acceptance_rate_zero_denominator(self) -> None:
        """When nothing has been speculated the rate is 0.0, never NaN."""
        profile = AcceptanceProfile()
        assert profile.acceptance_rate == 0.0
        assert profile.acceptance_rate == pytest.approx(
            float(profile.total_accepted) / max(profile.total_speculated, 1)
        )

    def test_acceptance_rate_positive_ratio(self) -> None:
        profile = AcceptanceProfile(total_accepted=7, total_speculated=10)
        assert profile.acceptance_rate == pytest.approx(0.7)

    def test_acceptance_rate_exceeds_one(self) -> None:
        """Rate is NOT clamped -- caller is responsible for sensible values."""
        profile = AcceptanceProfile(total_accepted=15, total_speculated=10)
        assert profile.acceptance_rate == pytest.approx(1.5)

    def test_acceptance_rate_larger_denominator(self) -> None:
        profile = AcceptanceProfile(total_accepted=1, total_speculated=100)
        assert profile.acceptance_rate == pytest.approx(0.01)

    def test_update_first_observation_sets_ema(self) -> None:
        profile = AcceptanceProfile()
        profile.update(0.8)
        assert profile.ema_acceptance == pytest.approx(0.8)
        assert profile.observation_count == 1
        assert profile.total_speculated == 1

    def test_update_subsequent_applies_smoothing(self) -> None:
        profile = AcceptanceProfile()
        profile.update(1.0, alpha=0.5)
        assert profile.ema_acceptance == pytest.approx(1.0)
        profile.update(0.0, alpha=0.5)
        # ema = (1 - 0.5) * 1.0 + 0.5 * 0.0 = 0.5
        assert profile.ema_acceptance == pytest.approx(0.5)

    def test_update_default_alpha(self) -> None:
        """Default alpha (0.3) weights recent observation at 30%."""
        profile = AcceptanceProfile()
        profile.update(1.0)  # ema = 1.0 (first obs)
        profile.update(0.0)  # ema = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert profile.ema_acceptance == pytest.approx(0.7)

    def test_update_accumulates_observation_count(self) -> None:
        profile = AcceptanceProfile()
        for _ in range(10):
            profile.update(0.5, alpha=0.1)
        assert profile.observation_count == 10
        assert profile.total_speculated == 10

    def test_update_increments_total_speculated_by_one(self) -> None:
        """Each update() call increments total_speculated by exactly 1."""
        profile = AcceptanceProfile()
        profile.update(0.3)
        profile.update(0.6)
        assert profile.total_speculated == 2

    def test_update_negative_rate(self) -> None:
        """Negative rates are accepted (caller's responsibility)."""
        profile = AcceptanceProfile()
        profile.update(-0.2)
        assert profile.ema_acceptance == pytest.approx(-0.2)

    def test_update_rate_above_one(self) -> None:
        profile = AcceptanceProfile()
        profile.update(2.0)
        assert profile.ema_acceptance == pytest.approx(2.0)

    def test_update_with_alpha_zero(self) -> None:
        """Alpha = 0 means no smoothing -- EMA never changes after first obs."""
        profile = AcceptanceProfile()
        profile.update(1.0, alpha=0.0)
        profile.update(0.0, alpha=0.0)
        assert profile.ema_acceptance == pytest.approx(1.0)

    def test_update_with_alpha_one(self) -> None:
        """Alpha = 1 means EMA always equals the latest observation."""
        profile = AcceptanceProfile()
        profile.update(0.5, alpha=1.0)
        assert profile.ema_acceptance == pytest.approx(0.5)
        profile.update(0.9, alpha=1.0)
        assert profile.ema_acceptance == pytest.approx(0.9)


# ══════════════════════════════════════════════════════════════════════
# AdaptiveSpeculatorConfig
# ══════════════════════════════════════════════════════════════════════


class TestAdaptiveSpeculatorConfig:
    """AdaptiveSpeculatorConfig -- configuration dataclass."""

    def test_default_values(self) -> None:
        config = AdaptiveSpeculatorConfig()
        assert config.max_candidates == 16
        assert config.min_candidates == 1
        assert config.target_acceptance == 0.7
        assert config.low_acceptance_threshold == 0.3
        assert config.ema_alpha == 0.3
        assert config.profile_cooldown_s == 5.0
        assert config.warmup_observations == 5

    def test_custom_values(self) -> None:
        config = AdaptiveSpeculatorConfig(
            max_candidates=8,
            min_candidates=2,
            target_acceptance=0.8,
            low_acceptance_threshold=0.4,
            ema_alpha=0.2,
            profile_cooldown_s=10.0,
            warmup_observations=3,
        )
        assert config.max_candidates == 8
        assert config.min_candidates == 2
        assert config.target_acceptance == 0.8
        assert config.low_acceptance_threshold == 0.4
        assert config.ema_alpha == 0.2
        assert config.profile_cooldown_s == 10.0
        assert config.warmup_observations == 3

    def test_zero_cooldown_disables_throttle(self) -> None:
        config = AdaptiveSpeculatorConfig(profile_cooldown_s=0.0)
        assert config.profile_cooldown_s == 0.0

    def test_zero_warmup_allows_immediate_adaptation(self) -> None:
        config = AdaptiveSpeculatorConfig(warmup_observations=0)
        assert config.warmup_observations == 0

    def test_min_greater_than_max_possible(self) -> None:
        """The config does not validate min/max consistency."""
        config = AdaptiveSpeculatorConfig(min_candidates=10, max_candidates=5)
        assert config.min_candidates > config.max_candidates

    def test_all_boundary_values_zero(self) -> None:
        config = AdaptiveSpeculatorConfig(
            max_candidates=0,
            min_candidates=0,
            target_acceptance=0.0,
            low_acceptance_threshold=0.0,
            ema_alpha=0.0,
            profile_cooldown_s=0.0,
            warmup_observations=0,
        )
        assert config.max_candidates == 0
        assert config.min_candidates == 0
        assert config.low_acceptance_threshold == 0.0


# ══════════════════════════════════════════════════════════════════════
# AdaptiveSpeculator
# ══════════════════════════════════════════════════════════════════════


class TestAdaptiveSpeculator:
    """AdaptiveSpeculator -- main adaptive speculative decoding class."""

    # ── __init__ ─────────────────────────────────────────────────────

    def test_init_defaults(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert speculator._target_forward is _target_forward
        assert speculator._draft_bank is None
        assert isinstance(speculator._config, AdaptiveSpeculatorConfig)

    def test_init_custom_config(self) -> None:
        config = AdaptiveSpeculatorConfig(max_candidates=4, min_candidates=2)
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        assert speculator._config.max_candidates == 4
        assert speculator._config.min_candidates == 2

    def test_init_explicit_draft_bank_none(self) -> None:
        speculator = AdaptiveSpeculator(
            target_forward=_target_forward,
            draft_bank=None,
        )
        assert speculator._draft_bank is None

    def test_init_internal_collections_empty(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert len(speculator._profiles) == 0
        assert len(speculator._candidate_counts) == 0
        assert len(speculator._latency_records) == 0
        assert speculator._global_profile.observation_count == 0

    # ── get_acceptance_rate ──────────────────────────────────────────

    def test_get_acceptance_rate_global_before_any_request(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert speculator.get_acceptance_rate() == 0.0

    def test_get_acceptance_rate_specific_before_any_request(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert speculator.get_acceptance_rate(WorkloadType.CODE) == 0.0

    def test_get_acceptance_rate_after_one_record(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.85)
        assert speculator.get_acceptance_rate(WorkloadType.CODE) == pytest.approx(0.85)

    def test_get_acceptance_rate_global_different_from_specific(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.99)
        speculator._record_outcome("diverse", None, 0.01)
        code_rate = speculator.get_acceptance_rate(WorkloadType.CODE)
        diverse_rate = speculator.get_acceptance_rate(WorkloadType.DIVERSE)
        global_rate = speculator.get_acceptance_rate()
        assert code_rate == pytest.approx(0.99)
        assert diverse_rate == pytest.approx(0.01)
        assert global_rate != code_rate
        assert global_rate != diverse_rate
        # Global is blended from both
        assert 0.01 < global_rate < 0.99

    def test_get_acceptance_rate_none_returns_global(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.6)
        assert speculator.get_acceptance_rate(None) == pytest.approx(0.6)

    def test_get_acceptance_rate_unknown_workload(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("unknown", None, 0.5)
        assert speculator.get_acceptance_rate(WorkloadType.UNKNOWN) == pytest.approx(0.5)

    # ── get_candidate_count ──────────────────────────────────────────

    def test_get_candidate_count_unseen_workload_uses_min(self) -> None:
        """For an unseen workload the .get() fallback is min_candidates (default 1)."""
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert speculator.get_candidate_count() == 1

    def test_get_candidate_count_unseen_code(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        assert speculator.get_candidate_count(WorkloadType.CODE) == 1

    def test_get_candidate_count_respects_high_min(self) -> None:
        config = AdaptiveSpeculatorConfig(min_candidates=10, max_candidates=20)
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        assert speculator.get_candidate_count(WorkloadType.CODE) == 10

    def test_get_candidate_count_unseen_none_workload(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        count = speculator.get_candidate_count(None)
        assert count == 1

    def test_get_candidate_count_min_candidates_zero_unseen(self) -> None:
        config = AdaptiveSpeculatorConfig(min_candidates=0)
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        assert speculator.get_candidate_count(WorkloadType.CODE) == 0

    # ── get_stats ────────────────────────────────────────────────────

    def test_get_stats_before_any_request(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        stats = speculator.get_stats()
        assert stats == {
            "profiles": {},
            "global_acceptance": 0.0,
            "global_observations": 0,
            "avg_latency_ms": 0.0,
            "total_requests": 0,
        }

    def test_get_stats_after_one_record(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.8)
        stats = speculator.get_stats()
        assert "code" in stats["profiles"]
        assert stats["profiles"]["code"]["ema_acceptance"] == 0.8
        assert stats["profiles"]["code"]["observations"] == 1
        assert stats["global_observations"] == 1

    def test_get_stats_multiple_profiles(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.9)
        speculator._record_outcome("code", None, 0.8)
        speculator._record_outcome("diverse", None, 0.3)
        stats = speculator.get_stats()
        assert stats["global_observations"] == 3
        assert len(stats["profiles"]) == 2
        assert stats["profiles"]["code"]["observations"] == 2
        assert stats["profiles"]["diverse"]["observations"] == 1

    def test_get_stats_after_generate_updates_latency(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[1, 2, 3]])
        asyncio.run(speculator.generate(input_ids, max_new_tokens=0))
        stats = speculator.get_stats()
        assert stats["total_requests"] == 1
        assert stats["avg_latency_ms"] >= 0

    # ── _record_outcome (internal, tested via get_acceptance_rate) ───

    def test_record_outcome_creates_per_class_profile(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("instruction", None, 0.4)
        assert "instruction" in speculator._profiles
        assert speculator._profiles["instruction"].observation_count == 1

    def test_record_outcome_updates_global_profile(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.6)
        assert speculator._global_profile.observation_count == 1
        assert speculator._global_profile.ema_acceptance == pytest.approx(0.6)

    def test_record_outcome_multiple_updates_same_class(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.5)
        speculator._record_outcome("code", None, 0.7)
        assert speculator._profiles["code"].observation_count == 2
        assert speculator._profiles["code"].ema_acceptance != 0.5  # smoothed

    def test_record_outcome_draft_id_is_accepted(self) -> None:
        """draft_id parameter is accepted but currently ignored -- verify no crash."""
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", "some-draft-cluster", 0.9)
        assert speculator._profiles["code"].observation_count == 1

    def test_record_outcome_different_classes_independent(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        speculator._record_outcome("code", None, 0.9)
        speculator._record_outcome("diverse", None, 0.1)
        speculator._record_outcome("repetitive", None, 0.5)
        assert len(speculator._profiles) == 3

    def test_record_outcome_high_observation_count(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        for _ in range(1000):
            speculator._record_outcome("code", None, 0.5)
        assert speculator._profiles["code"].observation_count == 1000
        assert speculator._global_profile.observation_count == 1000

    # ── _adapt_candidates (internal, verified via get_candidate_count) ─

    def _spec_with_fast_adapt(self, **kw: float | int) -> AdaptiveSpeculator:
        """Build a speculator with cooldown and warmup disabled for testing."""
        config = AdaptiveSpeculatorConfig(
            profile_cooldown_s=0,
            warmup_observations=0,
            **kw,  # type: ignore
        )
        return AdaptiveSpeculator(target_forward=_target_forward, config=config)

    def test_adapt_high_acceptance_doubles_candidates(self) -> None:
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.95)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.95)
        # ema=0.95 >= 0.7 target -> double
        expected = min(initial * 2, 16)
        assert speculator._candidate_counts["code"] == expected

    def test_adapt_low_acceptance_halves_candidates(self) -> None:
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.1)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.1)
        # ema=0.1 <= 0.3 low_threshold -> halve
        expected = max(initial // 2, 1)
        assert speculator._candidate_counts["code"] == expected

    def test_adapt_at_max_boundary_does_not_exceed(self) -> None:
        speculator = self._spec_with_fast_adapt(max_candidates=4)
        speculator._candidate_counts["code"] = 4
        speculator._record_outcome("code", None, 0.99)
        speculator._adapt_candidates("code", 0.99)
        assert speculator._candidate_counts["code"] == 4

    def test_adapt_at_min_boundary_does_not_go_below(self) -> None:
        speculator = self._spec_with_fast_adapt(min_candidates=2)
        speculator._candidate_counts["code"] = 2
        speculator._record_outcome("code", None, 0.01)
        speculator._adapt_candidates("code", 0.01)
        assert speculator._candidate_counts["code"] == 2

    def test_adapt_moderate_above_ema_increments_by_one(self) -> None:
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.5)  # ema=0.5 in moderate range
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.6)  # rate > ema -> +1
        assert speculator._candidate_counts["code"] == initial + 1

    def test_adapt_moderate_below_ema_decrements_by_one(self) -> None:
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.5)  # ema=0.5 in moderate range
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.4)  # rate < ema -> -1
        assert speculator._candidate_counts["code"] == initial - 1

    def test_adapt_moderate_equal_to_ema_decrements(self) -> None:
        """When rate equals ema, the 'else' branch treats it as below-or-equal."""
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.5)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.5)  # rate == ema -> -1
        assert speculator._candidate_counts["code"] == initial - 1

    def test_adapt_cooldown_prevents_change(self) -> None:
        import time
        config = AdaptiveSpeculatorConfig(
            profile_cooldown_s=100.0,  # long cooldown
            warmup_observations=0,
        )
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        # Set last adaptation to now so cooldown is active.
        speculator._last_adapt["code"] = time.time()
        speculator._record_outcome("code", None, 0.95)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.95)
        assert speculator._candidate_counts["code"] == initial

    def test_adapt_warmup_prevents_change(self) -> None:
        config = AdaptiveSpeculatorConfig(
            profile_cooldown_s=0,
            warmup_observations=5,
        )
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        speculator._record_outcome("code", None, 0.95)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.95)
        assert speculator._candidate_counts["code"] == initial

    def test_adapt_warmup_exactly_met_allows_change(self) -> None:
        config = AdaptiveSpeculatorConfig(
            profile_cooldown_s=0,
            warmup_observations=3,
        )
        speculator = AdaptiveSpeculator(target_forward=_target_forward, config=config)
        for _ in range(3):
            speculator._record_outcome("code", None, 0.95)
        initial = speculator._candidate_counts["code"]
        speculator._adapt_candidates("code", 0.95)
        assert speculator._candidate_counts["code"] != initial

    def test_adapt_different_workloads_independent(self) -> None:
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.95)
        speculator._record_outcome("diverse", None, 0.1)
        code_initial = speculator._candidate_counts["code"]
        diverse_initial = speculator._candidate_counts["diverse"]
        speculator._adapt_candidates("code", 0.95)
        speculator._adapt_candidates("diverse", 0.1)
        assert speculator._candidate_counts["code"] > code_initial
        assert speculator._candidate_counts["diverse"] < diverse_initial

    def test_adapt_unseen_workload_does_not_crash(self) -> None:
        """Adapting a workload with no records just uses defaults."""
        speculator = self._spec_with_fast_adapt()
        speculator._adapt_candidates("unseen", 0.5)
        # Should not raise -- default ema is 0.0, moderate range,
        # rate > ema -> +1
        assert speculator._candidate_counts["unseen"] > 0

    def test_adapt_only_logs_change_no_side_effects(self) -> None:
        """Verifies _adapt_candidates doesn't corrupt other state."""
        speculator = self._spec_with_fast_adapt()
        speculator._record_outcome("code", None, 0.5)
        stats_before = speculator.get_stats()
        speculator._adapt_candidates("code", 0.8)
        stats_after = speculator.get_stats()
        assert stats_after["global_observations"] == stats_before["global_observations"]
        assert stats_after["total_requests"] == stats_before["total_requests"]

    # ── generate (async) ─────────────────────────────────────────────

    def test_generate_returns_tensor(self) -> None:
        """generate with max_new_tokens=0 returns a copy of input_ids.

        This exercises the full async path without triggering the
        draft forwarder (since remaining=0 exits the loop immediately).
        """
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[1, 2, 3]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text="hello world")
        )
        assert isinstance(result, torch.Tensor)

    def test_generate_no_new_tokens_returns_input_copy(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[5, 10, 15]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text="test prompt")
        )
        assert result.shape == input_ids.shape
        assert torch.equal(result, input_ids)

    def test_generate_no_prompt_text_uses_unknown_workload(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[1, 2, 3]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text=None)
        )
        assert result.shape == input_ids.shape

    def test_generate_empty_input_ids(self) -> None:
        """Empty sequence (no tokens) does not crash."""
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        empty = torch.empty(1, 0, dtype=torch.long)
        result = asyncio.run(
            speculator.generate(empty, max_new_tokens=0, prompt_text="test")
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == empty.shape

    def test_generate_batch_size_two(self) -> None:
        """Batch dimension > 1 is accepted."""
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text="test")
        )
        assert result.shape == input_ids.shape

    def test_generate_single_token_input(self) -> None:
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[42]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text="hello")
        )
        assert torch.equal(result, input_ids)

    def test_generate_large_vocab_forward_no_error(self) -> None:
        """Target forward with a realistically sized vocabulary."""
        async def large_vocab_target(input_ids: torch.Tensor, **kw):
            batch, seq_len = input_ids.shape
            return torch.zeros(batch, seq_len, 32000)

        speculator = AdaptiveSpeculator(target_forward=large_vocab_target)
        input_ids = torch.tensor([[1, 2, 3]])
        result = asyncio.run(
            speculator.generate(input_ids, max_new_tokens=0, prompt_text="test")
        )
        assert result.shape == input_ids.shape

    def test_generate_preserves_input_values(self) -> None:
        """The input tensor values are unchanged after generate."""
        speculator = AdaptiveSpeculator(target_forward=_target_forward)
        input_ids = torch.tensor([[7, 8, 9]])
        original = input_ids.clone()
        asyncio.run(speculator.generate(input_ids, max_new_tokens=0))
        assert torch.equal(input_ids, original)
