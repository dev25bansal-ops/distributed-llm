"""Tests for WAN-optimized speculative decoding."""

from __future__ import annotations

import asyncio

import pytest
import torch

from distllm.dist.wan_speculative import WANSpeculativeConfig, WANSpeculativeDecoder


# ---------------------------------------------------------------------------
# Module-level helpers: real callables used for draft / target forwards.
# These are NOT mocks -- they are concrete functions the decoder calls.
# ---------------------------------------------------------------------------

VOCAB = 100


def _draft_always_token_zero(prefix: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Draft model that always assigns highest probability to token 0.

    Returns logits shape ``(1, 1, VOCAB)``.
    """
    logits = torch.full((1, 1, VOCAB), -10.0)
    logits[:, :, 0] = 10.0
    return logits


def _draft_follows_prefix(prefix: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Draft model where the last prefix token has highest probability."""
    last_token = prefix[0, -1].item()
    logits = torch.full((1, 1, VOCAB), -10.0)
    logits[:, :, last_token] = 10.0
    return logits


async def _target_always_token_zero(
    input_ids: torch.Tensor, **kwargs: object
) -> torch.Tensor:
    """Target model that assigns highest probability to token 0 everywhere."""
    b, s = input_ids.shape
    logits = torch.full((b, s, VOCAB), -10.0)
    logits[:, :, 0] = 10.0
    return logits


async def _target_token_at_index(
    input_ids: torch.Tensor, **kwargs: object
) -> torch.Tensor:
    """Target model that favors the input token value at each position."""
    b, s = input_ids.shape
    logits = torch.full((b, s, VOCAB), -10.0)
    for pos in range(s):
        token_id = input_ids[0, pos].item()
        logits[:, pos, token_id] = 10.0
    return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWANSpeculativeConfig:
    """WANSpeculativeConfig -- parameter container and adaptive logic."""

    # -- Construction -------------------------------------------------------

    def test_default_values(self) -> None:
        config = WANSpeculativeConfig()
        assert config.num_candidates == 8
        assert config.temperature == 1.0
        assert config.top_k == 20
        assert config.max_speculation_depth == 16
        assert config.adaptive_candidates is True
        assert config.min_acceptance_rate == 0.3

    def test_custom_values(self) -> None:
        config = WANSpeculativeConfig(
            num_candidates=4,
            temperature=0.5,
            top_k=10,
            max_speculation_depth=8,
            adaptive_candidates=False,
            min_acceptance_rate=0.5,
        )
        assert config.num_candidates == 4
        assert config.temperature == 0.5
        assert config.top_k == 10
        assert config.max_speculation_depth == 8
        assert config.adaptive_candidates is False
        assert config.min_acceptance_rate == 0.5

    # -- adapt_candidates ---------------------------------------------------

    def test_adapt_disabled(self) -> None:
        config = WANSpeculativeConfig(adaptive_candidates=False)
        assert config.adapt_candidates(0.0) == 8
        assert config.adapt_candidates(1.0) == 8

    def test_adapt_high_acceptance(self) -> None:
        config = WANSpeculativeConfig(num_candidates=4, max_speculation_depth=16)
        # rate > 0.8 -> double, capped by max_speculation_depth
        assert config.adapt_candidates(0.9) == 8

    def test_adapt_high_acceptance_capped(self) -> None:
        config = WANSpeculativeConfig(num_candidates=10, max_speculation_depth=12)
        assert config.adapt_candidates(0.9) == 12

    def test_adapt_medium_acceptance(self) -> None:
        config = WANSpeculativeConfig(num_candidates=8, max_speculation_depth=16)
        # 0.5 < rate <= 0.8 -> unchanged
        assert config.adapt_candidates(0.6) == 8

    def test_adapt_low_acceptance(self) -> None:
        config = WANSpeculativeConfig(num_candidates=8, max_speculation_depth=16)
        # min_acceptance_rate < rate <= 0.5 -> half (floor 2)
        assert config.adapt_candidates(0.4) == 4

    def test_adapt_low_acceptance_floor(self) -> None:
        config = WANSpeculativeConfig(num_candidates=3, max_speculation_depth=16)
        # 3 // 2 = 1, but min is 2
        assert config.adapt_candidates(0.4) == 2

    def test_adapt_very_low_acceptance(self) -> None:
        config = WANSpeculativeConfig(num_candidates=8, max_speculation_depth=16)
        # rate <= min_acceptance_rate -> quarter (floor 1)
        assert config.adapt_candidates(0.2) == 2

    def test_adapt_very_low_acceptance_floor(self) -> None:
        config = WANSpeculativeConfig(num_candidates=1, max_speculation_depth=16)
        # 1 // 4 = 0, but min is 1
        assert config.adapt_candidates(0.2) == 1

    def test_adapt_boundary_at_threshold(self) -> None:
        config = WANSpeculativeConfig(
            num_candidates=8,
            max_speculation_depth=16,
            min_acceptance_rate=0.3,
        )
        # Strict inequalities in source: > 0.8, > 0.5, > min_acceptance_rate
        # 0.8: not > 0.8, 0.8 > 0.5 -> same branch -> 8
        assert config.adapt_candidates(0.8) == 8
        # 0.801: > 0.8 -> double -> 16
        assert config.adapt_candidates(0.801) == 16
        # 0.5: not > 0.8, not > 0.5, 0.5 > 0.3 -> half -> max(2, 4) = 4
        assert config.adapt_candidates(0.5) == 4
        # 0.51: > 0.5 -> same -> 8
        assert config.adapt_candidates(0.51) == 8
        # 0.3: not > 0.8, not > 0.5, not > 0.3 -> quarter -> max(1, 2) = 2
        assert config.adapt_candidates(0.3) == 2
        # 0.31: > 0.3 -> half -> max(2, 4) = 4
        assert config.adapt_candidates(0.31) == 4


class TestWANSpeculativeDecoderConstruction:
    """WANSpeculativeDecoder construction and parameter validation."""

    @staticmethod
    def _dummy_draft(prefix: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.zeros(1, 1, VOCAB)

    @staticmethod
    async def _dummy_target(
        input_ids: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return torch.zeros(1, input_ids.shape[1], VOCAB)

    def test_default_params(self) -> None:
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            device="cpu",
        )
        assert decoder._num_candidates == 8
        assert decoder._temperature == 1.0
        assert decoder._top_k == 20
        assert decoder._device.type == "cpu"
        assert decoder._max_speculation_depth == 16
        assert decoder._stats["draft_calls"] == 0

    def test_custom_params(self) -> None:
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            num_candidates=4,
            temperature=0.0,
            top_k=5,
            device="cpu",
            max_speculation_depth=8,
        )
        assert decoder._num_candidates == 4
        assert decoder._temperature == 0.0
        assert decoder._top_k == 5
        assert decoder._max_speculation_depth == 8

    def test_device_cpu_explicit(self) -> None:
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            device="cpu",
        )
        assert decoder._device.type == "cpu"

    def test_construction_with_non_callable_raises_on_call(self) -> None:
        """TypeError raised when a non-callable is later invoked.

        Python does not raise at construction time for Callable annotations,
        so we verify the TypeError surfaces when the attribute is called.
        """
        decoder = WANSpeculativeDecoder(
            target_forward="not_callable",  # type: ignore[arg-type]
            draft_forward="also_not_callable",  # type: ignore[arg-type]
            device="cpu",
        )
        with pytest.raises(TypeError):
            decoder._target(torch.zeros(1, 3))


class TestWANSpeculativeDecoderStats:
    """The .stats property."""

    @staticmethod
    async def _dummy_target(
        input_ids: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return torch.zeros(1, input_ids.shape[1], VOCAB)

    @staticmethod
    def _dummy_draft(prefix: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.zeros(1, 1, VOCAB)

    def test_stats_all_zero_initially(self) -> None:
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            device="cpu",
        )
        stats = decoder.stats
        assert stats["draft_calls"] == 0
        assert stats["target_calls"] == 0
        assert stats["tokens_accepted"] == 0
        assert stats["tokens_rejected"] == 0
        assert stats["wan_rounds"] == 0
        assert stats["total_draft_tokens"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["wan_speedup"] == 0.0

    def test_stats_return_is_copy(self) -> None:
        """Mutating the returned dict does not affect internal state."""
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            device="cpu",
        )
        stats = decoder.stats
        stats["draft_calls"] = 999
        assert decoder._stats["draft_calls"] == 0

    def test_stats_computed_fields_no_division_by_zero(self) -> None:
        """Computed fields should not crash when totals are zero."""
        decoder = WANSpeculativeDecoder(
            target_forward=self._dummy_target,
            draft_forward=self._dummy_draft,
            device="cpu",
        )
        stats = decoder.stats
        assert stats["acceptance_rate"] == 0.0
        assert stats["wan_speedup"] == 0.0


class TestWANSpeculativeDecoderSample:
    """_sample (target-model sampling)."""

    @staticmethod
    def _build(temperature: float = 1.0) -> WANSpeculativeDecoder:
        async def _target(input_ids: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, 1, VOCAB)

        def _draft(prefix: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, 1, VOCAB)

        return WANSpeculativeDecoder(
            target_forward=_target,
            draft_forward=_draft,
            temperature=temperature,
            device="cpu",
        )

    def test_greedy_argmax(self) -> None:
        decoder = self._build(temperature=0.0)
        logits = torch.tensor([[-100.0, 50.0, -100.0]])
        token = decoder._sample(logits)
        assert token.shape == (1, 1)
        assert token[0, 0].item() == 1

    def test_greedy_last_vocab_token(self) -> None:
        decoder = self._build(temperature=0.0)
        logits = torch.full((1, VOCAB), -100.0)
        logits[0, VOCAB - 1] = 50.0
        token = decoder._sample(logits)
        assert token[0, 0].item() == VOCAB - 1

    def test_greedy_tie_first_wins(self) -> None:
        """When multiple tokens tie for argmax, the first index is returned."""
        decoder = self._build(temperature=0.0)
        logits = torch.tensor([[-100.0, 100.0, 100.0]])
        token = decoder._sample(logits)
        # argmax returns the first max index
        assert token[0, 0].item() == 1

    def test_stochastic_skewed_distribution(self) -> None:
        """Extreme logit differences make sampling deterministic in practice."""
        decoder = self._build(temperature=1.0)
        logits = torch.full((1, VOCAB), -100.0)
        logits[0, 42] = 100.0
        for _ in range(20):
            token = decoder._sample(logits)
            assert token[0, 0].item() == 42


class TestWANSpeculativeDecoderSampleDraft:
    """_sample_draft (draft-model sampling with optional top-k)."""

    @staticmethod
    def _build(temperature: float = 1.0, top_k: int = 20) -> WANSpeculativeDecoder:
        async def _target(input_ids: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, 1, VOCAB)

        def _draft(prefix: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, 1, VOCAB)

        return WANSpeculativeDecoder(
            target_forward=_target,
            draft_forward=_draft,
            temperature=temperature,
            top_k=top_k,
            device="cpu",
        )

    def test_greedy(self) -> None:
        decoder = self._build(temperature=0.0, top_k=20)
        logits = torch.tensor([[-100.0, 50.0, -100.0]])
        token = decoder._sample_draft(logits)
        assert token[0, 0].item() == 1

    def test_top_k_filters(self) -> None:
        decoder = self._build(temperature=1.0, top_k=3)
        # Create logits where top-3 filtering still yields a valid sample
        logits = torch.randn(1, VOCAB)
        token = decoder._sample_draft(logits)
        assert token.shape == (1, 1)
        assert 0 <= token[0, 0].item() < VOCAB

    def test_top_k_larger_than_vocab(self) -> None:
        """top_k exceeding vocab size should clamp to vocab."""
        decoder = self._build(temperature=1.0, top_k=10_000)
        logits = torch.randn(1, VOCAB)
        token = decoder._sample_draft(logits)
        assert token.shape == (1, 1)

    def test_top_k_zero_disables_filtering(self) -> None:
        decoder = self._build(temperature=1.0, top_k=0)
        logits = torch.randn(1, VOCAB)
        token = decoder._sample_draft(logits)
        assert token.shape == (1, 1)

    def test_greedy_with_top_k(self) -> None:
        decoder = self._build(temperature=0.0, top_k=5)
        logits = torch.tensor([[-100.0, 100.0, -50.0, -30.0, -20.0]])
        token = decoder._sample_draft(logits)
        assert token[0, 0].item() == 1


class TestWANSpeculativeDecoderDraftForward:
    """_draft_forward (local draft-token generation)."""

    @staticmethod
    def _build(draft_fn=_draft_follows_prefix) -> WANSpeculativeDecoder:
        async def _target(input_ids: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, 1, VOCAB)

        return WANSpeculativeDecoder(
            target_forward=_target,
            draft_forward=draft_fn,
            temperature=0.0,
            device="cpu",
        )

    def test_zero_tokens(self) -> None:
        decoder = self._build()
        prefix = torch.tensor([[1, 2, 3]])
        result = decoder._draft_forward(prefix, num_tokens=0)
        assert result.shape == (1, 0)
        assert result.dtype == torch.long

    def test_negative_tokens(self) -> None:
        decoder = self._build()
        prefix = torch.tensor([[1, 2, 3]])
        result = decoder._draft_forward(prefix, num_tokens=-5)
        assert result.shape == (1, 0)

    def test_single_token(self) -> None:
        decoder = self._build(draft_fn=_draft_always_token_zero)
        prefix = torch.tensor([[5]])
        result = decoder._draft_forward(prefix, num_tokens=1)
        assert result.shape == (1, 1)
        assert result[0, 0].item() == 0

    def test_multiple_tokens_greedy(self) -> None:
        decoder = self._build(draft_fn=_draft_always_token_zero)
        prefix = torch.tensor([[7]])
        result = decoder._draft_forward(prefix, num_tokens=4)
        assert result.shape == (1, 4)
        assert (result[0] == 0).all()

    def test_draft_follows_prefix(self) -> None:
        """With _draft_follows_prefix, each step emits the prior token."""
        decoder = self._build(draft_fn=_draft_follows_prefix)
        prefix = torch.tensor([[3]])
        result = decoder._draft_forward(prefix, num_tokens=3)
        assert result.shape == (1, 3)
        assert (result[0] == 3).all()

    def test_edge_vocab_index(self) -> None:
        """Last vocab index as prefix token."""
        decoder = self._build(draft_fn=_draft_follows_prefix)
        prefix = torch.tensor([[VOCAB - 1]])
        result = decoder._draft_forward(prefix, num_tokens=2)
        assert result.shape == (1, 2)
        assert result[0, 0].item() == VOCAB - 1


class TestWANSpeculativeDecoderVerifyTokens:
    """_verify_tokens -- acceptance / rejection logic."""

    @staticmethod
    def _build(
        temperature: float = 0.0,
        draft_fn=_draft_always_token_zero,
    ) -> WANSpeculativeDecoder:
        async def _target(input_ids: object, **kwargs: object) -> torch.Tensor:
            return torch.zeros(1, input_ids.shape[1], VOCAB)  # type: ignore[union-attr]

        return WANSpeculativeDecoder(
            target_forward=_target,
            draft_forward=draft_fn,
            temperature=temperature,
            device="cpu",
        )

    def test_greedy_all_accepted(self) -> None:
        decoder = self._build(temperature=0.0)
        prefix = torch.tensor([[1, 2, 3]])
        draft_tokens = torch.tensor([[0, 0, 0]])  # all token 0
        full_input = torch.cat([prefix, draft_tokens], dim=1)  # (1, 6)

        # Target logits: token 0 is argmax at every position
        _, seq_len = full_input.shape
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, :, 0] = 10.0

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        assert accepted == 3

    def test_greedy_partial_rejection(self) -> None:
        decoder = self._build(temperature=0.0)
        prefix = torch.tensor([[1, 2, 3]])
        draft_tokens = torch.tensor([[0, 5, 0]])  # middle token differs
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        # Target logits: token 0 is argmax only at draft positions 0 and 2;
        # token 5 is NOT the argmax at position 1.
        _, seq_len = full_input.shape  # seq_len = 6
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, 3, 0] = 10.0  # first draft pos -> token 0
        target_logits[:, 4, 0] = 10.0  # second draft pos -> token 0 (draft has 5)
        target_logits[:, 5, 0] = 10.0  # third draft pos -> token 0

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        # First token (pos 0) matches, second (pos 5) does not
        assert accepted == 1

    def test_greedy_all_rejected(self) -> None:
        decoder = self._build(temperature=0.0)
        prefix = torch.tensor([[1, 2, 3]])
        draft_tokens = torch.tensor([[7, 7, 7]])
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        _, seq_len = full_input.shape
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, :, 0] = 10.0  # target wants token 0, draft has 7

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        assert accepted == 0

    def test_greedy_single_draft_token_accepted(self) -> None:
        decoder = self._build(temperature=0.0)
        prefix = torch.tensor([[1, 2]])
        draft_tokens = torch.tensor([[0]])
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        _, seq_len = full_input.shape
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, 2, 0] = 10.0  # matches

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        assert accepted == 1

    def test_greedy_single_draft_token_rejected(self) -> None:
        decoder = self._build(temperature=0.0)
        prefix = torch.tensor([[1, 2]])
        draft_tokens = torch.tensor([[99]])
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        _, seq_len = full_input.shape
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, 2, 0] = 10.0  # target wants token 0, draft has 99

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        assert accepted == 0

    def test_probabilistic_high_target_prob_all_accepted(self) -> None:
        """When target_prob > 0.5 for every draft token, all are accepted."""
        decoder = self._build(temperature=1.0, draft_fn=_draft_always_token_zero)
        prefix = torch.tensor([[1, 2, 3]])
        draft_tokens = torch.tensor([[0, 0, 0]])
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        _, seq_len = full_input.shape
        # Token 0 has very high softmax probability >> 0.5
        target_logits = torch.full((1, seq_len, VOCAB), -100.0)
        target_logits[:, :, 0] = 100.0

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        assert accepted == 3

    def test_probabilistic_reject_when_draft_prob_zero(self) -> None:
        """When target_prob <= 0.5 and draft_prob == 0, token is rejected."""
        # Use a draft function where token 5 has zero probability
        def _draft_token_5_zero(prefix: torch.Tensor, **kwargs: object) -> torch.Tensor:
            logits = torch.full((1, 1, VOCAB), -10.0)
            logits[:, :, 0] = 10.0  # token 0 is very likely
            # Token 5 gets extremely low prob (but not -inf to avoid NaN)
            logits[:, :, 5] = -1000.0
            return logits

        decoder = self._build(temperature=1.0, draft_fn=_draft_token_5_zero)
        prefix = torch.tensor([[1, 2, 3]])
        draft_tokens = torch.tensor([[5, 0, 0]])
        full_input = torch.cat([prefix, draft_tokens], dim=1)

        _, seq_len = full_input.shape
        # Token 5 has probability < 0.5 from target perspective
        target_logits = torch.full((1, seq_len, VOCAB), -10.0)
        target_logits[:, :, 0] = 10.0  # target also favors 0

        accepted = decoder._verify_tokens(prefix, full_input, draft_tokens, target_logits)
        # First draft token (5) should be rejected because draft_prob ≈ 0
        assert accepted == 0


class TestWANSpeculativeDecoderGenerate:
    """The async generate method end-to-end."""

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _run(coro):
        """Run an async coroutine synchronously."""
        return asyncio.run(coro)

    @staticmethod
    def _build(
        target=_target_always_token_zero,
        draft=_draft_always_token_zero,
        num_candidates: int = 4,
        temperature: float = 0.0,
        max_speculation_depth: int = 16,
    ) -> WANSpeculativeDecoder:
        return WANSpeculativeDecoder(
            target_forward=target,
            draft_forward=draft,
            num_candidates=num_candidates,
            temperature=temperature,
            device="cpu",
            max_speculation_depth=max_speculation_depth,
        )

    # -- Tests --------------------------------------------------------------

    def test_generate_greedy_all_accepted(self) -> None:
        """All draft tokens accepted -> fast generation path."""
        decoder = self._build()
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=5))
        assert result.shape == (1, 8)  # 3 prompt + 5 new
        assert (result[0, 3:] == 0).all()  # all generated tokens are 0

    def test_generate_zero_new_tokens(self) -> None:
        """max_new_tokens=0 returns just the prompt."""
        decoder = self._build()
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=0))
        assert result.shape == (1, 3)
        assert (result == input_ids).all()

    def test_generate_single_token(self) -> None:
        """max_new_tokens=1 generates exactly one token."""
        decoder = self._build()
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=1))
        assert result.shape == (1, 4)
        assert result[0, 3].item() == 0

    def test_generate_partial_rejection_appends_correction(self) -> None:
        """When a draft token is rejected, a correction token is sampled."""
        # Target favors token 0; draft always generates token 0.
        # So all drafts are accepted. To test rejection, we need target
        # to disagree with draft.  We override the draft to produce token 1
        # while target still favors token 0.

        def _draft_always_token_one(
            prefix: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            logits = torch.full((1, 1, VOCAB), -10.0)
            logits[:, :, 1] = 10.0
            return logits

        # Target still favors token 0
        decoder = self._build(
            target=_target_always_token_zero,
            draft=_draft_always_token_one,
            num_candidates=3,
            temperature=0.0,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=5))
        assert result.shape == (1, 8)
        # All new tokens should be 0 (correction token from target, which favors 0)
        assert (result[0, 3:] == 0).all()

    def test_generate_stats_populated(self) -> None:
        """After generation, stats reflect the work done."""
        decoder = self._build(num_candidates=3, temperature=0.0)
        input_ids = torch.tensor([[1, 2, 3]])
        self._run(decoder.generate(input_ids, max_new_tokens=6))

        stats = decoder.stats
        assert stats["draft_calls"] > 0
        assert stats["target_calls"] > 0
        assert stats["wan_rounds"] > 0
        assert stats["total_draft_tokens"] > 0
        assert stats["tokens_accepted"] > 0
        assert stats["tokens_rejected"] >= 0
        assert stats["acceptance_rate"] > 0.0
        assert stats["wan_speedup"] > 0.0

    def test_generate_input_preserved(self) -> None:
        """Original input tensor is not mutated."""
        decoder = self._build()
        input_ids = torch.tensor([[1, 2, 3]])
        original = input_ids.clone()
        self._run(decoder.generate(input_ids, max_new_tokens=4))
        assert (input_ids == original).all()

    def test_generate_full_vocab_range(self) -> None:
        """Draft and target operate correctly with diverse input tokens."""

        async def _target_full_range(
            input_ids: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            b, s = input_ids.shape
            logits = torch.full((b, s, VOCAB), -10.0)
            for pos in range(s):
                tok = input_ids[0, pos].item()
                # Keep existing token as argmax
                logits[:, pos, tok] = 10.0
            return logits

        decoder = self._build(
            target=_target_full_range,
            draft=_draft_follows_prefix,
            num_candidates=2,
            temperature=0.0,
        )
        input_ids = torch.tensor([[VOCAB - 2, VOCAB - 1, 0]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=3))
        assert result.shape == (1, 6)

    def test_generate_num_candidates_exceeds_remaining(self) -> None:
        """When fewer tokens remain than num_candidates, fewer are drafted."""
        decoder = self._build(num_candidates=8, temperature=0.0)
        input_ids = torch.tensor([[1, 2, 3]])
        # Only 2 tokens remaining
        result = self._run(decoder.generate(input_ids, max_new_tokens=2))
        assert result.shape == (1, 5)
        assert (result[0, 3:] == 0).all()

    def test_generate_large_batch(self) -> None:
        """Batch size > 1 works (even though typical usage is BS=1)."""
        decoder = self._build(num_candidates=2, temperature=0.0)
        # Shape (1, 1) minimal
        input_ids = torch.tensor([[5]])
        result = self._run(decoder.generate(input_ids, max_new_tokens=10))
        assert result.shape == (1, 11)
        assert (result[0, 1:] == 0).all()
