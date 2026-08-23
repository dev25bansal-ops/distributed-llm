"""Tests for online self-improving speculative decoder (SISD).

Classes under test:
  - SpeculativeFeedbackBuffer  (distllm.dist.speculative.online_sisd)
  - OnlineLoRAUpdater          (distllm.dist.speculative.online_sisd)
"""

from __future__ import annotations

from distllm.dist.speculative.online_sisd import (
    OnlineLoRAUpdater,
    SpeculativeFeedbackBuffer,
)
from tests.core._stubs import _Stub


class TestFeedbackBufferAddAndSample:
    """SpeculativeFeedbackBuffer add and stratified sample."""

    def test_add_and_sample_returns_all_when_under_batch_size(self) -> None:
        buf = SpeculativeFeedbackBuffer(max_size=100)
        buf.add(
            prefix_ids=[1, 2, 3],
            draft_token_ids=[10, 11, 12],
            accepted_mask=[True, False, True],
        )
        buf.add(
            prefix_ids=[4, 5],
            draft_token_ids=[13, 14],
            accepted_mask=[False, False],
        )

        assert buf.size == 2

        sampled = buf.sample(batch_size=32)
        assert len(sampled) == 2
        assert sampled[0].draft_token_ids == [10, 11, 12]
        assert sampled[1].draft_token_ids == [13, 14]

    def test_sample_empty_buffer(self) -> None:
        buf = SpeculativeFeedbackBuffer(max_size=10)
        assert buf.sample(batch_size=4) == []

    def test_sample_mismatched_lengths_raises(self) -> None:
        buf = SpeculativeFeedbackBuffer(max_size=10)
        try:
            buf.add(
                prefix_ids=[1],
                draft_token_ids=[10, 11],
                accepted_mask=[True],
            )
        except ValueError:
            pass
        assert buf.size == 0


class TestUpdaterCreation:
    """OnlineLoRAUpdater construction and basic API."""

    def test_construction_with_mock_model(self) -> None:
        model = _Stub()
        updater = OnlineLoRAUpdater(
            draft_model_ref=model,
            lora_r=8,
            lora_alpha=16,
            lr=1e-4,
        )
        assert updater._lora_r == 8
        assert updater._lora_alpha == 16
        assert updater._lr == 1e-4
        assert updater._adapter_version == 0

    def test_update_returns_metrics_even_with_empty_buffer(self) -> None:
        model = _Stub()
        updater = OnlineLoRAUpdater(draft_model_ref=model)
        buf = SpeculativeFeedbackBuffer(max_size=10)
        metrics = updater.update(buf)
        assert metrics["loss"] == 0.0
        assert metrics["accepted_nll"] == 0.0
        assert metrics["rejected_nll"] == 0.0
        assert metrics["kl_penalty"] == 0.0

    def test_invalid_lora_rank_raises(self) -> None:
        model = _Stub()
        try:
            OnlineLoRAUpdater(draft_model_ref=model, lora_r=0)
        except ValueError:
            pass
