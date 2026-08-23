"""Tests for FlashAttention integration using real PyTorch tensors (no GPU required).

Tests cover FlashAttentionWrapper construction, forward pass with various
configurations, input format auto-detection, and the model-level patching
function (which gracefully returns 0 when flash-attn is unavailable).
"""

from __future__ import annotations

import torch
import pytest

from distllm.dist.flash_attention import (
    FlashAttentionWrapper,
    apply_flash_attention_to_model,
)


# ---------------------------------------------------------------------------
# Construction, properties, and stats
# ---------------------------------------------------------------------------


class TestFlashAttentionWrapperConstruction:
    """FlashAttentionWrapper constructor and simple property access."""

    def test_default_construction(self) -> None:
        wrapper = FlashAttentionWrapper()
        assert wrapper.causal is True

    def test_non_causal(self) -> None:
        wrapper = FlashAttentionWrapper(causal=False)
        assert wrapper.causal is False

    def test_with_head_params(self) -> None:
        wrapper = FlashAttentionWrapper(num_heads=8, head_dim=64)
        assert wrapper._num_heads == 8
        assert wrapper._head_dim == 64

    def test_is_available_false_without_flash_attn(self) -> None:
        wrapper = FlashAttentionWrapper()
        assert wrapper.is_available is False

    def test_stats_default(self) -> None:
        wrapper = FlashAttentionWrapper()
        s = wrapper.stats()
        assert s == {
            "available": False,
            "causal": True,
            "backend": "sdpa_fallback",
        }

    def test_stats_non_causal(self) -> None:
        wrapper = FlashAttentionWrapper(causal=False)
        assert wrapper.stats()["causal"] is False
        assert wrapper.stats()["backend"] == "sdpa_fallback"

    def test_class_level_mask_cache_is_dict(self) -> None:
        assert isinstance(FlashAttentionWrapper._mask_cache, dict)
        assert len(FlashAttentionWrapper._mask_cache) == 0


# ---------------------------------------------------------------------------
# Forward pass -- standard shapes, parameters, and edge cases
# ---------------------------------------------------------------------------


class TestFlashAttentionWrapperForward:
    """Forward pass behaviour with real tensors (all use SDPA fallback).

    All shapes use S >= H so that _ensure_format leaves the [B, S, H, D]
    layout alone.
    """

    B = 2
    S = 8  # Must be >= H to avoid auto-transpose in _ensure_format
    H = 4
    D = 16

    def _inputs(
        self,
        B: int | None = None,
        S: int | None = None,
        H: int | None = None,
        D: int | None = None,
    ):
        B = B or self.B
        S = S or self.S
        H = H or self.H
        D = D or self.D
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        return q, k, v

    # -- simple forward shape checks --

    def test_basic(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_non_causal(self) -> None:
        wrapper = FlashAttentionWrapper(causal=False)
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_with_attention_mask(self) -> None:
        wrapper = FlashAttentionWrapper(causal=False)
        q, k, v = self._inputs()
        mask = torch.triu(
            torch.full((self.S, self.S), float("-inf")), diagonal=1
        )
        out = wrapper.forward(q, k, v, attention_mask=mask)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_with_dropout(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v, dropout_p=0.5)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_with_softmax_scale(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v, softmax_scale=0.125)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_one_batch(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs(B=1)
        out = wrapper.forward(q, k, v)
        assert out.shape == (1, self.S, self.H * self.D)

    # -- causal vs non-causal behaviour --

    def test_causal_and_non_causal_differ(self) -> None:
        wrapper_on = FlashAttentionWrapper(causal=True)
        wrapper_off = FlashAttentionWrapper(causal=False)
        q, k, v = self._inputs()
        out_on = wrapper_on.forward(q, k, v)
        out_off = wrapper_off.forward(q, k, v)
        assert not torch.allclose(out_on, out_off)

    # -- edge shapes --

    def test_single_head_single_sequence(self) -> None:
        wrapper = FlashAttentionWrapper()
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)
        v = torch.randn(1, 1, 1, 8)
        out = wrapper.forward(q, k, v)
        # _ensure_format: shape[1]=1, shape[2]=1, 1<1 is False => no transpose
        assert out.shape == (1, 1, 8)

    def test_large_batch_small_sequence(self) -> None:
        wrapper = FlashAttentionWrapper()
        # [B=16, S=4, H=2, D=32] -- S >= H, no auto-transpose
        q = torch.randn(16, 4, 2, 32)
        k = torch.randn(16, 4, 2, 32)
        v = torch.randn(16, 4, 2, 32)
        out = wrapper.forward(q, k, v)
        assert out.shape == (16, 4, 64)

    def test_many_heads(self) -> None:
        wrapper = FlashAttentionWrapper()
        # [B=2, S=32, H=4, D=8] -- S=32 >= H=4, no transpose
        q = torch.randn(2, 32, 4, 8)
        k = torch.randn(2, 32, 4, 8)
        v = torch.randn(2, 32, 4, 8)
        out = wrapper.forward(q, k, v)
        assert out.shape == (2, 32, 32)

    def test_cross_attention_same_heads(self) -> None:
        """Cross-attention with same num_heads but different seq lengths."""
        wrapper = FlashAttentionWrapper(causal=False)
        # S=8 >= H=4 for all tensors -> no auto-transpose
        q = torch.randn(2, 4, 4, 16)   # Q seq_len = 4
        k = torch.randn(2, 8, 4, 16)   # K seq_len = 8
        v = torch.randn(2, 8, 4, 16)   # V seq_len = 8
        out = wrapper.forward(q, k, v)
        assert out.shape == (2, 4, 64)

    # -- mask edge cases --

    def test_mask_with_causal_true(self) -> None:
        """attention_mask + causal=True => explicit mask path is taken."""
        wrapper = FlashAttentionWrapper(causal=True)
        q, k, v = self._inputs()
        mask = torch.triu(
            torch.full((self.S, self.S), float("-inf")), diagonal=1
        )
        out = wrapper.forward(q, k, v, attention_mask=mask)
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_none_mask_explicit(self) -> None:
        wrapper = FlashAttentionWrapper(causal=True)
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v, attention_mask=None)
        assert out.shape == (self.B, self.S, self.H * self.D)

    # -- determinism and gradients --

    def test_deterministic(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        torch.manual_seed(42)
        out1 = wrapper.forward(q.clone(), k.clone(), v.clone())
        torch.manual_seed(42)
        out2 = wrapper.forward(q.clone(), k.clone(), v.clone())
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self) -> None:
        wrapper = FlashAttentionWrapper()
        q = torch.randn(2, 4, 2, 8, requires_grad=True)
        k = torch.randn(2, 4, 2, 8, requires_grad=True)
        v = torch.randn(2, 4, 2, 8, requires_grad=True)
        out = wrapper.forward(q, k, v)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape

    def test_zero_dropout_identical(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        out1 = wrapper.forward(q, k, v, dropout_p=0.0)
        out2 = wrapper.forward(q, k, v, dropout_p=0.0)
        assert torch.allclose(out1, out2)

    def test_output_can_reshape_back(self) -> None:
        wrapper = FlashAttentionWrapper()
        q, k, v = self._inputs()
        out = wrapper.forward(q, k, v)
        reshaped = out.view(self.B, self.S, self.H, self.D)
        assert reshaped.shape == (self.B, self.S, self.H, self.D)


# ---------------------------------------------------------------------------
# Input format auto-detection
# ---------------------------------------------------------------------------


class TestInputFormatDetection:
    """The wrapper auto-detects [B, H, S, D] vs [B, S, H, D] layout."""

    def test_bshd_unchanged(self) -> None:
        """[B, S, H, D] passes through when S >= H."""
        wrapper = FlashAttentionWrapper()
        q = torch.randn(2, 8, 4, 16)
        k = torch.randn(2, 8, 4, 16)
        v = torch.randn(2, 8, 4, 16)
        out = wrapper.forward(q, k, v)
        assert out.shape == (2, 8, 64)

    def test_bhsd_transposed_when_params_match(self) -> None:
        """[B, H, S, D] with matching head_dim/num_heads is transposed."""
        wrapper = FlashAttentionWrapper(num_heads=4, head_dim=16)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = wrapper.forward(q, k, v)
        # After transpose: (2, 8, 4, 16) -> out (2, 8, 64)
        assert out.shape == (2, 8, 64)

    def test_auto_detect_transpose_when_h_gt_s(self) -> None:
        """When dim[1] < dim[2], treat as [B, H, S, D] and transpose."""
        wrapper = FlashAttentionWrapper()
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = wrapper.forward(q, k, v)
        assert out.shape == (2, 8, 64)

    def test_auto_detect_no_transpose_when_s_ge_h(self) -> None:
        """When dim[1] >= dim[2], treat as [B, S, H, D] and keep."""
        wrapper = FlashAttentionWrapper()
        q = torch.randn(2, 8, 4, 16)
        k = torch.randn(2, 8, 4, 16)
        v = torch.randn(2, 8, 4, 16)
        out = wrapper.forward(q, k, v)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# Model-level patching (always returns 0 without flash-attn)
# ---------------------------------------------------------------------------


class TestApplyFlashAttentionToModel:
    """apply_flash_attention_to_model gracefully returns 0 without flash-attn."""

    def test_returns_zero_for_linear(self) -> None:
        model = torch.nn.Linear(10, 10)
        assert apply_flash_attention_to_model(model) == 0

    def test_returns_zero_for_sequential_no_attention(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        assert apply_flash_attention_to_model(model) == 0

    def test_returns_zero_for_module_named_attention(self) -> None:
        class FakeAttention(torch.nn.Module):  # noqa: E302
            def __init__(self) -> None:
                super().__init__()
                self.dense = torch.nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = FakeAttention()
        model.__class__.__name__ = "FalconAttention"
        assert apply_flash_attention_to_model(model) == 0

    def test_returns_zero_for_model_with_projections(self) -> None:
        """Even with proper attention projections, returns 0 (no flash-attn)."""
        class MyAttention(torch.nn.Module):  # noqa: E302
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 64)
                self.k_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.o_proj = torch.nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = MyAttention()
        model.__class__.__name__ = "MyAttention"
        assert apply_flash_attention_to_model(model) == 0

    def test_returns_zero_for_nested_attention(self) -> None:
        """Attention inside a Sequential container."""
        class Attn(torch.nn.Module):  # noqa: E302
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = torch.nn.Linear(32, 32)
                self.k_proj = torch.nn.Linear(32, 32)
                self.v_proj = torch.nn.Linear(32, 32)
                self.o_proj = torch.nn.Linear(32, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            Attn(),
            torch.nn.Linear(32, 32),
        )
        assert apply_flash_attention_to_model(model) == 0

    def test_returns_zero_for_falcon_style_attention(self) -> None:
        """Falcon-style (query_key_value) attention also returns 0."""
        class FalconAttention(torch.nn.Module):  # noqa: E302
            def __init__(self) -> None:
                super().__init__()
                self.query_key_value = torch.nn.Linear(64, 192)
                self.dense = torch.nn.Linear(64, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = FalconAttention()
        assert apply_flash_attention_to_model(model) == 0
