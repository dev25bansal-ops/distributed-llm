"""Tests for CompressedSpeculativeDecoder -- KV cache compression with verification.

Covers:
- Construction with callable target, KV cache, and verifier
- generate with no verifier (compressed fallback only)
- stats property tracks acceptance/re-run rates
- Verifier-only initialization
- Trainer integration via CompressedSpeculativeDecoder constructor

No MagicMock -- real stubs for forward pass and KV cache.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_csd_mod = load_module("distllm/core/compressed_speculative.py")
CompressedSpeculativeDecoder = _csd_mod.CompressedSpeculativeDecoder
LightweightVerifier = _csd_mod.LightweightVerifier
CompressionVerifierTrainer = _csd_mod.CompressionVerifierTrainer


class _StubKV:
    """Minimal KV cache stub."""

    def __init__(self: Any) -> None:
        self._compressed = False

    def compress(self: Any, method: str) -> None:
        self._compressed = True


def _forward_fn(input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Stub forward returning logits and hidden states."""
    batch, seq = input_ids.shape
    vocab = 100
    logits = torch.randn(batch, seq, vocab)
    hidden = torch.randn(batch, seq, 64)
    return logits, hidden


class TestCompressedSpeculativeDecoderConstruction:
    """Construction and initial state."""

    def test_minimal_construction(self) -> None:
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
        )
        assert decoder._target is _forward_fn
        assert decoder._verifier is None
        assert decoder._trainer is None
        assert decoder._re_run_threshold == 0.3
        assert decoder._max_re_runs == 3

    def test_construction_with_verifier(self) -> None:
        verifier = LightweightVerifier(hidden_size=64, num_heads=4, head_dim=16, intermediate_size=128, device="cpu")
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
            verifier=verifier,
        )
        assert decoder._verifier is verifier

    def test_construction_with_trainer(self) -> None:
        verifier = LightweightVerifier(hidden_size=64, num_heads=4, head_dim=16, intermediate_size=128, device="cpu")
        trainer = CompressionVerifierTrainer(verifier=verifier, lr=1e-4)
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
            verifier=verifier,
            trainer=trainer,
        )
        assert decoder._trainer is trainer

    def test_initial_stats(self) -> None:
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
        )
        stats = decoder.stats
        assert stats["compressed_calls"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["re_run_rate"] == 0.0


class TestCompressedSpeculativeDecoderGenerate:
    """Generate with and without verification."""

    def test_generate_without_verifier(self) -> None:
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = decoder.generate(input_ids, max_new_tokens=5)
        assert output.shape[0] == 1
        assert output.shape[1] == 3 + 5

    def test_generate_increments_stats(self) -> None:
        decoder = CompressedSpeculativeDecoder(
            target_forward=_forward_fn,
            kv_cache=_StubKV(),
        )
        input_ids = torch.tensor([[1]], dtype=torch.long)
        decoder.generate(input_ids, max_new_tokens=3)
        stats = decoder.stats
        assert stats["compressed_calls"] == 3

    def test_verifier_cpu_construction(self) -> None:
        """LightweightVerifier can be constructed on CPU with small dims."""
        verifier = LightweightVerifier(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            intermediate_size=128,
            device="cpu",
        )
        assert verifier._device.type == "cpu"
        # Run a forward pass to verify shapes
        hidden = torch.randn(1, 1, 64)
        logits = torch.randn(1, 1, 100)
        out = verifier(hidden, logits)
        assert out.shape == (1, 1, 1)
