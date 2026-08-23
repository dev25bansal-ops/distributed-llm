"""Tests for distllm.dist.scheduling.profiler module.

Covers the full public API: get_memory_per_sequence, estimate_max_batch,
and the re-exported group_by_length utility.
Deterministic -- no GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import pytest

from distllm.dist.scheduling.profiler import (
    estimate_max_batch,
    get_memory_per_sequence,
    group_by_length,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeSequence:
    """Minimal stand-in for objects with a total_len attribute."""

    def __init__(self, total_len: int) -> None:
        self.total_len = total_len

    def __repr__(self) -> str:
        return f"FakeSeq({self.total_len})"


# ── get_memory_per_sequence ──────────────────────────────────────────────────


class TestGetMemoryPerSequence:
    """get_memory_per_sequence: KV-cache memory estimation per sequence."""

    def test_defaults_with_minimal_model_info(self) -> None:
        """All keys missing => uses built-in defaults (hidden_size=768 ...)."""
        mem = get_memory_per_sequence({}, seq_len=256)
        # hidden=768, layers=12, heads=12, kv_heads=12, hidden_per_head=64
        # 2 * 12 * 12 * 64 * 256 * 2 = 9,437,184
        assert mem == 2 * 12 * 12 * (768 // 12) * 256 * 2
        assert mem == 9_437_184

    def test_custom_values(self) -> None:
        model_info = {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA / MQA
        }
        mem = get_memory_per_sequence(model_info, seq_len=1024, dtype_bytes=4)
        # hidden_per_head = 4096 // 32 = 128
        # 2 * 32 * 8 * 128 * 1024 * 4 = 268,435,456
        assert mem == 268_435_456

    def test_zero_num_heads_returns_zero(self) -> None:
        model_info = {"num_attention_heads": 0}
        assert get_memory_per_sequence(model_info, seq_len=512) == 0

    def test_zero_seq_len(self) -> None:
        mem = get_memory_per_sequence({}, seq_len=0)
        assert mem == 0

    def test_num_kv_heads_defaults_to_num_attention_heads(self) -> None:
        """When num_key_value_heads is absent, it falls back to num_attention_heads."""
        model_info = {
            "hidden_size": 1024,
            "num_layers": 4,
            "num_attention_heads": 16,
        }
        mem = get_memory_per_sequence(model_info, seq_len=128)
        # no kv_heads key -> num_kv_heads = num_heads = 16
        # hidden_per_head = 1024 // 16 = 64
        # 2 * 4 * 16 * 64 * 128 * 2 = 2,097,152
        assert mem == 2_097_152

    def test_gqa_smaller_kv_heads(self) -> None:
        """GQA config: fewer KV heads than query heads reduces memory."""
        gqa = {"hidden_size": 1024, "num_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 4}
        mha = {"hidden_size": 1024, "num_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 16}
        mem_gqa = get_memory_per_sequence(gqa, seq_len=128)
        mem_mha = get_memory_per_sequence(mha, seq_len=128)
        assert mem_gqa < mem_mha
        assert mem_gqa * 4 == mem_mha  # 4x fewer kv heads => 4x less memory

    def test_dtype_bytes_scales_linearly(self) -> None:
        mem_fp16 = get_memory_per_sequence({}, seq_len=256, dtype_bytes=2)
        mem_fp32 = get_memory_per_sequence({}, seq_len=256, dtype_bytes=4)
        assert mem_fp32 == 2 * mem_fp16

    def test_seq_len_scales_linearly(self) -> None:
        mem_128 = get_memory_per_sequence({}, seq_len=128)
        mem_256 = get_memory_per_sequence({}, seq_len=256)
        assert mem_256 == 2 * mem_128

    def test_num_layers_scales_linearly(self) -> None:
        model_4 = {"num_layers": 4}
        model_8 = {"num_layers": 8}
        mem_4 = get_memory_per_sequence(model_4, seq_len=256)
        mem_8 = get_memory_per_sequence(model_8, seq_len=256)
        assert mem_8 == 2 * mem_4

    def test_very_large_values_no_overflow(self) -> None:
        """Large model + long sequence should produce a large but correct int."""
        model_info = {
            "hidden_size": 16384,
            "num_layers": 128,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
        }
        mem = get_memory_per_sequence(model_info, seq_len=16384, dtype_bytes=4)
        # hidden_per_head = 16384 // 128 = 128
        # 2 * 128 * 128 * 128 * 16384 * 4 = 2 * 128^4 * 4
        assert isinstance(mem, int)
        assert mem > 0
        # Sanity: ~34 billion bytes ~= 34 GB for a single sequence
        expected = 2 * 128 * 128 * (16384 // 128) * 16384 * 4
        assert mem == expected


# ── estimate_max_batch ──────────────────────────────────────────────────────


class TestEstimateMaxBatch:
    """estimate_max_batch: optimal batch-size / token-count estimation."""

    def test_default_model(self) -> None:
        """With an empty model_info dict (defaults) and 16 GB device memory."""
        batch, tokens = estimate_max_batch({}, device_memory_bytes=16 * 1024**3)
        assert isinstance(batch, int)
        assert isinstance(tokens, int)
        assert 1 <= batch <= 128
        assert tokens >= 512

    def test_zero_mem_per_sequence_returns_fallback(self) -> None:
        """When num_attention_heads=0 => mem_per_seq=0 => fallback (32, 4096)."""
        model_info = {"num_attention_heads": 0}
        batch, tokens = estimate_max_batch(model_info, device_memory_bytes=999)
        assert batch == 32
        assert tokens == 4096

    def test_very_small_memory(self) -> None:
        """Tiny device memory => at least 1 sequence possible, min 2 batch."""
        model_info = {"hidden_size": 768, "num_layers": 1, "num_attention_heads": 1}
        # mem_per_seq = 2 * 1 * 1 * (768//1) * 256 * 2 = 786,432
        # usable = 100 * 0.6 = 60 => max_seqs = 60 // 786432 = 0 => max(1, 0) = 1
        # max_batch = min(1, 128) = 1, then max(1, 2) = 2
        batch, tokens = estimate_max_batch(model_info, device_memory_bytes=100)
        assert batch == 2
        assert tokens == 512  # max(2*256, 512) = 512

    def test_large_memory_capped_at_128(self) -> None:
        """Large device memory should saturate at max_batch_size=128."""
        model_info = {"hidden_size": 768, "num_layers": 1, "num_attention_heads": 1}
        # mem_per_seq = 2 * 1 * 1 * 768 * 256 * 2 = 786,432
        # usable = 1TB * 0.6 = 600GB => max_seqs >> 128 => capped at 128
        batch, tokens = estimate_max_batch(model_info, device_memory_bytes=1024**4)
        assert batch == 128
        assert tokens == 128 * 256

    def test_safety_factor_zero(self) -> None:
        """safety_factor=0 => usable_memory=0 => max_seqs=0 => max(1,0)=1 => batch=2 cap."""
        batch, tokens = estimate_max_batch({}, device_memory_bytes=16 * 1024**3, safety_factor=0.0)
        assert batch == 2
        assert tokens >= 512

    def test_safety_factor_one(self) -> None:
        """safety_factor=1 => usable=full device memory => larger batch."""
        batch_low, _ = estimate_max_batch({}, device_memory_bytes=16 * 1024**3, safety_factor=0.1)
        batch_high, _ = estimate_max_batch({}, device_memory_bytes=16 * 1024**3, safety_factor=1.0)
        assert batch_high >= batch_low

    def test_target_latency_accepted_but_not_used(self) -> None:
        """target_latency_ms is accepted as a parameter (currently unused)."""
        # Should not raise even though the param is not consumed
        estimate_max_batch({}, device_memory_bytes=8 * 1024**3, target_latency_ms=50.0)

    def test_returns_tuple_of_ints(self) -> None:
        result = estimate_max_batch({}, device_memory_bytes=8 * 1024**3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_exact_boundary_usable_memory(self) -> None:
        """When usable_memory exactly equals mem_per_seq => max_seqs = 1 => batch capped at 2."""
        model_info = {"hidden_size": 768, "num_layers": 1, "num_attention_heads": 1}
        mem_per_seq = get_memory_per_sequence(model_info, seq_len=256)
        # usable must be exactly mem_per_seq
        device = int(mem_per_seq / 0.6)  # solve: mem_per_seq = device * 0.6
        batch, tokens = estimate_max_batch(model_info, device_memory_bytes=device)
        assert batch == 2
        assert tokens >= 512

    def test_different_avg_seq_len_scaling(self) -> None:
        """Larger avg_seq_len (via different defaults) => more memory per seq => smaller batch."""
        # We can't inject avg_seq_len, so we use a model with larger hidden size
        # to increase mem_per_seq => smaller batch for a fixed memory.
        small = {"hidden_size": 768, "num_layers": 1, "num_attention_heads": 1}
        large = {"hidden_size": 4096, "num_layers": 1, "num_attention_heads": 1}
        batch_small, _ = estimate_max_batch(small, device_memory_bytes=256 * 1024**2)
        batch_large, _ = estimate_max_batch(large, device_memory_bytes=256 * 1024**2)
        assert batch_large <= batch_small


# ── group_by_length (re-export) ─────────────────────────────────────────────


class TestGroupByLength:
    """group_by_length re-exported from distllm.utils.scheduling via profiler."""

    def test_empty_list(self) -> None:
        result = group_by_length([])
        assert result == {0: [], 1: [], 2: [], 3: []}

    def test_single_sequence(self) -> None:
        seqs = [_FakeSequence(100)]
        result = group_by_length(seqs, num_buckets=4)
        # All go to bucket 0 since min==max
        assert result[0] == seqs
        for i in range(1, 4):
            assert result[i] == []

    def test_custom_num_buckets(self) -> None:
        seqs = [_FakeSequence(10), _FakeSequence(1000)]
        result = group_by_length(seqs, num_buckets=8)
        assert sum(len(v) for v in result.values()) == 2
        assert len([k for k, v in result.items() if v]) == 2

    def test_all_same_length(self) -> None:
        """When all lengths are equal, everything goes into bucket 0."""
        seqs = [_FakeSequence(42) for _ in range(5)]
        result = group_by_length(seqs, num_buckets=4)
        assert result[0] == seqs
        for i in range(1, 4):
            assert result[i] == []

    def test_all_zero_length(self) -> None:
        """Zero-length sequences are handled (total_len=0 => log(1) clamping)."""
        seqs = [_FakeSequence(0), _FakeSequence(0)]
        result = group_by_length(seqs, num_buckets=4)
        assert sum(len(v) for v in result.values()) == 2

    def test_bucket_distribution_is_exhaustive(self) -> None:
        """Every sequence ends up in exactly one bucket."""
        seqs = [_FakeSequence(i * 10) for i in range(1, 21)]
        result = group_by_length(seqs, num_buckets=4)
        total = sum(len(v) for v in result.values())
        assert total == 20

    def test_default_num_buckets(self) -> None:
        seqs = [_FakeSequence(50), _FakeSequence(500)]
        result = group_by_length(seqs)
        assert len(result) == 4  # default num_buckets=4