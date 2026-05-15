"""Tests for model-aware batch profiling and scheduling."""

import pytest
from distllm.core.batch_profiler import (
    get_memory_per_sequence,
    estimate_max_batch,
    group_by_length,
)


class TestMemoryPerSequence:
    """Test KV cache memory estimation."""

    def test_small_model(self):
        info = {"hidden_size": 64, "num_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4}
        mem = get_memory_per_sequence(info, 128)
        assert mem > 0
        # 2 * 4 * 4 * (64/4) * 128 * 2 = 2 * 4 * 4 * 16 * 128 * 2 = 131072 bytes
        assert mem == 131072

    def test_medium_model(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        mem = get_memory_per_sequence(info, 256)
        assert mem > 0

    def test_gqa_model(self):
        # Grouped query attention: fewer KV heads
        info = {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8}
        mem_gqa = get_memory_per_sequence(info, 512)
        # Same model without GQA
        info_mqa = {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32}
        mem_mqa = get_memory_per_sequence(info_mqa, 512)
        # GQA should use 8/32 = 1/4 of the KV cache memory
        assert mem_gqa == mem_mqa // 4

    def test_longer_sequence_more_memory(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        mem_128 = get_memory_per_sequence(info, 128)
        mem_256 = get_memory_per_sequence(info, 256)
        assert mem_256 == mem_128 * 2

    def test_fp32_double_fp16(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        mem_32 = get_memory_per_sequence(info, 256, dtype_bytes=4)
        mem_16 = get_memory_per_sequence(info, 256, dtype_bytes=2)
        assert mem_32 == mem_16 * 2


class TestEstimateMaxBatch:
    """Test max batch size estimation."""

    def test_returns_tuple(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        batch_size, tokens = estimate_max_batch(info, 8e9)
        assert isinstance(batch_size, int)
        assert isinstance(tokens, int)
        assert batch_size >= 2
        assert tokens >= 512

    def test_more_memory_allows_larger_batch(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        bs_4gb, _ = estimate_max_batch(info, 4e9)
        bs_16gb, _ = estimate_max_batch(info, 16e9)
        assert bs_16gb >= bs_4gb

    def test_fallback_on_zero_memory(self):
        info = {"hidden_size": 0, "num_layers": 0, "num_attention_heads": 0}
        batch_size, tokens = estimate_max_batch(info, 8e9)
        assert batch_size == 32
        assert tokens == 4096

    def test_safety_factor_affects_result(self):
        info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        bs_conservative, _ = estimate_max_batch(info, 8e9, safety_factor=0.3)
        bs_aggressive, _ = estimate_max_batch(info, 8e9, safety_factor=0.9)
        assert bs_aggressive >= bs_conservative


class TestGroupByLength:
    """Test length-based sequence grouping."""

    def test_empty_list(self):
        result = group_by_length([])
        assert all(len(v) == 0 for v in result.values())

    def test_same_length_all_in_one_bucket(self):
        seqs = [_make_seq(i, 100) for i in range(5)]
        result = group_by_length(seqs)
        total = sum(len(v) for v in result.values())
        assert total == 5

    def test_different_lengths_spread_across_buckets(self):
        seqs = [
            _make_seq(0, 10),    # very short
            _make_seq(1, 50),    # short
            _make_seq(2, 200),   # medium
            _make_seq(3, 500),   # long
            _make_seq(4, 1000),  # very long
        ]
        result = group_by_length(seqs, num_buckets=4)
        non_empty = sum(1 for v in result.values() if v)
        assert non_empty >= 2  # Should spread across at least 2 buckets

    def test_respects_num_buckets(self):
        seqs = [_make_seq(i, (i + 1) * 100) for i in range(10)]
        result = group_by_length(seqs, num_buckets=3)
        assert len(result) == 3

    def test_single_sequence(self):
        seqs = [_make_seq(0, 100)]
        result = group_by_length(seqs)
        total = sum(len(v) for v in result.values())
        assert total == 1


def _make_seq(rid, length):
    """Helper to create a mock Sequence with total_len."""
    from distllm.core.batch_scheduler import Sequence
    return Sequence(request_id=f"req-{rid}", prompt_tokens=list(range(length)))
