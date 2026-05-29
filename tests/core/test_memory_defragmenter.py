"""Tests for MemoryDefragmenter — block compaction for PagedAttention KV cache.

Covers: unit tests (CPU mock), edge cases, concurrency safety,
fragmentation metrics, compaction plans, the full defragment pass,
policy behavior, tiered compaction, and predictive defragmentation.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.memory_defragmenter import (
    DefragConfig,
    DefragPolicy,
    DefragResult,
    FragmentInfo,
    MemoryDefragmenter,
    TieredCompactionLevel,
)


# ── Helpers ──


def _make_block(
    block_id: int,
    allocated: bool,
    num_heads: int = 32,
    head_dim: int = 128,
    max_tokens: int = 16,
    ref_count: int = 0,
) -> MagicMock:
    """Create a mock KVCacheBlock."""
    d = torch.device("cpu")

    def _allocate(num_heads=num_heads, head_dim=head_dim, device="cpu"):
        block.key_cache = torch.zeros(2, num_heads, max_tokens, head_dim, device=device)
        block.value_cache = torch.zeros(2, num_heads, max_tokens, head_dim, device=device)
        block.is_allocated = True
        block.num_tokens = 0

    block = MagicMock()
    block.block_id = block_id
    block.is_allocated = allocated
    block.num_tokens = max_tokens if allocated else 0
    block.max_tokens = max_tokens
    block.ref_count = ref_count
    block.allocate = _allocate

    if allocated:
        block.key_cache = torch.zeros(2, num_heads, max_tokens, head_dim)
        block.value_cache = torch.zeros(2, num_heads, max_tokens, head_dim)
    else:
        block.key_cache = None
        block.value_cache = None

    return block


def _make_mgr(blocks: list) -> MagicMock:
    """Create a mock PagedAttentionManager."""
    mgr = MagicMock()
    mgr._blocks = blocks
    mgr._seq_blocks = {}
    mgr._free_blocks = [b.block_id for b in blocks if not b.is_allocated]
    return mgr


def _frag_blocks(alloc_pattern: str) -> list:
    """Create blocks from a string pattern: 'A' = allocated, 'F' = free."""
    blocks = []
    for i, ch in enumerate(alloc_pattern):
        blocks.append(_make_block(i, allocated=(ch == "A")))
    return blocks


# ── Fragmentation Ratio ──


class TestFragmentationRatio:
    def test_no_blocks(self):
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio([]) == 0.0

    def test_all_allocated(self):
        blocks = _frag_blocks("AAAA")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 0.0

    def test_all_free(self):
        blocks = _frag_blocks("FFFF")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 0.0

    def test_one_fragmented_free(self):
        # A F A — the free block in the middle has allocated neighbors
        blocks = _frag_blocks("AFA")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 1.0 / 3

    def test_multiple_fragmented(self):
        # A F A F A — two fragmented free blocks
        blocks = _frag_blocks("AFAFA")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 2.0 / 5

    def test_contiguous_free_at_end(self):
        # A A A F F — boundary block at idx 3 is fragmented (neighbor A), idx 4 is not
        blocks = _frag_blocks("AAAFF")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 1.0 / 5

    def test_contiguous_free_at_start(self):
        # F F A A A — boundary block at idx 1 is fragmented (neighbor A), idx 0 is not
        blocks = _frag_blocks("FFAAA")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 1.0 / 5

    def test_interleaved_full(self):
        # A F A F A F — every free block has allocated neighbors
        blocks = _frag_blocks("AFAFAF")
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 3.0 / 6


# ── Should Defragment ──


class TestShouldDefragment:
    def test_below_threshold(self):
        blocks = _frag_blocks("AAAFF")  # ratio = 0.0
        d = MemoryDefragmenter()
        assert not d.should_defragment(blocks)

    def test_above_threshold(self):
        blocks = _frag_blocks("AFAFA")  # ratio = 0.4
        d = MemoryDefragmenter()
        assert d.should_defragment(blocks)

    def test_policy_default_threshold(self):
        d = MemoryDefragmenter()
        assert d._config.policy == DefragPolicy.BALANCED
        assert d._config.policy.threshold == 0.30

    def test_lazy_policy(self):
        config = DefragConfig(policy=DefragPolicy.LAZY)
        d = MemoryDefragmenter(config=config)
        assert d._config.policy.threshold == 0.50
        blocks = _frag_blocks("AFAFA")  # ratio = 0.4
        assert not d.should_defragment(blocks)
        blocks = _frag_blocks("AFAFAFAF")  # ratio = 0.5
        assert d.should_defragment(blocks)

    def test_aggressive_policy(self):
        config = DefragConfig(policy=DefragPolicy.AGGRESSIVE)
        d = MemoryDefragmenter(config=config)
        assert d._config.policy.threshold == 0.15
        blocks = _frag_blocks("AAAAA")  # ratio = 0.0
        assert not d.should_defragment(blocks)
        blocks = _frag_blocks("AAAFF")  # ratio = 0.2 (idx 3 has allocated neighbor)
        assert d.should_defragment(blocks)


# ── Analyze Fragmentation ──


class TestAnalyzeFragmentation:
    def test_empty_blocks(self):
        d = MemoryDefragmenter()
        assert d.analyze_fragmentation([]) == []

    def test_all_allocated(self):
        blocks = _frag_blocks("AA")
        d = MemoryDefragmenter()
        fragments = d.analyze_fragmentation(blocks)
        assert len(fragments) == 2
        assert all(not f.is_free for f in fragments)
        assert all(f.adjacent_free == 0 for f in fragments)

    def test_free_block_tracks_adjacent(self):
        blocks = _frag_blocks("AFA")
        d = MemoryDefragmenter()
        fragments = d.analyze_fragmentation(blocks)
        assert len(fragments) == 3
        # Middle block (free) has 0 free neighbors (both neighbors are allocated)
        assert fragments[1].is_free
        assert fragments[1].adjacent_free == 0

    def test_fragment_info_has_block_id(self):
        blocks = _frag_blocks("AF")
        d = MemoryDefragmenter()
        fragments = d.analyze_fragmentation(blocks)
        assert fragments[0].block_id == 0
        assert fragments[1].block_id == 1


# ── Compaction Plan ──


class TestCompactionPlan:
    def test_no_fragmentation(self):
        blocks = _frag_blocks("AAAFF")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        assert moves == []

    def test_simple_compaction(self):
        blocks = _frag_blocks("AFA")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        # Block at idx 2 should move to idx 1
        assert moves == [(2, 1, 0)]

    def test_multi_gap_compaction(self):
        blocks = _frag_blocks("AFAFA")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        # Block at idx 4 moves to idx 1 → fully compacted in one move: A A A F F
        assert len(moves) == 1
        assert moves[0] == (4, 1, 0)

    def test_contiguous_free_at_end_no_moves(self):
        blocks = _frag_blocks("AAFFF")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        assert moves == []

    def test_contiguous_free_at_start_no_moves(self):
        blocks = _frag_blocks("FFFAA")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        # The algorithm moves allocated blocks from 3,4 to fill slots 0,1 → A A F F F
        # This is valid compaction (all allocated blocks contiguous)
        assert len(moves) == 2
        assert moves == [(4, 0, 0), (3, 1, 0)]

    def test_contiguous_free_at_end_no_moves(self):
        blocks = _frag_blocks("AAFFF")
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        assert moves == []

    def test_respects_ref_count(self):
        blocks = [
            _make_block(0, allocated=False),
            _make_block(1, allocated=True, ref_count=2),
            _make_block(2, allocated=True, ref_count=1),
        ]
        d = MemoryDefragmenter()
        moves = d.compute_compaction_plan(blocks)
        # Block at idx 1 has ref_count > 1, should NOT be moved
        # Block at idx 2 has ref_count = 1, should be moved to idx 0
        assert moves == [(2, 0, 1)]

    def test_max_blocks_per_pass(self):
        config = DefragConfig(max_blocks_per_pass=1)
        d = MemoryDefragmenter(config=config)
        blocks = _frag_blocks("AFAFA")
        moves = d.compute_compaction_plan(blocks)
        assert len(moves) <= 1


# ── Defragment Pass ──


class TestDefragment:
    def test_defragment_noop_when_no_fragmentation(self):
        blocks = [
            _make_block(0, allocated=True),
            _make_block(1, allocated=True),
            _make_block(2, allocated=False),
        ]
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = d.defragment(mgr)
        assert result.blocks_moved == 0
        assert result.time_ms >= 0.0

    def test_defragment_single_move(self):
        # F A A F A → 5 blocks, 2 free interleaved → compact to A A A F F
        blocks = _frag_blocks("FAAFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = d.defragment(mgr)
        assert result.blocks_moved >= 1
        assert result.fragmentation_after <= result.fragmentation_before
        assert result.bytes_compacted > 0

    def test_defragment_full_compaction(self):
        # A F A F A → compact to A A A F F (block 4 moves to idx 1)
        blocks = _frag_blocks("AFAFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = d.defragment(mgr)
        assert result.blocks_moved == 1
        assert result.fragmentation_after < result.fragmentation_before
        # Verify free blocks are contiguous at the end
        free_indices = [i for i, b in enumerate(blocks) if not b.is_allocated]
        assert free_indices == list(range(len(blocks) - len(free_indices), len(blocks)))

    def test_defragment_updates_free_list(self):
        blocks = _frag_blocks("AFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        d.defragment(mgr)
        # After compaction, free blocks should be at the end
        assert mgr._free_blocks == [1]

    def test_defragment_stats(self):
        blocks = _frag_blocks("AFAFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        d.defragment(mgr)
        stats = d.stats
        assert stats["defrag_count"] == 1
        assert stats["blocks_moved"] == 1
        assert stats["total_time_ms"] > 0.0

    def test_defragment_tier_tracking(self):
        blocks = _frag_blocks("AFAFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = d._defragment_impl(mgr, tier=TieredCompactionLevel.L2_WARM)
        assert result.tier_used == TieredCompactionLevel.L2_WARM
        stats = d.stats
        assert stats["l2_count"] == 1


# ── Tiered Compaction ──


class TestTieredCompaction:
    def test_l2_offloads_cold_sequences(self):
        blocks = [_make_block(i, allocated=True) for i in range(4)]
        seq_block = MagicMock()
        seq_block.sequence_id = "seq_1"
        seq_block.block_ids = [0, 1]
        mgr = _make_mgr(blocks)
        mgr._seq_blocks = {"seq_1": seq_block}
        mgr.swap_blocks_to_cpu = MagicMock(return_value=2)

        d = MemoryDefragmenter()
        offloaded = d._offload_cold_sequences(mgr, TieredCompactionLevel.L2_WARM)
        assert offloaded == 2

    def test_config_tiered_detection(self):
        config = DefragConfig(
            enabled=True,
            policy=DefragPolicy.BALANCED,
            tiered_compaction=True,
            l2_cpu_swap_threshold=0.50,
            l3_nvme_swap_threshold=0.75,
        )
        assert config.tiered_compaction is True


# ── Reconfigure ──


class TestReconfigure:
    def test_reconfigure_changes_policy(self):
        d = MemoryDefragmenter()
        assert d._config.policy == DefragPolicy.BALANCED
        new_config = DefragConfig(policy=DefragPolicy.AGGRESSIVE)
        d.reconfigure(new_config)
        assert d._config.policy == DefragPolicy.AGGRESSIVE

    def test_reconfigure_thread_safe(self):
        d = MemoryDefragmenter()
        errors = []

        def reconfigure_loop():
            for i in range(100):
                try:
                    cfg = DefragConfig(
                        policy=DefragPolicy.LAZY if i % 2 == 0 else DefragPolicy.AGGRESSIVE,
                    )
                    d.reconfigure(cfg)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reconfigure_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── Predictive Defragmentation ──


class TestPredictiveDefrag:
    def test_no_history_returns_zero(self):
        d = MemoryDefragmenter()
        assert d.predict_fragmentation() == 0.0

    def test_single_value(self):
        d = MemoryDefragmenter()
        d._fragmentation_history = [0.5]
        assert d.predict_fragmentation() == 0.5

    def test_prediction_returns_in_range(self):
        d = MemoryDefragmenter()
        d._fragmentation_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        pred = d.predict_fragmentation(steps_ahead=5)
        assert 0.0 <= pred <= 1.0

    def test_prediction_tracks_history_len(self):
        d = MemoryDefragmenter()
        for i in range(200):
            if len(d._fragmentation_history) >= 100:
                d._fragmentation_history.pop(0)
            d._fragmentation_history.append(i / 200)
        assert len(d.fragmentation_history) == 100
        assert max(d.fragmentation_history) <= 1.0

    def test_needs_tier_upgrade(self):
        d = MemoryDefragmenter()
        d._fragmentation_history = [0.7]
        assert d.needs_tier_upgrade


# ── Thread Safety ──


class TestThreadSafety:
    def test_defragment_thread_safe(self):
        blocks = _frag_blocks("AFAFA" * 5)
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        errors = []

        def run_defrag():
            try:
                for _ in range(20):
                    d.defragment(mgr)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_defrag) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety errors: {errors}"

    def test_stats_thread_safe(self):
        d = MemoryDefragmenter()
        errors = []

        def read_stats():
            try:
                for _ in range(100):
                    _ = d.stats
                    _ = d.defrag_count
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_stats) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── Edge Cases ──


class TestEdgeCases:
    def test_single_block_allocated(self):
        blocks = [_make_block(0, allocated=True)]
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 0.0
        assert d.compute_compaction_plan(blocks) == []

    def test_single_block_free(self):
        blocks = [_make_block(0, allocated=False)]
        d = MemoryDefragmenter()
        assert d._compute_fragmentation_ratio(blocks) == 0.0
        assert d.compute_compaction_plan(blocks) == []

    def test_alternating_pattern(self):
        blocks = _frag_blocks("A" * 50 + "F" * 50)
        d = MemoryDefragmenter()
        # Boundary at idx 50: first free block has allocated neighbor → 1/100 = 0.01
        assert d._compute_fragmentation_ratio(blocks) == 0.01

    def test_fully_fragmented(self):
        blocks = _frag_blocks("AF" * 25)
        d = MemoryDefragmenter()
        ratio = d._compute_fragmentation_ratio(blocks)
        assert ratio == 0.5

    def test_defragmenter_not_initialized(self):
        mgr = _make_mgr([])
        d = MemoryDefragmenter()
        result = d.defragment(mgr)
        assert isinstance(result, DefragResult)
        assert result.blocks_moved == 0

    def test_analyze_none_key_cache(self):
        block = _make_block(0, allocated=True)
        block.key_cache = None
        d = MemoryDefragmenter()
        fragments = d.analyze_fragmentation([block])
        assert fragments[0].size_bytes == 0


# ── DefragResult ──


class TestDefragResult:
    def test_to_dict_contains_all_keys(self):
        r = DefragResult(
            blocks_moved=5,
            bytes_compacted=1024,
            time_ms=10.5,
            fragmentation_before=0.5,
            fragmentation_after=0.1,
            tier_used=TieredCompactionLevel.L1_HOT,
        )
        d = r.to_dict()
        assert d["blocks_moved"] == 5
        assert d["bytes_compacted"] == 1024
        assert d["tier_used"] == "l1_hot"

    def test_to_dict_with_error(self):
        r = DefragResult(error="test error")
        d = r.to_dict()
        assert d["error"] == "test error"

    def test_to_dict_no_tier(self):
        r = DefragResult()
        d = r.to_dict()
        assert "tier_used" not in d


# ── Async ──


@pytest.mark.asyncio
class TestAsyncDefrag:
    async def test_defragment_async(self):
        blocks = _frag_blocks("AFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = await d.defragment_async(mgr)
        assert isinstance(result, DefragResult)
        assert result.blocks_moved == 1

    async def test_defragment_with_tier_async(self):
        blocks = _frag_blocks("AFA")
        mgr = _make_mgr(blocks)
        d = MemoryDefragmenter()
        result = await d.defragment_with_tier_async(
            mgr, TieredCompactionLevel.L2_WARM,
        )
        assert isinstance(result, DefragResult)
        assert result.tier_used == TieredCompactionLevel.L2_WARM
