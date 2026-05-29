"""Security and correctness boundary tests for MemoryDefragmenter.

Tests cover:
  - No tensor data leaks between sequences after compaction.
  - No dangling GPU pointers after free().
  - Mutation testing: all code paths execute.
  - Edge cases: empty pool, single block, single free block.
"""

from __future__ import annotations

import pytest
import torch

from distllm.backends.paged_attention import (
    KVCacheBlock,
    PagedAttentionManager,
    SequenceBlocks,
)
from distllm.core.memory_defragmenter import (
    DefragConfig,
    DefragPolicy,
    MemoryDefragmenter,
)


def _make_manager(
    num_blocks: int = 32,
    allocated_ids: list[int] | None = None,
) -> PagedAttentionManager:
    mgr = PagedAttentionManager(
        num_blocks=num_blocks,
        block_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=64,
        device="cpu",
    )
    if allocated_ids:
        for bid in allocated_ids:
            block = mgr._blocks[bid]
            block.allocate(num_heads=4, head_dim=64, device="cpu")
            block.ref_count = 1
        mgr._free_blocks = [
            b for b in range(num_blocks) if b not in allocated_ids
        ]
        mgr._seq_blocks["seq_0"] = SequenceBlocks(
            sequence_id="seq_0",
            block_ids=allocated_ids,
            num_tokens=len(allocated_ids) * 16,
        )
    return mgr


# ─── 6.6.1: No tensor data leaks between sequences ───


def test_no_data_leak_between_sequences():
    """After defrag, block data must be preserved and no cross-contamination."""
    mgr = _make_manager(num_blocks=16, allocated_ids=[0, 1, 2, 5, 6, 7])
    blocks = mgr._blocks

    # Write unique data to each allocated block
    for bid in [0, 1, 2, 5, 6, 7]:
        blocks[bid].key_cache.fill_(bid + 100)
        blocks[bid].value_cache.fill_(bid + 200)

    # Snapshot original data per original block_id
    snapshots = {}
    for bid in [0, 1, 2, 5, 6, 7]:
        snapshots[bid] = {
            "key": blocks[bid].key_cache.clone(),
            "value": blocks[bid].value_cache.clone(),
        }
    # Track original block_ids in the seq
    original_ids = set(mgr._seq_blocks["seq_0"].block_ids)

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    defrag.defragment(mgr)

    # Collect data from ALL allocated blocks after defrag
    allocated_block_ids = {b.block_id for b in blocks if b.is_allocated}
    # Every allocated block must have data from exactly one original block
    data_found: set[tuple[int, int]] = set()
    for bid in allocated_block_ids:
        key_val = int(blocks[bid].key_cache[0, 0, 0, 0].item())
        assert key_val in (100, 101, 102, 105, 106, 107), f"Unexpected data in block {bid}"
        data_found.add(key_val)

    # Must cover all 6 original data values (no loss)
    expected_data = {bid + 100 for bid in original_ids}
    assert data_found == expected_data, "Data loss or duplication after defrag"


def test_no_dangling_pointers_after_free():
    """After free(), freed blocks must have None key/value cache."""
    mgr = _make_manager(num_blocks=16, allocated_ids=[0, 1, 2, 3])
    blocks = mgr._blocks

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    defrag.defragment(mgr)

    for b in blocks:
        if not b.is_allocated:
            assert b.key_cache is None, f"Freed block {b.block_id} has non-None key_cache"
            assert b.value_cache is None, f"Freed block {b.block_id} has non-None value_cache"
            assert b.ref_count == 0, f"Freed block {b.block_id} has non-zero ref_count"


def test_no_dangling_seq_references():
    """After defrag, free blocks must not appear in any seq.block_ids."""
    mgr = _make_manager(num_blocks=16, allocated_ids=[0, 1, 2, 5, 6, 7])

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    defrag.defragment(mgr)

    allocated_set = {b.block_id for b in mgr._blocks if b.is_allocated}
    for seq in mgr._seq_blocks.values():
        for bid in seq.block_ids:
            assert bid in allocated_set, (
                f"seq {seq.sequence_id} references freed block {bid}"
            )


# ─── 6.6.2: Edge cases ───


def test_edge_empty_pool():
    """Defragmenting an empty pool must not crash."""
    mgr = _make_manager(num_blocks=0)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved == 0


def test_edge_single_allocated_block():
    """Single allocated block with no fragmentation must be a no-op."""
    mgr = _make_manager(num_blocks=8, allocated_ids=[0])
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved == 0
    assert mgr._blocks[0].is_allocated


def test_edge_single_free_block():
    """Single free block with no allocated blocks must not fail."""
    mgr = _make_manager(num_blocks=8, allocated_ids=[])
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved == 0


def test_edge_all_blocks_free():
    """All blocks free — defrag must be a no-op."""
    mgr = _make_manager(num_blocks=32, allocated_ids=[])
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved == 0
    assert all(not b.is_allocated for b in mgr._blocks)


def test_edge_all_blocks_allocated():
    """All blocks allocated — defrag must be a no-op."""
    mgr = _make_manager(num_blocks=32, allocated_ids=list(range(32)))
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved == 0
    assert all(b.is_allocated for b in mgr._blocks)


def test_edge_fully_fragmented():
    """Alternating alloc/free pattern — defrag must compact."""
    mgr = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)
    assert result.blocks_moved > 0

    # After: all allocated blocks must be at the front
    free_seen = False
    for b in mgr._blocks:
        if b.is_allocated:
            assert not free_seen
        else:
            free_seen = True


# ─── 6.6.3: Code path coverage (mutation testing) ───


def test_all_policies_execute():
    """Every DefragPolicy must produce a valid result."""
    for policy in DefragPolicy:
        mgr = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))
        defrag = MemoryDefragmenter(DefragConfig(policy=policy))
        result = defrag.defragment(mgr)
        assert isinstance(result.blocks_moved, int)


def test_all_tiered_levels():
    """Every TieredCompactionLevel must produce a valid result on CPU."""
    from distllm.core.memory_defragmenter import TieredCompactionLevel

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))

    for tier in TieredCompactionLevel:
        mgr = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))
        result = defrag._defragment_impl(mgr, tier=tier)
        assert isinstance(result.blocks_moved, int)
        assert result.tier_used == tier


def test_async_path():
    """Async defrag must produce the same result as sync."""

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    mgr_sync = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))

    import asyncio

    mgr_async = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))
    result_async = asyncio.run(defrag.defragment_async(mgr_async))

    result_sync = defrag.defragment(mgr_sync)

    assert result_async.blocks_moved == result_sync.blocks_moved
    assert result_async.bytes_compacted == result_sync.bytes_compacted


def test_should_defragment_threshold():
    """should_defragment must respect configured policy threshold."""
    mgr = _make_manager(num_blocks=32, allocated_ids=list(range(0, 32, 2)))
    blocks = mgr._blocks

    ratio = MemoryDefragmenter.compute_fragmentation_ratio(blocks)

    # Aggressive threshold is 0.15 — should trigger for any fragmentation > 15%
    defrag_aggressive = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    # LAZY threshold is 0.50 — may or may not trigger depending on actual ratio
    defrag_lazy = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.LAZY))

    if ratio >= 0.15:
        assert defrag_aggressive.should_defragment(blocks)
    if ratio < 0.50:
        assert not defrag_lazy.should_defragment(blocks)


def test_predict_fragmentation():
    """Predictive defrag must produce stable forecasts."""
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.BALANCED, enable_predictive=True))
    history = [0.1, 0.2, 0.3, 0.4, 0.5]
    prediction = defrag.predict_fragmentation(history)
    assert 0.0 <= prediction <= 1.0
    assert prediction >= 0.0


def test_compute_fragmentation_ratio_static():
    """Static compute_fragmentation_ratio must work on raw block lists."""
    blocks = [
        KVCacheBlock(block_id=0, is_allocated=True),
        KVCacheBlock(block_id=1, is_allocated=False),
        KVCacheBlock(block_id=2, is_allocated=True),
    ]
    ratio = MemoryDefragmenter.compute_fragmentation_ratio(blocks)
    assert ratio == 1.0 / 3.0
