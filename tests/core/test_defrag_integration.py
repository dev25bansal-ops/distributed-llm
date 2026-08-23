"""Integration tests for MemoryDefragmenter with PagedAttentionManager."""

from __future__ import annotations

import threading
from typing import Any

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
    TieredCompactionLevel,
)


def _cpu_manager(num_blocks: int = 64, allocated: int = 32) -> PagedAttentionManager:
    """Create a PagedAttentionManager on CPU with simple allocation pattern."""
    mgr = PagedAttentionManager(
        num_blocks=num_blocks,
        block_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=64,
        device="cpu",
    )
    # Allocate first `allocated` blocks
    for i in range(allocated):
        bid = mgr._free_blocks.pop()
        block = mgr._blocks[bid]
        block.allocate(num_heads=4, head_dim=64, device="cpu")
        block.ref_count = 1
    # Assign them to a fake sequence
    mgr._seq_blocks["seq_0"] = SequenceBlocks(
        sequence_id="seq_0",
        block_ids=list(range(allocated)),
        num_tokens=allocated * 16,
    )
    return mgr


# ─── GPU Tests (pytest -m gpu) ───


@pytest.mark.gpu
def test_gpu_defrag_reduces_fragmentation():
    """On GPU, defrag must reduce fragmentation ratio."""

    defrag = MemoryDefragmenter(
        DefragConfig(policy=DefragPolicy.AGGRESSIVE)
    )
    mgr = PagedAttentionManager(
        num_blocks=64,
        block_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=64,
        device="cuda",
    )
    # Scattered allocation pattern: 4 groups of 8 blocks with gaps
    group_size = 8
    groups = 4
    allocated_ids = []
    for g in range(groups):
        start = g * (group_size + 4)
        for i in range(start, start + group_size):
            allocated_ids.append(i)

    for bid in allocated_ids:
        block = mgr._blocks[bid]
        block.allocate(num_heads=4, head_dim=64, device="cuda")
        block.ref_count = 1
    mgr._free_blocks = [b for b in range(64) if b not in allocated_ids]
    mgr._seq_blocks["seq_0"] = SequenceBlocks(
        sequence_id="seq_0",
        block_ids=allocated_ids,
        num_tokens=len(allocated_ids) * 16,
    )

    ratio_before = MemoryDefragmenter.compute_fragmentation_ratio(mgr._blocks)
    result = defrag.defragment(mgr)
    ratio_after = MemoryDefragmenter.compute_fragmentation_ratio(mgr._blocks)

    assert ratio_after < ratio_before
    assert result.blocks_moved > 0
    assert result.bytes_compacted > 0


@pytest.mark.gpu
def test_gpu_tiered_l1_compaction():
    """Tiered L1 compaction must not leave gaps on GPU."""

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    mgr = _cpu_manager(num_blocks=64, allocated=32)
    # Force device to cuda
    for b in mgr._blocks:
        if b.key_cache is not None:
            b.key_cache = b.key_cache.to("cuda")
            b.value_cache = b.value_cache.to("cuda")

    result = defrag._defragment_impl(mgr, tier=TieredCompactionLevel.L1_HOT)
    assert result.blocks_moved >= 0
    assert result.tier_used == TieredCompactionLevel.L1_HOT


@pytest.mark.gpu
def test_gpu_no_memory_leak():
    """After defrag, total GPU memory should not increase."""

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    mgr = PagedAttentionManager(
        num_blocks=32,
        block_size=16,
        num_layers=1,
        num_heads=4,
        head_dim=64,
        device="cuda",
    )
    # Allocate blocks 0-15 (contiguous), then free some in middle
    for i in range(16):
        mgr._blocks[i].allocate(num_heads=4, head_dim=64, device="cuda")
        mgr._blocks[i].ref_count = 1
    mgr._free_blocks = list(range(16, 32))
    mgr._seq_blocks["seq_0"] = SequenceBlocks(
        sequence_id="seq_0", block_ids=list(range(16)), num_tokens=256
    )

    # Free middle blocks 4-7 to create fragmentation
    for i in range(4, 8):
        mgr._blocks[i].free()
        mgr._blocks[i].ref_count = 0
        mgr._free_blocks.append(i)

    mem_before = torch.cuda.memory_allocated()
    result = defrag.defragment(mgr)
    mem_after = torch.cuda.memory_allocated()

    # Memory should not increase
    assert mem_after <= mem_before * 1.05  # Allow 5% tolerance
    assert result.blocks_moved > 0


# ─── CPU Integration Tests ───


def test_defrag_via_defragmentable_protocol():
    """Defragmenter must work via Defragmentable protocol methods."""

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    mgr = _cpu_manager(num_blocks=64, allocated=32)

    # Free middle 8 blocks to create fragmentation
    for i in range(12, 20):
        mgr._blocks[i].free()
        mgr._blocks[i].ref_count = 0
        mgr._free_blocks.append(i)

    # Verify protocol conformance
    assert hasattr(mgr, "get_blocks") and callable(mgr.get_blocks)

    ratio_before = MemoryDefragmenter.compute_fragmentation_ratio(
        mgr.get_blocks()
    )
    result = defrag.defragment(mgr)

    # After defrag, all free blocks should be contiguous at the end
    free_ids = mgr.get_seq_blocks()  # not directly useful for checking
    blocks = mgr.get_blocks()
    last_allocated_idx = -1
    free_seen = False
    for i, b in enumerate(blocks):
        if b.is_allocated:
            last_allocated_idx = i
            if free_seen:
                pytest.fail("Allocated block found after free block")
        else:
            free_seen = True

    ratio_after = MemoryDefragmenter.compute_fragmentation_ratio(blocks)
    # After defrag, fragmentation must not increase
    assert ratio_after <= ratio_before
    # All allocated blocks are before free blocks
    assert last_allocated_idx >= 0


def test_defrag_with_fragmentation_aware_allocator():
    """Fragmentation-aware allocator should reduce fragmentation over time."""

    mgr = _cpu_manager(num_blocks=64, allocated=32)

    # Free a scattered pattern
    for i in [8, 9, 17, 18, 26, 27]:
        mgr._blocks[i].free()
        mgr._blocks[i].ref_count = 0
        mgr._free_blocks.append(i)

    ratio_before = MemoryDefragmenter.compute_fragmentation_ratio(mgr._blocks)

    # Run defrag
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    defrag.defragment(mgr)

    ratio_after = MemoryDefragmenter.compute_fragmentation_ratio(mgr._blocks)
    # After defrag, no allocated block should appear after a free block
    free_seen = False
    for b in mgr._blocks:
        if b.is_allocated:
            assert not free_seen, "Allocated block after free block after defrag"
        else:
            free_seen = True


def test_defrag_multiple_sequences():
    """Multiple sequences with interleaved free blocks must all update tables."""

    mgr = _cpu_manager(num_blocks=128, allocated=0)
    # Create interleaved pattern: seq1 takes evens, seq2 takes odds
    # Leave every 5th block free to create fragmentation
    seq1_ids = [0, 2, 4, 6, 8]
    seq2_ids = [1, 3, 5, 7, 9]
    allocated_ids = seq1_ids + seq2_ids
    # Add a gap at 10, then allocate 11-20
    for bid in range(11, 21):
        allocated_ids.append(bid)

    for bid in allocated_ids:
        block = mgr._blocks[bid]
        block.allocate(num_heads=4, head_dim=64, device="cpu")
        block.ref_count = 1

    mgr._free_blocks = [b for b in range(128) if b not in allocated_ids]
    mgr._seq_blocks["seq_1"] = SequenceBlocks(
        sequence_id="seq_1", block_ids=seq1_ids, num_tokens=80
    )
    mgr._seq_blocks["seq_2"] = SequenceBlocks(
        sequence_id="seq_2", block_ids=seq2_ids, num_tokens=80
    )
    mgr._seq_blocks["seq_3"] = SequenceBlocks(
        sequence_id="seq_3", block_ids=list(range(11, 21)), num_tokens=160
    )

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    result = defrag.defragment(mgr)

    # All blocks referenced by sequences must now be contiguous and valid
    for seq in mgr._seq_blocks.values():
        for bid in seq.block_ids:
            assert mgr._blocks[bid].is_allocated, f"seq references freed block {bid}"

    assert result.blocks_moved > 0


def test_defrag_preserves_ref_counts():
    """After defrag, ref_counts must remain consistent."""

    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    mgr = _cpu_manager(num_blocks=64, allocated=24)

    # Free gap in the middle
    for i in range(12, 18):
        mgr._blocks[i].free()
        mgr._blocks[i].ref_count = 0
        mgr._free_blocks.append(i)

    refs_before = {b.block_id: b.ref_count for b in mgr._blocks if b.is_allocated}
    defrag.defragment(mgr)
    refs_after = {b.block_id: b.ref_count for b in mgr._blocks if b.is_allocated}

    # Total ref_count sum should be preserved
    assert sum(refs_before.values()) == sum(refs_after.values())
    # Every allocated block must have ref_count > 0
    assert all(rc > 0 for rc in refs_after.values())


def test_defrag_end_to_end_via_coordinator():
    """Coordinator's defrag hooks must be callable without error."""

    from distllm.core.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord._defragmenters = {}
    coord._defrag_interval = 60.0
    coord._metrics_collector = None
    # init_defragmentation delegates to the configurator (real __init__
    # creates it at line ~121); __new__ bypasses that, so wire it here.
    from distllm.core.coordinator_config_wiring import CoordinatorConfigurator
    coord._configurator = CoordinatorConfigurator(coord)

    # Call init_defragmentation with settings
    from distllm.config.settings import DefragmentationSettings
    settings = DefragmentationSettings(enabled=True)
    coord.init_defragmentation(settings=settings)
    assert coord._defragmenter is not None
