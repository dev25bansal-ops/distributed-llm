"""Stress, fuzz, and property-based tests for MemoryDefragmenter.

Properties asserted:
  - After defrag, all free blocks are contiguous at the end.
  - No allocated blocks are ever freed.
  - Sequence block tables always reference allocated blocks.
  - Total ref_count sum is preserved.
  - Fragmentation ratio never exceeds 1.0 or goes below 0.0.
"""

from __future__ import annotations

import random

import pytest

from distllm.backends.paged_attention import (
    PagedAttentionManager,
    SequenceBlocks,
)
from distllm.core.memory_defragmenter import (
    DefragConfig,
    DefragPolicy,
    MemoryDefragmenter,
)


def _random_manager(
    num_blocks: int,
    num_sequences: int,
    block_size: int = 16,
    seed: int = 42,
) -> PagedAttentionManager:
    """Create a PagedAttentionManager with random allocation patterns."""
    rng = random.Random(seed)
    mgr = PagedAttentionManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=1,
        num_heads=4,
        head_dim=64,
        device="cpu",
    )

    allocated_blocks: set[int] = set()
    seq_blocks: dict[str, list[int]] = {}

    for seq_idx in range(num_sequences):
        seq_id = f"seq_{seq_idx}"
        num_blocks_for_seq = rng.randint(1, max(1, num_blocks // num_sequences))
        ids = []
        for _ in range(num_blocks_for_seq):
            # Pick a random block that isn't allocated
            candidates = [
                b
                for b in range(num_blocks)
                if b not in allocated_blocks
            ]
            if not candidates:
                break
            bid = rng.choice(candidates)
            block = mgr._blocks[bid]
            block.allocate(num_heads=4, head_dim=64, device="cpu")
            block.ref_count = 1
            allocated_blocks.add(bid)
            ids.append(bid)

        if ids:
            seq_blocks[seq_id] = ids

    # Update free list
    mgr._free_blocks = [b for b in range(num_blocks) if b not in allocated_blocks]

    # Register sequences
    for seq_id, ids in seq_blocks.items():
        mgr._seq_blocks[seq_id] = SequenceBlocks(
            sequence_id=seq_id,
            block_ids=ids,
            num_tokens=len(ids) * block_size,
        )

    return mgr


# ─── Property: after defrag, all free blocks are contiguous at end ───


@pytest.mark.parametrize("seed", range(10))
def test_property_free_blocks_contiguous_at_end(seed: int):
    """After defrag, no allocated block appears after a free block."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    blocks = mgr.get_blocks()

    defrag.defragment(mgr)

    free_seen = False
    for b in blocks:
        if b.is_allocated:
            assert not free_seen, "Allocated block after free block"
        else:
            free_seen = True


@pytest.mark.parametrize("seed", range(10))
def test_property_seq_tables_consistent(seed: int):
    """Every block referenced by a sequence must be allocated after defrag."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    blocks = mgr.get_blocks()

    defrag.defragment(mgr)

    for seq in mgr.get_seq_blocks().values():
        for bid in seq.block_ids:
            assert blocks[bid].is_allocated, f"seq {seq.sequence_id} refs freed block {bid}"


@pytest.mark.parametrize("seed", range(10))
def test_property_ref_count_preserved(seed: int):
    """Sum of ref_counts must not change after defrag."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    blocks = mgr.get_blocks()

    refs_before = sum(b.ref_count for b in blocks if b.is_allocated)
    defrag.defragment(mgr)
    refs_after = sum(b.ref_count for b in blocks if b.is_allocated)

    assert refs_before == refs_after


@pytest.mark.parametrize("seed", range(10))
def test_property_fragmentation_ratio_bounds(seed: int):
    """Fragmentation ratio must always be in [0.0, 1.0]."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    blocks = mgr.get_blocks()

    ratio = MemoryDefragmenter.compute_fragmentation_ratio(blocks)
    assert 0.0 <= ratio <= 1.0

    # After defrag, ratio should be lower or equal
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    defrag.defragment(mgr)
    ratio_after = MemoryDefragmenter.compute_fragmentation_ratio(blocks)
    assert 0.0 <= ratio_after <= 1.0
    assert ratio_after <= ratio


@pytest.mark.parametrize("seed", range(10))
def test_property_no_dangling_pointers(seed: int):
    """Every allocated block must have non-None key/value cache."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    blocks = mgr.get_blocks()

    defrag.defragment(mgr)

    for b in blocks:
        if b.is_allocated:
            assert b.key_cache is not None, f"allocated block {b.block_id} has None key_cache"
            assert b.value_cache is not None, f"allocated block {b.block_id} has None value_cache"
        else:
            assert b.key_cache is None, f"free block {b.block_id} has non-None key_cache"
            assert b.value_cache is None, f"free block {b.block_id} has non-None value_cache"


@pytest.mark.parametrize("seed", range(10))
def test_property_no_block_id_duplication(seed: int):
    """No block_id may appear in more than one sequence's block_ids."""
    mgr = _random_manager(num_blocks=128, num_sequences=8, seed=seed)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))
    blocks = mgr.get_blocks()

    defrag.defragment(mgr)

    seen: set[int] = set()
    for seq in mgr.get_seq_blocks().values():
        for bid in seq.block_ids:
            assert bid not in seen, f"block {bid} duplicated across sequences"
            seen.add(bid)


# ─── Concurrent stress test ───


def test_concurrent_alloc_free_during_defrag():
    """Alloc/free operations concurrent with defrag must not corrupt state."""

    import concurrent.futures
    import threading

    mgr = _random_manager(num_blocks=256, num_sequences=16, seed=0)
    defrag = MemoryDefragmenter(DefragConfig(policy=DefragPolicy.AGGRESSIVE))

    errors: list[str] = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()

    def alloc_worker():
        seq_counter = 0
        while not stop_event.is_set():
            try:
                mgr.acquire_lock()
                try:
                    seq_id = f"stress_seq_{seq_counter}"
                    seq_counter += 1
                    if mgr.num_free_blocks >= 2:
                        mgr.allocate_sequence(seq_id, num_tokens=32)
                finally:
                    mgr.release_lock()
            except RuntimeError:
                pass
            except Exception as e:
                with errors_lock:
                    errors.append(f"alloc: {e}")

    def free_worker():
        while not stop_event.is_set():
            try:
                mgr.acquire_lock()
                try:
                    seq_ids = list(mgr.get_seq_blocks().keys())
                    if seq_ids:
                        mgr.free_sequence(seq_ids[0])
                finally:
                    mgr.release_lock()
            except RuntimeError:
                pass
            except Exception as e:
                with errors_lock:
                    errors.append(f"free: {e}")

    def defrag_worker():
        while not stop_event.is_set():
            try:
                mgr.acquire_lock()
                try:
                    defrag.defragment(mgr)
                finally:
                    mgr.release_lock()
            except RuntimeError:
                pass
            except Exception as e:
                with errors_lock:
                    errors.append(f"defrag: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(alloc_worker) for _ in range(2)
        ] + [
            pool.submit(free_worker) for _ in range(2)
        ] + [
            pool.submit(defrag_worker) for _ in range(2)
        ]

        import time
        time.sleep(2.0)
        stop_event.set()

    for f in futures:
        f.result(timeout=10)

    assert not errors, f"Errors during concurrent stress: {errors}"

    # Verify invariants still hold
    blocks = mgr.get_blocks()
    for b in blocks:
        if b.is_allocated:
            assert b.key_cache is not None
            assert b.value_cache is not None
        else:
            assert b.key_cache is None
            assert b.value_cache is None

    # All seq references must point to allocated blocks
    for seq in mgr.get_seq_blocks().values():
        for bid in seq.block_ids:
            assert blocks[bid].is_allocated, f"seq {seq.sequence_id} refs freed block {bid}"
