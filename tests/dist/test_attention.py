"""Real tests for PagedAttention — BlockPool with real CPU/CUDA tensors.

No mocks — uses actual BlockPool, BlockTable, and PagedAttentionManager.
CPU-only for most tests; CUDA for device-specific operations.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.attention import BlockPool, PagedAttentionManager


class TestBlockPool:
    def test_create_pool(self):
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        assert bp.num_blocks == 64
        assert bp.free_count == 64
        assert bp.used_count == 0

    def test_allocate_block(self):
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        block = bp.allocate_block()
        assert block is not None
        assert bp.used_count == 1
        assert bp.free_count == 63

    def test_allocate_all_blocks(self):
        bp = BlockPool(num_blocks=8, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        block_ids = []
        for _ in range(8):
            bid = bp.allocate_block()
            block_ids.append(bid)
        assert all(b is not None for b in block_ids)
        assert bp.free_count == 0  # pool exhausted

    def test_free_block(self):
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        block = bp.allocate_block()
        bp.free_block(block)
        assert bp.used_count == 0
        assert bp.free_count == 64

    def test_stats(self):
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        s = bp.stats()
        assert "free_blocks" in s
        assert s["free_blocks"] == 64

    def test_utilization(self):
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cpu')
        util = bp.utilization
        assert util == 0.0
        bp.allocate_block()
        assert bp.utilization > 0.0


class TestBlockPoolOnCuda:
    def test_create_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        bp = BlockPool(num_blocks=64, block_size=16, num_layers=4, num_heads=8, head_dim=64, device='cuda')
        assert bp.num_blocks == 64
        assert bp.free_count == 64


class TestExplicitBlockBudget:
    """Regression: audit 2026-11-13 attention.py finding.

    PagedAttentionManager did not expose auto_expand, so BlockPool always
    doubled (+64 min) past an explicitly configured num_blocks — an
    operator's DISTLLM_PAGED_NUM_BLOCKS could never be a hard memory
    ceiling. auto_expand now defaults to False on the manager.
    """

    def test_manager_pool_does_not_expand_past_explicit_budget(self):
        mgr = PagedAttentionManager(
            num_blocks=4, block_size=2, num_layers=1,
            num_heads=1, head_dim=2, device="cpu",
        )
        ids = [mgr.pool.allocate_block() for _ in range(4)]
        assert all(i is not None for i in ids)
        # 5th allocation must fail (None) rather than silently growing.
        assert mgr.pool.allocate_block() is None
        assert mgr.pool.num_blocks == 4

    def test_manager_opt_in_auto_expand_still_works(self):
        mgr = PagedAttentionManager(
            num_blocks=4, block_size=2, num_layers=1,
            num_heads=1, head_dim=2, device="cpu",
            auto_expand=True,
        )
        for _ in range(4):
            assert mgr.pool.allocate_block() is not None
        assert mgr.pool.allocate_block() is not None  # expanded on demand
        assert mgr.pool.num_blocks > 4

    def test_create_sequence_raises_instead_of_auto_expanding(self):
        mgr = PagedAttentionManager(
            num_blocks=1, block_size=2, num_layers=1,
            num_heads=1, head_dim=2, device="cpu",
        )
        mgr.create_sequence("a")  # takes the only block
        with pytest.raises(RuntimeError, match="No free blocks"):
            mgr.create_sequence("b")
