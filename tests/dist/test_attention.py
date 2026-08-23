"""Real tests for PagedAttention — BlockPool with real CPU/CUDA tensors.

No mocks — uses actual BlockPool, BlockTable, and PagedAttentionManager.
CPU-only for most tests; CUDA for device-specific operations.
"""

from __future__ import annotations

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
