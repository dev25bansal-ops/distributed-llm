"""Unit tests for backends/paged_attention.py.

Covers:
- KVCacheBlock allocation, free, ref_count
- SequenceBlocks tracking
- PagedAttentionManager: allocate, free, append, COW, swap, reset
- Input validation
- Thread safety (via concurrent tests in stress/)
"""

import pytest
import torch

from distllm.backends.paged_attention import (
    KVCacheBlock,
    PagedAttentionManager,
    SequenceBlocks,
)


# ── KVCacheBlock ────────────────────────────────────────────────────────


class TestKVCacheBlock:
    def test_initial_state(self):
        block = KVCacheBlock(block_id=0)
        assert block.block_id == 0
        assert block.num_tokens == 0
        assert block.is_allocated is False
        assert block.ref_count == 0
        assert block.key_cache is None
        assert block.value_cache is None

    def test_allocate(self):
        block = KVCacheBlock(block_id=0, max_tokens=8)
        block.allocate(num_heads=2, head_dim=4, device="cpu")
        assert block.is_allocated is True
        assert block.key_cache.shape == (2, 2, 8, 4)
        assert block.value_cache.shape == (2, 2, 8, 4)
        assert block.key_cache.device.type == "cpu"

    def test_free(self):
        block = KVCacheBlock(block_id=0, max_tokens=8)
        block.allocate(num_heads=2, head_dim=4, device="cpu")
        block.num_tokens = 5
        block.ref_count = 2
        block.free()
        assert block.is_allocated is False
        assert block.num_tokens == 0
        assert block.ref_count == 0
        assert block.key_cache is None

    def test_repr(self):
        block = KVCacheBlock(block_id=42, max_tokens=16)
        assert "42" in repr(block)
        block.allocate(num_heads=2, head_dim=4, device="cpu")
        block.num_tokens = 8
        block.ref_count = 1
        r = repr(block)
        assert "8/16" in r
        assert "ref=1" in r


# ── SequenceBlocks ──────────────────────────────────────────────────────


class TestSequenceBlocks:
    def test_defaults(self):
        seq = SequenceBlocks(sequence_id="req-1")
        assert seq.sequence_id == "req-1"
        assert seq.block_ids == []
        assert seq.num_tokens == 0


# ── PagedAttentionManager ──────────────────────────────────────────────


class TestPagedAttentionManager:
    def _make(self, **kwargs):
        defaults = dict(
            num_blocks=32,
            block_size=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            device="cpu",
        )
        defaults.update(kwargs)
        return PagedAttentionManager(**defaults)

    # -- Init & validation --

    def test_init(self):
        pam = self._make()
        assert pam.num_free_blocks == 32
        assert pam.num_used_blocks == 0

    def test_validation_negative_blocks(self):
        with pytest.raises(ValueError, match="positive"):
            self._make(num_blocks=0)

    def test_validation_bad_block_size(self):
        with pytest.raises(ValueError, match="power of 2"):
            self._make(block_size=3)

    def test_validation_negative_heads(self):
        with pytest.raises(ValueError, match="positive"):
            self._make(num_heads=-1)

    # -- Allocate / free --

    def test_allocate_sequence(self):
        pam = self._make()
        bids = pam.allocate_sequence("seq-1", num_tokens=20)
        assert len(bids) == 3  # ceil(20/8) = 3
        assert pam.num_used_blocks == 3
        assert pam.num_free_blocks == 29

    def test_allocate_exhausts_pool(self):
        pam = self._make(num_blocks=4)
        pam.allocate_sequence("s1", num_tokens=24)  # 3 blocks
        with pytest.raises(RuntimeError, match="Not enough"):
            pam.allocate_sequence("s2", num_tokens=16)  # 2 blocks, only 1 free

    def test_free_sequence(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=20)
        assert pam.num_used_blocks == 3
        pam.free_sequence("seq-1")
        assert pam.num_used_blocks == 0
        assert pam.num_free_blocks == 32

    def test_free_sequence_double_free(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=8)
        pam.free_sequence("seq-1")
        pam.free_sequence("seq-1")  # should not crash or corrupt
        assert pam.num_free_blocks == 32

    def test_free_unknown_sequence(self):
        pam = self._make()
        pam.free_sequence("nonexistent")  # should not crash

    # -- Append token --

    def test_append_token(self):
        pam = self._make(block_size=4)
        pam.allocate_sequence("seq-1", num_tokens=4)  # 1 block
        bid = pam.append_token("seq-1")
        assert isinstance(bid, int)
        assert pam._blocks[bid].num_tokens > 0

    def test_append_token_fills_new_block(self):
        pam = self._make(block_size=2)
        pam.allocate_sequence("seq-1", num_tokens=2)  # 1 block, full (num_tokens set by allocation)
        # Manually fill the block so append triggers a new one
        bid = pam.get_block_table("seq-1")[0]
        pam._blocks[bid].num_tokens = 2  # mark as full
        new_bid = pam.append_token("seq-1")  # should allocate new block
        seq = pam._seq_blocks["seq-1"]
        assert len(seq.block_ids) == 2

    def test_append_token_unknown(self):
        pam = self._make()
        with pytest.raises(ValueError, match="Unknown"):
            pam.append_token("nonexistent")

    # -- Block table --

    def test_get_block_table(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=20)
        table = pam.get_block_table("seq-1")
        assert len(table) == 3

    def test_get_block_table_unknown(self):
        pam = self._make()
        assert pam.get_block_table("nonexistent") == []

    # -- Copy-on-write --

    def test_copy_on_write(self):
        pam = self._make()
        pam.allocate_sequence("src", num_tokens=16)  # 2 blocks
        pam.copy_on_write("src", "dst")
        src_table = pam.get_block_table("src")
        dst_table = pam.get_block_table("dst")
        assert src_table == dst_table
        for bid in src_table:
            assert pam._blocks[bid].ref_count == 2

    def test_cow_unshare_on_write(self):
        pam = self._make(block_size=4)
        pam.allocate_sequence("src", num_tokens=4)  # 1 block
        pam.copy_on_write("src", "dst")
        shared_bid = pam.get_block_table("src")[0]
        assert pam._blocks[shared_bid].ref_count == 2

        # Write to src — should unshare
        pam.append_token("src")
        new_bid = pam.get_block_table("src")[0]
        # The first block should have been unshared
        assert pam._blocks[shared_bid].ref_count == 1 or new_bid != shared_bid

    def test_copy_on_write_unknown(self):
        pam = self._make()
        pam.copy_on_write("nonexistent", "dst")  # should not crash

    # -- Swap --

    def test_swap_to_cpu_and_back(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=16)
        swapped = pam.swap_blocks_to_cpu("seq-1")
        assert swapped == 2
        restored = pam.swap_blocks_to_gpu("seq-1")
        assert restored == 2

    def test_swap_unknown(self):
        pam = self._make()
        assert pam.swap_blocks_to_cpu("nonexistent") == 0
        assert pam.swap_blocks_to_gpu("nonexistent") == 0

    # -- Stats --

    def test_get_stats(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=16)
        stats = pam.get_stats()
        assert stats["num_blocks"] == 32
        assert stats["used_blocks"] == 2
        assert stats["active_sequences"] == 1

    def test_repr(self):
        pam = self._make()
        r = repr(pam)
        assert "32" in r
        assert "PagedAttentionManager" in r

    # -- Reset --

    def test_reset(self):
        pam = self._make()
        pam.allocate_sequence("seq-1", num_tokens=16)
        pam.reset()
        assert pam.num_free_blocks == 32
        assert pam.num_used_blocks == 0
        assert len(pam._seq_blocks) == 0
        assert pam._stats["peak_blocks_used"] == 0

    # -- KV cache retrieval --

    def test_get_kv_cache(self):
        pam = self._make(block_size=4, num_heads=2, head_dim=4)
        pam.allocate_sequence("seq-1", num_tokens=4)
        # Manually write some data
        bid = pam.get_block_table("seq-1")[0]
        block = pam._blocks[bid]
        block.key_cache[0, :, 0, :] = torch.ones(2, 4)
        k, v = pam.get_kv_cache("seq-1", layer_idx=0)
        assert k is not None
        assert k.shape[2] == 4  # seq_len

    def test_get_kv_cache_unknown(self):
        pam = self._make()
        k, v = pam.get_kv_cache("nonexistent", 0)
        assert k is None and v is None
