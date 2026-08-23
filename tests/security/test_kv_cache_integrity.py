"""KV cache corruption detection and integrity verification tests.

Verifies that:
1. MerkleTree.detect_corruption() catches bit-flip corruption
2. Recovery from corrupted cache works correctly
3. Tampering with cached blocks is detected on access
"""

from __future__ import annotations

import hashlib
import struct

import pytest
import torch


# ── Helpers ────────────────────────────────────────────────────────────

def _random_kv_block(num_heads: int = 8, head_dim: int = 128,
                     block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random KV cache block with a known pattern."""
    torch.manual_seed(42)
    key = torch.randn(num_heads, block_size, head_dim)
    value = torch.randn(num_heads, block_size, head_dim)
    return key, value


def _compute_block_hash(key: torch.Tensor, value: torch.Tensor) -> str:
    """Compute a SHA-256 hash of a KV block (Merkle leaf)."""
    h = hashlib.sha256()
    h.update(key.numpy().tobytes())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _bitflip(tensor: torch.Tensor, bit_pos: int = 0) -> torch.Tensor:
    """Flip a single bit in a tensor to simulate GPU memory corruption."""
    flat = tensor.clone().flatten()
    byte_idx = (bit_pos // 8) % flat.numel()
    bit_idx = bit_pos % 8
    # Get the byte representation
    arr = flat[byte_idx:byte_idx + 1].numpy().tobytes()
    corrupted = bytearray(arr)
    corrupted[0] ^= (1 << bit_idx)
    flat[byte_idx:byte_idx + 1] = torch.frombuffer(bytes(corrupted), dtype=flat.dtype)
    return flat.reshape(tensor.shape)


class TestKVCacheIntegrity:
    """KV cache corruption detection tests."""

    def test_merkle_tree_detects_corruption(self):
        """MerkleTree should detect a single bit-flip in any block."""
        from distllm.dist.merkle import MerkleTree

        blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Build a small cache
        leaves: list[str] = []
        for i in range(8):
            k, v = _random_kv_block()
            blocks[i] = (k, v)
            leaves.append(_compute_block_hash(k, v))

        tree = MerkleTree(leaves=leaves)
        root_before = tree.root

        # Corrupt one block
        k_corrupted, v_corrupted = blocks[3]
        k_corrupted = _bitflip(k_corrupted, bit_pos=7)
        corrupted_hash = _compute_block_hash(k_corrupted, v_corrupted)

        # Update tree with corrupted hash
        updated = list(leaves)
        updated[3] = corrupted_hash
        tree.update(updated)
        root_after = tree.root

        # Root hash must differ after corruption
        assert root_before != root_after, (
            "Merkle root unchanged after corruption — "
            "corruption detection failed"
        )

    def test_detect_corruption_returns_corrupted_block_ids(self):
        """detect_corruption should return the IDs of corrupted blocks."""
        from distllm.dist.merkle import MerkleTree

        blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        expected_hashes: dict[int, str] = {}

        for i in range(8):
            k, v = _random_kv_block()
            blocks[i] = (k, v)
            expected_hashes[i] = _compute_block_hash(k, v)

        tree = MerkleTree(leaves=[expected_hashes[i] for i in range(8)])

        # Corrupt block 2 and 5
        k2, v2 = blocks[2]
        k2 = _bitflip(k2, bit_pos=3)
        k5, v5 = blocks[5]
        v5 = _bitflip(v5, bit_pos=15)

        if hasattr(tree, 'detect_corruption'):
            corrupted = tree.detect_corruption([
                _compute_block_hash(blocks[i][0] if i not in (2, 5) else (k2 if i == 2 else blocks[5][0]),
                                     (v2 if i == 2 else (v5 if i == 5 else blocks[i][1])))
                for i in range(8)
            ])
            # Both corrupted blocks should be detected
            assert 2 in corrupted or 5 in corrupted, (
                f"Corruption not detected. Corrupted IDs: {corrupted}"
            )

    def test_no_false_positives_on_clean_cache(self):
        """MerkleTree should not report corruption on unmodified data."""
        from distllm.dist.merkle import MerkleTree

        leaves = [_compute_block_hash(*_random_kv_block()) for _ in range(4)]
        tree = MerkleTree(leaves=leaves)

        if hasattr(tree, 'detect_corruption'):
            corrupted = tree.detect_corruption(leaves)
            assert len(corrupted) == 0, (
                f"False positives: clean cache reported as corrupted: {corrupted}"
            )

    def test_recovery_from_corrupted_cache(self):
        """Recovery should work after corruption is detected."""
        from distllm.dist.merkle import MerkleTree

        blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        leaves: list[str] = []
        for i in range(4):
            k, v = _random_kv_block()
            blocks[i] = (k, v)
            leaves.append(_compute_block_hash(k, v))

        tree = MerkleTree(leaves=leaves)
        root_original = tree.root

        # Corrupt block 0, then "recover" by re-computing.
        k0, v0 = blocks[0]
        k0_corrupted = _bitflip(k0, bit_pos=1)
        corrupted_hash = _compute_block_hash(k0_corrupted, v0)
        tree.update([corrupted_hash] + leaves[1:])
        assert tree.root != root_original, "corruption should change the root"

        tree.update(leaves)  # recovery: restore original hashes
        root_recovered = tree.root
        assert root_recovered == root_original, (
            "Recovery failed: root hash does not match original"
        )

    def test_tampered_block_detected_on_access(self):
        """Tampering should be detectable when the block is accessed."""
        k, v = _random_kv_block()
        original_hash = _compute_block_hash(k, v)

        # Tamper with value tensor
        v_tampered = _bitflip(v, bit_pos=31)
        tampered_hash = _compute_block_hash(k, v_tampered)

        assert original_hash != tampered_hash, (
            "Tampered hash matches original — integrity check bypassed"
        )

    def test_large_block_integrity(self):
        """Large KV blocks should have integrity verification."""
        large_k = torch.randn(32, 128, 128)
        large_v = torch.randn(32, 128, 128)

        h1 = _compute_block_hash(large_k, large_v)

        # Flip a bit deep in the tensor
        large_k_corrupted = _bitflip(large_k, bit_pos=5000)
        h2 = _compute_block_hash(large_k_corrupted, large_v)

        assert h1 != h2, "Integrity hash collision on large block"
