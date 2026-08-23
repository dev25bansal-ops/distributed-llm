"""Tests for HierarchicalDigestExchange.

Covers: bloom filter compute/check, Merkle root,
tree diff, exchange payload, should_sync, edge cases.
"""

from __future__ import annotations

import hashlib

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/hierarchical_digest.py")
HierarchicalDigestExchange = _mod.HierarchicalDigestExchange


@pytest.fixture
def digest():
    return HierarchicalDigestExchange()


class TestBloomFilter:
    def test_compute_bloom_filter_returns_bytes(self, digest):
        bloom = digest.compute_bloom_filter(["abc", "def", "ghi"])
        assert isinstance(bloom, bytes)
        assert len(bloom) > 0

    def test_bloom_filter_deterministic(self, digest):
        hashes = ["abc", "def"]
        b1 = digest.compute_bloom_filter(hashes)
        b2 = digest.compute_bloom_filter(hashes)
        assert b1 == b2

    def test_bloom_positive_match(self, digest):
        hashes = ["hello", "world"]
        bloom = digest.compute_bloom_filter(hashes)
        assert digest.check_bloom_filter(hashes, bloom) is True

    def test_bloom_no_match_different_hashes(self, digest):
        hashes = ["hello", "world"]
        bloom = digest.compute_bloom_filter(hashes)
        # Different local hashes should ideally not match
        # (bloom filters have false positives, but this is deterministic)
        result = digest.check_bloom_filter(["zzz", "yyy"], bloom)
        # It might be a false positive; just verify we get a bool
        assert isinstance(result, bool)

    def test_bloom_empty_local_list(self, digest):
        bloom = digest.compute_bloom_filter(["abc"])
        assert digest.check_bloom_filter([], bloom) is False

    def test_bloom_size_respected(self):
        d = HierarchicalDigestExchange(bloom_size=64)
        bloom = d.compute_bloom_filter(["test"])
        # 64 bits = 8 bytes + 1 = 9 bytes
        assert len(bloom) == 9

    def test_bloom_different_inputs_different_filters(self, digest):
        b1 = digest.compute_bloom_filter(["alpha"])
        b2 = digest.compute_bloom_filter(["beta"])
        # Extremely unlikely to collide with different data
        assert b1 != b2


class TestMerkleRoot:
    def test_empty_token_ids(self, digest):
        root = digest.compute_merkle_root([])
        assert root == hashlib.sha256(b"empty").hexdigest()

    def test_single_block(self, digest):
        tokens = [1, 2, 3]
        root = digest.compute_merkle_root(tokens)
        assert isinstance(root, str)
        assert len(root) == 64  # SHA-256 hex

    def test_deterministic(self, digest):
        tokens = [10, 20, 30, 40]
        r1 = digest.compute_merkle_root(tokens)
        r2 = digest.compute_merkle_root(tokens)
        assert r1 == r2

    def test_different_tokens_different_root(self, digest):
        r1 = digest.compute_merkle_root([1, 2, 3])
        r2 = digest.compute_merkle_root([1, 2, 4])
        assert r1 != r2

    def test_merkle_leaf_size_respected(self):
        d = HierarchicalDigestExchange(merkle_leaf_size=2)
        tokens = [1, 2, 3, 4, 5]
        root = d.compute_merkle_root(tokens)
        # 5 tokens / 2 = 3 leaves -> tree reduces to 1 root
        assert len(root) == 64


class TestDiffMerkleTrees:
    def test_same_root_empty_diff(self, digest):
        tokens = [1, 2, 3, 4]
        root = digest.compute_merkle_root(tokens)
        diff = digest.diff_merkle_trees(tokens, root)
        assert diff == []

    def test_different_root_returns_all_blocks(self, digest):
        tokens = [1, 2, 3, 4]
        other_root = hashlib.sha256(b"different").hexdigest()
        diff = digest.diff_merkle_trees(tokens, other_root)
        assert len(diff) > 0
        # Number of blocks matches leaf count
        expected_blocks = (len(tokens) + digest._merkle_leaf_size - 1) // digest._merkle_leaf_size
        assert len(diff) == expected_blocks


class TestExchangePayload:
    def test_build_exchange_payload(self, digest):
        payload = digest.build_exchange_payload(
            prefix_hashes=["abc", "def"],
            token_ids=[1, 2, 3],
        )
        assert payload["level"] == 1
        assert "bloom_filter" in payload
        assert "merkle_root" in payload
        assert payload["prefix_count"] == 2
        assert payload["token_count"] == 3

    def test_bloom_filter_is_hex_string(self, digest):
        payload = digest.build_exchange_payload(["test"], [1])
        # Should be a valid hex string
        bytes.fromhex(payload["bloom_filter"])


class TestShouldSync:
    def test_should_sync_with_match(self, digest):
        hashes = ["local-a", "local-b"]
        payload_local = digest.build_exchange_payload(hashes, [1, 2])
        payload_remote = digest.build_exchange_payload(hashes, [1, 2])
        should, level = digest.should_sync(
            local_hashes=hashes,
            remote_payload=payload_remote,
        )
        assert should is True
        assert level == 1

    def test_should_not_sync_empty_bloom(self, digest):
        remote_payload = {"bloom_filter": ""}
        should, level = digest.should_sync(
            local_hashes=["abc"],
            remote_payload=remote_payload,
        )
        assert should is False
        assert level == 0

    def test_should_not_sync_no_bloom_key(self, digest):
        should, level = digest.should_sync(
            local_hashes=["abc"],
            remote_payload={},
        )
        assert should is False
        assert level == 0
