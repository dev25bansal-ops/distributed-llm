"""Unit tests for dist/merkle.py MerkleTree.

Covers:
- Empty tree
- Single leaf
- Multiple leaves (power-of-2 and non-power-of-2)
- Proof generation and verification
- Diff between trees
- xxhash vs SHA-256 fallback
"""

import importlib.util
import os
import sys

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "distllm.dist.merkle",
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "distllm", "dist", "merkle.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merkle():
    return _load_module()


class TestEmptyTree:
    def test_empty_root(self, merkle):
        tree = merkle.MerkleTree()
        assert tree.root == merkle.EMPTY_HASH
        assert tree.leaf_count == 0

    def test_empty_levels(self, merkle):
        tree = merkle.MerkleTree()
        assert tree._levels == []


class TestSingleLeaf:
    def test_single(self, merkle):
        tree = merkle.MerkleTree(["a" * 64])
        assert tree.leaf_count == 1
        assert tree.root != merkle.EMPTY_HASH
        assert len(tree.root) == merkle._HASH_HEX_LEN


class TestMultipleLeaves:
    def test_power_of_two(self, merkle):
        tree = merkle.MerkleTree(["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        assert tree.leaf_count == 4
        assert len(tree._levels) == 3  # 4 -> 2 -> 1

    def test_non_power_of_two(self, merkle):
        tree = merkle.MerkleTree(["a" * 64, "b" * 64, "c" * 64])
        assert tree.leaf_count == 3
        assert tree.root != merkle.EMPTY_HASH

    def test_deterministic(self, merkle):
        t1 = merkle.MerkleTree(["a" * 64, "b" * 64])
        t2 = merkle.MerkleTree(["a" * 64, "b" * 64])
        assert t1.root == t2.root

    def test_different_leaves_different_root(self, merkle):
        t1 = merkle.MerkleTree(["a" * 64, "b" * 64])
        t2 = merkle.MerkleTree(["a" * 64, "c" * 64])
        assert t1.root != t2.root


class TestUpdate:
    def test_update_changes_root(self, merkle):
        tree = merkle.MerkleTree(["a" * 64, "b" * 64])
        old_root = tree.root
        tree.update(["x" * 64, "y" * 64])
        assert tree.root != old_root


class TestProof:
    def test_proof_roundtrip(self, merkle):
        leaves = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        tree = merkle.MerkleTree(leaves)
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            assert merkle.verify_proof(leaf, proof, tree.root, i)

    def test_proof_non_power_of_two(self, merkle):
        leaves = ["a" * 64, "b" * 64, "c" * 64]
        tree = merkle.MerkleTree(leaves)
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            assert merkle.verify_proof(leaf, proof, tree.root, i)

    def test_proof_wrong_leaf_fails(self, merkle):
        tree = merkle.MerkleTree(["a" * 64, "b" * 64])
        proof = tree.get_proof(0)
        assert not merkle.verify_proof("x" * 64, proof, tree.root, 0)

    def test_proof_wrong_root_fails(self, merkle):
        tree = merkle.MerkleTree(["a" * 64, "b" * 64])
        proof = tree.get_proof(0)
        assert not merkle.verify_proof("a" * 64, proof, "0" * merkle._HASH_HEX_LEN, 0)


class TestDiff:
    def test_identical_trees(self, merkle):
        t1 = merkle.MerkleTree(["a" * 64, "b" * 64])
        t2 = merkle.MerkleTree(["a" * 64, "b" * 64])
        assert t1.diff(t2) == []

    def test_one_changed(self, merkle):
        t1 = merkle.MerkleTree(["a" * 64, "b" * 64, "c" * 64])
        t2 = merkle.MerkleTree(["a" * 64, "x" * 64, "c" * 64])
        diff = t1.diff(t2)
        assert 1 in diff

    def test_all_different(self, merkle):
        t1 = merkle.MerkleTree(["a" * 64, "b" * 64])
        t2 = merkle.MerkleTree(["x" * 64, "y" * 64])
        diff = t1.diff(t2)
        assert len(diff) > 0


class TestHashBackend:
    def test_uses_xxhash_if_available(self, merkle):
        try:
            import xxhash
            assert merkle._USE_XXHASH is True
            assert merkle._HASH_HEX_LEN == 16
        except ImportError:
            assert merkle._USE_XXHASH is False
            assert merkle._HASH_HEX_LEN == 64
