"""Merkle tree for efficient page-table sync between nodes.

Each leaf is a SHA-256 hash of a KV cache block. Two nodes compare
their Merkle roots to detect divergence, then walk the tree to find
exactly which blocks differ — avoiding full page-table transfer.
"""

import hashlib
from typing import List

EMPTY_HASH = "0" * 64


class MerkleTree:
    """Merkle tree over a list of block hashes.

    Leaves are SHA-256 hashes of KV cache block data.
    Internal nodes are SHA-256(children concatenated).
    The tree is padded to a power-of-two with EMPTY_HASH leaves.

    Usage::

        tree = MerkleTree(["hash1", "hash2", "hash3"])
        other = MerkleTree(["hash1", "hash2", "hash4"])
        differing = tree.diff(other)  # → [2]
    """

    def __init__(self, leaves: List[str] | None = None):
        self._leaves: list[str] = leaves or []
        self._levels: list[list[str]] = []
        self._rebuild()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> str:
        """Top-level Merkle root hash (64-char hex)."""
        if not self._levels:
            return EMPTY_HASH
        return self._levels[-1][0]

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    def update(self, leaves: List[str]) -> None:
        """Replace all leaves and rebuild the tree."""
        self._leaves = list(leaves)
        self._rebuild()

    def get_proof(self, index: int) -> List[str]:
        """Merkle proof for the leaf at *index*.

        Returns a list of sibling hashes from leaf to root.
        """
        proof: list[str] = []
        for level in self._levels[:-1]:
            sibling_idx = index ^ 1
            if sibling_idx < len(level):
                proof.append(level[sibling_idx])
            index //= 2
        return proof

    def diff(self, other: "MerkleTree") -> List[int]:
        """Return indices of leaves that differ from *other*'s tree.

        Recursively walks down from the root; O(log n) when trees are
        identical, O(k log n) when *k* leaves differ.
        """
        if self.root == other.root:
            return []

        if not self._levels or not other._levels:
            return list(range(max(len(self._leaves), len(other._leaves))))

        # Start from the top (root) level and recurse down
        top = len(self._levels) - 1
        span = len(self._levels[0])
        return self._diff_range(other, top, 0, span)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Build the tree bottom-up from current leaves."""
        if not self._leaves:
            self._levels = []
            return

        n = 1
        while n < len(self._leaves):
            n <<= 1
        padded: list[str] = self._leaves + [EMPTY_HASH] * (n - len(self._leaves))

        self._levels = [padded]
        while len(self._levels[-1]) > 1:
            top = self._levels[-1]
            nxt: list[str] = []
            for i in range(0, len(top), 2):
                combined = top[i] + top[i + 1]
                nxt.append(hashlib.sha256(combined.encode()).hexdigest())
            self._levels.append(nxt)

    def _node_hash(self, level: int, offset: int) -> str | None:
        """Get the hash at *(level, offset)* or None if out of range."""
        if level < 0 or level >= len(self._levels):
            return None
        if offset < 0 or offset >= len(self._levels[level]):
            return None
        return self._levels[level][offset]

    def _other_hash(self, other: "MerkleTree", level: int, offset: int) -> str | None:
        """Get the hash at *(level, offset)* from *other* or None."""
        if level < 0 or level >= len(other._levels):
            return None
        if offset < 0 or offset >= len(other._levels[level]):
            return None
        return other._levels[level][offset]

    def _node_width(self, level: int) -> int:
        """Number of leaves spanned by one node at this level."""
        return 1 << (len(self._levels) - 1 - level) if self._levels else 0

    def _diff_range(
        self, other: "MerkleTree", level: int, offset: int, span: int
    ) -> List[int]:
        """Recursively compare the range *(offset, offset + span)*.

        Returns leaf indices that differ.  Starts at the root level
        and recurses down to the leaves.
        """
        my_h = self._node_hash(level, offset)
        ot_h = self._other_hash(other, level, offset)

        if my_h == ot_h:
            return []
        if my_h is None or ot_h is None:
            # One tree has no node here — all leaves in range differ
            results: list[int] = []
            leaf_offset = offset * (1 << (len(self._levels) - 1 - level)) if self._levels else offset
            for i in range(span):
                idx = leaf_offset + i
                if idx < len(self._leaves) or idx < len(other._leaves):
                    results.append(idx)
            return results

        if level == 0:
            return [offset] if offset < len(self._leaves) else []

        half = span // 2
        return self._diff_range(other, level - 1, offset * 2, half) + self._diff_range(
            other, level - 1, offset * 2 + 1, half
        )


def verify_proof(leaf_hash: str, proof: List[str], root: str, leaf_index: int = 0) -> bool:
    """Verify a Merkle proof for *leaf_hash* against *root*.

    Args:
        leaf_hash: SHA-256 hex hash of the leaf.
        proof: List of sibling hashes from :meth:`MerkleTree.get_proof`.
        root: Expected Merkle root hash.
        leaf_index: Index of the leaf (to determine left/right ordering).

    Returns:
        True if the proof is valid.
    """
    current = leaf_hash
    idx = leaf_index
    for sibling in proof:
        if idx % 2 == 0:
            combined = current + sibling  # leaf is left child
        else:
            combined = sibling + current  # leaf is right child
        current = hashlib.sha256(combined.encode()).hexdigest()
        idx //= 2
    return current == root
