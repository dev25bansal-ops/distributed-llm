"""Merkle tree for efficient page-table sync between nodes.

Uses xxhash (xxh64) when available for ~50x faster hashing of small
leaf digests.  Falls back to SHA-256 if xxhash is not installed.
"""


from __future__ import annotations
import hashlib
from typing import List

try:
    import xxhash as _xxhash
    _USE_XXHASH = True
except ImportError:
    _USE_XXHASH = False

# Hash output length in hex chars: xxh64=16, sha256=64
_HASH_HEX_LEN = 16 if _USE_XXHASH else 64
EMPTY_HASH = "0" * _HASH_HEX_LEN
_EMPTY_HASH_BYTES = EMPTY_HASH.encode("ascii")


def _hash_pair(left: bytes, right: bytes) -> str:
    """Hash two byte strings and return a hex digest string."""

    if _USE_XXHASH:
        h = _xxhash.xxh64()
        h.update(left)
        h.update(right)
        return h.hexdigest()
    return hashlib.sha256(left + right).hexdigest()


class MerkleTree:
    def __init__(self, leaves: List[str] | None = None):
        self._leaves: list[str] = leaves or []
        self._levels: list[list[str]] = []
        self._rebuild()

    @property
    def root(self) -> str:
        if not self._levels:
            return EMPTY_HASH
        return self._levels[-1][0]

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    def update(self, leaves: List[str]) -> None:
        self._leaves = list(leaves)
        self._rebuild()

    def get_proof(self, index: int) -> List[str]:
        proof: list[str] = []
        for level in self._levels[:-1]:
            sibling_idx = index ^ 1
            if sibling_idx < len(level):
                proof.append(level[sibling_idx])
            index //= 2
        return proof

    def diff(self, other: "MerkleTree") -> List[int]:
        if self.root == other.root:
            return []

        if not self._levels or not other._levels:
            return list(range(max(len(self._leaves), len(other._leaves))))

        top = len(self._levels) - 1
        span = len(self._levels[0])
        return self._diff_range(other, top, 0, span)

    @staticmethod
    def _hash_pair(left_bytes: bytes, right_bytes: bytes) -> str:
        """Hash two pre-encoded leaf/node digests and return hex string."""

        return _hash_pair(left_bytes, right_bytes)

    def _rebuild(self) -> None:
        if not self._leaves:
            self._levels = []
            return

        n = 1
        while n < len(self._leaves):
            n <<= 1

        # Normalize leaves: hash any that aren't already the right length
        normalized: list[str] = []
        for s in self._leaves:
            if len(s) == _HASH_HEX_LEN:
                normalized.append(s)
            else:
                normalized.append(_hash_pair(s.encode("ascii"), b""))
        normalized += [EMPTY_HASH] * (n - len(normalized))

        # Pre-encode all leaves to bytes once
        current_bytes: list[bytes] = [s.encode("ascii") for s in normalized]
        self._levels = [normalized]

        while len(current_bytes) > 1:
            nxt: list[str] = []
            for i in range(0, len(current_bytes), 2):
                h = _hash_pair(current_bytes[i], current_bytes[i + 1])
                nxt.append(h)
            current_bytes = [s.encode("ascii") for s in nxt]
            self._levels.append(nxt)

    def _node_hash(self, level: int, offset: int) -> str | None:
        if level < 0 or level >= len(self._levels):
            return None
        if offset < 0 or offset >= len(self._levels[level]):
            return None
        return self._levels[level][offset]

    def _other_hash(self, other: "MerkleTree", level: int, offset: int) -> str | None:
        if level < 0 or level >= len(other._levels):
            return None
        if offset < 0 or offset >= len(other._levels[level]):
            return None
        return other._levels[level][offset]

    def _node_width(self, level: int) -> int:
        return 1 << (len(self._levels) - 1 - level) if self._levels else 0

    def _diff_range(
        self, other: "MerkleTree", level: int, offset: int, span: int
    ) -> List[int]:
        """Iterative diff: avoids recursion depth issues for large trees."""

        results: list[int] = []
        # Stack of (level, offset, span) work items
        stack = [(level, offset, span)]
        max_leaf_level = len(self._levels) - 1 if self._levels else 0

        while stack:
            lv, off, sp = stack.pop()
            my_h = self._node_hash(lv, off)
            ot_h = self._other_hash(other, lv, off)

            if my_h == ot_h:
                continue
            if my_h is None or ot_h is None:
                leaf_offset = off * (1 << (max_leaf_level - lv)) if self._levels else off
                for i in range(sp):
                    idx = leaf_offset + i
                    if idx < len(self._leaves) or idx < len(other._leaves):
                        results.append(idx)
                continue
            if lv == 0:
                if off < len(self._leaves):
                    results.append(off)
                continue

            half = sp // 2
            stack.append((lv - 1, off * 2 + 1, half))
            stack.append((lv - 1, off * 2, half))

        return results


def verify_proof(leaf_hash: str, proof: List[str], root: str, leaf_index: int = 0) -> bool:
    # Normalize leaf hash to match tree's internal representation
    if len(leaf_hash) != _HASH_HEX_LEN:
        current = _hash_pair(leaf_hash.encode("ascii"), b"").encode("ascii")
    else:
        current = leaf_hash.encode("ascii")
    idx = leaf_index
    for sibling in proof:
        sibling_bytes = sibling.encode("ascii")
        if idx % 2 == 0:
            current = _hash_pair(current, sibling_bytes).encode("ascii")
        else:
            current = _hash_pair(sibling_bytes, current).encode("ascii")
        idx //= 2
    return current.decode("ascii") == root
