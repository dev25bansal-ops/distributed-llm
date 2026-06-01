"""Cache eviction policies: TTL-based expiration and semantic grouping."""

from __future__ import annotations

import hashlib
import time
from typing import Any


class TTLPolicy:
    """Time-to-live based cache eviction policy.

    Tracks when each cache entry was stored and its TTL.
    Entries are considered expired when the current time exceeds
    the stored time plus the TTL.

    Usage::

        policy = TTLPolicy(default_ttl_seconds=3600)
        policy.set_ttl(entry_id=1, ttl_seconds=1800)
        if policy.is_expired(1):
            evict(1)
    """

    def __init__(self, default_ttl_seconds: float = 3600.0):
        self._default_ttl = default_ttl_seconds
        self._entry_stored_at: dict[int, float] = {}
        self._entry_ttl: dict[int, float] = {}

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: float) -> None:
        self._default_ttl = value

    def set_ttl(self, entry_id: int, ttl_seconds: float) -> None:
        """Set TTL for an entry. Creates the entry if it doesn't exist."""
        self._entry_ttl[entry_id] = ttl_seconds
        if entry_id not in self._entry_stored_at:
            self._entry_stored_at[entry_id] = time.time()

    def is_expired(self, entry_id: int, now: float | None = None) -> bool:
        """Check if an entry has expired.

        Args:
            entry_id: The cache entry ID.
            now: Current time (uses time.time() if None).

        Returns:
            True if the entry is expired or unknown.
        """
        if entry_id not in self._entry_stored_at:
            return True
        stored_at = self._entry_stored_at[entry_id]
        ttl = self._entry_ttl.get(entry_id, self._default_ttl)
        current = now if now is not None else time.time()
        return current > stored_at + ttl

    def get_expired_keys(self, keys: list[int], now: float | None = None) -> list[int]:
        """Return which keys are expired."""
        return [k for k in keys if self.is_expired(k, now=now)]

    def record_access(self, entry_id: int) -> None:
        """Record an access, refreshing the stored timestamp."""
        self._entry_stored_at[entry_id] = time.time()

    def remove(self, entry_id: int) -> None:
        """Stop tracking an entry."""
        self._entry_stored_at.pop(entry_id, None)
        self._entry_ttl.pop(entry_id, None)

    def clear(self) -> None:
        """Clear all tracking state."""
        self._entry_stored_at.clear()
        self._entry_ttl.clear()


class SemanticGrouping:
    """Groups cache entries by semantic similarity of their token sequences.

    Uses MinHash-style signatures to efficiently find similar sequences
    and group them together for batch eviction or cache warming.

    Usage::

        grouping = SemanticGrouping(threshold=0.8)
        gid = grouping.find_or_create_group([1, 2, 3, 4])
        grouping.find_or_create_group([1, 2, 3, 5])  # joins same group
        members = grouping.get_group_members(gid)
    """

    def __init__(self, num_permutations: int = 128, threshold: float = 0.8):
        self._num_permutations = num_permutations
        self._threshold = threshold
        self._groups: dict[str, list[list[int]]] = {}
        self._signatures: dict[str, list[int]] = {}
        self._token_to_group: dict[str, str] = {}
        self._next_group_id = 0

    def compute_signature(self, tokens: list[int]) -> list[int]:
        """Compute a MinHash signature for a token sequence.

        Uses k-shingle approach: treats the token sequence as a set of
        elements and computes MinHash for Jaccard similarity estimation.

        Returns a fixed-size vector of hash values.
        """
        if not tokens:
            return [0] * self._num_permutations

        # Treat tokens as a set (unique elements)
        token_set = set(tokens)

        signature = []
        for i in range(self._num_permutations):
            min_hash = float("inf")
            for token in token_set:
                # Use a proper hash function that distributes well
                h = hash(f"{token}_{i}") & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(int(min_hash) if min_hash != float("inf") else 0)
        return signature

    def _similarity(self, sig1: list[int], sig2: list[int]) -> float:
        """Compute Jaccard similarity between two signatures."""
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _tokens_key(self, tokens: list[int]) -> str:
        """Create a string key for a token sequence."""
        return ",".join(str(t) for t in tokens)

    def find_or_create_group(self, tokens: list[int]) -> str:
        """Find an existing group for similar tokens, or create a new one.

        Args:
            tokens: The token sequence to group.

        Returns:
            The group ID.
        """
        key = self._tokens_key(tokens)
        if key in self._token_to_group:
            return self._token_to_group[key]

        sig = self.compute_signature(tokens)

        # Check existing groups for similarity
        for gid, group_sig in self._signatures.items():
            if self._similarity(sig, group_sig) >= self._threshold:
                self._groups[gid].append(tokens)
                self._token_to_group[key] = gid
                return gid

        # Create new group
        gid = f"semantic_{self._next_group_id}"
        self._next_group_id += 1
        self._groups[gid] = [tokens]
        self._signatures[gid] = sig
        self._token_to_group[key] = gid
        return gid

    def get_group_id(self, tokens: list[int]) -> str | None:
        """Get the group ID for a token sequence, or None if not grouped."""
        key = self._tokens_key(tokens)
        return self._token_to_group.get(key)

    def get_group_members(self, group_id: str) -> list[list[int]]:
        """Get all token sequences in a group."""
        return self._groups.get(group_id, [])

    def clear(self) -> None:
        """Clear all groups."""
        self._groups.clear()
        self._signatures.clear()
        self._token_to_group.clear()
        self._next_group_id = 0
