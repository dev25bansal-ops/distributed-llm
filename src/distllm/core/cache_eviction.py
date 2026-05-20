"""Cache eviction policies for KV prefix caches.

Provides TTL-based eviction and semantic-aware grouping
that can be composed with PrefixCache and RadixTreeCache.
"""

from __future__ import annotations

import time
from collections import defaultdict


class TTLPolicy:
    """Time-to-live eviction policy for cache entries.

    Each entry can have a custom TTL. Expired entries are evicted first
    during memory pressure, before falling back to LRU. TTL is refreshed
    on each access (lookup or store).

    Usage:
        policy = TTLPolicy(default_ttl_seconds=3600.0)
        policy.set_ttl(hash_key, ttl_seconds=1800.0)  # optional per-entry TTL
        policy.record_access(hash_key)  # refresh TTL on access
        if policy.is_expired(hash_key): ...
        expired = policy.get_expired_keys(list(self._cache.keys()))
    """

    def __init__(self, default_ttl_seconds: float = 3600.0):
        self._default_ttl = default_ttl_seconds
        self._entry_ttl: dict[int, float] = {}       # hash -> ttl_seconds
        self._entry_stored_at: dict[int, float] = {}  # hash -> timestamp

    def set_ttl(self, key_hash: int, ttl_seconds: float) -> None:
        """Set a custom TTL for a specific entry."""
        self._entry_ttl[key_hash] = ttl_seconds
        self._entry_stored_at[key_hash] = time.time()

    def is_expired(self, key_hash: int, now: float | None = None) -> bool:
        """Check if an entry has expired."""
        if key_hash not in self._entry_stored_at:
            return True
        now = now or time.time()
        ttl = self._entry_ttl.get(key_hash, self._default_ttl)
        return (now - self._entry_stored_at[key_hash]) > ttl

    def get_expired_keys(self, all_keys: list[int], now: float | None = None) -> list[int]:
        """Return all keys that have expired."""
        now = now or time.time()
        return [k for k in all_keys if self.is_expired(k, now)]

    def record_access(self, key_hash: int, now: float | None = None) -> None:
        """Refresh TTL on access (store or lookup hit)."""
        now = now or time.time()
        self._entry_stored_at[key_hash] = now

    def remove(self, key_hash: int) -> None:
        """Remove an entry from TTL tracking."""
        self._entry_ttl.pop(key_hash, None)
        self._entry_stored_at.pop(key_hash, None)

    def clear(self) -> None:
        """Clear all TTL tracking."""
        self._entry_ttl.clear()
        self._entry_stored_at.clear()

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: float) -> None:
        self._default_ttl = value


class SemanticGrouping:
    """Groups similar prompts semantically using minhash for cache sharing.

    Computes a minhash signature for each token sequence and groups entries
    with similar signatures. This allows the cache to identify prompts that
    share semantic structure even if they differ in specific tokens.

    Usage:
        grouping = SemanticGrouping(threshold=0.8)
        group_id = grouping.find_or_create_group(token_ids)
        members = grouping.get_group_members(group_id)
    """

    def __init__(self, num_permutations: int = 128, threshold: float = 0.8):
        self._num_perm = num_permutations
        self._threshold = threshold
        self._groups: dict[str, list[int]] = defaultdict(list)  # group_id -> list of hashes
        self._hash_to_group: dict[int, str] = {}
        self._next_group_id: int = 0

    def compute_signature(self, token_ids: list[int]) -> list[int]:
        """Compute a minhash signature for a token sequence.

        Uses simple hash-based minhashing with bit-level permutations.
        """
        if not token_ids:
            return [0] * self._num_perm

        # Generate a set of hash values using different hash functions
        signature = []
        for i in range(self._num_perm):
            # Use different prime multipliers as hash function variants
            min_val = float('inf')
            for tok in token_ids:
                h = ((tok * 31337 + i * 7919) % (2**31 - 1))
                min_val = min(min_val, h)
            signature.append(int(min_val) if min_val != float('inf') else 0)

        return signature

    def _similarity(self, sig_a: list[int], sig_b: list[int]) -> float:
        """Compute Jaccard similarity between two minhash signatures."""
        if not sig_a or not sig_b:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def find_or_create_group(self, token_ids: list[int]) -> str:
        """Find an existing group or create a new one for this token sequence.

        Returns:
            group_id string for the matched or created group.
        """
        signature = self.compute_signature(token_ids)
        h = hash(tuple(token_ids))

        # Check existing groups for a match
        best_group = None
        best_score = 0.0

        for group_id, member_hashes in self._groups.items():
            if not member_hashes:
                continue
            # Compare with first member's signature as representative
            rep_hash = member_hashes[0]
            # We use stored signatures for comparison
            score = self._get_group_similarity(group_id, signature)
            if score > best_score:
                best_score = score
                best_group = group_id

        if best_score >= self._threshold and best_group is not None:
            self._groups[best_group].append(h)
            self._hash_to_group[h] = best_group
            return best_group

        # Create new group
        group_id = f"semantic_{self._next_group_id}"
        self._next_group_id += 1
        self._groups[group_id].append(h)
        self._hash_to_group[h] = group_id
        # Store signature for future comparisons
        self._group_signatures[group_id] = signature
        return group_id

    def _get_group_similarity(self, group_id: str, signature: list[int]) -> float:
        """Get similarity between a signature and a group's representative."""
        if group_id not in self._group_signatures:
            return 0.0
        return self._similarity(self._group_signatures[group_id], signature)

    def get_group_members(self, group_id: str) -> list[int]:
        """Return all token hashes in a group."""
        return list(self._groups.get(group_id, []))

    def get_group_id(self, token_ids: list[int]) -> str | None:
        """Return the group ID for a token sequence, or None if not grouped."""
        h = hash(tuple(token_ids))
        return self._hash_to_group.get(h)

    def clear(self) -> None:
        """Clear all groups."""
        self._groups.clear()
        self._hash_to_group.clear()
        self._group_signatures.clear()
        self._next_group_id = 0

    # Initialize signature storage
    def __init__(self, num_permutations: int = 128, threshold: float = 0.8):
        self._num_perm = num_permutations
        self._threshold = threshold
        self._groups: dict[str, list[int]] = defaultdict(list)
        self._hash_to_group: dict[int, str] = {}
        self._group_signatures: dict[str, list[int]] = {}
        self._next_group_id: int = 0
