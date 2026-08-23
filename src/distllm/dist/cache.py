"""Cache eviction policies for KV prefix caches.

Provides TTL-based eviction and semantic-aware grouping
that can be composed with PrefixCache and RadixTreeCache.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict


class TTLPolicy:
    def __init__(self, default_ttl_seconds: float = 3600.0):
        self._default_ttl = default_ttl_seconds
        self._entry_ttl: dict[int, float] = {}
        self._entry_stored_at: dict[int, float] = {}

    def set_ttl(self, key_hash: int, ttl_seconds: float) -> None:
        self._entry_ttl[key_hash] = ttl_seconds
        self._entry_stored_at[key_hash] = time.time()

    def is_expired(self, key_hash: int, now: float | None = None) -> bool:
        if key_hash not in self._entry_stored_at:
            return True
        now = now or time.time()
        ttl = self._entry_ttl.get(key_hash, self._default_ttl)
        return (now - self._entry_stored_at[key_hash]) > ttl

    def get_expired_keys(self, all_keys: list[int], now: float | None = None) -> list[int]:
        now = now or time.time()
        return [k for k in all_keys if self.is_expired(k, now)]

    def record_access(self, key_hash: int, now: float | None = None) -> None:
        now = now or time.time()
        self._entry_stored_at[key_hash] = now

    def remove(self, key_hash: int) -> None:
        self._entry_ttl.pop(key_hash, None)
        self._entry_stored_at.pop(key_hash, None)

    def clear(self) -> None:
        self._entry_ttl.clear()
        self._entry_stored_at.clear()

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: float) -> None:
        self._default_ttl = value


class SemanticGrouping:
    def __init__(self, num_permutations: int = 128, threshold: float = 0.8):
        self._num_perm = num_permutations
        self._threshold = threshold
        self._groups: dict[str, list[int]] = defaultdict(list)
        self._hash_to_group: dict[int, str] = {}
        self._group_signatures: dict[str, list[int]] = {}
        self._next_group_id: int = 0
        # LSH index for O(1) candidate lookup instead of O(n) linear scan
        self._lsh_index: dict[int, list[str]] = defaultdict(list)
        self._num_bands = 16
        self._rows_per_band = max(1, num_permutations // self._num_bands)

    def compute_signature(self, token_ids: list[int]) -> list[int]:
        """Compute MinHash signature using fast pre-computed per-token hashes.

        Pre-computes a single hash per token, then derives each permutation
        via cheap XOR mixing instead of per-(tok, i) modular arithmetic.
        This reduces the constant factor of the O(n * k) inner loop
        significantly by eliminating the expensive ``% (2**31 - 1)``.
        """
        if not token_ids:
            return [0] * self._num_perm

        # Fast C-accelerated per-token hash, normalised to 31 bits
        token_hashes = [hash(tok) & 0x7FFFFFFF for tok in token_ids]

        signature = []
        MASK = 0x7FFFFFFF
        PHI = 0x9E3779B9  # golden-ratio constant for permutation mixing
        for i in range(self._num_perm):
            perm_seed = (PHI * i) & MASK
            min_val = MASK
            for h in token_hashes:
                v = (h ^ perm_seed) & MASK
                if v < min_val:
                    min_val = v
            signature.append(min_val)

        return signature

    def _similarity(self, sig_a: list[int], sig_b: list[int]) -> float:
        if not sig_a or not sig_b:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
        return matches / len(sig_a)

    def _band_hash(self, signature: list[int], band_idx: int) -> int:
        """Compute a single hash for one LSH band of the signature."""
        start = band_idx * self._rows_per_band
        end = min(start + self._rows_per_band, len(signature))
        return hash(tuple(signature[start:end]))

    def find_or_create_group(self, token_ids: list[int]) -> str:
        signature = self.compute_signature(token_ids)
        h = hash(tuple(token_ids))

        best_group = None
        best_score = 0.0

        # ---- Fast path: LSH-based candidate lookup (O(bands) vs O(groups)) ----
        candidates: set[str] = set()
        for band_idx in range(self._num_bands):
            band_key = self._band_hash(signature, band_idx)
            for group_id in self._lsh_index.get(band_key, ()):
                candidates.add(group_id)

        if candidates:
            for group_id in candidates:
                score = self._get_group_similarity(group_id, signature)
                if score > best_score:
                    best_score = score
                    best_group = group_id

            if best_score >= self._threshold:
                self._groups[best_group].append(h)
                self._hash_to_group[h] = best_group
                return best_group

        # ---- Fallback: linear scan of groups not found by LSH ----
        for group_id, member_hashes in self._groups.items():
            if group_id in candidates:
                continue
            if not member_hashes:
                continue
            score = self._get_group_similarity(group_id, signature)
            if score > best_score:
                best_score = score
                best_group = group_id

        if best_score >= self._threshold and best_group is not None:
            self._groups[best_group].append(h)
            self._hash_to_group[h] = best_group
            return best_group

        # ---- Create new group ----
        group_id = f"semantic_{self._next_group_id}"
        self._next_group_id += 1
        self._groups[group_id].append(h)
        self._hash_to_group[h] = group_id
        self._group_signatures[group_id] = signature
        # Index each band of the new group's signature in the LSH table
        for band_idx in range(self._num_bands):
            band_key = self._band_hash(signature, band_idx)
            self._lsh_index[band_key].append(group_id)
        return group_id

    def _get_group_similarity(self, group_id: str, signature: list[int]) -> float:
        if group_id not in self._group_signatures:
            return 0.0
        return self._similarity(self._group_signatures[group_id], signature)

    def get_group_members(self, group_id: str) -> list[int]:
        return list(self._groups.get(group_id, []))

    def get_group_id(self, token_ids: list[int]) -> str | None:
        h = hash(tuple(token_ids))
        return self._hash_to_group.get(h)

    def clear(self) -> None:
        self._groups.clear()
        self._hash_to_group.clear()
        self._group_signatures.clear()
        self._lsh_index.clear()
        self._next_group_id = 0


"""Prefix-hash-based KV cache index for gossip protocol.

Tracks which nodes hold which cache entries, enabling peer-to-peer
cache lookups across the distributed cluster.
"""


class CacheIndex:
    def __init__(self):
        self._index: dict[str, set[str]] = {}
        self._refs: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    def index_tokens(self, tokens: list[int]) -> str:
        h = hashlib.sha256()
        for t in tokens:
            h.update(t.to_bytes(4, "little", signed=True))
        return f"h{h.hexdigest()[:32]}"

    def rolling_prefix_hash(self, tokens: list[int], window_size: int = 4) -> list[str]:
        """Compute rolling prefix hashes for incremental prefix matching.

        Returns one hash per prefix position, so that two token sequences
        that share a common prefix produce identical hashes for the shared
        portion.  The old :meth:`index_tokens` hashes the *entire* sequence
        (all-or-nothing), which defeats prefix matching.

        Args:
            tokens: Token ID list.
            window_size: Tokens per hash segment.  Larger values reduce
                storage but coarsen the granularity of prefix matching.

        Returns:
            A list of segment hashes ``["h<sha256-32>", ...]``, one per
            ``window_size``-token segment.  Two sequences share a prefix
            of length N iff their first ``N // window_size`` hashes match.
        """
        segment_hasher = hashlib.sha256()
        hashes: list[str] = []
        for i, tok in enumerate(tokens):
            segment_hasher.update(tok.to_bytes(4, "little", signed=True))
            if (i + 1) % window_size == 0 or i == len(tokens) - 1:
                hashes.append(f"h{segment_hasher.hexdigest()[:32]}")
                segment_hasher = hashlib.sha256()
        return hashes

    def longest_prefix_match(
        self,
        tokens: list[int],
        window_size: int = 4,
    ) -> tuple[str | None, int]:
        """Find the stored node whose prefix best matches *tokens*.

        Returns ``(node_id, matched_tokens)`` or ``(None, 0)`` if no match.
        """
        rolling = self.rolling_prefix_hash(tokens, window_size)
        matched = 0
        best_node: str | None = None
        for i, seg_hash in enumerate(rolling):
            nodes = self._index.get(seg_hash)
            if nodes:
                matched = (i + 1) * window_size
                best_node = next(iter(nodes))
            else:
                break
        return best_node, matched

    def store(self, prefix_hash: str, node_id: str, entry_ref: str) -> None:
        if prefix_hash not in self._index:
            self._index[prefix_hash] = set()
        self._index[prefix_hash].add(node_id)
        self._refs[prefix_hash] = entry_ref

    def lookup(self, prefix_hash: str) -> str | None:
        nodes = self._index.get(prefix_hash)
        if nodes:
            self._hits += 1
            return next(iter(nodes))
        self._misses += 1
        return None

    def lookup_all(self, prefix_hash: str) -> list[str]:
        nodes = self._index.get(prefix_hash)
        if nodes:
            self._hits += 1
            return list(nodes)
        self._misses += 1
        return []

    def get_ref(self, prefix_hash: str) -> str | None:
        return self._refs.get(prefix_hash)

    def remove(self, prefix_hash: str, node_id: str | None = None) -> None:
        if prefix_hash not in self._index:
            return

        if node_id is None:
            del self._index[prefix_hash]
            self._refs.pop(prefix_hash, None)
        else:
            self._index[prefix_hash].discard(node_id)
            if not self._index[prefix_hash]:
                del self._index[prefix_hash]
                self._refs.pop(prefix_hash, None)

    def clear(self) -> None:
        self._index.clear()
        self._refs.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        all_nodes: set[str] = set()
        for nodes in self._index.values():
            all_nodes.update(nodes)

        return {
            "hit_count": self._hits,
            "miss_count": self._misses,
            "total_entries": len(self._index),
            "unique_nodes": len(all_nodes),
        }
