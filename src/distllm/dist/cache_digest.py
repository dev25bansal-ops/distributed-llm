"""KV cache digest for content-based federated routing.

Provides:
- :class:`KVCacheDigest` — rolling-hash digest of cached prefixes (Merkle-tree-like)
- :class:`ContentRouter` — routes requests to the cluster with the most relevant cache
- :class:`CacheDigestExchange` — compact digest serialization for heartbeat exchange

Enables content-based routing across federated clusters: requests are sent to the
cluster that already has the most relevant KV cache (few-shot examples, system prompt),
avoiding redundant recomputation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_HASH_BASE = 31337
_HASH_MOD = (1 << 61) - 1
_DIGEST_VERSION = 1


def _rolling_hash(token_ids: list[int], window_size: int = 128) -> dict[int, int]:
    """Compute polynomial rolling hash over sliding windows in O(n) time.

    Uses Rabin-Karp style rolling hash: subtracts the oldest term's contribution
    and adds the newest term in O(1) per step, avoiding O(n * window_size)
    recomputation.

    Returns a dict mapping ``window_start`` to a 61-bit hash.
    Only windows that start at multiples of ``window_size // 4`` are included
    (stride = window_size // 4) to keep the digest compact.
    """
    if not token_ids:
        return {}

    stride = max(1, window_size // 4)
    n = len(token_ids)
    hashes: dict[int, int] = {}

    # All stride-aligned start positions
    stride_starts = list(range(0, n, stride))

    # --- Full-size windows (exactly window_size tokens) ---
    # Compute via O(n) rolling hash (Rabin-Karp style).
    full_starts = [s for s in stride_starts if s + window_size <= n]
    if full_starts:
        k = window_size

        # First window [0, k) computed directly
        h = 0
        for tid in token_ids[:k]:
            h = (h * _HASH_BASE + tid) % _HASH_MOD
        hashes[0] = h

        # Precompute B^(k-1) mod M to remove the oldest term in O(1)
        pow_k_minus_1 = pow(_HASH_BASE, k - 1, _HASH_MOD)
        needed = frozenset(full_starts)

        # Slide one position at a time; record only stride-aligned starts
        for start in range(1, n - k + 1):
            # Remove token_ids[start-1], add token_ids[start + k - 1]
            h = ((h - token_ids[start - 1] * pow_k_minus_1) * _HASH_BASE
                 + token_ids[start + k - 1]) % _HASH_MOD
            if start in needed:
                hashes[start] = h

    # --- Partial trailing windows (< window_size) ---
    # At most ``stride`` such windows; compute directly.
    for start in (s for s in stride_starts if s + window_size > n):
        h = 0
        for tid in token_ids[start:]:
            h = (h * _HASH_BASE + tid) % _HASH_MOD
        hashes[start] = h

    return hashes


def _hash_token_ids_sha256(token_ids: list[int]) -> str:
    """Compute a SHA-256 digest of a token sequence for Merkle leaf hashing."""
    h = hashlib.sha256()
    for tid in token_ids:
        h.update(struct.pack("!I", tid))
    return h.hexdigest()


def compute_prefix_hash(token_ids: list[int]) -> str:
    """Compute a single hash for the full prefix (used for exact-match lookups)."""
    h = hashlib.sha256()
    for tid in token_ids:
        h.update(struct.pack("!I", tid))
    return h.hexdigest()


@dataclass
class CachedPrefixInfo:
    """Metadata about a cached KV entry on a cluster."""
    prefix_hash: str
    prefix_length: int
    token_window_hash: int  # rolling hash of the window
    cluster_id: str
    node_id: str | None = None
    memory_bytes: int = 0


@dataclass
class RouterScore:
    """Routing decision for a single cluster."""
    cluster_id: str
    cache_affinity: float  # 0.0 (none) to 1.0 (perfect match)
    load_score: float       # 0.0 (idle) to 1.0 (overloaded)
    combined: float         # weighted combination
    matched_length: int = 0 # tokens of common cached prefix


class KVCacheDigest:
    """Computes a compact digest of cached KV prefixes for cross-cluster exchange.

    Uses sliding-window rolling hashes, wrapped in a SHA-256 Merkle tree for
    efficient ``diff()`` between clusters.
    """

    def __init__(self, window_size: int = 128):
        self.window_size = window_size

    def compute(self, token_ids: list[int]) -> dict[str, Any]:
        """Compute a full digest for a token sequence.

        Returns:
            dict with ``version``, ``hash`` (SHA-256 of all tokens),
            ``prefix_hash`` (rolling hash per window), ``length``, and
            ``window_size``.
        """
        return {
            "version": _DIGEST_VERSION,
            "hash": compute_prefix_hash(token_ids),
            "prefix_hash": _rolling_hash(token_ids, self.window_size),
            "length": len(token_ids),
            "window_size": self.window_size,
        }

    @staticmethod
    def similarity(
        prompt_digest: dict[str, Any],
        cached_digest: dict[str, Any],
    ) -> float:
        """Compute how much of the prompt prefix overlaps with a cached entry.

        Returns 0.0 (no match) to 1.0 (all cached tokens match the prompt prefix).
        """
        prompt_hashes = prompt_digest.get("prefix_hash", {})
        cached_hashes = cached_digest.get("prefix_hash", {})

        if not prompt_hashes or not cached_hashes:
            return 0.0

        prompt_len = prompt_digest.get("length", 0)
        if prompt_len == 0:
            return 0.0

        common = 0
        for start, ph in prompt_hashes.items():
            ch = cached_hashes.get(start)
            if ch is not None and ph == ch:
                common += 1

        return min(common / max(len(prompt_hashes), 1), 1.0)

    @staticmethod
    def longest_common_prefix_len(
        prompt_digest: dict[str, Any],
        cached_digest: dict[str, Any],
    ) -> int:
        """Find the longest continuous matching prefix length in tokens."""
        prompt_hashes = sorted(prompt_digest.get("prefix_hash", {}).items())
        cached_hashes = dict(cached_digest.get("prefix_hash", {}))

        window_size = prompt_digest.get("window_size", 128)
        stride = max(1, window_size // 4)
        matched_windows = 0

        for start, ph in prompt_hashes:
            ch = cached_hashes.get(start)
            if ch is not None and ph == ch:
                matched_windows += 1
            else:
                break

        if matched_windows == 0:
            return 0
        return min(matched_windows * stride + window_size, prompt_digest.get("length", 0))


class ContentRouter:
    """Routes requests based on KV cache affinity across federated clusters.

    Combines cache similarity scores with load metrics for the final routing
    decision.
    """

    def __init__(self, cache_weight: float = 0.6, load_weight: float = 0.4):
        """Args:
            cache_weight: How much to favor cache affinity over load (0.0–1.0).
            load_weight: How much to favor low load over cache (0.0–1.0).
        """
        self.cache_weight = cache_weight
        self.load_weight = load_weight

    def score_cluster(
        self,
        prompt_digest: dict[str, Any],
        cluster_digests: dict[str, dict[str, Any]],
        cluster_loads: dict[str, float],
    ) -> list[RouterScore]:
        """Score each cluster by combined cache affinity and load.

        Args:
            prompt_digest: Digest of the current request's prompt.
            cluster_digests: ``{cluster_id: digest}`` for cached prefixes.
            cluster_loads: ``{cluster_id: load_fraction}`` (0.0 idle, 1.0 overloaded).

        Returns:
            Sorted list of ``RouterScore``, highest combined first.
        """
        scores: list[RouterScore] = []

        for cid, cached in cluster_digests.items():
            affinity = KVCacheDigest.similarity(prompt_digest, cached)
            lcp = KVCacheDigest.longest_common_prefix_len(prompt_digest, cached)
            load = cluster_loads.get(cid, 1.0)
            combined = self.cache_weight * affinity + self.load_weight * (1.0 - load)

            scores.append(RouterScore(
                cluster_id=cid,
                cache_affinity=affinity,
                load_score=load,
                combined=combined,
                matched_length=lcp,
            ))

        scores.sort(key=lambda s: s.combined, reverse=True)
        return scores

    def route(
        self,
        prompt_digest: dict[str, Any],
        cluster_digests: dict[str, dict[str, Any]],
        cluster_loads: dict[str, float],
    ) -> str | None:
        """Select the best cluster for a request.

        Returns the cluster ID with the best combined score, or ``None`` if
        no clusters are available.
        """
        scores = self.score_cluster(prompt_digest, cluster_digests, cluster_loads)
        if not scores:
            return None
        return scores[0].cluster_id


class CacheDigestExchange:
    """Serializes/deserializes KV cache digests for cross-cluster heartbeat.

    Format: compact binary with version header, then per-entry:
    - cluster_id (length-prefixed string)
    - prefix_hash (SHA-256 hex, 64 bytes)
    - length (uint32)
    - window_hash_count (uint32)
    - window_hashes (each: uint32 start + uint64 hash = 12 bytes)
    """

    @staticmethod
    def serialize(digests: dict[str, dict[str, Any]]) -> bytes:
        """Pack cluster digest dict into compact binary."""
        parts = [struct.pack("!B", _DIGEST_VERSION)]

        for cid, digest in digests.items():
            cid_bytes = cid.encode("utf-8")
            prefix_hash = digest.get("hash", "").encode("utf-8")
            length = digest.get("length", 0)
            window_hashes = digest.get("prefix_hash", {})
            window_count = len(window_hashes)

            entry = struct.pack("!H", len(cid_bytes))
            entry += cid_bytes
            entry += struct.pack("!64s", prefix_hash.ljust(64, b"\x00")[:64])
            entry += struct.pack("!I", length)
            entry += struct.pack("!I", window_count)

            for start, wh in window_hashes.items():
                entry += struct.pack("!IQ", start, wh)

            parts.append(entry)

        return b"".join(parts)

    @staticmethod
    def deserialize(data: bytes) -> dict[str, dict[str, Any]]:
        """Unpack binary back into cluster digest dict."""
        if len(data) < 1:
            return {}

        version = struct.unpack("!B", data[0:1])[0]
        if version > _DIGEST_VERSION:
            logger.warning("Unknown digest version {}", version)
            return {}

        result: dict[str, dict[str, Any]] = {}
        offset = 1

        while offset < len(data):
            if offset + 2 > len(data):
                break
            (cid_len,) = struct.unpack("!H", data[offset:offset + 2])
            offset += 2

            if offset + cid_len > len(data):
                break
            cid = data[offset:offset + cid_len].decode("utf-8")
            offset += cid_len

            if offset + 64 > len(data):
                break
            raw_hash = data[offset:offset + 64]
            prefix_hash = raw_hash.rstrip(b"\x00").decode("utf-8")
            offset += 64

            if offset + 4 > len(data):
                break
            (length,) = struct.unpack("!I", data[offset:offset + 4])
            offset += 4

            if offset + 4 > len(data):
                break
            (wc,) = struct.unpack("!I", data[offset:offset + 4])
            offset += 4

            window_hashes: dict[int, int] = {}
            for _ in range(wc):
                if offset + 12 > len(data):
                    break
                start, wh = struct.unpack("!IQ", data[offset:offset + 12])
                window_hashes[start] = wh
                offset += 12

            result[cid] = {
                "version": version,
                "hash": prefix_hash,
                "prefix_hash": window_hashes,
                "length": length,
                "window_size": 128,
            }

        return result

    @staticmethod
    def build_merkle_digest(cluster_id: str, token_ids: list[int]) -> dict[str, Any]:
        """Build a full digest including SHA-256 leaf hashes for Merkle comparison.

        The Merkle tree is built over 128-token leaf blocks.  The root can be
        exchanged compactly; the full tree enables ``diff()`` between clusters.
        """
        from distllm.dist.merkle import MerkleTree

        leaves = []
        for i in range(0, len(token_ids), 128):
            block = token_ids[i:i + 128]
            leaves.append(_hash_token_ids_sha256(block))

        tree = MerkleTree(leaves)
        digest = KVCacheDigest(window_size=128).compute(token_ids)
        digest["merkle_root"] = tree.root
        digest["merkle_leaves"] = leaves
        digest["cluster_id"] = cluster_id
        return digest

    @staticmethod
    def diff_merkle(
        local_digest: dict[str, Any],
        remote_digest: dict[str, Any],
    ) -> list[int]:
        """Find differing leaf block indices between two Merkle digests.

        Returns indices of leaf blocks that differ (blocks the remote cluster
        has that the local one doesn't, or vice versa).
        """
        from distllm.dist.merkle import MerkleTree

        local_leaves = local_digest.get("merkle_leaves", [])
        remote_leaves = remote_digest.get("merkle_leaves", [])

        if not local_leaves or not remote_leaves:
            return list(range(max(len(local_leaves), len(remote_leaves))))

        local_tree = MerkleTree(local_leaves)
        remote_tree = MerkleTree(remote_leaves)

        diff_indices: set[int] = set()
        for idx in local_tree.diff(remote_tree):
            diff_indices.add(idx)
        return sorted(diff_indices)
