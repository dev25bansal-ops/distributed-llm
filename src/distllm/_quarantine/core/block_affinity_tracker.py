"""Block-to-request affinity tracking for smarter CoW sharing.

Tracks which blocks belong to which request, request metadata,
and access patterns.  Enables smarter copy-on-write sharing across
requests with similar prefixes.

Usage::

    from distllm.core.block_affinity_tracker import BlockAffinityTracker

    tracker = BlockAffinityTracker()
    tracker.register("req-1", block_ids=[0, 1, 2], prefix_hash="abc123")
    tracker.register("req-2", block_ids=[3, 4, 5], prefix_hash="abc123")

    # Find requests sharing the same prefix
    siblings = tracker.find_prefix_siblings("abc123")
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestInfo:
    """Metadata for a tracked request."""
    request_id: str
    block_ids: list[int]
    prefix_hash: str = ""
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    adapter_id: str | None = None
    priority: int = 2


class BlockAffinityTracker:
    """Tracks block-to-request mappings and prefix sharing relationships.

    Enables:
    - Finding all requests sharing a prefix (CoW candidates)
    - Tracking which blocks are shared vs unique
    - Computing sharing ratios for memory estimation
    - Identifying hot prefixes worth caching
    """

    def __init__(self) -> None:
        self._requests: dict[str, RequestInfo] = {}
        self._block_to_requests: dict[int, set[str]] = defaultdict(set)
        self._prefix_to_requests: dict[str, set[str]] = defaultdict(set)
        self._shared_blocks: dict[int, int] = {}  # block_id -> ref_count

    def register(
        self,
        request_id: str,
        block_ids: list[int],
        prefix_hash: str = "",
        adapter_id: str | None = None,
        priority: int = 2,
    ) -> None:
        """Register a request and its block allocation."""
        info = RequestInfo(
            request_id=request_id,
            block_ids=list(block_ids),
            prefix_hash=prefix_hash,
            adapter_id=adapter_id,
            priority=priority,
        )
        self._requests[request_id] = info

        for bid in block_ids:
            self._block_to_requests[bid].add(request_id)
            self._shared_blocks[bid] = len(self._block_to_requests[bid])

        if prefix_hash:
            self._prefix_to_requests[prefix_hash].add(request_id)

    def unregister(self, request_id: str) -> None:
        """Remove a request and update block tracking."""
        info = self._requests.pop(request_id, None)
        if info is None:
            return

        for bid in info.block_ids:
            reqs = self._block_to_requests.get(bid)
            if reqs:
                reqs.discard(request_id)
                if not reqs:
                    del self._block_to_requests[bid]
                    self._shared_blocks.pop(bid, None)
                else:
                    self._shared_blocks[bid] = len(reqs)

        if info.prefix_hash:
            prefix_set = self._prefix_to_requests.get(info.prefix_hash)
            if prefix_set:
                prefix_set.discard(request_id)
                if not prefix_set:
                    del self._prefix_to_requests[info.prefix_hash]

    def record_access(self, request_id: str) -> None:
        """Record an access to a request."""
        info = self._requests.get(request_id)
        if info:
            info.last_access = time.time()
            info.access_count += 1

    def find_prefix_siblings(self, prefix_hash: str) -> list[str]:
        """Find all request IDs sharing the same prefix hash."""
        return list(self._prefix_to_requests.get(prefix_hash, set()))

    def get_shared_blocks(self) -> list[int]:
        """Return block IDs shared by 2+ requests (CoW candidates)."""
        return [bid for bid, count in self._shared_blocks.items() if count >= 2]

    def get_unique_blocks(self, request_id: str) -> list[int]:
        """Return block IDs unique to a specific request."""
        info = self._requests.get(request_id)
        if info is None:
            return []
        return [bid for bid in info.block_ids if self._shared_blocks.get(bid, 0) <= 1]

    def sharing_ratio(self, request_id: str) -> float:
        """Fraction of blocks shared with other requests."""
        info = self._requests.get(request_id)
        if info is None or not info.block_ids:
            return 0.0
        shared = sum(1 for bid in info.block_ids if self._shared_blocks.get(bid, 0) >= 2)
        return shared / len(info.block_ids)

    def hot_prefixes(self, min_requests: int = 2) -> list[tuple[str, int]]:
        """Return prefix hashes shared by at least min_requests requests."""
        return [
            (prefix, len(reqs))
            for prefix, reqs in self._prefix_to_requests.items()
            if len(reqs) >= min_requests
        ]

    def request_info(self, request_id: str) -> RequestInfo | None:
        """Get metadata for a request."""
        return self._requests.get(request_id)

    def stats(self) -> dict[str, Any]:
        total_blocks = len(self._block_to_requests)
        shared = len(self.get_shared_blocks())
        return {
            "tracked_requests": len(self._requests),
            "tracked_blocks": total_blocks,
            "shared_blocks": shared,
            "unique_blocks": total_blocks - shared,
            "prefix_groups": len(self._prefix_to_requests),
        }

    def __repr__(self) -> str:
        shared = len(self.get_shared_blocks())
        return (
            f"BlockAffinityTracker(requests={len(self._requests)}, "
            f"blocks={len(self._block_to_requests)}, shared={shared})"
        )
