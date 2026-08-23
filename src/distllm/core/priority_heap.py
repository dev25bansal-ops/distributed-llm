"""Priority heap operations for the batch scheduler.

Extracted from ``BatchScheduler`` in ``batch_scheduler.py``.
"""

from __future__ import annotations

import heapq
from typing import Any


def promote_request(
    pending_heap: list,
    request_id: str,
    new_priority: int,
    pending_index: dict[str, int] | None,
) -> tuple[bool, dict[str, int] | None]:
    """Change the priority of a pending request.

    Uses O(log n) indexed heap update instead of O(n) linear scan + heapify.

    Args:
        pending_heap: The heap of pending (priority, counter, sequence) tuples.
        request_id: The request to promote.
        new_priority: The new priority level.
        pending_index: Cached index mapping request_id → heap position (or None).

    Returns:
        Tuple of (success, updated_index) where updated_index is None when
        the index was invalidated (caller should rebuild on next access).
    """
    # Build index if not cached
    if pending_index is None:
        pending_index = {
            seq.request_id: i
            for i, (_, _, seq) in enumerate(pending_heap)
        }

    idx = pending_index.get(request_id)
    if idx is None:
        return False, pending_index

    _pri, _cnt, seq = pending_heap[idx]
    seq.priority = new_priority
    pending_heap[idx] = (new_priority, _cnt, seq)

    # Bubble up or down -- O(log n) instead of O(n) heapify.
    if new_priority < _pri:
        heapq._siftdown(pending_heap, 0, idx)
    else:
        heapq._siftup(pending_heap, idx)

    # Invalidate the index cache — after sifting, positions are stale.
    return True, None


def rebuild_pending_index(pending_heap: list) -> dict[str, int]:
    """Rebuild the request_id → heap-position index from scratch."""
    return {
        seq.request_id: i
        for i, (_, _, seq) in enumerate(pending_heap)
    }
