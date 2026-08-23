"""Starvation detection and aging boost for the batch scheduler.

Extracted from ``BatchScheduler`` in ``batch_scheduler.py`` to reduce
that file below the 800-line ceiling.

These are called with explicit scheduler state rather than ``self``,
so they remain pure functions that can be tested independently.
"""

from __future__ import annotations

import time
from loguru import logger


def check_starvation(
    pending_heap: list,
    starvation_threshold_s: float,
    last_starvation_warn: set[str],
) -> set[str]:
    """Check *pending_heap* for requests waiting too long.

    Samples the top of the heap (highest-priority items) to avoid O(n)
    copy on every schedule() call.  Logs a warning for any request
    exceeding the starvation threshold.

    Returns the updated set of starved request IDs.
    """
    now = time.time()
    current_starved: set[str] = set()
    sample_count = min(20, len(pending_heap))
    for i in range(sample_count):
        _pri, _cnt, seq = pending_heap[i]
        elapsed = now - seq.created_at
        if elapsed > starvation_threshold_s:
            current_starved.add(seq.request_id)
            if seq.request_id not in last_starvation_warn:
                logger.warning(
                    f"Request {seq.request_id} pending for {elapsed:.0f}s "
                    f"(priority={seq.priority}) -- possible starvation"
                )
    return current_starved


def aging_boost(
    created_at: float,
    aging_enabled: bool,
    aging_interval_s: float,
    aging_max_boost: int,
) -> int:
    """Calculate priority boost from aging (starvation prevention).

    The longer a request waits in the pending heap, the more its
    effective priority is boosted, ensuring low-priority requests
    are eventually served even under continuous high-priority load.
    """
    if not aging_enabled:
        return 0
    elapsed = time.time() - created_at
    boost = int(elapsed / aging_interval_s)
    return min(boost, aging_max_boost)


def priority_weight(priority_weights: dict[int, float], priority: int) -> float:
    """Return token allocation weight for a given priority level.

    Higher priority sequences get a larger share of the prefill
    token budget within a batch.
    """
    return priority_weights.get(priority, 1.0)
