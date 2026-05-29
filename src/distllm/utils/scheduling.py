"""Shared scheduling utilities used by batch_scheduler and profiler."""

import math


def group_by_length(
    sequences: list[object],
    num_buckets: int = 4,
) -> dict[int, list[object]]:
    """Group sequences by similar total length into log-scale buckets.

    Sequences with similar lengths are placed in the same bucket so they
    can be batched together, reducing ragged-attention overhead.

    Args:
        sequences: Objects with a ``total_len`` attribute (e.g. Sequence).
        num_buckets: Number of log-scale buckets to create.

    Returns:
        Dict mapping bucket index (0..num_buckets-1) to lists of sequences.
    """
    buckets: dict[int, list[object]] = {i: [] for i in range(num_buckets)}
    lengths = [s.total_len for s in sequences]
    if not lengths:
        return buckets
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        buckets[0] = list(sequences)
        return buckets
    log_min = math.log(max(min_len, 1))
    log_max = math.log(max_len)
    log_range = log_max - log_min
    for seq in sequences:
        ln = math.log(max(seq.total_len, 1))
        bucket = min(int((ln - log_min) / log_range * num_buckets), num_buckets - 1)
        buckets[bucket].append(seq)
    return buckets
