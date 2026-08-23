"""Property-based tests: partitioner sum-of-shards invariant.

The real partitioner is ``best_fit_decreasing_partition`` in
``distllm.core.auto_partitioner`` -- a memory-aware (VRAM-capped) bin-packer
that places model *layers* onto *devices*.

Invariant under test (when every layer fits within total capacity): the
placement produced by ``best_fit_decreasing_partition`` covers every input
layer exactly once -- no layer is lost and no layer is duplicated.  That is
the shard-sum property: ``union(buckets) == set(all_layer_indices)`` and
``sum(len(bucket) for bucket in buckets) == len(layer_bytes)``.

To avoid the ``ValueError`` raised on a true OOM (a layer that fits nowhere),
we *shrink* the per-layer footprints to a fraction of the smallest device cap,
guaranteeing feasibility while still exercising arbitrary counts of layers
(N) and devices (K).  K >= 1 is required.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.auto_partitioner import best_fit_decreasing_partition


_PBT_SETTINGS = dict(max_examples=30, deadline=None)


@settings(**_PBT_SETTINGS)
@given(
    n_layers=st.integers(0, 40),
    k_devices=st.integers(1, 10),
    seed=st.integers(0, 2**31 - 1),
)
def test_partitioner_sum_of_shards(n_layers, k_devices, seed):
    import random

    rng = random.Random(seed)

    # Device VRAM caps in bytes (1 GB .. 80 GB).  K >= 1 guaranteed.
    caps = {f"dev{i}": rng.randint(1 * 1024**3, 80 * 1024**3) for i in range(k_devices)}

    # Feasibility guarantee (avoid the legitimate OOM ValueError): every layer
    # fits within the SMALLEST device's headroom-reserved slot and the TOTAL
    # layer footprint fits within the total usable capacity.  We bound each
    # layer to at most min_cap*headroom / max(n_layers, 1):
    #   * each layer <= min_cap*headroom  -> fits the smallest device
    #   * sum of layers   <= n_layers * (min_cap*headroom / n_layers)
    #                     = min_cap*headroom <= total usable (>= K*min_cap*headroom)
    # so the whole partition is always feasible regardless of N and K.
    headroom = 0.10
    min_cap = min(caps.values())
    max_layer = max(1, int(0.9 * min_cap / max(n_layers, 1)))
    layer_bytes = [rng.randint(1, max_layer) for _ in range(n_layers)]

    buckets = best_fit_decreasing_partition(dict(caps), layer_bytes)

    # 1) Sum-of-shards: total placed layers == number of input layers.
    placed_total = sum(len(v) for v in buckets.values())
    assert placed_total == n_layers, (
        f"layer loss/dup: placed {placed_total} != input {n_layers}"
    )

    # 2) Every input layer index appears exactly once (no loss, no dup).
    all_placed = sorted(idx for v in buckets.values() for idx in v)
    assert all_placed == list(range(n_layers)), (
        f"layer set mismatch: {all_placed} != range({n_layers})"
    )

    # 3) Used VRAM never exceeds the usable (headroom-reserved) capacity.
    headroom = 0.10
    for dev, idxs in buckets.items():
        used = sum(layer_bytes[i] for i in idxs)
        assert used <= caps[dev] * (1.0 - headroom) + 1
