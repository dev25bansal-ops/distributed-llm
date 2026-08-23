"""Regression test for HIGH fix H13: radix-tree eviction was O(n^2).

The memory-budget eviction path in ``RadixTreeCache._evict_until_fit`` called
``_find_lru_leaf`` once per evicted leaf, re-scanning the whole tree on every
pass.  Evicting ``k`` leaves out of ``n`` therefore cost O(k * n) ~ O(n^2).

After the fix, all evictable leaves are collected in a SINGLE traversal (sorted
by recency) and removed in one batch -> O(n).

This test asserts:
  * the full-tree walk (``RadixNode._find_lru_leaf``) is invoked a *constant*
    number of times regardless of how many leaves are evicted (O(n), not O(n^2));
  * the cache ends up under capacity;
  * most-recently-used entries are retained.
"""

from __future__ import annotations

import functools
import time

import pytest

import distllm.core.radix_tree_cache as rtc
from distllm.core.radix_tree_cache import RadixNode, RadixTreeCache


class _FakeClock:
    """Monotonic, always-increasing clock so LRU ordering is deterministic."""

    def __init__(self):
        self._t = 0.0

    def __call__(self):
        self._t += 1.0
        return self._t


def _kv(elems: int):
    """KV data whose byte size the cache's estimator understands (dict wrap)."""
    import torch

    return {"t": torch.zeros(elems, dtype=torch.float32)}


def _make_cache_with_leaves(n: int, per_bytes: int, fake_clock: _FakeClock):
    """Build a cache holding ``n`` distinct leaves, each ~``per_bytes`` big.

    ``max_entries=0`` disables the entry-count eviction path, so the only
    eviction that can run is the memory-budget path (the one with the bug).
    """
    elems = max(1, per_bytes // 4)  # float32 == 4 bytes
    cache = RadixTreeCache(
        max_entries=0,
        min_prefix_len=1,
        memory_budget_bytes=10 ** 12,  # huge: no eviction while filling
    )
    for i in range(n):
        # distinct 2-token path per leaf so nothing shares a leaf
        cache.store([i, i + 1_000_000], _kv(elems))
    return cache


def _instrument_walk():
    """Patch ``RadixNode._find_lru_leaf`` to count full-tree walk invocations."""
    calls = {"n": 0}
    orig = RadixNode._find_lru_leaf

    @functools.wraps(orig)
    def _counting(self):
        calls["n"] += 1
        return orig(self)

    RadixNode._find_lru_leaf = _counting
    return calls, orig


def test_h13_radix_eviction_single_traversal():
    fake = _FakeClock()
    real_time = time.time
    time.time = fake
    try:
        n = 1000
        per = 400  # bytes per leaf (100 float32 in the dict-wrapped tensor)
        cache = _make_cache_with_leaves(n, per, fake)

        # Mark the last 10 leaves as most-recently-used under the fake clock.
        mru = list(range(n - 10, n))
        for i in mru:
            cache.lookup([i, i + 1_000_000])
    finally:
        time.time = real_time

    calls, orig = _instrument_walk()
    try:
        # Shrink the budget hard -> forces eviction of ~all leaves.
        cache.adjust_memory_budget(12 * per)
    finally:
        RadixNode._find_lru_leaf = orig

    remaining = cache.stats()["prefix_cache_entries"]
    evicted = n - remaining

    # We forced a large eviction.
    assert evicted > n * 0.5, f"expected a large eviction, got {evicted}"

    # CORE REGRESSION ASSERTION: the tree walk must run a *constant* number of
    # times, not once per evicted leaf.  A per-leaf loop would call it ~evicted
    # times (~988 here); the fixed single-traversal path calls it 0 times.
    assert calls["n"] <= 2, (
        f"eviction re-walked the tree {calls['n']} times while evicting "
        f"{evicted} leaves -- O(n^2) bug not fixed (expected <= 2)"
    )

    # Capacity invariant: we kept at most ~12 leaves (12 * per bytes).
    assert remaining <= 14, f"cache exceeded capacity: {remaining} leaves"

    # Recency invariant: the MRU leaves must have been retained.
    for i in mru:
        match_len, _ = cache.lookup([i, i + 1_000_000])
        assert match_len == 2, f"MRU leaf {i} was wrongly evicted"


def test_h13_entry_count_eviction_single_traversal():
    """The entry-count eviction path must also be single-traversal (M6)."""
    fake = _FakeClock()
    real_time = time.time
    time.time = fake
    try:
        n = 500
        cache = RadixTreeCache(
            max_entries=20,  # keep only 20 of 500 entries
            min_prefix_len=1,
            memory_budget_bytes=10 ** 12,
        )
        for i in range(n):
            cache.store([i, i + 2_000_000], _kv(50))
        mru = list(range(n - 10, n))
        for i in mru:
            cache.lookup([i, i + 2_000_000])
    finally:
        time.time = real_time

    calls, orig = _instrument_walk()
    try:
        # Force the entry-count path to run again.
        cache.adjust_memory_budget(10 ** 12)
    finally:
        RadixNode._find_lru_leaf = orig

    remaining = cache.stats()["prefix_cache_entries"]
    assert remaining <= 20, f"entry-count eviction exceeded capacity: {remaining}"
    assert calls["n"] <= 2, (
        f"entry-count eviction re-walked the tree {calls['n']} times -- O(n^2)"
    )
    for i in mru:
        match_len, _ = cache.lookup([i, i + 2_000_000])
        assert match_len == 2, f"MRU leaf {i} wrongly evicted (count path)"


if __name__ == "__main__":
    test_h13_radix_eviction_single_traversal()
    test_h13_entry_count_eviction_single_traversal()
    print("OK")
