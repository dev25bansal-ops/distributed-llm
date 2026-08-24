"""Regression tests for verified High bug C10: radix-cache eviction stall.

The old eviction cleared ``kv_data`` but left the dead leaf attached to the
tree. Because ``_find_lru_leaf`` only selects childless leaves that hold data,
a data-bearing ancestor buried under a dead child chain became permanently
invisible to eviction, so ``max_entries`` / the byte budget were silently
exceeded and memory grew without bound. Overwrites also added the new value's
bytes without subtracting the replaced value's, inflating the memory counter
monotonically, and ``_count_entries`` was a full O(n) recursion run twice per
eviction-loop iteration.

These tests pin the fixed behavior:
  * dead chains are pruned upward after every removal;
  * shared-prefix floods stay within both the entry cap and byte budget;
  * repeated overwrites leave the size/memory counters exactly accurate;
  * eviction keeps finding victims after deep-chain evictions;
  * LRU order decides who survives;
  * the running entry counter stays consistent with a brute-force recount.
"""

from __future__ import annotations

import random

import pytest
import torch

from distllm.core.radix_tree_cache import RadixNode, RadixTreeCache


def _kv(elems: int):
    """KV payload the cache's estimator can measure: dict-wrapped float32."""
    return {"t": torch.zeros(elems, dtype=torch.float32)}


def _entry_bytes(elems: int) -> int:
    return elems * 4


def _walk_to(root: RadixNode, token_ids: list[int]) -> RadixNode | None:
    node = root
    for tok in token_ids:
        node = node.children.get(tok)
        if node is None:
            return None
    return node


def _assert_no_dead_nodes(root: RadixNode) -> None:
    """No node may be dataless AND childless except the root itself."""
    stack: list[RadixNode] = [root]
    while stack:
        node = stack.pop()
        if node is not root:
            assert node.kv_data is not None or node.children, (
                f"dead leaf left attached at token {node.token} "
                "(C10 stall regression)"
            )
        stack.extend(node.children.values())


def _bruteforce_entry_count(node: RadixNode) -> int:
    total = 1 if node.kv_data is not None else 0
    for child in node.children.values():
        total += _bruteforce_entry_count(child)
    return total


class TestSharedPrefixFloodWithinCaps:
    """(a) Inserting max_entries+N shared-prefix prompts must stay bounded."""

    def test_count_and_budget_respected(self):
        max_entries = 8
        per = _entry_bytes(64)
        budget = max_entries * per
        cache = RadixTreeCache(
            max_entries=max_entries,
            min_prefix_len=1,
            memory_budget_bytes=budget,
        )
        base = list(range(16))  # long shared prefix across all prompts
        for i in range(max_entries + 10):
            cache.store(base + [1000 + i], _kv(64))
            snapshot = cache.stats()
            assert snapshot["prefix_cache_entries"] <= max_entries
            assert snapshot["prefix_cache_memory_bytes"] <= budget

        _assert_no_dead_nodes(cache._root)

        # The most recently stored prompt must still be retrievable.
        matched, kv = cache.lookup(base + [1000 + max_entries + 9])
        assert matched == len(base) + 1
        assert kv is not None

    def test_memory_budget_enforced_per_store_with_shared_prefixes(self):
        per = _entry_bytes(64)
        cache = RadixTreeCache(
            min_prefix_len=1, memory_budget_bytes=4 * per, max_entries=0
        )
        base = list(range(10))
        for i in range(30):
            cache.store(base + [i], _kv(64))
            assert cache.stats()["prefix_cache_memory_bytes"] <= 4 * per
        _assert_no_dead_nodes(cache._root)


class TestOverwriteAccounting:
    """(b) Overwriting the same prompt must keep the counters exact."""

    def test_repeated_overwrite_same_prompt(self):
        cache = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=10**9)
        toks = [7, 7, 7]
        for _ in range(50):
            cache.store(toks, _kv(64))

        stats = cache.stats()
        assert stats["prefix_cache_entries"] == 1
        # Exactly one entry's worth of bytes -- not 50x accumulated.
        assert stats["prefix_cache_memory_bytes"] == _entry_bytes(64)

    def test_overwrite_between_different_sizes(self):
        cache = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=10**9)
        toks = [1, 2]

        cache.store(toks, _kv(64))
        assert cache.stats()["prefix_cache_memory_bytes"] == _entry_bytes(64)

        bigger = {
            "t": torch.zeros(128, dtype=torch.float32),
            "u": torch.zeros(32, dtype=torch.float32),
        }  # 512 + 128 = 640 bytes
        cache.store(toks, bigger)
        assert cache.stats()["prefix_cache_memory_bytes"] == 640

        cache.store(toks, _kv(64))
        assert cache.stats()["prefix_cache_memory_bytes"] == _entry_bytes(64)
        assert cache.stats()["prefix_cache_entries"] == 1

    def test_overwrite_does_not_inflate_memory_across_distinct_keys(self):
        cache = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=10**9)
        for round_ in range(25):
            for i in range(10):
                cache.store([i], _kv(64))
        # 10 distinct keys overwritten 25 times: exactly 10 entries of bytes.
        stats = cache.stats()
        assert stats["prefix_cache_entries"] == 10
        assert stats["prefix_cache_memory_bytes"] == 10 * _entry_bytes(64)


class TestEvictionAfterDeepChain:
    """(c) Victims must remain findable after evictions down shared chains."""

    def test_remove_entry_prunes_dead_chain_upward(self):
        root = RadixNode()
        root.insert([1, 2, 3], "x")
        root.insert([9, 9], "y")

        node3 = root.children[1].children[2].children[3]
        assert node3._remove_entry() == "x"
        # Entire now-dead branch detached, sibling untouched.
        assert 1 not in root.children
        assert root.children[9].children[9].kv_data == "y"
        assert root._count_entries() == 1
        assert node3._remove_entry() is None  # idempotent

    def test_lru_leaf_found_after_deep_chain_pruned(self):
        root = RadixNode()
        root.insert([1, 2, 3, 4, 5], "deep")
        root.insert([1, 2, 3], "mid")  # data-bearing ancestor of the leaf

        deep = _walk_to(root, [1, 2, 3, 4, 5])
        deep.last_access = 100.0

        root.evict_lru(max_entries=1)

        # Dead chain below "mid" pruned -> "mid" is now a selectable leaf.
        mid = _walk_to(root, [1, 2, 3])
        assert mid is not None
        assert mid.kv_data == "mid"
        assert mid.children == {}
        best_time, victim = root._find_lru_leaf()
        assert victim is mid

    def test_nested_prefix_cache_keeps_evicting_after_deep_chain(self):
        """End-to-end repro of the C10 stall: nested prefixes then a flood."""
        cache = RadixTreeCache(
            max_entries=1, min_prefix_len=1, memory_budget_bytes=10**9
        )
        cache.store([1, 2, 3, 4, 5], "deep")
        deep = _walk_to(cache._root, [1, 2, 3, 4, 5])
        deep.last_access = 100.0

        cache.store([1, 2, 3], "mid")
        assert cache.stats()["prefix_cache_entries"] == 1
        matched, kv = cache.lookup([1, 2, 3])
        assert matched == 3 and kv == "mid"

        # Under the old bug the tree was now stuck: every further insert
        # pushed the count past the cap with no victim ever found again.
        for i in range(20):
            cache.store([9000, i], f"new{i}")
            assert cache.stats()["prefix_cache_entries"] <= 1
            _assert_no_dead_nodes(cache._root)

        matched, kv = cache.lookup([9000, 19])
        assert matched == 2 and kv == "new19"


class TestLRUOrdering:
    """(d) Eviction must follow last-access order exactly."""

    def test_oldest_leaves_evicted_newest_retained(self):
        cache = RadixTreeCache(
            max_entries=0, min_prefix_len=1, memory_budget_bytes=10**9
        )
        toks = [10, 20, 30, 40, 50, 60]
        # Fill with no cap so nothing is pruned while we pin recency.
        for i, tok in enumerate(toks):
            cache.store([tok], f"v{i}")
        for i, tok in enumerate(toks[:-1]):
            cache._root.children[tok].last_access = float(i)
        cache._root.children[60].last_access = 100.0

        cache._max_entries = 3
        cache._evict_until_fit()

        survivors = {tok for tok in toks if tok in cache._root.children}
        assert survivors == {40, 50, 60}

        for tok in (10, 20, 30):
            matched, _ = cache.lookup([tok])
            assert matched == 0, f"stale token {tok} survived eviction"
        for tok in (40, 50, 60):
            matched, kv = cache.lookup([tok])
            assert matched == 1 and kv is not None

    def test_lookup_touch_refreshes_protection(self):
        cache = RadixTreeCache(
            max_entries=0, min_prefix_len=1, memory_budget_bytes=10**9
        )
        for tok in (1, 2, 3, 4):
            cache.store([tok], f"v{tok}")
        cache._root.children[1].last_access = 10.0
        cache._root.children[2].last_access = 20.0
        cache._root.children[3].last_access = 30.0
        cache._root.children[4].last_access = 100.0

        cache.lookup([1])  # touch the oldest -> it becomes protected
        cache._root.children[1].last_access = 99.0

        cache._max_entries = 2
        cache._evict_until_fit()

        survivors = {tok for tok in (1, 2, 3, 4) if tok in cache._root.children}
        assert survivors == {1, 4}


class TestRunningCounterConsistency:
    """Requirement 3: the maintained counter must match a brute-force recount."""

    def test_mixed_ops_keep_counters_exact(self):
        rng = random.Random(1234)
        cache = RadixTreeCache(
            max_entries=25, min_prefix_len=1, memory_budget_bytes=10**9
        )

        def expected_memory(node: RadixNode) -> int:
            total = 0
            stack = [node]
            while stack:
                n = stack.pop()
                if n.kv_data is not None:
                    total += cache._estimate_entry_memory(n.kv_data)
                stack.extend(n.children.values())
            return total

        keys: list[list[int]] = []
        for step in range(400):
            roll = rng.random()
            key = [rng.randrange(500) for _ in range(rng.randint(1, 3))]
            if roll < 0.70 or not keys:
                cache.store(key, _kv(rng.choice([16, 32])))
                keys.append(key)
            elif roll < 0.85:
                cache.evict(key)
            elif roll < 0.95:
                cache.lookup(key)
            else:
                victim = rng.choice(keys)
                cache.evict(victim)

            if step % 50 == 0:
                stats = cache.stats()
                assert stats["radix_tree_nodes"] >= 1
                assert cache._root._count_entries() == _bruteforce_entry_count(
                    cache._root
                )
                assert stats["prefix_cache_memory_bytes"] == expected_memory(
                    cache._root
                )
                _assert_no_dead_nodes(cache._root)

        assert cache._root._count_entries() == _bruteforce_entry_count(
            cache._root
        )
        assert cache.stats()["prefix_cache_memory_bytes"] == expected_memory(
            cache._root
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
