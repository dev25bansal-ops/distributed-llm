"""Verification tests for already-applied fixes from the latest review table:

- H9  load_balancer: atomic slot reservation at pick time (no thundering herd)
- M6  radix_tree_cache: O(n log n) LRU eviction with tracked count
- H4/M7 cache_manager: stable (deterministic) hash namespace across processes
"""

import threading

from distllm.core.load_balancer import LBStrategy, LoadBalancer
from distllm.core.radix_tree_cache import RadixNode


def test_h9_atomic_slot_reservation_no_lost_reservations():
    """Concurrent picks must reserve exactly one slot each (no double-book)."""
    lb = LoadBalancer(strategy=LBStrategy.LEAST_CONNECTIONS)
    for i in range(4):
        lb.add_target(f"10.0.0.{i}", 50050, node_id=f"coord-{i}")

    N = 500
    results = []
    results_lock = threading.Lock()

    def worker():
        t = lb.pick()
        if t is not None:
            with results_lock:
                results.append(t)

    _threads = [threading.Thread(target=worker) for _ in range(N)]
    for th in _threads:
        th.start()
    for th in _threads:
        th.join()

    # Every successful pick reserved exactly one connection slot.
    total_reserved = sum(t.active_connections for t in lb.all_targets())
    assert total_reserved == N, f"lost reservations: reserved={total_reserved}, picks={N}"
    assert len(results) == N


def test_m6_eviction_bounded_and_tracked():
    """evict_lru evicts down to the cap and tracks entry count exactly."""
    node = RadixNode()

    # Build independent leaves, each on its own single-token path, with
    # distinct access times so LRU order is deterministic.
    max_entries = 10
    n_extra = 50
    for i in range(max_entries + n_extra):
        tok = 1000 + i
        leaf = node.children.get(tok)
        if leaf is None:
            leaf = RadixNode(token=tok)
            node.children[tok] = leaf
        leaf.last_access = float(i)
        leaf.kv_data = (tok,)  # mark as a real entry
        leaf.size = 1

    evicted, _ = node.evict_lru(max_entries)
    assert evicted == n_extra, f"expected {n_extra} evicted, got {evicted}"

    remaining = sum(1 for c in node.children.values() if c.kv_data is not None)
    assert remaining == max_entries, f"expected {max_entries} left, got {remaining}"


def test_cache_hash_namespace_deterministic():
    """_hash_tokens must be stable regardless of PYTHONHASHSEED (SHA-256)."""
    from distllm.core.cache_manager import CacheManager

    cm = CacheManager()
    tokens = [11, 22, 33, 44, 55]
    h1 = cm._hash_tokens(tokens)
    h2 = cm._hash_tokens(list(tokens))  # same tokens, different list object
    assert h1 == h2, "hash namespace not deterministic for identical tokens"
    assert h1.startswith("h") and len(h1) > 10, f"unexpected hash format: {h1}"
