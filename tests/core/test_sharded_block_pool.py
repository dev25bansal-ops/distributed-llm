"""Enhancement #2: sharded PagedAttention block pool tests.

16 sequence-keyed RLock shards replace the single global mutex so
different sequences allocate/append/free concurrently. Free-list-mutating
ops additionally hold the pool lock. These tests pin correctness under
concurrency and verify the concurrency actually happens.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from distllm.backends.paged_attention import PagedAttentionManager


def _mgr(**kw):
    return PagedAttentionManager(
        num_blocks=kw.get("num_blocks", 512),
        block_size=16,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        device="cpu",
    )


def test_shards_exist_and_distinct():
    m = _mgr()
    assert len(m._lock_shards) == 16
    # Two different sequence ids usually land on different shards.
    shards = {m._shard_for(f"seq-{i}") for i in range(64)}
    assert len(shards) > 1, "expected more than one distinct shard across 64 ids"


def test_concurrent_allocations_are_consistent():
    """32 threads x 8 sequences each: no double-allocation, no leaks."""
    m = _mgr(num_blocks=512)
    errors: list[str] = []
    per_thread_blocks: dict[int, set[int]] = {}

    def worker(tid: int):
        try:
            for s in range(8):
                sid = f"t{tid}-s{s}"
                bids = m.allocate_sequence(sid, num_tokens=16)
                per_thread_blocks.setdefault(tid, set()).update(bids)
        except Exception as e:  # pragma: no cover
            errors.append(f"t{tid}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Every block handed out exactly once across all threads
    all_blocks: list[int] = []
    for blocks in per_thread_blocks.values():
        all_blocks.extend(blocks)
    assert len(all_blocks) == len(set(all_blocks)), "double allocation detected"
    # 32 threads x 8 seqs x 1 block = 256 blocks used
    assert m.num_used_blocks == 256


def test_concurrent_alloc_and_free_no_leak():
    """Interleaved alloc/free across threads returns the pool to baseline."""
    m = _mgr(num_blocks=256)
    baseline_free = m.num_free_blocks

    def worker(tid: int):
        for i in range(20):
            sid = f"t{tid}-{i}"
            m.allocate_sequence(sid, num_tokens=16)
            m.free_sequence(sid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.num_free_blocks == baseline_free


def test_concurrent_append_tokens():
    """append_token under concurrency: block counts stay exact."""
    m = _mgr(num_blocks=512)

    def worker(tid: int):
        sid = f"seq-{tid}"
        m.allocate_sequence(sid, num_tokens=16)
        for _ in range(32):  # 2 more blocks each
            m.append_token(sid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Lazy allocation contract: a block counts as used on allocation; the
    # append-after-full allocation of block N+1 happens inside the same
    # append_token() call. 16 init + 32 appends = 48 tokens = 3 blocks per
    # seq (16 in block 1; tokens 17-32 fill block 2; token 33 opens block 3,
    # tokens 34-48 fill it). 16 seqs x 3 blocks = 48.
    assert m.num_used_blocks == 16 * 3


def test_cow_under_concurrency():
    """copy_on_write across sequences holds both shards; no deadlock."""
    m = _mgr(num_blocks=256)
    m.allocate_sequence("src", num_tokens=16)

    stop = threading.Event()
    errors: list[str] = []

    def copier():
        try:
            while not stop.is_set():
                m.copy_on_write("src", "dst")
                m.free_sequence("dst")
        except Exception as e:  # pragma: no cover
            errors.append(str(e))

    t = threading.Thread(target=copier)
    t.start()
    import time

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        m.append_token("src")  # contends with the COW loop
        if m.num_free_blocks == 0:
            break
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive(), "COW thread deadlocked"
    assert not errors, errors


def test_shard_locks_allow_parallel_holders():
    """Two threads on DIFFERENT shards can hold locks simultaneously."""
    m = _mgr()
    held = []
    barrier = threading.Barrier(2, timeout=10)

    def grab(sid: str):
        with m._shard_for(sid):
            barrier.wait()  # both must reach here => locks held concurrently
            held.append(sid)

    # Find two ids on distinct shards
    a = "aaa"
    b = next(x for x in (f"b{i}" for i in range(1000)) if m._shard_for(x) is not m._shard_for(a))
    t1 = threading.Thread(target=grab, args=(a,))
    t2 = threading.Thread(target=grab, args=(b,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(held) == 2, "barrier timeout — different shards serialized (bug)"
