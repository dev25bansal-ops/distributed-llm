"""Concurrency stress tests for PagedAttention.

Run: pytest tests/stress/test_paged_attention_concurrent.py -v --timeout=60
"""

import concurrent.futures
import random
import threading

import pytest
import torch

from distllm.backends.paged_attention import PagedAttentionManager


@pytest.fixture
def pam():
    return PagedAttentionManager(
        num_blocks=512, block_size=8,
        num_layers=2, num_heads=2, head_dim=4,
        device="cpu",
    )


class TestConcurrentAllocation:
    def test_parallel_allocate_free(self, pam):
        """16 threads allocating and freeing sequences concurrently."""
        errors = []
        counter = threading.atomic = 0
        lock = threading.Lock()

        def worker(thread_id):
            nonlocal counter
            for i in range(50):
                with lock:
                    sid = f"t{thread_id}-s{i}"
                    counter += 1
                try:
                    pam.allocate_sequence(sid, num_tokens=random.randint(8, 64))
                    pam.free_sequence(sid)
                except RuntimeError:
                    pass  # pool exhaustion is OK
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(worker, tid) for tid in range(16)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        assert pam.num_free_blocks + pam.num_used_blocks == 512

    def test_concurrent_allocate_no_corruption(self, pam):
        """Allocate from multiple threads, verify no double-allocation."""
        allocated = []
        alloc_lock = threading.Lock()

        def worker():
            for i in range(20):
                sid = f"seq-{threading.current_thread().name}-{i}"
                try:
                    bids = pam.allocate_sequence(sid, num_tokens=16)
                    with alloc_lock:
                        allocated.extend(bids)
                except RuntimeError:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker) for _ in range(8)]
            concurrent.futures.wait(futures)

        # No duplicate block IDs
        assert len(allocated) == len(set(allocated))

    def test_concurrent_cow(self, pam):
        """Copy-on-write from multiple threads."""
        pam.allocate_sequence("src", num_tokens=32)
        errors = []

        def worker(tid):
            try:
                pam.copy_on_write("src", f"dst-{tid}")
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, tid) for tid in range(8)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        # All should share the same blocks
        src_table = pam.get_block_table("src")
        for tid in range(8):
            dst_table = pam.get_block_table(f"dst-{tid}")
            assert dst_table == src_table


class TestConcurrentSwap:
    def test_concurrent_swap_restore(self, pam):
        """Multiple threads swapping and restoring simultaneously."""
        for i in range(8):
            pam.allocate_sequence(f"s{i}", num_tokens=16)

        errors = []

        def worker(tid):
            try:
                pam.swap_blocks_to_cpu(f"s{tid}")
                pam.swap_blocks_to_gpu(f"s{tid}")
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, tid) for tid in range(8)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0


class TestInvariantUnderStress:
    def test_invariant_after_random_ops(self, pam):
        """Random allocate/free/append for 1000 ops, verify invariants."""
        active = set()
        errors = []

        for i in range(1000):
            op = random.choice(["alloc", "free", "append"])
            sid = f"seq-{random.randint(0, 50)}"

            try:
                if op == "alloc" and sid not in active:
                    pam.allocate_sequence(sid, num_tokens=random.randint(8, 64))
                    active.add(sid)
                elif op == "free" and sid in active:
                    pam.free_sequence(sid)
                    active.discard(sid)
                elif op == "append" and sid in active:
                    pam.append_token(sid)
            except RuntimeError:
                pass  # pool exhaustion
            except Exception as e:
                errors.append(e)

        # Clean up
        for sid in list(active):
            try:
                pam.free_sequence(sid)
            except Exception:
                pass

        assert len(errors) == 0
        assert pam.num_free_blocks + pam.num_used_blocks == 512

    def test_ref_count_invariant(self, pam):
        """After all sequences are freed, all ref_counts should be 0."""
        for i in range(20):
            pam.allocate_sequence(f"s{i}", num_tokens=32)

        for i in range(20):
            pam.free_sequence(f"s{i}")

        for block in pam._blocks:
            assert block.ref_count == 0
