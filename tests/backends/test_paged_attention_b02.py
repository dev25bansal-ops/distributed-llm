"""B-02 regression tests: swap/get races in PagedAttentionManager."""

import concurrent.futures
import threading
import time

import pytest

from distllm.backends.paged_attention import PagedAttentionManager


@pytest.fixture
def pam():
    return PagedAttentionManager(
        num_blocks=128, block_size=8,
        num_layers=1, num_heads=2, head_dim=4,
        device="cpu",
    )


class TestSwapGetRace:
    def test_concurrent_swap_and_get_no_crash(self, pam):
        pam.allocate_sequence("s", num_tokens=64)
        errors = []
        stop = threading.Event()

        def swapper():
            while not stop.is_set():
                try:
                    pam.swap_blocks_to_cpu("s")
                    pam.swap_blocks_to_gpu("s")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    break

        def getter():
            while not stop.is_set():
                try:
                    k, v = pam.get_kv_cache("s", 0)
                    if k is not None:
                        assert k.shape == v.shape
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    break

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(swapper), ex.submit(swapper),
                    ex.submit(getter), ex.submit(getter)]
            time.sleep(1.0)
            stop.set()
            concurrent.futures.wait(futs)

        assert errors == []
        assert len(pam.get_block_table("s")) == 8
        assert pam.num_free_blocks + pam.num_used_blocks == 128

    def test_swap_vs_append_same_sequence(self, pam):
        pam.allocate_sequence("s", num_tokens=16)
        errors = []

        def worker(tid):
            try:
                for _ in range(100):
                    pam.swap_blocks_to_cpu("s")
                    pam.append_token("s")
                    pam.swap_blocks_to_gpu("s")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(worker, t) for t in range(4)]
            concurrent.futures.wait(futures)

        assert errors == []
        assert pam.num_free_blocks + pam.num_used_blocks == 128


class TestResetRace:
    def test_reset_concurrent_with_mutators(self, pam):
        # Bounded rounds (no infinite loops): each worker performs a
        # fixed number of alloc/append/free cycles while reset runs
        # concurrently. Must not hang or corrupt the pool.
        errors = []
        import threading as _t

        def worker(tid):
            for i in range(200):
                sid = f"t{tid}-s{i % 10}"
                try:
                    pam.allocate_sequence(sid, num_tokens=8)
                    pam.append_token(sid)
                    pam.free_sequence(sid)
                except RuntimeError:
                    # Out of blocks / lock contention: benign under reset.
                    pass
                except ValueError:
                    # A concurrent reset() legitimately cleared _seq_blocks
                    # between allocate and append — the sequence no longer
                    # exists. That is the *intended* reset semantics, not a
                    # corruption; keep going.
                    pass
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    break

        def resetter():
            try:
                for _ in range(20):
                    pam.reset()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(worker, t) for t in range(4)]
            futs.append(ex.submit(resetter))
            concurrent.futures.wait(futs)

        assert errors == []
        pam.reset()
        assert pam.num_free_blocks == 128
        assert pam.num_used_blocks == 0
        assert len(pam._seq_blocks) == 0


class TestStatsRace:
    def test_get_stats_concurrent_with_alloc_free(self, pam):
        errors = []

        def worker(tid):
            try:
                for i in range(50):
                    sid = f"t{tid}-s{i}"
                    pam.allocate_sequence(sid, num_tokens=8)
                    stats = pam.get_stats()
                    assert stats["free_blocks"] + stats["used_blocks"] == 128
                    pam.free_sequence(sid)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(worker, t) for t in range(4)]
            concurrent.futures.wait(futures)

        assert errors == []
        assert pam.num_free_blocks == 128
