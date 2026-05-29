"""Performance benchmarks for PagedAttention block operations.

Run with: pytest tests/performance/bench_paged_attention.py -v --benchmark-only
"""

import pytest
import torch

from distllm.backends.paged_attention import PagedAttentionManager


@pytest.fixture
def pam():
    return PagedAttentionManager(
        num_blocks=4096, block_size=16,
        num_layers=32, num_heads=32, head_dim=128,
        device="cpu",
    )


class TestAllocationLatency:
    def test_allocate_single(self, pam, benchmark):
        counter = [0]
        def alloc():
            sid = f"seq-{counter[0]}"
            counter[0] += 1
            pam.allocate_sequence(sid, num_tokens=128)
        benchmark(alloc)

    def test_allocate_and_free(self, pam, benchmark):
        counter = [0]
        def cycle():
            sid = f"seq-{counter[0]}"
            counter[0] += 1
            pam.allocate_sequence(sid, num_tokens=128)
            pam.free_sequence(sid)
        benchmark(cycle)


class TestAppendToken:
    def test_append_single(self, pam, benchmark):
        pam.allocate_sequence("seq-1", num_tokens=256)
        benchmark(pam.append_token, "seq-1")

    def test_append_many(self, pam, benchmark):
        pam.allocate_sequence("seq-1", num_tokens=256)
        def append_16():
            for _ in range(16):
                pam.append_token("seq-1")
        benchmark(append_16)


class TestCopyOnWrite:
    def test_cow(self, pam, benchmark):
        pam.allocate_sequence("src", num_tokens=512)
        counter = [0]
        def cow():
            counter[0] += 1
            pam.copy_on_write("src", f"dst-{counter[0]}")
        benchmark(cow)


class TestSwapLatency:
    def test_swap_out_in(self, pam, benchmark):
        pam.allocate_sequence("seq-1", num_tokens=1024)
        def swap_cycle():
            pam.swap_blocks_to_cpu("seq-1")
            pam.swap_blocks_to_gpu("seq-1")
        benchmark(swap_cycle)


class TestGetKVCache:
    @pytest.mark.parametrize("seq_len", [64, 256, 1024, 4096])
    def test_gather_kv(self, seq_len, benchmark):
        pam = PagedAttentionManager(
            num_blocks=512, block_size=16,
            num_layers=2, num_heads=2, head_dim=64,
            device="cpu",
        )
        pam.allocate_sequence("seq-1", num_tokens=seq_len)
        benchmark(pam.get_kv_cache, "seq-1", 0)


class TestMemoryPressure:
    def test_allocate_95_percent(self):
        pam = PagedAttentionManager(
            num_blocks=1000, block_size=16,
            num_layers=1, num_heads=1, head_dim=16, device="cpu",
        )
        sequences = []
        for i in range(950 // 2):  # ~475 sequences of 32 tokens = 2 blocks each
            sid = f"seq-{i}"
            try:
                pam.allocate_sequence(sid, num_tokens=32)
                sequences.append(sid)
            except RuntimeError:
                break
        assert pam.num_used_blocks > 900
        # Free all
        for sid in sequences:
            pam.free_sequence(sid)
        assert pam.num_free_blocks == 1000
