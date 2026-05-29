"""Unit tests for PagedKVCacheBackend (core/kv_cache.py).

Covers:
- attach / append_kv / get_kv / free
- Per-request memory budget (max_blocks_per_request)
- available property
- memory_usage / pool_utilization
"""

import importlib.util
import os
import sys

import pytest
import torch

from distllm.core.kv_cache import PagedKVCacheBackend


def _load_dist_pam():
    """Load dist/attention.py PagedAttentionManager directly."""
    # Load merkle first so the import in attention.py works
    merkle_spec = importlib.util.spec_from_file_location(
        "distllm.dist.merkle",
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "distllm", "dist", "merkle.py"),
    )
    merkle_mod = importlib.util.module_from_spec(merkle_spec)
    sys.modules["distllm.dist.merkle"] = merkle_mod
    merkle_spec.loader.exec_module(merkle_mod)

    spec = importlib.util.spec_from_file_location(
        "distllm.dist.attention",
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "distllm", "dist", "attention.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        pytest.skip(f"Cannot load dist/attention.py: {e}")
    return mod.PagedAttentionManager


@pytest.fixture
def pam():
    cls = _load_dist_pam()
    return cls(
        num_blocks=64, block_size=8,
        num_layers=2, num_heads=2, head_dim=4,
        device="cpu",
    )


@pytest.fixture
def backend(pam):
    return PagedKVCacheBackend(paged_mgr=pam)


class TestAttachAndFree:
    def test_attach_creates_sequence(self, backend, pam):
        backend.attach("req-1")
        assert pam.get_block_table("req-1") != []

    def test_free_removes_sequence(self, backend, pam):
        backend.attach("req-1")
        backend.free("req-1")
        # dist version returns None for unknown sequences
        assert pam.get_block_table("req-1") is None


class TestAppendAndGet:
    def test_append_kv(self, backend):
        backend.attach("req-1")
        backend.append_kv(layer_idx=0, new_key=torch.randn(2, 4, 4), new_value=torch.randn(2, 4, 4))

    def test_get_kv(self, backend):
        backend.attach("req-1")
        k_in = torch.randn(2, 8, 4)
        v_in = torch.randn(2, 8, 4)
        backend.append_kv(0, k_in, v_in)
        k_out, v_out = backend.get_kv("req-1", layer_idx=0, seq_len=8)
        assert k_out.shape == (2, 8, 4)


class TestMemoryBudget:
    def test_within_budget(self, pam):
        backend = PagedKVCacheBackend(paged_mgr=pam, max_blocks_per_request=5)
        backend.attach("req-1")
        # Should work — 1 block
        backend.append_kv(0, torch.randn(2, 4, 4), torch.randn(2, 4, 4))

    def test_exceeds_budget(self, pam):
        backend = PagedKVCacheBackend(paged_mgr=pam, max_blocks_per_request=1)
        backend.attach("req-1")
        # First append OK
        backend.append_kv(0, torch.randn(2, 4, 4), torch.randn(2, 4, 4))
        # Second append should exceed budget
        with pytest.raises(RuntimeError, match="exceeded block budget"):
            backend.append_kv(0, torch.randn(2, 4, 4), torch.randn(2, 4, 4))


class TestProperties:
    def test_available(self, pam):
        backend = PagedKVCacheBackend(paged_mgr=pam)
        assert backend.available is True

    def test_available_no_mgr(self):
        backend = PagedKVCacheBackend(paged_mgr=None)
        assert backend.available is False

    def test_pool_utilization(self, backend):
        assert backend.pool_utilization >= 0.0

    def test_memory_usage(self, backend):
        backend.attach("req-1")
        backend.append_kv(0, torch.randn(2, 4, 4), torch.randn(2, 4, 4))
        usage = backend.memory_usage()
        assert usage > 0
