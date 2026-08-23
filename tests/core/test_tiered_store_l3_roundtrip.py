"""Regression tests for B3: TieredMemoryPool L3 (NVMe) round-trip preserves
tensors.

Entries demoted to / stored in the COLD tier used to be serialized to raw
bytes with ``numpy().tobytes()`` (losing dtype/shape), and ``get()`` returned
those opaque bytes as if they were the original KV tensors.  Tensors are now
persisted with dtype/shape metadata and reconstructed on load.
"""

import torch

from distllm.core.advanced_scheduling.tiered_store import StorageTier, TieredMemoryPool


def _pool(tmp_path):
    return TieredMemoryPool(
        gpu_memory_gb=0.01,
        cpu_memory_gb=0.01,
        nvme_path=str(tmp_path),
        nvme_max_gb=1.0,
    )


def test_put_cold_roundtrip_preserves_tensor(tmp_path):
    """Storing a tensor directly in L3 and reading it back must give a tensor."""
    pool = _pool(tmp_path)
    t = torch.randn(2, 3, 4, dtype=torch.float16)
    assert pool.put("k", t, tier=StorageTier.COLD)

    out = pool.get("k")
    assert isinstance(out, torch.Tensor), f"expected tensor, got {type(out)}"
    assert out.dtype == t.dtype
    assert out.shape == t.shape
    assert torch.allclose(out.float(), t.float())


def test_demote_to_l3_roundtrip_preserves_tensor(tmp_path):
    """Demoting a WARM tensor to L3 then reading it back must give a tensor."""
    pool = _pool(tmp_path)
    t = torch.randn(1, 8, 8)
    assert pool.put("k", t, tier=StorageTier.WARM)

    entry = pool._l2_cache.pop("k")
    pool._demote_to_l3("k", entry)
    assert pool._l3_cache["k"].data is None  # data is on disk

    out = pool.get("k")
    assert isinstance(out, torch.Tensor), f"expected tensor, got {type(out)}"
    assert out.dtype == t.dtype
    assert out.shape == t.shape
    assert torch.allclose(out, t)


def test_evict_demotes_hot_to_cold_and_roundtrips(tmp_path):
    """The full eviction path (HOT -> WARM -> COLD) must survive a read-back."""
    pool = TieredMemoryPool(
        gpu_memory_gb=0.0005,   # tiny L1 so HOT overflows quickly
        cpu_memory_gb=0.0005,   # tiny L2 so WARM overflows quickly
        nvme_path=str(tmp_path),
        nvme_max_gb=1.0,
    )
    t = torch.randn(4, 4)
    pool.put("a", t, tier=StorageTier.HOT)
    pool.put("b", torch.randn(4, 4), tier=StorageTier.HOT)  # evicts "a" to L2/L3

    # Force demotion of whatever is in WARM down to L3.
    while pool._l2_cache:
        key, entry = pool._l2_cache.popitem(last=False)
        pool._demote_to_l3(key, entry)

    out = pool.get("a")
    assert isinstance(out, torch.Tensor), f"expected tensor, got {type(out)}"
    assert out.shape == t.shape
