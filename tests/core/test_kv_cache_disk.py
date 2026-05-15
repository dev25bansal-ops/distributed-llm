"""Tests for KV cache disk persistence."""

import tempfile
from pathlib import Path

import torch

from distllm.core.kv_cache import KVCache, save_kv_cache_to_disk, load_kv_cache_from_disk


class TestKVCacheDisk:
    """Tests for KV cache save/load to disk."""

    def test_kv_cache_save_to_disk(self):
        """save_to_disk writes file."""
        cache = KVCache()
        cache.cache = [
            (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
        ]

        tmpdir = tempfile.mkdtemp()
        filepath = Path(tmpdir) / "test.pt"
        save_kv_cache_to_disk(cache, str(filepath))
        assert filepath.exists()
        filepath.unlink()

    def test_kv_cache_load_from_disk(self):
        """load_from_disk restores cache."""
        cache = KVCache()
        cache.cache = [
            (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
        ]

        tmpdir = tempfile.mkdtemp()
        filepath = Path(tmpdir) / "test.pt"
        save_kv_cache_to_disk(cache, str(filepath))
        loaded = load_kv_cache_from_disk(str(filepath))

        assert len(loaded.cache) == 1
        filepath.unlink()

    def test_kv_cache_roundtrip(self):
        """Save then load preserves data."""
        cache = KVCache()
        k1 = torch.randn(1, 2, 4, 8)
        v1 = torch.randn(1, 2, 4, 8)
        cache.cache = [(k1, v1)]

        tmpdir = tempfile.mkdtemp()
        filepath = Path(tmpdir) / "test.pt"
        save_kv_cache_to_disk(cache, str(filepath))
        loaded = load_kv_cache_from_disk(str(filepath))

        assert len(loaded.cache) == 1
        k2, v2 = loaded.cache[0]
        assert torch.allclose(k1, k2)
        assert torch.allclose(v1, v2)
        filepath.unlink()
