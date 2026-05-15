"""Tests for Feature 26: Model Cache Management."""

import json
import time
from pathlib import Path

import pytest

from distllm.models.cache import ModelCache


class TestModelCache:
    def test_init_creates_cache_dir(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        assert (tmp_path / "cache").exists()

    def test_init_default_cache_dir(self):
        cache = ModelCache()
        assert "distributed-llm" in str(cache.cache_dir)

    def test_get_disk_usage_starts_at_zero(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get_disk_usage() == 0

    def test_get_disk_usage_gb(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=1.0)
        usage = cache.get_disk_usage_gb()
        assert usage == 0.0

    def test_get_usage_pct(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=1.0)
        assert cache.get_usage_pct() == 0.0

    def test_ensure_model_tracked_records_entry(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "file.bin").write_bytes(b"\x00" * 100)

        cache.ensure_model_tracked(str(model_path), "test-org/test-model")

        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0]["model_id"] == "test-org/test-model"
        assert entries[0]["size_bytes"] > 0

    def test_list_entries_returns_empty(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        assert cache.list_entries() == []

    def test_list_entries_sorted_by_lru(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))

        # Add model A
        path_a = tmp_path / "model_a"
        path_a.mkdir()
        cache.ensure_model_tracked(str(path_a), "model-a")

        # Small delay to ensure different timestamps
        time.sleep(0.01)

        # Add model B (more recent)
        path_b = tmp_path / "model_b"
        path_b.mkdir()
        cache.ensure_model_tracked(str(path_b), "model-b")

        entries = cache.list_entries()
        assert len(entries) == 2
        # LRU first (model-a was added first)
        assert entries[0]["model_id"] == "model-a"
        assert entries[1]["model_id"] == "model-b"

    def test_touch_updates_last_accessed(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        path = tmp_path / "model"
        path.mkdir()
        cache.ensure_model_tracked(str(path), "model-x")

        original_time = cache._metadata["entries"]["model-x"]["last_accessed"]
        time.sleep(0.01)

        cache.touch("model-x")
        new_time = cache._metadata["entries"]["model-x"]["last_accessed"]

        assert new_time > original_time

    def test_remove_entry_deletes_files(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        path = tmp_path / "model"
        path.mkdir()
        cache.ensure_model_tracked(str(path), "model-to-remove")

        result = cache.remove_entry("model-to-remove", delete_files=True)

        assert result is True
        assert not path.exists()

    def test_remove_entry_keeps_files_when_flag_false(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        path = tmp_path / "model"
        path.mkdir()
        cache.ensure_model_tracked(str(path), "model-to-keep")

        result = cache.remove_entry("model-to-keep", delete_files=False)

        assert result is True
        assert path.exists()

    def test_remove_entry_returns_false_for_missing(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        assert cache.remove_entry("nonexistent") is False

    def test_evict_if_needed_removes_lru_entries(self, tmp_path):
        # Small cache: 1 byte max
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=0.000000001)

        # Add models with tracked sizes
        for i in range(3):
            path = tmp_path / f"model_{i}"
            path.mkdir()
            cache.ensure_model_tracked(str(path), f"model-{i}")
            # Manually set size to be large enough to trigger eviction
            cache._metadata["entries"][f"model-{i}"]["size_bytes"] = 1000
            time.sleep(0.01)
        cache._recalculate_total()
        cache._save_metadata()

        evicted = cache.evict_if_needed(target_pct=50.0)

        assert len(evicted) > 0
        # LRU entries should be evicted first
        remaining = cache.list_entries()
        assert len(remaining) < 3

    def test_can_fit_returns_true_when_space_available(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=1.0)
        assert cache.can_fit(1000) is True

    def test_can_fit_returns_false_when_no_space(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=0.000000001)
        cache._metadata["total_size_bytes"] = 1000  # Already full
        assert cache.can_fit(1000) is False

    def test_get_available_space(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"), max_size_gb=1.0)
        available = cache.get_available_space()
        assert available == int(1.0 * 1024 * 1024 * 1024)

    def test_reset_clears_metadata(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        path = tmp_path / "model"
        path.mkdir()
        cache.ensure_model_tracked(str(path), "model-to-reset")

        cache.reset()

        assert cache.list_entries() == []
        assert cache.get_disk_usage() == 0

    def test_metadata_persistence(self, tmp_path):
        cache = ModelCache(cache_dir=str(tmp_path / "cache"))
        path = tmp_path / "model"
        path.mkdir()
        cache.ensure_model_tracked(str(path), "persistent-model")

        # Create new cache instance (should reload metadata)
        cache2 = ModelCache(cache_dir=str(tmp_path / "cache"))
        entries = cache2.list_entries()

        assert len(entries) == 1
        assert entries[0]["model_id"] == "persistent-model"
