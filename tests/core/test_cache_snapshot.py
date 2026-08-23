"""Tests for CacheSnapshot (point-in-time cache export/import)."""

import json
import tempfile
from pathlib import Path

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/cache_snapshot.py")
CacheSnapshot = _mod.CacheSnapshot


class TestCacheSnapshot:
    def test_init(self):
        snap = CacheSnapshot()
        assert snap._cache_manager is None

    def test_init_with_manager(self):
        snap = CacheSnapshot(cache_manager=object())
        assert snap._cache_manager is not None

    def test_export_snapshot_no_manager(self):
        snap = CacheSnapshot()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        result = snap.export_snapshot(path)
        assert result["version"] == 1
        assert "timestamp" in result
        assert "exported_at" in result
        Path(path).unlink(missing_ok=True)

    def test_export_snapshot_with_stats(self):
        class FakeManager:
            prefix_cache = None
            _predictive_cache = None
            def get_tier_stats(self):
                return {"local": {"hits": 5, "misses": 3}}

        snap = CacheSnapshot(FakeManager())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        result = snap.export_snapshot(path)
        assert result["tier_stats"]["local"]["hits"] == 5
        Path(path).unlink(missing_ok=True)

    def test_import_snapshot_json(self):
        data = {"version": 1, "timestamp": 100.0, "tier_stats": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        snap = CacheSnapshot()
        result = snap.import_snapshot(path)
        assert result["version"] == 1
        assert result["timestamp"] == 100.0
        Path(path).unlink(missing_ok=True)

    def test_import_snapshot_missing_file(self):
        import pytest
        snap = CacheSnapshot()
        with pytest.raises(FileNotFoundError):
            snap.import_snapshot("/nonexistent/snapshot.json")

    def test_diff_snapshots(self):
        snap = CacheSnapshot()
        a = {
            "timestamp": 1000.0,
            "prefix_cache": {"stats": {"hits": 10, "misses": 5}},
        }
        b = {
            "timestamp": 2000.0,
            "prefix_cache": {"stats": {"hits": 20, "misses": 5}},
        }
        diff = snap.diff_snapshots(a, b)
        assert diff["time_diff_s"] == 1000.0
        assert diff["prefix_cache_diff"]["hits"]["before"] == 10
        assert diff["prefix_cache_diff"]["hits"]["after"] == 20
        assert diff["prefix_cache_diff"]["hits"]["delta"] == 10
        assert "misses" not in diff["prefix_cache_diff"]

    def test_diff_snapshots_no_prefix_cache(self):
        snap = CacheSnapshot()
        a = {"timestamp": 1.0}
        b = {"timestamp": 2.0}
        diff = snap.diff_snapshots(a, b)
        assert diff["time_diff_s"] == 1.0

    def test_serialize_prefix_cache(self):
        """_serialize_prefix_cache handles cache without _cache attr."""
        class FakePrefixCache:
            pass

        snap = CacheSnapshot()
        result = snap._serialize_prefix_cache(FakePrefixCache())
        assert result == []
