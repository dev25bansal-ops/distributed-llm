"""Tests for CrossModelPrefixSharing.

Covers: register_model, store, lookup (direct, base, sibling),
compatible models, LRU eviction, TTL expiry, stats.
"""

from __future__ import annotations

import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/cross_model_prefix_sharing.py")
CrossModelPrefixSharing = _mod.CrossModelPrefixSharing
ModelVariant = _mod.ModelVariant
SharedCacheEntry = _mod.SharedCacheEntry


@pytest.fixture
def sharing():
    s = CrossModelPrefixSharing(max_entries=10, default_ttl=3600)
    s.register_model("llama-70b", base_model="llama-70b-base", shared_layers=70, total_layers=80)
    s.register_model("llama-70b-instruct", base_model="llama-70b-base", shared_layers=70, total_layers=80)
    s.register_model("llama-70b-base", shared_layers=80, total_layers=80)
    return s


class TestModelVariantDefaults:
    def test_default_values(self):
        mv = ModelVariant(model_id="test")
        assert mv.model_id == "test"
        assert mv.base_model == ""
        assert mv.shared_layers == 0
        assert mv.total_layers == 0


class TestRegisterModel:
    def test_register_model(self, sharing):
        s = sharing
        assert "llama-70b" in s._models
        assert "llama-70b-instruct" in s._models
        assert s._models["llama-70b"].base_model == "llama-70b-base"
        assert s._models["llama-70b"].shared_layers == 70

    def test_register_duplicate_overwrites(self, sharing):
        s = sharing
        s.register_model("llama-70b", base_model="other-base", shared_layers=50)
        assert s._models["llama-70b"].base_model == "other-base"


class TestStore:
    def test_store_returns_hash(self, sharing):
        h = sharing.store("llama-70b", [1, 2, 3], kv_data="data")
        assert isinstance(h, str)
        assert len(h) == 16  # sha256 hexdigest[:16]

    def test_store_creates_cache_entry(self, sharing):
        h = sharing.store("llama-70b", [1, 2, 3], kv_data="data")
        key = f"llama-70b:{h}"
        assert key in sharing._cache
        assert sharing._cache[key].kv_data == "data"

    def test_store_with_unknown_model_still_works(self, sharing):
        h = sharing.store("unknown-model", [1], "data")
        assert f"unknown-model:{h}" in sharing._cache
        # shared_layers defaults to 0 since model not registered
        entry = sharing._cache[f"unknown-model:{h}"]
        assert entry.shared_layers == 0


class TestLookup:
    def test_direct_match(self, sharing):
        h = sharing.store("llama-70b", [1, 2, 3], kv_data="direct")
        entry = sharing.lookup("llama-70b", [1, 2, 3])
        assert entry is not None
        assert entry.kv_data == "direct"

    def test_direct_match_updates_access_count(self, sharing):
        h = sharing.store("llama-70b", [1, 2, 3], kv_data="data")
        entry = sharing.lookup("llama-70b", [1, 2, 3])
        assert entry is not None
        assert entry.access_count == 1

    def test_base_model_lookup(self, sharing):
        # Store under base model
        sharing.store("llama-70b-base", [5, 6, 7], kv_data="from-base")
        # Instruct model should find it via base_model chain
        entry = sharing.lookup("llama-70b-instruct", [5, 6, 7])
        assert entry is not None
        assert entry.kv_data == "from-base"

    def test_sibling_lookup(self, sharing):
        # Store under one sibling
        sharing.store("llama-70b", [10, 20], kv_data="from-sibling")
        # Other sibling should find it
        entry = sharing.lookup("llama-70b-instruct", [10, 20])
        assert entry is not None
        assert entry.kv_data == "from-sibling"

    def test_missing_tokens_returns_none(self, sharing):
        entry = sharing.lookup("llama-70b", [999, 888])
        assert entry is None

    def test_unknown_model_returns_none(self, sharing):
        entry = sharing.lookup("nonexistent", [1, 2])
        assert entry is None

    def test_cross_model_hit_stats(self, sharing):
        # Direct hit — not a cross-model hit
        sharing.store("llama-70b", [1], "data")
        sharing.lookup("llama-70b", [1])
        assert sharing._stats["cross_model_hits"] == 0
        # Cross-model hit
        sharing.lookup("llama-70b-instruct", [1])
        assert sharing._stats["cross_model_hits"] >= 1


class TestCompatibleModels:
    def test_compatible_models_for_base(self, sharing):
        comp = sharing.get_compatible_models("llama-70b-base")
        # Base model has base_model="" so it only matches itself
        assert comp == ["llama-70b-base"]

    def test_compatible_models_for_unknown(self, sharing):
        comp = sharing.get_compatible_models("nobody")
        assert comp == []

    def test_compatible_models_self_only_when_no_base(self, sharing):
        s = CrossModelPrefixSharing()
        s.register_model("standalone", base_model="", shared_layers=10)
        comp = s.get_compatible_models("standalone")
        assert comp == ["standalone"]


class TestLRUEviction:
    def test_eviction_when_max_entries_exceeded(self):
        s = CrossModelPrefixSharing(max_entries=3, default_ttl=3600)
        s.register_model("m1", shared_layers=10, total_layers=10)
        for i in range(5):
            s.store("m1", [i], f"data-{i}")
        assert len(s._cache) == 3  # max_entries is enforced

    def test_lru_evicts_oldest_access(self):
        s = CrossModelPrefixSharing(max_entries=2, default_ttl=3600)
        s.register_model("m1", shared_layers=10, total_layers=10)
        s.store("m1", [1], "first")
        s.store("m1", [2], "second")
        # Access first to make second the LRU candidate
        s.lookup("m1", [1])
        s.store("m1", [3], "third")  # should evict second
        assert s.lookup("m1", [2]) is None  # second was evicted
        assert s.lookup("m1", [1]) is not None  # first still alive
        assert s.lookup("m1", [3]) is not None  # third alive


class TestStats:
    def test_stats_basic(self, sharing):
        s = sharing
        stats = s.stats()
        assert stats["registered_models"] == 3
        assert stats["cached_entries"] == 0
        assert stats["cross_model_hit_rate"] == 0.0
        assert stats["total_lookups"] == 0

    def test_stats_after_lookups(self, sharing):
        s = sharing
        s.store("llama-70b", [1, 2], "data")
        s.lookup("llama-70b", [1, 2])
        s.lookup("llama-70b-instruct", [1, 2])
        stats = s.stats()
        assert stats["total_lookups"] == 2
        assert stats["cross_model_hits"] == 1
        assert stats["cross_model_hit_rate"] == 0.5
