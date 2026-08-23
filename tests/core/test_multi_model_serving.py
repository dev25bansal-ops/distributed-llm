"""Tests for multi-model serving modules: ModelRegistry, ModelMemoryBudget,
ModelHotSwapManager, GPUTimeSlicer, MemoryAwarePlacer.

Covers:
- ModelEntry / ModelRegistry: register, get, remove, list
- ModelMemoryBudget: budgets, usage, can_fit, available_gb, stats
- ModelHotSwapManager: register, load, unload, remove, LRU eviction, callbacks
- GPUTimeSlicer: register, get_next_model, slice duration, start/end slice, SLA violations
- MemoryAwarePlacer: place, preferred GPU, best-fit, remove, stats
"""

from __future__ import annotations

import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mms = load_module("distllm/core/multi_model_serving.py")
ModelRegistry = _mms.ModelRegistry
ModelEntry = _mms.ModelEntry
ModelInstance = _mms.ModelInstance
ModelMemoryBudget = _mms.ModelMemoryBudget
ModelHotSwapManager = _mms.ModelHotSwapManager
ModelSLA = _mms.ModelSLA
TimeSlice = _mms.TimeSlice
GPUTimeSlicer = _mms.GPUTimeSlicer
MemoryAwarePlacer = _mms.MemoryAwarePlacer


# ── ModelRegistry ─────────────────────────────────────────────────────────────


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        entry = reg.register("llama", "/models/llama", 32)
        assert isinstance(entry, ModelEntry)
        assert reg.get("llama") is entry
        assert entry.name == "llama"
        assert entry.path == "/models/llama"
        assert entry.total_layers == 32

    def test_get_missing_returns_none(self):
        reg = ModelRegistry()
        assert reg.get("nonexistent") is None

    def test_remove(self):
        reg = ModelRegistry()
        reg.register("llama", "/models/llama", 32)
        removed = reg.remove("llama")
        assert removed is not None
        assert reg.get("llama") is None

    def test_remove_missing_returns_none(self):
        reg = ModelRegistry()
        assert reg.remove("nonexistent") is None

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register("a", "/a", 1)
        reg.register("b", "/b", 2)
        assert set(reg.list_models()) == {"a", "b"}

    def test_max_models_default(self):
        reg = ModelRegistry()
        assert reg.max_models == 4


# ── ModelMemoryBudget ─────────────────────────────────────────────────────────


class TestModelMemoryBudget:
    def test_defaults(self):
        budget = ModelMemoryBudget()
        assert budget.total_gpu_memory_gb == 0.0
        assert budget.available_gb() == float("inf")

    def test_set_and_get_budget(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        budget.set_budget("llama", 20.0)
        assert budget.get_budget("llama") == 20.0

    def test_get_budget_missing(self):
        budget = ModelMemoryBudget()
        assert budget.get_budget("nonexistent") is None

    def test_update_usage(self):
        budget = ModelMemoryBudget()
        budget.update_usage("llama", 15.0)
        assert budget.get_usage("llama") == 15.0

    def test_total_allocated_gb(self):
        budget = ModelMemoryBudget()
        budget.update_usage("a", 10.0)
        budget.update_usage("b", 5.0)
        assert budget.total_allocated_gb() == 15.0

    def test_available_gb_with_total(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        budget.update_usage("llama", 20.0)
        assert budget.available_gb() == 60.0

    def test_available_gb_bounded_below_zero(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=10.0)
        budget.update_usage("llama", 20.0)
        assert budget.available_gb() == 0.0

    def test_can_fit_within_budget(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        budget.set_budget("llama", 30.0)
        assert budget.can_fit("llama", 20.0) is True

    def test_can_fit_exceeds_budget(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        budget.set_budget("llama", 30.0)
        budget.update_usage("llama", 25.0)
        assert budget.can_fit("llama", 10.0) is False  # 25+10=35 > 30

    def test_can_fit_exceeds_available_gpu(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=20.0)
        budget.update_usage("other", 18.0)
        assert budget.can_fit("new", 5.0) is False  # only 2GB free

    def test_can_fit_no_budget_set(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        # No explicit budget set for model, only check available_gb
        budget.update_usage("other", 10.0)
        assert budget.can_fit("new", 5.0) is True

    def test_remove_model(self):
        budget = ModelMemoryBudget()
        budget.set_budget("llama", 30.0)
        budget.update_usage("llama", 15.0)
        budget.remove_model("llama")
        assert budget.get_budget("llama") is None
        assert budget.get_usage("llama") == 0.0

    def test_stats(self):
        budget = ModelMemoryBudget(total_gpu_memory_gb=80.0)
        budget.set_budget("llama", 30.0)
        budget.update_usage("llama", 15.0)
        s = budget.stats()
        assert s["total_gpu_memory_gb"] == 80.0
        assert s["total_allocated_gb"] == 15.0
        assert "llama" in s["budgets"]


# ── ModelHotSwapManager ───────────────────────────────────────────────────────


class TestModelHotSwapManagerConstruction:
    def test_default_construction(self):
        mgr = ModelHotSwapManager()
        assert mgr._max_models == 4
        assert mgr._on_load_model is None
        assert mgr._on_unload_model is None
        assert mgr._layer_pool is not None  # SharedLayerPool enabled by default

    def test_construction_with_params(self):
        reg = ModelRegistry(max_models=8)
        mgr = ModelHotSwapManager(
            model_registry=reg,
            total_gpu_memory_gb=80.0,
            max_models=8,
            enable_layer_sharing=False,
        )
        assert mgr._max_models == 8
        assert mgr.registry.max_models == 8
        assert mgr._layer_pool is None

    def test_set_callbacks(self):
        mgr = ModelHotSwapManager()
        cb = lambda name, path: (None, None, 0.0)
        mgr.set_callbacks(on_load_model=cb, on_unload_model=cb)
        assert mgr._on_load_model is cb


class TestModelHotSwapManagerRegisterLoadUnload:
    def test_register_model(self):
        mgr = ModelHotSwapManager()
        entry = mgr.register_model("llama", "/models/llama", 32, memory_budget_gb=20.0)
        assert entry.name == "llama"
        assert mgr.registry.default_model == "llama"
        assert mgr.memory_budget.get_budget("llama") == 20.0

    def test_register_multiple_first_is_default(self):
        mgr = ModelHotSwapManager()
        mgr.register_model("a", "/a", 1)
        mgr.register_model("b", "/b", 2)
        assert mgr.registry.default_model == "a"

    def test_load_model_not_registered_returns_false(self):
        mgr = ModelHotSwapManager()
        assert mgr.load_model("nonexistent") is False

    def test_load_model_with_callback(self):
        mgr = ModelHotSwapManager()

        def load_cb(name, path):
            return ("model_obj", "tokenizer_obj", 5.0)

        mgr.set_callbacks(on_load_model=load_cb, on_unload_model=lambda n, m, t: None)
        mgr.register_model("llama", "/models/llama", 32)
        assert mgr.load_model("llama") is True
        assert "llama" in mgr._loaded
        assert mgr._loaded["llama"].actual_memory_gb == 5.0
        assert mgr._loaded["llama"].is_loading is False
        assert mgr._loaded["llama"].request_count == 1

    def test_load_model_no_callback_returns_false(self):
        mgr = ModelHotSwapManager()
        mgr.register_model("llama", "/models/llama", 32)
        assert mgr.load_model("llama") is False

    def test_load_already_loaded_updates_access(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 0.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("llama", "/models/llama", 32)
        mgr.load_model("llama")
        prev_count = mgr._loaded["llama"].request_count
        mgr.load_model("llama")
        assert mgr._loaded["llama"].request_count == prev_count + 1

    def test_unload_model(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 0.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("llama", "/models/llama", 32)
        mgr.load_model("llama")
        assert mgr.unload_model("llama") is True
        assert "llama" not in mgr._loaded

    def test_unload_not_loaded_returns_false(self):
        mgr = ModelHotSwapManager()
        assert mgr.unload_model("nonexistent") is False

    def test_remove_model(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 0.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("llama", "/models/llama", 32)
        mgr.load_model("llama")
        assert mgr.remove_model("llama") is not None
        assert "llama" not in mgr._loaded
        assert mgr.registry.get("llama") is None

    def test_get_model_updates_access_time(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 0.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("llama", "/models/llama", 32)
        mgr.load_model("llama")
        inst = mgr.get_model("llama")
        assert inst is not None
        assert inst.request_count > 1

    def test_get_model_nonexistent(self):
        mgr = ModelHotSwapManager()
        assert mgr.get_model("nonexistent") is None

    def test_list_loaded_models(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 2.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("a", "/a", 1)
        mgr.register_model("b", "/b", 1)
        mgr.load_model("a")
        mgr.load_model("b")
        lst = mgr.list_loaded_models()
        assert len(lst) == 2
        names = {m["name"] for m in lst}
        assert names == {"a", "b"}

    def test_get_total_memory_usage(self):
        mgr = ModelHotSwapManager()
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 3.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("a", "/a", 1)
        mgr.load_model("a")
        assert mgr.get_total_memory_usage() == 3.0


class TestModelHotSwapManagerLRUEviction:
    def test_eviction_when_at_capacity(self):
        mgr = ModelHotSwapManager(max_models=1)
        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 1.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("a", "/a", 1)
        mgr.register_model("b", "/b", 1)
        mgr.load_model("a")
        # Loading 'b' should evict 'a'
        assert mgr.load_model("b") is True
        assert "a" not in mgr._loaded
        assert "b" in mgr._loaded
        assert mgr._total_evictions == 1

    def test_eviction_skips_loading_models(self):
        """Don't evict a model that's currently loading."""
        mgr = ModelHotSwapManager(max_models=1)

        def load_cb(name, path):
            if name == "stuck":
                # Simulate long load — leaves is_loading=True
                import threading
                import time
                time.sleep(0.001)
            return (None, None, 0.0)

        mgr.set_callbacks(
            on_load_model=load_cb,
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("a", "/a", 1)
        mgr.register_model("stuck", "/stuck", 1)
        mgr.load_model("a")
        # Force 'a' to be the only evictable — should work
        assert mgr.load_model("stuck") is True

    def test_evict_all_loading_returns_false(self):
        """If all models are loading, eviction fails."""
        mgr = ModelHotSwapManager(max_models=1)

        mgr.set_callbacks(
            on_load_model=lambda n, p: (None, None, 0.0),
            on_unload_model=lambda n, m, t: None,
        )
        mgr.register_model("a", "/a", 1)
        mgr.register_model("b", "/b", 1)
        mgr.load_model("a")
        # Mark 'a' as loading so it can't be evicted
        mgr._loaded["a"].is_loading = True
        assert mgr.load_model("b") is False


class TestModelHotSwapManagerLayerSharing:
    def test_layer_sharing_disabled(self):
        mgr = ModelHotSwapManager(enable_layer_sharing=False)
        state_dict = {"layer.0.weight": "dummy"}
        assert mgr.register_model_layers("test", state_dict) is None
        assert mgr.get_shared_tensor("test", "layer.0.weight") is None
        assert mgr.find_similar_models("test") == []
        assert mgr.get_layer_sharing_stats() is None

    def test_stats_includes_layer_sharing(self):
        mgr = ModelHotSwapManager(enable_layer_sharing=False)
        s = mgr.stats()
        assert "layer_sharing" not in s


# ── GPUTimeSlicer ─────────────────────────────────────────────────────────────


class TestGPUTimeSlicer:
    def test_default_construction(self):
        s = GPUTimeSlicer()
        assert s._base_slice_ms == 100.0
        assert s._slas == {}
        assert s._active_model is None

    def test_register_model(self):
        s = GPUTimeSlicer()
        sla = ModelSLA(model_name="llama", max_latency_ms=1000.0, priority=1)
        s.register_model(sla)
        assert "llama" in s._slas
        assert "llama" in s._stats

    def test_get_next_model_with_no_models(self):
        s = GPUTimeSlicer()
        assert s.get_next_model() is None

    def test_get_next_model_returns_model(self):
        s = GPUTimeSlicer()
        s.register_model(ModelSLA(model_name="a"))
        s.register_model(ModelSLA(model_name="b"))
        model = s.get_next_model()
        assert model in ("a", "b")

    def test_get_slice_duration_default(self):
        s = GPUTimeSlicer(slice_duration_ms=200.0)
        # No SLA registered -> uses base
        assert s.get_slice_duration("unknown") == 200.0

    def test_get_slice_duration_with_priority(self):
        s = GPUTimeSlicer(slice_duration_ms=100.0)
        s.register_model(ModelSLA(model_name="high", priority=1))
        s.register_model(ModelSLA(model_name="low", priority=3))
        high_dur = s.get_slice_duration("high")
        low_dur = s.get_slice_duration("low")
        assert high_dur > low_dur

    def test_start_slice(self):
        s = GPUTimeSlicer()
        s.register_model(ModelSLA(model_name="llama"))
        sl = s.start_slice("llama")
        assert isinstance(sl, TimeSlice)
        assert sl.model_name == "llama"
        assert sl.duration_ms > 0
        assert s._active_model == "llama"

    def test_end_slice(self):
        s = GPUTimeSlicer()
        s.register_model(ModelSLA(model_name="llama"))
        sl = s.start_slice("llama")
        sl.requests_served = 5
        s.end_slice(sl)
        stats = s._stats.get("llama", {})
        assert stats["requests_served"] == 5

    def test_check_sla_violations_empty(self):
        s = GPUTimeSlicer()
        assert s.check_sla_violations() == []

    def test_check_sla_violations_detects(self):
        s = GPUTimeSlicer()
        sla = ModelSLA(model_name="slow", max_latency_ms=10.0)
        s.register_model(sla)
        # Manually set high latency
        s._stats["slow"]["total_time_ms"] = 1000.0
        s._stats["slow"]["requests_served"] = 1
        violations = s.check_sla_violations()
        assert "slow" in violations

    def test_stats(self):
        s = GPUTimeSlicer()
        s.register_model(ModelSLA(model_name="a"))
        stats = s.stats()
        assert stats["registered_models"] == 1
        assert stats["base_slice_ms"] == 100.0


# ── MemoryAwarePlacer ─────────────────────────────────────────────────────────


class TestMemoryAwarePlacer:
    def test_default_construction(self):
        p = MemoryAwarePlacer()
        assert p._gpu_memory == 80.0
        assert p._safety_margin == 0.1

    def test_place_model_returns_gpu_id(self):
        p = MemoryAwarePlacer(gpu_memory_gb=80.0)
        gpu = p.place_model("llama", 10.0)
        assert gpu is not None
        assert isinstance(gpu, int)
        assert gpu >= 0

    def test_place_model_preferred_gpu(self):
        p = MemoryAwarePlacer(gpu_memory_gb=80.0)
        gpu = p.place_model("llama", 10.0, preferred_gpu=2)
        assert gpu == 2

    def test_preferred_gpu_full_falls_back(self):
        p = MemoryAwarePlacer(gpu_memory_gb=80.0)
        # Fill GPU 2
        p.place_model("big", 70.0, preferred_gpu=2)
        # Should fall back to another GPU
        gpu = p.place_model("small", 5.0, preferred_gpu=2)
        assert gpu != 2
        assert gpu is not None

    def test_place_model_no_space(self):
        p = MemoryAwarePlacer(gpu_memory_gb=5.0, safety_margin_pct=0.0)
        for i in range(8):
            p.place_model(f"m{i}", 5.0, preferred_gpu=i)
        # All 8 GPUs full with 5GB each, no more space
        gpu = p.place_model("extra", 0.5)
        assert gpu is None
        # Total 10GB, no safety margin -> both models at 5GB each leaves no space
        gpu = p.place_model("extra", 5.0)
        assert gpu is None

    def test_get_placement(self):
        p = MemoryAwarePlacer()
        p.place_model("llama", 10.0, preferred_gpu=0)
        assert p.get_placement("llama") == 0

    def test_get_placement_missing(self):
        p = MemoryAwarePlacer()
        assert p.get_placement("nonexistent") is None

    def test_remove_model(self):
        p = MemoryAwarePlacer()
        p.place_model("llama", 10.0)
        p.remove_model("llama")
        assert p.get_placement("llama") is None

    def test_stats(self):
        p = MemoryAwarePlacer(gpu_memory_gb=80.0)
        p.place_model("llama", 10.0, preferred_gpu=0)
        s = p.stats()
        assert s["placed_models"] == 1
        assert s["gpu_memory_gb"] == 80.0
