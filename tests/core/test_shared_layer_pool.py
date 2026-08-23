"""Tests for distllm.core.shared_layer_pool -- SharedLayerPool.
from __future__ import annotations

Detects and shares common layers (embeddings, attention, MLP) across
similar models to reduce GPU memory usage.

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""


import pytest

try:
    import torch
    _ = torch.float16  # canary: real torch always has this; pollution replaces torch with an empty stub
except (ModuleNotFoundError, ImportError, AttributeError) as _e:
    pytest.skip(f"requires working torch / distllm.core.shared_layer_pool (not available): {_e}", allow_module_level=True)


import threading
from typing import Any

import pytest
import torch

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the module under test
_pool_mod = load_module("distllm/core/shared_layer_pool.py")

# Re-export symbols for test readability
LayerFingerprint = _pool_mod.LayerFingerprint
SharedLayer = _pool_mod.SharedLayer
SharedLayerPool = _pool_mod.SharedLayerPool


# ===================================================================
# Helpers
# ===================================================================

def make_tensor(shape: tuple[int, ...],
                dtype: torch.dtype = torch.float32,
                fill_value: float | int = 1.0) -> torch.Tensor:
    """Deterministic tensor factory for tests."""
    return torch.full(shape, fill_value, dtype=dtype)


def make_state_dict(layers: dict[str, tuple[int, ...]],
                    dtype: torch.dtype = torch.float32,
                    fill_value: float | int = 1.0) -> dict[str, torch.Tensor]:
    """Build a state dict from layer-name -> shape mappings."""
    return {
        name: make_tensor(shape, dtype, fill_value)
        for name, shape in layers.items()
    }


# ===================================================================
# LAYER FINGERPRINT TESTS
# ===================================================================

class TestLayerFingerprint:
    """LayerFingerprint dataclass -- construction and defaults."""

    def test_default_construction(self) -> None:
        fp = LayerFingerprint(
            layer_name="layers.0.self_attn.q_proj.weight",
            shape=(4096, 4096),
            dtype="torch.float32",
            param_hash="abc123",
            param_count=16777216,
            memory_bytes=67108864,
        )
        assert fp.layer_name == "layers.0.self_attn.q_proj.weight"
        assert fp.shape == (4096, 4096)
        assert fp.dtype == "torch.float32"
        assert fp.param_hash == "abc123"
        assert fp.param_count == 16777216
        assert fp.memory_bytes == 67108864

    def test_all_fields(self) -> None:
        fp = LayerFingerprint(
            layer_name="embed_tokens.weight",
            shape=(32000, 4096),
            dtype="torch.float16",
            param_hash="def456",
            param_count=131072000,
            memory_bytes=262144000,
        )
        assert fp.layer_name == "embed_tokens.weight"
        assert fp.param_hash == "def456"


# ===================================================================
# SHARED LAYER TESTS
# ===================================================================

class TestSharedLayer:
    """SharedLayer dataclass -- construction and defaults."""

    def test_default_construction(self) -> None:
        fp = LayerFingerprint(
            layer_name="test", shape=(4, 4), dtype="float32",
            param_hash="h1", param_count=16, memory_bytes=64,
        )
        sl = SharedLayer(fingerprint=fp)
        assert sl.fingerprint is fp
        assert sl.tensor is None
        assert sl.ref_count == 0
        assert sl.model_names == []

    def test_full_construction(self) -> None:
        fp = LayerFingerprint(
            layer_name="test", shape=(2, 2), dtype="float32",
            param_hash="h2", param_count=4, memory_bytes=16,
        )
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sl = SharedLayer(
            fingerprint=fp,
            tensor=tensor,
            ref_count=2,
            model_names=["model-a", "model-b"],
        )
        assert sl.tensor is tensor
        assert sl.ref_count == 2
        assert sl.model_names == ["model-a", "model-b"]

    def test_model_names_independent_default_factory(self) -> None:
        """Default factory for model_names should give independent lists."""
        fp1 = LayerFingerprint(
            layer_name="a", shape=(1,), dtype="float32",
            param_hash="h1", param_count=1, memory_bytes=4,
        )
        fp2 = LayerFingerprint(
            layer_name="b", shape=(1,), dtype="float32",
            param_hash="h2", param_count=1, memory_bytes=4,
        )
        sl1 = SharedLayer(fingerprint=fp1)
        sl2 = SharedLayer(fingerprint=fp2)
        sl1.model_names.append("m1")
        assert "m1" not in sl2.model_names


# ===================================================================
# SHARED LAYER POOL TESTS
# ===================================================================

class TestSharedLayerPoolConstruction:
    """SharedLayerPool construction and defaults."""

    def test_default_construction(self) -> None:
        pool = SharedLayerPool()
        assert pool._threshold == 1.0
        assert pool._shared_layers == {}
        assert pool._model_layers == {}
        assert pool._total_shared_bytes == 0
        assert pool._total_models == 0

    def test_custom_threshold(self) -> None:
        pool = SharedLayerPool(similarity_threshold=0.95)
        assert pool._threshold == 0.95

    def test_stats_empty(self) -> None:
        pool = SharedLayerPool()
        stats = pool.stats()
        assert stats["total_models"] == 0
        assert stats["unique_layers"] == 0
        assert stats["total_shared_bytes"] == 0
        assert stats["models"] == []


class TestSharedLayerPoolRegisterModel:
    """SharedLayerPool.register_model -- happy path, sharing detection."""

    def test_register_single_model(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({
            "embed_tokens.weight": (32000, 4096),
            "layers.0.self_attn.q_proj.weight": (4096, 4096),
        })
        result = pool.register_model("llama-3-8b", sd)
        assert result["model_name"] == "llama-3-8b"
        assert result["total_layers"] == 2
        assert result["shared_layers"] == 0
        assert result["unique_layers"] == 2
        assert result["saved_bytes"] == 0
        assert result["saved_mb"] == 0.0
        assert pool._total_models == 1

    def test_register_two_identical_models_shares_layers(self) -> None:
        pool = SharedLayerPool()
        sd1 = make_state_dict({
            "embed.weight": (3200, 4096),
        })
        sd2 = make_state_dict({
            "embed.weight": (3200, 4096),
        })
        pool.register_model("model-a", sd1)
        result = pool.register_model("model-b", sd2)
        assert result["shared_layers"] == 1
        assert result["unique_layers"] == 0
        assert result["saved_bytes"] > 0
        assert result["saved_mb"] > 0.0

    def test_register_two_different_models_no_sharing(self) -> None:
        pool = SharedLayerPool()
        sd1 = make_state_dict({"embed_tokens.weight": (32000, 4096)})
        sd2 = make_state_dict({"embed_tokens.weight": (16000, 2048)})
        pool.register_model("model-a", sd1)
        result = pool.register_model("model-b", sd2)
        # Different shapes => different fingerprints => no sharing
        assert result["shared_layers"] == 0
        assert result["unique_layers"] == 1
        assert result["saved_bytes"] == 0

    def test_register_model_with_same_tensor_content(self) -> None:
        """Two tensors with identical shape and content get the same fingerprint."""
        pool = SharedLayerPool()
        # Different fill values means different content => different hashes
        sd1 = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd2 = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd1)
        result = pool.register_model("b", sd2)
        assert result["shared_layers"] == 1

    def test_register_model_with_different_fill_no_sharing(self) -> None:
        """Tensors with same shape but different content => different fingerprint."""
        pool = SharedLayerPool()
        sd1 = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd2 = make_state_dict({"w": (4, 4)}, fill_value=2.0)
        pool.register_model("a", sd1)
        result = pool.register_model("b", sd2)
        assert result["shared_layers"] == 0

    def test_register_model_multiple_layers_mixed(self) -> None:
        """Mixed case: some layers shared, some unique."""
        pool = SharedLayerPool()
        # Each layer gets unique content to avoid hash collisions
        sd1 = {
            "embed.weight": make_tensor((32000, 4096), fill_value=1.0),
            "layers.0.attn.weight": make_tensor((4096, 4096), fill_value=2.0),
            "head.weight": make_tensor((32000, 4096), fill_value=3.0),
        }
        sd2 = {
            "embed.weight": make_tensor((32000, 4096), fill_value=1.0),   # shared
            "layers.0.attn.weight": make_tensor((4096, 4096), fill_value=2.0),  # shared
            "head.weight": make_tensor((32000, 2048), fill_value=4.0),   # different shape
        }
        pool.register_model("a", sd1)
        result = pool.register_model("b", sd2)
        assert result["shared_layers"] == 2  # embed, attn shared
        assert result["unique_layers"] == 1  # head unique
        assert result["saved_bytes"] > 0

    def test_register_model_empty_state_dict(self) -> None:
        pool = SharedLayerPool()
        result = pool.register_model("empty-model", {})
        assert result["total_layers"] == 0
        assert result["shared_layers"] == 0
        assert result["unique_layers"] == 0
        assert result["saved_bytes"] == 0
        assert pool._total_models == 1

    def test_register_model_three_models_chain_sharing(self) -> None:
        """Third model shares layers with the existing shared layer."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=42.0)
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        result = pool.register_model("c", sd)
        assert result["shared_layers"] == 1
        # The shared layer should now have ref_count=3
        fp_hash = pool._model_layers["c"]["w"]
        assert pool._shared_layers[fp_hash].ref_count == 3
        assert pool._shared_layers[fp_hash].model_names == ["a", "b", "c"]

    def test_register_model_returns_stats_match_method(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w1": (4, 4), "w2": (8, 8)})
        pool.register_model("a", sd)
        stats = pool.stats()
        assert stats["total_models"] == 1
        assert stats["unique_layers"] == 2

    def test_register_model_same_model_name_twice(self) -> None:
        """Registering the same model name overwrites previous registration."""
        pool = SharedLayerPool()
        sd1 = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd2 = make_state_dict({"w": (4, 4)}, fill_value=2.0)
        pool.register_model("same-name", sd1)
        pool.register_model("same-name", sd2)
        # The second call overwrites: _model_layers["same-name"] is replaced
        assert pool._model_layers["same-name"] is not None
        # But the shared layers dict still contains the first fingerprint too
        assert len(pool._shared_layers) == 2  # both unique, since different content

    def test_register_different_dtypes_no_sharing(self) -> None:
        """Different dtypes produce different fingerprints."""
        pool = SharedLayerPool()
        sd1 = {"w": make_tensor((4, 4), dtype=torch.float32, fill_value=1.0)}
        sd2 = {"w": make_tensor((4, 4), dtype=torch.float16, fill_value=1.0)}
        pool.register_model("a", sd1)
        result = pool.register_model("b", sd2)
        assert result["shared_layers"] == 0


class TestSharedLayerPoolUnregisterModel:
    """SharedLayerPool.unregister_model -- ref counting, layer freeing."""

    def test_unregister_model_single(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        assert len(pool._shared_layers) == 1
        pool.unregister_model("a")
        assert pool._model_layers == {}
        assert len(pool._shared_layers) == 0
        assert pool._total_models == 0

    def test_unregister_model_shared_layer_ref_count(self) -> None:
        """Unregistering one model decrements the ref_count of shared layers."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        fp_hash = pool._model_layers["a"]["w"]
        assert pool._shared_layers[fp_hash].ref_count == 2
        pool.unregister_model("a")
        assert pool._shared_layers[fp_hash].ref_count == 1
        assert pool._model_layers == {"b": {"w": fp_hash}}  # "b" still registered

    def test_unregister_model_frees_when_ref_count_reaches_zero(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        pool.unregister_model("a")
        assert len(pool._shared_layers) == 1  # still held by "b"
        pool.unregister_model("b")
        assert len(pool._shared_layers) == 0  # both freed

    def test_unregister_model_nonexistent(self) -> None:
        """Unregistering a model that was never registered should not crash."""
        pool = SharedLayerPool()
        pool.unregister_model("never-registered")
        assert pool._total_models == 0

    def test_unregister_model_removes_model_name_from_list(self) -> None:
        pool = SharedLayerPool()
        sd1 = make_state_dict({"w1": (4, 4)})
        sd2 = make_state_dict({"w1": (4, 4)})
        pool.register_model("a", sd1)
        pool.register_model("b", sd2)
        fp_hash = pool._model_layers["a"]["w1"]
        assert "a" in pool._shared_layers[fp_hash].model_names
        pool.unregister_model("a")
        assert "a" not in pool._shared_layers[fp_hash].model_names

    def test_unregister_model_updates_total_models(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}))
        pool.register_model("b", make_state_dict({"w": (4, 4)}))
        assert pool._total_models == 2
        pool.unregister_model("a")
        assert pool._total_models == 1
        pool.unregister_model("b")
        assert pool._total_models == 0

    def test_unregister_model_frees_tensor(self) -> None:
        """When ref_count reaches zero, tensor should be deleted."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        fp_hash = pool._model_layers["a"]["w"]
        assert pool._shared_layers[fp_hash].tensor is not None
        pool.unregister_model("a")
        # Shared layer is deleted, so accessing would raise KeyError
        assert fp_hash not in pool._shared_layers


class TestSharedLayerPoolGetSharedTensor:
    """SharedLayerPool.get_shared_tensor -- accessing shared tensors."""

    def test_get_shared_tensor_shared_layer(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=42.0)
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        tensor = pool.get_shared_tensor("a", "w")
        assert tensor is not None
        assert tensor.shape == (4, 4)

    def test_get_shared_tensor_not_registered_model(self) -> None:
        pool = SharedLayerPool()
        result = pool.get_shared_tensor("nonexistent", "w")
        assert result is None

    def test_get_shared_tensor_not_registered_layer(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        result = pool.get_shared_tensor("a", "nonexistent_layer")
        assert result is None

    def test_get_shared_tensor_after_unregister(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        pool.unregister_model("a")
        result = pool.get_shared_tensor("a", "w")
        assert result is None


class TestSharedLayerPoolGetModelLayers:
    """SharedLayerPool.get_model_layers -- fingerprint map access."""

    def test_get_model_layers_existing(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w1": (4, 4), "w2": (8, 8)})
        pool.register_model("a", sd)
        layers = pool.get_model_layers("a")
        assert len(layers) == 2
        assert "w1" in layers
        assert "w2" in layers
        # values should be hex fingerprints (16-char strings)
        assert len(layers["w1"]) == 16
        assert len(layers["w2"]) == 16

    def test_get_model_layers_nonexistent(self) -> None:
        pool = SharedLayerPool()
        layers = pool.get_model_layers("does-not-exist")
        assert layers == {}

    def test_get_model_layers_returns_copy(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}))
        layers = pool.get_model_layers("a")
        # Mutating the returned dict should not affect internal state
        layers["new_layer"] = "fake_hash"
        assert "new_layer" not in pool._model_layers["a"]

    def test_get_model_layers_fingerprint_format(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=3.14)
        pool.register_model("a", sd)
        layers = pool.get_model_layers("a")
        fp_hash = layers["w"]
        # SHA-256 hexdigest truncated to 16 chars = 16 hex characters
        assert isinstance(fp_hash, str)
        assert len(fp_hash) == 16
        assert all(c in "0123456789abcdef" for c in fp_hash)


class TestSharedLayerPoolGetSavings:
    """SharedLayerPool.get_savings -- memory savings reporting."""

    def test_get_savings_empty(self) -> None:
        pool = SharedLayerPool()
        savings = pool.get_savings()
        assert savings["total_models"] == 0
        assert savings["unique_layers"] == 0
        assert savings["shared_references"] == 0
        assert savings["total_saved_bytes"] == 0

    def test_get_savings_after_one_model(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd)
        savings = pool.get_savings()
        assert savings["total_models"] == 1
        assert savings["unique_layers"] == 1
        assert savings["shared_references"] == 0
        assert savings["total_saved_bytes"] == 0

    def test_get_savings_after_sharing(self) -> None:
        pool = SharedLayerPool()
        # Use a larger tensor so saved_bytes / (1024**2) > 0
        sd = make_state_dict({"w": (128, 256)}, fill_value=1.0)
        pool.register_model("a", sd)
        result = pool.register_model("b", sd)
        savings = pool.get_savings()
        assert savings["total_models"] == 2
        assert savings["unique_layers"] == 1
        # ref_count=2 => shared_references = 2 - 1 = 1
        assert savings["shared_references"] == 1
        # saved_bytes should match the per-registration saved_bytes
        assert savings["total_saved_bytes"] == result["saved_bytes"]
        assert savings["total_saved_mb"] >= 0.0
        assert savings["total_saved_gb"] >= 0.0

    def test_get_savings_three_models_two_shared(self) -> None:
        pool = SharedLayerPool()
        sd_common = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd_unique = make_state_dict({"w": (8, 8)}, fill_value=2.0)
        pool.register_model("a", sd_common)
        pool.register_model("b", sd_common)
        pool.register_model("c", sd_unique)
        savings = pool.get_savings()
        assert savings["total_models"] == 3
        assert savings["unique_layers"] == 2
        # "w" from a and b are shared (ref_count=2 => 1 reference saved)
        # "w" from c is unique (ref_count=1 => 0 references saved)
        assert savings["shared_references"] == 1

    def test_get_savings_multiple_shared_layers(self) -> None:
        pool = SharedLayerPool()
        sd_a = make_state_dict({"w1": (4, 4), "w2": (8, 8)}, fill_value=1.0)
        sd_b = make_state_dict({"w1": (4, 4), "w2": (8, 8)}, fill_value=1.0)
        pool.register_model("a", sd_a)
        pool.register_model("b", sd_b)
        savings = pool.get_savings()
        assert savings["unique_layers"] == 2
        assert savings["shared_references"] == 2  # one per shared layer


class TestSharedLayerPoolFindSimilarModels:
    """SharedLayerPool.find_similar_models -- identifying similar models."""

    def test_find_similar_models_no_sharing(self) -> None:
        pool = SharedLayerPool()
        sd_a = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd_b = make_state_dict({"w": (8, 8)}, fill_value=2.0)
        pool.register_model("a", sd_a)
        pool.register_model("b", sd_b)
        result = pool.find_similar_models("a")
        assert result == []

    def test_find_similar_models_with_sharing(self) -> None:
        pool = SharedLayerPool()
        sd_a = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd_b = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd_a)
        pool.register_model("b", sd_b)
        result = pool.find_similar_models("a")
        assert len(result) == 1
        assert result[0]["model_name"] == "b"
        assert result[0]["shared_layers"] == 1
        assert result[0]["total_layers"] == 1
        assert result[0]["similarity"] == 1.0

    def test_find_similar_models_for_nonexistent_model(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}))
        result = pool.find_similar_models("nonexistent")
        assert result == []

    def test_find_similar_models_sorted_by_similarity(self) -> None:
        """Results should be sorted by shared_layers count descending."""
        pool = SharedLayerPool()
        # Unique fill values per layer to avoid hash collisions
        sd_a = {
            "w1": make_tensor((4, 4), fill_value=1.0),
            "w2": make_tensor((8, 8), fill_value=2.0),
            "w3": make_tensor((16, 16), fill_value=3.0),
        }
        sd_b = {
            "w1": make_tensor((4, 4), fill_value=1.0),  # shared
            "w2": make_tensor((8, 8), fill_value=2.0),  # shared
            "w3": make_tensor((16, 16), fill_value=3.0),  # shared
        }
        sd_c = {
            "w1": make_tensor((4, 4), fill_value=1.0),  # shared
            "w2": make_tensor((8, 8), fill_value=2.0),  # shared
        }
        pool.register_model("a", sd_a)
        pool.register_model("b", sd_b)
        pool.register_model("c", sd_c)
        result = pool.find_similar_models("a")
        # Both "b" and "c" share layers with "a". "b" shares 3, "c" shares 2.
        assert len(result) == 2
        assert result[0]["model_name"] == "b"
        assert result[0]["shared_layers"] == 3
        assert result[1]["model_name"] == "c"
        assert result[1]["shared_layers"] == 2

    def test_find_similar_models_partial_similarity(self) -> None:
        """Similarity should be computed as shared / total_layers."""
        pool = SharedLayerPool()
        # Build state dicts: w1/w2 shared (same shape+content), w3/w4 unique per model
        sd_a = {
            "w1": make_tensor((4, 4), fill_value=1.0),
            "w2": make_tensor((4, 4), fill_value=2.0),
            "w3": make_tensor((4, 4), fill_value=3.0),
            "w4": make_tensor((4, 4), fill_value=4.0),
        }
        sd_b = {
            "w1": make_tensor((4, 4), fill_value=1.0),  # shared with a
            "w2": make_tensor((4, 4), fill_value=2.0),  # shared with a
            "w3": make_tensor((4, 4), fill_value=5.0),  # unique
            "w4": make_tensor((4, 4), fill_value=6.0),  # unique
        }
        pool.register_model("a", sd_a)
        pool.register_model("b", sd_b)
        result = pool.find_similar_models("a")
        assert len(result) == 1
        assert result[0]["shared_layers"] == 2  # w1, w2 shared
        assert result[0]["total_layers"] == 4
        assert result[0]["similarity"] == 0.5

    def test_find_similar_models_excludes_self(self) -> None:
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        result = pool.find_similar_models("a")
        # Only "a" is registered, so no other models to find
        assert result == []

    def test_find_similar_models_no_shared_layers_with_others(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w1": (4, 4)}, fill_value=1.0))
        pool.register_model("b", make_state_dict({"w2": (8, 8)}, fill_value=2.0))
        result = pool.find_similar_models("a")
        assert result == []


class TestSharedLayerPoolStats:
    """SharedLayerPool.stats -- pool statistics."""

    def test_stats_empty(self) -> None:
        pool = SharedLayerPool()
        s = pool.stats()
        assert s["total_models"] == 0
        assert s["unique_layers"] == 0
        assert s["total_shared_bytes"] == 0
        assert s["models"] == []

    def test_stats_after_registrations(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}))
        pool.register_model("b", make_state_dict({"w": (8, 8)}))
        s = pool.stats()
        assert s["total_models"] == 2
        assert s["unique_layers"] == 2
        assert "a" in s["models"]
        assert "b" in s["models"]

    def test_stats_after_unregister(self) -> None:
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}))
        pool.register_model("b", make_state_dict({"w": (4, 4)}))
        pool.unregister_model("a")
        s = pool.stats()
        assert s["total_models"] == 1
        assert s["models"] == ["b"]


class TestSharedLayerPoolEdgeCases:
    """SharedLayerPool edge cases and error paths."""

    def test_register_with_identical_layers_shares(self) -> None:
        """Layers with same shape and content are shared between models."""
        pool = SharedLayerPool()
        sd_a = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        sd_b = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd_a)
        result = pool.register_model("b", sd_b)
        assert result["shared_layers"] == 1  # shared

    def test_unregister_model_with_tensor_none(self) -> None:
        """unregister should handle layers where tensor is None gracefully."""
        pool = SharedLayerPool()
        # We can't easily get a SharedLayer with tensor=None into the pool
        # because register_model always sets the tensor.  But the unregister
        # code checks ``if shared.tensor is not None``, so this exercises
        # the safety branch.
        fp = LayerFingerprint(
            layer_name="orphan", shape=(1,), dtype="float32",
            param_hash="orphan_hash", param_count=1, memory_bytes=4,
        )
        pool._shared_layers["orphan_hash"] = SharedLayer(
            fingerprint=fp,
            tensor=None,
            ref_count=1,
            model_names=["zombie"],
        )
        pool._model_layers["zombie"] = {"orphan": "orphan_hash"}
        pool._total_models = 1
        pool.unregister_model("zombie")
        assert "orphan_hash" not in pool._shared_layers

    def test_register_does_not_raise_on_duplicate_fingerprint(self) -> None:
        """Registering the same layer repeatedly should not raise."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        pool.register_model("c", sd)  # third time
        assert pool._total_models == 3
        assert len(pool._shared_layers) == 1  # single unique layer

    def test_get_savings_returns_same_object_after_mutation(self) -> None:
        """get_savings returns a new dict each time (no accidental mutation)."""
        pool = SharedLayerPool()
        s1 = pool.get_savings()
        s1["total_models"] = 999  # try to corrupt
        s2 = pool.get_savings()
        assert s2["total_models"] == 0

    def test_find_similar_models_sorted_by_count(self) -> None:
        """find_similar_models always sorts descending by shared_layers."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        result = pool.find_similar_models("a")
        assert len(result) == 1
        # Verify the sorting key is integer shared_layers
        assert result[0]["similarity"] == 1.0


class TestSharedLayerPoolThreadSafety:
    """SharedLayerPool thread safety -- basic concurrent access."""

    def test_concurrent_register(self) -> None:
        """Multiple threads can register models without corrupting state."""
        pool = SharedLayerPool()
        n_threads = 8
        errors: list[Exception] = []
        lock = threading.Lock()

        def register(idx: int) -> None:
            try:
                sd = make_state_dict(
                    {f"layer_{idx}": (4, 4)},
                    fill_value=float(idx),
                )
                pool.register_model(f"model-{idx}", sd)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent register raised: {errors}"
        assert pool._total_models == n_threads
        assert len(pool._shared_layers) == n_threads  # all unique layers

    def test_concurrent_register_shared_layer(self) -> None:
        """Multiple threads registering the same layer should share correctly."""
        pool = SharedLayerPool()
        n_threads = 8
        errors: list[Exception] = []
        lock = threading.Lock()
        sd = make_state_dict({"shared_layer": (4, 4)}, fill_value=42.0)

        def register(idx: int) -> None:
            try:
                pool.register_model(f"model-{idx}", sd)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert pool._total_models == n_threads
        assert len(pool._shared_layers) == 1  # single shared layer
        savings = pool.get_savings()
        assert savings["shared_references"] == n_threads - 1

    def test_concurrent_register_and_unregister(self) -> None:
        """Register and unregister from different threads should not deadlock."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("perm", sd)  # keep this model alive

        errors: list[Exception] = []
        lock = threading.Lock()

        def reg_unreg(ident: int) -> None:
            try:
                name = f"tmp-{ident}"
                pool.register_model(name, sd)
                pool.unregister_model(name)
                pool.register_model(name, sd)
                pool.unregister_model(name)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=reg_unreg, args=(i,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert pool._total_models == 1  # only "perm" remains
        assert "perm" in pool._model_layers


class TestSharedLayerPoolFingerprint:
    """_fingerprint_layer -- the hashing mechanism."""

    def test_fingerprint_consistent(self) -> None:
        """Same tensor produces same fingerprint."""
        pool = SharedLayerPool()
        tensor = make_tensor((64, 64), fill_value=7.0)
        fp1 = pool._fingerprint_layer("test", tensor)
        fp2 = pool._fingerprint_layer("test", tensor)
        assert fp1.param_hash == fp2.param_hash

    def test_fingerprint_different_content(self) -> None:
        """Different tensor content produces different hash."""
        pool = SharedLayerPool()
        t1 = make_tensor((64, 64), fill_value=1.0)
        t2 = make_tensor((64, 64), fill_value=2.0)
        fp1 = pool._fingerprint_layer("test", t1)
        fp2 = pool._fingerprint_layer("test", t2)
        assert fp1.param_hash != fp2.param_hash

    def test_fingerprint_different_shape(self) -> None:
        """Different shape produces different hash."""
        pool = SharedLayerPool()
        t1 = make_tensor((32, 64), fill_value=1.0)
        t2 = make_tensor((64, 32), fill_value=1.0)
        fp1 = pool._fingerprint_layer("test", t1)
        fp2 = pool._fingerprint_layer("test", t2)
        assert fp1.param_hash != fp2.param_hash

    def test_fingerprint_includes_dtype(self) -> None:
        """Different dtype produces different hash."""
        pool = SharedLayerPool()
        t1 = make_tensor((4, 4), dtype=torch.float32, fill_value=1.0)
        t2 = make_tensor((4, 4), dtype=torch.float16, fill_value=1.0)
        fp1 = pool._fingerprint_layer("test", t1)
        fp2 = pool._fingerprint_layer("test", t2)
        assert fp1.param_hash != fp2.param_hash

    def test_fingerprint_fields(self) -> None:
        pool = SharedLayerPool()
        tensor = make_tensor((8, 16), dtype=torch.float32, fill_value=3.0)
        fp = pool._fingerprint_layer("my_layer", tensor)
        assert fp.layer_name == "my_layer"
        assert fp.shape == (8, 16)
        assert fp.dtype == str(tensor.dtype)
        assert fp.param_count == 128
        assert fp.memory_bytes == 128 * 4  # float32 = 4 bytes
        assert len(fp.param_hash) == 16

    def test_fingerprint_large_tensor_uses_first_1024_bytes(self) -> None:
        """Hashing only uses first 1024 bytes + shape."""
        pool = SharedLayerPool()
        # Tensor with more than 1024 bytes of data
        t_a = make_tensor((1, 1024), dtype=torch.float32, fill_value=1.0)
        t_b = make_tensor((1, 1024), dtype=torch.float32, fill_value=1.0)
        fp_a = pool._fingerprint_layer("same", t_a)
        fp_b = pool._fingerprint_layer("same", t_b)
        assert fp_a.param_hash == fp_b.param_hash
        # Now t_a and t_b differ only in the 1025th byte region (first 1024 are same)
        # With 1024 floats at 4 bytes = 4096 bytes of data, first 1024 bytes = 256 floats
        t_a_diff = torch.cat([torch.full((256,), 1.0), torch.full((768,), 2.0)])
        t_b_diff = torch.cat([torch.full((256,), 1.0), torch.full((768,), 3.0)])
        # Both have same first 256 elements = 1024 bytes
        fp_a_diff = pool._fingerprint_layer("test", t_a_diff.unsqueeze(0))
        fp_b_diff = pool._fingerprint_layer("test", t_b_diff.unsqueeze(0))
        # Same first 1024 bytes + same shape => same hash
        assert fp_a_diff.param_hash == fp_b_diff.param_hash

    def test_fingerprint_layer_name_does_not_affect_hash(self) -> None:
        """The fingerprint hash only considers tensor bytes + shape, not layer_name."""
        pool = SharedLayerPool()
        tensor = make_tensor((4, 4), fill_value=5.0)
        fp1 = pool._fingerprint_layer("any_name", tensor)
        fp2 = pool._fingerprint_layer("different_name", tensor)
        assert fp1.param_hash == fp2.param_hash  # same content, same hash


class TestSharedLayerPoolIntegration:
    """Integration-level scenarios combining multiple operations."""

    def test_register_unregister_reuse_model_name(self) -> None:
        """Register, unregister, re-register a model name."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)})
        pool.register_model("reuse", sd)
        pool.unregister_model("reuse")
        assert pool._total_models == 0
        pool.register_model("reuse", sd)
        assert pool._total_models == 1
        assert pool.get_shared_tensor("reuse", "w") is not None

    def test_full_workflow(self) -> None:
        """Complete workflow: register models, verify sharing, query, unregister."""
        pool = SharedLayerPool()

        # Register three models with varying similarity
        base_layers = {
            "embed.weight": (32000, 4096),
            "layers.0.attn.q.weight": (4096, 4096),
            "layers.0.attn.k.weight": (4096, 1024),
            "layers.0.attn.v.weight": (4096, 1024),
            "layers.0.mlp.gate.weight": (4096, 14336),
            "layers.0.mlp.up.weight": (4096, 14336),
            "layers.0.mlp.down.weight": (14336, 4096),
            "head.weight": (32000, 4096),
        }

        # Model A: base Llama
        sd_a = {
            "embed.weight": make_tensor((32000, 4096), fill_value=1.0),
            "layers.0.attn.q.weight": make_tensor((4096, 4096), fill_value=10.0),
            "layers.0.attn.k.weight": make_tensor((4096, 1024), fill_value=20.0),
            "layers.0.attn.v.weight": make_tensor((4096, 1024), fill_value=30.0),
            "layers.0.mlp.gate.weight": make_tensor((4096, 14336), fill_value=40.0),
            "layers.0.mlp.up.weight": make_tensor((4096, 14336), fill_value=50.0),
            "layers.0.mlp.down.weight": make_tensor((14336, 4096), fill_value=60.0),
            "head.weight": make_tensor((32000, 4096), fill_value=70.0),
        }
        r_a = pool.register_model("llama-3-8b", sd_a)
        assert r_a["unique_layers"] == 8

        # Model B: instruct variant -- all layers same as model A
        sd_b = {
            "embed.weight": make_tensor((32000, 4096), fill_value=1.0),
            "layers.0.attn.q.weight": make_tensor((4096, 4096), fill_value=10.0),
            "layers.0.attn.k.weight": make_tensor((4096, 1024), fill_value=20.0),
            "layers.0.attn.v.weight": make_tensor((4096, 1024), fill_value=30.0),
            "layers.0.mlp.gate.weight": make_tensor((4096, 14336), fill_value=40.0),
            "layers.0.mlp.up.weight": make_tensor((4096, 14336), fill_value=50.0),
            "layers.0.mlp.down.weight": make_tensor((14336, 4096), fill_value=60.0),
            "head.weight": make_tensor((32000, 4096), fill_value=70.0),
        }
        r_b = pool.register_model("llama-3-8b-instruct", sd_b)
        assert r_b["shared_layers"] == 8
        assert r_b["unique_layers"] == 0
        assert r_b["saved_bytes"] > 0

        # Model C: different head shape and MLP shapes; other layers same content
        sd_c = {
            "embed.weight": make_tensor((32000, 4096), fill_value=1.0),   # shared
            "layers.0.attn.q.weight": make_tensor((4096, 4096), fill_value=10.0),  # shared
            "layers.0.attn.k.weight": make_tensor((4096, 1024), fill_value=20.0),  # shared
            "layers.0.attn.v.weight": make_tensor((4096, 1024), fill_value=30.0),  # shared
            "layers.0.mlp.gate.weight": make_tensor((4096, 14336), fill_value=40.0),  # shared
            "layers.0.mlp.up.weight": make_tensor((4096, 14336), fill_value=50.0),  # shared
            "layers.0.mlp.down.weight": make_tensor((2048, 4096), fill_value=80.0),  # different shape
            "head.weight": make_tensor((32000, 2048), fill_value=90.0),  # different shape
        }
        r_c = pool.register_model("llama-3-8b-lora", sd_c)
        # embed, attn layers shared = 6 shared, head + mlp.down unique = 2 unique
        assert r_c["shared_layers"] == 6
        assert r_c["unique_layers"] == 2

        # Verify savings
        savings = pool.get_savings()
        assert savings["total_models"] == 3
        assert savings["unique_layers"] > 0

        # Find similar models for model A
        similar = pool.find_similar_models("llama-3-8b")
        assert len(similar) == 2
        assert similar[0]["model_name"] == "llama-3-8b-instruct"
        assert similar[0]["similarity"] == 1.0
        assert similar[1]["model_name"] == "llama-3-8b-lora"
        assert similar[1]["similarity"] == pytest.approx(6 / 8)

        # Unregister model B
        pool.unregister_model("llama-3-8b-instruct")
        savings_after = pool.get_savings()
        assert savings_after["total_models"] == 2

        # Remaining models still share
        remaining_similar = pool.find_similar_models("llama-3-8b")
        assert len(remaining_similar) == 1
        assert remaining_similar[0]["model_name"] == "llama-3-8b-lora"

        # Unregister all
        pool.unregister_model("llama-3-8b")
        pool.unregister_model("llama-3-8b-lora")
        assert pool._total_models == 0
        assert len(pool._shared_layers) == 0

    def test_large_tensor_handling(self) -> None:
        """Registering a model with large tensors should work correctly."""
        pool = SharedLayerPool()
        # Use moderate sizes that won't OOM but are large enough for realistic
        # byte-count computations
        sd = {
            "embed.weight": make_tensor((32000, 1024), fill_value=1.0),  # ~125MB float32
            "head.weight": make_tensor((32000, 1024), fill_value=2.0),
        }
        result = pool.register_model("large-model", sd)
        assert result["total_layers"] == 2
        assert result["shared_layers"] == 0
        assert result["saved_mb"] == 0.0

    def test_get_shared_tensor_returns_same_object_for_shared(self) -> None:
        """Shared tensors should return the same underlying tensor object."""
        pool = SharedLayerPool()
        sd = make_state_dict({"w": (4, 4)}, fill_value=1.0)
        pool.register_model("a", sd)
        pool.register_model("b", sd)
        tensor_a = pool.get_shared_tensor("a", "w")
        tensor_b = pool.get_shared_tensor("b", "w")
        assert tensor_a is tensor_b  # same object in memory

    def test_find_similar_models_is_thread_safe(self) -> None:
        """find_similar_models holds the internal lock and returns consistent results."""
        pool = SharedLayerPool()
        pool.register_model("a", make_state_dict({"w": (4, 4)}, fill_value=1.0))
        pool.register_model("b", make_state_dict({"w": (4, 4)}, fill_value=1.0))
        result = pool.find_similar_models("a")
        assert len(result) == 1
