"""Tests for ModelStore — shared model cache for distributed layer loading.

Tests the public API surface with real ModelStore instances backed by
temporary directories.  No mocks, no GPU, no network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from distllm.dist.model_store import ModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_layer(store: ModelStore, model: str, start: int, end: int,
                       revision: str = "main") -> Path:
    """Write a minimal fake .pt file so the store thinks layers exist."""
    path = Path(store.save_layer_weights(model, start, end, revision=revision))
    path.write_bytes(b"\x00\x01\x02\x03")
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestModelStoreConstruction:
    """ModelStore initialisation edge cases."""

    def test_default_cache_dir(self):
        """Default cache dir falls back to ~/.cache/distllm/models."""
        store = ModelStore()
        expected = Path.home() / ".cache" / "distllm" / "models"
        assert store._cache_dir == expected
        assert expected.exists()  # __init__ calls mkdir(parents=True, exist_ok=True)

    def test_custom_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "my_cache"
        store = ModelStore(cache_dir=str(cache_dir))
        assert store._cache_dir == cache_dir
        assert cache_dir.exists()

    def test_cache_dir_created_if_not_exist(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        store = ModelStore(cache_dir=str(nested))
        assert nested.exists()

    def test_cache_dir_already_exists(self, tmp_path: Path):
        nested = tmp_path / "already_exists"
        nested.mkdir(parents=True)
        (nested / "some_file.txt").write_text("hello")
        store = ModelStore(cache_dir=str(nested))
        assert nested.exists()
        assert (nested / "some_file.txt").exists()  # did not blow away

    def test_cache_dir_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DISTLLM_MODEL_CACHE", str(tmp_path / "from_env"))
        store = ModelStore()  # no argument -> reads env
        assert store._cache_dir == tmp_path / "from_env"
        assert store._cache_dir.exists()


# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------


class TestModelPath:
    """model_path returns the right directory for a model + revision."""

    def test_simple_name(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p = store.model_path("gpt2")
        assert p == tmp_path / "gpt2" / "main"

    def test_name_with_slash(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p = store.model_path("meta-llama/Llama-2-7b")
        # Slash is replaced with underscore
        assert p == tmp_path / "meta-llama_Llama-2-7b" / "main"

    def test_custom_revision(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p = store.model_path("gpt2", revision="v2.0")
        assert p == tmp_path / "gpt2" / "v2.0"

    def test_model_path_is_posix(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p = store.model_path("gpt2", revision="main")
        assert isinstance(p, Path)


# ---------------------------------------------------------------------------
# has_layers / get_layer_path
# ---------------------------------------------------------------------------


class TestHasLayers:
    """has_layers inspects the file system correctly."""

    def test_no_layers_initially(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert store.has_layers("gpt2", 0, 5) is False

    def test_after_writing_layer(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        assert store.has_layers("gpt2", 0, 5) is True

    def test_different_range_not_present(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        assert store.has_layers("gpt2", 6, 11) is False

    def test_different_model_not_present(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        assert store.has_layers("gpt3", 0, 5) is False

    def test_after_save_layer_manifest_only(self, tmp_path: Path):
        """Manifest alone should not make has_layers return True."""
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        assert store.has_layers("gpt2", 0, 5) is False

    def test_different_revision(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5, revision="v1")
        assert store.has_layers("gpt2", 0, 5, revision="v1") is True
        assert store.has_layers("gpt2", 0, 5, revision="main") is False

    def test_single_layer(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 3, 3)
        assert store.has_layers("gpt2", 3, 3) is True
        assert store.has_layers("gpt2", 2, 3) is False


class TestGetLayerPath:
    """get_layer_path returns the correct path or None."""

    def test_returns_none_for_missing(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert store.get_layer_path("gpt2", 0, 5) is None

    def test_returns_string_for_cached(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        path = store.get_layer_path("gpt2", 0, 5)
        assert isinstance(path, str)
        assert path.endswith("layers_0_5.pt")

    def test_returns_existing_file(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        path = store.get_layer_path("gpt2", 0, 5)
        assert path is not None
        assert Path(path).exists()

    def test_different_model_returns_none(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        assert store.get_layer_path("gpt3", 0, 5) is None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestSaveLayerManifest:
    """save_layer_manifest writes valid JSON metadata."""

    def test_creates_manifest(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        manifest_path = store.model_path("gpt2") / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_content(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12, revision="v2")
        manifest_path = store.model_path("gpt2", revision="v2") / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["model_name"] == "gpt2"
        assert data["revision"] == "v2"
        assert data["total_layers"] == 12

    def test_manifest_overwrite(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        store.save_layer_manifest("gpt2", 24)  # same model, updated layers
        data = json.loads((store.model_path("gpt2") / "manifest.json").read_text())
        assert data["total_layers"] == 24

    def test_manifest_with_weird_name(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("org/model", 8)
        data = json.loads(
            (store.model_path("org/model") / "manifest.json").read_text()
        )
        assert data["model_name"] == "org/model"


# ---------------------------------------------------------------------------
# Save layer weights (returns a path, doesn't write)
# ---------------------------------------------------------------------------


class TestSaveLayerWeights:
    """save_layer_weights returns a valid, writeable path."""

    def test_returns_str(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        path = store.save_layer_weights("gpt2", 0, 5)
        assert isinstance(path, str)

    def test_path_has_correct_format(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        path = store.save_layer_weights("gpt2", 2, 7)
        assert path.endswith("layers_2_7.pt")
        assert "gpt2" in path

    def test_directory_created(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p = Path(store.save_layer_weights("gpt2", 0, 5))
        assert p.parent.exists()

    def test_writing_to_returned_path(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        path = Path(store.save_layer_weights("gpt2", 0, 5))
        path.write_bytes(b"\x00" * 100)
        assert path.stat().st_size == 100

    def test_integer_edge_cases(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        p0 = Path(store.save_layer_weights("gpt2", 0, 0))
        assert p0.name == "layers_0_0.pt"
        p_large = Path(store.save_layer_weights("gpt2", 2**31 - 1, 2**31))
        assert "layers_2147483647_2147483648.pt" in p_large.name


# ---------------------------------------------------------------------------
# List cached models
# ---------------------------------------------------------------------------


class TestListCachedModels:
    """list_cached_models enumerates models from disk."""

    def test_empty_cache(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert store.list_cached_models() == []

    def test_single_model_with_manifest(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        models = store.list_cached_models()
        assert len(models) == 1
        assert models[0]["model_name"] == "gpt2"

    def test_multiple_models(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        store.save_layer_manifest("llama", 32)
        models = store.list_cached_models()
        assert len(models) == 2
        names = {m["model_name"] for m in models}
        assert names == {"gpt2", "llama"}

    def test_model_without_manifest(self, tmp_path: Path):
        """Model directory with no manifest should still appear with total_layers=0."""
        store = ModelStore(cache_dir=str(tmp_path))
        # Create a revision directory directly without calling save_layer_manifest
        d = store.model_path("gpt2")
        d.mkdir(parents=True, exist_ok=True)
        # Write a fake layer to give evidence of a model
        (d / "layers_0_5.pt").write_bytes(b"data")
        models = store.list_cached_models()
        assert len(models) == 1
        assert models[0]["model_name"] == "gpt2"
        assert models[0]["total_layers"] == 0

    def test_revisions_appear_as_separate_entries(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12, revision="v1")
        store.save_layer_manifest("gpt2", 12, revision="v2")
        models = store.list_cached_models()
        assert len(models) == 2
        revs = {m["revision"] for m in models}
        assert revs == {"v1", "v2"}

    def test_empty_cache_dir_returns_empty_list(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert store.list_cached_models() == []

    def test_cache_dir_does_not_exist(self, tmp_path: Path):
        non_existent = tmp_path / "nonexistent"
        store = ModelStore(cache_dir=str(non_existent))
        # __init__ creates it, but then we remove it to test the guard
        # Actually __init__ creates it, so we need a different approach:
        # just verify the internal guard works if dir is missing.
        import shutil
        shutil.rmtree(non_existent)
        assert store.list_cached_models() == []


# ---------------------------------------------------------------------------
# Cache size
# ---------------------------------------------------------------------------


class TestCacheSizeBytes:
    """cache_size_bytes returns aggregate file sizes."""

    def test_empty_cache(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert store.cache_size_bytes() == 0

    def test_single_file(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        assert store.cache_size_bytes() == 4

    def test_multiple_models(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        _write_fake_layer(store, "llama", 0, 5)
        assert store.cache_size_bytes() == 8  # 4 bytes each

    def test_manifest_counted(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12)
        size = store.cache_size_bytes()
        assert size > 0  # manifest has actual JSON content

    def test_result_is_int(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        assert isinstance(store.cache_size_bytes(), int)

    def test_empty_directory_returns_zero(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.model_path("empty_model").mkdir(parents=True, exist_ok=True)
        assert store.cache_size_bytes() == 0


# ---------------------------------------------------------------------------
# Integration-style: round trips
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end check that saving and reading back works."""

    def test_save_then_check(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        _write_fake_layer(store, "gpt2", 0, 5)
        store.save_layer_manifest("gpt2", 12)

        assert store.has_layers("gpt2", 0, 5) is True
        path = store.get_layer_path("gpt2", 0, 5)
        assert path is not None
        assert Path(path).read_bytes() == b"\x00\x01\x02\x03"

    def test_save_then_list(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("gpt2", 12, revision="main")
        models = store.list_cached_models()
        assert models[0]["model_name"] == "gpt2"
        assert models[0]["total_layers"] == 12
        assert models[0]["revision"] == "main"

    def test_persists_across_store_instances(self, tmp_path: Path):
        store_a = ModelStore(cache_dir=str(tmp_path))
        store_a.save_layer_manifest("gpt2", 12)
        _write_fake_layer(store_a, "gpt2", 0, 5)

        store_b = ModelStore(cache_dir=str(tmp_path))
        assert store_b.has_layers("gpt2", 0, 5) is True
        assert len(store_b.list_cached_models()) == 1

    def test_layer_with_none_range_edge(self, tmp_path: Path):
        """Edge case: very large layer numbers should still work."""
        store = ModelStore(cache_dir=str(tmp_path))
        # Use realistic layer indices (e.g., 40-79 for a 80-layer model shard)
        _write_fake_layer(store, "large_model", 40, 79)
        assert store.has_layers("large_model", 40, 79) is True


# ---------------------------------------------------------------------------
# Environment variable interaction
# ---------------------------------------------------------------------------


class TestEnvVar:
    """DISTLLM_MODEL_CACHE environment variable."""

    def test_env_var_not_set_uses_default(self, monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: Path):
        monkeypatch.delenv("DISTLLM_MODEL_CACHE", raising=False)
        store = ModelStore()
        expected = Path.home() / ".cache" / "distllm" / "models"
        assert store._cache_dir == expected

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path):
        monkeypatch.setenv("DISTLLM_MODEL_CACHE", str(tmp_path / "env_cache"))
        store = ModelStore()
        assert store._cache_dir == tmp_path / "env_cache"

    def test_explicit_constructor_wins_over_env(self, monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path):
        monkeypatch.setenv("DISTLLM_MODEL_CACHE", str(tmp_path / "env_cache"))
        store = ModelStore(cache_dir=str(tmp_path / "explicit"))
        assert store._cache_dir == tmp_path / "explicit"


# ---------------------------------------------------------------------------
# Edge cases: model names with special characters
# ---------------------------------------------------------------------------


class TestSpecialCharacters:
    """Model names with slashes, dots, or unusual characters."""

    def test_model_name_with_dashes(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("my-model-v2", 8)
        models = store.list_cached_models()
        assert models[0]["model_name"] == "my-model-v2"

    def test_model_name_with_dots(self, tmp_path: Path):
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("model.v3.1", 8)
        models = store.list_cached_models()
        assert models[0]["model_name"] == "model.v3.1"

    def test_model_name_with_org_slash_recovered_on_list(self, tmp_path: Path):
        """When listing, models stored with org/name have their slash restored
        only if they lack a manifest.  With a manifest, the original model_name
        is stored in the JSON."""
        store = ModelStore(cache_dir=str(tmp_path))
        store.save_layer_manifest("org/model", 12)
        models = store.list_cached_models()
        assert models[0]["model_name"] == "org/model"  # from manifest JSON
        assert models[0]["total_layers"] == 12

    def test_model_name_without_manifest_slash_restored(self, tmp_path: Path):
        """Without a manifest, the directory name has underscores that came from
        slashes — they get restored as slashes."""
        store = ModelStore(cache_dir=str(tmp_path))
        # Simulate what happens when save_layer_weights is called for org/model
        p = Path(store.save_layer_weights("org/model", 0, 5))
        p.write_bytes(b"data")
        models = store.list_cached_models()
        assert models[0]["model_name"] == "org/model"
