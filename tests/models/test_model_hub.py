"""Tests for Feature 26: Model Hub Integration."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distllm.models.model_hub import (
    ModelHub,
    ModelInfo,
    CachedModel,
    ModelHubError,
    ModelNotCachedError,
    DownloadError,
)


class TestModelHub:
    def test_init_uses_default_cache_dir(self):
        hub = ModelHub()
        assert "distributed-llm" in str(hub.cache_dir)

    def test_init_uses_custom_cache_dir(self, tmp_path):
        hub = ModelHub(cache_dir=str(tmp_path / "custom"))
        assert hub.cache_dir == tmp_path / "custom"

    def test_download_creates_model_files(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"

        def mock_download(repo_id, revision=None, token=None, cache_dir=None, allow_patterns=None, resume_download=True):
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}")
            manifest_path = model_dir / ".manifest"
            with open(manifest_path, "w") as f:
                json.dump({"model_id": repo_id, "revision": revision or "main", "downloaded_at": "", "size_bytes": 0, "files": ["config.json"]}, f)
            return str(model_dir)

        monkeypatch.setattr("distllm.models.model_hub.snapshot_download", mock_download)
        monkeypatch.setattr("distllm.models.model_hub.HAS_HF_HUB", True)

        hub = ModelHub(cache_dir=str(cache_dir))
        result = hub.download("test-org/test-model")

        assert result == str(model_dir)
        assert (model_dir / ".manifest").exists()

    def test_download_returns_cached_path_if_exists(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / ".manifest").write_text("{}")

        monkeypatch.setattr("distllm.models.model_hub.HAS_HF_HUB", True)
        hub = ModelHub(cache_dir=str(cache_dir))

        # Should not call snapshot_download since model is cached
        with patch("distllm.models.model_hub.snapshot_download") as mock_dl:
            result = hub.download("test-org/test-model")
            mock_dl.assert_not_called()

        assert result == str(model_dir)

    def test_is_available_returns_true_for_cached_model(self, tmp_path):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / ".manifest").write_text("{}")

        hub = ModelHub(cache_dir=str(cache_dir))
        assert hub.is_available("test-org/test-model") is True

    def test_is_available_returns_false_for_uncached_model(self, tmp_path):
        hub = ModelHub(cache_dir=str(tmp_path / "cache"))
        assert hub.is_available("nonexistent/model") is False

    def test_resolve_returns_local_path_if_directory(self, tmp_path):
        local_path = tmp_path / "local_model"
        local_path.mkdir()

        hub = ModelHub(cache_dir=str(tmp_path / "cache"))
        result = hub.resolve(str(local_path))
        assert result == str(local_path)

    def test_resolve_returns_cached_path_if_available(self, tmp_path):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / ".manifest").write_text("{}")

        hub = ModelHub(cache_dir=str(cache_dir))
        result = hub.resolve("test-org/test-model")
        assert result == str(model_dir)

    def test_resolve_raises_when_offline_and_not_cached(self, tmp_path):
        hub = ModelHub(cache_dir=str(tmp_path / "cache"))

        with pytest.raises(ModelNotCachedError):
            hub.resolve("nonexistent/model", offline_mode=True)

    def test_list_cached_returns_empty_for_empty_cache(self, tmp_path):
        hub = ModelHub(cache_dir=str(tmp_path / "cache"))
        assert hub.list_cached() == []

    def test_list_cached_returns_downloaded_models(self, tmp_path):
        cache_dir = tmp_path / "cache"
        # list_cached iterates {cache_dir}/{model_id}/{revision}/
        # So use a flat model_id without slashes
        model_dir = cache_dir / "test-model" / "main"
        model_dir.mkdir(parents=True)
        manifest = {
            "model_id": "test-model",
            "revision": "main",
            "downloaded_at": "2024-01-01T00:00:00Z",
            "size_bytes": 12345,
            "files": ["config.json"],
        }
        with open(model_dir / ".manifest", "w") as f:
            json.dump(manifest, f)

        hub = ModelHub(cache_dir=str(cache_dir))
        models = hub.list_cached()

        assert len(models) == 1
        assert models[0].model_id == "test-model"
        assert models[0].size_bytes == 12345

    def test_remove_deletes_model_files(self, tmp_path):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / ".manifest").write_text("{}")

        hub = ModelHub(cache_dir=str(cache_dir))
        result = hub.remove("test-org/test-model")

        assert result is True
        assert not model_dir.exists()

    def test_remove_returns_false_for_nonexistent_model(self, tmp_path):
        hub = ModelHub(cache_dir=str(tmp_path / "cache"))
        assert hub.remove("nonexistent/model") is False

    def test_get_cache_size_returns_bytes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "test-org" / "test-model" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / "file.txt").write_text("hello")
        (model_dir / ".manifest").write_text("{}")

        hub = ModelHub(cache_dir=str(cache_dir))
        size = hub.get_cache_size()
        assert size > 0

    def test_warm_cache_downloads_multiple_models(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        call_count = {"count": 0}

        def mock_download(repo_id, revision=None, token=None, cache_dir=None, allow_patterns=None, resume_download=True):
            model_dir = Path(cache_dir) / repo_id / (revision or "main")
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / ".manifest").write_text("{}")
            call_count["count"] += 1
            return str(model_dir)

        monkeypatch.setattr("distllm.models.model_hub.snapshot_download", mock_download)
        monkeypatch.setattr("distllm.models.model_hub.HAS_HF_HUB", True)

        hub = ModelHub(cache_dir=str(cache_dir))
        results = hub.warm_cache(["model-a", "model-b"])

        assert len(results) == 2
        assert "model-a" in results
        assert "model-b" in results
        assert call_count["count"] == 2


class TestModelHubErrorHandling:
    def test_model_hub_error(self):
        with pytest.raises(ModelHubError):
            raise ModelHubError("test error")

    def test_model_not_cached_error(self):
        with pytest.raises(ModelNotCachedError):
            raise ModelNotCachedError("not cached")

    def test_download_error(self):
        with pytest.raises(DownloadError):
            raise DownloadError("download failed")


class TestModelHubWithoutHFHub:
    def test_raises_when_huggingface_hub_not_installed(self, monkeypatch):
        monkeypatch.setattr("distllm.models.model_hub.HAS_HF_HUB", False)
        hub = ModelHub()

        with pytest.raises(ModelHubError, match="huggingface-hub is not installed"):
            hub.download("test/model")
