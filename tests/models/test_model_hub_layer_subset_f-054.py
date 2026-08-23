"""Regression tests for audit finding F-054.

Finding: ``ModelHub.download_layer_subset`` fetched shards via
``hf_hub_download`` WITHOUT ``local_dir`` (they landed in the HuggingFace
shared cache) and returned a layer directory containing only
``.layer_manifest`` -- no ``model.safetensors.index.json``, no shard
weights -- contradicting its documented contract and breaking any
consumer that loads weights from the returned directory.

These tests pin the fixed contract: the returned path physically
contains the safetensors index plus exactly the needed shard files,
stale manifest-only caches are re-downloaded, and incomplete downloads
raise ``DownloadError`` instead of returning a useless directory.
"""

import json
from pathlib import Path

import pytest

from distllm.models.model_hub import DownloadError, ModelHub
from distllm.models.safetensors_index import SafetensorsIndex

MODEL = "org/model"
INDEX_FILE = "model.safetensors.index.json"
SHARD_A = "model-00001-of-00002.safetensors"
SHARD_B = "model-00002-of-00002.safetensors"

# Shard A holds layer 0 (+ embeddings); shard B holds layer 5 only.
# Non-layer params (embed_tokens) are always treated as needed.
WEIGHT_MAP = {
    "model.embed_tokens.weight": SHARD_A,
    "model.layers.0.self_attn.q_proj.weight": SHARD_A,
    "model.layers.5.self_attn.q_proj.weight": SHARD_B,
}

PAYLOAD = b"fake-safetensors-payload"


def _write_manifest_only(layer_subdir: Path, shard_files: list[str]) -> None:
    """Recreate the cache state left behind by the pre-fix (buggy) code."""
    layer_subdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": MODEL,
        "revision": "main",
        "start_layer": 0,
        "end_layer": 1,
        "shard_files": shard_files,
        "num_params": 2,
        "total_size": 4096,
    }
    with open(layer_subdir / ".layer_manifest", "w") as f:
        json.dump(manifest, f)


def _make_fake_download(skip=(), write_empty=()):
    """Build an hf_hub_download stand-in enforcing the local_dir contract."""

    def fake_hf_hub_download(
        repo_id, filename, revision=None, token=None, local_dir=None
    ):
        assert local_dir is not None, (
            f"hf_hub_download called without local_dir for {filename!r}: "
            f"the file would land outside the returned layer directory "
            f"(F-054 regression)"
        )
        target = Path(local_dir) / filename
        if filename in skip:
            return str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PAYLOAD if filename not in write_empty else b"")
        return str(target)

    return fake_hf_hub_download


@pytest.fixture()
def hub(tmp_path):
    return ModelHub(cache_dir=str(tmp_path / "cache"))


@pytest.fixture()
def hf_mocks(monkeypatch):
    """Patch SafetensorsIndex.from_hub and count hf_hub_download calls."""
    index = SafetensorsIndex(
        {"weight_map": dict(WEIGHT_MAP), "metadata": {"total_size": 4096}}
    )
    monkeypatch.setattr(
        "distllm.models.safetensors_index.SafetensorsIndex.from_hub",
        lambda *a, **k: index,
    )
    calls: list[str] = []

    def fake_download(repo_id, filename, revision=None, token=None, local_dir=None):
        assert local_dir is not None, (
            f"hf_hub_download called without local_dir for {filename!r}: "
            f"the file would land outside the returned layer directory "
            f"(F-054 regression)"
        )
        calls.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PAYLOAD)
        return str(target)

    monkeypatch.setattr("distllm.models.model_hub.hf_hub_download", fake_download)
    return calls


class TestDownloadLayerSubsetF054:
    def test_returned_dir_contains_index_and_needed_shards(self, hub, hf_mocks):
        path = hub.download_layer_subset(MODEL, 0, 1)
        layer_dir = Path(path)

        # Returned path is the layer-scoped directory under the ModelHub cache.
        assert layer_dir == hub.cache_dir / MODEL / "main" / "layers_0_1"

        # The exact reliability probe from the finding must now hold.
        assert list(layer_dir.glob("*.safetensors")), (
            f"returned layer dir {layer_dir} contains no .safetensors files"
        )
        assert (layer_dir / INDEX_FILE).is_file()
        assert (layer_dir / SHARD_A).is_file()
        for name in (INDEX_FILE, SHARD_A):
            assert (layer_dir / name).stat().st_size > 0

        # Only the *needed* shards are present (shard B holds layer 5).
        assert not (layer_dir / SHARD_B).exists()

        # Manifest records the downloaded files and matches reality.
        manifest = json.loads((layer_dir / ".layer_manifest").read_text())
        assert manifest["shard_files"] == sorted([INDEX_FILE, SHARD_A])
        assert manifest["start_layer"] == 0
        assert manifest["end_layer"] == 1

    def test_cached_subset_short_circuits_without_redownload(self, hub, hf_mocks):
        first = hub.download_layer_subset(MODEL, 0, 1)
        assert len(hf_mocks) == 2  # index + one needed shard

        second = hub.download_layer_subset(MODEL, 0, 1)
        assert second == first
        assert len(hf_mocks) == 2  # no additional downloads on cache hit

    def test_stale_manifest_only_cache_is_redownloaded(self, hub, hf_mocks):
        # Simulate the directory the buggy version used to return:
        # only .layer_manifest, no weights, no index.
        stale_dir = hub.cache_dir / MODEL / "main" / "layers_0_1"
        _write_manifest_only(stale_dir, [INDEX_FILE, SHARD_A])

        path = hub.download_layer_subset(MODEL, 0, 1)

        # Must NOT trust the stale manifest: shards are re-downloaded
        # into the directory before it is handed back.
        assert len(hf_mocks) >= 2
        assert (Path(path) / SHARD_A).is_file()
        assert (Path(path) / INDEX_FILE).is_file()

    def test_resolve_layer_subset_ignores_stale_manifest_dir(self, hub, hf_mocks):
        stale_dir = hub.cache_dir / MODEL / "main" / "layers_0_1"
        _write_manifest_only(stale_dir, [INDEX_FILE, SHARD_A])

        path = hub.resolve_layer_subset(MODEL, 0, 1)

        assert Path(path) == stale_dir
        assert (Path(path) / SHARD_A).is_file(), (
            "resolve_layer_subset returned the stale manifest-only directory"
        )

    @pytest.mark.parametrize("kwargs", [{"skip": (SHARD_A,)}, {"write_empty": (SHARD_A,)}],
                             ids=["missing-file", "empty-file"])
    def test_incomplete_download_raises_and_writes_no_manifest(self, hub, monkeypatch, kwargs):
        index = SafetensorsIndex(
            {"weight_map": dict(WEIGHT_MAP), "metadata": {"total_size": 4096}}
        )
        monkeypatch.setattr(
            "distllm.models.safetensors_index.SafetensorsIndex.from_hub",
            lambda *a, **k: index,
        )
        monkeypatch.setattr(
            "distllm.models.model_hub.hf_hub_download",
            _make_fake_download(**kwargs),
        )

        with pytest.raises(DownloadError, match="incomplete"):
            hub.download_layer_subset(MODEL, 0, 1)

        layer_dir = hub.cache_dir / MODEL / "main" / "layers_0_1"
        assert not (layer_dir / ".layer_manifest").exists(), (
            "manifest written despite an incomplete download"
        )
