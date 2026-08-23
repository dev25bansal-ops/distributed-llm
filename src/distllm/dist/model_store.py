"""Shared model cache for distributed layer loading.

When multiple workers need the same model layers, the ModelStore
ensures only one download from HuggingFace — subsequent nodes load
from the shared cache (local FS or NFS mount).
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

from loguru import logger


class ModelStore:
    """Manages a shared directory of downloaded model layers.

    Workers check the store before downloading from HuggingFace.
    If layers exist in the store, they load from disk instead.

    The store is a directory structure:
        <cache_dir>/<model_name>/<revision>/layers_<start>_<end>.pt
        <cache_dir>/<model_name>/<revision>/manifest.json
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.environ.get(
                "DISTLLM_MODEL_CACHE",
                str(Path.home() / ".cache" / "distllm" / "models"),
            )
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelStore initialized at {self._cache_dir}")

    def model_path(self, model_name: str, revision: str = "main") -> Path:
        safe_name = model_name.replace("/", "_")
        return self._cache_dir / safe_name / revision

    def has_layers(self, model_name: str, start_layer: int, end_layer: int,
                   revision: str = "main") -> bool:
        """Check if specific layers exist in the cache."""
        path = self.model_path(model_name, revision)
        layer_file = path / f"layers_{start_layer}_{end_layer}.pt"
        return layer_file.exists()

    def get_layer_path(self, model_name: str, start_layer: int, end_layer: int,
                       revision: str = "main") -> str | None:
        """Get path to cached layers, or None if not cached."""
        path = self.model_path(model_name, revision)
        layer_file = path / f"layers_{start_layer}_{end_layer}.pt"
        if layer_file.exists():
            return str(layer_file)
        return None

    def save_layer_manifest(self, model_name: str, total_layers: int,
                            revision: str = "main") -> None:
        """Save or update the manifest for a model."""
        path = self.model_path(model_name, revision)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        manifest = {
            "model_name": model_name,
            "revision": revision,
            "total_layers": total_layers,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.debug(f"Saved manifest for {model_name}@{revision}")

    def save_layer_weights(self, model_name: str, start_layer: int, end_layer: int,
                           revision: str = "main") -> str:
        """Get the path to save layer weights to."""
        path = self.model_path(model_name, revision)
        path.mkdir(parents=True, exist_ok=True)
        return str(path / f"layers_{start_layer}_{end_layer}.pt")

    def list_cached_models(self) -> list[dict]:
        """List all models available in the cache."""
        results = []
        if not self._cache_dir.exists():
            return results
        for model_dir in self._cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for rev_dir in model_dir.iterdir():
                manifest_path = rev_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    results.append(manifest)
                else:
                    results.append({
                        "model_name": model_dir.name.replace("_", "/"),
                        "revision": rev_dir.name,
                        "total_layers": 0,
                    })
        return results

    def cache_size_bytes(self) -> int:
        """Total size of all cached models."""
        total = 0
        for f in self._cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
