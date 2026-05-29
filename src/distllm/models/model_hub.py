"""HuggingFace model hub integration with caching and offline mode."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
    )
    from huggingface_hub.utils import (
        RepositoryNotFoundError,
    )
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


class ModelHubError(Exception):
    """Base exception for model hub operations."""


class ModelNotCachedError(ModelHubError):
    """Raised when a model is not available in cache and offline mode is enabled."""


class DownloadError(ModelHubError):
    """Raised when a model download fails."""


@dataclass
class ModelInfo:
    """Metadata about a model from HuggingFace."""
    model_id: str
    size_bytes: int = 0
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    revisions: list[str] = field(default_factory=lambda: ["main"])


@dataclass
class CachedModel:
    """Info about a locally cached model."""
    model_id: str
    revision: str
    path: str
    size_bytes: int
    downloaded_at: str


class ModelHub:
    """Manages model downloads from HuggingFace with caching, retry, and offline mode.

    Supports both full-model downloads and **layer-aware** downloads
    that fetch only the safetensors shards needed for a specific layer
    range — critical for distributed inference where each node only
    needs a subset of the model.

    Usage:
        hub = ModelHub()
        path = hub.download("roneneldan/TinyStories-1M")
        # Layer-aware:
        path = hub.download_layer_subset("meta-llama/Llama-3.1-8B", 0, 11)
        models = hub.list_cached()
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_retries: int = 3,
        download_timeout_s: int = 300,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.max_retries = max_retries
        self.download_timeout_s = download_timeout_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Layer-aware download ----

    def download_layer_subset(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        revision: str | None = None,
        token: str | None = None,
    ) -> str:
        """Download **only** the safetensors shards needed for a layer range.

        Instead of downloading the entire 70B model (140 GB on disk),
        this method downloads the single ``model.safetensors.index.json``
        (a few KB), determines which shard files contain the requested
        layers, and downloads only those shards.

        Layers are downloaded into the HuggingFace shared cache
        (``~/.cache/huggingface/hub/``) which deduplicates across
        concurrent workers.  A lightweight manifest is written to the
        ModelHub cache so that subsequent calls are instant.

        Args:
            model_name: HuggingFace model ID (e.g. ``"meta-llama/Llama-3.1-8B"``).
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (inclusive).
            revision: Git revision (branch, tag, or commit hash).
            token: HuggingFace API token for gated models.

        Returns:
            Local path to the layer-scoped model directory.  The
            directory contains ``model.safetensors.index.json`` plus
            only the needed ``.safetensors`` shard files.

        Raises:
            ModelHubError: If ``huggingface-hub`` is not installed.
            DownloadError: If the index download fails after retries.
        """
        if not HAS_HF_HUB:
            raise ModelHubError(
                "huggingface-hub is not installed. Run: pip install huggingface-hub"
            )

        revision = revision or "main"
        model_cache_path = self.cache_dir / model_name / revision
        layer_subdir = model_cache_path / f"layers_{start_layer}_{end_layer}"
        manifest_path = layer_subdir / ".layer_manifest"

        # --- Check if already cached ---
        if manifest_path.exists():
            logger.info(
                f"Layer subset {start_layer}-{end_layer} for "
                f"{model_name}@{revision} already cached at {layer_subdir}"
            )
            return str(layer_subdir)

        # --- 1. Download & parse the index ---
        from distllm.models.safetensors_index import SafetensorsIndex

        last_error = None
        index: SafetensorsIndex | None = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching safetensors index for {model_name}@{revision} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                index = SafetensorsIndex.from_hub(model_name, revision, token)
                break
            except Exception as e:
                last_error = e
                wait = 2**attempt
                logger.warning(f"Index fetch failed: {e}. Retrying in {wait}s...")
                from time import sleep
                sleep(wait)

        if index is None:
            raise DownloadError(
                f"Failed to download safetensors index for {model_name} "
                f"after {self.max_retries} attempts: {last_error}"
            )

        needed_shards = index.get_shards_for_layer_range(start_layer, end_layer)
        logger.info(
            f"Layer range {start_layer}-{end_layer} needs "
            f"{len(needed_shards)}/{len(index.all_shard_files)} shards "
            f"({len(index.get_keys_for_layer_range(start_layer, end_layer))} params)"
        )

        if len(needed_shards) > 1 or (
            len(needed_shards) == 1
            and "index.json" not in next(iter(needed_shards))
        ):
            index_path = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors.index.json",
                revision=revision,
                token=token,
            )
            logger.debug(f"Index cached at {index_path}")

            for shard in sorted(needed_shards):
                if shard == "model.safetensors.index.json":
                    continue
                logger.debug(f"Downloading shard: {shard}")
                hf_hub_download(
                    repo_id=model_name,
                    filename=shard,
                    revision=revision,
                    token=token,
                )
        else:
            logger.info(
                f"Model {model_name} uses a single safetensors file; "
                f"falling back to standard download for non-shard files"
            )
            return self.download(
                model_name,
                revision=revision,
                token=token,
                allow_patterns=[
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "config.json",
                    "*.json",
                    "tokenizer*",
                    "special_tokens_map*",
                ],
            )

        # --- 2. Write layer-scoped manifest ---
        layer_subdir.mkdir(parents=True, exist_ok=True)
        import time as time_module

        needed_keys = index.get_keys_for_layer_range(start_layer, end_layer)
        manifest: dict = {
            "model_id": model_name,
            "revision": revision,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "downloaded_at": time_module.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time_module.gmtime()
            ),
            "shard_files": sorted(needed_shards),
            "num_params": len(needed_keys),
            "total_size": index.total_size,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Layer subset {start_layer}-{end_layer} for {model_name} "
            f"cached ({len(needed_shards)} shards, {len(needed_keys)} params)"
        )
        return str(layer_subdir)

    def resolve_layer_subset(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        revision: str | None = None,
        token: str | None = None,
    ) -> str:
        """Resolve a layer subset, downloading if necessary.

        Like :meth:`download_layer_subset` but returns the standard
        model cache path if the full model is already cached.
        """
        revision = revision or "main"
        model_cache_path = self.cache_dir / model_name / revision
        layer_subdir = model_cache_path / f"layers_{start_layer}_{end_layer}"
        manifest_path = layer_subdir / ".layer_manifest"

        if manifest_path.exists():
            return str(layer_subdir)

        if model_cache_path.exists() and (model_cache_path / ".manifest").exists():
            logger.info(
                f"Full model {model_name}@{revision} already cached; "
                f"using it instead of layer subset"
            )
            return str(model_cache_path)

        return self.download_layer_subset(
            model_name, start_layer, end_layer,
            revision=revision, token=token,
        )

    @staticmethod
    def _default_cache_dir() -> Path:
        """Default cache directory: ~/.cache/distributed-llm/models"""
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "distributed-llm" / "models"

    def download(
        self,
        model_name: str,
        revision: str | None = None,
        token: str | None = None,
        allow_patterns: list[str] | None = None,
        progress_callback=None,
    ) -> str:
        """Download a model from HuggingFace to the local cache.

        Args:
            model_name: HuggingFace model ID (e.g., "roneneldan/TinyStories-1M").
            revision: Specific git revision (branch, tag, or commit hash).
            token: HuggingFace API token for gated models.
            allow_patterns: Only download files matching these glob patterns.
            progress_callback: Optional callback(current, total, filename).

        Returns:
            Local path to the downloaded model directory.

        Raises:
            DownloadError: If download fails after retries.
            ModelHubError: If huggingface-hub is not installed.
        """
        if not HAS_HF_HUB:
            raise ModelHubError(
                "huggingface-hub is not installed. Run: pip install huggingface-hub"
            )

        revision = revision or "main"
        model_cache_path = self.cache_dir / model_name / revision

        # Check if already cached
        if model_cache_path.exists() and (model_cache_path / ".manifest").exists():
            logger.info(f"Model {model_name}@{revision} already cached at {model_cache_path}")
            return str(model_cache_path)

        # Download with retry
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Downloading {model_name}@{revision} (attempt {attempt + 1}/{self.max_retries})"
                )
                downloaded_path = snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    token=token,
                    cache_dir=str(self.cache_dir),
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
                # Write manifest
                self._write_manifest(model_name, revision, downloaded_path)
                logger.info(f"Downloaded {model_name} to {downloaded_path}")
                return downloaded_path
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise DownloadError(
            f"Failed to download {model_name} after {self.max_retries} attempts: {last_error}"
        )

    def is_available(
        self,
        model_name: str,
        revision: str | None = None,
    ) -> bool:
        """Check if a model is available in the local cache."""
        revision = revision or "main"
        model_path = self.cache_dir / model_name / revision
        return model_path.exists() and (model_path / ".manifest").exists()

    def resolve(
        self,
        model_name: str,
        revision: str | None = None,
        offline_mode: bool = False,
        token: str | None = None,
    ) -> str:
        """Resolve a model name to a local path, downloading if necessary.

        Args:
            model_name: Model ID or local path.
            revision: Specific revision to use.
            offline_mode: If True, only use cached models.
            token: HF API token.

        Returns:
            Local path to the model.

        Raises:
            ModelNotCachedError: If offline_mode is True and model not cached.
        """
        # If it looks like a local path, use it directly
        if os.path.isdir(model_name):
            return model_name

        revision = revision or "main"

        if self.is_available(model_name, revision):
            return str(self.cache_dir / model_name / revision)

        if offline_mode:
            raise ModelNotCachedError(
                f"Model {model_name} not in cache and offline mode is enabled"
            )

        return self.download(model_name, revision=revision, token=token)

    def list_cached(self) -> list[CachedModel]:
        """List all models in the local cache."""
        models = []
        if not self.cache_dir.exists():
            return models

        for model_dir in self.cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for rev_dir in model_dir.iterdir():
                if not rev_dir.is_dir():
                    continue
                manifest = rev_dir / ".manifest"
                if manifest.exists():
                    with open(manifest) as f:
                        data = json.load(f)
                    models.append(CachedModel(
                        model_id=model_dir.name,
                        revision=rev_dir.name,
                        path=str(rev_dir),
                        size_bytes=data.get("size_bytes", 0),
                        downloaded_at=data.get("downloaded_at", ""),
                    ))
        return models

    def remove(self, model_name: str, revision: str | None = None) -> bool:
        """Remove a cached model (and all revisions if revision is None).

        Returns:
            True if something was removed, False if nothing found.
        """
        model_dir = self.cache_dir / model_name
        if not model_dir.exists():
            return False

        if revision:
            rev_dir = model_dir / revision
            if rev_dir.exists():
                import shutil
                shutil.rmtree(rev_dir)
                logger.info(f"Removed {model_name}@{revision} from cache")
                # Remove model dir if empty
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
                return True
            return False
        else:
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"Removed {model_name} from cache")
            return True

    def get_info(self, model_name: str, token: str | None = None) -> ModelInfo | None:
        """Fetch model metadata from HuggingFace API."""
        if not HAS_HF_HUB:
            raise ModelHubError("huggingface-hub is not installed")

        try:
            api = HfApi()
            info = api.model_info(model_name, token=token)
            return ModelInfo(
                model_id=model_name,
                size_bytes=info.siblings_total_size if hasattr(info, "siblings_total_size") else 0,
                tags=info.tags if hasattr(info, "tags") else [],
                pipeline_tag=info.pipeline_tag if hasattr(info, "pipeline_tag") else "",
                downloads=info.downloads if hasattr(info, "downloads") else 0,
                likes=getattr(info, "likes", 0),
                last_modified=getattr(info, "last_modified", ""),
                revisions=["main"],  # Would need separate API call for full list
            )
        except RepositoryNotFoundError:
            return None
        except Exception as e:  # HfHubHTTPError, etc.
            logger.warning(f"Failed to fetch info for {model_name}: {e}")
            return None

    def warm_cache(
        self,
        model_names: list[str],
        revision: str | None = None,
        token: str | None = None,
        max_concurrent: int = 2,
    ) -> dict[str, str]:
        """Pre-download multiple models.

        Returns:
            Dict mapping model_name -> local_path for successfully downloaded models.
        """
        results = {}
        # Sequential download (concurrent would require async)
        for name in model_names:
            try:
                path = self.download(name, revision=revision, token=token)
                results[name] = path
            except DownloadError as e:
                logger.error(f"Failed to download {name}: {e}")
        return results

    def get_cache_size(self) -> int:
        """Total cache size in bytes."""
        total = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        return total

    @staticmethod
    def _write_manifest(model_name: str, revision: str, model_path: str) -> None:
        """Write a manifest file with download metadata."""
        manifest_path = Path(model_path) / ".manifest"
        size_bytes = 0
        for p in Path(model_path).rglob("*"):
            if p.is_file():
                size_bytes += p.stat().st_size

        manifest = {
            "model_id": model_name,
            "revision": revision,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size_bytes": size_bytes,
            "files": [str(p.relative_to(model_path)) for p in Path(model_path).rglob("*") if p.is_file()],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
