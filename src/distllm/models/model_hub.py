"""HuggingFace model hub integration with caching and offline mode."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

try:
    from huggingface_hub import (
        snapshot_download,
        hf_hub_download,
        HfApi,
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
    tags: List[str] = field(default_factory=list)
    pipeline_tag: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    revisions: List[str] = field(default_factory=lambda: ["main"])


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

    Usage:
        hub = ModelHub()
        path = hub.download("roneneldan/TinyStories-1M")
        models = hub.list_cached()
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        download_timeout_s: int = 300,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.max_retries = max_retries
        self.download_timeout_s = download_timeout_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_cache_dir() -> Path:
        """Default cache directory: ~/.cache/distributed-llm/models"""
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "distributed-llm" / "models"

    def download(
        self,
        model_name: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
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
        revision: Optional[str] = None,
    ) -> bool:
        """Check if a model is available in the local cache."""
        revision = revision or "main"
        model_path = self.cache_dir / model_name / revision
        return model_path.exists() and (model_path / ".manifest").exists()

    def resolve(
        self,
        model_name: str,
        revision: Optional[str] = None,
        offline_mode: bool = False,
        token: Optional[str] = None,
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

    def list_cached(self) -> List[CachedModel]:
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

    def remove(self, model_name: str, revision: Optional[str] = None) -> bool:
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

    def get_info(self, model_name: str, token: Optional[str] = None) -> Optional[ModelInfo]:
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
        model_names: List[str],
        revision: Optional[str] = None,
        token: Optional[str] = None,
        max_concurrent: int = 2,
    ) -> Dict[str, str]:
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
