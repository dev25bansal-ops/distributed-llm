"""Model cache management with LRU eviction and disk usage tracking."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class ModelCache:
    """Manages the model cache directory with LRU eviction.

    Tracks disk usage and evicts least-recently-used models when
    the cache exceeds the maximum size.

    Usage:
        cache = ModelCache(cache_dir="/path/to/cache", max_size_gb=50.0)
        cache.ensure_model_tracked("model/path", "roneneldan/TinyStories-1M")
        cache.evict_if_needed()
    """

    METADATA_FILE = ".cache_metadata.json"

    def __init__(self, cache_dir: Optional[str] = None, max_size_gb: float = 50.0):
        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()

    @staticmethod
    def _default_cache_dir() -> Path:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "distributed-llm" / "models"

    def _load_metadata(self) -> dict:
        """Load cache metadata file."""
        meta_path = self.cache_dir / self.METADATA_FILE
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted cache metadata, rebuilding")
        return {"entries": {}, "total_size_bytes": 0}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        meta_path = self.cache_dir / self.METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def get_disk_usage(self) -> int:
        """Current total cache size in bytes."""
        return self._metadata.get("total_size_bytes", 0)

    def get_disk_usage_gb(self) -> float:
        """Current total cache size in GB."""
        return self.get_disk_usage() / (1024 * 1024 * 1024)

    def get_usage_pct(self) -> float:
        """Cache usage as percentage of max."""
        if self.max_size_bytes == 0:
            return 0.0
        return min(100.0, (self.get_disk_usage() / self.max_size_bytes) * 100.0)

    def list_entries(self) -> List[dict]:
        """List all cached models with metadata."""
        entries = self._metadata.get("entries", {})
        result = []
        for model_id, info in entries.items():
            result.append({
                "model_id": model_id,
                "path": info.get("path", ""),
                "size_bytes": info.get("size_bytes", 0),
                "last_accessed": info.get("last_accessed", 0),
                "last_modified": info.get("last_modified", 0),
            })
        # Sort by last_accessed (LRU)
        result.sort(key=lambda x: x["last_accessed"])
        return result

    def ensure_model_tracked(self, model_path: str, model_id: str) -> None:
        """Register a model in the cache metadata."""
        path = Path(model_path)
        if not path.exists():
            return

        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        now = time.time()

        self._metadata["entries"][model_id] = {
            "path": str(path),
            "size_bytes": size_bytes,
            "last_accessed": now,
            "last_modified": now,
        }
        self._recalculate_total()
        self._save_metadata()

    def touch(self, model_id: str) -> None:
        """Update last_accessed time for a model (marks as recently used)."""
        entries = self._metadata.get("entries", {})
        if model_id in entries:
            entries[model_id]["last_accessed"] = time.time()
            self._save_metadata()

    def remove_entry(self, model_id: str, delete_files: bool = True) -> bool:
        """Remove a model from cache metadata and optionally delete files."""
        entries = self._metadata.get("entries", {})
        if model_id not in entries:
            return False

        info = entries[model_id]
        if delete_files:
            path = Path(info["path"])
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Deleted cache files for {model_id}")

        del entries[model_id]
        self._recalculate_total()
        self._save_metadata()
        return True

    def evict_if_needed(self, target_pct: float = 90.0) -> List[str]:
        """Evict least-recently-used models until cache is below target percentage.

        Args:
            target_pct: Evict until cache usage is below this percentage.

        Returns:
            List of evicted model IDs.
        """
        evicted = []
        target_bytes = int(self.max_size_bytes * target_pct / 100.0)

        while self.get_disk_usage() > target_bytes:
            entries = self.list_entries()
            if not entries:
                break

            # Evict the LRU entry
            lru = entries[0]
            model_id = lru["model_id"]
            logger.info(
                f"Evicting {model_id} ({lru['size_bytes'] / 1e9:.1f} GB) "
                f"to free cache space"
            )
            self.remove_entry(model_id, delete_files=True)
            evicted.append(model_id)

        return evicted

    def get_available_space(self) -> int:
        """Remaining cache space in bytes."""
        return max(0, self.max_size_bytes - self.get_disk_usage())

    def can_fit(self, size_bytes: int) -> bool:
        """Check if a model of given size can fit in the cache."""
        return size_bytes <= self.get_available_space()

    def _recalculate_total(self) -> None:
        """Recalculate total cache size from entries."""
        entries = self._metadata.get("entries", {})
        total = sum(info.get("size_bytes", 0) for info in entries.values())
        self._metadata["total_size_bytes"] = total

    def reset(self) -> None:
        """Clear all cache metadata (does not delete files)."""
        self._metadata = {"entries": {}, "total_size_bytes": 0}
        self._save_metadata()
