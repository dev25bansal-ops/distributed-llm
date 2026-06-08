"""N3: Point-in-time cache export/import.

Enables debugging cache issues offline, migrating caches between
clusters, and reproducing cache-dependent bugs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


class CacheSnapshot:
    """Export/import cache state for debugging and migration.

    Captures the full state of all cache tiers at a point in time.
    """

    def __init__(self, cache_manager: Any = None):
        self._cache_manager = cache_manager

    def export_snapshot(self, path: str) -> dict:
        """Export cache state to a file.

        Args:
            path: Path to save the snapshot (.json or .pt).

        Returns:
            Snapshot metadata dict.
        """
        snapshot = {
            "version": 1,
            "timestamp": time.time(),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Export prefix cache
        if self._cache_manager and hasattr(self._cache_manager, 'prefix_cache'):
            cache = self._cache_manager.prefix_cache
            if cache is not None:
                try:
                    snapshot["prefix_cache"] = {
                        "stats": cache.stats(),
                        "entries": self._serialize_prefix_cache(cache),
                    }
                except Exception as e:
                    logger.warning(f"Failed to export prefix cache: {e}")

        # Export predictive cache patterns
        if self._cache_manager and hasattr(self._cache_manager, '_predictive_cache'):
            pc = self._cache_manager._predictive_cache
            if pc is not None and hasattr(pc, 'learner'):
                try:
                    patterns = pc.learner.top_patterns(1000)
                    snapshot["predictive_patterns"] = [
                        {
                            "prefix": list(p.prefix_tokens),
                            "frequency": p.frequency,
                            "hit_count": p.hit_count,
                            "score": p.score,
                        }
                        for p in patterns
                    ]
                except Exception as e:
                    logger.warning(f"Failed to export predictive patterns: {e}")

        # Export tier stats
        if self._cache_manager and hasattr(self._cache_manager, 'get_tier_stats'):
            try:
                snapshot["tier_stats"] = self._cache_manager.get_tier_stats()
            except Exception:
                pass

        # Save to file
        filepath = Path(path)
        if filepath.suffix == ".pt":
            try:
                import torch
                torch.save(snapshot, path)
            except ImportError:
                # Fallback to JSON
                filepath = filepath.with_suffix(".json")
                with open(filepath, "w") as f:
                    json.dump(snapshot, f, indent=2, default=str)
        else:
            with open(filepath, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)

        logger.info(f"Cache snapshot exported to {filepath}")
        return snapshot

    def import_snapshot(self, path: str) -> dict:
        """Import cache state from a file.

        Args:
            path: Path to the snapshot file.

        Returns:
            The imported snapshot dict.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        if filepath.suffix == ".pt":
            import torch
            snapshot = torch.load(path, weights_only=True, map_location="cpu")
        else:
            with open(filepath) as f:
                snapshot = json.load(f)

        logger.info(f"Cache snapshot imported from {path}")

        # Restore predictive patterns
        if "predictive_patterns" in snapshot:
            if self._cache_manager and hasattr(self._cache_manager, '_predictive_cache'):
                pc = self._cache_manager._predictive_cache
                if pc is not None and hasattr(pc, 'learner'):
                    try:
                        for pdata in snapshot["predictive_patterns"]:
                            prefix = tuple(pdata["prefix"])
                            if prefix not in pc.learner._patterns:
                                from distllm.dist.predictive_cache import PrefixPattern
                                pc.learner._patterns[prefix] = PrefixPattern(
                                    prefix_tokens=prefix,
                                    frequency=pdata.get("frequency", 0),
                                    hit_count=pdata.get("hit_count", 0),
                                    score=pdata.get("score", 0),
                                    last_seen=pdata.get("last_seen", time.time()),
                                )
                        logger.info(f"Restored {len(snapshot['predictive_patterns'])} predictive patterns")
                    except Exception as e:
                        logger.warning(f"Failed to restore predictive patterns: {e}")

        return snapshot

    def _serialize_prefix_cache(self, cache: Any) -> list[dict]:
        """Serialize prefix cache entries for export."""
        entries = []
        try:
            if hasattr(cache, '_cache'):
                for key, entry in cache._cache.items():
                    entries.append({
                        "key": str(key),
                        "token_count": len(entry.get("tokens", [])),
                        "has_kv_data": entry.get("kv_data") is not None,
                        "stored_at": entry.get("stored_at", 0),
                    })
        except Exception:
            pass
        return entries

    def diff_snapshots(self, snapshot_a: dict, snapshot_b: dict) -> dict:
        """Compare two snapshots.

        Returns:
            Dict with differences.
        """
        diff = {
            "time_diff_s": snapshot_b.get("timestamp", 0) - snapshot_a.get("timestamp", 0),
            "prefix_cache_diff": {},
        }

        stats_a = snapshot_a.get("prefix_cache", {}).get("stats", {})
        stats_b = snapshot_b.get("prefix_cache", {}).get("stats", {})

        for key in set(list(stats_a.keys()) + list(stats_b.keys())):
            val_a = stats_a.get(key, 0)
            val_b = stats_b.get(key, 0)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                if val_a != val_b:
                    diff["prefix_cache_diff"][key] = {"before": val_a, "after": val_b, "delta": val_b - val_a}

        return diff
