"""Thread-safe model registry with version pinning and A/B testing.

Features:
- Versioned model entries (multiple versions per model name)
- A/B traffic splitting between model versions
- Performance metric tracking per version
- Thread-safe registry with eviction
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field


class ModelNotFoundError(KeyError):
    pass


@dataclass
class ModelVersion:
    """A single version of a model."""
    version_id: str
    path: str
    total_layers: int
    precision: str = "float16"
    vram_mb: float = 0.0
    throughput_tok_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    registered_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelEntry:
    """A model name with its versions and A/B routing config."""
    name: str
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    default_version: str | None = None
    ab_test_enabled: bool = False
    ab_traffic_split: dict[str, float] = field(default_factory=dict)

    @property
    def version_count(self) -> int:
        return len(self.versions)

    def get_active_version(self, routing_key: str | None = None) -> str | None:
        if not self.versions:
            return None
        if not self.ab_test_enabled or not self.ab_traffic_split:
            return self.default_version or next(iter(self.versions))

        if routing_key is not None:
            total = sum(self.ab_traffic_split.values())
            if total <= 0:
                return self.default_version or next(iter(self.versions))
            bucket = abs(hash(routing_key)) % 10000 / 100.0
            cumulative = 0.0
            for ver, pct in sorted(self.ab_traffic_split.items()):
                cumulative += pct
                if bucket < cumulative:
                    return ver if ver in self.versions else self.default_version

        return self.default_version or next(iter(self.versions))


class ModelRegistry:
    """Thread-safe model registry with versioning, A/B testing, and metrics."""

    def __init__(self, max_models: int = 8):
        self._max_models = max_models
        self._models: dict[str, ModelEntry] = {}
        self._default_model: str | None = None
        self._lock = threading.RLock()

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register_version(
        self,
        name: str,
        version_id: str,
        path: str,
        total_layers: int,
        precision: str = "float16",
        metadata: dict | None = None,
        set_as_default: bool = False,
    ) -> ModelVersion:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                if len(self._models) >= self._max_models:
                    raise ValueError(
                        f"Maximum models ({self._max_models}) exceeded. "
                        f"Remove a model before adding '{name}'."
                    )
                entry = ModelEntry(name=name)
                self._models[name] = entry
                if self._default_model is None:
                    self._default_model = name

            if version_id in entry.versions:
                existing = entry.versions[version_id]
                existing.path = path
                existing.total_layers = total_layers
                existing.precision = precision
                existing.registered_at = time.time()
                if metadata:
                    existing.metadata.update(metadata)
                return existing

            ver = ModelVersion(
                version_id=version_id,
                path=path,
                total_layers=total_layers,
                precision=precision,
                metadata=metadata or {},
            )
            entry.versions[version_id] = ver
            if entry.default_version is None or set_as_default:
                entry.default_version = version_id
            return ver

    def remove_version(self, name: str, version_id: str) -> bool:
        with self._lock:
            entry = self._models.get(name)
            if entry is None or version_id not in entry.versions:
                return False
            del entry.versions[version_id]
            if entry.default_version == version_id:
                entry.default_version = next(iter(entry.versions), None)
            if not entry.versions:
                del self._models[name]
                if self._default_model == name:
                    self._default_model = next(iter(self._models), None)
            return True

    def remove_model(self, name: str) -> bool:
        with self._lock:
            if name not in self._models:
                return False
            del self._models[name]
            if self._default_model == name:
                self._default_model = next(iter(self._models), None)
            return True

    # -----------------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------------

    def get(self, name: str) -> ModelEntry | None:
        with self._lock:
            return self._models.get(name)

    def get_version(self, name: str, version_id: str | None = None,
                    routing_key: str | None = None) -> ModelVersion | None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return None
            vid = version_id or entry.get_active_version(routing_key)
            if vid is None:
                return None
            ver = entry.versions.get(vid)
            if ver is not None:
                ver.last_used = time.time()
                ver.use_count += 1
            return ver

    def list_models(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())

    def list_versions(self, name: str) -> list[ModelVersion]:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return []
            return list(entry.versions.values())

    @property
    def default_model(self) -> str | None:
        with self._lock:
            return self._default_model

    @default_model.setter
    def default_model(self, name: str) -> None:
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(name)
            self._default_model = name

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._models

    # -----------------------------------------------------------------------
    # A/B Testing
    # -----------------------------------------------------------------------

    def configure_ab_test(self, name: str, traffic_split: dict[str, float]) -> None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                raise ModelNotFoundError(name)
            total = sum(traffic_split.values())
            if abs(total - 100.0) > 0.01:
                raise ValueError(f"Traffic split must total 100%, got {total}%")
            for vid in traffic_split:
                if vid not in entry.versions:
                    raise ValueError(f"Version '{vid}' not registered for model '{name}'")
            entry.ab_test_enabled = True
            entry.ab_traffic_split = traffic_split

    def disable_ab_test(self, name: str, pin_version: str | None = None) -> None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return
            entry.ab_test_enabled = False
            entry.ab_traffic_split = {}
            if pin_version is not None and pin_version in entry.versions:
                entry.default_version = pin_version

    def get_ab_status(self, name: str) -> dict | None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return None
            if not entry.ab_test_enabled:
                return {"enabled": False}
            return {
                "enabled": True,
                "traffic_split": dict(entry.ab_traffic_split),
                "versions": {
                    vid: {"model_path": ver.path, "use_count": ver.use_count}
                    for vid, ver in entry.versions.items()
                },
            }

    def set_default_version(self, name: str, version_id: str) -> bool:
        with self._lock:
            entry = self._models.get(name)
            if entry is None or version_id not in entry.versions:
                return False
            entry.default_version = version_id
            return True

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def record_metrics(
        self,
        name: str,
        version_id: str,
        throughput: float | None = None,
        latency_p50: float | None = None,
        latency_p99: float | None = None,
        vram_mb: float | None = None,
    ) -> None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return
            ver = entry.versions.get(version_id)
            if ver is None:
                return
            if throughput is not None:
                ver.throughput_tok_s = throughput
            if latency_p50 is not None:
                ver.latency_p50_ms = latency_p50
            if latency_p99 is not None:
                ver.latency_p99_ms = latency_p99
            if vram_mb is not None:
                ver.vram_mb = vram_mb

    def get_best_version(self, name: str, metric: str = "throughput") -> str | None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None or not entry.versions:
                return None
            best = None
            best_val = -1.0
            for vid, ver in entry.versions.items():
                val = getattr(ver, metric, 0.0)
                if val > best_val:
                    best_val = val
                    best = vid
            return best

    def promote_best_version(self, name: str, metric: str = "throughput") -> str | None:
        best = self.get_best_version(name, metric)
        if best is not None:
            self.set_default_version(name, best)
        return best

    def pin_model(self, name: str, version_id: str) -> dict:
        """Pin a model to a specific version, disabling A/B traffic splitting.

        Args:
            name: Model name.
            version_id: Version ID to pin.

        Returns:
            Dict confirming the pin.
        """
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                raise ValueError(f"Model '{name}' not found")
            if version_id not in entry.versions:
                raise ValueError(f"Version '{version_id}' not found for model '{name}'")
            entry.ab_test_enabled = False
            entry.ab_traffic_split.clear()
            entry.default_version = version_id
        return {"model": name, "pinned_version": version_id, "ab_test_enabled": False}

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        with self._lock:
            lines = [f"Model Registry ({len(self._models)} models)"]
            for name, entry in self._models.items():
                default = entry.default_version or "none"
                ab = " [A/B]" if entry.ab_test_enabled else ""
                lines.append(f"  {name}{ab}: {entry.version_count} versions, default={default}")
                for vid, ver in entry.versions.items():
                    mark = " [default]" if vid == default else ""
                    ab_pct = f" {entry.ab_traffic_split.get(vid, 0):.0f}%" if entry.ab_test_enabled else ""
                    lines.append(f"    {vid}{mark}{ab_pct}: {ver.path} ({ver.precision}, "
                                 f"{ver.throughput_tok_s:.0f} tok/s)")
            return "\n".join(lines)
