"""Model Version Manager — versioning, rollback, shadow/A/B testing.

Manages multiple versions of a model simultaneously, supports canary
rollouts, shadow deployments, A/B testing splits, and instant rollback
when metrics degrade.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class VersionStatus(Enum):
    STAGED = "staged"
    ACTIVE = "active"
    SHADOW = "shadow"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """A single version of a model."""
    version_id: str
    model_name: str
    model_path: str
    dtype: str = "float16"
    status: VersionStatus = VersionStatus.STAGED
    deployed_at: float = 0.0
    promoted_at: float | None = None
    sha256: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class ABTestSplit:
    """Traffic split configuration for A/B testing."""
    version_a: str = ""
    version_b: str = ""
    split_ratio: float = 0.5  # fraction going to version_b
    metrics_collected: int = 0
    started_at: float = 0.0


@dataclass
class CanaryStage:
    """A single stage in a canary rollout."""
    name: str = ""
    traffic_pct: float = 0.0
    min_duration_s: float = 120.0
    rollback_threshold: float = 0.05  # max acceptable metric degradation


class ModelVersionManager:
    """Manage model versions, rollouts, and A/B tests.

    Thread-safe for concurrent coordinator access::

        mgr = ModelVersionManager(model_name="llama-70b")
        mgr.register_version("v1", "/models/llama-70b-v1")
        mgr.register_version("v2", "/models/llama-70b-v2")
        mgr.promote("v2")  # activate v2
        mgr.rollback()      # revert to v1
    """

    def __init__(
        self,
        model_name: str = "",
        max_versions: int = 5,
        state_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_versions = max_versions
        self._state_path = Path(state_path) if state_path else Path(".model_versions.json")
        self._lock = threading.Lock()
        self._versions: dict[str, ModelVersion] = {}
        self._active_version: str = ""
        self._ab_test: ABTestSplit | None = None
        self._canary_stages: list[CanaryStage] = []
        self._rollback_history: list[str] = []
        self._promote_hooks: list[Callable] = []

    # ── Version registration ────────────────────────────────────────────

    def register_version(
        self,
        version_id: str,
        model_path: str,
        dtype: str = "float16",
        tags: list[str] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> bool:
        """Register a new model version.

        If this is the first version, it is automatically promoted.
        """
        with self._lock:
            if version_id in self._versions:
                logger.warning(f"Version {version_id} already registered")
                return False
            if len(self._versions) >= self.max_versions:
                oldest = min(self._versions.keys(),
                             key=lambda k: self._versions[k].deployed_at)
                self._versions.pop(oldest, None)
                logger.info(f"Evicted oldest version {oldest}")

            self._versions[version_id] = ModelVersion(
                version_id=version_id,
                model_name=self.model_name,
                model_path=model_path,
                dtype=dtype,
                status=VersionStatus.STAGED,
                deployed_at=time.time(),
                tags=tags or [],
                config_overrides=config_overrides or {},
            )
            logger.info(f"Registered version {version_id} ({model_path})")

            if not self._active_version:
                self._active_version = version_id
                self._versions[version_id].status = VersionStatus.ACTIVE
                self._versions[version_id].promoted_at = time.time()

            self._save_state()
            return True

    def unregister_version(self, version_id: str) -> bool:
        """Remove a version from management."""
        with self._lock:
            if version_id == self._active_version:
                logger.warning(f"Cannot unregister active version {version_id}")
                return False
            old = self._versions.pop(version_id, None)
            if old:
                self._save_state()
            return old is not None

    # ── Promotion / rollback ────────────────────────────────────────────

    def promote(self, version_id: str) -> bool:
        """Promote *version_id* to active. Returns False if not found."""
        with self._lock:
            if version_id not in self._versions:
                logger.error(f"Version {version_id} not found")
                return False
            previous = self._active_version
            self._active_version = version_id
            now = time.time()
            self._versions[version_id].status = VersionStatus.ACTIVE
            self._versions[version_id].promoted_at = now
            if previous and previous in self._versions:
                self._versions[previous].status = VersionStatus.DEPRECATED
                self._rollback_history.append(previous)
            logger.info(f"Promoted version {version_id} to active")
            self._save_state()
            for hook in self._promote_hooks:
                try:
                    hook(version_id, previous)
                except Exception as e:
                    logger.warning(f"Promote hook failed: {e}")
            return True

    def rollback(self) -> str | None:
        """Rollback to the previous active version.

        Returns the version ID rolled back to, or None if no history.
        """
        with self._lock:
            if not self._rollback_history:
                logger.warning("No rollback history available")
                return None
            previous = self._rollback_history.pop()
            if previous not in self._versions:
                logger.warning(f"Previous version {previous} no longer registered")
                return None
            current = self._active_version
            self._active_version = previous
            self._versions[previous].status = VersionStatus.ACTIVE
            self._versions[previous].promoted_at = time.time()
            if current and current in self._versions:
                self._versions[current].status = VersionStatus.ROLLED_BACK
            logger.info(f"Rolled back from {current} to {previous}")
            self._save_state()
            return previous

    # ── Shadow deployment ───────────────────────────────────────────────

    def set_shadow(self, version_id: str) -> bool:
        """Set *version_id* as shadow (receives copy of traffic, not served)."""
        with self._lock:
            if version_id not in self._versions:
                return False
            self._versions[version_id].status = VersionStatus.SHADOW
            logger.info(f"Set version {version_id} as shadow")
            self._save_state()
            return True

    def clear_shadow(self) -> None:
        """Remove shadow status from all versions."""
        with self._lock:
            for v in self._versions.values():
                if v.status == VersionStatus.SHADOW:
                    v.status = VersionStatus.DEPRECATED
            self._save_state()

    # ── A/B testing ─────────────────────────────────────────────────────

    def start_ab_test(
        self, version_a: str, version_b: str, split_ratio: float = 0.5
    ) -> bool:
        """Start an A/B test between two versions."""
        with self._lock:
            if version_a not in self._versions or version_b not in self._versions:
                return False
            self._ab_test = ABTestSplit(
                version_a=version_a, version_b=version_b,
                split_ratio=split_ratio, started_at=time.time(),
            )
            logger.info(f"Started A/B test: {version_a} vs {version_b} "
                        f"(split={split_ratio:.0%})")
            self._save_state()
            return True

    def stop_ab_test(self) -> ABTestSplit | None:
        """Stop the current A/B test."""
        with self._lock:
            result = self._ab_test
            self._ab_test = None
            self._save_state()
            return result

    def select_ab_version(self, request_id: str) -> str | None:
        """Select which version serves *request_id* based on A/B split."""
        if self._ab_test is None:
            return self._active_version or None
        # Deterministic selection based on request_id hash
        h = hash(request_id + self._ab_test.version_a)
        if (h % 1000) / 1000.0 < self._ab_test.split_ratio:
            return self._ab_test.version_b
        return self._ab_test.version_a

    # ── Canary rollout ──────────────────────────────────────────────────

    def set_canary_stages(self, stages: list[CanaryStage]) -> None:
        """Set the canary rollout stages."""
        with self._lock:
            self._canary_stages = stages
            self._save_state()

    def canary_current_stage(self, traffic_pct: float) -> int | None:
        """Return the stage index for a given traffic percentage."""
        for i, stage in enumerate(self._canary_stages):
            if traffic_pct <= stage.traffic_pct:
                return i
        return None

    def should_rollback_canary(
        self, metrics_before: dict[str, float], metrics_after: dict[str, float]
    ) -> tuple[bool, str]:
        """Check if canary metrics degraded beyond threshold.

        Returns (should_rollback, reason).
        """
        for key in metrics_before:
            if key in metrics_after:
                delta = abs(metrics_after[key] - metrics_before[key]) / max(abs(metrics_before[key]), 1e-10)
                threshold = self._canary_stages[-1].rollback_threshold if self._canary_stages else 0.05
                if delta > threshold:
                    return True, f"Metric {key} degraded by {delta:.1%} (threshold {threshold:.1%})"
        return False, ""

    # ── Queries ─────────────────────────────────────────────────────────

    def active_version(self) -> ModelVersion | None:
        """Return the currently active model version."""
        v = self._versions.get(self._active_version)
        if v is None and self._versions:
            first = list(self._versions.values())[0]
            self._active_version = first.version_id
            return first
        return v

    def shadow_version(self) -> ModelVersion | None:
        """Return the current shadow version, if any."""
        for v in self._versions.values():
            if v.status == VersionStatus.SHADOW:
                return v
        return None

    def get_version(self, version_id: str) -> ModelVersion | None:
        return self._versions.get(version_id)

    def list_versions(self) -> list[ModelVersion]:
        return list(self._versions.values())

    def on_promote(self, hook: Callable[[str, str | None], None]) -> None:
        """Register a hook called after each promotion.

        ``hook(version_id, previous_version_id)``
        """
        self._promote_hooks.append(hook)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            data = {
                "model_name": self.model_name,
                "active_version": self._active_version,
                "rollback_history": self._rollback_history,
                "versions": {
                    vid: {
                        "version_id": v.version_id,
                        "model_name": v.model_name,
                        "model_path": v.model_path,
                        "dtype": v.dtype,
                        "status": v.status.value,
                        "deployed_at": v.deployed_at,
                        "promoted_at": v.promoted_at,
                        "sha256": v.sha256,
                        "config_overrides": v.config_overrides,
                        "metrics": v.metrics,
                        "tags": v.tags,
                    }
                    for vid, v in self._versions.items()
                },
            }
            self._state_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save version state: {e}")

    def load_state(self) -> None:
        """Restore version state from disk."""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
            self.model_name = data.get("model_name", self.model_name)
            self._active_version = data.get("active_version", "")
            self._rollback_history = data.get("rollback_history", [])
            for vid, vdata in data.get("versions", {}).items():
                self._versions[vid] = ModelVersion(
                    version_id=vdata["version_id"],
                    model_name=vdata["model_name"],
                    model_path=vdata["model_path"],
                    dtype=vdata.get("dtype", "float16"),
                    status=VersionStatus(vdata.get("status", "staged")),
                    deployed_at=vdata.get("deployed_at", 0.0),
                    promoted_at=vdata.get("promoted_at"),
                    sha256=vdata.get("sha256", ""),
                    config_overrides=vdata.get("config_overrides", {}),
                    metrics=vdata.get("metrics", {}),
                    tags=vdata.get("tags", []),
                )
        except Exception as e:
            logger.warning(f"Failed to load version state: {e}")
