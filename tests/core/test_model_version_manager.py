"""Tests for model_version_manager.py -- versioning, rollback, shadow/A/B testing.

Covers:
    ModelVersion        -- dataclass construction and defaults
    ABTestSplit         -- dataclass construction and defaults
    CanaryStage         -- dataclass construction and defaults
    ModelVersionManager -- registration, promotion, rollback, shadow, canary,
                           A/B testing, lineage, persistence, hooks

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight plain-Python stubs only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the module under test
_mvm_mod = load_module("distllm/core/model_version_manager.py")

# Re-export symbols for test readability
ModelVersion = _mvm_mod.ModelVersion
ABTestSplit = _mvm_mod.ABTestSplit
CanaryStage = _mvm_mod.CanaryStage
VersionStatus = _mvm_mod.VersionStatus
ModelVersionManager = _mvm_mod.ModelVersionManager


# ===================================================================
# Deterministic time helpers
# ===================================================================

class _FrozenTime:
    """Deterministic replacement for time.time() in tests.

    Usage::

        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
    """

    def __init__(self, now: float = 1000.0):
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


# ===================================================================
# MODEL VERSION DATACLASS TESTS
# ===================================================================

class TestModelVersion:
    """ModelVersion dataclass -- construction, defaults, fields."""

    def test_default_construction(self) -> None:
        v = ModelVersion(version_id="v1", model_name="llama", model_path="/models/v1")
        assert v.version_id == "v1"
        assert v.model_name == "llama"
        assert v.model_path == "/models/v1"
        assert v.dtype == "float16"
        assert v.status == VersionStatus.STAGED
        assert v.deployed_at == 0.0
        assert v.promoted_at is None
        assert v.sha256 == ""
        assert v.config_overrides == {}
        assert v.metrics == {}
        assert v.tags == []
        assert v.parent_version == ""
        assert v.training_round == 0
        assert v.training_node == ""

    def test_custom_values(self) -> None:
        v = ModelVersion(
            version_id="v2",
            model_name="llama",
            model_path="/models/v2",
            dtype="float32",
            status=VersionStatus.ACTIVE,
            deployed_at=100.0,
            promoted_at=200.0,
            sha256="abc123",
            config_overrides={"temperature": 0.8},
            metrics={"accuracy": 0.95},
            tags=["prod", "canary"],
            parent_version="v1",
            training_round=3,
            training_node="node-1",
        )
        assert v.version_id == "v2"
        assert v.dtype == "float32"
        assert v.status == VersionStatus.ACTIVE
        assert v.deployed_at == 100.0
        assert v.promoted_at == 200.0
        assert v.sha256 == "abc123"
        assert v.config_overrides == {"temperature": 0.8}
        assert v.metrics == {"accuracy": 0.95}
        assert v.tags == ["prod", "canary"]
        assert v.parent_version == "v1"
        assert v.training_round == 3
        assert v.training_node == "node-1"


class TestABTestSplit:
    """ABTestSplit dataclass -- defaults."""

    def test_default_construction(self) -> None:
        ab = ABTestSplit()
        assert ab.version_a == ""
        assert ab.version_b == ""
        assert ab.split_ratio == 0.5
        assert ab.metrics_collected == 0
        assert ab.started_at == 0.0

    def test_custom_values(self) -> None:
        ab = ABTestSplit(
            version_a="v1", version_b="v2",
            split_ratio=0.3, metrics_collected=42, started_at=100.0,
        )
        assert ab.version_a == "v1"
        assert ab.version_b == "v2"
        assert ab.split_ratio == 0.3
        assert ab.metrics_collected == 42
        assert ab.started_at == 100.0


class TestCanaryStage:
    """CanaryStage dataclass -- defaults."""

    def test_default_construction(self) -> None:
        cs = CanaryStage()
        assert cs.name == ""
        assert cs.traffic_pct == 0.0
        assert cs.min_duration_s == 120.0
        assert cs.rollback_threshold == 0.05

    def test_custom_values(self) -> None:
        cs = CanaryStage(
            name="10pct", traffic_pct=0.1,
            min_duration_s=60.0, rollback_threshold=0.03,
        )
        assert cs.name == "10pct"
        assert cs.traffic_pct == 0.1
        assert cs.min_duration_s == 60.0
        assert cs.rollback_threshold == 0.03


class TestVersionStatus:
    """VersionStatus enum -- values."""

    def test_all_values(self) -> None:
        assert VersionStatus.STAGED.value == "staged"
        assert VersionStatus.ACTIVE.value == "active"
        assert VersionStatus.SHADOW.value == "shadow"
        assert VersionStatus.ROLLED_BACK.value == "rolled_back"
        assert VersionStatus.DEPRECATED.value == "deprecated"


# ===================================================================
# MODEL VERSION MANAGER TESTS
# ===================================================================

class TestModelVersionManagerConstruction:
    """ModelVersionManager __init__ -- construction, defaults, state_path."""

    def test_default_construction(self) -> None:
        mgr = ModelVersionManager(model_name="llama-70b")
        assert mgr.model_name == "llama-70b"
        assert mgr.max_versions == 5
        assert mgr._state_path.name == ".model_versions.json"
        assert mgr._versions == {}
        assert mgr._active_version == ""
        assert mgr._ab_test is None
        assert mgr._canary_stages == []
        assert mgr._rollback_history == []
        assert mgr._promote_hooks == []

    def test_custom_max_versions(self) -> None:
        mgr = ModelVersionManager(model_name="llama", max_versions=2)
        assert mgr.max_versions == 2

    def test_custom_state_path(self) -> None:
        mgr = ModelVersionManager(
            model_name="llama", state_path="/tmp/custom_state.json",
        )
        assert mgr._state_path == Path("/tmp/custom_state.json")


class TestModelVersionManagerRegister:
    """ModelVersionManager.register_version -- registration logic."""

    def test_first_version_auto_promotes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.register_version("v1", "/models/v1")

        assert result is True
        v = mgr._versions["v1"]
        assert v.status == VersionStatus.ACTIVE
        assert v.promoted_at == 1000.0
        assert mgr._active_version == "v1"

    def test_second_version_stays_staged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        ft.advance(10.0)
        result = mgr.register_version("v2", "/models/v2")

        assert result is True
        assert mgr._versions["v2"].status == VersionStatus.STAGED
        assert mgr._versions["v2"].deployed_at == 1010.0
        assert mgr._active_version == "v1"  # unchanged

    def test_duplicate_version_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        result = mgr.register_version("v1", "/models/v1-dupe")
        assert result is False
        # Original still present
        assert mgr._versions["v1"].model_path == "/models/v1"

    def test_evicts_oldest_when_at_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama", max_versions=2)

        mgr.register_version("v1", "/models/v1")  # deployed_at=1000
        ft.advance(1.0)
        mgr.register_version("v2", "/models/v2")  # deployed_at=1001
        ft.advance(1.0)
        # At capacity (2), so v3 evicts oldest (v1)
        mgr.register_version("v3", "/models/v3")  # deployed_at=1002

        assert "v1" not in mgr._versions
        assert "v2" in mgr._versions
        assert "v3" in mgr._versions

    def test_register_with_tags_and_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version(
            "v1", "/models/v1",
            dtype="bfloat16",
            tags=["prod", "gpu"],
            config_overrides={"temperature": 0.5, "top_p": 0.9},
        )
        v = mgr._versions["v1"]
        assert v.dtype == "bfloat16"
        assert v.tags == ["prod", "gpu"]
        assert v.config_overrides == {"temperature": 0.5, "top_p": 0.9}


class TestModelVersionManagerUnregister:
    """ModelVersionManager.unregister_version -- removal logic."""

    def test_unregister_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        result = mgr.unregister_version("v2")
        assert result is True
        assert "v2" not in mgr._versions

    def test_cannot_unregister_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")  # auto-promotes

        result = mgr.unregister_version("v1")
        assert result is False  # active, can't remove
        assert "v1" in mgr._versions

    def test_unregister_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.unregister_version("nonexistent")
        assert result is False


class TestModelVersionManagerPromote:
    """ModelVersionManager.promote -- activation and deprecation."""

    def test_promote_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        ft.advance(10.0)
        mgr.register_version("v2", "/models/v2")

        result = mgr.promote("v2")
        assert result is True
        assert mgr._active_version == "v2"
        assert mgr._versions["v2"].status == VersionStatus.ACTIVE
        assert mgr._versions["v2"].promoted_at == 1010.0

        # v1 should be deprecated
        assert mgr._versions["v1"].status == VersionStatus.DEPRECATED
        assert mgr._rollback_history == ["v1"]

    def test_promote_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.promote("nonexistent")
        assert result is False

    def test_promote_triggers_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        calls: list[tuple[str, str | None]] = []
        mgr.on_promote(lambda vid, prev: calls.append((vid, prev)))

        mgr.promote("v2")
        assert len(calls) == 1
        assert calls[0] == ("v2", "v1")

    def test_promote_hook_failure_does_not_break(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        def failing_hook(vid: str, prev: str | None) -> None:
            raise RuntimeError("hook failure")

        mgr.on_promote(failing_hook)
        # Should not raise, should still succeed
        result = mgr.promote("v2")
        assert result is True
        assert mgr._active_version == "v2"

    def test_promote_same_version_deprecates_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Promoting the already-active version deprecates it (source quirk).

        The source code sets the version to ACTIVE, then overwrites its status
        to DEPRECATED because *previous* == *version_id* and the deprecation
        block runs unconditionally after the promotion block.  The active pointer
        still points to this version even though its status says deprecated.
        """
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        result = mgr.promote("v1")
        assert result is True
        assert mgr._rollback_history == ["v1"]
        assert mgr._active_version == "v1"
        # Quirk: the version ends up DEPRECATED because the
        # deprecation block runs after setting status = ACTIVE.
        assert mgr._versions["v1"].status == VersionStatus.DEPRECATED


class TestModelVersionManagerRollback:
    """ModelVersionManager.rollback -- revert to previous active."""

    def test_rollback_no_history(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.rollback()
        assert result is None

    def test_rollback_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")  # v1 -> history

        ft.advance(5.0)
        result = mgr.rollback()
        assert result == "v1"
        assert mgr._active_version == "v1"
        assert mgr._versions["v1"].status == VersionStatus.ACTIVE
        assert mgr._versions["v1"].promoted_at == 1005.0
        assert mgr._versions["v2"].status == VersionStatus.ROLLED_BACK

    def test_rollback_previous_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rollback returns None when previous version was unregistered."""
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")
        # Manually remove v1 from _versions to simulate external removal
        mgr._versions.pop("v1", None)

        result = mgr.rollback()
        assert result is None  # previous version no longer registered


class TestModelVersionManagerShadow:
    """ModelVersionManager shadow deployment."""

    def test_set_shadow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        result = mgr.set_shadow("v2")
        assert result is True
        assert mgr._versions["v2"].status == VersionStatus.SHADOW

    def test_set_shadow_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.set_shadow("nonexistent")
        assert result is False

    def test_clear_shadow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.set_shadow("v2")
        assert mgr._versions["v2"].status == VersionStatus.SHADOW

        mgr.clear_shadow()
        assert mgr._versions["v2"].status == VersionStatus.DEPRECATED

    def test_clear_shadow_does_not_affect_non_shadow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")  # auto-promoted to ACTIVE
        mgr.register_version("v2", "/models/v2")  # STAGED
        mgr.set_shadow("v2")
        # v1 should remain ACTIVE
        mgr.clear_shadow()
        assert mgr._versions["v1"].status == VersionStatus.ACTIVE


class TestModelVersionManagerABTest:
    """ModelVersionManager A/B test -- start, stop, select."""

    def test_start_ab_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        result = mgr.start_ab_test("v1", "v2", split_ratio=0.3)
        assert result is True
        assert mgr._ab_test is not None
        assert mgr._ab_test.version_a == "v1"
        assert mgr._ab_test.version_b == "v2"
        assert mgr._ab_test.split_ratio == 0.3
        assert mgr._ab_test.started_at == 1000.0

    def test_start_ab_test_default_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        mgr.start_ab_test("v1", "v2")
        assert mgr._ab_test is not None
        assert mgr._ab_test.split_ratio == 0.5

    def test_start_ab_test_version_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.start_ab_test("v1", "nonexistent")
        assert result is False

    def test_stop_ab_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.start_ab_test("v1", "v2", split_ratio=0.3)

        result = mgr.stop_ab_test()
        assert result is not None
        assert result.version_a == "v1"
        assert result.version_b == "v2"
        assert mgr._ab_test is None  # cleared

    def test_stop_ab_test_when_none(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.stop_ab_test()
        assert result is None

    def test_select_ab_version_no_test_returns_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")  # auto-promoted

        selected = mgr.select_ab_version("req-1")
        assert selected == "v1"

    def test_select_ab_version_no_test_no_active(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        selected = mgr.select_ab_version("req-1")
        assert selected is None

    def test_select_ab_version_split_zero_always_a(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.start_ab_test("v1", "v2", split_ratio=0.0)

        # With split_ratio=0, always returns version_a
        selected = mgr.select_ab_version("req-1")
        assert selected == "v1"

    def test_select_ab_version_split_one_always_b(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.start_ab_test("v1", "v2", split_ratio=1.0)

        # With split_ratio=1, the condition ``(h % 1000) / 1000.0 < 1.0``
        # is always True (max value is 0.999), so always returns version_b.
        selected = mgr.select_ab_version("req-1")
        assert selected == "v2"

    def test_select_ab_version_returns_valid_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.start_ab_test("v1", "v2", split_ratio=0.5)

        selected = mgr.select_ab_version("req-1")
        assert selected in ("v1", "v2")


class TestModelVersionManagerCanary:
    """ModelVersionManager canary rollout -- stages, threshold checks."""

    def test_set_canary_stages(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="5pct", traffic_pct=0.05, min_duration_s=60.0),
            CanaryStage(name="25pct", traffic_pct=0.25, min_duration_s=120.0),
            CanaryStage(name="50pct", traffic_pct=0.50, min_duration_s=180.0),
            CanaryStage(name="100pct", traffic_pct=1.0, min_duration_s=300.0),
        ]
        mgr.set_canary_stages(stages)
        assert len(mgr._canary_stages) == 4
        assert mgr._canary_stages[0].name == "5pct"
        assert mgr._canary_stages[3].traffic_pct == 1.0

    def test_canary_current_stage_first(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="5pct", traffic_pct=0.05),
            CanaryStage(name="25pct", traffic_pct=0.25),
            CanaryStage(name="100pct", traffic_pct=1.0),
        ]
        mgr.set_canary_stages(stages)

        idx = mgr.canary_current_stage(0.03)
        assert idx == 0  # 0.03 <= 0.05

    def test_canary_current_stage_middle(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="5pct", traffic_pct=0.05),
            CanaryStage(name="25pct", traffic_pct=0.25),
            CanaryStage(name="100pct", traffic_pct=1.0),
        ]
        mgr.set_canary_stages(stages)

        idx = mgr.canary_current_stage(0.15)
        assert idx == 1  # 0.15 <= 0.25

    def test_canary_current_stage_exact_match(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="5pct", traffic_pct=0.05),
            CanaryStage(name="25pct", traffic_pct=0.25),
        ]
        mgr.set_canary_stages(stages)

        idx = mgr.canary_current_stage(0.05)
        assert idx == 0  # <= 0.05

    def test_canary_current_stage_beyond_max(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="5pct", traffic_pct=0.05),
            CanaryStage(name="25pct", traffic_pct=0.25),
        ]
        mgr.set_canary_stages(stages)

        idx = mgr.canary_current_stage(0.50)
        assert idx is None  # 0.50 > 0.25

    def test_canary_current_stage_no_stages(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        idx = mgr.canary_current_stage(0.1)
        assert idx is None

    def test_should_rollback_canary_threshold_exceeded(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="100pct", traffic_pct=1.0, rollback_threshold=0.05),
        ]
        mgr.set_canary_stages(stages)

        before = {"accuracy": 0.95, "latency_ms": 100.0}
        after = {"accuracy": 0.85, "latency_ms": 100.0}
        # accuracy degraded from 0.95 to 0.85 => delta = 0.10/0.95 = 0.105 > 0.05
        should, reason = mgr.should_rollback_canary(before, after)
        assert should is True
        assert "accuracy" in reason
        assert "degraded" in reason

    def test_should_rollback_canary_ok(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="100pct", traffic_pct=1.0, rollback_threshold=0.05),
        ]
        mgr.set_canary_stages(stages)

        before = {"accuracy": 0.95, "latency_ms": 100.0}
        after = {"accuracy": 0.94, "latency_ms": 102.0}
        # Both within 5% threshold
        should, reason = mgr.should_rollback_canary(before, after)
        assert should is False
        assert reason == ""

    def test_should_rollback_canary_no_stages_default_threshold(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        # No stages set => uses default threshold of 0.05
        before = {"accuracy": 1.0}
        after = {"accuracy": 0.8}
        # delta = 0.2/1.0 = 0.2 > 0.05
        should, reason = mgr.should_rollback_canary(before, after)
        assert should is True

    def test_should_rollback_canary_missing_key_skipped(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="100pct", traffic_pct=1.0, rollback_threshold=0.05),
        ]
        mgr.set_canary_stages(stages)

        before = {"accuracy": 0.95}
        after = {"latency_ms": 200.0}  # different key, no overlap
        should, reason = mgr.should_rollback_canary(before, after)
        assert should is False

    def test_should_rollback_canary_zero_before_value(self) -> None:
        """Division by near-zero in before value should not crash."""
        mgr = ModelVersionManager(model_name="llama")
        stages = [
            CanaryStage(name="100pct", traffic_pct=1.0, rollback_threshold=0.05),
        ]
        mgr.set_canary_stages(stages)

        before = {"accuracy": 0.0}
        after = {"accuracy": 0.1}
        # delta = 0.1 / max(0.0, 1e-10) = 1e9 >> threshold
        should, reason = mgr.should_rollback_canary(before, after)
        assert should is True


class TestModelVersionManagerQueries:
    """ModelVersionManager query methods."""

    def test_active_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")

        v = mgr.active_version()
        assert v is not None
        assert v.version_id == "v2"

    def test_active_version_fallback(self) -> None:
        """When _active_version is empty but versions exist, return first."""
        mgr = ModelVersionManager(model_name="llama")
        # Manually insert a version without setting _active_version
        mgr._versions["v1"] = ModelVersion(
            version_id="v1", model_name="llama", model_path="/models/v1",
        )
        v = mgr.active_version()
        assert v is not None
        assert v.version_id == "v1"
        assert mgr._active_version == "v1"  # side-effect

    def test_active_version_empty(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        v = mgr.active_version()
        assert v is None

    def test_shadow_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.set_shadow("v2")

        v = mgr.shadow_version()
        assert v is not None
        assert v.version_id == "v2"

    def test_shadow_version_none(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        v = mgr.shadow_version()
        assert v is None

    def test_get_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")

        v = mgr.get_version("v1")
        assert v is not None
        assert v.version_id == "v1"

    def test_get_version_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        v = mgr.get_version("nonexistent")
        assert v is None

    def test_list_versions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        versions = mgr.list_versions()
        assert len(versions) == 2
        ids = {v.version_id for v in versions}
        assert ids == {"v1", "v2"}

    def test_list_versions_empty(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        versions = mgr.list_versions()
        assert versions == []


class TestModelVersionManagerHooks:
    """ModelVersionManager.on_promote -- hook registration."""

    def test_on_promote_registers_hook(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        assert len(mgr._promote_hooks) == 0

        def my_hook(vid: str, prev: str | None) -> None:
            pass

        mgr.on_promote(my_hook)
        assert len(mgr._promote_hooks) == 1
        assert mgr._promote_hooks[0] is my_hook

    def test_on_promote_multiple_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")

        calls: list[str] = []

        def hook1(vid: str, prev: str | None) -> None:
            calls.append(f"h1:{vid}")

        def hook2(vid: str, prev: str | None) -> None:
            calls.append(f"h2:{vid}")

        mgr.on_promote(hook1)
        mgr.on_promote(hook2)
        mgr.promote("v2")

        assert calls == ["h1:v2", "h2:v2"]


class TestModelVersionManagerLineage:
    """ModelVersionManager lineage tracking (parent_version chain)."""

    def _make_mgr_with_chain(self, monkeypatch: pytest.MonkeyPatch) -> ModelVersionManager:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")

        # Create a chain: v0 -> v1 -> v2 -> v3
        for i in range(4):
            vid = f"v{i}"
            mgr.register_version(vid, f"/models/{vid}")
            if i > 0:
                mgr._versions[vid].parent_version = f"v{i - 1}"
                mgr._versions[vid].training_round = i
                mgr._versions[vid].training_node = f"node-{i}"
        return mgr

    def test_get_lineage_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._make_mgr_with_chain(monkeypatch)
        lineage = mgr.get_lineage("v3")

        # Returns from v3 back to v0
        assert len(lineage) == 4
        assert lineage[0]["version_id"] == "v3"
        assert lineage[0]["parent_version"] == "v2"
        assert lineage[1]["version_id"] == "v2"
        assert lineage[1]["training_round"] == 2
        assert lineage[2]["version_id"] == "v1"
        assert lineage[3]["version_id"] == "v0"
        assert lineage[3]["parent_version"] == ""

    def test_get_lineage_single_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")

        lineage = mgr.get_lineage("v1")
        assert len(lineage) == 1
        assert lineage[0]["version_id"] == "v1"

    def test_get_lineage_not_found(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        lineage = mgr.get_lineage("nonexistent")
        assert lineage == []

    def test_get_lineage_breaks_on_missing_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr._versions["v2"].parent_version = "v_missing"

        lineage = mgr.get_lineage("v2")
        # Should return just v2 since v_missing is not in _versions
        assert len(lineage) == 1
        assert lineage[0]["version_id"] == "v2"

    def test_get_lineage_circular_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        # Create a cycle: v1 -> v2 -> v1
        mgr._versions["v1"].parent_version = "v2"
        mgr._versions["v2"].parent_version = "v1"

        lineage = mgr.get_lineage("v1")
        # Should return [v1, v2] and stop before revisiting v1
        assert len(lineage) == 2
        assert lineage[0]["version_id"] == "v1"
        assert lineage[1]["version_id"] == "v2"

    def test_get_version_tree_adjacency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._make_mgr_with_chain(monkeypatch)
        tree = mgr.get_version_tree()

        # v0 has no parent -> under __root__
        assert "__root__" in tree
        assert "v0" in tree["__root__"]
        # v0 -> v1 -> v2 -> v3
        assert tree["v0"] == ["v1"]
        assert tree["v1"] == ["v2"]
        assert tree["v2"] == ["v3"]

    def test_get_version_tree_with_branching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("root", "/models/root")
        mgr.register_version("a", "/models/a")
        mgr.register_version("b", "/models/b")
        mgr._versions["a"].parent_version = "root"
        mgr._versions["b"].parent_version = "root"

        tree = mgr.get_version_tree()
        assert set(tree["root"]) == {"a", "b"}

    def test_get_version_tree_empty(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        tree = mgr.get_version_tree()
        assert tree == {}


class TestModelVersionManagerPersistence:
    """ModelVersionManager _save_state / load_state -- file round-trip."""

    def test_save_and_load_state(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        state_file = tmp_path / "versions.json"

        mgr = ModelVersionManager(
            model_name="llama", state_path=str(state_file),
        )
        mgr.register_version("v1", "/models/v1", tags=["prod"])
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")
        mgr.set_shadow("v1")

        # Create a second manager and load state
        mgr2 = ModelVersionManager(
            model_name="other", state_path=str(state_file),
        )
        mgr2.load_state()

        assert mgr2.model_name == "llama"  # restored from file
        assert mgr2._active_version == "v2"
        assert len(mgr2._versions) == 2
        assert mgr2._versions["v1"].status == VersionStatus.SHADOW
        assert mgr2._versions["v2"].status == VersionStatus.ACTIVE
        assert mgr2._versions["v1"].tags == ["prod"]
        assert mgr2._versions["v2"].promoted_at == 1000.0

    def test_load_state_no_file(self) -> None:
        mgr = ModelVersionManager(
            model_name="llama", state_path="/nonexistent/path.json",
        )
        # Should not raise
        mgr.load_state()
        assert mgr._versions == {}

    def test_load_state_corrupt_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("This is not valid JSON")

        mgr = ModelVersionManager(
            model_name="llama", state_path=str(state_file),
        )
        # Should not raise, just log a warning
        mgr.load_state()
        assert mgr._versions == {}

    def test_save_and_load_empty_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "empty.json"
        mgr = ModelVersionManager(
            model_name="llama", state_path=str(state_file),
        )
        # Save empty state
        mgr._save_state()

        mgr2 = ModelVersionManager(
            model_name="other", state_path=str(state_file),
        )
        mgr2.load_state()
        assert mgr2.model_name == "llama"
        assert mgr2._versions == {}

    def test_load_state_restores_rollback_history(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        state_file = tmp_path / "with_history.json"

        mgr = ModelVersionManager(
            model_name="llama", state_path=str(state_file),
        )
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")
        mgr.register_version("v3", "/models/v3")
        mgr.promote("v3")
        # history: ["v1", "v2"]

        mgr2 = ModelVersionManager(
            model_name="other", state_path=str(state_file),
        )
        mgr2.load_state()
        assert mgr2._rollback_history == ["v1", "v2"]


class TestModelVersionManagerEdgeCases:
    """ModelVersionManager edge cases and error paths."""

    def test_register_version_evicts_correctly_with_ties(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When versions have identical deployed_at, min() still picks one."""
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama", max_versions=2)
        # All deployed at the same time (1000.0)
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        # v3 will evict one (tie broken by min() on key string)
        mgr.register_version("v3", "/models/v3")

        assert len(mgr._versions) == 2
        assert "v3" in mgr._versions

    def test_rollback_empty_history_on_empty_manager(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.rollback()
        assert result is None

    def test_promote_then_promote_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.register_version("v2", "/models/v2")
        mgr.promote("v2")
        mgr.promote("v1")  # promote back to v1

        assert mgr._active_version == "v1"
        assert mgr._versions["v1"].status == VersionStatus.ACTIVE
        assert mgr._versions["v2"].status == VersionStatus.DEPRECATED
        # history: ["v1", "v2"]
        assert "v1" in mgr._rollback_history

    def test_shadow_no_shadow_versions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        mgr.clear_shadow()  # should not raise nor change anything
        assert mgr._versions["v1"].status == VersionStatus.ACTIVE

    def test_unregister_non_existent(self) -> None:
        mgr = ModelVersionManager(model_name="llama")
        result = mgr.unregister_version("ghost")
        assert result is False

    def test_promote_nonexistent_in_non_empty_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")
        result = mgr.promote("ghost")
        assert result is False
        assert mgr._active_version == "v1"  # unchanged

    def test_list_versions_are_copies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_versions returns a new list each time."""
        ft = _FrozenTime(1000.0)
        monkeypatch.setattr(time, "time", ft)
        mgr = ModelVersionManager(model_name="llama")
        mgr.register_version("v1", "/models/v1")

        list1 = mgr.list_versions()
        list2 = mgr.list_versions()
        assert list1 is not list2  # different lists
        assert list1 == list2
