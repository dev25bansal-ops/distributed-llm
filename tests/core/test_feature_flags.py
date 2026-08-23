"""Tests for FeatureFlags system.

Covers: file loading, is_enabled resolution, env overrides,
rollout percentages, user/group targeting, time windows,
runtime updates, save/reload, empty config.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/feature_flags.py")
FeatureFlags = _mod.FeatureFlags
FlagConfig = _mod.FlagConfig


@pytest.fixture
def flags_path():
    p = Path(tempfile.gettempdir()) / f"test_flags_{time.time_ns()}.json"
    yield p
    for _ in range(3):
        try:
            if p.exists():
                p.unlink()
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def populated_flags(flags_path):
    data = {
        "feature_a": {"enabled": True, "description": "Feature A"},
        "feature_b": {"enabled": False, "description": "Feature B"},
        "rollout_feature": {"enabled": True, "rollout_pct": 50.0},
        "user_feature": {
            "enabled": True,
            "allowed_users": ["user-1", "user-2"],
        },
        "group_feature": {
            "enabled": True,
            "allowed_groups": ["beta-testers"],
        },
    }
    flags_path.write_text(json.dumps(data))
    return flags_path


class TestFlagConfigDefaults:
    def test_default_values(self):
        fc = FlagConfig(name="test")
        assert fc.name == "test"
        assert fc.enabled is False
        assert fc.description == ""
        assert fc.rollout_pct == 100.0
        assert fc.allowed_users == []
        assert fc.allowed_groups == []
        assert fc.enabled_from == 0.0
        assert fc.enabled_until == 0.0


class TestFeatureFlagsInit:
    def test_no_path_creates_empty(self):
        flags = FeatureFlags()
        assert flags.get_all_flags() == {}

    def test_loads_from_file(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        all_flags = flags.get_all_flags()
        assert "feature_a" in all_flags
        assert all_flags["feature_a"]["enabled"] is True

    def test_nonexistent_path_does_not_raise(self):
        flags = FeatureFlags(config_path="/nonexistent/flags.json")
        assert flags.get_all_flags() == {}

    def test_simple_bool_in_file(self, flags_path):
        flags_path.write_text(json.dumps({"simple_flag": True}))
        flags = FeatureFlags(config_path=str(flags_path))
        assert flags.is_enabled("simple_flag") is True


class TestIsEnabled:
    def test_enabled_flag_returns_true(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("feature_a") is True

    def test_disabled_flag_returns_false(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("feature_b") is False

    def test_unknown_flag_returns_default(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("nonexistent") is False
        assert flags.is_enabled("nonexistent", default=True) is True

    def test_rollout_50pct_deterministic(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        results = {}
        for uid in [f"user-{i}" for i in range(200)]:
            results[uid] = flags.is_enabled("rollout_feature", user_id=uid)
        # With 50% rollout over 200 users, roughly 80-120 should be enabled
        enabled_count = sum(1 for v in results.values() if v)
        assert 60 <= enabled_count <= 140

    def test_user_targeting(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("user_feature", user_id="user-1") is True
        assert flags.is_enabled("user_feature", user_id="user-3") is False

    def test_group_targeting(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("group_feature", group="beta-testers") is True
        # Non-matching group falls through to rollout (default 100%) so
        # it returns True because the flag is enabled.
        assert flags.is_enabled("group_feature", group="nonexistent") is True


class TestEnvOverride:
    def test_env_override_true(self, populated_flags, monkeypatch):
        monkeypatch.setenv("DISTLLM_FLAG_FEATURE_B", "1")
        flags = FeatureFlags(config_path=str(populated_flags))
        # feature_b is disabled in file but env says 1
        assert flags.is_enabled("feature_b") is True

    def test_env_override_false(self, populated_flags, monkeypatch):
        monkeypatch.setenv("DISTLLM_FLAG_FEATURE_A", "0")
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("feature_a") is False

    def test_env_override_not_present(self, populated_flags):
        flags = FeatureFlags(config_path=str(populated_flags))
        assert flags.is_enabled("feature_a") is True


class TestRuntimeUpdate:
    def test_set_flag_enables(self, flags_path):
        flags_path.write_text(json.dumps({"test_flag": {"enabled": False}}))
        flags = FeatureFlags(config_path=str(flags_path))
        assert flags.is_enabled("test_flag") is False
        flags.set_flag("test_flag", True)
        assert flags.is_enabled("test_flag") is True

    def test_set_flag_creates_new(self, flags_path):
        flags_path.write_text(json.dumps({}))
        flags = FeatureFlags(config_path=str(flags_path))
        flags.set_flag("new_flag", True)
        assert flags.is_enabled("new_flag") is True

    def test_save_and_reload(self, flags_path):
        flags_path.write_text(json.dumps({"save_test": {"enabled": False}}))
        flags = FeatureFlags(config_path=str(flags_path))
        flags.set_flag("save_test", True)
        flags.save()
        # Reload from file
        flags2 = FeatureFlags(config_path=str(flags_path))
        assert flags2.is_enabled("save_test") is True


class TestTimeWindow:
    def test_enabled_from_future_returns_false(self, flags_path):
        data = {
            "future_flag": {
                "enabled": True,
                "enabled_from": time.time() + 3600,
            }
        }
        flags_path.write_text(json.dumps(data))
        flags = FeatureFlags(config_path=str(flags_path))
        assert flags.is_enabled("future_flag") is False

    def test_enabled_until_past_returns_false(self, flags_path):
        data = {
            "expired_flag": {
                "enabled": True,
                "enabled_until": time.time() - 3600,
            }
        }
        flags_path.write_text(json.dumps(data))
        flags = FeatureFlags(config_path=str(flags_path))
        assert flags.is_enabled("expired_flag") is False

    def test_enabled_until_future_returns_true(self, flags_path):
        data = {
            "valid_flag": {
                "enabled": True,
                "enabled_until": time.time() + 3600,
            }
        }
        flags_path.write_text(json.dumps(data))
        flags = FeatureFlags(config_path=str(flags_path))
        assert flags.is_enabled("valid_flag") is True


class TestAutoReload:
    def test_auto_reload_detects_file_changes(self, flags_path):
        data = {"reload_flag": {"enabled": False}}
        flags_path.write_text(json.dumps(data))
        flags = FeatureFlags(
            config_path=str(flags_path),
            auto_reload=True,
            reload_interval_s=0.0,
        )
        assert flags.is_enabled("reload_flag") is False
        # Update file
        data["reload_flag"]["enabled"] = True
        flags_path.write_text(json.dumps(data))
        # Force reload by sleeping past interval
        time.sleep(0.01)
        assert flags.is_enabled("reload_flag") is True
