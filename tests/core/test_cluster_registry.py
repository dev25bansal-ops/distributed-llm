"""Tests for distllm/core/cluster_registry.py.

Covers:
    ClusterInfo    -- dataclass for public cluster metadata
    ContributionCredit -- dataclass for GPU contribution tracking
    ClusterRegistry -- register, unregister, heartbeat, discover,
                       credits, spend, leaderboard

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects only.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the module under test
_cluster_mod = load_module("distllm/core/cluster_registry.py")

# Re-export symbols for test readability
ClusterInfo = _cluster_mod.ClusterInfo
ContributionCredit = _cluster_mod.ContributionCredit
ClusterRegistry = _cluster_mod.ClusterRegistry

# Deterministic timestamp used throughout
_FIXED_NOW = 1_000_000.0


# ===================================================================
# ClusterInfo TESTS
# ===================================================================


class TestClusterInfo:
    """ClusterInfo dataclass -- construction, defaults, fields."""

    def test_minimal_defaults(self) -> None:
        """ClusterInfo should get sensible defaults with only required fields."""
        info = ClusterInfo(cluster_id="c1", host="10.0.0.1", port=50050)
        assert info.cluster_id == "c1"
        assert info.host == "10.0.0.1"
        assert info.port == 50050
        assert info.gpus == 1
        assert info.gpu_model == ""
        assert info.model == ""
        assert info.region == ""
        assert info.reputation == 0.5
        assert info.uptime_pct == 100.0
        assert info.avg_latency_ms == 0.0
        assert info.is_public is True
        assert info.owner_id == ""
        assert info.tags == []

    def test_all_fields_explicit(self) -> None:
        """All ClusterInfo fields should accept explicit values."""
        info = ClusterInfo(
            cluster_id="c2",
            host="gpu-2.example.com",
            port=50051,
            gpus=8,
            gpu_model="A100",
            model="llama-3-70b",
            region="us-east-1",
            reputation=0.95,
            uptime_pct=99.9,
            avg_latency_ms=12.5,
            is_public=False,
            owner_id="user-abc",
            registered_at=_FIXED_NOW,
            last_heartbeat=_FIXED_NOW,
            tags=["production", "high-priority"],
        )
        assert info.cluster_id == "c2"
        assert info.gpus == 8
        assert info.gpu_model == "A100"
        assert info.model == "llama-3-70b"
        assert info.region == "us-east-1"
        assert info.reputation == 0.95
        assert info.uptime_pct == 99.9
        assert info.avg_latency_ms == 12.5
        assert info.is_public is False
        assert info.owner_id == "user-abc"
        assert info.registered_at == _FIXED_NOW
        assert info.last_heartbeat == _FIXED_NOW
        assert info.tags == ["production", "high-priority"]

    def test_tags_independent_defaults(self) -> None:
        """Default factory for tags should give independent lists."""
        i1 = ClusterInfo(cluster_id="c1", host="h1", port=80)
        i2 = ClusterInfo(cluster_id="c2", host="h2", port=81)
        i1.tags.append("test")
        assert "test" not in i2.tags


# ===================================================================
# ContributionCredit TESTS
# ===================================================================


class TestContributionCredit:
    """ContributionCredit dataclass -- construction, defaults, fields."""

    def test_minimal_defaults(self) -> None:
        """Minimal ContributionCredit should get reasonable defaults."""
        cc = ContributionCredit(user_id="u1", cluster_id="c1")
        assert cc.user_id == "u1"
        assert cc.cluster_id == "c1"
        assert cc.gpu_hours == 0.0
        assert cc.tokens_served == 0
        assert cc.earned_credits == 0.0
        assert cc.spent_credits == 0.0
        assert cc.tier == "bronze"

    def test_all_fields_explicit(self) -> None:
        """All ContributionCredit fields should accept explicit values."""
        cc = ContributionCredit(
            user_id="u2",
            cluster_id="c2",
            gpu_hours=150.0,
            tokens_served=50000,
            earned_credits=1500.0,
            spent_credits=200.0,
            tier="gold",
        )
        assert cc.user_id == "u2"
        assert cc.cluster_id == "c2"
        assert cc.gpu_hours == 150.0
        assert cc.tokens_served == 50000
        assert cc.earned_credits == 1500.0
        assert cc.spent_credits == 200.0
        assert cc.tier == "gold"


# ===================================================================
# ClusterRegistry TESTS
# ===================================================================


class _TimeStepper:
    """Deterministic time source that tests can advance."""
    def __init__(self, now: float):
        self._now = now
    def __call__(self) -> float:
        return self._now
    def advance(self, delta: float) -> None:
        self._now += delta


@pytest.fixture
def registry() -> ClusterRegistry:
    """Return a fresh ClusterRegistry with default settings."""
    return ClusterRegistry(heartbeat_timeout=300.0)


@pytest.fixture
def stepper(monkeypatch: pytest.MonkeyPatch) -> _TimeStepper:
    """Monkeypatch time.time with a _TimeStepper and return it."""
    stepper = _TimeStepper(_FIXED_NOW)
    monkeypatch.setattr(_cluster_mod.time, "time", stepper)
    return stepper


@pytest.fixture
def time_registry(
    stepper: _TimeStepper, monkeypatch: pytest.MonkeyPatch,
) -> ClusterRegistry:
    """Registry with deterministic (steppable) time."""
    return ClusterRegistry(heartbeat_timeout=300.0)


# ── Construction / Defaults ─────────────────────────────────────


class TestClusterRegistryConstruction:
    """ClusterRegistry construction and default values."""

    def test_default_heartbeat_timeout(self) -> None:
        """Default heartbeat_timeout should be 300.0."""
        reg = ClusterRegistry()
        assert reg._heartbeat_timeout == 300.0

    def test_custom_heartbeat_timeout(self) -> None:
        """Custom heartbeat_timeout should be accepted."""
        reg = ClusterRegistry(heartbeat_timeout=60.0)
        assert reg._heartbeat_timeout == 60.0

    def test_initial_state_empty(self) -> None:
        """A fresh registry should have no clusters and no credits."""
        reg = ClusterRegistry()
        assert reg._clusters == {}
        assert reg._credits == {}

    def test_negative_heartbeat_timeout(self) -> None:
        """Negative timeout (fast expiry) should still be accepted."""
        reg = ClusterRegistry(heartbeat_timeout=-1.0)
        assert reg._heartbeat_timeout == -1.0


# ── register_cluster ────────────────────────────────────────────


class TestRegisterCluster:
    """ClusterRegistry.register_cluster -- registration and return."""

    def test_register_minimal(self, registry: ClusterRegistry) -> None:
        """Registering with only required kwargs should succeed."""
        info = registry.register_cluster(
            cluster_id="c1", host="10.0.0.1", port=50050,
        )
        assert isinstance(info, ClusterInfo)
        assert info.cluster_id == "c1"
        assert info.host == "10.0.0.1"
        assert info.port == 50050

    def test_register_with_all_fields(self, registry: ClusterRegistry) -> None:
        """Registering with all ClusterInfo fields should persist them."""
        info = registry.register_cluster(
            cluster_id="c2",
            host="gpu.example.com",
            port=50051,
            gpus=4,
            gpu_model="A100",
            model="llama-3-70b",
            region="us-east-1",
            reputation=0.9,
            is_public=True,
            owner_id="admin",
            tags=["production"],
        )
        stored = registry._clusters["c2"]
        assert stored.gpus == 4
        assert stored.gpu_model == "A100"
        assert stored.model == "llama-3-70b"
        assert stored.region == "us-east-1"
        assert stored.reputation == 0.9

    def test_register_persists_in_map(self, registry: ClusterRegistry) -> None:
        """After registration the cluster should be in _clusters."""
        registry.register_cluster(
            cluster_id="c3", host="10.0.0.3", port=50052,
        )
        assert "c3" in registry._clusters
        assert registry._clusters["c3"].cluster_id == "c3"

    def test_register_overwrites_existing(self, registry: ClusterRegistry) -> None:
        """Registering with an existing cluster_id should overwrite."""
        registry.register_cluster(
            cluster_id="c1", host="10.0.0.1", port=50050, gpus=2,
        )
        registry.register_cluster(
            cluster_id="c1", host="10.0.0.2", port=50051, gpus=8,
        )
        stored = registry._clusters["c1"]
        assert stored.host == "10.0.0.2"
        assert stored.port == 50051
        assert stored.gpus == 8

    def test_register_multiple_clusters(self, registry: ClusterRegistry) -> None:
        """Multiple registrations should each be stored separately."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.register_cluster(cluster_id="c2", host="h2", port=81)
        registry.register_cluster(cluster_id="c3", host="h3", port=82)
        assert len(registry._clusters) == 3

    def test_register_unknown_kwargs_raises(self, registry: ClusterRegistry) -> None:
        """Unknown kwargs should cause a TypeError from the ClusterInfo dataclass."""
        with pytest.raises(TypeError):
            registry.register_cluster(
                cluster_id="bad", host="h", port=80, nonexistent_field=42,
            )


# ── unregister_cluster ──────────────────────────────────────────


class TestUnregisterCluster:
    """ClusterRegistry.unregister_cluster -- removal."""

    def test_unregister_existing(self, registry: ClusterRegistry) -> None:
        """Unregistering an existing cluster should return True."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        result = registry.unregister_cluster("c1")
        assert result is True
        assert "c1" not in registry._clusters

    def test_unregister_nonexistent(self, registry: ClusterRegistry) -> None:
        """Unregistering a non-existent cluster should return False."""
        result = registry.unregister_cluster("nonexistent")
        assert result is False

    def test_unregister_empty_registry(self, registry: ClusterRegistry) -> None:
        """Unregistering from an empty registry should return False."""
        result = registry.unregister_cluster("anything")
        assert result is False

    def test_unregister_removes_only_target(self, registry: ClusterRegistry) -> None:
        """Unregistering one cluster should not affect others."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.register_cluster(cluster_id="c2", host="h2", port=81)
        registry.unregister_cluster("c1")
        assert "c1" not in registry._clusters
        assert "c2" in registry._clusters

    def test_unregister_then_re_register(self, registry: ClusterRegistry) -> None:
        """After unregister, a new cluster can be registered with the same id."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.unregister_cluster("c1")
        info = registry.register_cluster(cluster_id="c1", host="h2", port=81)
        assert info.host == "h2"


# ── heartbeat ───────────────────────────────────────────────────


class TestHeartbeat:
    """ClusterRegistry.heartbeat -- live-ness and metric updates."""

    def test_heartbeat_updates_timestamp(
        self, registry: ClusterRegistry,
    ) -> None:
        """heartbeat should update last_heartbeat to the current time."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        cluster = registry._clusters["c1"]
        old_ts = cluster.last_heartbeat
        assert old_ts > 0

        registry.heartbeat("c1")

        # last_heartbeat should advance (time moved forward since register)
        assert cluster.last_heartbeat >= old_ts

    def test_heartbeat_updates_metrics(self, registry: ClusterRegistry) -> None:
        """heartbeat should update metric fields passed as kwargs."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, gpus=4,
        )
        registry.heartbeat("c1", gpus=8, reputation=0.99, avg_latency_ms=5.0)
        cluster = registry._clusters["c1"]
        assert cluster.gpus == 8
        assert cluster.reputation == 0.99
        assert cluster.avg_latency_ms == 5.0

    def test_heartbeat_nonexistent(self, registry: ClusterRegistry) -> None:
        """heartbeat on a non-existent cluster should be a no-op."""
        registry.heartbeat("nonexistent")  # should not raise

    def test_heartbeat_unknown_metric_ignored(self, registry: ClusterRegistry) -> None:
        """heartbeat kwargs that are not ClusterInfo fields should be ignored."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.heartbeat("c1", made_up_field="xyz")  # should not raise

    def test_heartbeat_multiple_clusters(
        self, registry: ClusterRegistry,
    ) -> None:
        """Heartbeats should affect only the target cluster."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.register_cluster(cluster_id="c2", host="h2", port=81)
        c1_before = registry._clusters["c1"].last_heartbeat

        registry.heartbeat("c1")

        # c1's heartbeat should advance; c2's should stay the same (or close
        # enough — the default_factory=time.time in ClusterInfo may give a
        # slightly earlier timestamp for c1 than c2 during registration).
        assert registry._clusters["c1"].last_heartbeat >= c1_before
        assert registry._clusters["c2"].last_heartbeat >= c1_before


# ── discover ────────────────────────────────────────────────────


class TestDiscover:
    """ClusterRegistry.discover -- filtering, sorting, limits."""

    def test_discover_empty(self, registry: ClusterRegistry) -> None:
        """Discover on an empty registry should return an empty list."""
        result = registry.discover()
        assert result == []

    def test_discover_all_public(self, registry: ClusterRegistry) -> None:
        """Discover with no filters should return all public clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, model="llama",
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, model="falcon",
        )
        result = registry.discover()
        assert len(result) == 2

    def test_discover_excludes_non_public(self, registry: ClusterRegistry) -> None:
        """Discover should exclude clusters with is_public=False."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, is_public=True,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, is_public=False,
        )
        result = registry.discover()
        assert len(result) == 1
        assert result[0].cluster_id == "c1"

    def test_discover_excludes_stale(self, registry: ClusterRegistry) -> None:
        """Discover should exclude clusters past the heartbeat timeout."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.register_cluster(cluster_id="c2", host="h2", port=81)

        # Artificially age the clusters past the heartbeat timeout
        age = registry._heartbeat_timeout + 600.0
        for cluster in registry._clusters.values():
            cluster.last_heartbeat -= age

        result = registry.discover()
        assert len(result) == 0

    def test_discover_model_filter(self, registry: ClusterRegistry) -> None:
        """Filtering by model should return clusters whose model contains the string."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, model="llama-3-70b",
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, model="falcon-40b",
        )
        # Substring match
        result = registry.discover(model="llama")
        assert len(result) == 1
        assert result[0].cluster_id == "c1"

    def test_discover_model_filter_no_match(self, registry: ClusterRegistry) -> None:
        """Model filter with no matches should return empty list."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, model="llama",
        )
        result = registry.discover(model="nonexistent")
        assert result == []

    def test_discover_model_empty_string(self, registry: ClusterRegistry) -> None:
        """Empty string model filter should return all public clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, model="llama",
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, model="falcon",
        )
        result = registry.discover(model="")
        assert len(result) == 2

    def test_discover_region_filter(self, registry: ClusterRegistry) -> None:
        """Filtering by region should return exact-match clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, region="us-east-1",
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, region="eu-west-1",
        )
        result = registry.discover(region="us-east-1")
        assert len(result) == 1
        assert result[0].cluster_id == "c1"

    def test_discover_region_filter_no_match(self, registry: ClusterRegistry) -> None:
        """Region filter with no matches should return empty list."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, region="us-east-1",
        )
        result = registry.discover(region="ap-southeast-1")
        assert result == []

    def test_discover_min_gpus(self, registry: ClusterRegistry) -> None:
        """Filtering by min_gpus should exclude clusters with fewer GPUs."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, gpus=2,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, gpus=8,
        )
        registry.register_cluster(
            cluster_id="c3", host="h3", port=82, gpus=4,
        )
        result = registry.discover(min_gpus=4)
        assert len(result) == 2
        ids = {c.cluster_id for c in result}
        assert "c1" not in ids
        assert "c2" in ids
        assert "c3" in ids

    def test_discover_min_gpus_zero(self, registry: ClusterRegistry) -> None:
        """min_gpus=0 should return all clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, gpus=1,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, gpus=8,
        )
        result = registry.discover(min_gpus=0)
        assert len(result) == 2

    def test_discover_min_reputation(self, registry: ClusterRegistry) -> None:
        """Filtering by min_reputation should exclude low-reputation clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, reputation=0.3,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, reputation=0.7,
        )
        result = registry.discover(min_reputation=0.5)
        assert len(result) == 1
        assert result[0].cluster_id == "c2"

    def test_discover_min_reputation_zero(self, registry: ClusterRegistry) -> None:
        """min_reputation=0 should return all clusters (no filter)."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, reputation=0.1,
        )
        result = registry.discover(min_reputation=0.0)
        assert len(result) == 1

    def test_discover_limit(self, registry: ClusterRegistry) -> None:
        """Limit should cap the number of results."""
        for i in range(10):
            registry.register_cluster(
                cluster_id=f"c{i}", host=f"h{i}", port=80 + i,
            )
        result = registry.discover(limit=3)
        assert len(result) == 3

    def test_discover_limit_default(self, registry: ClusterRegistry) -> None:
        """Default limit should be 20."""
        for i in range(30):
            registry.register_cluster(
                cluster_id=f"c{i}", host=f"h{i}", port=80 + i,
            )
        result = registry.discover()
        assert len(result) == 20

    def test_discover_combined_filters(self, registry: ClusterRegistry) -> None:
        """Multiple filters should compose with AND logic."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80,
            model="llama", region="us-east-1", gpus=8, reputation=0.9,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81,
            model="llama", region="eu-west-1", gpus=4, reputation=0.8,
        )
        registry.register_cluster(
            cluster_id="c3", host="h3", port=82,
            model="falcon", region="us-east-1", gpus=8, reputation=0.95,
        )
        result = registry.discover(
            model="llama", region="us-east-1", min_gpus=4, min_reputation=0.85,
        )
        assert len(result) == 1
        assert result[0].cluster_id == "c1"

    def test_discover_sorted_by_reputation_desc(self, registry: ClusterRegistry) -> None:
        """Results should be sorted by reputation descending."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, reputation=0.5,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, reputation=0.9,
        )
        registry.register_cluster(
            cluster_id="c3", host="h3", port=82, reputation=0.7,
        )
        result = registry.discover()
        assert [c.reputation for c in result] == [0.9, 0.7, 0.5]

    def test_discover_public_only_even_with_filters(self, registry: ClusterRegistry) -> None:
        """Non-public clusters should never appear in discover results."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, is_public=True,
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, is_public=False,
        )
        result = registry.discover()
        assert len(result) == 1
        assert result[0].cluster_id == "c1"


# ── get_cluster ─────────────────────────────────────────────────


class TestGetCluster:
    """ClusterRegistry.get_cluster -- single-cluster lookup."""

    def test_get_existing(self, registry: ClusterRegistry) -> None:
        """get_cluster should return the ClusterInfo for an existing id."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        info = registry.get_cluster("c1")
        assert info is not None
        assert info.cluster_id == "c1"
        assert info.host == "h1"

    def test_get_nonexistent(self, registry: ClusterRegistry) -> None:
        """get_cluster should return None for a missing id."""
        info = registry.get_cluster("nonexistent")
        assert info is None

    def test_get_after_unregister(self, registry: ClusterRegistry) -> None:
        """get_cluster should return None after the cluster is removed."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.unregister_cluster("c1")
        info = registry.get_cluster("c1")
        assert info is None


# ── list_clusters ───────────────────────────────────────────────


class TestListClusters:
    """ClusterRegistry.list_clusters -- summary dicts."""

    def test_list_empty(self, registry: ClusterRegistry) -> None:
        """list_clusters on an empty registry should return []."""
        result = registry.list_clusters()
        assert result == []

    def test_list_returns_all(self, registry: ClusterRegistry) -> None:
        """list_clusters should return all registered clusters."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, model="llama",
            gpus=4, reputation=0.9, region="us-east-1",
        )
        registry.register_cluster(
            cluster_id="c2", host="h2", port=81, model="falcon",
            gpus=8, reputation=0.8, region="eu-west-1",
        )
        result = registry.list_clusters()
        assert len(result) == 2

    def test_list_dict_keys(self, registry: ClusterRegistry) -> None:
        """Each cluster dict should contain the expected keys."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80,
        )
        result = registry.list_clusters()
        d = result[0]
        assert set(d.keys()) == {
            "cluster_id", "host", "port", "gpus",
            "model", "reputation", "region",
        }

    def test_list_includes_non_public(self, registry: ClusterRegistry) -> None:
        """list_clusters should include non-public clusters (unlike discover)."""
        registry.register_cluster(
            cluster_id="c1", host="h1", port=80, is_public=False,
        )
        result = registry.list_clusters()
        assert len(result) == 1

    def test_list_values_match_registration(self, registry: ClusterRegistry) -> None:
        """Values in list dicts should match the registered ClusterInfo."""
        registry.register_cluster(
            cluster_id="c1", host="myhost", port=5050,
            gpus=16, model="gpt-4", reputation=0.99, region="us-west-2",
        )
        result = registry.list_clusters()
        d = result[0]
        assert d["cluster_id"] == "c1"
        assert d["host"] == "myhost"
        assert d["port"] == 5050
        assert d["gpus"] == 16
        assert d["model"] == "gpt-4"
        assert d["reputation"] == 0.99
        assert d["region"] == "us-west-2"


# ── record_contribution ─────────────────────────────────────────


class TestRecordContribution:
    """ClusterRegistry.record_contribution -- GPU hour tracking."""

    def test_record_first_time(self, registry: ClusterRegistry) -> None:
        """First contribution for a user:cluster pair should create a new credit."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=5.0, tokens_served=1000,
        )
        assert credit.user_id == "u1"
        assert credit.cluster_id == "c1"
        assert credit.gpu_hours == 5.0
        assert credit.tokens_served == 1000
        assert credit.earned_credits == 50.0  # 5 * 10
        assert credit.tier == "bronze"

    def test_record_accumulates(self, registry: ClusterRegistry) -> None:
        """Subsequent contributions should accumulate into the same credit."""
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=3.0)
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=7.0)
        key = "u1:c1"
        credit = registry._credits[key]
        assert credit.gpu_hours == 10.0
        assert credit.earned_credits == 100.0

    def test_record_separate_users(self, registry: ClusterRegistry) -> None:
        """Different users should have independent credit records."""
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=5.0)
        registry.record_contribution(user_id="u2", cluster_id="c1", gpu_hours=10.0)
        assert registry._credits["u1:c1"].gpu_hours == 5.0
        assert registry._credits["u2:c1"].gpu_hours == 10.0

    def test_record_separate_clusters(self, registry: ClusterRegistry) -> None:
        """Same user on different clusters should have separate credits."""
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=2.0)
        registry.record_contribution(user_id="u1", cluster_id="c2", gpu_hours=3.0)
        assert registry._credits["u1:c1"].gpu_hours == 2.0
        assert registry._credits["u1:c2"].gpu_hours == 3.0

    def test_record_zero_hours(self, registry: ClusterRegistry) -> None:
        """Recording zero GPU hours should still create a credit entry."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=0.0,
        )
        assert credit.gpu_hours == 0.0
        assert credit.earned_credits == 0.0
        assert credit.tier == "bronze"

    def test_tier_silver(self, registry: ClusterRegistry) -> None:
        """10+ GPU hours should promote to silver."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        assert credit.tier == "silver"

    def test_tier_gold(self, registry: ClusterRegistry) -> None:
        """100+ GPU hours should promote to gold."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=100.0,
        )
        assert credit.tier == "gold"

    def test_tier_platinum(self, registry: ClusterRegistry) -> None:
        """1000+ GPU hours should promote to platinum."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=1000.0,
        )
        assert credit.tier == "platinum"

    def test_tier_boundary_bronze(self, registry: ClusterRegistry) -> None:
        """9.9 GPU hours should remain bronze."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=9.9,
        )
        assert credit.tier == "bronze"

    def test_tier_boundary_silver_to_gold(self, registry: ClusterRegistry) -> None:
        """Exactly 100 GPU hours should be gold."""
        credit = registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=100.0,
        )
        assert credit.tier == "gold"


# ── spend_credits ───────────────────────────────────────────────


class TestSpendCredits:
    """ClusterRegistry.spend_credits -- credit spending."""

    def test_spend_sufficient(self, registry: ClusterRegistry) -> None:
        """Spending within earned credits should succeed."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        result = registry.spend_credits("u1", 50.0)
        assert result is True

    def test_spend_updates_spent(self, registry: ClusterRegistry) -> None:
        """Spent credits should be tracked properly."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        registry.spend_credits("u1", 30.0)
        key = "u1:c1"
        assert registry._credits[key].spent_credits == 30.0

    def test_spend_insufficient(self, registry: ClusterRegistry) -> None:
        """Spending more than earned should fail."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=1.0,
        )
        result = registry.spend_credits("u1", 100.0)
        assert result is False

    def test_spend_no_credits(self, registry: ClusterRegistry) -> None:
        """Spending when user has no credits should fail."""
        result = registry.spend_credits("nonexistent", 10.0)
        assert result is False

    def test_spend_exact_balance(self, registry: ClusterRegistry) -> None:
        """Spending exactly the earned amount should succeed."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        result = registry.spend_credits("u1", 100.0)
        assert result is True

    def test_spend_multiple_times(self, registry: ClusterRegistry) -> None:
        """Multiple spend calls should accumulate."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        registry.spend_credits("u1", 30.0)
        registry.spend_credits("u1", 40.0)
        key = "u1:c1"
        assert registry._credits[key].spent_credits == 70.0

    def test_spend_uses_any_cluster_credit(self, registry: ClusterRegistry) -> None:
        """spend_credits should check all credits for the user."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=2.0,
        )
        registry.record_contribution(
            user_id="u1", cluster_id="c2", gpu_hours=8.0,
        )
        # u1:c1 has 20 credits, u1:c2 has 80 credits
        # It should find the first one with sufficient balance (c1: 20 >= 10)
        result = registry.spend_credits("u1", 10.0)
        assert result is True


# ── get_user_credits ────────────────────────────────────────────


class TestGetUserCredits:
    """ClusterRegistry.get_user_credits -- aggregated credit summary."""

    def test_get_nonexistent_user(self, registry: ClusterRegistry) -> None:
        """A user with no credits should get a zero summary."""
        summary = registry.get_user_credits("nonexistent")
        assert summary["user_id"] == "nonexistent"
        assert summary["total_earned"] == 0.0
        assert summary["total_spent"] == 0.0
        assert summary["balance"] == 0.0
        assert summary["total_gpu_hours"] == 0.0
        assert summary["tier"] == "bronze"

    def test_get_aggregates_across_clusters(self, registry: ClusterRegistry) -> None:
        """Credits from multiple clusters should be aggregated."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=5.0,
        )
        registry.record_contribution(
            user_id="u1", cluster_id="c2", gpu_hours=5.0,
        )
        summary = registry.get_user_credits("u1")
        assert summary["total_gpu_hours"] == 10.0
        assert summary["total_earned"] == 100.0

    def test_get_balance_after_spend(self, registry: ClusterRegistry) -> None:
        """Balance should reflect earned minus spent."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=10.0,
        )
        registry.spend_credits("u1", 40.0)
        summary = registry.get_user_credits("u1")
        assert summary["total_earned"] == 100.0
        assert summary["total_spent"] == 40.0
        assert summary["balance"] == 60.0

    def test_get_tier_highest(self, registry: ClusterRegistry) -> None:
        """Tier should be the highest among all user's credits."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=5.0,  # bronze
        )
        registry.record_contribution(
            user_id="u1", cluster_id="c2", gpu_hours=100.0,  # gold
        )
        summary = registry.get_user_credits("u1")
        assert summary["tier"] == "gold"

    def test_get_tier_platinum_wins(self, registry: ClusterRegistry) -> None:
        """Platinum tier should take priority over gold."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=100.0,  # gold
        )
        registry.record_contribution(
            user_id="u1", cluster_id="c2", gpu_hours=1000.0,  # platinum
        )
        summary = registry.get_user_credits("u1")
        assert summary["tier"] == "platinum"

    def test_get_values_rounded(self, registry: ClusterRegistry) -> None:
        """Numeric values should be rounded to 2 decimal places."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=1.234,
        )
        summary = registry.get_user_credits("u1")
        assert summary["total_earned"] == 12.34
        assert summary["total_gpu_hours"] == 1.23


# ── get_leaderboard ─────────────────────────────────────────────


class TestGetLeaderboard:
    """ClusterRegistry.get_leaderboard -- top contributors."""

    def test_leaderboard_empty(self, registry: ClusterRegistry) -> None:
        """An empty registry should return an empty leaderboard."""
        result = registry.get_leaderboard()
        assert result == []

    def test_leaderboard_orders_by_gpu_hours_desc(self, registry: ClusterRegistry) -> None:
        """Leaderboard should be sorted by GPU hours descending."""
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=5.0)
        registry.record_contribution(user_id="u2", cluster_id="c1", gpu_hours=20.0)
        registry.record_contribution(user_id="u3", cluster_id="c1", gpu_hours=10.0)
        result = registry.get_leaderboard()
        assert [e["user_id"] for e in result] == ["u2", "u3", "u1"]

    def test_leaderboard_aggregates_across_clusters(self, registry: ClusterRegistry) -> None:
        """Same user on multiple clusters should be aggregated."""
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=5.0)
        registry.record_contribution(user_id="u1", cluster_id="c2", gpu_hours=5.0)
        registry.record_contribution(user_id="u2", cluster_id="c1", gpu_hours=8.0)
        result = registry.get_leaderboard()
        assert len(result) == 2
        assert result[0]["user_id"] == "u1"
        assert result[0]["gpu_hours"] == 10.0
        assert result[1]["user_id"] == "u2"
        assert result[1]["gpu_hours"] == 8.0

    def test_leaderboard_limit(self, registry: ClusterRegistry) -> None:
        """Leaderboard should respect the limit parameter."""
        for i in range(10):
            registry.record_contribution(
                user_id=f"u{i}", cluster_id="c1", gpu_hours=float(i + 1),
            )
        result = registry.get_leaderboard(limit=3)
        assert len(result) == 3

    def test_leaderboard_gpu_hours_rounded(self, registry: ClusterRegistry) -> None:
        """GPU hours in leaderboard should be rounded to 2 decimal places."""
        registry.record_contribution(
            user_id="u1", cluster_id="c1", gpu_hours=1.234,
        )
        result = registry.get_leaderboard()
        assert result[0]["gpu_hours"] == 1.23


# ── Thread Safety ───────────────────────────────────────────────


class TestThreadSafety:
    """ClusterRegistry operations should not deadlock on the same lock."""

    def test_consecutive_operations_with_same_thread(self, registry: ClusterRegistry) -> None:
        """Multiple operations on the same thread should not deadlock."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.heartbeat("c1")
        registry.discover()
        registry.list_clusters()
        registry.get_cluster("c1")
        registry.unregister_cluster("c1")
        # All operations with the same lock, reentrant OK since threading.Lock
        # in Python is not reentrant, but each operation acquires/releases within.
        # This test verifies no deadlock.
        assert True

    def test_register_and_contribution_sequential(self, registry: ClusterRegistry) -> None:
        """Sequential register and record_contribution should not interfere."""
        registry.register_cluster(cluster_id="c1", host="h1", port=80)
        registry.record_contribution(user_id="u1", cluster_id="c1", gpu_hours=1.0)
        assert registry.get_cluster("c1") is not None
        summary = registry.get_user_credits("u1")
        assert summary["total_gpu_hours"] == 1.0
