"""Tests for federated draft banks.

Covers:
- DraftBankEntry properties and edge cases
- FederationDraftConfig construction
- FederatedDraftBank registration, recording, decay, and status
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.draft_bank import (
    DraftBankEntry,
    FederatedDraftBank,
    FederationDraftConfig,
)


class TestDraftBankEntry:
    """Tests for DraftBankEntry data class and its computed properties."""

    def test_defaults(self) -> None:
        """Default field values are set correctly."""
        entry = DraftBankEntry(
            cluster_id="test", endpoint_url="http://localhost:9000"
        )
        assert entry.cluster_id == "test"
        assert entry.endpoint_url == "http://localhost:9000"
        assert entry.model_name == ""
        assert entry.hardware == "cpu"
        assert entry.cost_per_hour == 0.0
        assert entry.avg_latency_ms == 0.0
        assert entry.max_concurrent == 10
        assert entry.current_load == 0
        assert entry.total_served == 0
        assert entry.total_errors == 0
        assert entry.reputation_score == 1.0
        assert entry.region == ""

    def test_is_overloaded_true(self) -> None:
        """Entry is overloaded when current_load >= max_concurrent."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=10,
            max_concurrent=10,
        )
        assert entry.is_overloaded is True

    def test_is_overloaded_false(self) -> None:
        """Entry is not overloaded when current_load < max_concurrent."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=5,
            max_concurrent=10,
        )
        assert entry.is_overloaded is False

    def test_is_overloaded_exceeds(self) -> None:
        """Entry is overloaded when current_load exceeds max_concurrent."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=15,
            max_concurrent=10,
        )
        assert entry.is_overloaded is True

    def test_availability_score_full(self) -> None:
        """Entry with no load and recent timestamp scores 1.0."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=0,
            max_concurrent=10,
            last_seen=time.time(),
        )
        assert entry.availability_score == pytest.approx(1.0)

    def test_availability_score_partial(self) -> None:
        """Partial load yields proportional availability score."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=3,
            max_concurrent=10,
            last_seen=time.time(),
        )
        expected = 1.0 - 3.0 / 10.0
        assert entry.availability_score == pytest.approx(expected)

    def test_availability_score_overloaded(self) -> None:
        """Overloaded entry scores 0.0."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=10,
            max_concurrent=10,
            last_seen=time.time(),
        )
        assert entry.availability_score == 0.0

    def test_availability_score_stale(self) -> None:
        """Stale entry scores 0.0 regardless of load."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=0,
            max_concurrent=10,
            last_seen=0.0,  # Epoch -- far in the past
        )
        assert entry.availability_score == 0.0

    def test_availability_score_zero_max_concurrent(self) -> None:
        """Zero max_concurrent means overloaded (0 >= 0), score = 0.0."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            current_load=0,
            max_concurrent=0,
            last_seen=time.time(),
        )
        # current_load >= max_concurrent triggers overloaded
        assert entry.is_overloaded is True
        assert entry.availability_score == 0.0

    def test_is_stale_property_true(self) -> None:
        """is_stale property returns True for old entries."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            last_seen=0.0,
        )
        # Accessed as a property (not called as method); uses default 60s threshold
        assert entry.is_stale is True

    def test_is_stale_property_false(self) -> None:
        """is_stale property returns False for fresh entries."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            last_seen=time.time(),
        )
        assert entry.is_stale is False

    def test_is_stale_boundary(self) -> None:
        """Fresh entry within threshold is not stale."""
        entry = DraftBankEntry(
            cluster_id="a",
            endpoint_url="http://a",
            last_seen=time.time() - 59.0,  # Just under 60s
        )
        assert entry.is_stale is False


class TestFederationDraftConfig:
    """Tests for FederationDraftConfig data class."""

    def test_defaults(self) -> None:
        """Default configuration values."""
        config = FederationDraftConfig()
        assert config.own_cluster_id == ""
        assert config.own_host == ""
        assert config.own_port == 9000
        assert config.own_model_name == ""
        assert config.own_hardware == "cpu"
        assert config.own_cost_per_hour == 0.05
        assert config.own_max_concurrent == 10
        assert config.discovery_interval_s == 30.0
        assert config.stale_threshold_s == 60.0
        assert config.reputation_decay == 0.95
        assert config.min_reputation == 0.1

    def test_custom(self) -> None:
        """Explicit configuration values are stored."""
        config = FederationDraftConfig(
            own_cluster_id="my-cluster",
            own_host="10.0.0.1",
            own_port=9001,
            own_model_name="draft-v2",
            own_hardware="cuda",
            own_cost_per_hour=0.10,
            own_max_concurrent=20,
            stale_threshold_s=120.0,
            min_reputation=0.2,
        )
        assert config.own_cluster_id == "my-cluster"
        assert config.own_host == "10.0.0.1"
        assert config.own_port == 9001
        assert config.own_model_name == "draft-v2"
        assert config.own_hardware == "cuda"
        assert config.own_cost_per_hour == 0.10
        assert config.own_max_concurrent == 20
        assert config.stale_threshold_s == 120.0
        assert config.min_reputation == 0.2


class TestFederatedDraftBank:
    """Tests for FederatedDraftBank orchestration."""

    def test_init_default_config(self) -> None:
        """Bank without explicit config uses constructor args in default config."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        assert bank._config.own_cluster_id == "c1"
        assert bank._config.own_host == "10.0.0.1"
        assert bank._config.own_port == 9000
        assert len(bank._entries) == 0

    def test_init_with_config(self) -> None:
        """Bank with explicit config uses it directly."""
        config = FederationDraftConfig(
            own_cluster_id="c2",
            own_host="10.0.0.2",
            own_port=9001,
        )
        bank = FederatedDraftBank("c2", "10.0.0.2", 9001, config=config)
        assert bank._config.own_port == 9001

    def test_register_local_capacity(self) -> None:
        """Register local capacity adds an entry for own cluster."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(
            model_name="draft-v1",
            hardware="cpu",
            cost_per_hour=0.05,
            max_concurrent=10,
        )
        assert "c1" in bank._entries
        entry = bank._entries["c1"]
        assert entry.model_name == "draft-v1"
        assert entry.hardware == "cpu"
        assert entry.endpoint_url == "http://10.0.0.1:9000"
        assert entry.cost_per_hour == 0.05
        assert entry.max_concurrent == 10
        assert entry.reputation_score == 1.0
        assert entry.current_load == 0

    def test_register_local_capacity_overwrites(self) -> None:
        """Re-registering local capacity overwrites existing entry."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="v1", max_concurrent=5)
        bank.register_local_capacity(model_name="v2", max_concurrent=10)
        assert len(bank._entries) == 1
        assert bank._entries["c1"].model_name == "v2"

    def test_record_success(self) -> None:
        """Record success updates metrics and boosts reputation."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]
        entry.reputation_score = 0.9  # Start below max to observe boost

        bank.record_success(
            "c1", latency_s=0.1, tokens_generated=100, acceptance_rate=0.8
        )

        assert entry.total_served == 1
        # EWMA: 0 * 0.8 + (0.1 * 1000) * 0.2 = 20.0
        assert entry.avg_latency_ms == pytest.approx(20.0)
        # EWMA: 0 * 0.8 + 0.8 * 0.2 = 0.16
        assert entry.avg_acceptance_rate == pytest.approx(0.16)
        # Boost: 0.9 * 1.01 = 0.909
        assert entry.reputation_score == pytest.approx(0.909)

    def test_record_success_caps_reputation_at_one(self) -> None:
        """Record success caps reputation boost at 1.0."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]
        entry.reputation_score = 0.99

        bank.record_success("c1", latency_s=0.1, tokens_generated=100)

        # 0.99 * 1.01 = 0.9999, still below 1.0
        assert entry.reputation_score == pytest.approx(0.9999)

    def test_record_success_caps_at_one(self) -> None:
        """Reputation is capped at 1.0 even with repeated success."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]

        bank.record_success("c1", latency_s=0.01, tokens_generated=10)
        bank.record_success("c1", latency_s=0.01, tokens_generated=10)

        assert entry.reputation_score == 1.0  # Already at cap

    def test_record_success_unknown_cluster(self) -> None:
        """Record success for unknown cluster silently returns."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        # Should not raise
        bank.record_success("unknown", latency_s=0.1, tokens_generated=10)
        assert len(bank._entries) == 0

    def test_record_error(self) -> None:
        """Record error increases error count and reduces reputation."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]

        bank.record_error("c1", "timeout")

        assert entry.total_errors == 1
        # 1.0 * 0.9 = 0.9
        assert entry.reputation_score == pytest.approx(0.9)

    def test_record_error_clamps_reputation(self) -> None:
        """Record error does not push reputation below min_reputation."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]
        entry.reputation_score = 0.11

        bank.record_error("c1", "timeout")

        # 0.11 * 0.9 = 0.099, clamped to min_reputation (0.1)
        assert entry.reputation_score == pytest.approx(0.1)

    def test_record_error_multiple(self) -> None:
        """Multiple errors compound reputation penalty."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        entry = bank._entries["c1"]

        bank.record_error("c1", "timeout")
        bank.record_error("c1", "timeout")
        # 1.0 * 0.9 = 0.9, then 0.9 * 0.9 = 0.81
        assert entry.reputation_score == pytest.approx(0.81)

    def test_record_error_unknown_cluster(self) -> None:
        """Record error for unknown cluster silently returns."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.record_error("unknown", "timeout")
        assert len(bank._entries) == 0

    def test_record_request_start(self) -> None:
        """Record request start increments current_load."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank.record_request_start("c1")
        assert bank._entries["c1"].current_load == 1

    def test_record_request_start_multiple(self) -> None:
        """Multiple concurrent requests accumulate load."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank.record_request_start("c1")
        bank.record_request_start("c1")
        bank.record_request_start("c1")
        assert bank._entries["c1"].current_load == 3

    def test_record_request_start_unknown(self) -> None:
        """Record request start for unknown cluster silently returns."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.record_request_start("unknown")
        # No error, no entry created

    def test_record_request_end(self) -> None:
        """Record request end decrements current_load."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank._entries["c1"].current_load = 5
        bank.record_request_end("c1")
        assert bank._entries["c1"].current_load == 4

    def test_record_request_end_unknown(self) -> None:
        """Record request end for unknown cluster silently returns."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.record_request_end("unknown")

    def test_record_request_end_below_zero(self) -> None:
        """Record request end does not decrement below zero."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        # current_load starts at 0
        bank.record_request_end("c1")
        assert bank._entries["c1"].current_load == 0

    def test_decay_reputation(self) -> None:
        """Decay reputation multiplies all scores by decay factor."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank._entries["c1"].reputation_score = 1.0

        bank.decay_reputation()

        assert bank._entries["c1"].reputation_score == pytest.approx(0.95)

    def test_decay_reputation_multiple_entries(self) -> None:
        """Decay applies to all entries in the bank."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank._entries["c1"].reputation_score = 1.0
        bank._entries["c2"] = DraftBankEntry(
            cluster_id="c2",
            endpoint_url="http://10.0.0.2:9000",
            reputation_score=0.5,
            last_seen=time.time(),
        )

        bank.decay_reputation()

        assert bank._entries["c1"].reputation_score == pytest.approx(0.95)
        assert bank._entries["c2"].reputation_score == pytest.approx(0.475)

    def test_decay_reputation_minimum(self) -> None:
        """Decay reputation does not go below min_reputation."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")
        bank._entries["c1"].reputation_score = 0.1

        bank.decay_reputation()

        # 0.1 * 0.95 = 0.095, clamped to min_reputation (0.1)
        assert bank._entries["c1"].reputation_score == pytest.approx(0.1)

    def test_get_best_draft_endpoint_empty(self) -> None:
        """get_best_draft_endpoint returns None when no entries exist."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        result = bank.get_best_draft_endpoint()
        assert result is None

    def test_get_best_draft_endpoint_empty_with_args(self) -> None:
        """get_best_draft_endpoint with custom criteria returns None when empty."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        result = bank.get_best_draft_endpoint(
            workload_type="code",
            max_latency_ms=50.0,
            max_cost_per_hour=5.0,
            prefer_local=False,
        )
        assert result is None

    def test_get_federation_status_empty(self) -> None:
        """get_federation_status returns empty status when no entries exist."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        status = bank.get_federation_status()
        assert status["total_endpoints"] == 0
        assert status["healthy_endpoints"] == 0
        assert status["endpoints"] == []

    def test_discover_and_register_no_seeds(self) -> None:
        """discover_and_register with no seed nodes returns empty list."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        result = bank.discover_and_register(seed_nodes=None)
        assert result == []

    def test_discover_and_register_empty_seeds(self) -> None:
        """discover_and_register with empty seed list returns empty list."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        result = bank.discover_and_register(seed_nodes=[])
        assert result == []

    def test_record_operations_independent(self) -> None:
        """Success and error recording on different clusters are independent."""
        bank = FederatedDraftBank("c1", "10.0.0.1", 9000)
        bank.register_local_capacity(model_name="draft-v1")

        entry_a = bank._entries["c1"]
        entry_a.reputation_score = 0.9
        entry_b = DraftBankEntry(
            cluster_id="c2",
            endpoint_url="http://10.0.0.2:9000",
            last_seen=time.time(),
        )
        bank._entries["c2"] = entry_b

        bank.record_success("c1", latency_s=0.05, tokens_generated=50, acceptance_rate=0.9)
        bank.record_error("c2", "timeout")

        # Entry A was sucessful
        assert entry_a.total_served == 1
        assert entry_a.total_errors == 0
        assert entry_a.reputation_score == pytest.approx(0.909)  # 0.9 * 1.01

        # Entry B had an error
        assert entry_b.total_served == 0
        assert entry_b.total_errors == 1
        assert entry_b.reputation_score == pytest.approx(0.9)  # 1.0 * 0.9
