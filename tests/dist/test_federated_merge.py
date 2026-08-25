"""Comprehensive tests for federated LoRA merging.

Tests FederatedMergeCoordinator (registration, rounds, submission, merge
strategies) and SecureAggregator (additive secret sharing split/aggregate).
No mocks — uses real torch tensors saved to temp files for merge tests.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from itertools import combinations

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric import ed25519
from loguru import logger

from distllm.dist.byzantine import _sign_bytes
from distllm.dist.federated_merge import (
    AdapterVersion,
    FederatedMergeCoordinator,
    FederatedRound,
    NodeTrainingState,
    SecureAggregator,
)


# =========================================================================
# Dataclass defaults
# =========================================================================


class TestFederatedRound:
    """FederatedRound dataclass construction and defaults."""

    def test_default_status(self):
        """Default status should be 'pending'."""
        r = FederatedRound(round_id="r1", round_number=1)
        assert r.status == "pending"

    def test_default_timestamps(self):
        """started_at and completed_at should default to 0.0."""
        r = FederatedRound(round_id="r1", round_number=1)
        assert r.started_at == 0.0
        assert r.completed_at == 0.0

    def test_participating_nodes_default_empty(self):
        """participating_nodes should default to empty list."""
        r = FederatedRound(round_id="r1", round_number=1)
        assert r.participating_nodes == []

    def test_full_construction(self):
        """All fields can be set via constructor."""
        r = FederatedRound(
            round_id="r1",
            round_number=5,
            started_at=100.0,
            completed_at=200.0,
            participating_nodes=["n1", "n2"],
            node_weights={"n1": 100, "n2": 200},
            merged_adapter_path="/tmp/adapter.pt",
            status="completed",
            loss_values={"n1": 0.5, "n2": 0.3},
        )
        assert r.round_id == "r1"
        assert r.round_number == 5
        assert r.started_at == 100.0
        assert r.completed_at == 200.0
        assert r.participating_nodes == ["n1", "n2"]
        assert r.node_weights == {"n1": 100, "n2": 200}
        assert r.merged_adapter_path == "/tmp/adapter.pt"
        assert r.status == "completed"
        assert r.loss_values == {"n1": 0.5, "n2": 0.3}


class TestAdapterVersion:
    """AdapterVersion dataclass construction and defaults."""

    def test_default_created_at(self):
        """created_at should default to a positive timestamp."""
        v = AdapterVersion(version_id="v1", adapter_id="a1", round_number=3)
        assert v.created_at > 0

    def test_default_path_empty(self):
        """path should default to empty string."""
        v = AdapterVersion(version_id="v1", adapter_id="a1", round_number=3)
        assert v.path == ""

    def test_default_metrics_empty_dict(self):
        """metrics should default to empty dict."""
        v = AdapterVersion(version_id="v1", adapter_id="a1", round_number=3)
        assert v.metrics == {}

    def test_default_parent_versions_empty(self):
        """parent_versions should default to empty list."""
        v = AdapterVersion(version_id="v1", adapter_id="a1", round_number=3)
        assert v.parent_versions == []

    def test_full_construction(self):
        """All fields can be set via constructor."""
        v = AdapterVersion(
            version_id="v-abc",
            adapter_id="adapter-x",
            round_number=7,
            created_at=500.0,
            path="/tmp/v.pt",
            metrics={"avg_loss": 0.4},
            parent_versions=["v-001", "v-002"],
        )
        assert v.version_id == "v-abc"
        assert v.adapter_id == "adapter-x"
        assert v.round_number == 7
        assert v.created_at == 500.0
        assert v.path == "/tmp/v.pt"
        assert v.metrics == {"avg_loss": 0.4}
        assert v.parent_versions == ["v-001", "v-002"]


class TestNodeTrainingState:
    """NodeTrainingState dataclass construction and defaults."""

    def test_default_status(self):
        """Default status should be 'idle'."""
        s = NodeTrainingState(node_id="n1")
        assert s.status == "idle"

    def test_default_epochs_and_batch_size(self):
        """Default local_epochs=3 and local_batch_size=4."""
        s = NodeTrainingState(node_id="n1")
        assert s.local_epochs == 3
        assert s.local_batch_size == 4

    def test_default_learning_rate(self):
        """Default learning_rate=2e-4."""
        s = NodeTrainingState(node_id="n1")
        assert s.learning_rate == 2e-4

    def test_default_adapter_path_empty(self):
        """adapter_path should default to empty string."""
        s = NodeTrainingState(node_id="n1")
        assert s.adapter_path == ""

    def test_default_dataset_size_zero(self):
        """dataset_size should default to 0."""
        s = NodeTrainingState(node_id="n1")
        assert s.dataset_size == 0

    def test_default_last_loss_zero(self):
        """last_loss should default to 0.0."""
        s = NodeTrainingState(node_id="n1")
        assert s.last_loss == 0.0

    def test_default_last_sync_zero(self):
        """last_sync should default to 0.0."""
        s = NodeTrainingState(node_id="n1")
        assert s.last_sync == 0.0

    def test_full_construction(self):
        """All fields can be set via constructor."""
        s = NodeTrainingState(
            node_id="n42",
            current_round=3,
            local_epochs=5,
            local_batch_size=8,
            learning_rate=1e-3,
            adapter_path="/tmp/n42.pt",
            dataset_size=1000,
            last_loss=0.25,
            last_sync=1234.0,
            status="training",
        )
        assert s.node_id == "n42"
        assert s.current_round == 3
        assert s.local_epochs == 5
        assert s.local_batch_size == 8
        assert s.learning_rate == 1e-3
        assert s.adapter_path == "/tmp/n42.pt"
        assert s.dataset_size == 1000
        assert s.last_loss == 0.25
        assert s.last_sync == 1234.0
        assert s.status == "training"


# =========================================================================
# FederatedMergeCoordinator
# =========================================================================


class TestFederatedMergeCoordinatorRegister:
    """Node registration / unregistration."""

    def test_register_node_returns_state(self):
        """register_node returns a NodeTrainingState with matching node_id."""
        coord = FederatedMergeCoordinator()
        state = coord.register_node("node-a", dataset_size=500, local_epochs=3)
        assert isinstance(state, NodeTrainingState)
        assert state.node_id == "node-a"
        assert state.dataset_size == 500
        assert state.local_epochs == 3

    def test_register_node_stores_internal(self):
        """Registered node is accessible via get_node_states."""
        coord = FederatedMergeCoordinator()
        coord.register_node("node-a")
        states = coord.get_node_states()
        assert "node-a" in states
        assert states["node-a"].node_id == "node-a"

    def test_register_multiple_nodes(self):
        """Multiple nodes can be registered."""
        coord = FederatedMergeCoordinator()
        coord.register_node("n1")
        coord.register_node("n2")
        coord.register_node("n3")
        assert len(coord.get_node_states()) == 3

    def test_unregister_node_removes(self):
        """Unregistered node is no longer in get_node_states."""
        coord = FederatedMergeCoordinator()
        coord.register_node("node-a")
        coord.register_node("node-b")
        coord.unregister_node("node-a")
        states = coord.get_node_states()
        assert "node-a" not in states
        assert "node-b" in states

    def test_unregister_nonexistent_node_no_error(self):
        """Unregistering a non-existent node does not raise."""
        coord = FederatedMergeCoordinator()
        coord.unregister_node("ghost")  # should not raise


class TestFederatedMergeCoordinatorStartRound:
    """Start round logic."""

    def test_start_round_insufficient_nodes(self):
        """start_round returns None when fewer than min_nodes."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=3)
        coord.register_node("n1")
        coord.register_node("n2")
        # Only 2 nodes, need 3
        assert coord.start_round() is None

    def test_start_round_sufficient_nodes(self):
        """start_round returns a FederatedRound when enough nodes."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1")
        coord.register_node("n2")
        round_ = coord.start_round()
        assert round_ is not None
        assert isinstance(round_, FederatedRound)
        assert round_.round_number == 1
        assert round_.status == "collecting"
        assert "n1" in round_.participating_nodes
        assert "n2" in round_.participating_nodes

    def test_start_round_increments_round_number(self):
        """Each start_round increments the round number (based on completed rounds)."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n1", dataset_size=10)
        r1 = coord.start_round()
        assert r1 is not None
        assert r1.round_number == 1

        # Complete the round so _rounds list grows
        adapter = _create_adapter_file({"w": torch.tensor([1.0])})
        try:
            coord.submit_node_adapter("n1", adapter, 0.1, dataset_size=10)
            coord.merge_adapters()
            r2 = coord.start_round()
            assert r2 is not None
            assert r2.round_number == 2
        finally:
            os.unlink(adapter)

    def test_start_round_sets_nodes_to_training(self):
        """Starting a round marks idle nodes as 'training'."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1")
        coord.register_node("n2")
        coord.start_round()
        states = coord.get_node_states()
        assert states["n1"].status == "training"
        assert states["n2"].status == "training"

    def test_start_round_only_idle_or_completed_nodes(self):
        """Nodes that are not idle/completed should not participate."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n-idle")
        coord.register_node("n-busy")
        coord._nodes["n-busy"].status = "training"
        round_ = coord.start_round()
        assert round_ is not None
        assert "n-idle" in round_.participating_nodes
        assert "n-busy" not in round_.participating_nodes


class TestFederatedMergeCoordinatorSubmitAdapter:
    """Submit adapter logic."""

    def test_submit_no_current_round(self):
        """Submit returns False when no round is active."""
        coord = FederatedMergeCoordinator()
        assert coord.submit_node_adapter("n1", "/tmp/a.pt", 0.5) is False

    def test_submit_node_not_participating(self):
        """Submit returns False when node is not in the current round."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n1")
        coord.start_round()
        assert coord.submit_node_adapter("n2", "/tmp/a.pt", 0.5) is False

    def test_submit_success(self):
        """Successful submission returns True."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=200)
        coord.start_round()
        assert coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=100) is True

    def test_submit_updates_state(self):
        """Submission updates node state fields."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=200)
        coord.start_round()
        coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.42, dataset_size=100)
        state = coord.get_node_states()["n1"]
        assert state.adapter_path == "/tmp/n1.pt"
        assert state.last_loss == 0.42
        assert state.status == "completed"

    def test_submit_triggers_merge_when_all_submitted(self):
        """When all nodes submit, the round status switches to 'merging'."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=200)
        coord.start_round()
        coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1)
        # Not yet all submitted
        assert coord.get_current_round() is not None
        assert coord.get_current_round().status == "collecting"
        # Second node triggers merge
        coord.submit_node_adapter("n2", "/tmp/n2.pt", 0.2)
        assert coord.get_current_round().status == "merging"

    def test_submit_round_not_collecting(self):
        """Submit returns False if round status is not 'collecting'."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n1", dataset_size=100)
        coord.start_round()
        # Manually set status past collecting
        coord._current_round.status = "merging"
        assert coord.submit_node_adapter("n1", "/tmp/a.pt", 0.1) is False

    def test_submit_updates_round_loss_and_weights(self):
        """Submission records node loss and dataset weight."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=50)
        coord.register_node("n2", dataset_size=150)
        coord.start_round()
        coord.submit_node_adapter("n1", "/tmp/a.pt", 0.3, dataset_size=50)
        cr = coord.get_current_round()
        assert cr.node_weights["n1"] == 50
        assert cr.loss_values["n1"] == 0.3


# =========================================================================
# Merge strategy integration tests — real torch tensors
# =========================================================================


def _create_adapter_file(tensors: dict[str, torch.Tensor]) -> str:
    """Create a temporary .pt file with the given state dict and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(tensors, tmp.name)
    tmp.close()
    return tmp.name


class TestFederatedMergeCoordinatorMerge:
    """Merge adapters with real tensor files."""

    @pytest.fixture(autouse=True)
    def setup_coord(self):
        """Create a coordinator with two registered nodes and a started round."""
        self.coord = FederatedMergeCoordinator(min_nodes_per_round=2, merge_strategy="fedavg")
        self.coord.register_node("n1", dataset_size=100)
        self.coord.register_node("n2", dataset_size=300)
        self.coord.start_round()

        # Create two adapters with known values
        self.t1 = _create_adapter_file({"weight": torch.tensor([1.0, 2.0, 3.0])})
        self.t2 = _create_adapter_file({"weight": torch.tensor([4.0, 5.0, 6.0])})

        yield

        # Cleanup
        for p in [self.t1, self.t2]:
            if os.path.exists(p):
                os.unlink(p)

    def submit_both(self):
        """Helper: submit both nodes' adapters."""
        self.coord.submit_node_adapter("n1", self.t1, 0.2, dataset_size=100)
        self.coord.submit_node_adapter("n2", self.t2, 0.1, dataset_size=300)

    def test_merge_fedavg_default_strategy(self):
        """merge_adapters returns a path under fedavg."""
        self.submit_both()
        result = self.coord.merge_adapters()
        assert result is not None
        assert isinstance(result, str)
        assert result.endswith(".pt")
        assert os.path.exists(result)

    def test_merge_weighted_strategy(self):
        """merge_adapters with weighted strategy returns a path."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2, merge_strategy="weighted")
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=300)
        coord.start_round()
        coord.submit_node_adapter("n1", self.t1, 0.2, dataset_size=100)
        coord.submit_node_adapter("n2", self.t2, 0.1, dataset_size=300)
        result = coord.merge_adapters()
        assert result is not None
        assert os.path.exists(result)

    def test_merge_reputation_strategy(self):
        """merge_adapters with reputation falls back to fedavg and succeeds."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2, merge_strategy="reputation")
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=300)
        coord.start_round()
        coord.submit_node_adapter("n1", self.t1, 0.2, dataset_size=100)
        coord.submit_node_adapter("n2", self.t2, 0.1, dataset_size=300)
        result = coord.merge_adapters()
        assert result is not None
        assert os.path.exists(result)

    def test_merge_result_is_weighted_average(self):
        """The merged values should match the FedAvg-weighted expectation."""
        self.submit_both()
        result_path = self.coord.merge_adapters()
        assert result_path is not None
        merged = torch.load(result_path, map_location="cpu", weights_only=True)
        # n1 weight=100, n2 weight=300, total=400
        # Expected: (1.0*100 + 4.0*300)/400 = (100 + 1200)/400 = 3.25
        #           (2.0*100 + 5.0*300)/400 = (200 + 1500)/400 = 4.25
        #           (3.0*100 + 6.0*300)/400 = (300 + 1800)/400 = 5.25
        expected = torch.tensor([3.25, 4.25, 5.25])
        assert "weight" in merged
        assert torch.allclose(merged["weight"], expected, atol=1e-5)

    def test_merge_records_version(self):
        """After a successful merge, a version is recorded."""
        self.submit_both()
        self.coord.merge_adapters()
        versions = self.coord.get_versions()
        assert len(versions) == 1
        assert versions[0].round_number == 1
        assert "avg_loss" in versions[0].metrics

    def test_merge_appends_to_rounds_history(self):
        """After merge, completed rounds appear in get_rounds."""
        self.submit_both()
        self.coord.merge_adapters()
        rounds = self.coord.get_rounds()
        assert len(rounds) == 1
        assert rounds[0].status == "completed"
        assert rounds[0].round_number == 1

    def test_merge_none_when_not_merging_status(self):
        """merge_adapters returns None if round status is not 'merging'."""
        self.coord.submit_node_adapter("n1", self.t1, 0.2, dataset_size=100)
        # Only one submitted, still 'collecting'
        assert self.coord.merge_adapters() is None

    def test_merge_none_when_no_current_round(self):
        """merge_adapters returns None if no current round."""
        coord = FederatedMergeCoordinator()
        assert coord.merge_adapters() is None

    def test_merge_no_adapters_returns_none(self):
        """If no nodes have adapters, merge_adapters returns None."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n1", dataset_size=100)
        coord.start_round()
        coord._current_round.status = "merging"  # force merging status
        result = coord.merge_adapters()
        assert result is None


class TestFederatedMergeCoordinatorGetStatus:
    """Status query methods."""

    def test_get_current_round_none_initially(self):
        """get_current_round returns None before any round."""
        coord = FederatedMergeCoordinator()
        assert coord.get_current_round() is None

    def test_get_rounds_empty_initially(self):
        """get_rounds returns empty list initially."""
        coord = FederatedMergeCoordinator()
        assert coord.get_rounds() == []

    def test_get_versions_empty_initially(self):
        """get_versions returns empty list initially."""
        coord = FederatedMergeCoordinator()
        assert coord.get_versions() == []

    def test_get_node_states_empty_initially(self):
        """get_node_states returns empty dict initially."""
        coord = FederatedMergeCoordinator()
        assert coord.get_node_states() == {}

    def test_get_versions_sorted_descending(self, tmp_path):
        """get_versions returns versions sorted by round_number descending."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=100)

        # Do two complete rounds
        for _ in range(2):
            a1 = _create_adapter_file({"w": torch.tensor([1.0])})
            a2 = _create_adapter_file({"w": torch.tensor([2.0])})
            coord.start_round()
            coord.submit_node_adapter("n1", a1, 0.1, dataset_size=100)
            coord.submit_node_adapter("n2", a2, 0.1, dataset_size=100)
            coord.merge_adapters()

        versions = coord.get_versions()
        assert len(versions) == 2
        assert versions[0].round_number == 2
        assert versions[1].round_number == 1

    def test_stats_structure(self):
        """get_stats returns a dict with expected keys."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=1)
        coord.register_node("n1", dataset_size=100)
        stats = coord.get_stats()
        assert isinstance(stats, dict)
        assert "total_rounds" in stats
        assert "registered_nodes" in stats
        assert "active_nodes" in stats
        assert "total_versions" in stats
        assert "merge_strategy" in stats
        assert "current_round" in stats
        assert "current_round_status" in stats
        assert "avg_loss_last_round" in stats

    def test_stats_values(self):
        """Stats reflect the actual state."""
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=200)
        stats = coord.get_stats()
        assert stats["registered_nodes"] == 2
        assert stats["total_rounds"] == 0
        assert stats["merge_strategy"] == "fedavg"


# =========================================================================
# SecureAggregator — pairwise-cancelling masks (SecAgg correctness)
#
# Regression tests for audit finding A-C1: the original implementation
# summed each party's own gradient with pure-random peer shares, so the
# masks NEVER cancelled and the "aggregate" was wrong by construction.
# These tests pin the corrected invariant: with full participation, every
# party recovers the exact global sum/mean of all gradients.
# =========================================================================


def _full_mesh_keys(parties: list[str]) -> dict[tuple[str, str], bytes]:
    """One 32-byte symmetric key per unordered party pair."""
    return {pair: os.urandom(32) for pair in combinations(sorted(parties), 2)}


def _make_aggregators(
    parties: list[str],
    round_nonce: bytes = b"",
) -> list[SecureAggregator]:
    """Build one SecureAggregator per party over a full mesh + shared keys."""
    keys = _full_mesh_keys(parties)
    aggs = []
    for pid in parties:
        peers = [q for q in parties if q != pid]
        aggs.append(
            SecureAggregator(
                node_id=pid,
                peer_ids=peers,
                pairwise_keys=keys,
                round_nonce=round_nonce,
            )
        )
    return aggs


class _CaptureWarnings:
    """Collect loguru records at WARNING level inside a with-block."""

    def __init__(self):
        self.messages: list[str] = []

    def __enter__(self):
        self._hid = logger.add(self._sink, level="WARNING")
        return self

    def __exit__(self, *exc):
        logger.remove(self._hid)
        return False

    def _sink(self, message):
        self.messages.append(str(message))


class TestSecureAggregatorExactness:
    """Core property: masks cancel, so the aggregate is the exact sum."""

    @pytest.mark.parametrize("num_parties", [2, 3, 4, 5, 7])
    def test_sum_recovered_exactly_integer_gradients(self, num_parties):
        """Each party recovers the bit-exact SUM of all gradients.

        Odd and even party counts both covered. Small integer values keep
        float32 arithmetic exact, making torch.equal legitimate.
        """
        parties = [f"n{i}" for i in range(num_parties)]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}

        grads = [
            [
                torch.tensor([float(k + j), -float(k * j + 1)])
                for j in range(num_parties)
            ]
            for k in range(num_parties)
        ]

        sent = [aggs[k].split_gradients(grads[k]) for k in range(num_parties)]

        expected_total = [
            torch.zeros_like(grads[0][j]) for j in range(len(grads[0]))
        ]
        for k in range(num_parties):
            for j in range(len(grads[0])):
                expected_total[j] = expected_total[j] + grads[k][j]

        for k, agg in enumerate(aggs):
            received = {q: sent[idx[q]][agg._node_id] for q in agg._peer_ids}
            result = agg.aggregate_received_shares(grads[k], received)
            for j in range(len(expected_total)):
                assert torch.equal(result[j], expected_total[j]), (
                    f"party {k} slot {j}: got {result[j]}, "
                    f"want exact {expected_total[j]}"
                )

    def test_three_peers_hand_computed_values(self):
        """Hand-computed: g_a=[1,2], g_b=[10,20], g_c=[100,200] → [111,222].

        Each of the three parties independently recovers the same exact
        total despite random per-party masks.
        """
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {
            "a": [torch.tensor([1.0, 2.0])],
            "b": [torch.tensor([10.0, 20.0])],
            "c": [torch.tensor([100.0, 200.0])],
        }

        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}
        expected = torch.tensor([111.0, 222.0])

        for p in parties:
            agg = aggs[idx[p]]
            received = {q: sent[q][p] for q in agg._peer_ids}
            result = agg.aggregate_received_shares(grads[p], received)
            assert torch.equal(result[0], expected), (
                f"party {p} did not recover the exact sum"
            )

    def test_mean_of_gradients_bit_exact(self):
        """mean(gradients) recovered exactly when a party divides by N."""
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {
            "a": [torch.tensor([3.0, 6.0])],
            "b": [torch.tensor([6.0, 12.0])],
            "c": [torch.tensor([9.0, 18.0])],
        }
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}

        expected_mean = torch.tensor([6.0, 12.0])
        for p in parties:
            agg = aggs[idx[p]]
            received = {q: sent[q][p] for q in agg._peer_ids}
            total = agg.aggregate_received_shares(grads[p], received)
            assert torch.equal(total[0] / len(parties), expected_mean)

    def test_float_gradients_within_tolerance(self):
        """Arbitrary floats recover the sum within fp rounding tolerance.

        Uses a small mask_bound so mask magnitudes stay far below gradient
        scale: float32 addition of grad + mask rounds at ulp(mask), so
        smaller masks mean tighter reconstruction (documented trade-off —
        see class docstring).
        """
        parties = ["n0", "n1", "n2", "n3"]
        keys = _full_mesh_keys(parties)
        idx = {p: i for i, p in enumerate(parties)}
        aggs = [
            SecureAggregator(
                node_id=p,
                peer_ids=[q for q in parties if q != p],
                pairwise_keys=keys,
                round_nonce=b"float-test",
                mask_bound=256,
            )
            for p in parties
        ]

        base = [0.123456789, -987.654321e-3, 42.000001]
        grads = [[torch.tensor(base) + k * 0.25] for k in range(len(parties))]

        sent = [aggs[k].split_gradients(grads[k]) for k in range(len(parties))]

        expected_total = sum(
            (grads[k][0] for k in range(len(parties))), torch.zeros(3)
        )
        for k, agg in enumerate(aggs):
            received = {q: sent[idx[q]][agg._node_id] for q in agg._peer_ids}
            result = agg.aggregate_received_shares(grads[k], received)
            assert torch.allclose(result[0], expected_total, rtol=1e-3, atol=1e-3)

    def test_multiple_gradient_slots_all_cancel(self):
        """Every tensor slot (per-layer gradient) cancels independently."""
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        vals = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        grads = {p: [torch.tensor([v]) for v in vals[p]] for p in parties}
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}
        expected = [torch.tensor([12.0]), torch.tensor([15.0]), torch.tensor([18.0])]
        for p in parties:
            agg = aggs[idx[p]]
            received = {q: sent[q][p] for q in agg._peer_ids}
            result = agg.aggregate_received_shares(grads[p], received)
            for j in range(3):
                assert torch.equal(result[j], expected[j])

    def test_no_party_view_equals_any_single_gradient(self):
        """A party's output is the SUM, not own-gradient-plus-noise.

        Guards the A-C1 failure mode where the result carried no trace of
        the other parties' actual gradients.
        """
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {
            "a": [torch.tensor([1000.0])],
            "b": [torch.tensor([1.0])],
            "c": [torch.tensor([2.0])],
        }
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}

        agg_a = aggs[0]
        received_a = {q: sent[q]["a"] for q in agg_a._peer_ids}
        out_a = agg_a.aggregate_received_shares(grads["a"], received_a)
        assert not torch.allclose(out_a[0], grads["a"][0])
        assert not torch.allclose(out_a[0], grads["b"][0])
        assert not torch.allclose(out_a[0], grads["c"][0])
        assert torch.equal(out_a[0], torch.tensor([1003.0]))

    def test_masks_vary_by_round_nonce_and_pair(self):
        """Different nonces give different masked vectors; same keys + nonce
        reproduce identical output (deterministic); per-pair masks differ
        even though every peer receives the same masked vector."""
        keys = _full_mesh_keys(["a", "b"])
        outs = []
        for nonce in (b"round-1", b"round-2"):
            agg = SecureAggregator(
                node_id="a",
                peer_ids=["b"],
                pairwise_keys=keys,
                round_nonce=nonce,
            )
            outs.append(agg.split_gradients([torch.tensor([1.0])])["b"][0])
        assert not torch.equal(outs[0], outs[1])

        replay = SecureAggregator(
            node_id="a", peer_ids=["b"], pairwise_keys=keys, round_nonce=b"round-1"
        ).split_gradients([torch.tensor([1.0])])["b"][0]
        assert torch.equal(replay, outs[0])

        # Same masked vector goes to every peer (clones), but the
        # underlying per-pair masks are pairwise-distinct streams.
        tri_keys = _full_mesh_keys(["a", "b", "c"])
        agg_a = SecureAggregator(
            node_id="a",
            peer_ids=["b", "c"],
            pairwise_keys=tri_keys,
            round_nonce=b"r",
        )
        shares = agg_a.split_gradients([torch.tensor([5.0])])
        assert torch.equal(shares["b"][0], shares["c"][0])  # same payload...
        m_ab = agg_a._pairwise_mask("b", torch.zeros(1))
        m_ac = agg_a._pairwise_mask("c", torch.zeros(1))
        assert not torch.equal(m_ab, m_ac)  # ...distinct pair masks


class TestSecureAggregatorDropout:
    """Documented behavior when a participant never delivers."""

    def test_dropout_residual_is_shared_and_warned(self):
        """With a dropped peer, dead-edge mask terms cannot cancel.

        Documented semantics (see SecureAggregator docstring):
          1. The dropped party contributes nothing and leaks nothing.
          2. Every SURVIVING party computes the IDENTICAL residual-shifted
             sum (they all miss the same cancellation terms).
          3. Exactness therefore requires full participation; a WARNING
             naming the dropped peer is emitted.
        Production deployments wanting exactness under dropout need the
        Bonawitz-style Shamir seed-recovery round (not implemented here).
        """
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {
            "a": [torch.tensor([1.0])],
            "b": [torch.tensor([10.0])],
            "c": [torch.tensor([100.0])],  # c drops: sends nothing
        }
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in ("a", "b")}

        with _CaptureWarnings() as cap:
            out_a = aggs[0].aggregate_received_shares(
                grads["a"], {"b": sent["b"]["a"]}
            )
            out_b = aggs[1].aggregate_received_shares(
                grads["b"], {"a": sent["a"]["b"]}
            )

        # Both survivors agree on the same shifted total...
        assert torch.equal(out_a[0], out_b[0])
        # ...which is NOT the participant sum (dead-edge residual) and is
        # not influenced by c's actual gradient value.
        assert not torch.equal(out_a[0], torch.tensor([11.0]))
        assert not torch.allclose(out_a[0], torch.tensor([111.0]))
        # The dropout is called out loudly, not swallowed silently.
        assert any("c" in m for m in cap.messages), cap.messages

    def test_all_participants_present_emits_no_dropout_warning(self):
        parties = ["a", "b"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {"a": [torch.tensor([1.0])], "b": [torch.tensor([2.0])]}
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}

        with _CaptureWarnings() as cap:
            aggs[0].aggregate_received_shares(grads["a"], {"b": sent["b"]["a"]})
        assert not any("dropout" in m.lower() for m in cap.messages)


class TestSecureAggregatorContract:
    """Input validation and API-shape contracts."""

    def test_missing_pairwise_keys_rejected(self):
        """Peers without key material fail at construction (fail closed)."""
        with pytest.raises(ValueError, match="pairwise key"):
            SecureAggregator(node_id="n1", peer_ids=["n2"])

    def test_wrong_key_length_rejected(self):
        keys = {("n1", "n2"): b"short"}
        with pytest.raises(ValueError, match="32 bytes"):
            SecureAggregator(node_id="n1", peer_ids=["n2"], pairwise_keys=keys)

    def test_self_pair_key_rejected(self):
        keys = {("n1", "n1"): os.urandom(32)}
        with pytest.raises(ValueError, match="itself"):
            SecureAggregator(node_id="n1", peer_ids=["n1"], pairwise_keys=keys)

    def test_node_listed_as_own_peer_rejected(self):
        keys = _full_mesh_keys(["n1"])
        with pytest.raises(ValueError, match="contains the node itself"):
            SecureAggregator(node_id="n1", peer_ids=["n1"], pairwise_keys=keys)

    def test_single_party_without_peers_needs_no_keys(self):
        """Degenerate single-party aggregator: aggregate = own gradient."""
        agg = SecureAggregator(node_id="solo", peer_ids=[])
        grads = [torch.tensor([7.0, 8.0])]
        assert agg.split_gradients(grads) == {}
        result = agg.aggregate_received_shares(grads, {})
        assert len(result) == 1
        assert torch.equal(result[0], grads[0])
        # Returned tensors are fresh objects, not aliases of the inputs.
        assert result[0] is not grads[0]

    def test_split_returns_one_entry_per_peer_with_matching_lengths(self):
        agg = SecureAggregator(
            node_id="n1",
            peer_ids=["n2", "n3"],
            pairwise_keys=_full_mesh_keys(["n1", "n2", "n3"]),
        )
        grads = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        shares = agg.split_gradients(grads)
        assert set(shares.keys()) == {"n2", "n3"}
        assert len(shares["n2"]) == 3
        assert len(shares["n3"]) == 3

    def test_positional_construction_still_binds(self):
        """Backward-compatible positional (node_id, peer_ids) signature —
        though keys are mandatory whenever peers exist (tested above)."""
        agg = SecureAggregator("solo", [])
        assert agg._node_id == "solo"
        assert agg._peer_ids == []

    def test_extra_received_slots_beyond_own_gradients_ignored(self):
        parties = ["a", "b"]
        aggs = _make_aggregators(parties)
        grads_a = [torch.tensor([1.0])]
        masked_b = aggs[1].split_gradients([torch.tensor([2.0])])["a"]
        # b accidentally appends a junk extra slot
        received = {"b": masked_b + [torch.tensor([999.0])]}
        result = aggs[0].aggregate_received_shares(grads_a, received)
        assert torch.equal(result[0], torch.tensor([3.0]))

    def test_shape_mismatch_rejected(self):
        parties = ["a", "b"]
        aggs = _make_aggregators(parties)
        grads = [torch.tensor([1.0, 2.0])]
        bad = {"b": [torch.tensor([[1.0, 2.0]])]}  # wrong rank
        with pytest.raises(ValueError, match="shape"):
            aggs[0].aggregate_received_shares(grads, bad)

    def test_dtype_mismatch_rejected(self):
        parties = ["a", "b"]
        aggs = _make_aggregators(parties)
        grads = [torch.tensor([1.0])]
        bad = {"b": [torch.tensor([1.0], dtype=torch.float64)]}
        with pytest.raises(ValueError, match="dtype"):
            aggs[0].aggregate_received_shares(grads, bad)

    def test_integer_dtype_unsupported_with_clear_error(self):
        """Integer tensors need modular arithmetic for exact cancellation;
        not implemented — fail loudly instead of producing wrong results."""
        agg = SecureAggregator(
            node_id="a",
            peer_ids=["b"],
            pairwise_keys=_full_mesh_keys(["a", "b"]),
        )
        with pytest.raises(NotImplementedError, match="[Ii]nteger"):
            agg.split_gradients([torch.tensor([1, 2], dtype=torch.int64)])

    def test_unknown_peer_contribution_skipped_with_warning(self):
        """Data from a non-peer is excluded from the total (which stays
        exact over the configured participants) and warned about."""
        parties = ["a", "b", "c"]
        aggs = _make_aggregators(parties)
        idx = {p: i for i, p in enumerate(parties)}
        grads = {
            "a": [torch.tensor([1.0])],
            "b": [torch.tensor([10.0])],
            "c": [torch.tensor([100.0])],
        }
        sent = {p: aggs[idx[p]].split_gradients(grads[p]) for p in parties}
        stranger = torch.tensor([500.0])  # garbage from non-peer "z"

        with _CaptureWarnings() as cap:
            result = aggs[0].aggregate_received_shares(
                grads["a"],
                {"b": sent["b"]["a"], "c": sent["c"]["a"], "z": [stranger]},
            )
        assert torch.equal(result[0], torch.tensor([111.0]))
        assert any("z" in m for m in cap.messages), cap.messages

    def test_aggregate_does_not_mutate_inputs(self):
        parties = ["a", "b"]
        aggs = _make_aggregators(parties)
        grads = [torch.tensor([10.0])]
        masked_b = aggs[1].split_gradients([torch.tensor([5.0])])["a"]
        received = {"b": [masked_b[0].clone()]}
        grad_before = grads[0].clone()
        received_before = received["b"][0].clone()
        aggs[0].aggregate_received_shares(grads, received)
        assert torch.equal(grads[0], grad_before)
        assert torch.equal(received["b"][0], received_before)

    def test_split_outputs_are_per_peer_clones(self):
        """Split outputs must not alias shared storage (A-C10 aliasing
        pattern): mutating one peer's share never corrupts another's."""
        agg = SecureAggregator(
            node_id="a",
            peer_ids=["b", "c"],
            pairwise_keys=_full_mesh_keys(["a", "b", "c"]),
        )
        shares = agg.split_gradients([torch.tensor([1.0])])
        snapshot_c = shares["c"][0].clone()
        shares["b"][0].add_(100.0)
        assert torch.equal(shares["c"][0], snapshot_c)


# =========================================================================
# Security: authenticated submissions + dataset_size capping
# =========================================================================


class TestFederatedMergeSecurity:
    """P0: untrusted adapter + self-reported dataset_size must not enable
    model poisoning or unfair weight dominance."""

    def _coord_with_round(self, min_nodes=1, **kwargs):
        coord = FederatedMergeCoordinator(min_nodes_per_round=min_nodes, **kwargs)
        coord.register_node("n1", dataset_size=100)
        coord.start_round()
        return coord

    def test_dataset_size_is_capped(self):
        coord = self._coord_with_round()
        assert coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=10**15) is True
        state = coord.get_node_states()["n1"]
        assert state.dataset_size == coord._max_dataset_size

    def test_dataset_size_capped_custom_max(self):
        coord = self._coord_with_round(max_dataset_size=5000)
        coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=10**9)
        assert coord.get_node_states()["n1"].dataset_size == 5000

    def test_capped_weight_does_not_dominate_fedavg(self):
        coord = FederatedMergeCoordinator(min_nodes_per_round=2)
        coord.register_node("n1", dataset_size=100)
        coord.register_node("n2", dataset_size=100)
        coord.start_round()
        coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=10**15)
        coord.submit_node_adapter("n2", "/tmp/n2.pt", 0.1, dataset_size=100)
        round_ = coord.get_current_round()
        assert round_ is not None
        # n1's inflated self-report is capped; it cannot claim 10^15 and win
        # the federated average against n2's honest 100.
        assert round_.node_weights["n1"] == coord._max_dataset_size
        assert round_.node_weights["n2"] == 100

    def test_signature_required_when_key_registered(self):
        key = ed25519.Ed25519PrivateKey.generate()
        coord = self._coord_with_round()
        coord.register_node_public_key("n1", key.public_key())
        # No signature -> rejected (fail closed).
        assert coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=100) is False

    def test_invalid_signature_rejected(self):
        key = ed25519.Ed25519PrivateKey.generate()
        other_key = ed25519.Ed25519PrivateKey.generate()
        coord = self._coord_with_round()
        coord.register_node_public_key("n1", key.public_key())
        round_id = coord.get_current_round().round_id
        wrong = _sign_bytes(
            other_key, f"n1|{round_id}|/tmp/n1.pt|0.1|100"
        )
        assert coord.submit_node_adapter(
            "n1", "/tmp/n1.pt", 0.1, dataset_size=100, signature=wrong
        ) is False

    def test_tampered_values_rejected(self):
        key = ed25519.Ed25519PrivateKey.generate()
        coord = self._coord_with_round()
        coord.register_node_public_key("n1", key.public_key())
        round_id = coord.get_current_round().round_id
        # Signed over dataset_size=100 but submitted with 1000.
        sig = _sign_bytes(key, f"n1|{round_id}|/tmp/n1.pt|0.1|100")
        assert coord.submit_node_adapter(
            "n1", "/tmp/n1.pt", 0.1, dataset_size=1000, signature=sig
        ) is False

    def test_valid_signature_accepted(self):
        key = ed25519.Ed25519PrivateKey.generate()
        coord = self._coord_with_round()
        coord.register_node_public_key("n1", key.public_key())
        round_id = coord.get_current_round().round_id
        sig = _sign_bytes(key, f"n1|{round_id}|/tmp/n1.pt|0.1|100")
        assert coord.submit_node_adapter(
            "n1", "/tmp/n1.pt", 0.1, dataset_size=100, signature=sig
        ) is True

    def test_unkeyed_node_still_accepted(self):
        # Backward compatibility: nodes without a registered key keep working.
        coord = self._coord_with_round()
        assert coord.submit_node_adapter("n1", "/tmp/n1.pt", 0.1, dataset_size=100) is True
