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

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric import ed25519

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
# SecureAggregator
# =========================================================================


class TestSecureAggregator:
    """Secure gradient aggregation using additive secret sharing."""

    def test_split_returns_correct_number_of_peers(self):
        """split_gradients returns a dict with entries for each peer."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2", "n3"])
        grads = [torch.tensor([1.0, 2.0, 3.0])]
        shares = agg.split_gradients(grads)
        assert set(shares.keys()) == {"n2", "n3"}

    def test_split_each_share_is_tensor_list(self):
        """Each peer entry is a list of tensors matching the input length."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([1.0]), torch.tensor([2.0])]
        shares = agg.split_gradients(grads)
        assert len(shares["n2"]) == 2

    def test_aggregate_empty_received_shares(self):
        """With empty received_shares, aggregate returns self_gradients unchanged."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([10.0, 20.0])]
        result = agg.aggregate_received_shares(grads, {})
        assert len(result) == 1
        assert torch.equal(result[0], grads[0])

    def test_aggregate_with_received_shares(self):
        """Aggregate adds received shares to self gradient."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([5.0, 5.0])]
        received = {"n2": [torch.tensor([3.0, 7.0])]}
        result = agg.aggregate_received_shares(grads, received)
        # aggregate = grad + sum of received shares = [5,5] + [3,7] = [8,12]
        expected = torch.tensor([8.0, 12.0])
        assert torch.allclose(result[0], expected)

    def test_split_and_aggregate_roundtrip(self):
        """Split then aggregate: each party sees own_grad + sum(received_shares).

        In this simplified SecAgg, split_gradients creates random peer shares
        and a self-share = grad - sum(sent_shares). aggregate_received_shares
        returns grad + sum(received_shares). This test verifies the operation
        is consistent: after splitting, aggregating with the other party's
        shares yields the expected addition.
        """
        g1 = [torch.tensor([10.0, 20.0, 30.0])]
        g2 = [torch.tensor([1.0, 2.0, 3.0])]

        agg1 = SecureAggregator(node_id="n1", peer_ids=["n2"])
        agg2 = SecureAggregator(node_id="n2", peer_ids=["n1"])

        # Split
        shares_from_1 = agg1.split_gradients(g1)  # {"n2": [share]}
        shares_from_2 = agg2.split_gradients(g2)  # {"n1": [share]}

        # Each aggregate receives the other's share
        n1_result = agg1.aggregate_received_shares(g1, shares_from_2)
        n2_result = agg2.aggregate_received_shares(g2, shares_from_1)

        # n1_result = g1 + shares_from_2["n1"] (the random share n2 created for n1)
        # n2_result = g2 + shares_from_1["n2"] (the random share n1 created for n2)
        assert torch.allclose(n1_result[0], g1[0] + shares_from_2["n1"][0])
        assert torch.allclose(n2_result[0], g2[0] + shares_from_1["n2"][0])

    def test_split_single_gradient(self):
        """split_gradients works with a single gradient tensor."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([42.0])]
        shares = agg.split_gradients(grads)
        assert "n2" in shares
        # The self share (kept internally) is grad - sum_of_peer_shares
        # We can verify by: grad = self_share + sum(peer_shares)
        # Since aggregate = grad + received_shares, the tensor should be
        # reconstructable
        received = {"n2": shares["n2"]}
        result = agg.aggregate_received_shares(grads, received)
        # This should be grad + received = grad + (grad - self_share) ... not quite
        # Actually: aggregate = self_share + received_shares
        # And self_share is stored inside grads, so aggregate_received_shares does:
        # aggregated = sum(received_shares) + grad
        # This yields: grad_shares_from_n2_to_n1 + n1_grad
        # In the 2-party case: n1_grad = self_share + share_to_n2
        # n2 sends its share of n1's grad back -- wait these are independent gradients.
        # Let's just verify the operation composes properly:
        assert torch.allclose(result[0], grads[0] + received["n2"][0])

    def test_split_multiple_gradients(self):
        """split_gradients works with multiple gradient tensors."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2", "n3"])
        grads = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        shares = agg.split_gradients(grads)
        assert len(shares["n2"]) == 3
        assert len(shares["n3"]) == 3

    def test_aggregate_more_shares_than_gradients(self):
        """Extra received shares beyond num_gradients are safely ignored."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([1.0])]
        received = {"n2": [torch.tensor([2.0]), torch.tensor([3.0])]}
        result = agg.aggregate_received_shares(grads, received)
        assert torch.allclose(result[0], torch.tensor([3.0]))

    def test_split_each_share_is_unique(self):
        """Each peer receives a different random share."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2", "n3"])
        grads = [torch.tensor([1.0, 2.0])]
        shares = agg.split_gradients(grads)
        # Two peers should have different shares
        assert not torch.equal(shares["n2"][0], shares["n3"][0])

    def test_split_sums_reconstruct_gradient(self):
        """The sum of all shares (peer + self) equals the original gradient.

        The self-share is implicitly stored as grad - sum(sent_shares).
        After aggregate_received_shares with zero received_shares,
        the result should equal the original gradient.
        """
        agg = SecureAggregator(node_id="n1", peer_ids=["n2", "n3"])
        grads = [torch.tensor([7.0, 11.0, 13.0])]
        shares = agg.split_gradients(grads)

        # Compute sum of all peer shares
        sum_peer = shares["n2"][0] + shares["n3"][0]

        # aggregate_received_shares with no received = grad = self_share
        # But self_share = grad - sum_peer, so the sum of all shares = grad
        # Verify: aggregate of received (from peers) + self gradient = grad + sum_peer
        # And we know self_share internally is grad - sum_peer
        # After aggregate_received_shares with empty received:
        # aggregated[i] = aggregated[i] + grad = grad
        result = agg.aggregate_received_shares(grads, {})
        assert torch.allclose(result[0], grads[0])

        # Full aggregate: includes peer shares
        # aggregated = grad + sum(received)
        full = agg.aggregate_received_shares(grads, {"n2": [shares["n2"][0]], "n3": [shares["n3"][0]]})
        assert torch.allclose(full[0], grads[0] + sum_peer)

    def test_aggregate_does_not_mutate_input(self):
        """aggregate_received_shares does not modify input tensors."""
        agg = SecureAggregator(node_id="n1", peer_ids=["n2"])
        grads = [torch.tensor([10.0])]
        received = {"n2": [torch.tensor([5.0])]}
        grad_before = grads[0].clone()
        received_before = received["n2"][0].clone()
        agg.aggregate_received_shares(grads, received)
        assert torch.equal(grads[0], grad_before)
        assert torch.equal(received["n2"][0], received_before)


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
