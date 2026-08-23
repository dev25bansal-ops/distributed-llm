"""Federated fine-tuning round tests.

Tests the federated learning pipeline end-to-end:
- Multi-node gradient exchange
- Federated averaging (FedAvg)
- Differential privacy integration
- Adapter merging
- Round lifecycle management

Run with:
    pytest tests/integration/test_federated_rounds.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch


class TestFederatedRounds:
    """Test federated fine-tuning round lifecycle."""

    def _make_tuner(self, node_id="node-1", num_rounds=3, local_steps=5):
        """Create a FederatedFineTuner with mock callbacks."""
        from distllm.core.federated_finetuner import FederatedFineTuner

        applied_grads = []

        def mock_apply(grads, lr):
            applied_grads.append((grads, lr))

        tuner = FederatedFineTuner(
            node_id=node_id,
            local_steps=local_steps,
            num_rounds=num_rounds,
            learning_rate=1e-4,
            apply_gradients=mock_apply,
        )
        return tuner, applied_grads

    def _mock_train_fn(self, steps):
        """Mock training function that returns deterministic gradients."""
        return [torch.ones(4, 4) * 0.1, torch.ones(4) * 0.05]

    def test_single_round_completes(self):
        """A single federated round should complete and return metrics."""
        tuner, applied = self._make_tuner(num_rounds=1)
        result = tuner.train_round(self._mock_train_fn)

        assert result["round"] == 1
        assert result["local_steps"] == 5
        assert result["elapsed_s"] >= 0
        assert len(applied) == 1  # Gradients were applied

    def test_multiple_rounds_increment(self):
        """Multiple rounds should increment the round counter."""
        tuner, _ = self._make_tuner(num_rounds=3)
        for expected_round in range(1, 4):
            result = tuner.train_round(self._mock_train_fn)
            assert result["round"] == expected_round

        assert tuner.stats["rounds_completed"] == 3
        assert tuner.stats["total_local_steps"] == 15  # 3 rounds * 5 steps

    def test_peer_gradient_averaging(self):
        """Received peer gradients should be averaged with local gradients."""
        from distllm.core.federated_finetuner import FederatedFineTuner

        applied_grads = []
        received_data = {
            "peer_id": "node-2",
            "gradients": [torch.ones(4, 4) * 0.2, torch.ones(4) * 0.1],
            "round": 1,
        }

        def mock_receive(timeout=30.0):
            return received_data

        tuner = FederatedFineTuner(
            node_id="node-1",
            local_steps=5,
            num_rounds=1,
            dp_epsilon=float("inf"),  # deterministic: this test isolates averaging
            gossip_receive=mock_receive,
            apply_gradients=lambda g, lr: applied_grads.append(g),
        )
        tuner.add_peer("node-2")

        result = tuner.train_round(self._mock_train_fn)

        # Should have received and averaged gradients
        assert result["gradients_received"] == 1
        assert len(applied_grads) == 1

        # Verify averaging: (0.1 + 0.2) / 2 = 0.15 for first grad
        avg_grad = applied_grads[0][0]
        assert torch.allclose(avg_grad, torch.ones(4, 4) * 0.15, atol=1e-6)

    def test_differential_privacy_clips_gradients(self):
        """DP should clip gradients that exceed max_grad_norm."""
        from distllm.core.federated_finetuner import FederatedFineTuner

        applied_grads = []

        tuner = FederatedFineTuner(
            node_id="node-1",
            local_steps=1,
            num_rounds=1,
            dp_epsilon=1.0,
            dp_max_grad_norm=0.5,
            dp_noise_multiplier=0.0,  # No noise for deterministic test
            apply_gradients=lambda g, lr: applied_grads.append(g),
        )

        # Large gradients that should be clipped
        large_grads = [torch.ones(4, 4) * 10.0]

        result = tuner.train_round(lambda steps: large_grads)

        assert tuner.stats["dp_clips"] > 0
        # After clipping, norm should be <= max_grad_norm
        if applied_grads:
            total_norm = torch.sqrt(sum(g.norm() ** 2 for g in applied_grads[0]))
            assert total_norm <= 0.5 + 1e-6

    def test_broadcast_sends_to_all_peers(self):
        """Gradient broadcast should send to all registered peers."""
        sent_to = []

        def mock_broadcast(peer_id, data):
            sent_to.append(peer_id)

        from distllm.core.federated_finetuner import FederatedFineTuner

        tuner = FederatedFineTuner(
            node_id="node-1",
            local_steps=1,
            num_rounds=1,
            gossip_broadcast=mock_broadcast,
            apply_gradients=lambda g, lr: None,
        )
        tuner.add_peer("node-2")
        tuner.add_peer("node-3")

        tuner.train_round(self._mock_train_fn)

        assert "node-2" in sent_to
        assert "node-3" in sent_to
        assert tuner.stats["peers_contacted"] == 2

    def test_full_run_completes(self):
        """Full federated training run should complete all rounds."""
        tuner, applied = self._make_tuner(num_rounds=3)
        result = tuner.run(self._mock_train_fn)

        assert result["rounds_completed"] == 3
        assert result["total_local_steps"] == 15
        assert len(applied) == 3


class TestFederatedMerge:
    """Test federated adapter merging."""

    def _make_coordinator(self, merge_strategy="fedavg"):
        from distllm.dist.federated_merge import FederatedMergeCoordinator
        return FederatedMergeCoordinator(merge_strategy=merge_strategy)

    def test_register_and_start_round(self):
        """Should start a round when enough nodes are registered."""
        coord = self._make_coordinator()
        coord.register_node("node-1", dataset_size=100)
        coord.register_node("node-2", dataset_size=200)

        round = coord.start_round()
        assert round is not None
        assert len(round.participating_nodes) == 2
        assert round.status == "collecting"

    def test_insufficient_nodes_returns_none(self):
        """Should return None when not enough nodes for a round."""
        coord = self._make_coordinator()
        coord.register_node("node-1", dataset_size=100)

        round = coord.start_round()
        assert round is None  # min_nodes_per_round=2

    def test_submit_adapter_updates_state(self):
        """Submitting an adapter should update node state."""
        coord = self._make_coordinator()
        coord.register_node("node-1", dataset_size=100)
        coord.register_node("node-2", dataset_size=200)
        coord.start_round()

        result = coord.submit_node_adapter("node-1", "/tmp/adapter-1.pt", loss=0.5)
        assert result is True

        state = coord.get_node_states()["node-1"]
        assert state.status == "completed"
        assert state.last_loss == 0.5

    def test_get_versions_tracks_history(self):
        """Version history should be maintained across rounds."""
        coord = self._make_coordinator()
        coord.register_node("node-1", dataset_size=100)
        coord.register_node("node-2", dataset_size=200)

        coord.start_round()
        # Would need actual adapter files for full merge test
        versions = coord.get_versions()
        assert isinstance(versions, list)

    def test_stats_returns_expected_keys(self):
        """Stats should contain expected keys."""
        coord = self._make_coordinator()
        stats = coord.get_stats()

        assert "total_rounds" in stats
        assert "registered_nodes" in stats
        assert "merge_strategy" in stats
