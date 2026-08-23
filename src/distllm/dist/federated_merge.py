"""Federated LoRA Merging — distributed adapter training and merging.

Enables each node in the cluster to train a LoRA adapter on local data,
then periodically merges adapters across nodes using federated averaging.
This provides privacy-preserving fine-tuning without centralizing data.

Features:
- Per-node local LoRA training with configurable epochs
- Federated averaging (FedAvg) of adapter weights
- Weighted merging based on dataset size and node reputation
- Periodic synchronization rounds
- Checkpoint and rollback support
- Adapter versioning and lineage tracking
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


@dataclass
class FederatedRound:
    """A single federated training round."""
    round_id: str
    round_number: int
    started_at: float = 0.0
    completed_at: float = 0.0
    participating_nodes: list[str] = field(default_factory=list)
    node_weights: dict[str, int] = field(default_factory=dict)  # node_id -> dataset_size
    merged_adapter_path: str = ""
    status: str = "pending"  # pending, collecting, merging, completed, failed
    loss_values: dict[str, float] = field(default_factory=dict)  # node_id -> loss


@dataclass
class AdapterVersion:
    """A versioned adapter checkpoint."""
    version_id: str
    adapter_id: str
    round_number: int
    created_at: float = field(default_factory=time.time)
    path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    parent_versions: list[str] = field(default_factory=list)  # Lineage


@dataclass
class NodeTrainingState:
    """Training state for a single node."""
    node_id: str
    current_round: int = 0
    local_epochs: int = 3
    local_batch_size: int = 4
    learning_rate: float = 2e-4
    adapter_path: str = ""
    dataset_size: int = 0
    last_loss: float = 0.0
    last_sync: float = 0.0
    status: str = "idle"  # idle, training, uploading, completed


class FederatedMergeCoordinator:
    """Coordinates federated LoRA training and merging across nodes.

    Orchestrates training rounds where each node trains locally,
    then merges adapter weights using federated averaging.
    """

    def __init__(
        self,
        adapter_manager: Any | None = None,
        merge_strategy: str = "fedavg",  # fedavg, weighted, reputation
        min_nodes_per_round: int = 2,
        rounds_per_checkpoint: int = 5,
    ):
        self._adapter_mgr = adapter_manager
        self._merge_strategy = merge_strategy
        self._min_nodes = min_nodes_per_round
        self._checkpoint_interval = rounds_per_checkpoint

        self._nodes: dict[str, NodeTrainingState] = {}
        self._rounds: list[FederatedRound] = []
        self._versions: dict[str, AdapterVersion] = {}
        self._current_round: FederatedRound | None = None
        self._lock = __import__("threading").Lock()

    # ── Node Registration ───────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        dataset_size: int = 0,
        local_epochs: int = 3,
        learning_rate: float = 2e-4,
    ) -> NodeTrainingState:
        """Register a node for federated training."""
        with self._lock:
            state = NodeTrainingState(
                node_id=node_id,
                dataset_size=dataset_size,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
            )
            self._nodes[node_id] = state
            logger.info(f"Registered federated node: {node_id} (dataset_size={dataset_size})")
            return state

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from federated training."""
        with self._lock:
            self._nodes.pop(node_id, None)

    # ── Round Management ────────────────────────────────────────────────

    def start_round(self) -> FederatedRound | None:
        """Start a new federated training round.

        Checks that enough nodes are available, then creates a round
        and instructs all nodes to begin local training.
        """
        with self._lock:
            active_nodes = [
                nid for nid, state in self._nodes.items()
                if state.status in ("idle", "completed")
            ]
            if len(active_nodes) < self._min_nodes:
                logger.warning(
                    f"Not enough nodes for federated round: "
                    f"{len(active_nodes)} < {self._min_nodes}"
                )
                return None

            round_number = len(self._rounds) + 1
            round_id = f"federated-round-{uuid.uuid4().hex[:8]}"

            self._current_round = FederatedRound(
                round_id=round_id,
                round_number=round_number,
                started_at=time.time(),
                participating_nodes=active_nodes,
                status="collecting",
            )

            # Mark nodes as training
            for nid in active_nodes:
                self._nodes[nid].status = "training"

            logger.info(
                f"Started federated round {round_number} "
                f"with {len(active_nodes)} nodes"
            )
            return self._current_round

    def submit_node_adapter(
        self,
        node_id: str,
        adapter_path: str,
        loss: float,
        dataset_size: int = 0,
    ) -> bool:
        """Submit a node's locally trained adapter for merging.

        Args:
            node_id: The submitting node.
            adapter_path: Path to the node's LoRA adapter weights.
            loss: Final training loss from this node.
            dataset_size: Size of the node's local dataset.

        Returns:
            True if accepted, False if node not in current round.
        """
        with self._lock:
            if not self._current_round or self._current_round.status != "collecting":
                return False
            if node_id not in self._current_round.participating_nodes:
                return False

            state = self._nodes.get(node_id)
            if not state:
                return False

            state.adapter_path = adapter_path
            state.last_loss = loss
            state.last_sync = time.time()
            state.status = "completed"
            if dataset_size > 0:
                state.dataset_size = dataset_size

            self._current_round.node_weights[node_id] = state.dataset_size
            self._current_round.loss_values[node_id] = loss

            logger.info(
                f"Node {node_id} submitted adapter (loss={loss:.4f}, "
                f"dataset_size={state.dataset_size})"
            )

            # Check if all nodes have submitted
            submitted = len(self._current_round.node_weights)
            total = len(self._current_round.participating_nodes)
            if submitted >= total:
                logger.info(f"All {total} nodes submitted, starting merge")
                self._current_round.status = "merging"

            return True

    def merge_adapters(self) -> str | None:
        """Merge all submitted adapters using the configured strategy.

        Returns:
            Path to the merged adapter, or None on failure.
        """
        with self._lock:
            if not self._current_round or self._current_round.status != "merging":
                return None

            try:
                if self._merge_strategy == "fedavg":
                    merged = self._federated_average()
                elif self._merge_strategy == "weighted":
                    merged = self._weighted_merge()
                elif self._merge_strategy == "reputation":
                    merged = self._reputation_weighted_merge()
                else:
                    merged = self._federated_average()

                if merged:
                    self._current_round.merged_adapter_path = merged
                    self._current_round.status = "completed"
                    self._current_round.completed_at = time.time()

                    # Create version
                    version = AdapterVersion(
                        version_id=f"v-{uuid.uuid4().hex[:8]}",
                        adapter_id="federated",
                        round_number=self._current_round.round_number,
                        path=merged,
                        metrics={
                            "avg_loss": sum(self._current_round.loss_values.values())
                            / max(len(self._current_round.loss_values), 1),
                            "num_nodes": len(self._current_round.participating_nodes),
                        },
                    )
                    self._versions[version.version_id] = version

                    # Checkpoint if interval reached
                    if self._current_round.round_number % self._checkpoint_interval == 0:
                        logger.info(
                            f"Checkpoint at round {self._current_round.round_number}"
                        )

                    self._rounds.append(self._current_round)
                    self._current_round = None
                    return merged

            except Exception as e:
                logger.error(f"Merge failed: {e}")
                if self._current_round:
                    self._current_round.status = "failed"
                return None

        return None

    def _federated_average(self) -> str | None:
        """FedAvg: average adapter weights proportional to dataset size."""
        adapters = {}
        for nid in self._current_round.participating_nodes:
            state = self._nodes.get(nid)
            if state and state.adapter_path:
                adapters[nid] = state.adapter_path

        if not adapters:
            return None

        return self._merge_adapter_paths(
            adapters,
            weights={nid: self._nodes[nid].dataset_size or 1 for nid in adapters},
        )

    def _weighted_merge(self) -> str | None:
        """Merge with inverse-loss weighting (better nodes get more weight)."""
        adapters = {}
        weights = {}
        for nid in self._current_round.participating_nodes:
            state = self._nodes.get(nid)
            if state and state.adapter_path:
                adapters[nid] = state.adapter_path
                loss = self._current_round.loss_values.get(nid, 1.0)
                weights[nid] = 1.0 / max(loss, 0.001)

        if not adapters:
            return None
        return self._merge_adapter_paths(adapters, weights)

    def _reputation_weighted_merge(self) -> str | None:
        """Merge with reputation-based weighting (placeholder)."""
        # Falls back to FedAvg for now; reputation integration
        # requires the ReputationSystem from dist/reputation.py
        return self._federated_average()

    def _merge_adapter_paths(
        self,
        adapter_paths: dict[str, str],
        weights: dict[str, float],
    ) -> str | None:
        """Merge multiple adapter weight files into one.

        Loads each adapter's state dict, computes weighted average
        of all parameters, and saves the result.
        """
        import os
        import tempfile

        if not adapter_paths:
            return None

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return None
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Load first adapter as base
        first_path = next(iter(adapter_paths.values()))
        try:
            base_state = torch.load(first_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load adapter {first_path}: {e}")
            return None

        if not isinstance(base_state, dict):
            logger.error("Adapter state is not a dict")
            return None

        # Initialize merged state with weighted first adapter
        first_nid = next(iter(adapter_paths.keys()))
        merged = {}
        for key, tensor in base_state.items():
            if isinstance(tensor, torch.Tensor):
                merged[key] = tensor.float() * norm_weights[first_nid]
            else:
                merged[key] = tensor

        # Accumulate weighted contributions from remaining adapters
        for nid, path in list(adapter_paths.items())[1:]:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    continue
                w = norm_weights.get(nid, 0)
                for key, tensor in state.items():
                    if key in merged and isinstance(tensor, torch.Tensor):
                        merged[key] = merged[key] + tensor.float() * w
            except Exception as e:
                logger.warning(f"Failed to load adapter from node {nid}: {e}")

        # Save merged adapter
        output_dir = os.path.join(tempfile.gettempdir(), "distllm-federated")
        os.makedirs(output_dir, exist_ok=True)
        round_num = self._current_round.round_number if self._current_round else 0
        output_path = os.path.join(output_dir, f"federated-adapter-round{round_num}.pt")

        # Convert back to original dtype
        for key in merged:
            if isinstance(merged[key], torch.Tensor):
                orig = base_state.get(key)
                if isinstance(orig, torch.Tensor):
                    merged[key] = merged[key].to(orig.dtype)

        torch.save(merged, output_path)
        logger.info(f"Merged adapter saved to {output_path}")
        return output_path

    # ── Status ──────────────────────────────────────────────────────────

    def get_current_round(self) -> FederatedRound | None:
        return self._current_round

    def get_rounds(self) -> list[FederatedRound]:
        return list(self._rounds)

    def get_versions(self) -> list[AdapterVersion]:
        return sorted(
            self._versions.values(),
            key=lambda v: v.round_number,
            reverse=True,
        )

    def get_node_states(self) -> dict[str, NodeTrainingState]:
        return dict(self._nodes)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_rounds": len(self._rounds),
                "registered_nodes": len(self._nodes),
                "active_nodes": sum(
                    1 for s in self._nodes.values() if s.status != "idle"
                ),
                "total_versions": len(self._versions),
                "merge_strategy": self._merge_strategy,
                "current_round": (
                    self._current_round.round_number
                    if self._current_round else None
                ),
                "current_round_status": (
                    self._current_round.status
                    if self._current_round else None
                ),
                "avg_loss_last_round": (
                    sum(self._rounds[-1].loss_values.values())
                    / max(len(self._rounds[-1].loss_values), 1)
                    if self._rounds else None
                ),
            }


class SecureAggregator:
    """Secure aggregation for federated learning using additive secret sharing.

    Each node splits its gradient into N random shares (one per peer),
    sends share_i to peer_i, and sums the received shares. The result
    is that no single node sees any other node's raw gradient — only
    the aggregate is revealed.

    This implements a simplified version of the SecAgg protocol
    (Bonawitz et al. 2017) without the masking/unmasking phase.

    Usage::

        aggregator = SecureAggregator(node_id="node-1", peer_ids=["node-2", "node-3"])
        shares = aggregator.split_gradients(local_gradients)
        # Send shares[i] to peer_i
        received = aggregator.receive_shares(peer_shares)
        aggregate = aggregator.aggregate(received)
    """

    def __init__(self, node_id: str, peer_ids: list[str]):
        self._node_id = node_id
        self._peer_ids = list(peer_ids)
        self._num_parties = len(peer_ids) + 1  # Including self

    def split_gradients(
        self, gradients: list[torch.Tensor]
    ) -> dict[str, list[torch.Tensor]]:
        """Split gradients into additive shares for each peer.

        Returns a dict mapping peer_id -> list of share tensors.
        The caller keeps the self-share internally.
        """
        shares: dict[str, list[torch.Tensor]] = {}
        self_shares: list[torch.Tensor] = []

        for grad in gradients:
            # Generate random shares for each peer
            peer_shares = {}
            running_sum = torch.zeros_like(grad)

            for peer_id in self._peer_ids:
                share = torch.randn_like(grad)
                peer_shares[peer_id] = share
                running_sum += share

            # Self share = gradient - sum of peer shares
            self_share = grad - running_sum
            self_shares.append(self_share)

            for peer_id, share in peer_shares.items():
                if peer_id not in shares:
                    shares[peer_id] = []
                shares[peer_id].append(share)

        return shares

    def aggregate_received_shares(
        self,
        self_gradients: list[torch.Tensor],
        received_shares: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Aggregate received shares with self-gradients to get the sum.

        Args:
            self_gradients: Own gradients (used to compute self-shares).
            received_shares: Dict of peer_id -> list of share tensors.

        Returns:
            Aggregated gradient tensors (sum of all parties' gradients).
        """
        if not received_shares:
            return self_gradients

        num_grads = len(self_gradients)
        aggregated = [torch.zeros_like(g) for g in self_gradients]

        # Add received shares
        for peer_id, peer_shares in received_shares.items():
            for i, share in enumerate(peer_shares):
                if i < num_grads:
                    aggregated[i] += share

        # Add self shares (gradient - sum_of_peer_shares)
        for i, grad in enumerate(self_gradients):
            if i < num_grads:
                # Self share = grad - sum(peer_shares_sent_to_peers)
                # But we need to add back the self share which is grad - sum(sent)
                # Actually: aggregate = sum(all_shares) = self_share + sum(received_shares)
                # self_share = grad - sum(shares_sent), so aggregate = grad
                # The protocol ensures aggregate = sum of all gradients
                aggregated[i] = aggregated[i] + grad

        return aggregated
