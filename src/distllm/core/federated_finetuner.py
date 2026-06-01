"""Federated fine-tuning over P2P — distributed LoRA training across nodes.

Each node trains on its private data.  Only gradient updates (not raw data
or model weights) are exchanged over the P2P gossip protocol.

Supported algorithms:
- **FedAvg**: Standard federated averaging (McMahan et al. 2017)
- **FedProx**: Proximal term regularization for heterogeneous data (Li et al. 2020)

Architecture::

    Node A (data_a) ── LoRA grads ──┐
    Node B (data_b) ── LoRA grads ──┼── P2P All-Reduce ──▸ Unified LoRA weights
    Node C (data_c) ── LoRA grads ──┘

Privacy:
- Gradient clipping bounds sensitivity
- Gaussian noise addition for differential privacy
- Secure aggregation via additive secret sharing (see federated_merge.py)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import torch
from loguru import logger


class FederatedFineTuner:
    """Coordinates distributed LoRA fine-tuning across P2P nodes.

    Each node:
      1. Loads the base model + LoRA adapters
      2. Trains on local data for ``local_steps`` steps
      3. Broadcasts LoRA gradient updates to peers via the gossip protocol
      4. Averages received gradients with local gradients
      5. Applies the merged gradients to the local LoRA weights
      6. Repeats for ``num_rounds`` rounds

    Args:
        node_id: Unique identifier for this node.
        lora_adapter: Callable returning LoRA parameters.
        apply_gradients: Callable to apply merged gradients to LoRA weights.
        gossip_broadcast: Callable to send data to peers.
        gossip_receive: Callable to receive data from peers.
        local_steps: Training steps per round.
        num_rounds: Total federated rounds.
        learning_rate: SGD learning rate for gradient updates.
    """

    def __init__(
        self,
        node_id: str,
        lora_adapter: Any = None,
        apply_gradients: Callable | None = None,
        gossip_broadcast: Callable | None = None,
        gossip_receive: Callable | None = None,
        local_steps: int = 100,
        num_rounds: int = 10,
        learning_rate: float = 1e-4,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_max_grad_norm: float = 1.0,
        dp_noise_multiplier: float = 0.0,
        algorithm: str = "fedavg",
        fedprox_mu: float = 0.0,
        global_model_params: list[torch.Tensor] | None = None,
    ):
        self._node_id = node_id
        self._lora_adapter = lora_adapter
        self._apply_gradients = apply_gradients
        self._broadcast = gossip_broadcast
        self._receive = gossip_receive
        self._local_steps = local_steps
        self._num_rounds = num_rounds
        self._lr = learning_rate

        self._round = 0
        self._lock = threading.Lock()
        self._peers: set[str] = set()
        self._received_grads: dict[str, list[torch.Tensor]] = {}

        # Algorithm selection
        self._algorithm = algorithm.lower()
        self._fedprox_mu = fedprox_mu  # FedProx proximal term coefficient
        self._global_model_params = global_model_params  # Global model for FedProx

        # Differential privacy settings
        self._dp_epsilon = dp_epsilon
        self._dp_delta = dp_delta
        self._dp_max_grad_norm = dp_max_grad_norm
        self._dp_noise_multiplier = dp_noise_multiplier

        self._stats = {
            "rounds_completed": 0,
            "total_local_steps": 0,
            "peers_contacted": 0,
            "dp_clips": 0,
            "dp_noise_added": False,
            "algorithm": self._algorithm,
            "fedprox_proximal_terms": 0,
        }

    def _clip_gradients(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Clip gradients to max_grad_norm for differential privacy.

        Computes the total gradient norm across all parameters and clips
        if it exceeds max_grad_norm. This bounds sensitivity for noise addition.
        """
        total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
        if total_norm > self._dp_max_grad_norm:
            clip_factor = self._dp_max_grad_norm / total_norm
            self._stats["dp_clips"] += 1
            return [g * clip_factor for g in grads]
        return grads

    def _add_dp_noise(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add calibrated Gaussian noise for differential privacy.

        Noise scale: sigma = max_grad_norm * noise_multiplier
        where noise_multiplier = sqrt(2 * ln(1.25/delta)) / epsilon
        """
        if self._dp_noise_multiplier > 0:
            sigma = self._dp_max_grad_norm * self._dp_noise_multiplier
        else:
            import math
            sigma = self._dp_max_grad_norm * math.sqrt(
                2 * math.log(1.25 / self._dp_delta)
            ) / self._dp_epsilon

        self._stats["dp_noise_added"] = True
        return [g + torch.randn_like(g) * sigma for g in grads]

    def add_peer(self, peer_id: str) -> None:
        self._peers.add(peer_id)

    def remove_peer(self, peer_id: str) -> None:
        self._peers.discard(peer_id)

    def train_round(self, local_train_fn: Callable[[int], list[torch.Tensor]]) -> dict[str, Any]:
        """Execute one federated round.

        Args:
            local_train_fn: Callable that runs local training and returns
                a list of LoRA gradient tensors.  Receives the number of
                local steps as argument.

        Returns:
            Dict with round metrics.
        """
        self._round += 1
        round_start = time.time()

        # Step 1: Local training
        local_grads = local_train_fn(self._local_steps)
        self._stats["total_local_steps"] += self._local_steps

        # Step 1.5: Apply differential privacy (clip + noise)
        if self._dp_epsilon < float("inf"):
            local_grads = self._clip_gradients(local_grads)
            local_grads = self._add_dp_noise(local_grads)

        # Step 2: Broadcast gradients to peers
        if self._broadcast is not None:
            for peer_id in self._peers:
                try:
                    self._broadcast(peer_id, {"gradients": local_grads, "round": self._round})
                    self._stats["peers_contacted"] += 1
                except Exception as e:
                    logger.warning(f"Failed to broadcast to {peer_id}: {e}")

        # Step 3: Receive peer gradients (with timeout)
        if self._receive is not None:
            try:
                peer_data = self._receive(timeout=30.0)
                if peer_data and "gradients" in peer_data:
                    peer_grads = peer_data["gradients"]
                    self._received_grads[peer_data.get("peer_id", "unknown")] = peer_grads
            except Exception:
                pass

        # Step 4: Average gradients
        merged = self._average_gradients(local_grads)

        # Step 5: Apply merged gradients
        if self._apply_gradients is not None:
            self._apply_gradients(merged, self._lr)

        self._stats["rounds_completed"] = self._round

        elapsed = time.time() - round_start
        return {
            "round": self._round,
            "local_steps": self._local_steps,
            "peers": len(self._peers),
            "gradients_received": len(self._received_grads),
            "elapsed_s": round(elapsed, 2),
        }

    def _average_gradients(self, local_grads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Average local gradients with all received peer gradients.

        Uses the configured algorithm:
        - FedAvg: Simple weighted average
        - FedProx: Average with proximal term regularization
        """
        all_grads = [local_grads]
        for peer_id, peer_grads in self._received_grads.items():
            if len(peer_grads) == len(local_grads):
                all_grads.append(peer_grads)

        num_sources = len(all_grads)
        if num_sources <= 1:
            return local_grads

        # FedProx: add proximal term to gradients
        if self._algorithm == "fedprox" and self._global_model_params is not None:
            local_grads = self._apply_fedprox_term(local_grads)
            all_grads[0] = local_grads

        # Weighted average (FedAvg or FedProx)
        averaged = []
        for i in range(len(local_grads)):
            grad_sum = sum(g[i] for g in all_grads)
            averaged.append(grad_sum / num_sources)

        return averaged

    def _apply_fedprox_term(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply FedProx proximal term to gradients.

        FedProx adds a regularization term: mu * (w - w_global)
        This penalizes local model for drifting too far from the global model,
        which helps with heterogeneous data distributions.

        Args:
            grads: Local gradients.

        Returns:
            Gradients with proximal term added.
        """
        if self._global_model_params is None or self._fedprox_mu <= 0:
            return grads

        proximal_grads = []
        for i, grad in enumerate(grads):
            if i < len(self._global_model_params):
                global_param = self._global_model_params[i]
                # Proximal term: mu * (w_local - w_global)
                # This is added to the gradient to penalize divergence
                proximal = self._fedprox_mu * (grad - global_param.detach())
                proximal_grads.append(grad + proximal)
                self._stats["fedprox_proximal_terms"] += 1
            else:
                proximal_grads.append(grad)

        return proximal_grads

    def set_global_model(self, params: list[torch.Tensor]) -> None:
        """Update the global model parameters for FedProx.

        Called after each round with the aggregated global model.
        """
        self._global_model_params = [p.detach().clone() for p in params]

    def run(self, local_train_fn: Callable[[int], list[torch.Tensor]]) -> dict[str, Any]:
        """Run the full federated training loop.

        Args:
            local_train_fn: Callable that runs local training and returns
                a list of LoRA gradient tensors per round.

        Returns:
            Final training stats.
        """
        for round_idx in range(self._num_rounds):
            logger.info(f"Federated round {round_idx + 1}/{self._num_rounds}")
            metrics = self.train_round(local_train_fn)
            logger.info(f"  Round complete: {metrics['elapsed_s']}s, "
                         f"{metrics['gradients_received']} peer gradients")

        return dict(self._stats)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)
