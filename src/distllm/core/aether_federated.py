"""Federated fine-tuning across the cluster -- LoRA adapters, secure gradient
aggregation, and federated training orchestration.

Provides four main classes::

    LoRAAdapterManager   -- create / load / merge / update / unload LoRA adapters
    GradientAggregator   -- FedAvg, secure masked sum, Byzantine-robust trimmed mean
    FederatedTrainer     -- client local training + server aggregation loop
    Aether               -- high-level orchestrator combining all three

Architecture::

    Aether
      |-- LoRAAdapterManager     create/load/merge/unload LoRA adapters
      |-- FederatedTrainer       client training + server aggregation
      |-- GradientAggregator     fedavg / secure / robust aggregation

Torch is optional -- the module is importable without it, and a clear
RuntimeError is raised when any torch-dependent method is called.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

# ---------------------------------------------------------------------------
# Optional torch
# ---------------------------------------------------------------------------
_HAS_TORCH = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for this operation. "
            "Install it with: pip install torch"
        )


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter creation.

    Attributes:
        rank: LoRA rank (r).  Lower means fewer trainable parameters.
        alpha: LoRA scaling factor.  The update is scaled by ``alpha / rank``.
        dropout: Dropout probability applied to the input before the LoRA path.
        target_modules: Module-name substrings to target.  If empty every
            linear (2-D weight) layer in the supplied weights dict is adapted.
        device: Device to place adapter tensors on (``"cpu"``, ``"cuda"``, …).
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)
    device: str = "cpu"


@dataclass
class FederatedConfig:
    """Configuration for federated training rounds.

    Attributes:
        num_rounds: Number of federated training rounds.
        local_epochs: Number of local training epochs per round.
        local_batch_size: Batch size for local training.
        learning_rate: Learning rate for local SGD.
        aggregation: Aggregation strategy -- ``"fedavg"``, ``"secure"``,
            or ``"robust"``.
        robust_f: Number of Byzantine workers to tolerate (for robust
            trimmed-mean aggregation).
        min_clients: Minimum number of clients required per round.
        timeout: Maximum seconds to wait for client gradients per round.
        seed: Random seed for reproducibility.
    """
    num_rounds: int = 10
    local_epochs: int = 1
    local_batch_size: int = 32
    learning_rate: float = 1e-4
    aggregation: str = "fedavg"
    robust_f: int = 0
    min_clients: int = 2
    timeout: float = 300.0
    seed: int = 42


@dataclass
class AetherState:
    """Mutable state tracked by ``Aether`` during fine-tuning.

    Attributes:
        round: Current round index (0-based).
        participants: Number of clients in the last round.
        total_rounds: Total rounds completed.
        loss_history: Per-round loss values.
        convergence_metric: Estimated convergence metric (average gradient
            norm from the last round).
        start_time: ``time.time()`` when fine-tuning started.
    """
    round: int = 0
    participants: int = 0
    total_rounds: int = 0
    loss_history: list[float] = field(default_factory=list)
    convergence_metric: float = float("inf")
    start_time: float = 0.0


# ===================================================================
# LoRAAdapterManager
# ===================================================================


class LoRAAdapterManager:
    """Manages on-device LoRA adapters for a base model.

    Each adapter stores low-rank A (in_features x rank) and B (out_features x rank)
    matrices for every target layer.  The LoRA update is ``delta_W = B @ A^T``,
    scaled by ``alpha / rank``.

    Thread-safe.
    """

    def __init__(self, device: str = "cpu"):
        _require_torch()
        self._device = device
        self._adapters: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        # Per-adapter, per-layer (rank, alpha) so merge() can apply the
        # canonical ``alpha / rank`` scaling without a caller-supplied scale.
        self._adapter_meta: dict[str, dict[str, tuple[int, float]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------
    def create(
        self,
        base_model_weights: dict[str, torch.Tensor],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
        adapter_id: str | None = None,
    ) -> str:
        """Create a new LoRA adapter for the given base model weights.

        Args:
            base_model_weights: Mapping of layer name -> weight tensor.  Every
                2-D weight is treated as a linear layer candidate.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: Dropout probability (applied at forward time by the
                caller; stored for reference).
            target_modules: If provided, only layers whose name contains any
                of these substrings are adapted.  If ``None``, all 2-D weights
                are adapted.
            adapter_id: Optional explicit ID.  Auto-generated if omitted.

        Returns:
            The adapter ID string.

        Raises:
            ValueError: If ``rank`` < 1 or the base model dict is empty.
        """
        _require_torch()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        if not base_model_weights:
            raise ValueError("base_model_weights dict is empty -- nothing to adapt")

        if adapter_id is None:
            adapter_id = f"lora_{secrets.token_hex(8)}"

        # Filter target layers
        layers_to_adapt = self._select_layers(base_model_weights, target_modules)

        adapter_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, weight in layers_to_adapt.items():
            out_features, in_features = weight.shape
            # A: in_features x rank  (random init)
            # B: out_features x rank (zero init)
            a = torch.randn(in_features, rank, device=self._device) * 0.02
            b = torch.zeros(out_features, rank, device=self._device)
            adapter_data[name] = (a, b)

        with self._lock:
            if adapter_id in self._adapters:
                raise ValueError(f"Adapter {adapter_id!r} already exists")
            self._adapters[adapter_id] = adapter_data
            self._adapter_meta[adapter_id] = {
                name: (rank, alpha) for name in adapter_data
            }

        logger.info(
            "Created LoRA adapter {!r} (rank={}, alpha={}, dropout={}, "
            "{} layers)".format(adapter_id, rank, alpha, dropout, len(adapter_data))
        )
        return adapter_id

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(
        self, adapter_id: str
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Load adapter weights.

        Args:
            adapter_id: The adapter identifier returned by ``create``.

        Returns:
            A dict mapping layer name to ``(A_matrix, B_matrix)``.

        Raises:
            KeyError: If the adapter does not exist.
        """
        _require_torch()
        with self._lock:
            if adapter_id not in self._adapters:
                raise KeyError(f"Adapter {adapter_id!r} not found")
            data = self._adapters[adapter_id]
            # Return copies so callers cannot mutate internal state.
            return {
                name: (a.clone(), b.clone()) for name, (a, b) in data.items()
            }

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------
    def merge(
        self,
        adapter_id: str,
        base_model_weights: dict[str, torch.Tensor],
        scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Merge adapter weights into a copy of the base model weights.

        For each adapted layer::

            W_new = W + (alpha / rank) * (B @ A^T)

        Args:
            adapter_id: The adapter to merge.
            base_model_weights: Original base model weights (unchanged).
            scale: Override the scaling factor.  Defaults to ``alpha / rank``
                persisted at creation time.

        Returns:
            A new dict of merged weights.  The original dict is not modified.

        Raises:
            KeyError: If the adapter does not exist.
        """
        _require_torch()
        with self._lock:
            if adapter_id not in self._adapters:
                raise KeyError(f"Adapter {adapter_id!r} not found")
            adapter_data = self._adapters[adapter_id]
        adapter_meta = self._adapter_meta.get(adapter_id, {})

        merged = {}
        for name, weight in base_model_weights.items():
            if name in adapter_data:
                a, b = adapter_data[name]
                delta = b @ a.T  # (out_features, out_features) @ (out_features, in_features)
                effective_scale = scale
                if effective_scale is None:
                    # Canonical LoRA scaling: W_new = W + (alpha / rank) * delta.
                    rank, alpha = adapter_meta.get(name, (0, 1.0))
                    effective_scale = (alpha / rank) if rank else 1.0
                delta = delta * effective_scale
                merged[name] = (weight + delta).detach().clone()
            else:
                merged[name] = weight.detach().clone()
        return merged

    # ------------------------------------------------------------------
    # update  (write trained A/B back into the adapter)
    # ------------------------------------------------------------------
    def update(
        self,
        adapter_id: str,
        data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write trained ``(A, B)`` matrices back into an adapter.

        Used by :meth:`FederatedTrainer.train_lora` to persist the learned
        low-rank parameters so a subsequent :meth:`merge` incorporates them.

        Args:
            adapter_id: The adapter to update.
            data: Mapping of layer name -> ``(A, B)``.  Every name must be a
                layer that exists in the adapter and both matrices must keep
                their original shapes.

        Raises:
            KeyError: If the adapter does not exist.
            ValueError: If a layer is unknown to the adapter or a matrix
                shape does not match the stored one.
        """
        _require_torch()
        with self._lock:
            if adapter_id not in self._adapters:
                raise KeyError(f"Adapter {adapter_id!r} not found")
            stored = self._adapters[adapter_id]
            for name, (a, b) in data.items():
                if name not in stored:
                    raise ValueError(
                        f"Layer {name!r} is not part of adapter {adapter_id!r}"
                    )
                ref_a, ref_b = stored[name]
                if a.shape != ref_a.shape or b.shape != ref_b.shape:
                    raise ValueError(
                        f"Shape mismatch for layer {name!r}: got "
                        f"A={tuple(a.shape)}, B={tuple(b.shape)}, expected "
                        f"A={tuple(ref_a.shape)}, B={tuple(ref_b.shape)}"
                    )
            for name, (a, b) in data.items():
                stored[name] = (a.detach().clone(), b.detach().clone())

    # ------------------------------------------------------------------
    # adapter_scales
    # ------------------------------------------------------------------
    def adapter_scales(self, adapter_id: str) -> dict[str, float]:
        """Return the canonical ``alpha / rank`` scale per adapted layer.

        Args:
            adapter_id: The adapter identifier.

        Returns:
            Mapping of layer name -> ``alpha / rank`` (1.0 fallback when no
            metadata was persisted, mirroring :meth:`merge`).

        Raises:
            KeyError: If the adapter does not exist.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                raise KeyError(f"Adapter {adapter_id!r} not found")
            meta = self._adapter_meta.get(adapter_id, {})
        return {
            name: (alpha / rank) if rank else 1.0
            for name, (rank, alpha) in meta.items()
        }

    # ------------------------------------------------------------------
    # unload
    # ------------------------------------------------------------------
    def unload(self, adapter_id: str) -> bool:
        """Remove an adapter from memory.

        Args:
            adapter_id: The adapter to remove.

        Returns:
            ``True`` if the adapter was found and removed, ``False`` otherwise.
        """
        with self._lock:
            if adapter_id in self._adapters:
                del self._adapters[adapter_id]
                self._adapter_meta.pop(adapter_id, None)
                logger.info("Unloaded LoRA adapter {!r}", adapter_id)
                return True
            return False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def list_adapters(self) -> list[str]:
        """Return a list of known adapter IDs."""
        with self._lock:
            return list(self._adapters.keys())

    def adapter_count(self) -> int:
        """Return the number of managed adapters."""
        with self._lock:
            return len(self._adapters)

    @staticmethod
    def _select_layers(
        weights: dict[str, torch.Tensor],
        target_modules: list[str] | None,
    ) -> dict[str, torch.Tensor]:
        """Return the subset of *weights* that should be adapted."""
        if not target_modules:
            return {n: w for n, w in weights.items() if w.dim() == 2}
        return {
            n: w
            for n, w in weights.items()
            if w.dim() == 2 and any(t in n for t in target_modules)
        }


# ===================================================================
# GradientAggregator
# ===================================================================


class GradientAggregator:
    """Secure and robust gradient aggregation strategies.

    All methods operate on lists of gradient tensors.  Each element in the
    outer list represents one client's contribution; each inner list has
    the same length (one tensor per model layer) and the tensors are
    assumed to be in the same canonical order.
    """

    # ------------------------------------------------------------------
    # FedAvg
    # ------------------------------------------------------------------
    @staticmethod
    def fedavg(gradients: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        """Standard Federated Averaging.

        Computes the element-wise mean across all clients.

        Args:
            gradients: ``gradients[i][j]`` is the j-th layer gradient of the
                i-th client.

        Returns:
            Averaged gradient list (one tensor per layer).

        Raises:
            ValueError: If the input is empty or layer counts differ.
        """
        _require_torch()
        GradientAggregator._validate(gradients)
        num_clients = len(gradients)
        if num_clients == 1:
            return [g.detach().clone() for g in gradients[0]]

        num_layers = len(gradients[0])
        averaged: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            stacked = torch.stack([g[layer_idx] for g in gradients])
            averaged.append(stacked.mean(dim=0))
        return averaged

    # ------------------------------------------------------------------
    # Secure aggregation  (additive masking)
    # ------------------------------------------------------------------
    @staticmethod
    def secure_aggregate(
        masked_gradients: list[list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Secure aggregation via additive masking.

        Assumes each client has masked its gradients so that the random
        masks cancel when summed.  The server simply computes the
        element-wise sum then divides by the number of participants.

        **Client-side masking** (example -- normally done before sending)::

            # Each pair (i, j) agrees on a random seed.
            # Client i adds mask_{i,j}; client j subtracts mask_{i,j}.
            mask = torch.randn_like(gradient) * shared_seed
            masked = gradient + mask   # client i
            # masked = gradient - mask # client j

        Args:
            masked_gradients: One gradient list per client, already masked.

        Returns:
            Averaged gradient list (one tensor per layer).
        """
        _require_torch()
        GradientAggregator._validate(masked_gradients)
        num_clients = len(masked_gradients)
        if num_clients == 0:
            raise ValueError("No client gradients provided")

        num_layers = len(masked_gradients[0])
        result: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            stacked = torch.stack([g[layer_idx] for g in masked_gradients])
            result.append(stacked.sum(dim=0) / num_clients)
        return result

    @staticmethod
    def generate_pairwise_masks(
        num_clients: int,
        ref_gradients: list[torch.Tensor],
        seed: int = 0,
        device: str = "cpu",
    ) -> list[list[list[torch.Tensor]]]:
        """Generate pairwise additive masks that cancel on sum.

        Returns a 3-D structure ``masks[i][j]`` = mask that client *i*
        should **add** for the pair *(i, j)*.  Client *j* will **subtract**
        the same mask, so they cancel in the sum.

        Usage::

            masks = aggregator.generate_pairwise_masks(3, ref_grads)
            # client 0:  masked_0 = [g + masks[0][1] + masks[0][2] for g in grads_0]
            # client 1:  masked_1 = [g - masks[1][0] + masks[1][2] for g in grads_1]
            # client 2:  masked_2 = [g - masks[2][0] - masks[2][1] for g in grads_2]

        Args:
            num_clients: Number of clients.
            ref_gradients: Reference gradient list (used for shapes / dtypes).
            seed: Base random seed.
            device: Device for generated masks.

        Returns:
            ``masks[i][j]`` for *i != j* (entries where *i == j* are empty
            lists and should be ignored).
        """
        _require_torch()
        rng = torch.Generator(device=device).manual_seed(seed)
        masks: list[list[list[torch.Tensor]]] = [
            [[] for _ in range(num_clients)] for _ in range(num_clients)
        ]
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                pair_masks: list[torch.Tensor] = []
                for ref in ref_gradients:
                    m = torch.zeros_like(ref, device=device, dtype=ref.dtype)
                    m.normal_(generator=rng)
                    pair_masks.append(m)
                masks[i][j] = pair_masks
                # Client j subtracts the same mask.
                masks[j][i] = pair_masks
        return masks

    # ------------------------------------------------------------------
    # Robust aggregation  (trimmed mean)
    # ------------------------------------------------------------------
    @staticmethod
    def robust_aggregate(
        gradients: list[list[torch.Tensor]],
        f: int = 0,
    ) -> list[torch.Tensor]:
        """Byzantine-robust aggregation via trimmed mean.

        For each layer, the element-wise values are sorted across clients,
        the *f* smallest and *f* largest are removed, and the remainder is
        averaged.

        Requires ``num_clients >= 2 * f + 1``.

        Args:
            gradients: One gradient list per client.
            f: Number of Byzantine workers to tolerate.

        Returns:
            Robustly averaged gradient list.

        Raises:
            ValueError: If the client count is insufficient for the given *f*.
        """
        _require_torch()
        GradientAggregator._validate(gradients)
        num_clients = len(gradients)
        if num_clients < 2 * f + 1:
            raise ValueError(
                f"Need at least {2 * f + 1} clients for trimmed mean with f={f}, "
                f"got {num_clients}"
            )
        if f == 0:
            return GradientAggregator.fedavg(gradients)

        num_layers = len(gradients[0])
        result: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            stacked = torch.stack([g[layer_idx] for g in gradients])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[f : num_clients - f]
            result.append(trimmed.mean(dim=0))
        return result

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(gradients: list[list[torch.Tensor]]) -> None:
        if not gradients:
            raise ValueError("gradients list is empty")
        ref_len = len(gradients[0])
        if ref_len == 0:
            raise ValueError("First client has zero gradient tensors")
        for idx, g in enumerate(gradients):
            if len(g) != ref_len:
                raise ValueError(
                    f"Client {idx} has {len(g)} layers, expected {ref_len}"
                )


# ===================================================================
# FederatedTrainer
# ===================================================================


class FederatedTrainer:
    """Distributed LoRA training via federated averaging.

    Operates in one of two roles:

    * **Server** -- orchestrates rounds, collects client gradients, aggregates,
      and broadcasts the updated global model.
    * **Client** -- receives global model weights, trains on local data, and
      returns gradient deltas.

    The high-level :meth:`train` method runs the full multi-round protocol
    given a list of data shards and a local training callback.
    """

    def __init__(
        self,
        *,
        role: str = "server",
        aggregator: GradientAggregator | None = None,
        config: FederatedConfig | None = None,
    ):
        _require_torch()
        if role not in ("server", "client"):
            raise ValueError(f"role must be 'server' or 'client', got {role!r}")
        self._role = role
        self._aggregator = aggregator or GradientAggregator()
        self._config = config or FederatedConfig()
        self._global_weights: dict[str, torch.Tensor] | None = None
        self._lock = threading.Lock()
        self._round_stats: list[dict[str, Any]] = []

        # Client registry (server side)
        self._clients: dict[str, Any] = {}

        # Convergence
        self._best_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Client registration (server)
    # ------------------------------------------------------------------
    def register_client(self, client_id: str, metadata: Any = None) -> None:
        """Register a client with the server.

        Args:
            client_id: Unique client identifier.
            metadata: Optional info (device type, dataset size, …).
        """
        with self._lock:
            self._clients[client_id] = metadata

    def unregister_client(self, client_id: str) -> bool:
        """Remove a client from the server registry.

        Returns:
            ``True`` if the client was registered.
        """
        with self._lock:
            return self._clients.pop(client_id, None) is not None

    @property
    def registered_clients(self) -> list[str]:
        """List of registered client IDs."""
        with self._lock:
            return list(self._clients.keys())

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def train(
        self,
        model_weights: dict[str, torch.Tensor],
        dataset: list[Any],
        local_train_fn: Callable[
            [dict[str, torch.Tensor], Any, int, float],
            tuple[list[torch.Tensor], float],
        ],
        num_rounds: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full federated training protocol.

        For each round:

        1. Broadcast the current global weights to every client (by calling
           ``local_train_fn`` with the weights and the client's data shard).
        2. Collect the returned gradient list and loss.
        3. Aggregate all client gradients using the configured strategy.
        4. Apply the aggregated update to the global weights.
        5. Record metrics.

        The ``local_train_fn`` signature is::

            def local_train_fn(
                model_weights: dict[str, torch.Tensor],
                data_shard: Any,
                local_epochs: int,
                learning_rate: float,
            ) -> tuple[list[torch.Tensor], float]:
                \"\"\"Returns (gradients_list, loss).\"\"\"

        The gradient list must be in the same order as
        ``list(model_weights.values())`` and of the same length.

        Args:
            model_weights: Initial base model weights.
            dataset: List of data shards, one per client.  ``len(dataset)``
                is the number of participants.
            local_train_fn: Callable that performs local training and returns
                gradients and loss.
            num_rounds: Override the configured round count.

        Returns:
            Final aggregated model weights.
        """
        _require_torch()
        rounds = num_rounds or self._config.num_rounds
        if len(dataset) < self._config.min_clients:
            raise ValueError(
                f"Need at least {self._config.min_clients} clients, "
                f"got {len(dataset)}"
            )

        # Deep-copy initial weights so we never mutate the caller's dict.
        self._global_weights = {
            k: v.detach().clone() for k, v in model_weights.items()
        }
        self._round_stats.clear()
        self._best_loss = float("inf")

        for round_idx in range(rounds):
            logger.info(
                "Federated round {}/{} ({} clients)",
                round_idx + 1,
                rounds,
                len(dataset),
            )
            round_start = time.time()

            # -- Collect client gradients -----------------------------------
            client_grads: list[list[torch.Tensor]] = []
            round_losses: list[float] = []

            for shard in dataset:
                grads, loss = local_train_fn(
                    self._global_weights,
                    shard,
                    self._config.local_epochs,
                    self._config.learning_rate,
                )
                client_grads.append(grads)
                round_losses.append(loss)

            # -- Aggregate ---------------------------------------------------
            agg_strategy = self._config.aggregation
            if agg_strategy == "fedavg":
                averaged = self._aggregator.fedavg(client_grads)
            elif agg_strategy == "secure":
                averaged = self._aggregator.secure_aggregate(client_grads)
            elif agg_strategy == "robust":
                averaged = self._aggregator.robust_aggregate(
                    client_grads, self._config.robust_f
                )
            else:
                raise ValueError(f"Unknown aggregation strategy: {agg_strategy!r}")

            # -- Apply update ------------------------------------------------
            layer_names = list(self._global_weights.keys())
            for idx, name in enumerate(layer_names):
                self._global_weights[name] = (
                    self._global_weights[name] - self._config.learning_rate * averaged[idx]
                )

            # -- Track metrics -----------------------------------------------
            avg_loss = sum(round_losses) / max(len(round_losses), 1)
            avg_grad_norm = float(
                sum(g.norm().item() for grad_list in client_grads for g in grad_list)
                / max(sum(len(g) for g in client_grads), 1)
            )

            stats = {
                "round": round_idx + 1,
                "participants": len(dataset),
                "avg_loss": avg_loss,
                "avg_grad_norm": avg_grad_norm,
                "elapsed_s": round(time.time() - round_start, 2),
                "aggregation": agg_strategy,
            }
            self._round_stats.append(stats)

            if avg_loss < self._best_loss:
                self._best_loss = avg_loss

            logger.info(
                "  Round {} complete -- loss={:.4f}, grad_norm={:.4f}, {}s",
                round_idx + 1,
                avg_loss,
                avg_grad_norm,
                stats["elapsed_s"],
            )

        return {k: v.detach().clone() for k, v in self._global_weights.items()}

    # ------------------------------------------------------------------
    # LoRA training loop (base weights frozen)
    # ------------------------------------------------------------------
    def train_lora(
        self,
        *,
        adapter_manager: Any,
        adapter_id: str,
        base_model: dict[str, torch.Tensor],
        dataset: list[Any],
        local_train_fn: Callable[
            [dict[str, torch.Tensor], Any, int, float],
            tuple[list[torch.Tensor], float],
        ],
        num_rounds: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run federated training over the LoRA parameters only.

        Unlike :meth:`train`, the base weights stay **frozen**; only the
        adapter's ``A`` / ``B`` matrices are updated (``local_epochs`` /
        ``learning_rate`` come from the trainer config).  Each round:

        1. Compose the effective weight of every adapted layer,
           ``W_eff = W + scale * (B @ A^T)`` with ``scale = alpha / rank``.
        2. Ask every client for gradients w.r.t. the effective weights via
           ``local_train_fn`` (same signature as in :meth:`train`).
        3. Aggregate the full-weight gradients with the configured strategy.
        4. Convert the aggregated gradient ``G`` into LoRA parameter
           gradients and apply SGD::

               grad_B = scale * (G @ A)
               grad_A = scale * (G.T @ B)
               B <- B - lr * grad_B ;  A <- A - lr * grad_A

        Gradients for non-adapted layers are ignored (the base is frozen),
        so communication cost is unchanged even though only the low-rank
        parameters accumulate state.

        Args:
            adapter_manager: The :class:`LoRAAdapterManager` owning the
                adapter; trained ``A`` / ``B`` are written back into it.
            adapter_id: Adapter ID returned by ``create()``.
            base_model: Frozen base model weights.
            dataset: One data shard per client participant.
            local_train_fn: Callable returning ``(gradients, loss)`` for one
                client, gradients aligned with ``list(base_model)`` order.
            num_rounds: Override the configured round count.

        Returns:
            The untouched base weights.  Call ``adapter_manager.merge()`` to
            obtain the adapted model.

        Raises:
            ValueError: If there are too few clients or the adapter adapts
                no layers.
        """
        _require_torch()
        rounds = num_rounds or self._config.num_rounds
        if len(dataset) < self._config.min_clients:
            raise ValueError(
                f"Need at least {self._config.min_clients} clients, "
                f"got {len(dataset)}"
            )

        base = {k: v.detach().clone() for k, v in base_model.items()}
        loaded = adapter_manager.load(adapter_id)
        if not loaded:
            raise ValueError(
                f"Adapter {adapter_id!r} adapts no layers -- nothing to train"
            )
        scales = adapter_manager.adapter_scales(adapter_id)

        # Align adapter matrices with the base weights' device/dtype so the
        # composed effective weights and gradient math stay consistent.
        adapter_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, (a, b) in loaded.items():
            w = base[name]
            adapter_data[name] = (
                a.to(device=w.device, dtype=w.dtype),
                b.to(device=w.device, dtype=w.dtype),
            )

        self._round_stats.clear()
        self._best_loss = float("inf")
        self._global_weights = {k: v.clone() for k, v in base.items()}
        lr = self._config.learning_rate

        for round_idx in range(rounds):
            logger.info(
                "Federated LoRA round {}/{} ({} clients)",
                round_idx + 1,
                rounds,
                len(dataset),
            )
            round_start = time.time()

            # -- Compose effective weights for this round --------------------
            effective: dict[str, torch.Tensor] = {}
            for name, w in base.items():
                if name in adapter_data:
                    a, b = adapter_data[name]
                    effective[name] = w + scales[name] * (b @ a.T)
                else:
                    effective[name] = w

            # -- Collect client gradients ------------------------------------
            client_grads: list[list[torch.Tensor]] = []
            round_losses: list[float] = []
            for shard in dataset:
                grads, loss = local_train_fn(
                    effective,
                    shard,
                    self._config.local_epochs,
                    lr,
                )
                client_grads.append(grads)
                round_losses.append(loss)

            # -- Aggregate full-weight gradients ------------------------------
            agg_strategy = self._config.aggregation
            if agg_strategy == "fedavg":
                averaged = self._aggregator.fedavg(client_grads)
            elif agg_strategy == "secure":
                averaged = self._aggregator.secure_aggregate(client_grads)
            elif agg_strategy == "robust":
                averaged = self._aggregator.robust_aggregate(
                    client_grads, self._config.robust_f
                )
            else:
                raise ValueError(f"Unknown aggregation strategy: {agg_strategy!r}")

            # -- Convert to LoRA parameter gradients and update A/B ----------
            for idx, name in enumerate(base.keys()):
                if name not in adapter_data:
                    continue  # frozen layer -- ignore its gradient
                g = averaged[idx]
                a, b = adapter_data[name]
                s = scales[name]
                grad_a = s * (g.T @ b)
                grad_b = s * (g @ a)
                adapter_data[name] = (a - lr * grad_a, b - lr * grad_b)

            # -- Track metrics ------------------------------------------------
            avg_loss = sum(round_losses) / max(len(round_losses), 1)
            avg_grad_norm = float(
                sum(g.norm().item() for grad_list in client_grads for g in grad_list)
                / max(sum(len(g) for g in client_grads), 1)
            )
            stats = {
                "round": round_idx + 1,
                "participants": len(dataset),
                "avg_loss": avg_loss,
                "avg_grad_norm": avg_grad_norm,
                "elapsed_s": round(time.time() - round_start, 2),
                "aggregation": agg_strategy,
            }
            self._round_stats.append(stats)
            if avg_loss < self._best_loss:
                self._best_loss = avg_loss

            logger.info(
                "  Round {} complete -- loss={:.4f}, grad_norm={:.4f}, {}s",
                round_idx + 1,
                avg_loss,
                avg_grad_norm,
                stats["elapsed_s"],
            )

        # Persist the learned A/B so merge() incorporates the trained delta.
        adapter_manager.update(adapter_id, adapter_data)
        return {k: v.detach().clone() for k, v in base.items()}

    @property
    def global_weights(self) -> dict[str, torch.Tensor] | None:
        """Latest global model weights, or ``None`` if no training has run."""
        if self._global_weights is None:
            return None
        return {k: v.detach().clone() for k, v in self._global_weights.items()}

    @property
    def round_stats(self) -> list[dict[str, Any]]:
        """Per-round statistics from the last :meth:`train` run."""
        return list(self._round_stats)

    @property
    def best_loss(self) -> float:
        """Best (lowest) average loss observed during training."""
        return self._best_loss

    @property
    def config(self) -> FederatedConfig:
        """Return a copy of the current config."""
        return self._config

    # ------------------------------------------------------------------
    # Server-only: aggregate from explicitly submitted gradients
    # ------------------------------------------------------------------
    def aggregate_round(
        self,
        client_gradients: dict[str, list[torch.Tensor]],
        global_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Server-side aggregation of explicitly submitted client gradients.

        This is an alternative to :meth:`train` when clients submit gradients
        out-of-band (e.g. over a network).  The caller must supply the
        current global weights to apply the update to.

        Args:
            client_gradients: Maps client ID to gradient list.
            global_weights: Current global model weights (will be updated).

        Returns:
            Updated global weights.
        """
        _require_torch()
        if self._role != "server":
            raise RuntimeError("Only a server can call aggregate_round()")

        grad_list = list(client_gradients.values())
        agg_strategy = self._config.aggregation
        if agg_strategy == "fedavg":
            averaged = self._aggregator.fedavg(grad_list)
        elif agg_strategy == "secure":
            averaged = self._aggregator.secure_aggregate(grad_list)
        elif agg_strategy == "robust":
            averaged = self._aggregator.robust_aggregate(
                grad_list, self._config.robust_f
            )
        else:
            raise ValueError(f"Unknown aggregation strategy: {agg_strategy!r}")

        layer_names = list(global_weights.keys())
        for idx, name in enumerate(layer_names):
            global_weights[name] = (
                global_weights[name] - self._config.learning_rate * averaged[idx]
            )

        self._global_weights = {
            k: v.detach().clone() for k, v in global_weights.items()
        }
        return self._global_weights


# ===================================================================
# Aether  --  high-level orchestrator
# ===================================================================


class Aether:
    """High-level orchestrator for federated fine-tuning.

    Combines :class:`LoRAAdapterManager`, :class:`FederatedTrainer`, and
    :class:`GradientAggregator` into a single entry point.

    Typical usage::

        aether = Aether()

        result = aether.start_finetuning(
            base_model=model_weights,
            dataset=[shard_1, shard_2, shard_3],
            local_train_fn=my_train_fn,
            num_rounds=10,
        )

        final_weights = aether.get_global_model()
        print(aether.stats())
    """

    def __init__(
        self,
        adapter_manager: LoRAAdapterManager | None = None,
        trainer: FederatedTrainer | None = None,
        aggregator: GradientAggregator | None = None,
        config: FederatedConfig | None = None,
    ):
        self._aggregator = aggregator or GradientAggregator()
        self._adapter_manager = adapter_manager  # lazy-create on demand
        self._config = config or FederatedConfig()
        self._trainer = trainer or FederatedTrainer(
            role="server",
            aggregator=self._aggregator,
            config=self._config,
        )
        self._state = AetherState()
        self._global_model: dict[str, torch.Tensor] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # start_finetuning
    # ------------------------------------------------------------------
    def start_finetuning(
        self,
        base_model: dict[str, torch.Tensor],
        dataset: list[Any],
        local_train_fn: Callable | None = None,
        *,
        num_rounds: int | None = None,
        aggregation: str | None = None,
        lora_config: LoRAConfig | None = None,
    ) -> dict[str, Any]:
        """Initiate federated fine-tuning.

        The workflow is:

        1. Optionally create a LoRA adapter on top of the base model.
        2. Run federated training rounds via the ``FederatedTrainer``.
        3. Merge the trained adapter back into the base weights (if LoRA was
           used).
        4. Return summary statistics.

        Args:
            base_model: Base model weights (``dict[str, torch.Tensor]``).
            dataset: One data shard per client participant.
            local_train_fn: Callable with the same signature as
                :meth:`FederatedTrainer.train` requires.  If omitted, a
                default identity function is used (returns zero gradients),
                which is only useful for testing the aggregation machinery.
            num_rounds: Override the configured number of rounds.
            aggregation: Override the aggregation strategy (``"fedavg"``,
                ``"secure"``, ``"robust"``).
            lora_config: If provided, a LoRA adapter is created on the base
                model and **only the adapter's A/B parameters are trained**
                (the base weights stay frozen).  The trained adapter is
                merged back after training completes, so the returned global
                model reflects the adaptation.

        Returns:
            A dict with keys ``rounds_completed``, ``participants``,
            ``best_loss``, ``final_loss``, ``convergence_metric``,
            ``aggregation``, ``elapsed_s``, and ``used_lora``.

        Raises:
            ValueError: If ``dataset`` is empty.
        """
        _require_torch()
        if not dataset:
            raise ValueError("dataset must contain at least one shard")

        rounds = num_rounds or self._config.num_rounds
        agg_strategy = aggregation or self._config.aggregation
        self._state.start_time = time.time()

        # -- Create LoRA adapter (optional) ----------------------------------
        used_lora = lora_config is not None
        if used_lora:
            if self._adapter_manager is None:
                self._adapter_manager = LoRAAdapterManager(
                    device=lora_config.device  # type: ignore[union-attr]
                )
            adapter_id = self._adapter_manager.create(
                base_model_weights=base_model,
                rank=lora_config.rank,  # type: ignore[union-attr]
                alpha=lora_config.alpha,  # type: ignore[union-attr]
                dropout=lora_config.dropout,  # type: ignore[union-attr]
                target_modules=lora_config.target_modules,  # type: ignore[union-attr]
            )
            logger.info("Created LoRA adapter {!r} for fine-tuning", adapter_id)

        # -- Default training function (zero gradients) ----------------------
        if local_train_fn is None:
            local_train_fn = _zero_gradient_fn

        # -- Override aggregation in config for this run ---------------------
        original_agg = self._config.aggregation
        if aggregation is not None:
            self._config.aggregation = aggregation

        try:
            # -- Run federated training --------------------------------------
            if used_lora and self._adapter_manager is not None:
                # Train only the LoRA parameters; the base stays frozen and
                # the learned delta is merged in below.  (Training the full
                # base weights here would leave the zero-init adapter unused,
                # making lora_config a silent no-op.)
                final_weights = self._trainer.train_lora(
                    adapter_manager=self._adapter_manager,
                    adapter_id=adapter_id,  # type: ignore[possibly-undefined]
                    base_model=base_model,
                    dataset=dataset,
                    local_train_fn=local_train_fn,
                    num_rounds=rounds,
                )
            else:
                final_weights = self._trainer.train(
                    model_weights=base_model,
                    dataset=dataset,
                    local_train_fn=local_train_fn,
                    num_rounds=rounds,
                )
        finally:
            self._config.aggregation = original_agg

        # -- Merge LoRA adapter if one was created ---------------------------
        if used_lora and self._adapter_manager is not None:
            merged = self._adapter_manager.merge(adapter_id, final_weights)  # type: ignore[possibly-undefined]
            self._global_model = merged
            # Unload the adapter -- it is now merged.
            self._adapter_manager.unload(adapter_id)
        else:
            self._global_model = final_weights

        # -- Update state ----------------------------------------------------
        with self._lock:
            self._state.total_rounds = rounds
            self._state.loss_history = [
                s["avg_loss"] for s in self._trainer.round_stats
            ]
            self._state.participants = len(dataset)
            self._state.convergence_metric = (
                self._trainer.round_stats[-1]["avg_grad_norm"]
                if self._trainer.round_stats
                else float("inf")
            )
            self._state.round = rounds

        elapsed = time.time() - self._state.start_time

        results = {
            "rounds_completed": rounds,
            "participants": len(dataset),
            "best_loss": self._trainer.best_loss,
            "final_loss": (
                self._trainer.round_stats[-1]["avg_loss"]
                if self._trainer.round_stats
                else 0.0
            ),
            "convergence_metric": self._state.convergence_metric,
            "aggregation": agg_strategy,
            "elapsed_s": round(elapsed, 2),
            "used_lora": used_lora,
        }

        logger.info(
            "Fine-tuning complete -- {} rounds, {} clients, "
            "best_loss={:.4f}, elapsed={}s",
            rounds,
            len(dataset),
            results["best_loss"],
            results["elapsed_s"],
        )
        return results

    # ------------------------------------------------------------------
    # get_global_model
    # ------------------------------------------------------------------
    def get_global_model(self) -> dict[str, torch.Tensor]:
        """Return the latest aggregated model weights.

        Returns:
            A dict of layer name -> weight tensor.

        Raises:
            RuntimeError: If no fine-tuning run has completed.
        """
        _require_torch()
        if self._global_model is None:
            raise RuntimeError(
                "No global model available -- call start_finetuning() first"
            )
        return {k: v.detach().clone() for k, v in self._global_model.items()}

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return training statistics.

        Returns:
            Dict with keys ``rounds_completed``, ``participants``,
            ``loss_history``, ``best_loss``, ``convergence_metric``,
            ``aggregation``, ``total_elapsed_s`` (0 if not started), and
            ``num_adapters`` (from the adapter manager if present).
        """
        elapsed = (
            time.time() - self._state.start_time if self._state.start_time > 0 else 0.0
        )
        num_adapters = (
            self._adapter_manager.adapter_count()
            if self._adapter_manager is not None
            else 0
        )
        return {
            "rounds_completed": self._state.total_rounds,
            "participants": self._state.participants,
            "loss_history": list(self._state.loss_history),
            "best_loss": self._trainer.best_loss,
            "convergence_metric": self._state.convergence_metric,
            "aggregation": self._config.aggregation,
            "total_elapsed_s": round(elapsed, 2),
            "num_adapters": num_adapters,
        }


# ===================================================================
# Internal helpers
# ===================================================================


def _zero_gradient_fn(
    model_weights: dict[str, torch.Tensor],
    data_shard: Any,
    local_epochs: int = 1,
    learning_rate: float = 1e-4,
) -> tuple[list[torch.Tensor], float]:
    """Default local-training function that returns zero gradients.

    Used when the caller does not provide a real training function
    (e.g. for testing the aggregation machinery).
    """
    _require_torch()
    grads = [torch.zeros_like(w) for w in model_weights.values()]
    return grads, 0.0
