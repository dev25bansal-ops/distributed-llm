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

Privacy (honest contract — audit A-C2):
- Gradient clipping bounds gradient sensitivity.  This is the DEFAULT
  (``dp_mode="clip_only"``) and clipping alone is **not** differential
  privacy: without calibrated noise no (epsilon, delta) guarantee holds,
  so a WARNING is logged whenever this mode runs with a finite epsilon.
- Real DP requires opting in with ``dp_mode="enabled"``: every round adds
  Gaussian noise calibrated to that round's share of the (epsilon, delta)
  budget, spend accumulates in ``stats["dp_cumulative_epsilon"]``, and
  ``run()`` stops when the budget is exhausted.
- Secure aggregation via additive secret sharing (see federated_merge.py)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import torch
from loguru import logger


class DPBudgetExhausted(RuntimeError):
    """Raised when another round would exceed the stated (epsilon, delta) budget.

    Fail-closed by design: broadcasting gradients whose noise no longer
    corresponds to an honest privacy claim is worse than stopping.
    """


def _load_rdp_accounting() -> Any | None:
    """Return a fresh RDPAccounting instance, or None if unavailable.

    Tries the normal dotted import first; falls back to loading
    ``dp_inference/accounting.py`` directly from disk.  The fallback keeps
    privacy accounting working in test/embedded contexts where the package
    ``__init__`` chain is stubbed or partially initialized.
    """
    try:
        from distllm.core.dp_inference.accounting import RDPAccounting

        return RDPAccounting()
    except Exception:
        pass
    try:
        import importlib.util
        import sys
        from pathlib import Path

        accounting_path = Path(__file__).resolve().parent / "dp_inference" / "accounting.py"
        dotted = "distllm.core.dp_inference.accounting"
        mod = sys.modules.get(dotted)
        spec = None
        if mod is None or getattr(mod, "RDPAccounting", None) is None:
            spec = importlib.util.spec_from_file_location(dotted, str(accounting_path))
            if spec is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[dotted] = mod
                spec.loader.exec_module(mod)
        cls = getattr(mod, "RDPAccounting", None)
        return cls() if cls is not None else None
    except Exception:
        return None


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
        dp_mode: ``"clip_only"`` (default) or ``"enabled"``.  Clipping-only
            bounds gradient sensitivity but provides **no** (epsilon, delta)
            differential-privacy guarantee; a WARNING is emitted whenever it
            runs while a finite epsilon is configured.  ``"enabled"`` adds
            Gaussian noise calibrated to the per-round privacy budget and
            tracks cumulative spend in ``stats["dp_cumulative_epsilon"]``.
        dp_epsilon: Total (epsilon, delta) budget.  In ``"enabled"`` mode the
            budget is divided across ``num_rounds`` so per-round sigma grows
            with round count (composition-safe by construction).  Set to
            ``float("inf")`` in either mode to disable DP entirely.
        dp_delta: Delta target of the DP guarantee; must lie in (0, 1).
        dp_max_grad_norm: L2 clip bound; also scales the noise
            (sigma = max_grad_norm * noise_multiplier).
        dp_noise_multiplier: Explicit noise multiplier (> 0).  When left at
            0.0, sigma is auto-derived from the per-round share of
            (dp_epsilon, dp_delta).
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
        dp_mode: str = "clip_only",
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
        # Federated round each peer's stored gradients belong to (audit F-018).
        # Entries absent from this map (e.g. injected directly by callers or
        # sent by legacy peers without a ``round`` field) are treated as
        # belonging to the current round.
        self._received_rounds: dict[str, int] = {}

        # Algorithm selection
        self._algorithm = algorithm.lower()
        self._fedprox_mu = fedprox_mu  # FedProx proximal term coefficient
        self._global_model_params = global_model_params  # Global model for FedProx

        # Differential privacy settings (audit A-C2: honest contract).
        mode = dp_mode.lower()
        if mode not in ("clip_only", "enabled"):
            raise ValueError(
                f"dp_mode must be 'clip_only' or 'enabled', got {dp_mode!r}"
            )
        if not (0.0 < dp_delta < 1.0):
            raise ValueError(f"dp_delta must be in (0, 1), got {dp_delta}")
        if math.isfinite(dp_epsilon) and dp_epsilon <= 0:
            raise ValueError(
                f"dp_epsilon must be > 0 (or float('inf') to disable DP), "
                f"got {dp_epsilon}"
            )
        if dp_noise_multiplier < 0:
            raise ValueError(
                f"dp_noise_multiplier must be >= 0 (0 auto-derives sigma "
                f"from dp_epsilon/dp_delta), got {dp_noise_multiplier}"
            )
        # An explicitly positive multiplier is itself an opt-in to noise;
        # honor it (with a note) instead of silently discarding the setting,
        # so configurations written against the pre-A-C2 docs keep adding
        # calibrated noise.  The DEFAULT configuration (no mode, no
        # multiplier) stays clip-only -- that is the dishonest case this
        # fix targets.
        if mode == "clip_only" and dp_noise_multiplier > 0:
            logger.info(
                f"[{self._node_id}] dp_noise_multiplier="
                f"{dp_noise_multiplier} given without dp_mode='enabled'; "
                f"resolving dp_mode to 'enabled'."
            )
            mode = "enabled"
        self._dp_mode = mode
        self._dp_epsilon = dp_epsilon
        self._dp_delta = dp_delta
        self._dp_max_grad_norm = dp_max_grad_norm
        self._dp_noise_multiplier = dp_noise_multiplier
        # Per-round budget share used to calibrate sigma when the multiplier
        # is not given explicitly.  Dividing the budget across rounds makes
        # the loop composition-safe by construction: more rounds means a
        # larger sigma per round, and run() never exceeds the total.
        rounds_for_budget = max(1, num_rounds)
        self._per_round_epsilon = (
            dp_epsilon / rounds_for_budget
            if math.isfinite(dp_epsilon)
            else float("inf")
        )

        self._stats = {
            "rounds_completed": 0,
            "total_local_steps": 0,
            "peers_contacted": 0,
            "dp_clips": 0,
            "dp_noise_added": False,
            "algorithm": self._algorithm,
            "fedprox_proximal_terms": 0,
            "dp_mode": self._dp_mode,
            "dp_sigma": None,
            "dp_cumulative_epsilon": 0.0,
        }

        # Fail loud when the configuration implies a privacy guarantee it
        # does not deliver (audit A-C2).  A WARNING (not an error) keeps
        # existing convergence behavior working while making the gap visible.
        if (
            self._dp_mode == "clip_only"
            and math.isfinite(self._dp_epsilon)
            and self._dp_epsilon > 0
        ):
            logger.warning(
                f"[{self._node_id}] DP NOT ENABLED: dp_mode='clip_only' adds "
                f"gradient clipping only -- NO calibrated noise is added, so "
                f"NO (epsilon={self._dp_epsilon}, delta={self._dp_delta}) "
                f"guarantee holds. Pass dp_mode='enabled' for real "
                f"differential privacy, or dp_epsilon=float('inf') to "
                f"declare non-private training and silence this warning."
            )

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

        Only runs in ``dp_mode="enabled"``; in ``"clip_only"`` mode gradients
        pass through untouched (that mode provides NO privacy guarantee).

        Noise scale ``sigma = max_grad_norm * noise_multiplier`` when an
        explicit positive multiplier was configured; otherwise sigma is
        derived from this round's share of the (epsilon, delta) budget::

            eps_round = dp_epsilon / num_rounds
            sigma = max_grad_norm * sqrt(2 * ln(1.25 / delta)) / eps_round

        Dividing the budget across rounds makes the loop composition-safe by
        construction: more rounds means larger sigma per round.
        """
        if self._dp_mode != "enabled" or not math.isfinite(self._dp_epsilon):
            self._stats["dp_noise_added"] = False
            return grads

        if self._dp_noise_multiplier > 0:
            sigma = self._dp_max_grad_norm * self._dp_noise_multiplier
            # The true (epsilon, delta) cost of this explicitly-sized
            # mechanism is measured with the RDP accountant below rather
            # than assumed.
            round_cost: float | None = None
        else:
            eps_round = max(self._per_round_epsilon, 1e-12)
            sigma = (
                self._dp_max_grad_norm
                * math.sqrt(2.0 * math.log(1.25 / self._dp_delta))
                / eps_round
            )
            # Sigma was calibrated so one round costs AT MOST eps_round
            # under the classic Gaussian bound; charging exactly eps_round
            # per round therefore yields a valid (conservative)
            # naive-composition total that sums to dp_epsilon.
            round_cost = eps_round

        self._stats["dp_sigma"] = float(sigma)
        self._stats["dp_noise_added"] = True
        self._record_privacy_spend(sigma, round_cost)

        return [g + torch.randn_like(g) * sigma for g in grads]

    def _record_privacy_spend(
        self, sigma: float, round_cost: float | None
    ) -> None:
        """Accumulate this round's privacy cost into the stats ledger.

        ``round_cost`` is the known per-round charge (auto-derived path).
        When None (explicit multiplier configured), the actual
        (epsilon, delta) cost of one Gaussian query with noise ``sigma`` is
        computed with the RDP accountant; if that module cannot be loaded a
        conservative fallback charges the full per-round budget share.
        """
        if round_cost is None:
            accountant = _load_rdp_accounting()
            if accountant is not None:
                accountant.add_query(sigma)
                round_cost = accountant.get_epsilon(self._dp_delta)
            else:
                logger.warning(
                    f"[{self._node_id}] RDP accountant unavailable; "
                    f"charging full per-round budget share "
                    f"{self._per_round_epsilon:.6f} -- total may be "
                    f"inaccurate."
                )
                round_cost = self._per_round_epsilon
        self._stats["dp_cumulative_epsilon"] = (
            self._stats["dp_cumulative_epsilon"] + round_cost
        )

    def _check_budget(self) -> None:
        """Raise :class:`DPBudgetExhausted` when the budget is spent.

        Fail-closed: continuing would broadcast gradients whose noise no
        longer corresponds to any honest privacy claim.
        """
        if self._dp_mode != "enabled" or not math.isfinite(self._dp_epsilon):
            return
        if self._stats["dp_cumulative_epsilon"] >= self._dp_epsilon:
            raise DPBudgetExhausted(
                f"[{self._node_id}] DP privacy budget exhausted: spent "
                f"{self._stats['dp_cumulative_epsilon']:.6f} of "
                f"(epsilon={self._dp_epsilon}, delta={self._dp_delta}). "
                f"Refusing to run another round with stale calibration."
            )

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

        Raises:
            DPBudgetExhausted: If ``dp_mode="enabled"`` and the accumulated
                privacy spend already meets or exceeds ``dp_epsilon``.
        """
        self._check_budget()
        self._round += 1
        round_start = time.time()

        # Prune gradient contributions left over from earlier rounds so a
        # peer that went silent (or a late/out-of-order gossip message
        # consumed in an earlier round) cannot have its stale submission
        # averaged into this round's update (audit F-018).
        stale_peers = [
            pid
            for pid, stored_round in self._received_rounds.items()
            if stored_round != self._round
        ]
        for pid in stale_peers:
            del self._received_grads[pid]
            del self._received_rounds[pid]

        # Step 1: Local training
        local_grads = local_train_fn(self._local_steps)
        self._stats["total_local_steps"] += self._local_steps

        # Step 1.5: Apply differential privacy (clip, then noise).
        # Clipping bounds sensitivity in both modes; noise is added ONLY in
        # dp_mode="enabled" (audit A-C2: clip_only is not DP).
        if math.isfinite(self._dp_epsilon):
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
                    msg_round = peer_data.get("round")
                    # Round-version guard (audit F-018): only accept
                    # gradients produced for the round in progress.
                    # Messages without a ``round`` field (legacy senders)
                    # are treated as current-round.
                    if msg_round is None or msg_round == self._round:
                        peer_id = peer_data.get("peer_id", "unknown")
                        self._received_grads[peer_id] = peer_data["gradients"]
                        self._received_rounds[peer_id] = self._round
                    else:
                        logger.warning(
                            f"Dropping stale gradients from "
                            f"{peer_data.get('peer_id', 'unknown')}: message "
                            f"round {msg_round} != current round {self._round}"
                        )
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
            # Defense in depth: never average a contribution tagged with a
            # round other than the one in progress (audit F-018).  Untagged
            # entries (no recorded round) are treated as current-round.
            stored_round = self._received_rounds.get(peer_id)
            if stored_round is not None and stored_round != self._round:
                continue
            # Guard against structural mismatches before averaging so
            # PyTorch broadcasting cannot silently corrupt the sum.
            if len(peer_grads) == len(local_grads) and all(
                pg.shape == local_grads[i].shape
                for i, pg in enumerate(peer_grads)
            ):
                all_grads.append(peer_grads)

        num_sources = len(all_grads)
        if num_sources <= 1:
            return local_grads

        # FedProx: add proximal term to gradients
        if self._algorithm == "fedprox" and self._global_model_params is not None:
            local_params = self._snapshot_local_params()
            local_grads = self._apply_fedprox_term(local_grads, local_params)
            all_grads[0] = local_grads

        # Weighted average (FedAvg or FedProx)
        averaged = []
        for i in range(len(local_grads)):
            grad_sum = sum(g[i] for g in all_grads)
            averaged.append(grad_sum / num_sources)

        return averaged

    def _snapshot_local_params(self) -> list[torch.Tensor] | None:
        """Return the current local model parameters (w_local) for FedProx.

        Uses the ``lora_adapter`` callback (the same source of local weights
        used for local training).  Returns None when no callback is wired or
        it fails, so callers can skip the proximal term instead of corrupting
        the gradient with a weight-mixed update.
        """
        if self._lora_adapter is None:
            return None
        try:
            params = list(self._lora_adapter())
        except Exception as e:
            logger.warning(f"Failed to snapshot local params for FedProx: {e}")
            return None
        return [p.detach().clone() for p in params]

    def _apply_fedprox_term(
        self,
        grads: list[torch.Tensor],
        local_params: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Apply FedProx proximal term to gradients.

        FedProx adds a regularization term to each gradient::

            grad_i <- grad_i + mu * (w_local_i - w_global_i)

        This penalizes the local model for drifting too far from the global
        model, which helps with heterogeneous data distributions
        (Li et al. 2020).

        Args:
            grads: Local gradients.
            local_params: Current local weights (w_local).  When not provided
                or shorter than ``grads``, falls back to ``lora_adapter``.
                If w_local cannot be determined for a parameter, its gradient
                is passed through unchanged rather than mixed with a weight
                tensor.

        Returns:
            Gradients with proximal term added.
        """
        if self._global_model_params is None or self._fedprox_mu <= 0:
            return grads

        if local_params is None:
            local_params = self._snapshot_local_params()

        proximal_grads = []
        for i, grad in enumerate(grads):
            if (
                i < len(self._global_model_params)
                and local_params is not None
                and i < len(local_params)
            ):
                global_param = self._global_model_params[i]
                # Proximal term: mu * (w_local - w_global)
                # This is added to the gradient to penalize divergence.
                proximal = self._fedprox_mu * (
                    local_params[i].detach() - global_param.detach()
                )
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
            Final training stats.  When ``dp_mode="enabled"`` the loop stops
            early (with a WARNING) if the (epsilon, delta) budget is
            exhausted before ``num_rounds`` complete.
        """
        for round_idx in range(self._num_rounds):
            logger.info(f"Federated round {round_idx + 1}/{self._num_rounds}")
            try:
                metrics = self.train_round(local_train_fn)
            except DPBudgetExhausted as e:
                logger.warning(f"Stopping federated training early: {e}")
                break
            logger.info(f"  Round complete: {metrics['elapsed_s']}s, "
                         f"{metrics['gradients_received']} peer gradients")

        return dict(self._stats)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)
