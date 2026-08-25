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
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

from distllm.dist.byzantine import _decode_public_key, _verify_bytes


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
        max_dataset_size: int = 1_000_000,
    ):
        self._adapter_mgr = adapter_manager
        self._merge_strategy = merge_strategy
        self._min_nodes = min_nodes_per_round
        self._checkpoint_interval = rounds_per_checkpoint
        # Upper bound on a node's self-reported dataset size so a single node
        # cannot dominate the federated average.
        self._max_dataset_size = max_dataset_size

        self._nodes: dict[str, NodeTrainingState] = {}
        self._rounds: list[FederatedRound] = []
        self._versions: dict[str, AdapterVersion] = {}
        self._current_round: FederatedRound | None = None
        self._node_public_keys: dict[str, Any] = {}
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

    def register_node_public_key(self, node_id: str, public_key: Any) -> None:
        """Register a node's Ed25519 public key for submission verification.

        Once a key is registered, ``submit_node_adapter`` REQUIRES a valid
        signature from that node (fail closed), so a submission cannot be
        forged or tampered with.  Accepts an Ed25519 key object, raw bytes, or
        the base64 string produced by ``distllm.dist.byzantine``.
        """
        with self._lock:
            self._node_public_keys[node_id] = _decode_public_key(public_key)

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
        signature: str = "",
    ) -> bool:
        """Submit a node's locally trained adapter for merging.

        Args:
            node_id: The submitting node.
            adapter_path: Path to the node's LoRA adapter weights.
            loss: Final training loss from this node.
            dataset_size: Size of the node's local dataset (capped at
                ``max_dataset_size`` so one node cannot dominate the average).
            signature: Base64 Ed25519 signature over
                ``node_id|round_id|adapter_path|loss|dataset_size``.  Required
                (fail closed) once the node's public key is registered.

        Returns:
            True if accepted, False if node not in current round or the
            submission is not authenticated.
        """
        with self._lock:
            if not self._current_round or self._current_round.status != "collecting":
                return False
            if node_id not in self._current_round.participating_nodes:
                return False

            state = self._nodes.get(node_id)
            if not state:
                return False

            # SECURITY: authenticate the submission when the node has a
            # registered public key.  The signature binds node_id, the round,
            # and the submitted values so they cannot be forged or tampered.
            public_key = self._node_public_keys.get(node_id)
            if public_key is not None:
                if not signature:
                    logger.warning(
                        f"Node {node_id}: signed adapter required but no signature supplied"
                    )
                    return False
                message = (
                    f"{node_id}|{self._current_round.round_id}|"
                    f"{adapter_path}|{loss}|{dataset_size}"
                )
                if not _verify_bytes(public_key, message, signature):
                    logger.warning(f"Node {node_id}: invalid adapter submission signature")
                    return False

            state.adapter_path = adapter_path
            state.last_loss = loss
            state.last_sync = time.time()
            state.status = "completed"
            if dataset_size > 0:
                # Cap self-reported dataset size to prevent weight dominance.
                state.dataset_size = min(dataset_size, self._max_dataset_size)

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
    """Secure aggregation for federated learning via pairwise-cancelling masks.

    Protocol (simplified SecAgg, Bonawitz et al. 2017, mask phase only):

    Every unordered pair of parties (i, j) shares a symmetric 32-byte key.
    For a given round nonce, party i derives ``m_ij = PRF(key_ij, nonce)``
    as an integer-valued float tensor and adds

        mask_i = Σ_{j>i} m_ij − Σ_{j<i} m_ij

    to its gradient before sending it anywhere.  Because each pairwise mask
    appears exactly once with a + sign and exactly once with a − sign across
    the whole party set, the masks cancel **by construction**:

        Σ_i (g_i + mask_i) = Σ_i g_i

    Any party that collects every participant's masked vector — including
    its own masked vector, which ``aggregate_received_shares`` recomputes —
    recovers the exact global sum (and hence the mean).  No individual
    gradient is revealed: what leaves a party is its gradient plus a
    deterministic pseudorandom mask that no single peer can strip alone.

    Trust model / documented limitations (read before shipping):
      * Full participation is required for exactness.  If a peer drops out,
        the dead edge's mask terms cannot cancel and every survivor computes
        the same residual-shifted sum.  A WARNING naming the missing peer is
        logged.  Recovering exactness under dropout requires the Shamir
        seed-recovery phase of full SecAgg (not implemented here).
      * Masks are integer-valued floats bounded by ``mask_bound`` so they
        cancel bit-exact on the float32 mantissa; arbitrary float gradients
        aggregate within fp rounding tolerance.  True finite-field arithmetic
        (exact for any magnitude) is future work.
      * Integer-dtype gradients are rejected: they need modular arithmetic.
      * Keys must come from an authenticated channel (e.g. X25519-derived);
        this class consumes key material, it does not establish it.
      * The coordinator/aggregator sees masked vectors only; it must not
        learn any party's pairwise keys.

    Usage::

        # Once per pair, over an authenticated channel:
        keys = {("node-1", "node-2"): shared_secret_32_bytes, ...}

        aggregator = SecureAggregator(
            node_id="node-1",
            peer_ids=["node-2", "node-3"],
            pairwise_keys=keys,
            round_nonce=b"round-17",
        )
        masked = aggregator.split_gradients(local_gradients)
        # Send masked[peer] to that peer (or to the collector) ...
        aggregate = aggregator.aggregate_received_shares(
            local_gradients, received_masked_vectors
        )
        mean = [t / num_parties for t in aggregate]
    """

    #: Per-element upper bound for mask magnitude.  Masks are integer-valued
    #: floats in (-bound, bound); float32 represents integers exactly up to
    #: 2**24, so a pair of cancelling masks plus typical gradient values stay
    #: exactly representable and cancel bit-exact.
    _MASK_BOUND = 1 << 17

    def __init__(
        self,
        node_id: str,
        peer_ids: list[str],
        pairwise_keys: dict[tuple[str, str], bytes] | None = None,
        round_nonce: bytes = b"",
        mask_bound: int | None = None,
    ):
        self._node_id = node_id
        self._peer_ids = list(peer_ids)

        if node_id in self._peer_ids:
            raise ValueError(
                f"peer list for node {node_id!r} contains the node itself"
            )

        if self._peer_ids:
            if not pairwise_keys:
                raise ValueError(
                    f"SecureAggregator for {node_id!r} requires a pairwise key "
                    f"for every peer; got no key material. Refusing to run "
                    f"mask-less 'secure' aggregation (fail closed)."
                )
            normalized: dict[tuple[str, str], bytes] = {}
            for pair, key in pairwise_keys.items():
                a, b = sorted(pair[:2])
                if a == b:
                    raise ValueError(
                        f"pairwise key entry {pair!r} pairs a node with itself"
                    )
                if len(key) != 32:
                    raise ValueError(
                        f"pairwise key for {pair!r} must be exactly 32 bytes, "
                        f"got {len(key)}"
                    )
                normalized[(a, b)] = key
            self._pair_keys = normalized

            missing = [
                p for p in self._peer_ids
                if tuple(sorted((node_id, p))) not in self._pair_keys
            ]
            if missing:
                raise ValueError(
                    f"missing pairwise key(s) between {node_id!r} and "
                    f"{missing}: secure aggregation cannot run without them"
                )
        else:
            self._pair_keys = {}

        self._round_nonce = bytes(round_nonce)
        self._mask_bound = (
            mask_bound if mask_bound is not None else self._MASK_BOUND
        )
        if self._mask_bound <= 0 or self._mask_bound > (1 << 23):
            raise ValueError(
                f"mask_bound must be in (0, 2**23], got {self._mask_bound}"
            )

    # ── Mask derivation ─────────────────────────────────────────────────

    def _pairwise_mask(self, other_id: str, like: torch.Tensor) -> torch.Tensor:
        """Derive this node's signed pairwise mask toward ``other_id``.

        Sign convention: positive when ``self._node_id < other_id``,
        negative otherwise — this is what makes Σ_i mask_i ≡ 0: the edge
        (i, j) contributes +m_ij to party i's total and −m_ij to party j's.
        """
        pair = tuple(sorted((self._node_id, other_id)))
        key = self._pair_keys[pair]
        material = (
            key + len(self._round_nonce).to_bytes(4, "big") + self._round_nonce
        )
        digest = hashlib.sha256(material).digest()

        # Expand the digest into enough uniform bytes to cover the tensor
        # (SHA-256 in counter mode).
        need = like.numel() * 4
        stream = bytearray()
        counter = 0
        while len(stream) < need:
            stream.extend(
                hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
            )
            counter += 1

        words = np.frombuffer(bytes(stream[:need]), dtype="<u4").reshape(
            like.shape
        )

        bound = self._mask_bound
        vals = (words.astype(np.int64) % (2 * bound)) - bound
        mask = torch.from_numpy(vals.astype(np.float32)).reshape(like.shape).to(
            like.dtype
        )
        if other_id < self._node_id:
            mask = -mask
        return mask

    def _own_mask_sum(self, like: torch.Tensor) -> torch.Tensor:
        """Total mask this node adds to every gradient tensor."""
        total = torch.zeros_like(like)
        for peer in self._peer_ids:
            total = total + self._pairwise_mask(peer, like)
        return total

    # ── Public API ──────────────────────────────────────────────────────

    def split_gradients(
        self, gradients: list[torch.Tensor]
    ) -> dict[str, list[torch.Tensor]]:
        """Return each peer's copy of this node's masked gradients.

        With pairwise cancelling masks every participant sends the SAME
        masked vector to all peers (the mask is the node's own), so the
        returned lists are independent clones of one masked vector — one
        entry per peer, never shared storage.
        """
        if not self._peer_ids:
            return {}

        for grad in gradients:
            if not grad.dtype.is_floating_point:
                raise NotImplementedError(
                    f"SecureAggregator supports floating-point tensors only; "
                    f"got {grad.dtype}. Integer gradients require modular "
                    f"(finite-field) arithmetic, which is not implemented."
                )

        masked_vecs = [grad + self._own_mask_sum(grad) for grad in gradients]

        out: dict[str, list[torch.Tensor]] = {}
        for peer_id in self._peer_ids:
            out[peer_id] = [m.clone() for m in masked_vecs]
        return out

    def aggregate_received_shares(
        self,
        self_gradients: list[torch.Tensor],
        received_shares: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Combine own masked gradient with peers' masked vectors.

        Args:
            self_gradients: Own raw gradients (one tensor per slot).
            received_shares: Dict of peer_id -> list of that peer's masked
                gradient tensors (same slot order and shapes).

        Returns:
            Aggregated tensors — the exact SUM over all participants'
            gradients when every configured peer contributed (full
            participation).  Divide by the participant count for the mean.

        Dropout behavior (documented): a missing peer's dead-edge mask terms
        cannot cancel, so the result shifts by a residual shared by all
        survivors; a WARNING names each dropped peer.  Contributions from
        ids outside the configured peer list are excluded with a WARNING.
        """
        expected_peers = set(self._peer_ids)
        delivered = {
            pid: shares
            for pid, shares in received_shares.items()
            if pid in expected_peers
        }
        for stranger in sorted(set(received_shares) - expected_peers):
            logger.warning(
                f"SecureAggregator[{self._node_id}]: ignoring contribution "
                f"from non-peer id {stranger!r}"
            )

        dropped = sorted(p for p in expected_peers if p not in delivered)
        if dropped:
            logger.warning(
                f"SecureAggregator[{self._node_id}]: dropout detected — no "
                f"masked vector from peer(s) {dropped}; the aggregate is "
                f"shifted by their dead-edge mask residual and will NOT "
                f"equal the sum over delivered participants."
            )

        aggregated: list[torch.Tensor] = []
        for i, grad in enumerate(self_gradients):
            if not grad.dtype.is_floating_point:
                raise NotImplementedError(
                    f"SecureAggregator supports floating-point tensors only; "
                    f"got {grad.dtype}. Integer gradients require modular "
                    f"(finite-field) arithmetic, which is not implemented."
                )
            # Self term must be this node's MASKED vector (grad + own mask) —
            # the same vector split_gradients distributes — so all N masks
            # enter the sum and cancel to zero:
            #   Σ_i (g_i + mask_i) = Σ_i g_i + Σ_i mask_i = Σ_i g_i.
            total = grad + self._own_mask_sum(grad)
            for pid, shares in delivered.items():
                if i < len(shares):
                    share = shares[i]
                    if share.shape != grad.shape:
                        raise ValueError(
                            f"shape mismatch from peer {pid!r} at slot {i}: "
                            f"{tuple(share.shape)} vs {tuple(grad.shape)}"
                        )
                    if share.dtype != grad.dtype:
                        raise ValueError(
                            f"dtype mismatch from peer {pid!r} at slot {i}: "
                            f"{share.dtype} vs {grad.dtype}"
                        )
                    total = total + share
            aggregated.append(total)
        return aggregated
