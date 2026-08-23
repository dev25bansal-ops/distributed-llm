"""PBFT-style Byzantine fault tolerance for distributed LLM inference.

Provides Byzantine fault tolerance primitives for the distributed inference
layer, including:

- **PBFTNode**: Practical Byzantine Fault Tolerance consensus engine with
  pre-prepare, prepare, and commit phases. Tolerates ``f`` faulty nodes
  with ``3f + 1`` total nodes. Includes view-change protocol for primary
  failure recovery.
- **QuorumManager**: Quorum-based decision making. Computes quorum sizes
  for PBFT (``2f + 1``) and crash-only (``f + 1``) failure models.
- **SplitBrainDetector**: Vector-clock-based conflict detection and
  resolution for partition-heal scenarios.
- **ByzantineCoordinator**: High-level coordinator that submits operations
  through PBFT consensus and exposes agreed state.

Usage::

    node = PBFTNode("node-0", node_ids=["node-0", "node-1", "node-2", "node-3"])
    node.handle_request({"op": "register_model", "model_id": "llama-70b"})

    # Quorum management
    qm = QuorumManager(total_nodes=4, f=1)
    assert qm.check_quorum(responses)

    # Split-brain detection
    sbd = SplitBrainDetector()
    sbd.update("node-0", "state-hash-a")
    conflicts = sbd.detect_conflicts(local_clocks, peer_clocks)

    # High-level coordinator
    coord = ByzantineCoordinator(node_ids=[...])
    result = await coord.submit({"op": "update_topology"})
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from loguru import logger


def _encode_public_key(public_key: ed25519.Ed25519PublicKey) -> str:
    """Serialize an Ed25519 public key to a compact base64 string."""
    raw = public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(raw).decode("ascii")


def _decode_public_key(
    value: ed25519.Ed25519PublicKey | bytes | str,
) -> ed25519.Ed25519PublicKey:
    """Coerce an Ed25519 public key from an object, raw bytes, or base64 str."""
    if isinstance(value, ed25519.Ed25519PublicKey):
        return value
    raw = base64.b64decode(value) if isinstance(value, str) else bytes(value)
    return ed25519.Ed25519PublicKey.from_public_bytes(raw)


def _sign_bytes(private_key: ed25519.Ed25519PrivateKey, payload: str) -> str:
    """Sign a canonical payload with an Ed25519 key, returning base64."""
    return base64.b64encode(
        private_key.sign(payload.encode("utf-8"))
    ).decode("ascii")


def _verify_bytes(
    public_key: ed25519.Ed25519PublicKey, payload: str, signature_b64: str
) -> bool:
    """Verify a base64 Ed25519 signature over a canonical payload."""
    try:
        public_key.verify(base64.b64decode(signature_b64), payload.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ===========================================================================
# PBFTNode -- Practical Byzantine Fault Tolerance consensus
# ===========================================================================


class PBFTPhase(Enum):
    """Phases of the PBFT consensus protocol."""

    PRE_PREPARE = auto()
    PREPARE = auto()
    COMMIT = auto()
    REPLY = auto()


class NodeStatus(Enum):
    """Status of a PBFT participant node."""

    ACTIVE = auto()
    SUSPECT = auto()
    FAULTY = auto()
    RECOVERING = auto()


@dataclass
class PBFTMessage:
    """A message exchanged during PBFT consensus phases.

    Attributes:
        phase: The PBFT phase this message belongs to.
        view: The current view number.
        sequence: Sequence number of the operation.
        digest: Digest (hash) of the operation.
        sender: Node ID of the sender.
        operation: The operation payload (only in PRE_PREPARE).
        signature: Optional message signature for authenticity.
    """

    phase: PBFTPhase
    view: int
    sequence: int
    digest: str
    sender: str
    operation: dict[str, Any] | None = None
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this message to a dict."""
        return {
            "phase": self.phase.name,
            "view": self.view,
            "sequence": self.sequence,
            "digest": self.digest,
            "sender": self.sender,
            "operation": self.operation,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PBFTMessage:
        """Deserialize from a dict."""
        return cls(
            phase=PBFTPhase[data["phase"]],
            view=data["view"],
            sequence=data["sequence"],
            digest=data["digest"],
            sender=data["sender"],
            operation=data.get("operation"),
            signature=data.get("signature"),
        )


class PBFTNode:
    """PBFT consensus participant implementing the core protocol phases.

    Implements Practical Byzantine Fault Tolerance consensus with:
    - Pre-prepare / prepare / commit three-phase commit protocol
    - ``3f + 1`` node requirement (tolerates ``f`` faulty nodes)
    - View change protocol for primary failure detection and recovery
    - Watermark-based garbage collection of committed sequences
    - Digest verification for operation integrity

    Args:
        node_id: Unique identifier for this node.
        node_ids: List of all node IDs in the cluster.
        f: Maximum number of faulty nodes to tolerate (default: auto-computed
            from ``(len(node_ids) - 1) // 3``).
        callback: Optional callback invoked when an operation is committed.
    """

    def __init__(
        self,
        node_id: str,
        node_ids: list[str],
        f: int | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        signing_key: ed25519.Ed25519PrivateKey | None = None,
        node_public_keys: dict[
            str, ed25519.Ed25519PublicKey | bytes | str
        ] | None = None,
    ) -> None:
        if node_id not in node_ids:
            raise ValueError(f"node_id {node_id!r} not in node_ids list")
        self._node_id = node_id
        self._node_ids = sorted(node_ids)
        self._total_nodes = len(self._node_ids)
        self._f = f if f is not None else (self._total_nodes - 1) // 3
        self._callback = callback

        if self._total_nodes < 3 * self._f + 1:
            logger.warning(
                f"PBFT requires 3f+1 nodes (have {self._total_nodes}, f={self._f}); "
                f"need {3 * self._f + 1}"
            )

        # Ed25519 identity: each node signs every message it originates, and
        # a message is only accepted once the sender's public key has been
        # registered (fail closed — unknown senders are rejected).
        self._signing_key = signing_key or ed25519.Ed25519PrivateKey.generate()
        self._public_keys: dict[str, ed25519.Ed25519PublicKey] = {}
        self.register_public_key(self._node_id, self._signing_key.public_key())
        if node_public_keys:
            self.register_public_keys(node_public_keys)

        # Current view
        self._view: int = 0
        self._last_sequence: int = 0
        self._committed_sequences: set[int] = set()

        # Message logs
        self._pre_prepare_log: dict[int, PBFTMessage] = {}  # seq -> pre-prepare
        self._prepare_log: dict[int, list[PBFTMessage]] = {}  # seq -> prepare msgs
        self._commit_log: dict[int, list[PBFTMessage]] = {}  # seq -> commit msgs

        # Watermarks (low / high watermarks for garbage collection)
        self._low_watermark: int = 0
        self._high_watermark: int = 0
        self._watermark_window: int = 128

        # View change state
        self._view_change_in_progress: bool = False
        self._view_change_messages: dict[int, list[dict[str, Any]]] = {}
        self._new_view_messages: dict[int, dict[str, Any]] = {}
        self._last_view_change: float = 0.0
        self._view_change_timeout: float = 10.0

        # Performance tracking
        self._committed_ops: int = 0
        self._rejected_ops: int = 0
        self._start_time: float = time.monotonic()
        self._node_status: dict[str, NodeStatus] = {
            nid: NodeStatus.ACTIVE for nid in self._node_ids
        }

        logger.info(
            f"PBFTNode {node_id} initialized: {self._total_nodes} nodes, "
            f"f={self._f}, active"
        )

    # ------------------------------------------------------------------ #
    # Property accessors
    # ------------------------------------------------------------------ #

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def view(self) -> int:
        return self._view

    @property
    def primary(self) -> str:
        """The primary node for the current view (``p = v mod |R|``)."""
        return self._node_ids[self._view % self._total_nodes]

    @property
    def is_primary(self) -> bool:
        return self._node_id == self.primary

    @property
    def faulty_count(self) -> int:
        return sum(
            1 for s in self._node_status.values() if s == NodeStatus.FAULTY
        )

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    @property
    def f(self) -> int:
        return self._f

    @property
    def committed_count(self) -> int:
        return self._committed_ops

    @property
    def quorum_size(self) -> int:
        """The minimum number of matching responses needed for PBFT."""
        return 2 * self._f + 1

    # ------------------------------------------------------------------ #
    # Core PBFT protocol
    # ------------------------------------------------------------------ #

    def _compute_digest(self, operation: dict[str, Any]) -> str:
        """Compute a SHA-256 digest of an operation."""
        serialized = json.dumps(operation, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def handle_request(self, operation: dict[str, Any]) -> PBFTMessage | None:
        """Handle a client request.

        If this node is the primary, initiates the PBFT consensus by
        broadcasting a PRE-PREPARE message. If not, the request is forwarded
        to the primary.

        Args:
            operation: The operation to submit to consensus.

        Returns:
            The PRE-PREPARE message if this node is primary, ``None`` otherwise.
        """
        if not self.is_primary:
            logger.debug(
                f"Forwarding request to primary {self.primary}"
            )
            return None

        if self._view_change_in_progress:
            logger.warning("View change in progress; request queued")
            return None

        digest = self._compute_digest(operation)
        sequence = self._last_sequence + 1

        if not self._check_sequence_watermark(sequence):
            logger.warning(f"Sequence {sequence} outside watermark window")
            return None

        self._last_sequence = sequence

        msg = PBFTMessage(
            phase=PBFTPhase.PRE_PREPARE,
            view=self._view,
            sequence=sequence,
            digest=digest,
            sender=self._node_id,
            operation=operation,
            signature=self._sign_message(digest, view=self._view, sequence=sequence),
        )

        self._pre_prepare_log[sequence] = msg
        logger.debug(
            f"PRE-PREPARE: view={self._view}, seq={sequence}, digest={digest[:12]}"
        )

        # Primary auto-accepts its own PRE-PREPARE by emitting PREPARE
        self._handle_pre_prepare(msg, from_primary=True)

        return msg

    def handle_pre_prepare(
        self, msg: PBFTMessage | None = None, **kwargs: Any
    ) -> bool:
        """Handle an incoming PRE-PREPARE message.

        Accepts either a ``PBFTMessage`` instance or keyword arguments to
        construct one (``phase``, ``view``, ``sequence``, ``digest``,
        ``sender``, ``operation``).

        Validates:
        - Message from the expected primary for the current view.
        - Sequence number within watermark.
        - Digest matches the operation payload.
        - No duplicate pre-prepare for this sequence.

        On acceptance, broadcasts a PREPARE message.

        Args:
            msg: The incoming PRE-PREPARE message.
            **kwargs: Alternative to ``msg`` for constructing a message.

        Returns:
            ``True`` if the message was accepted and a PREPARE was broadcast.
        """
        if msg is None:
            msg = PBFTMessage(phase=PBFTPhase.PRE_PREPARE, **kwargs)
        return self._handle_pre_prepare(msg, from_primary=False)

    def _handle_pre_prepare(
        self, msg: PBFTMessage, from_primary: bool
    ) -> bool:
        """Shared pre-prepare handling for primary and non-primary."""
        if msg.phase != PBFTPhase.PRE_PREPARE:
            logger.warning(f"Expected PRE_PREPARE, got {msg.phase}")
            return False

        # Authenticity: a PRE-PREPARE received from the network must carry a
        # valid Ed25519 signature from the sender's registered public key.
        if not from_primary and not self.verify_signature(msg):
            logger.warning(
                f"PRE-PREPARE from {msg.sender} failed signature verification"
            )
            return False

        # Only the primary in the current view can send PRE-PREPARE
        if not from_primary and msg.sender != self.primary:
            logger.warning(
                f"PRE-PREPARE from non-primary {msg.sender} "
                f"(expected {self.primary})"
            )
            return False

        if not from_primary and msg.view != self._view:
            logger.warning(
                f"PRE-PREPARE view mismatch: {msg.view} != {self._view}"
            )
            return False

        if not self._check_sequence_watermark(msg.sequence):
            logger.warning(
                f"PRE-PREPARE seq {msg.sequence} outside watermark"
            )
            return False

        # Verify digest matches operation (if operation present)
        if msg.operation is not None:
            computed = self._compute_digest(msg.operation)
            if computed != msg.digest:
                logger.error(
                    f"Digest mismatch: computed={computed[:12]}, "
                    f"claimed={msg.digest[:12]}"
                )
                return False
        else:
            msg.digest = msg.digest if msg.digest else ""

        # Check for duplicate
        existing = self._pre_prepare_log.get(msg.sequence)
        if existing is not None:
            if existing.digest != msg.digest:
                logger.warning(f"Conflict at seq {msg.sequence}: divergent pre-prepare")
                return False
            return True  # Already accepted

        self._pre_prepare_log[msg.sequence] = msg

        # Broadcast PREPARE
        prepare = PBFTMessage(
            phase=PBFTPhase.PREPARE,
            view=self._view,
            sequence=msg.sequence,
            digest=msg.digest,
            sender=self._node_id,
            signature=self._sign_message(
                msg.digest, view=self._view, sequence=msg.sequence
            ),
        )
        self._prepare_log.setdefault(msg.sequence, []).append(prepare)
        logger.debug(
            f"PREPARE: view={self._view}, seq={msg.sequence}, "
            f"digest={msg.digest[:12]}"
        )

        # Check if prepare quorum reached locally after adding
        self._try_prepare_quorum(msg.sequence, msg.digest)

        return True

    def handle_prepare(self, msg: PBFTMessage) -> bool:
        """Handle an incoming PREPARE message.

        Validates the prepare message and appends it to the prepare log.
        When ``2f`` matching PREPARE messages (from distinct nodes other than
        self) are collected for a sequence, a quorum is reached and this node
        broadcasts a COMMIT message.

        Args:
            msg: The incoming PREPARE message.

        Returns:
            ``True`` if the message was accepted.
        """
        if msg.phase != PBFTPhase.PREPARE:
            logger.warning(f"Expected PREPARE, got {msg.phase}")
            return False

        if msg.view != self._view:
            return False

        if msg.sender == self._node_id:
            return True  # Skip our own prepare (already logged)

        # Authenticity: only accept PREPARE messages with a valid signature
        # from the sender's registered public key.
        if not self.verify_signature(msg):
            logger.warning(
                f"PREPARE from {msg.sender} failed signature verification "
                f"at seq {msg.sequence}"
            )
            return False

        seq = msg.sequence

        # Ensure pre-prepare exists for this sequence
        if seq not in self._pre_prepare_log:
            logger.warning(f"PREPARE for unknown seq {seq}")
            return False

        # Ensure digest matches the pre-prepare
        expected_digest = self._pre_prepare_log[seq].digest
        if msg.digest != expected_digest:
            logger.warning(f"PREPARE digest mismatch at seq {seq}")
            return False

        # Deduplicate by sender
        existing_senders = {
            m.sender for m in self._prepare_log.get(seq, [])
        }
        if msg.sender in existing_senders:
            return True  # Already have this prepare

        self._prepare_log.setdefault(seq, []).append(msg)
        logger.debug(
            f"PREPARE from {msg.sender}: view={msg.view}, seq={seq}"
        )

        # Check if we can move to COMMIT
        self._try_prepare_quorum(seq, msg.digest)

        return True

    def _try_prepare_quorum(self, sequence: int, digest: str) -> None:
        """Check if prepare quorum is reached and broadcast COMMIT.

        PBFT: a node broadcasts COMMIT when it has received ``2f`` PREPARE
        messages from distinct *other* nodes that match the pre-prepare.
        The node's own PREPARE (already in the prepare log) brings the
        total to ``2f + 1``.
        """
        prep_messages = self._prepare_log.get(sequence, [])
        matching = [m for m in prep_messages if m.digest == digest]
        distinct = {m.sender for m in matching}

        # ``distinct`` already includes this node (its own PREPARE was
        # added in ``_handle_pre_prepare``), so the count must reach
        # the full quorum size (2f + 1) without any extra offset.
        if len(distinct) >= self.quorum_size:
            # Check we haven't already committed or broadcast COMMIT
            already_committed = sequence in self._committed_sequences
            already_broadcast = any(
                m.sender == self._node_id and m.phase == PBFTPhase.COMMIT
                for m in self._commit_log.get(sequence, [])
            )
            if not already_committed and not already_broadcast:
                commit = PBFTMessage(
                    phase=PBFTPhase.COMMIT,
                    view=self._view,
                    sequence=sequence,
                    digest=digest,
                    sender=self._node_id,
                    signature=self._sign_message(
                        digest, view=self._view, sequence=sequence
                    ),
                )
                self._commit_log.setdefault(sequence, []).append(commit)
                logger.debug(
                    f"COMMIT: view={self._view}, seq={sequence}, "
                    f"digest={digest[:12]}"
                )

                # Check if commit quorum reached
                self._try_commit_quorum(sequence, digest)

    def handle_commit(self, msg: PBFTMessage) -> bool:
        """Handle an incoming COMMIT message.

        Collects COMMIT messages from distinct nodes. When ``2f + 1`` matching
        COMMITs are received (including self), the operation is committed and
        the user-provided callback is invoked.

        Args:
            msg: The incoming COMMIT message.

        Returns:
            ``True`` if the message was accepted.
        """
        if msg.phase != PBFTPhase.COMMIT:
            logger.warning(f"Expected COMMIT, got {msg.phase}")
            return False

        if msg.view != self._view:
            return False

        # Authenticity: only accept COMMIT messages with a valid signature
        # from the sender's registered public key.
        if msg.sender != self._node_id and not self.verify_signature(msg):
            logger.warning(
                f"COMMIT from {msg.sender} failed signature verification "
                f"at seq {msg.sequence}"
            )
            return False

        seq = msg.sequence

        # Must have prepare quorum first.
        # ``matching_prep`` includes this node's own PREPARE (logged in
        # ``_handle_pre_prepare``), so compare directly to quorum size.
        prep_msgs = self._prepare_log.get(seq, [])
        matching_prep = {
            m.sender for m in prep_msgs if m.digest == msg.digest
        }
        if len(matching_prep) < self.quorum_size:
            logger.warning(
                f"COMMIT before prepare quorum at seq {seq} "
                f"(have {len(matching_prep) + 1}, need {self.quorum_size})"
            )
            return False

        existing_senders = {
            m.sender for m in self._commit_log.get(seq, [])
        }
        if msg.sender in existing_senders:
            return True

        self._commit_log.setdefault(seq, []).append(msg)
        logger.debug(f"COMMIT from {msg.sender}: view={msg.view}, seq={seq}")

        self._try_commit_quorum(seq, msg.digest)

        return True

    def _try_commit_quorum(self, sequence: int, digest: str) -> None:
        """Check if commit quorum is reached and finalize the operation."""
        commit_msgs = self._commit_log.get(sequence, [])
        matching = [m for m in commit_msgs if m.digest == digest]
        distinct = {m.sender for m in matching}

        if len(distinct) >= self.quorum_size:
            if sequence not in self._committed_sequences:
                self._committed_sequences.add(sequence)
                self._committed_ops += 1

                # Get the operation from pre-prepare
                pp = self._pre_prepare_log.get(sequence)
                operation = pp.operation if pp else None

                logger.info(
                    f"COMMITTED seq={sequence}, digest={digest[:12]}, "
                    f"view={self._view}, total_committed={self._committed_ops}"
                )

                if self._callback and operation is not None:
                    self._callback(operation)

                # Garbage collection
                self._advance_watermark()

    # ------------------------------------------------------------------ #
    # View change protocol
    # ------------------------------------------------------------------ #

    def start_view_change(self) -> None:
        """Initiate a view change.

        Called when the primary is suspected to be faulty (e.g., request
        timeout, invalid message, or explicit trigger). Broadcasts a
        view-change message containing the node's checkpoint state.
        """
        if self._view_change_in_progress:
            return

        new_view = self._view + 1
        self._view_change_in_progress = True
        self._last_view_change = time.monotonic()

        # Collect latest committed sequence
        last_committed = max(self._committed_sequences) if self._committed_sequences else 0

        vc_message = {
            "type": "view_change",
            "node_id": self._node_id,
            "new_view": new_view,
            "last_committed": last_committed,
            "signature": self._sign_view_message(
                "view_change", self._node_id, new_view, last_committed
            ),
            "pre_prepare_log": {
                str(k): v.to_dict() for k, v in self._pre_prepare_log.items()
                if k > self._low_watermark
            },
            "prepare_log": {
                str(k): [m.to_dict() for m in v]
                for k, v in self._prepare_log.items()
                if k > self._low_watermark
            },
        }

        self._view_change_messages.setdefault(new_view, []).append(vc_message)

        logger.info(
            f"View change started: {self._view} -> {new_view}, "
            f"last_committed={last_committed}"
        )

    def handle_view_change(self, message: dict[str, Any]) -> None:
        """Handle an incoming view-change message.

        Collects view-change messages from other nodes. Once ``2f + 1``
        view-change messages are collected for the new view, the new primary
        broadcasts a ``new_view`` message.

        Args:
            message: The view-change message dict.
        """
        new_view = message.get("new_view", 0)
        sender = message.get("node_id", "unknown")

        if sender not in self._node_ids:
            logger.warning(f"View change from unknown node {sender}")
            return

        # Authenticity: only accept view-change messages signed by the sender.
        if not self._verify_view_message("view_change", message):
            logger.warning(
                f"View change from {sender} failed signature verification"
            )
            return

        if new_view <= self._view:
            return

        self._view_change_messages.setdefault(new_view, []).append(message)
        collected = len(self._view_change_messages[new_view])

        logger.debug(
            f"View-change collected: {collected}/{self.quorum_size} "
            f"for view {new_view} (from {sender})"
        )

        if collected >= self.quorum_size:
            # The new primary broadcasts new-view
            new_primary = self._node_ids[new_view % self._total_nodes]
            if self._node_id == new_primary:
                self._broadcast_new_view(new_view)

    def handle_new_view(self, message: dict[str, Any]) -> None:
        """Handle a new-view message from the new primary.

        Accepts the new view and replays the pre-prepare / prepare log
        from the checkpoint state.

        Args:
            message: The new-view message dict.
        """
        new_view = message.get("new_view", 0)
        if new_view <= self._view:
            return

        # Authenticity: only accept new-view messages signed by the sender.
        if not self._verify_view_message("new_view", message):
            logger.warning("New-view message failed signature verification")
            return

        old_view = self._view
        self._view = new_view
        self._view_change_in_progress = False
        self._node_status[self.primary] = NodeStatus.ACTIVE

        # Replay operations from the new view's checkpoint
        checkpoint = message.get("checkpoint", {})
        for seq_str, pp_data in checkpoint.get("pre_prepare_log", {}).items():
            seq = int(seq_str)
            msg = PBFTMessage.from_dict(pp_data)
            msg.view = new_view
            if seq not in self._pre_prepare_log:
                self._pre_prepare_log[seq] = msg
                # Re-issue prepare for this sequence
                prepare = PBFTMessage(
                    phase=PBFTPhase.PREPARE,
                    view=new_view,
                    sequence=seq,
                    digest=msg.digest,
                    sender=self._node_id,
                    signature=self._sign_message(
                        msg.digest, view=new_view, sequence=seq
                    ),
                )
                self._prepare_log.setdefault(seq, []).append(prepare)

        logger.info(
            f"View changed: {old_view} -> {new_view}, "
            f"new primary={self.primary}"
        )

    def _broadcast_new_view(self, new_view: int) -> None:
        """Broadcast a new-view message as the new primary."""
        vc_msgs = self._view_change_messages.get(new_view, [])
        last_committed = max(
            m.get("last_committed", 0) for m in vc_msgs
        )

        # Build checkpoint from the most advanced node's logs
        checkpoint = {}
        for vc_msg in vc_msgs:
            if vc_msg.get("last_committed", 0) >= last_committed:
                checkpoint = {
                    "pre_prepare_log": vc_msg.get("pre_prepare_log", {}),
                    "prepare_log": vc_msg.get("prepare_log", {}),
                }

        nv_message = {
            "type": "new_view",
            "new_view": new_view,
            "sender": self._node_id,
            "signature": self._sign_view_message("new_view", self._node_id, new_view),
            "checkpoint": checkpoint,
        }

        self._new_view_messages[new_view] = nv_message
        self._view = new_view
        self._view_change_in_progress = False

        # Process locally
        self.handle_new_view(nv_message)

        logger.info(
            f"New view broadcast: view={new_view}, "
            f"last_committed={last_committed}"
        )

    def suspect_primary(self) -> None:
        """Mark the current primary as suspect and trigger view change."""
        current = self.primary
        if current == self._node_id:
            logger.warning("Primary cannot suspect itself")
            return

        self._node_status[current] = NodeStatus.SUSPECT
        logger.warning(f"Primary {current} suspected faulty")
        self.start_view_change()

    def recover_node(self, node_id: str) -> None:
        """Mark a node as recovered after failure."""
        if node_id in self._node_status:
            self._node_status[node_id] = NodeStatus.ACTIVE
            logger.info(f"Node {node_id} recovered")

    def mark_faulty(self, node_id: str) -> None:
        """Mark a node as confirmed faulty."""
        if node_id in self._node_status:
            self._node_status[node_id] = NodeStatus.FAULTY
            logger.warning(f"Node {node_id} marked FAULTY")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _check_sequence_watermark(self, sequence: int) -> bool:
        """Check whether a sequence number is within the watermark window.

        Sequences below the low watermark are garbage-collected.
        Sequences above high watermark are rejected until GC advances.
        """
        low = self._low_watermark
        high = low + self._watermark_window
        return low <= sequence <= high

    def _advance_watermark(self) -> None:
        """Advance the low watermark based on the oldest committed sequence."""
        if not self._committed_sequences:
            return
        oldest = min(self._committed_sequences)
        # Prune logs for sequences below the new watermark
        new_watermark = oldest
        for seq in list(self._pre_prepare_log.keys()):
            if seq < new_watermark:
                del self._pre_prepare_log[seq]
        for seq in list(self._prepare_log.keys()):
            if seq < new_watermark:
                del self._prepare_log[seq]
        for seq in list(self._commit_log.keys()):
            if seq < new_watermark:
                del self._commit_log[seq]
        self._low_watermark = new_watermark

    def public_key(self) -> str:
        """Return this node's serialized Ed25519 public key for distribution.

        Peers must call :meth:`register_public_key` with this value before
        they will accept messages signed by this node.
        """
        return _encode_public_key(self._signing_key.public_key())

    def register_public_key(
        self,
        node_id: str,
        public_key: ed25519.Ed25519PublicKey | bytes | str,
    ) -> None:
        """Register a peer's Ed25519 public key so its messages can be verified.

        Accepts an ``Ed25519PublicKey`` object, raw 32-byte key bytes, or the
        base64 string produced by :meth:`public_key` / :func:`_encode_public_key`.
        """
        self._public_keys[node_id] = _decode_public_key(public_key)

    def register_public_keys(
        self,
        mapping: dict[str, ed25519.Ed25519PublicKey | bytes | str],
    ) -> None:
        """Register public keys for multiple peers at once."""
        for nid, key in mapping.items():
            self.register_public_key(nid, key)

    def _sign_message(self, digest: str, view: int | None = None, sequence: int | None = None) -> str:
        """Sign a message's canonical payload with this node's Ed25519 key.

        The signature binds ``sender:view:sequence:digest`` so it cannot be
        replayed across views or sequence numbers.
        """
        view = self._view if view is None else view
        sequence = 0 if sequence is None else sequence
        payload = f"{self._node_id}:{view}:{sequence}:{digest}"
        return _sign_bytes(self._signing_key, payload)

    def verify_signature(self, msg: PBFTMessage, sender: str | None = None) -> bool:
        """Verify a message's Ed25519 signature.

        Args:
            msg: The message to verify.
            sender: Expected sender (defaults to ``msg.sender``).

        Returns:
            ``True`` only if the sender's public key is registered and the
            signature verifies over ``sender:view:sequence:digest``. Fail
            closed: unknown senders and unsigned messages are rejected.
        """
        if msg.signature is None:
            return False
        sender = sender or msg.sender
        pub = self._public_keys.get(sender)
        if pub is None:
            return False
        payload = f"{sender}:{msg.view}:{msg.sequence}:{msg.digest}"
        return _verify_bytes(pub, payload, msg.signature)

    def _sign_view_message(self, kind: str, node_id: str, new_view: int, extra: int = 0) -> str:
        """Sign a view-change / new-view dict payload with this node's key."""
        payload = f"{kind}:{node_id}:{new_view}:{extra}"
        return _sign_bytes(self._signing_key, payload)

    def _verify_view_message(self, kind: str, message: dict[str, Any]) -> bool:
        """Verify a view-change / new-view dict against its signature."""
        sender = message.get("node_id") or message.get("sender")
        signature = message.get("signature")
        if not sender or not signature:
            return False
        pub = self._public_keys.get(sender)
        if pub is None:
            return False
        new_view = message.get("new_view", 0)
        extra = message.get("last_committed", 0) if kind == "view_change" else 0
        payload = f"{kind}:{sender}:{new_view}:{extra}"
        return _verify_bytes(pub, payload, signature)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_logs(self) -> dict[str, Any]:
        """Return diagnostic information about the current log state."""
        return {
            "node_id": self._node_id,
            "view": self._view,
            "primary": self.primary,
            "is_primary": self.is_primary,
            "f": self._f,
            "quorum_size": self.quorum_size,
            "total_nodes": self._total_nodes,
            "last_sequence": self._last_sequence,
            "low_watermark": self._low_watermark,
            "pre_prepare_count": len(self._pre_prepare_log),
            "prepare_count": sum(len(v) for v in self._prepare_log.values()),
            "commit_count": sum(len(v) for v in self._commit_log.values()),
            "committed_ops": self._committed_ops,
            "committed_sequences": sorted(self._committed_sequences),
            "view_change_in_progress": self._view_change_in_progress,
            "node_status": {k: v.name for k, v in self._node_status.items()},
        }

    def get_committed_operations(self) -> list[dict[str, Any]]:
        """Return all committed operations in order."""
        ops: list[dict[str, Any]] = []
        for seq in sorted(self._committed_sequences):
            pp = self._pre_prepare_log.get(seq)
            if pp and pp.operation:
                ops.append({"sequence": seq, "operation": pp.operation})
        return ops


# ===========================================================================
# QuorumManager -- Quorum-based decision making
# ===========================================================================


class FaultModel(Enum):
    """Supported fault models for quorum computation."""

    PBFT = auto()  # Byzantine faults: quorum = 2f + 1
    CRASH_ONLY = auto()  # Crash faults: quorum = f + 1


@dataclass
class QuorumResult:
    """Result of a quorum collection attempt.

    Attributes:
        success: Whether the quorum succeeded.
        responses: The collected responses.
        faulty_nodes: Nodes that failed or returned invalid responses.
        elapsed: Wall-clock time for the collection.
    """

    success: bool
    responses: list[Any]
    faulty_nodes: list[str]
    elapsed: float


class QuorumManager:
    """Manages quorum-based decision making across a distributed system.

    Computes quorum sizes for different fault models and handles response
    collection with timeout. Supports both PBFT Byzantine quorums
    (``2f + 1`` of ``3f + 1`` total) and crash-only quorums (``f + 1`` of
    ``2f + 1`` total).

    Args:
        total_nodes: Total number of nodes in the cluster.
        f: Maximum number of faulty nodes to tolerate.
        fault_model: The fault model to use (PBFT or CRASH_ONLY).
    """

    def __init__(
        self,
        total_nodes: int,
        f: int | None = None,
        fault_model: FaultModel = FaultModel.PBFT,
    ) -> None:
        self._total = total_nodes
        self._f = (total_nodes - 1) // 3 if f is None else f
        self._model = fault_model

    # ------------------------------------------------------------------ #
    # Property accessors
    # ------------------------------------------------------------------ #

    @property
    def quorum_size(self) -> int:
        """Compute the required quorum size for the configured fault model.

        For PBFT: ``2f + 1`` (majority in a Byzantine environment).
        For CRASH_ONLY: ``f + 1`` (simple majority for crash tolerance).
        """
        if self._model == FaultModel.PBFT:
            return 2 * self._f + 1
        return self._f + 1

    @property
    def maximum_fault_tolerance(self) -> int:
        """Maximum number of simultaneous failures this configuration can tolerate."""
        if self._model == FaultModel.PBFT:
            return self._f
        # Crash-only can tolerate f failures out of 2f + 1
        return self._total - self.quorum_size

    @property
    def total_nodes(self) -> int:
        return self._total

    @property
    def f(self) -> int:
        return self._f

    # ------------------------------------------------------------------ #
    # Quorum checking
    # ------------------------------------------------------------------ #

    def check_quorum(self, responses: dict[str, Any]) -> bool:
        """Check whether a set of responses meets the quorum requirement.

        Accepts a dict of ``{node_id: response}``. Responses that are
        ``None`` are treated as faulty/absent. Returns ``True`` if the
        number of valid, matching responses meets the quorum threshold.

        Args:
            responses: Dict mapping node IDs to their responses. ``None``
                responses indicate no response or faulty node.

        Returns:
            ``True`` if enough matched responses exist for a quorum.
        """
        valid = {nid: resp for nid, resp in responses.items() if resp is not None}

        if len(valid) < self.quorum_size:
            logger.debug(
                f"Quorum check failed: {len(valid)} valid responses, "
                f"need {self.quorum_size}"
            )
            return False

        # Count agreement by response content (serialized to stable hash)
        agreement: dict[str, list[str]] = {}
        for nid, resp in valid.items():
            key = self._response_digest(resp)
            agreement.setdefault(key, []).append(nid)

        max_agree = max(len(nodes) for nodes in agreement.values())
        if max_agree >= self.quorum_size:
            logger.debug(
                f"Quorum reached: {max_agree} nodes agree, "
                f"need {self.quorum_size}"
            )
            return True

        logger.debug(
            f"Quorum check failed: most-agreed={max_agree}, "
            f"need {self.quorum_size}"
        )
        return False

    def check_consensus_quorum(
        self, votes: dict[str, Any]
    ) -> tuple[bool, Any]:
        """Check a consensus-style quorum where all nodes must agree.

        Returns a tuple of ``(quorum_reached, agreed_value)``. The
        ``agreed_value`` is ``None`` if no clear consensus exists.

        Args:
            votes: Dict mapping ``{node_id: vote_value}``.

        Returns:
            Tuple of ``(quorum_reached, agreed_value)``.
        """
        valid = {nid: v for nid, v in votes.items() if v is not None}
        if len(valid) < self.quorum_size:
            return False, None

        # Find the value with the most votes
        counter: Counter[Any] = Counter(
            self._response_digest(v) for v in valid.values()
        )
        if not counter:
            return False, None

        most_common = counter.most_common(1)[0]
        if most_common[1] >= self.quorum_size:
            # Map digest back to original value
            for nid, val in valid.items():
                if self._response_digest(val) == most_common[0]:
                    return True, val

        return False, None

    # ------------------------------------------------------------------ #
    # Response collection
    # ------------------------------------------------------------------ #

    async def collect_quorum(
        self,
        peers: list[Any],
        request: Any,
        timeout: float = 10.0,
        response_parser: Callable[[Any], Any] = lambda r: r,
    ) -> QuorumResult:
        """Collect responses from peers until quorum is reached or timeout.

        Sends the request to all peers and collects responses. Returns as
        soon as a quorum of matching responses is received, or when all
        peers have responded, or on timeout.

        Args:
            peers: List of peer objects, each with a ``send(request)`` method
                or ``node_id`` attribute.
            request: The request to send to each peer.
            timeout: Maximum wall-clock time to wait for responses (seconds).
            response_parser: Optional callable to transform each raw response.

        Returns:
            A ``QuorumResult`` with the collected responses and metadata.
        """
        import asyncio

        start = time.monotonic()
        responses: dict[str, Any] = {}
        faulty: list[str] = []

        async def query_peer(peer: Any) -> None:
            peer_id = getattr(peer, "node_id", str(id(peer)))
            try:
                if asyncio.iscoroutinefunction(peer.send):
                    raw = await peer.send(request)
                else:
                    raw = peer.send(request)
                resp = response_parser(raw)
                responses[peer_id] = resp

                # Early check: if we already have quorum, stop
                if self.check_quorum(responses):
                    return
            except Exception as exc:
                logger.debug(f"Peer {peer_id} failed: {exc}")
                faulty.append(peer_id)
                responses[peer_id] = None

        tasks = [query_peer(p) for p in peers]
        await asyncio.wait(
            [asyncio.create_task(t) for t in tasks],
            timeout=timeout,
        )

        elapsed = time.monotonic() - start
        success = self.check_quorum(responses)

        peer_responses: list[Any] = [
            r for r in responses.values() if r is not None
        ]

        return QuorumResult(
            success=success,
            responses=peer_responses,
            faulty_nodes=faulty,
            elapsed=elapsed,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _response_digest(response: Any) -> str:
        """Create a stable hash of a response for agreement checking."""
        if response is None:
            return ""
        if isinstance(response, (str, bytes, int, float, bool)):
            return str(response)
        try:
            serialized = json.dumps(response, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()
        except (TypeError, ValueError):
            return str(id(response))

    @staticmethod
    def required_total_for_f(f: int) -> int:
        """Calculate the minimum total nodes needed to tolerate ``f`` Byzantine faults.

        Args:
            f: Number of faulty nodes to tolerate.

        Returns:
            Minimum total nodes (``3f + 1``).
        """
        return 3 * f + 1

    @staticmethod
    def max_byzantine_faults(total_nodes: int) -> int:
        """Calculate the maximum number of Byzantine faults tolerable.

        Args:
            total_nodes: Total number of nodes.

        Returns:
            Maximum ``f`` such that ``3f + 1 <= total_nodes``.
        """
        return (total_nodes - 1) // 3


# ===========================================================================
# SplitBrainDetector -- Vector-clock-based conflict detection
# ===========================================================================


@dataclass
class VectorClock:
    """A vector clock mapping node IDs to logical timestamps.

    Each node maintains its own entry in the clock. When a node performs
    an operation, it increments its own timestamp.

    Attributes:
        timestamps: Dict mapping ``{node_id: logical_timestamp}``.
    """

    timestamps: dict[str, int] = field(default_factory=dict)


@dataclass
class Conflict:
    """A detected conflict between two nodes' states.

    Attributes:
        node_a: First conflicting node ID.
        node_b: Second conflicting node ID.
        key: The key or attribute that differs.
        value_a: Value on node_a.
        value_b: Value on node_b.
        reason: Human-readable explanation of the conflict.
    """

    node_a: str
    node_b: str
    key: str
    value_a: Any
    value_b: Any
    reason: str = ""


@dataclass
class ResolutionResult:
    """Result of a conflict resolution operation.

    Attributes:
        merged_state: The merged state dict after resolution.
        conflicts_resolved: Number of conflicts that were resolved.
        remaining_conflicts: Number of conflicts that could not be auto-resolved.
        log: Human-readable log of resolution steps.
    """

    merged_state: dict[str, Any]
    conflicts_resolved: int = 0
    remaining_conflicts: int = 0
    log: list[str] = field(default_factory=list)


class SplitBrainDetector:
    """Detects and resolves split-brain conditions using vector clocks.

    Each node maintains a vector clock that tracks logical time across
    all nodes. On partition heal, clocks are compared to detect causal
    conflicts (concurrent updates to the same key without knowledge of
    each other).

    Resolution strategies:
    - **Last-writer-wins** using vector clock ordering when causal
      relationship exists.
    - **Merge with explicit conflict markers** for concurrent updates.
    - **Custom resolution** via registered callbacks for domain-specific
      merge logic.

    Args:
        node_id: This detector's node identifier.
    """

    def __init__(self, node_id: str | None = None) -> None:
        self._node_id: str | None = node_id
        self._clocks: dict[str, VectorClock] = {}
        self._state: dict[str, Any] = {}
        self._conflict_history: list[Conflict] = []

    # ------------------------------------------------------------------ #
    # Vector clock operations
    # ------------------------------------------------------------------ #

    def get_clock(self, node_id: str) -> VectorClock:
        """Get or create a vector clock for a node."""
        if node_id not in self._clocks:
            self._clocks[node_id] = VectorClock(timestamps={})
        return self._clocks[node_id]

    def update(self, node_id: str, key: str, value: Any = None) -> None:
        """Record an update from a node on a key.

        Increments the node's own timestamp in its vector clock.

        Args:
            node_id: The node that performed the update.
            key: The state key that was updated.
            value: Optional value to associate with the key.
        """
        clock = self.get_clock(node_id)
        clock.timestamps[node_id] = clock.timestamps.get(node_id, 0) + 1

        # Store the value
        if value is not None:
            # Track per-node value history for conflict detection
            state_key = f"{key}:{node_id}"
            self._state[state_key] = {
                "value": value,
                "timestamp": clock.timestamps[node_id],
                "clock": dict(clock.timestamps),
            }

        logger.debug(f"Clock updated: {node_id} ++ ({key})")

    def update_all(self, node_id: str, state: dict[str, Any]) -> None:
        """Record multiple state updates from a node at once.

        Args:
            node_id: The node that performed the updates.
            state: Dict of ``{key: value}`` pairs.
        """
        for key, value in state.items():
            self.update(node_id, key, value)

    # ------------------------------------------------------------------ #
    # Clock comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_causally_before(
        clock_a: VectorClock, clock_b: VectorClock
    ) -> bool:
        """Check if ``clock_a`` is strictly before ``clock_b`` (causal order).

        ``clock_a`` is before ``clock_b`` if every node's timestamp in
        ``clock_a`` is less than or equal to its counterpart in ``clock_b``,
        and at least one is strictly less.

        Args:
            clock_a: First vector clock.
            clock_b: Second vector clock.

        Returns:
            ``True`` if ``clock_a`` happened before ``clock_b``.
        """
        all_keys = set(clock_a.timestamps) | set(clock_b.timestamps)
        has_strictly_less = False

        for key in sorted(all_keys):
            a_ts = clock_a.timestamps.get(key, 0)
            b_ts = clock_b.timestamps.get(key, 0)
            if a_ts > b_ts:
                return False
            if a_ts < b_ts:
                has_strictly_less = True

        return has_strictly_less

    @staticmethod
    def is_concurrent(
        clock_a: VectorClock, clock_b: VectorClock
    ) -> bool:
        """Check if two clocks are concurrent (conflict detected).

        Two clocks are concurrent if neither is causally before the other:
        there exist nodes where ``a > b`` and other nodes where ``b > a``.

        Args:
            clock_a: First vector clock.
            clock_b: Second vector clock.

        Returns:
            ``True`` if the clocks represent concurrent updates.
        """
        return (
            not SplitBrainDetector.is_causally_before(clock_a, clock_b)
            and not SplitBrainDetector.is_causally_before(clock_b, clock_a)
        )

    @staticmethod
    def merge_clocks(
        clock_a: VectorClock, clock_b: VectorClock
    ) -> VectorClock:
        """Merge two vector clocks by taking the maximum timestamp per node.

        Args:
            clock_a: First vector clock.
            clock_b: Second vector clock.

        Returns:
            A new VectorClock with per-element maxima.
        """
        all_keys = set(clock_a.timestamps) | set(clock_b.timestamps)
        merged = {
            key: max(
                clock_a.timestamps.get(key, 0),
                clock_b.timestamps.get(key, 0),
            )
            for key in all_keys
        }
        return VectorClock(timestamps=merged)

    # ------------------------------------------------------------------ #
    # Conflict detection
    # ------------------------------------------------------------------ #

    def detect_conflicts(
        self,
        local_clocks: dict[str, VectorClock],
        peer_clocks: dict[str, VectorClock],
    ) -> list[Conflict]:
        """Compare local and peer clocks to detect split-brain conflicts.

        For each node that appears in both clock sets, checks whether the
        clocks are concurrent. Concurrent clocks indicate that the two
        partitions independently performed updates that may conflict.

        Args:
            local_clocks: Dict of ``{node_id: VectorClock}`` from the local
                partition.
            peer_clocks: Dict of ``{node_id: VectorClock}`` from the remote
                partition being compared against.

        Returns:
            List of detected ``Conflict`` objects.
        """
        conflicts: list[Conflict] = []
        common_nodes = set(local_clocks) & set(peer_clocks)

        for nid in sorted(common_nodes):
            local = local_clocks[nid]
            peer = peer_clocks[nid]

            if self.is_concurrent(local, peer):
                # Determine which keys diverged
                diverged_keys = set(local.timestamps) | set(peer.timestamps)
                for key in sorted(diverged_keys):
                    lv = local.timestamps.get(key, 0)
                    pv = peer.timestamps.get(key, 0)
                    if lv != pv:
                        conflict = Conflict(
                            node_a=self._node_id or "local",
                            node_b=nid,
                            key=key,
                            value_a=lv,
                            value_b=pv,
                            reason=(
                                f"Concurrent updates on node {nid}: "
                                f"local ts={lv}, peer ts={pv}"
                            ),
                        )
                        conflicts.append(conflict)

        # Merge new clocks into our tracking
        all_nodes = set(local_clocks) | set(peer_clocks)
        for nid in all_nodes:
            local = local_clocks.get(nid, VectorClock())
            peer = peer_clocks.get(nid, VectorClock())
            merged = self.merge_clocks(local, peer)
            self._clocks[nid] = merged

        self._conflict_history.extend(conflicts)

        return conflicts

    def detect_partition_heal(
        self,
        local_node_ids: list[str],
        peer_node_ids: list[str],
    ) -> list[Conflict]:
        """Detect conflicts that arose during a network partition.

        Called when a partition heals and nodes from two partitions reconnect.
        Compares the clock state for each node pair.

        Args:
            local_node_ids: Node IDs from the local partition.
            peer_node_ids: Node IDs from the peer partition.

        Returns:
            List of conflicts detected between the partitions.
        """
        local_clocks = {
            nid: self.get_clock(nid)
            for nid in local_node_ids
        }
        peer_clocks = {
            nid: self.get_clock(nid)
            for nid in peer_node_ids
        }
        return self.detect_conflicts(local_clocks, peer_clocks)

    # ------------------------------------------------------------------ #
    # Conflict resolution
    # ------------------------------------------------------------------ #

    def resolve_conflict(
        self,
        local_state: dict[str, Any],
        peer_state: dict[str, Any],
        peer_id: str,
        resolution_strategy: str = "lww",
        custom_resolver: Callable[[str, Any, Any], Any] | None = None,
    ) -> ResolutionResult:
        """Resolve conflicts between local and peer state.

        Resolution strategies:

        - ``"lww"``: Last-writer-wins using vector clock timestamps.
        - ``"merge"``: Merge dictionaries recursively; conflicting values
          are wrapped in a ``ConflictMarker``.
        - ``"peer_wins"``: Always prefer the peer's value.
        - ``"local_wins"``: Always prefer the local value.
        - ``"custom"``: Use the provided ``custom_resolver`` callable.

        Args:
            local_state: The local state dict.
            peer_state: The peer state dict.
            peer_id: Node ID of the peer.
            resolution_strategy: One of ``"lww"``, ``"merge"``,
                ``"peer_wins"``, ``"local_wins"``, ``"custom"``.
            custom_resolver: Callable ``(key, local_val, peer_val) -> resolved_val``.
                Required when ``resolution_strategy="custom"``.

        Returns:
            A ``ResolutionResult`` describing the merged state and conflicts.
        """
        merged: dict[str, Any] = {}
        resolved_count = 0
        remaining = 0
        log: list[str] = []

        all_keys = set(local_state) | set(peer_state)

        for key in sorted(all_keys):
            local_val = local_state.get(key)
            peer_val = peer_state.get(key)

            if local_val == peer_val:
                merged[key] = local_val
                continue

            # There's a conflict
            if resolution_strategy == "lww":
                resolved = self._resolve_lww(key, local_val, peer_val, peer_id)
                merged[key] = resolved
                resolved_count += 1
                log.append(f"lww({key}): chose {resolved!r} over "
                           f"local={local_val!r}, peer={peer_val!r}")

            elif resolution_strategy == "merge":
                result = self._resolve_merge(key, local_val, peer_val)
                if result["resolved"]:
                    merged[key] = result["value"]
                    resolved_count += 1
                    log.append(f"merge({key}): {result['value']!r}")
                else:
                    merged[key] = ConflictMarker(
                        local=local_val, peer=peer_val
                    )
                    remaining += 1
                    log.append(f"merge({key}): CONFLICT - cannot auto-merge "
                               f"local={local_val!r}, peer={peer_val!r}")

            elif resolution_strategy == "peer_wins":
                merged[key] = peer_val
                resolved_count += 1
                log.append(f"peer_wins({key}): chose peer value {peer_val!r}")

            elif resolution_strategy == "local_wins":
                merged[key] = local_val
                resolved_count += 1
                log.append(f"local_wins({key}): kept local value {local_val!r}")

            elif resolution_strategy == "custom":
                if custom_resolver is None:
                    raise ValueError(
                        "custom_resolver required for strategy='custom'"
                    )
                resolved_val = custom_resolver(key, local_val, peer_val)
                merged[key] = resolved_val
                resolved_count += 1
                log.append(f"custom({key}): resolved to {resolved_val!r}")

            else:
                raise ValueError(
                    f"Unknown resolution strategy: {resolution_strategy!r}"
                )

        return ResolutionResult(
            merged_state=merged,
            conflicts_resolved=resolved_count,
            remaining_conflicts=remaining,
            log=log,
        )

    def _resolve_lww(
        self,
        key: str,
        local_val: Any,
        peer_val: Any,
        peer_id: str,
    ) -> Any:
        """Last-writer-wins resolution using vector clocks."""
        local_key = f"{key}:{self._node_id}" if self._node_id else f"{key}:local"
        peer_key = f"{key}:{peer_id}"

        local_entry = self._state.get(local_key, {})
        peer_entry = self._state.get(peer_key, {})

        local_ts = local_entry.get("timestamp", 0)
        peer_ts = peer_entry.get("timestamp", 0)

        if peer_ts > local_ts:
            return peer_val
        elif local_ts > peer_ts:
            return local_val
        # Same timestamp: prefer peer value for determinism
        return peer_val

    def _resolve_merge(
        self,
        key: str,
        local_val: Any,
        peer_val: Any,
    ) -> dict[str, Any]:
        """Attempt recursive merge of two values."""
        if isinstance(local_val, dict) and isinstance(peer_val, dict):
            merged: dict[str, Any] = {}
            sub_keys = set(local_val) | set(peer_val)
            all_resolved = True
            for sk in sorted(sub_keys):
                lv = local_val.get(sk)
                pv = peer_val.get(sk)
                if lv == pv:
                    merged[sk] = lv
                elif isinstance(lv, dict) and isinstance(pv, dict):
                    result = self._resolve_merge(sk, lv, pv)
                    if result["resolved"]:
                        merged[sk] = result["value"]
                    else:
                        merged[sk] = ConflictMarker(local=lv, peer=pv)
                        all_resolved = False
                else:
                    merged[sk] = ConflictMarker(local=lv, peer=pv)
                    all_resolved = False
            return {"value": merged, "resolved": all_resolved}

        if isinstance(local_val, list) and isinstance(peer_val, list):
            merged_list = local_val + [
                x for x in peer_val if x not in local_val
            ]
            return {"value": merged_list, "resolved": True}

        if local_val is None:
            return {"value": peer_val, "resolved": True}
        if peer_val is None:
            return {"value": local_val, "resolved": True}

        # Fall through: values are incompatible for auto-merge
        return {"value": None, "resolved": False}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_clocks(self) -> dict[str, dict[str, int]]:
        """Return a snapshot of all vector clocks."""
        return {
            nid: dict(clock.timestamps)
            for nid, clock in sorted(self._clocks.items())
        }

    def get_conflict_history(self) -> list[Conflict]:
        """Return the full conflict history."""
        return list(self._conflict_history)

    def clear_history(self) -> None:
        """Clear the accumulated conflict history."""
        self._conflict_history.clear()

    def state_summary(self) -> dict[str, Any]:
        """Return a summary of the detector's current state."""
        return {
            "node_id": self._node_id,
            "tracked_nodes": sorted(self._clocks.keys()),
            "state_entries": len(self._state),
            "conflict_history_count": len(self._conflict_history),
            "clocks": self.get_clocks(),
        }


class ConflictMarker:
    """Placeholder for an unresolvable conflict during merge.

    Attributes:
        local: The local side's value.
        peer: The peer side's value.
    """

    def __init__(self, local: Any, peer: Any) -> None:
        self.local = local
        self.peer = peer

    def __repr__(self) -> str:
        return f"ConflictMarker(local={self.local!r}, peer={self.peer!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConflictMarker):
            return self.local == other.local and self.peer == other.peer
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"__conflict__": True, "local": self.local, "peer": self.peer}


# ===========================================================================
# ByzantineCoordinator -- High-level BFT coordination
# ===========================================================================


@dataclass
class BFTStats:
    """Statistics for the Byzantine coordinator.

    Attributes:
        total_operations: Total operations submitted.
        committed_operations: Operations successfully committed via PBFT.
        failed_operations: Operations that failed consensus.
        pending_operations: Operations still in progress.
        view_changes: Number of view changes that have occurred.
        faulty_nodes: List of node IDs currently marked faulty.
        current_view: The current PBFT view number.
        uptime_seconds: Coordinator uptime in seconds.
    """

    total_operations: int = 0
    committed_operations: int = 0
    failed_operations: int = 0
    pending_operations: int = 0
    view_changes: int = 0
    faulty_nodes: list[str] = field(default_factory=list)
    current_view: int = 0
    uptime_seconds: float = 0.0


class ByzantineCoordinator:
    """High-level coordinator for Byzantine fault-tolerant operations.

    Provides a simplified interface for submitting operations through PBFT
    consensus, managing node membership, and querying the agreed state.

    The coordinator wraps a ``PBFTNode`` instance and optionally integrates
    a ``QuorumManager`` and ``SplitBrainDetector`` for comprehensive BFT
    coordination.

    Args:
        node_id: This coordinator's node ID.
        node_ids: All node IDs in the cluster.
        f: Maximum Byzantine faults (auto-computed from ``node_ids`` if not given).
        pbft_node: An optional pre-configured ``PBFTNode`` (otherwise created).
        quorum_manager: An optional pre-configured ``QuorumManager`` (otherwise
            created).
        split_brain_detector: An optional pre-configured ``SplitBrainDetector``
            (otherwise created).
        commit_callback: Callback invoked when an operation is committed.
    """

    def __init__(
        self,
        node_ids: list[str],
        node_id: str | None = None,
        f: int | None = None,
        pbft_node: PBFTNode | None = None,
        quorum_manager: QuorumManager | None = None,
        split_brain_detector: SplitBrainDetector | None = None,
        commit_callback: Callable[[dict[str, Any]], None] | None = None,
        signing_key: ed25519.Ed25519PrivateKey | None = None,
        node_public_keys: dict[
            str, ed25519.Ed25519PublicKey | bytes | str
        ] | None = None,
    ) -> None:
        self._node_ids = sorted(node_ids)
        self._total_nodes = len(self._node_ids)
        self._f = f if f is not None else (self._total_nodes - 1) // 3
        self._start_time = time.monotonic()

        # Use provided instances or create defaults
        if node_id is None and pbft_node is not None:
            node_id = pbft_node.node_id
        elif node_id is None and node_ids:
            node_id = node_ids[0]

        self._node_id = node_id or "coordinator"

        self._pbft = pbft_node or PBFTNode(
            node_id=self._node_id,
            node_ids=self._node_ids,
            f=self._f,
            callback=commit_callback or self._default_commit_callback,
            signing_key=signing_key,
            node_public_keys=node_public_keys,
        )

        self._qm = quorum_manager or QuorumManager(
            total_nodes=self._total_nodes,
            f=self._f,
            fault_model=FaultModel.PBFT,
        )

        self._sbd = split_brain_detector or SplitBrainDetector(
            node_id=self._node_id,
        )

        # Coordinator-level state
        self._agreed_state: dict[str, Any] = {}
        self._pending_ops: dict[int, dict[str, Any]] = {}
        self._committed_ops: list[dict[str, Any]] = []
        self._failed_ops: list[dict[str, Any]] = []
        self._view_changes: int = 0
        self._total_ops: int = 0
        self._peer_nodes: dict[str, Any] = {}

        # Commit callback state for our default
        self._commit_callback = commit_callback

        logger.info(
            f"ByzantineCoordinator {self._node_id}: {self._total_nodes} nodes, "
            f"f={self._f}, pbft_node={self._pbft.node_id}"
        )

    # ------------------------------------------------------------------ #
    # Operation submission
    # ------------------------------------------------------------------ #

    async def submit(
        self,
        operation: dict[str, Any],
        required_ack: int | None = None,
    ) -> bool:
        """Submit an operation through PBFT consensus.

        If this node is the PBFT primary, initiates consensus directly.
        Otherwise, the request is forwarded to the primary.

        Args:
            operation: The operation dict to submit for consensus.
            required_ack: Minimum acknowledgments required (defaults to
                the PBFT quorum size ``2f + 1``).

        Returns:
            ``True`` if the operation was successfully committed.
        """
        self._total_ops += 1
        op_id = self._total_ops

        self._pending_ops[op_id] = {
            "operation": operation,
            "submitted_at": time.monotonic(),
        }

        if required_ack is None:
            required_ack = self._qm.quorum_size

        logger.debug(
            f"Submitting operation {op_id}: {operation.get('op', 'unknown')}, "
            f"required_ack={required_ack}"
        )

        # Phase 1: Pre-prepare
        pre_prepare = self._pbft.handle_request(operation)
        if pre_prepare is None and not self._pbft.is_primary:
            # Non-primary: we need to wait for the PBFT protocol to progress
            # In a full implementation this would poll or subscribe
            logger.debug(
                f"Non-primary; forwarded op {op_id} to {self._pbft.primary}"
            )
            # Mark as pending until committed through callback
            return False

        # Phase 2-3 are handled asynchronously through the prepare/commit flow.
        # The callback will mark operations as committed.
        return True

    def _default_commit_callback(self, operation: dict[str, Any]) -> None:
        """Default callback when an operation is committed."""
        # Merge the operation into agreed state
        if isinstance(operation, dict):
            op_type = operation.get("op", operation.get("type", "unknown"))
            self._agreed_state[f"last_{op_type}"] = operation
            self._agreed_state["last_operation"] = operation
            self._agreed_state["timestamp"] = time.time()

        self._committed_ops.append(operation)
        self._pbft._committed_ops = len(self._committed_ops)

        # Update pending ops tracking
        for op_id, pending in list(self._pending_ops.items()):
            if pending.get("operation") == operation:
                self._pending_ops.pop(op_id, None)
                break

        logger.debug(f"Operation committed via callback: {str(operation)[:80]}")

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        """Return the current agreed state.

        The agreed state is built from all committed operations and
        represents the single source of truth across the cluster.

        Returns:
            Dict containing the current agreed state.
        """
        return dict(self._agreed_state)

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Return the full history of committed operations."""
        return list(self._committed_ops)

    # ------------------------------------------------------------------ #
    # Peer management
    # ------------------------------------------------------------------ #

    def register_peer(self, peer_id: str, peer_handle: Any = None) -> None:
        """Register a peer node for message forwarding.

        Args:
            peer_id: Node ID of the peer.
            peer_handle: Optional handle/connection to the peer.
        """
        self._peer_nodes[peer_id] = peer_handle or {}
        logger.debug(f"Peer registered: {peer_id}")

    def unregister_peer(self, peer_id: str) -> None:
        """Remove a peer registration.

        Args:
            peer_id: Node ID of the peer to remove.
        """
        self._peer_nodes.pop(peer_id, None)
        logger.debug(f"Peer unregistered: {peer_id}")

    def get_peers(self) -> list[str]:
        """Return the list of registered peer node IDs."""
        return list(self._peer_nodes.keys())

    # ------------------------------------------------------------------ #
    # View management
    # ------------------------------------------------------------------ #

    def trigger_view_change(self) -> None:
        """Manually trigger a PBFT view change.

        Increments the view change counter and delegates to the PBFT node.
        """
        self._pbft.suspect_primary()
        self._view_changes += 1
        logger.info(f"View change triggered: total={self._view_changes}")

    def get_current_view(self) -> int:
        """Return the current PBFT view number."""
        return self._pbft.view

    def get_primary(self) -> str:
        """Return the current primary node ID."""
        return self._pbft.primary

    def is_primary(self) -> bool:
        """Check if this coordinator is the current primary."""
        return self._pbft.is_primary

    # ------------------------------------------------------------------ #
    # Split-brain detection
    # ------------------------------------------------------------------ #

    def record_state_update(self, key: str, value: Any) -> None:
        """Record a state update for split-brain tracking.

        Args:
            key: The state key being updated.
            value: The new value.
        """
        self._sbd.update(self._node_id, key, value)

    def check_split_brain(
        self,
        peer_clocks: dict[str, VectorClock],
    ) -> list[Conflict]:
        """Check for split-brain conflicts with a peer's clocks.

        Args:
            peer_clocks: Vector clocks from the peer partition.

        Returns:
            List of detected conflicts.
        """
        local_clocks = self._sbd.get_clocks()
        reconciled_clocks = {
            nid: VectorClock(timestamps=ts)
            for nid, ts in local_clocks.items()
        }
        return self._sbd.detect_conflicts(reconciled_clocks, peer_clocks)

    def resolve_partition_conflict(
        self,
        local_state: dict[str, Any],
        peer_state: dict[str, Any],
        peer_id: str,
        strategy: str = "lww",
    ) -> ResolutionResult:
        """Resolve state conflicts after a partition heal.

        Args:
            local_state: Local partition state.
            peer_state: Peer partition state.
            peer_id: The peer node ID.
            strategy: Resolution strategy (see ``SplitBrainDetector.resolve_conflict``).

        Returns:
            Resolution result with merged state.
        """
        result = self._sbd.resolve_conflict(
            local_state, peer_state, peer_id, resolution_strategy=strategy,
        )
        if result.conflicts_resolved > 0 or result.remaining_conflicts == 0:
            self._agreed_state = dict(result.merged_state)
        return result

    # ------------------------------------------------------------------ #
    # Fault management
    # ------------------------------------------------------------------ #

    def report_faulty_node(self, node_id: str) -> None:
        """Report a node as faulty.

        Args:
            node_id: Node ID to mark as faulty.
        """
        self._pbft.mark_faulty(node_id)
        logger.warning(f"Node {node_id} reported as faulty")

    def recover_node(self, node_id: str) -> None:
        """Mark a previously faulty node as recovered.

        Args:
            node_id: Node ID to mark as recovered.
        """
        self._pbft.recover_node(node_id)
        logger.info(f"Node {node_id} recovered")

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> BFTStats:
        """Return BFT statistics.

        Returns:
            A ``BFTStats`` dataclass with current metrics.
        """
        return BFTStats(
            total_operations=self._total_ops,
            committed_operations=len(self._committed_ops),
            failed_operations=len(self._failed_ops),
            pending_operations=len(self._pending_ops),
            view_changes=self._view_changes,
            faulty_nodes=[
                nid
                for nid, status in self._pbft._node_status.items()
                if status in (NodeStatus.FAULTY, NodeStatus.SUSPECT)
            ],
            current_view=self._pbft.view,
            uptime_seconds=time.monotonic() - self._start_time,
        )

    # ------------------------------------------------------------------ #
    # Serialization / snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of the coordinator's full state.

        Useful for checkpointing and state transfer.

        Returns:
            A serializable dict of the coordinator state.
        """
        return {
            "node_id": self._node_id,
            "view": self._pbft.view,
            "agreed_state": dict(self._agreed_state),
            "committed_ops": list(self._committed_ops),
            "failed_ops": list(self._failed_ops),
            "total_ops": self._total_ops,
            "view_changes": self._view_changes,
            "faulty_nodes": [
                nid
                for nid, status in self._pbft._node_status.items()
                if status == NodeStatus.FAULTY
            ],
            "clocks": self._sbd.get_clocks(),
        }

    def restore_snapshot(self, data: dict[str, Any]) -> None:
        """Restore coordinator state from a snapshot.

        Args:
            data: Snapshot dict previously returned by ``snapshot()``.
        """
        self._agreed_state = dict(data.get("agreed_state", {}))
        self._committed_ops = list(data.get("committed_ops", []))
        self._failed_ops = list(data.get("failed_ops", []))
        self._total_ops = data.get("total_ops", 0)
        self._view_changes = data.get("view_changes", 0)
        logger.info(
            f"Snapshot restored: view={data.get('view')}, "
            f"committed={len(self._committed_ops)}"
        )
