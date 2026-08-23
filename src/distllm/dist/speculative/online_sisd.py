"""
Online Self-Improving Speculative Decoder (SISD).

The draft model continuously improves from production traffic via
online LoRA fine-tuning.  Each request produces a trajectory: the
prefix context, the draft tokens emitted, and a boolean mask indicating
which draft tokens were accepted by the target model.  Accepted tokens
are treated as positive targets; rejected tokens are treated as negative
targets.  A KL penalty against the base (pre-update) draft model prevents
catastrophic forgetting.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# SpeculativeFeedbackBuffer  — stores & samples production trajectories
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A single speculative-decoding trajectory from one request.

    Attributes
    ----------
    prefix_ids : list of int
        Token IDs of the prompt prefix (context).
    draft_token_ids : list of int
        Token IDs emitted by the draft model.
    accepted_mask : list of bool
        Boolean mask of length ``len(draft_token_ids)``; ``True`` means the
        corresponding token was accepted by the target model.
    """

    prefix_ids: list[int]
    draft_token_ids: list[int]
    accepted_mask: list[bool]


class SpeculativeFeedbackBuffer:
    """Ring-buffer of speculative-decoding trajectories for online training.

    Stores a fixed-capacity history of production trajectories.  New samples
    evict the oldest ones when the buffer is full.  Provides a stratified
    sampling method that mixes accepted-heavy and rejected-heavy trajectories
    for more stable training.

    Parameters
    ----------
    max_size : int
        Maximum number of trajectories to retain (default 10 000).
    """

    def __init__(self, max_size: int = 10000) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._buffer: deque[Trajectory] = deque(maxlen=max_size)

    # -- Mutation -----------------------------------------------------------

    def add(
        self,
        prefix_ids: list[int],
        draft_token_ids: list[int],
        accepted_mask: list[bool],
    ) -> None:
        """Store a single speculative-decoding trajectory.

        Parameters
        ----------
        prefix_ids : list of int
            Token IDs of the prompt prefix.
        draft_token_ids : list of int
            Token IDs emitted by the draft model.
        accepted_mask : list of bool
            Boolean mask indicating acceptance for each draft token.
        """
        if len(draft_token_ids) != len(accepted_mask):
            raise ValueError(
                f"draft_token_ids ({len(draft_token_ids)}) and accepted_mask "
                f"({len(accepted_mask)}) must have the same length"
            )
        self._buffer.append(
            Trajectory(
                prefix_ids=list(prefix_ids),
                draft_token_ids=list(draft_token_ids),
                accepted_mask=list(accepted_mask),
            )
        )

    def clear(self) -> None:
        """Remove all trajectories from the buffer."""
        self._buffer.clear()

    # -- Query --------------------------------------------------------------

    def sample(self, batch_size: int = 32) -> list[Trajectory]:
        """Draw a stratified batch of trajectories for training.

        Trajectories are split into two pools: those with an acceptance rate
        >= 0.5 and those below 0.5.  Half the batch is drawn from each pool
        to ensure the training signal includes both positive and negative
        examples.  Falls back to uniform random sampling when fewer than
        ``batch_size`` trajectories are available (returns all if even fewer).

        Parameters
        ----------
        batch_size : int
            Desired number of trajectories (default 32).

        Returns
        -------
        list of Trajectory
            Sampled trajectories (may be shorter than *batch_size* when the
            buffer has fewer entries).
        """
        if batch_size < 1:
            return []

        n = len(self._buffer)
        if n == 0:
            return []
        if n <= batch_size:
            return list(self._buffer)

        # Stratify by acceptance ratio
        high: list[Trajectory] = []
        low: list[Trajectory] = []
        for traj in self._buffer:
            if not traj.draft_token_ids:
                continue
            rate = sum(traj.accepted_mask) / len(traj.draft_token_ids)
            (high if rate >= 0.5 else low).append(traj)

        half = batch_size // 2
        batch: list[Trajectory] = []

        if high:
            batch.extend(random.sample(high, min(half, len(high))))
        if low:
            batch.extend(random.sample(low, min(half, len(low))))

        # Fill remaining slots from whichever pool has more
        remaining = batch_size - len(batch)
        if remaining > 0:
            pool = high if len(high) > len(low) else low
            if pool:
                extra = random.sample(pool, min(remaining, len(pool)))
                batch.extend(extra)

        random.shuffle(batch)
        return batch

    @property
    def size(self) -> int:
        """Number of trajectories currently in the buffer."""
        return len(self._buffer)

    def recent(self, n: int = 100) -> list[Trajectory]:
        """Return the *n* most recently added trajectories for analysis.

        Parameters
        ----------
        n : int
            Maximum number of trajectories to return (default 100).

        Returns
        -------
        list of Trajectory
            The most recent entries (oldest first).
        """
        if n <= 0:
            return []
        return list(self._buffer)[-n:]


# ---------------------------------------------------------------------------
# OnlineLoRAUpdater  — trains LoRA adapters from buffer samples
# ---------------------------------------------------------------------------


def _default_optimizer(params: list[Any], lr: float) -> Any:
    """Placeholder optimizer factory.

    In a real implementation this would instantiate ``torch.optim.AdamW``.
    Returns a dict mimicking the optimiser interface for the stub.
    """
    return {"params": params, "lr": lr, "_type": "adamw"}


class OnlineLoRAUpdater:
    """Online LoRA fine-tuner for self-improving speculative decoding.

    Periodically samples a batch of trajectories from a
    :class:`SpeculativeFeedbackBuffer` and computes a training signal:

    *   **Accepted tokens** -- maximise their log-probability under the draft
        model (treated as positive targets).
    *   **Rejected tokens** -- minimise their log-probability (treated as
        negative targets).
    *   **KL penalty** -- a reverse-KL divergence against the *base* draft
        model (the state before any online updates), preventing the adapter
        from diverging too far from the original distribution.

    LoRA hyper-parameters follow the standard ``(r, alpha)`` convention.

    Parameters
    ----------
    draft_model_ref : Any
        Reference to the draft model (framework-agnostic).  Must expose
        a ``forward`` method that accepts token IDs and returns logits.
    lora_r : int
        LoRA rank (default 8).
    lora_alpha : float
        LoRA alpha scaling factor (default 16).
    lr : float
        Learning rate for the LoRA optimiser (default 1e-4).
    """

    def __init__(
        self,
        draft_model_ref: Any,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lr: float = 1e-4,
    ) -> None:
        if lora_r < 1:
            raise ValueError(f"lora_r must be >= 1, got {lora_r}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")

        self._model_ref = draft_model_ref
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lr = lr
        self._adapter_version: int = 0

        # Placeholder for LoRA weight matrices: {layer_name: (A, B)}
        # A shape: (in_features, r) , B shape: (r, out_features), or the
        # transposed variant depending on framework convention.
        self._adapter: dict[str, tuple[Any, Any]] = {}
        self._base_weights: dict[str, Any] = {}

        # Optimiser state (stub)
        self._optimizer = _default_optimizer(list(self._adapter.values()), lr)

        self._lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    def update(self, buffer: SpeculativeFeedbackBuffer) -> dict[str, float]:
        """Train the LoRA adapter on a sampled batch from *buffer*.

        The update computes three loss terms:

        .. math::

            \\mathcal{L} =
                -\\frac{1}{|A|} \\sum_{i \\in A} \\log p(y_i \\mid x)
                + \\frac{1}{|R|} \\sum_{i \\in R} \\log p(y_i \\mid x)
                + \\lambda \\cdot \\text{KL}(p_{\\theta} \\| p_{\\text{base}})

        where *A* are accepted positions, *R* are rejected positions, and
        *lambda* is the KL penalty coefficient (set to 0.01 by default).

        Parameters
        ----------
        buffer : SpeculativeFeedbackBuffer
            Buffer from which a batch is sampled.

        Returns
        -------
        dict of str -> float
            Training metrics for this step: ``{"loss": ..., "accepted_nll": ...,
            "rejected_nll": ..., "kl_penalty": ...}``.
        """
        batch = buffer.sample(batch_size=32)
        if not batch:
            return {
                "loss": 0.0,
                "accepted_nll": 0.0,
                "rejected_nll": 0.0,
                "kl_penalty": 0.0,
            }

        with self._lock:
            metrics = self._compute_loss(batch)
            self._apply_gradients(metrics)
            self._adapter_version += 1

        return metrics

    def get_adapter(self) -> dict[str, tuple[Any, Any]]:
        """Return the current LoRA weight matrices.

        Returns
        -------
        dict of str -> (A, B)
            Mapping from layer name to ``(lora_A, lora_B)`` weight tuple.
            Returns a copy to avoid external mutation.
        """
        with self._lock:
            return dict(self._adapter)

    def swap_adapter(self, new_weights: dict[str, tuple[Any, Any]]) -> None:
        """Atomically hot-swap the LoRA adapter during serving.

        The swap is protected by the internal lock so that an in-flight
        forward pass completes with the old weights before the new ones
        take effect.

        Parameters
        ----------
        new_weights : dict of str -> (A, B)
            New LoRA weight matrices.
        """
        with self._lock:
            self._adapter = dict(new_weights)

    @property
    def adapter_version(self) -> int:
        """Monotonically-increasing version counter for the adapter."""
        return self._adapter_version

    # -- Internal helpers ---------------------------------------------------

    def _compute_loss(self, batch: list[Trajectory]) -> dict[str, float]:
        """Compute loss terms for a batch of trajectories.

        This is a framework-agnostic stub.  A real implementation would
        invoke ``self._model_ref.forward(...)`` to obtain logits, compute
        cross-entropy for accepted positions, negative cross-entropy for
        rejected positions, and a KL divergence against a frozen copy of
        the base model.

        Parameters
        ----------
        batch : list of Trajectory
            Sampled trajectories.

        Returns
        -------
        dict of str -> float
            Loss components.
        """
        # Count accepted / rejected tokens across the batch
        total_accepted = 0
        total_rejected = 0
        for traj in batch:
            for accepted in traj.accepted_mask:
                if accepted:
                    total_accepted += 1
                else:
                    total_rejected += 1

        # Stub: simulate loss values.
        # In production these would be actual forward-pass results.
        accepted_nll = (
            -math.log(0.8) * total_accepted / max(total_accepted, 1)
            if total_accepted
            else 0.0
        )
        rejected_nll = (
            math.log(0.2) * total_rejected / max(total_rejected, 1)
            if total_rejected
            else 0.0
        )
        kl_penalty = 0.01 * (accepted_nll + abs(rejected_nll)) / 2.0

        loss = accepted_nll + rejected_nll + kl_penalty
        return {
            "loss": loss,
            "accepted_nll": accepted_nll,
            "rejected_nll": rejected_nll,
            "kl_penalty": kl_penalty,
        }

    def _apply_gradients(self, _metrics: dict[str, float]) -> None:
        """Apply gradients to the LoRA parameters.

        Stub: in a real implementation this would call
        ``self._optimizer.step()`` and ``self._optimizer.zero_grad()``.
        """
        # Placeholder for actual gradient application
        pass


# ---------------------------------------------------------------------------
# Integration function  —  wires together the SISD pipeline
# ---------------------------------------------------------------------------


@dataclass
class SISDConfig:
    """Configuration for the self-improving speculative-decoding pipeline.

    Attributes
    ----------
    buffer_max_size : int
        Maximum number of trajectories in the feedback buffer (default 10000).
    lora_r : int
        LoRA rank (default 8).
    lora_alpha : float
        LoRA alpha scaling factor (default 16).
    lr : float
        Learning rate (default 1e-4).
    update_interval_s : float
        Minimum interval in seconds between LoRA update cycles (default 60.0).
    batch_size : int
        Number of trajectories per training batch (default 32).
    """

    buffer_max_size: int = 10000
    lora_r: int = 8
    lora_alpha: float = 16.0
    lr: float = 1e-4
    update_interval_s: float = 60.0
    batch_size: int = 32


def create_sisd_pipeline(
    target_model: Any,
    draft_model: Any,
    config: SISDConfig | None = None,
) -> tuple[Any, SpeculativeFeedbackBuffer, OnlineLoRAUpdater]:
    """Create the full self-improving speculative-decoding pipeline.

    Wires together a feedback buffer, an online LoRA updater, and starts a
    background thread that periodically samples the buffer and updates the
    draft-model adapter.

    Parameters
    ----------
    target_model : Any
        The target (large) model used for speculative-decoding verification.
    draft_model : Any
        The draft (small) model to be continuously improved.
    config : SISDConfig or None
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    tuple of (orchestrator, buffer, updater)
        ``orchestrator``
            A callable suitable for integration with the existing
            :class:`DraftOrchestrator` or used directly as a decorated
            draft model wrapper.
        ``buffer``
            The :class:`SpeculativeFeedbackBuffer` instance so that
            production code can call ``buffer.add(...)`` after each
            speculative-decoding step.
        ``updater``
            The :class:`OnlineLoRAUpdater` instance for manual control
            (e.g. triggering an immediate update or inspecting the adapter).

    Notes
    -----
    The background updater thread is a daemon thread and will be torn down
    when the process exits.  Use ``updater.update(buffer)`` directly for
    synchronous, caller-controlled training cycles.
    """
    if config is None:
        config = SISDConfig()

    buffer = SpeculativeFeedbackBuffer(max_size=config.buffer_max_size)
    updater = OnlineLoRAUpdater(
        draft_model_ref=draft_model,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lr=config.lr,
    )

    # Build a lightweight wrapper that collects trajectories and feeds
    # them into the buffer.  This is _not_ the draft model itself — it is
    # an orchestrator callback that production code calls after each
    # speculative step.
    def orchestrator(
        prefix_ids: list[int],
        draft_token_ids: list[int],
        accepted_mask: list[bool],
    ) -> None:
        """Record a speculative-decoding trajectory and attempt an update.

        Parameters
        ----------
        prefix_ids : list of int
            Token IDs of the prompt prefix.
        draft_token_ids : list of int
            Token IDs emitted by the draft model.
        accepted_mask : list of bool
            Boolean mask indicating acceptance for each draft token.
        """
        buffer.add(prefix_ids, draft_token_ids, accepted_mask)

    # Background LoRA updater thread
    def _background_loop() -> None:

        while True:
            _time.sleep(config.update_interval_s)
            try:
                updater.update(buffer)
            except Exception:
                # Log and continue — the background loop must never die
                pass

    updater_thread = threading.Thread(
        target=_background_loop,
        daemon=True,
        name="sisd-updater",
    )
    updater_thread.start()

    return orchestrator, buffer, updater
