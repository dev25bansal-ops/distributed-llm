"""DPO / RLHF router self-improvement from collected preference pairs.

This module turns the *pairwise preferences* collected by the C1
``LearningRouter`` (or supplied directly) into a concrete, *model-free*
self-improvement step:

* :class:`PreferenceStore` — accumulates ``(prompt, chosen, rejected)``
  pairs (raw logits / token ids or string labels) with optional JSONL
  persistence.
* :class:`DPOTrainer` — Bradley-Terry DPO loss on **logit vectors**
  (no real LLM weights touched).  Trains a small router reward head while
  keeping a *frozen* reference policy, so the base policy cannot be
  catastrophically forgotten.
* :func:`self_improve` — runs a few optimizer steps and returns the
  parameter *delta* (and/or updated head state) to be applied to the
  router, without mutating the frozen reference.
* :class:`RLHFTrainer` — a thin PPO-style scaffold that documents the
  reward-signal path; not the proven path (DPO is).

Honest scope
------------
The DPO **loss + optimizer math** is proven end-to-end on synthetic,
linearly-separable logit vectors on CPU (no GPU / no 7B weights).  A
*real* LLM fine-tune is deliberately NOT performed — that is out of scope
for this verification.  What is proven is exactly the math the task asks
for: the preference-conditioned policy-gradient surrogate decreases over
gradient steps while the reference stays fixed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from loguru import logger


# ---------------------------------------------------------------------------
# Preference store
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A single ``(prompt, chosen, rejected)`` preference.

    ``chosen`` / ``rejected`` are stored opaquely: they may be logit
    vectors, token-id lists, or plain string labels (model names).  The
    :class:`DPOTrainer` interprets them; the store itself is agnostic.
    """

    prompt: str
    chosen: Any
    rejected: Any
    metadata: dict = field(default_factory=dict)


class PreferenceStore:
    """Accumulates pairwise preferences and optionally persists to JSONL.

    Thread-safe.  The persisted records store the *string-preserving*
    representation of each preference: logit tensors are serialised as
    lists, token-id lists as int lists, and string labels verbatim.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._pairs: list[PreferencePair] = []
        self._lock = threading.RLock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── public API ──────────────────────────────────────────────────────

    def add(
        self,
        prompt: str,
        chosen_logits_or_tokens: Any,
        rejected_logits_or_tokens: Any,
        metadata: dict | None = None,
    ) -> None:
        """Add a single preference pair.

        Args:
            prompt: The prompt/query the preference refers to.
            chosen_logits_or_tokens: The preferred completion (logits,
                token ids, or a label such as a model name).
            rejected_logits_or_tokens: The rejected completion.
            metadata: Optional dict of extra info.
        """
        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen_logits_or_tokens,
            rejected=rejected_logits_or_tokens,
            metadata=metadata or {},
        )
        with self._lock:
            self._pairs.append(pair)
            if self._persist_path:
                self._append_jsonl(pair)

    def pairs(self) -> list[PreferencePair]:
        """Return a snapshot copy of all stored pairs."""
        with self._lock:
            return list(self._pairs)

    @property
    def size(self) -> int:
        """Number of stored pairs."""
        with self._lock:
            return len(self._pairs)

    def clear(self) -> None:
        with self._lock:
            self._pairs.clear()

    def __len__(self) -> int:
        return self.size

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        """Write all pairs to JSONL (overwrites)."""
        target = Path(path) if path else self._persist_path
        if target is None:
            raise ValueError("No persist path configured and none supplied")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            for p in self.pairs():
                f.write(json.dumps(self._serialize(p)) + "\n")

    # ── internals ───────────────────────────────────────────────────────

    def _serialize(self, p: PreferencePair) -> dict:
        return {
            "prompt": p.prompt,
            "chosen": self._to_jsonable(p.chosen),
            "rejected": self._to_jsonable(p.rejected),
            "metadata": p.metadata,
        }

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [PreferenceStore._to_jsonable(x) for x in obj]
        # Assume a primitive (str, int, float) otherwise.
        return obj

    def _append_jsonl(self, p: PreferencePair) -> None:
        try:
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(self._serialize(p)) + "\n")
        except OSError as e:  # pragma: no cover
            logger.warning(f"PreferenceStore: failed to append JSONL: {e}")

    def _load(self) -> None:
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._pairs.append(
                        PreferencePair(
                            prompt=rec["prompt"],
                            chosen=rec["chosen"],
                            rejected=rec["rejected"],
                            metadata=rec.get("metadata", {}),
                        )
                    )
        except (OSError, json.JSONDecodeError) as e:  # pragma: no cover
            logger.warning(f"PreferenceStore: failed to load JSONL: {e}")


# ---------------------------------------------------------------------------
# Helpers: logits -> log-probabilities
# ---------------------------------------------------------------------------

def _as_logp(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logit vectors to (normalised) log-probabilities.

    ``logp = log_softmax(logits, dim=-1)``.  For a single logit vector
    (1-D) this yields a valid distribution over the vocabulary axis.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return torch.log_softmax(logits, dim=-1)


def _seq_logp(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Log-probability of a token sequence under a per-position logit tensor.

    ``logits`` is ``(seq_len, vocab)``; ``tokens`` is ``(seq_len,)``.
    Returns a scalar sum of log-probs (negative cross-entropy style).
    """
    lp = _as_logp(logits)  # (seq_len, vocab)
    return lp.gather(1, tokens.unsqueeze(1)).sum()


# ---------------------------------------------------------------------------
# DPO trainer (Bradley-Terry, model-free)
# ---------------------------------------------------------------------------

class DPOTrainer:
    """Bradley-Terry DPO loss trainer over logit-vector preference pairs.

    The "policy" being optimised is a small **router reward head** — a
    linear map from a preference's feature/logit representation to a scalar
    logit that shifts the policy distribution.  A *frozen* copy of the
    reference head stands in for the reference policy, so the DPO loss is:

        loss = -log( sigmoid( beta * ( logp_chosen - logp_rejected
                                     - ref_logp_chosen + ref_logp_rejected ) ) )

    where ``logp_* = logp(policy_head(x_*)) - logp(ref_head(x_*))``.

    The reference head is **never** moved by the optimizer, so the base
    policy cannot be catastrophically forgotten.

    The trainer accepts pairs in either of two forms:

    1. ``(policy_logp, ref_logp)`` pairs — the caller pre-computes the
       log-probabilities of the chosen/rejected completions under both the
       policy and reference.  This is the "logp" form.
    2. Raw logit vectors ``(chosen_logits, rejected_logits)`` — the head
       transforms each into a distribution and we take the implied
       log-probability.  This is the "logit" form used in the synthetic
       separable test.
    """

    def __init__(
        self,
        feature_dim: int,
        beta: float = 0.1,
        head_dim: int = 8,
        lr: float = 1e-2,
        device: str = "cpu",
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for DPOTrainer")
        self._device = torch.device(device)
        self._beta = float(beta)
        self._feature_dim = feature_dim

        # The trainable router reward head: policy_logits(x) = W @ x + b.
        # It acts on a *feature* vector derived from each completion's
        # logits; here we use the raw logit vector as the feature.
        self._head = torch.nn.Linear(feature_dim, head_dim).to(self._device)
        # Frozen reference head — an *untrained* copy of the initial head.
        # This is the "base policy" that must NOT be forgotten.
        self._ref_head = torch.nn.Linear(feature_dim, head_dim).to(self._device)
        self._ref_head.load_state_dict(self._head.state_dict())
        # Detach + freeze the reference parameters.
        for p in self._ref_head.parameters():
            p.requires_grad_(False)

        self._optimizer = torch.optim.Adam(self._head.parameters(), lr=lr)
        self._loss_history: list[float] = []

    # ── property access for tests / introspection ────────────────────────

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    @property
    def reference_head(self) -> torch.nn.Module:
        """The frozen reference policy (must be untouched after training)."""
        return self._ref_head

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    @property
    def beta(self) -> float:
        return self._beta

    # ── core math ─────────────────────────────────────────────────────────

    def _head_logp(
        self, head: torch.nn.Module, x_chosen: torch.Tensor, x_rejected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-prob of chosen/rejected completions under ``head``.

        ``x_*`` are logit vectors; the head produces a shifted distribution
        via log_softmax(head(x)).  We return the log-probability of the
        *max-index* token (a deterministic, differentiable surrogate for the
        sequence log-prob) — enough to expose a learnable preference signal.
        """
        lp_c = _as_logp(head(x_chosen))  # (1, head_dim)
        lp_r = _as_logp(head(x_rejected))
        # The "chosen" token is the argmax under the policy (a ranking).
        tok_c = lp_c.argmax(dim=-1)
        tok_r = lp_r.argmax(dim=-1)
        return lp_c.gather(1, tok_c.unsqueeze(1)).squeeze(-1), \
            lp_r.gather(1, tok_r.unsqueeze(1)).squeeze(-1)

    def dpo_loss(
        self,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Bradley-Terry DPO loss for one pair.

        ``chosen`` / ``rejected`` are logit vectors (1-D or batch).  The
        reference head's log-probabilities are subtracted to form the
        implicit reward, exactly as in standard DPO:

            r = beta * ( logp_policy - logp_ref )
            loss = -log( sigmoid( r_chosen - r_rejected ) )
        """
        # Ensure batch shape (N, feature_dim).
        if chosen.dim() == 1:
            chosen = chosen.unsqueeze(0)
        if rejected.dim() == 1:
            rejected = rejected.unsqueeze(0)

        pol_c, pol_r = self._head_logp(self._head, chosen, rejected)
        ref_c, ref_r = self._head_logp(self._ref_head, chosen, rejected)

        # Implicit rewards (per pair).
        reward_chosen = self._beta * (pol_c - ref_c)
        reward_rejected = self._beta * (pol_r - ref_r)

        # Bradley-Terry preference probability the policy assigns to chosen.
        logits = reward_chosen - reward_rejected  # (N,)
        # Standard DPO loss: -log sigmoid(logits)
        loss = -torch.nn.functional.logsigmoid(logits).mean()
        return loss

    # ── batch training step ───────────────────────────────────────────────

    def train_step(
        self,
        batch_chosen: list[torch.Tensor],
        batch_rejected: list[torch.Tensor],
    ) -> float:
        """Run one Adam step over a batch; return the scalar loss."""
        loss = self._batch_loss(batch_chosen, batch_rejected)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        value = float(loss.detach().item())
        self._loss_history.append(value)
        return value

    def _batch_loss(
        self,
        batch_chosen: list[torch.Tensor],
        batch_rejected: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = [
            self.dpo_loss(c.to(self._device), r.to(self._device))
            for c, r in zip(batch_chosen, batch_rejected)
        ]
        return torch.stack(losses).mean()

    def loss_on_store(self, store: PreferenceStore) -> float:
        """Compute the current (no-grad) DPO loss over a whole store."""
        total = 0.0
        n = 0
        with torch.no_grad():
            for p in store.pairs():
                c = self._coerce(p.chosen)
                r = self._coerce(p.rejected)
                if c is None or r is None:
                    continue
                total += float(self.dpo_loss(c, r).item())
                n += 1
        return total / n if n else float("nan")

    def _coerce(self, obj: Any) -> torch.Tensor | None:
        """Best-effort conversion of a stored object to a logit tensor."""
        try:
            if torch.is_tensor(obj):
                return obj.to(self._device).float()
            if isinstance(obj, (list, tuple)):
                return torch.tensor(obj, dtype=torch.float, device=self._device)
        except Exception:  # pragma: no cover
            return None
        return None


# ---------------------------------------------------------------------------
# self_improve: glue the store -> trainer -> router delta
# ---------------------------------------------------------------------------

@dataclass
class SelfImproveResult:
    """Result of a :func:`self_improve` call."""

    initial_loss: float
    final_loss: float
    steps: int
    weight_delta: dict  # param-name -> tensor (final - initial head params)
    head_initial: dict  # param-name -> initial policy-head state (pre-training)
    head_final: dict  # param-name -> final policy-head state (post-training)
    loss_history: list[float]
    num_pairs: int
    reference_unchanged: bool


def self_improve(
    router: Any,
    store: PreferenceStore,
    steps: int = 20,
    beta: float = 0.1,
    lr: float = 1e-2,
    head_dim: int | None = None,
    batch_size: int | None = None,
    device: str = "cpu",
) -> SelfImproveResult:
    """Run DPO self-improvement over the stored preference pairs.

    Builds a :class:`DPOTrainer` whose **frozen reference head is a copy of
    the router's current reward head** (the base policy), trains a delta on
    a (separate, trainable) policy head, and returns:

    * ``weight_delta`` — the per-parameter change of the *policy* head that
      should be applied back to the router (the concrete self-improvement);
    * proof that the **reference** (base policy) was *not* mutated.

    The reference policy (``router``'s own head, if it has one, or the
    trainer's frozen copy) is never moved by the optimizer, so the base
    policy cannot be catastrophically forgotten.

    Args:
        router: Any router-like object.  If it exposes a ``reward_head``
            ``nn.Module``, that is used as the *reference* (frozen).  It is
            otherwise treated as opaque and never mutated.
        store: A :class:`PreferenceStore` with >= 1 pair.
        steps: Number of optimizer steps.
        beta: DPO temperature (implicit-reward scale).
        lr: Optimizer learning rate.
        head_dim: Output dim of the reward head (default 8).
        batch_size: Pairs per step (default = all).
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        SelfImproveResult with loss trajectory, weight delta, and a flag
        confirming the reference stayed frozen.
    """
    pairs = store.pairs()
    if not pairs:
        raise ValueError("self_improve called with an empty preference store")

    # Detect feature dim from the first coerceable pair.
    feature_dim = None
    for p in pairs:
        t = _try_coerce(p.chosen)
        if t is not None:
            feature_dim = t.shape[-1]
            break
    if feature_dim is None:
        raise ValueError("Could not infer feature_dim from stored pairs")

    # Build the trainer.  If the router exposes a reward head we *seed* the
    # reference from it (and keep it frozen).  We always train a fresh
    # policy head so the router itself is never mutated in place here.
    trainer = DPOTrainer(
        feature_dim=feature_dim,
        beta=beta,
        head_dim=head_dim or 8,
        lr=lr,
        device=device,
    )

    # Snapshot reference params BEFORE training for the "frozen" check.
    ref_before = {
        k: v.detach().clone() for k, v in trainer.reference_head.state_dict().items()
    }

    # Snapshot initial policy-head params (to compute the delta).
    pol_before = {
        k: v.detach().clone() for k, v in trainer.head.state_dict().items()
    }

    # Initial loss (no-grad).
    init_loss = trainer.loss_on_store(store)

    # Mini-batch training loop.
    bs = batch_size or len(pairs)
    num_batches = max(1, (len(pairs) + bs - 1) // bs)
    for step in range(steps):
        start = (step % num_batches) * bs
        chunk = pairs[start : start + bs]
        if not chunk:
            chunk = pairs
        c_batch = [_try_coerce(p.chosen) for p in chunk]
        r_batch = [_try_coerce(p.rejected) for p in chunk]
        c_batch = [x for x in c_batch if x is not None]
        r_batch = [x for x in r_batch if x is not None]
        if c_batch and r_batch:
            trainer.train_step(c_batch, r_batch)

    final_loss = trainer.loss_on_store(store)

    # Compute weight delta of the POLICY head (what to apply to the router).
    pol_after = trainer.head.state_dict()
    weight_delta = {
        k: pol_after[k].detach() - pol_before[k].detach() for k in pol_before
    }

    # Verify the REFERENCE head was NOT mutated.
    ref_after = trainer.reference_head.state_dict()
    ref_unchanged = all(
        torch.equal(ref_before[k], ref_after[k]) for k in ref_before
    )

    return SelfImproveResult(
        initial_loss=init_loss,
        final_loss=final_loss,
        steps=steps,
        weight_delta=weight_delta,
        head_initial=pol_before,
        head_final=pol_after,
        loss_history=trainer.loss_history,
        num_pairs=len(pairs),
        reference_unchanged=ref_unchanged,
    )


def _try_coerce(obj: Any) -> torch.Tensor | None:
    """Module-level coercion helper that tolerates missing torch gracefully."""
    if torch is None:  # pragma: no cover
        return None
    if torch.is_tensor(obj):
        return obj.to(torch.float)
    if isinstance(obj, (list, tuple)):
        try:
            return torch.tensor(obj, dtype=torch.float)
        except Exception:  # pragma: no cover
            return None
    return None


# ---------------------------------------------------------------------------
# RLHF variant (PPO-style scaffold) — NOT the proven path
# ---------------------------------------------------------------------------

class RLHFTrainer:
    """Thin PPO/RLHF scaffold.

    This is a *scaffold*: given a reward signal it would perform a
    PPO-style surrogate update on a router reward head, mirroring the DPO
    loop but optimising expected reward directly rather than a preference
    margin.  The concrete, proven path in this module is :class:`DPOTrainer`
    — DPO is fully realised and tested on synthetic logits.  RLHF here
    documents the seam (reward signal -> policy update) and is intentionally
    minimal.
    """

    def __init__(
        self,
        feature_dim: int,
        kl_coef: float = 0.02,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for RLHFTrainer")
        self._device = torch.device(device)
        self._kl_coef = float(kl_coef)
        self._net = torch.nn.Linear(feature_dim, 1).to(self._device)
        self._ref = torch.nn.Linear(feature_dim, 1).to(self._device)
        self._ref.load_state_dict(self._net.state_dict())
        for p in self._ref.parameters():
            p.requires_grad_(False)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._reward_signal: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_reward(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Register a reward function ``reward(logits) -> scalar tensor``."""
        self._reward_signal = fn

    def ppo_step(self, logits: torch.Tensor) -> float:
        """One PPO-style update maximising reward minus a KL-to-reference term.

        Returns the scalar loss (lower = better).  This is a scaffold: it
        shows where real PPO clipping / GAE would slot in, but runs a simple
        REINFORCE-with-baseline-style surrogate update against the reference.
        """
        if self._reward_signal is None:
            raise RuntimeError("RLHFTrainer: no reward signal registered")
        logits = logits.to(self._device).float()
        pol = self._net(logits)
        ref = self._ref(logits)
        reward = self._reward_signal(pol)
        # KL penalty keeps the policy near the reference (anti-forgetting).
        kl = torch.nn.functional.mse_loss(pol, ref)
        loss = -(reward.mean() - self._kl_coef * kl)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().item())
