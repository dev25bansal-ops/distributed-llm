"""Verification report history store for the dashboard.

Records, per ``(model_id, partition)``, a rolling history of speculative
verification quality:

* ``logit_cosine`` — cosine similarity between the draft model's logits and
  the target model's logits at a verified position.  A value of ``1.0`` means
  the two distributions point in exactly the same direction; ``~0`` means they
  are orthogonal.
* ``token_match`` — ``1`` if the draft token id equals the target argmax
  token id (i.e. greedy acceptance), else ``0``.
* ``acceptance`` — the boolean acceptance decision fed in by the caller (kept
  as an int for easy aggregation), defaulting to ``token_match`` when omitted.

This module is *additive* observability: it never influences the acceptance
math in :mod:`distllm.core.spec_verify`.  Callers compute acceptance there and
then hand the raw logits/token ids here purely for reporting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F


def compute_logit_cosine(
    draft_logits: torch.Tensor | None,
    target_logits: torch.Tensor | None,
) -> float | None:
    """Cosine similarity between two 1-D logit vectors.

    Guards for ``None`` inputs and shape mismatches, returning ``None`` when a
    meaningful cosine cannot be computed.  Multi-dimensional tensors are
    flattened to a single vector before comparison.

    Returns:
        A float in ``[-1.0, 1.0]``, or ``None`` if inputs are missing or their
        flattened shapes disagree.
    """
    if draft_logits is None or target_logits is None:
        return None
    try:
        d = draft_logits.reshape(-1).float()
        t = target_logits.reshape(-1).float()
    except Exception:
        return None
    if d.numel() == 0 or t.numel() == 0:
        return None
    if d.shape != t.shape:
        return None
    cos = F.cosine_similarity(d.unsqueeze(0), t.unsqueeze(0), dim=-1)
    return float(cos.item())


class VerificationHistoryStore:
    """Rolling per-(model, partition) history of verification quality metrics."""

    def __init__(self, max_history: int = 300):
        self.max_history = max_history
        self._history: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    @staticmethod
    def _key(model: str, partition: str) -> tuple[str, str]:
        return (str(model), str(partition))

    def record(
        self,
        model: str,
        partition: str,
        draft_logits: torch.Tensor | None,
        target_logits: torch.Tensor | None,
        draft_token_id: int | None,
        target_token_id: int | None,
        *,
        acceptance: bool | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Record one verification event and return the stored entry.

        Args:
            model: Model identifier.
            partition: Partition identifier (e.g. ``"0-15"`` layer range).
            draft_logits: Draft model logits for this position (any shape), or
                ``None``.
            target_logits: Target model logits for this position, or ``None``.
            draft_token_id: The draft-proposed token id, or ``None``.
            target_token_id: The target argmax token id, or ``None``.
            acceptance: Optional explicit acceptance decision.  When omitted,
                defaults to ``token_match``.
            timestamp: Optional explicit timestamp (defaults to ``time.time()``).

        Returns:
            The recorded entry dict.
        """
        cosine = compute_logit_cosine(draft_logits, target_logits)

        if draft_token_id is None or target_token_id is None:
            token_match = 0
        else:
            token_match = 1 if int(draft_token_id) == int(target_token_id) else 0

        if acceptance is None:
            accept_val = token_match
        else:
            accept_val = 1 if acceptance else 0

        entry = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "model": str(model),
            "partition": str(partition),
            "logit_cosine": cosine,
            "token_match": token_match,
            "acceptance": accept_val,
        }
        self._history[self._key(model, partition)].append(entry)
        return entry

    def history(
        self, model: str, partition: str, window: int | None = None
    ) -> list[dict[str, Any]]:
        """Return recorded entries for ``(model, partition)``.

        Args:
            model: Model identifier.
            partition: Partition identifier.
            window: If given, return only the most recent ``window`` entries.

        Returns:
            A list of entry dicts, oldest first.
        """
        entries = list(self._history.get(self._key(model, partition), ()))
        if window is not None and window >= 0:
            entries = entries[-window:]
        return entries

    def aggregate(
        self, model: str, partition: str, window: int | None = None
    ) -> dict[str, Any]:
        """Rolling aggregate for ``(model, partition)``.

        Returns:
            ``{count, mean_logit_cosine, token_match_rate, acceptance_rate}``.
            ``mean_logit_cosine`` is ``None`` when no cosine values exist in the
            window.
        """
        entries = self.history(model, partition, window=window)
        count = len(entries)
        if count == 0:
            return {
                "count": 0,
                "mean_logit_cosine": None,
                "token_match_rate": 0.0,
                "acceptance_rate": 0.0,
            }
        cosines = [e["logit_cosine"] for e in entries if e["logit_cosine"] is not None]
        mean_cos = sum(cosines) / len(cosines) if cosines else None
        token_match_rate = sum(e["token_match"] for e in entries) / count
        acceptance_rate = sum(e["acceptance"] for e in entries) / count
        return {
            "count": count,
            "mean_logit_cosine": mean_cos,
            "token_match_rate": token_match_rate,
            "acceptance_rate": acceptance_rate,
        }

    def keys(self) -> list[tuple[str, str]]:
        """Return all recorded ``(model, partition)`` keys."""
        return list(self._history.keys())

    def snapshot(self, window: int | None = 20) -> list[dict[str, Any]]:
        """Per-(model, partition) reporting snapshot for the dashboard.

        Returns a list of dicts, one per key, each with recent ``entries`` and
        the rolling ``aggregate`` — the shape consumed by the UI tab.
        """
        out: list[dict[str, Any]] = []
        for model, partition in self.keys():
            out.append(
                {
                    "model": model,
                    "partition": partition,
                    "entries": self.history(model, partition, window=window),
                    "aggregate": self.aggregate(model, partition, window=window),
                }
            )
        return out
