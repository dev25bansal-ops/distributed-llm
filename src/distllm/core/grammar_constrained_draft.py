"""Speculative + structured-output fusion.

Makes the draft model propose **only grammar-valid tokens** so speculative
decoding and constrained (structured) generation reinforce each other
instead of fighting: the draft stage is masked by the grammar, so nearly
every proposed token is accepted and the final sequence is guaranteed
grammar-valid (the verifier still confirms against the target).

The core primitive is :func:`mask_draft_logits`: given a grammar object that
exposes ``get_logits_mask(vocab_size, tokenizer, device) -> bool tensor``
(True = allowed), it sets disallowed draft logits to ``-inf`` before sampling.
This is wired into :class:`~distllm.core.speculative_decoder.SpecDecoderBase`
and :class:`~distllm.core.distributed_speculative.DistributedSpeculativeDecoder`
via an optional ``grammar`` argument (opt-in, no behavior change when absent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_draft_logits(
    logits: torch.Tensor,
    grammar_mask: torch.Tensor,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Zero out (``-inf``) draft logits for grammar-forbidden token ids.

    Args:
        logits: ``(1, vocab_size)`` draft logits for one position.
        grammar_mask: Boolean tensor, True where the token is grammar-valid.
            May be shorter than ``vocab_size`` (extra ids treated as forbidden).
        vocab_size: Expected vocab size; if the mask is shorter, ids beyond
            its length are forbidden.

    Returns:
        A new logits tensor with forbidden positions set to ``-inf``.
    """
    if grammar_mask is None:
        return logits
    v = vocab_size or logits.shape[-1]
    if grammar_mask.shape[0] < v:
        # Pad the mask with False (forbidden) for ids the grammar didn't cover.
        pad = torch.zeros(v - grammar_mask.shape[0], dtype=torch.bool, device=grammar_mask.device)
        grammar_mask = torch.cat([grammar_mask, pad], dim=0)
    masked = logits.clone()
    masked[:, ~grammar_mask] = float("-inf")
    return masked


class GrammarConstrainedDraftPolicy:
    """Drives grammar-constrained drafting for a speculative decoder.

    Holds a grammar (any object with ``get_logits_mask(vocab_size, tokenizer,
    device)`` and an optional ``reset()``) and a tokenizer, and produces a
    ``mask_fn(logits) -> masked_logits`` that the decoder calls right before
    sampling each draft token.  The grammar position is advanced by the
    caller as tokens are committed (see ``advance``).
    """

    def __init__(self, grammar: object, tokenizer: object, device: str | torch.device = "cpu") -> None:
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._device = torch.device(device)

    def reset(self) -> None:
        if hasattr(self._grammar, "reset"):
            self._grammar.reset()

    def mask_fn(self, logits: torch.Tensor) -> torch.Tensor:
        """Return draft logits with grammar-forbidden tokens zeroed."""
        vocab_size = logits.shape[-1]
        grammar_mask = self._grammar.get_logits_mask(
            vocab_size, self._tokenizer, str(self._device)
        )
        return mask_draft_logits(logits, grammar_mask, vocab_size)

    def advance(self, token_id: int) -> None:
        """Advance the grammar position after a token is committed.

        Best-effort: many grammar objects track position internally as they
        are queried; this hook lets callers push explicit progress for
        stateful grammars.
        """
        fn = getattr(self._grammar, "advance", None)
        if callable(fn):
            try:
                fn(token_id)
            except Exception:
                pass
