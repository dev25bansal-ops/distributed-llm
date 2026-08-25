"""Shared speculative-decoding verification primitives.

This module is the single source of truth for the draft-token
``prefix_len`` indexing and the acceptance decision used across the
speculative-decoding family:

* ``distributed_speculative.DistributedSpeculative._verify_tokens``
* ``speculative_decoder.SpeculativeDecoder._verify_tokens``
* ``multi_draft_verifier.MultiDraftVerifier`` (flat chains)
* ``draft_tree.DraftTree.verify_tree`` (tree paths)

Previously each module re-implemented the causal-LM logits indexing and
rejection-sampling loop independently.  That duplication is exactly how
the C3 off-by-one bug (verifying draft token *i* against the wrong logits
row) propagated to some modules and not others.  Everything now funnels
through :func:`prefix_len` and :func:`accept_token`, so a fix lands once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prefix_len(prefix: torch.Tensor) -> int:
    """Causal-LM draft index for a ``prefix`` tensor of shape ``(1, L)``.

    ``target_logits[:, j, :]`` predicts the token at absolute position
    ``j + 1``.  A draft token sitting at absolute prefix position
    ``(L + i)`` is therefore verified against logits row ``(L - 1 + i)``.

    Returns ``L - 1``.
    """
    return prefix.shape[1] - 1


def accept_token(
    target_logits: torch.Tensor,
    pos: int,
    token_id: int,
    *,
    draft_prob: float | None = None,
    temperature: float = 1.0,
    vocab_size: int | None = None,
    rng: torch.Generator | None = None,
) -> bool:
    """Decide whether a single draft ``token_id`` is accepted at logits row ``pos``.

    Greedy (``temperature == 0``): accept iff the token equals the target
    argmax.

    Sampled (``temperature > 0``): proper rejection sampling with
    ``min(1, p / q)`` where ``p`` is the target probability of the token and
    ``q`` is the draft probability.  When ``draft_prob`` (``q``) is ``None``,
    falls back to the uniform-draft approximation ``p * vocab_size``.

    Args:
        target_logits: ``(1, seq_len, vocab)`` target distribution.
        pos: Logits row index (already corrected for the causal shift).
        token_id: Candidate draft token id.
        draft_prob: Draft probability ``q`` for this position, or ``None``.
        temperature: Sampling temperature (0 = greedy).
        vocab_size: Vocabulary size (only needed for the uniform fallback).
        rng: Optional generator for reproducible sampling.

    Returns:
        ``True`` if the token should be accepted.
    """
    if temperature == 0:
        return int(target_logits[:, pos, :].argmax(dim=-1).item()) == int(token_id)

    target_probs = F.softmax(target_logits[:, pos, :] / temperature, dim=-1)
    p = float(target_probs[0, int(token_id)].item())

    if draft_prob is not None:
        q = float(draft_prob)
        if q <= 0:
            return False
        acceptance = min(1.0, p / q)
    else:
        if vocab_size is None:
            vocab_size = target_logits.shape[-1]
        acceptance = min(1.0, p * vocab_size)

    if rng is not None:
        return torch.rand(1, generator=rng).item() < acceptance
    return torch.rand(1).item() < acceptance


def eos_cutoff(tokens: list[int] | torch.Tensor, eos_token_id: int | None) -> int:
    """Length of the leading span of *tokens* that may be emitted before EOS.

    C14 shared kernel: returns the number of tokens up to and including the
    first ``eos_token_id`` occurrence, or ``len(tokens)`` when no EOS is
    configured (``None``) or absent.  Decoders truncate each round's newly
    produced tokens with this and end generation when it cuts anything off,
    so post-EOS hallucinations never reach the output.

    Accepts a list of ints or a 1-D/2-D token tensor (2-D uses row 0).
    """
    if eos_token_id is None:
        return len(tokens)
    if isinstance(tokens, torch.Tensor):
        flat = tokens.reshape(-1).tolist()
    else:
        flat = list(tokens)
    for i, t in enumerate(flat):
        if int(t) == int(eos_token_id):
            return i + 1  # include EOS itself
    return len(flat)


def verify_chain(
    prefix: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    draft_probs: list[float] | None = None,
    temperature: float = 1.0,
    vocab_size: int | None = None,
    rng: torch.Generator | None = None,
) -> int:
    """Verify a flat chain of draft tokens and return how many are accepted.

    Iterates over ``draft_tokens[:, 0..num-1]``, using :func:`prefix_len` for
    the causal shift and :func:`accept_token` for each decision.  Stops at the
    first rejected token (standard speculative-decoding semantics: a rejected
    token and everything after it are discarded).

    Args:
        prefix: ``(1, L)`` prefix tokens.
        draft_tokens: ``(1, num_draft)`` candidate tokens.
        target_logits: ``(1, seq_len, vocab)`` target distribution.
        draft_probs: Optional per-position draft probabilities ``q``
            (probabilities, not log-probabilities).  ``None`` entries fall
            back to the uniform-draft approximation for that position.
        temperature: Sampling temperature.
        vocab_size: Vocabulary size (uniform fallback only).
        rng: Optional generator for reproducible sampling.

    Returns:
        Number of leading draft tokens accepted (``0 <= n <= num_draft``).
    """
    num = draft_tokens.shape[1]
    L = prefix_len(prefix)
    for i in range(num):
        pos = L + i
        if pos >= target_logits.shape[1]:
            break
        q = draft_probs[i] if draft_probs is not None and i < len(draft_probs) else None
        if not accept_token(
            target_logits,
            pos,
            int(draft_tokens[0, i].item()),
            draft_prob=q,
            temperature=temperature,
            vocab_size=vocab_size,
            rng=rng,
        ):
            return i
    return num


def verify_draft_acceptance(
    draft_ids: list[int],
    target_ids: list[int],
    draft_probs: list[float] | None = None,
    target_probs: list[float] | None = None,
    *,
    rng: torch.Generator | None = None,
) -> list[bool]:
    """Pure, token-id-level facade over the canonical acceptance decision.

    This is a *thin* wrapper around the same per-position logic used by the
    runtime decoder (see :func:`accept_token` / :func:`verify_chain`).  It is
    provided so the draft-acceptance invariant can be verified with pure
    property/symbolic tests **without** touching the runtime engine.  It does
    **not** change decoder behavior.

    The function answers the core speculative-decoding invariant directly:

        accepted_ids == target_ids over the accepted span, and
        accepted positions form a *leading prefix* of the draft positions
        (the chain stops at the first rejected token — standard
        speculative-decoding semantics).

    Args:
        draft_ids: Draft token ids proposed by the draft model.
        target_ids: The *verified* target token ids over the same span — i.e.
            ``target_ids[j]`` is the target model's argmax (greedy) at the
            j-th draft position.  Truncated/empty here means the draft ran past
            the target span (those positions are rejected).
        draft_probs: Optional per-position draft probabilities ``q``
            (sampled/ rejection-sampling mode).  When ``None`` the function
            runs in **greedy** mode (accept iff ``draft_ids[j] == target_ids[j]``).
        target_probs: Optional per-position target probabilities ``p`` of the
            draft token (sampled mode only; required when ``draft_probs`` is
            given).
        rng: Optional generator for reproducible sampled-mode decisions.

    Returns:
        ``accepted_mask`` — a boolean list of length ``len(draft_ids)`` where
        ``accepted_mask[j]`` is ``True`` iff the j-th draft token is accepted.
        It is always a *leading prefix*: ``True`` for ``j < k`` and ``False``
        for ``j >= k`` for some ``k`` (the number of accepted tokens).
    """
    n = len(draft_ids)
    mask: list[bool] = []

    # Greedy mode: accept iff the draft token equals the target argmax.
    if draft_probs is None:
        for j in range(n):
            if j < len(target_ids) and int(draft_ids[j]) == int(target_ids[j]):
                mask.append(True)
            else:
                # First mismatch (or draft ran past the target span): stop.
                mask.append(False)
                break
        # Any draft positions beyond the (broken) loop are already omitted;
        # pad the remainder with False so the mask is a full leading prefix.
        mask.extend([False] * (n - len(mask)))
        return mask

    # Sampled mode: proper rejection sampling min(1, p/q) per position.
    if target_probs is None:
        raise ValueError("target_probs is required when draft_probs is given")
    for j in range(n):
        if j >= len(target_ids):
            # Draft ran past the target span -> rejected.
            mask.append(False)
            break
        q = float(draft_probs[j]) if j < len(draft_probs) else 0.0
        p = float(target_probs[j])
        if q <= 0.0:
            mask.append(False)
            break
        acceptance = min(1.0, p / q)
        if rng is not None:
            draw = torch.rand(1, generator=rng).item()
        else:
            draw = torch.rand(1).item()
        if draw < acceptance:
            mask.append(True)
        else:
            mask.append(False)
            break
    mask.extend([False] * (n - len(mask)))
    return mask
