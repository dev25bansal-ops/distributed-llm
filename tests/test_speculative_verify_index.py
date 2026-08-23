"""Regression tests for speculative-decoding verification index alignment.

These tests guard against the off-by-one bug class where a draft token *i*
(position L + i) was verified against the wrong logits slice, silently
corrupting generated output.

The verification oracle is *greedy* decoding (temperature == 0): a correct
speculative decoder must produce exactly the same token sequence as greedy
autoregressive decoding of the target model.  With temperature == 0 the
verifier compares ``target_logits[:, L-1+i, :].argmax()`` to draft token *i*,
so the test deterministically exercises the logits-index math that the
fix corrected.
"""

from __future__ import annotations

import pytest

try:
    import torch
    _ = torch.float16  # canary: real torch always has this; pollution replaces torch with an empty stub
except (ModuleNotFoundError, ImportError, AttributeError) as _e:
    pytest.skip(f"requires working torch / distllm.core.speculative_decoder (not available): {_e}", allow_module_level=True)

import torch

from distllm.core.speculative_decoder import SpeculativeDecoder


# A deterministic target model over a tiny vocab. ``target_seq`` fully defines
# the target distribution: at absolute position ``j`` (0-indexed) the model's
# argmax is ``target_seq[j]``.  Because logits are constructed directly from
# the sequence, ``target_logits[:, j, :].argmax() == target_seq[j]``.
VOCAB = 16


def _make_target_forward(target_seq: list[int]):
    """Return a target_forward callable matching ``target_seq`` as argmax.

    Causal-LM convention: ``target_logits[:, j, :]`` predicts the token at
    position ``j+1``.  So the argmax at index ``j`` must be ``target_seq[j+1]``
    (i.e. the token that *follows* index j).  The token that appears at
    absolute position ``p`` is therefore ``target_seq[p]``.
    """

    # Pad so every index is defined.
    seq = list(target_seq) + [0] * (VOCAB - len(target_seq) % VOCAB)

    def target_forward(input_ids: torch.Tensor, **kwargs):
        # input_ids: (1, T).  Produce logits[:, t, :] == one-hot(seq[t+1]).
        T = input_ids.shape[1]
        logits = torch.full((1, T, VOCAB), -10.0)
        for t in range(T):
            tok = seq[(t + 1) % len(seq)]
            logits[0, t, tok] = 10.0
        return logits

    return target_forward


def _make_draft_forward(draft_seq: list[int]):
    """Draft model that emits ``draft_seq`` greedily, regardless of prefix.

    At each autoregressive step (current prefix length ``n``) it predicts the
    token that belongs at absolute position ``n`` — i.e. ``draft_seq[n]``
    (cycled).  For the "perfect" tests ``draft_seq`` is set to the target's
    own sequence so every proposal matches the target distribution.
    """

    def draft_forward(input_ids: torch.Tensor, **kwargs):
        n = input_ids.shape[1]
        tok = draft_seq[n % len(draft_seq)] if draft_seq else 0
        logits = torch.full((1, 1, VOCAB), -10.0)
        logits[0, 0, tok] = 10.0
        return logits

    return draft_forward


def _greedy_oracle(target_seq: list[int], prompt_len: int, max_new: int) -> list[int]:
    """What greedy decoding of the target produces after the prompt."""
    out = []
    # At step s (0-indexed, generating token #prompt_len+s), the target's
    # argmax at absolute position (prompt_len + s) is seq[prompt_len + s].
    for s in range(max_new):
        pos = prompt_len + s
        tok = target_seq[pos % len(target_seq)] if pos < len(target_seq) else 0
        out.append(tok)
    return out


def _extract_generated(generated: torch.Tensor, prompt_len: int) -> list[int]:
    return generated[0, prompt_len:].tolist()


def test_speculative_matches_greedy_basic():
    """A correct verifier yields exactly the greedy target sequence."""
    prompt = torch.tensor([[1, 2, 3]])
    prompt_len = prompt.shape[1]
    max_new = 6

    # Target argmax sequence (defines the true distribution at every position).
    target_seq = [1, 2, 3, 7, 7, 8, 8, 9, 9, 5, 5, 6, 6, 0]
    # Draft proposals: a mix that matches in some spots, mismatches in others.
    draft_seq = [7, 8, 9, 5, 6, 0]

    sd = SpeculativeDecoder(
        target_forward=_make_target_forward(target_seq),
        draft_forward=_make_draft_forward(draft_seq),
        num_candidates=4,
        temperature=0.0,  # greedy path exercises the off-by-one-sensitive branch
        device="cpu",
    )
    generated = sd.generate(prompt, max_new_tokens=max_new)
    produced = _extract_generated(generated, prompt_len)

    expected = _greedy_oracle(target_seq, prompt_len, max_new)
    assert produced == expected, (
        f"speculative output {produced} != greedy oracle {expected}. "
        "This indicates a logits-index misalignment in _verify_tokens."
    )


def test_speculative_matches_greedy_all_accept():
    """When the draft perfectly matches the target, all drafts are accepted
    and the sequence is still exactly greedy."""
    prompt = torch.tensor([[4, 4, 4]])
    prompt_len = prompt.shape[1]
    max_new = 8

    # Make draft == target predictions so every position is accepted.
    target_seq = [4, 4, 4, 11, 12, 13, 14, 15, 10, 10, 11, 12, 0]
    # draft_seq must predict, at step producing token for pos p, the same as target.
    sd = SpeculativeDecoder(
        target_forward=_make_target_forward(target_seq),
        draft_forward=_make_draft_forward(target_seq[3:]),  # proposals follow target
        num_candidates=5,
        temperature=0.0,
        device="cpu",
    )
    generated = sd.generate(prompt, max_new_tokens=max_new)
    produced = _extract_generated(generated, prompt_len)
    expected = _greedy_oracle(target_seq, prompt_len, max_new)
    assert produced == expected, (
        f"all-accept path diverged: {produced} != {expected}"
    )


def test_verify_tokens_uses_correct_logits_index():
    """Direct unit test of ``_verify_tokens`` index math.

    With a target whose argmax at position j is (j+1) mod VOCAB, a draft token
    *i* at absolute position L+i is correct iff it equals (L+i).  A buggy
    ``prefix_len = prefix.shape[1]`` (no -1) would verify against position
    (L+i+1) instead, so it would reject tokens that should be accepted and
    accept tokens that should be rejected — this asserts the correct behaviour.
    """
    L = 4
    k = 3
    VOCAB_LOCAL = 32
    # logits[j].argmax == (j+1) % VOCAB_LOCAL  -> "perfect next-token predictor"
    T = L + k
    logits = torch.full((1, T, VOCAB_LOCAL), -10.0)
    for j in range(T):
        logits[0, j, (j + 1) % VOCAB_LOCAL] = 10.0

    prefix = torch.arange(L, dtype=torch.long).unsqueeze(0)          # (1, L)
    # Draft token i should be the model's true prediction at abs pos L+i:
    # that is (L + i) % VOCAB_LOCAL.
    draft_tokens = torch.tensor(
        [[(L + i) % VOCAB_LOCAL for i in range(k)]], dtype=torch.long
    )

    sd = SpeculativeDecoder(
        target_forward=lambda x, **k: logits,
        draft_forward=lambda x, **k: logits[:, -1:, :],
        num_candidates=k,
        temperature=0.0,
        device="cpu",
    )
    # full_input is only used in the rejection-sampling (non-greedy) fallback,
    # which is not taken at temperature 0; pass a dummy of correct length.
    full_input = torch.cat([prefix, draft_tokens], dim=1)
    accepted = sd._verify_tokens(prefix, full_input, draft_tokens, logits)

    assert accepted == k, (
        f"expected all {k} draft tokens accepted (they match the target "
        f"distribution), but got {accepted}. Off-by-one in verification index."
    )
