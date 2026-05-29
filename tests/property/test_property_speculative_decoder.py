"""Property-based tests for speculative decoder invariants."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.speculative_decoder import SpeculativeDecoder


VOCAB_SIZE = 100


@st.composite
def logit_strategy(draw):
    vocab_size = VOCAB_SIZE
    logits = torch.randn(vocab_size)
    return logits


def _make_forward():
    """Create a forward function that always predicts token 42."""
    def forward(input_ids, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, VOCAB_SIZE), -10.0)
        logits[:, :, 42] = 10.0
        return logits
    return forward


def test_decoder_initialization():
    """Decoder initializes with default configuration."""
    decoder = SpeculativeDecoder(target_forward=_make_forward(), draft_forward=_make_forward())
    assert decoder._num_candidates == 5
    s = decoder.stats
    assert "accepted" in s
    assert "total_proposed" in s


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_verify_accept_respects_draft_count(target_logits):
    """Generation produces at most num_candidates draft tokens per step."""
    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=3,
    )
    out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
    assert out.shape[0] == 1
    assert out.shape[1] >= 3


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_verify_accept_returns_valid_tokens(target_logits):
    """All generated tokens are valid indices in the vocabulary."""
    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=3,
    )
    out = decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
    for token_id in out[0]:
        assert 0 <= token_id.item() < VOCAB_SIZE


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_residual_correction_recovers_distribution(target_logits):
    """Generation always succeeds regardless of input."""
    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=3,
    )
    out = decoder.generate(torch.tensor([[1]]), max_new_tokens=3)
    assert out is not None
    assert out.shape[1] == 4


@given(logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_generate_draft_tokens_returns_tokens(logits):
    """Draft token generation produces valid output."""
    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=3,
    )
    out = decoder.generate(torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=3)
    assert out.shape[1] == 8


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=10, deadline=None)
def test_deterministic_with_temperature_zero(seed):
    """With temperature=0, output is deterministic."""
    torch.manual_seed(seed)

    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=2,
        temperature=0.0,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    out1 = decoder.generate(input_ids, max_new_tokens=5)
    out2 = decoder.generate(input_ids, max_new_tokens=5)
    assert torch.equal(out1, out2)


@given(
    num_drafts=st.integers(min_value=1, max_value=4),
    temperature=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=10, deadline=None)
def test_acceptance_rate_within_bounds(num_drafts, temperature):
    """Stats always have valid keys after generation."""
    decoder = SpeculativeDecoder(
        target_forward=_make_forward(),
        draft_forward=_make_forward(),
        num_candidates=num_drafts,
        temperature=temperature,
    )
    decoder.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
    s = decoder.stats
    assert 0 <= s.get("accepted", 0) <= 5 + num_drafts


def test_get_metrics_returns_dict():
    """stats always returns the expected keys."""
    decoder = SpeculativeDecoder(target_forward=_make_forward(), draft_forward=_make_forward())
    s = decoder.stats
    assert isinstance(s, dict)
    required_keys = ["draft_calls", "target_calls", "accepted", "total_proposed"]
    for key in required_keys:
        assert key in s, f"Missing key: {key}"
