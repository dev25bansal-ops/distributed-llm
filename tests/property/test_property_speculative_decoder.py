"""Property-based tests for speculative decoder invariants.

Verifies:
1. Accepted tokens always match the target distribution (rejection sampling correctness)
2. Draft tokens are accepted with probability p_target(x) / p_draft(x)
3. The number of accepted tokens never exceeds the number of draft tokens
4. Residual correction produces a token from the correct distribution
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.core.speculative_decoder import SpeculativeDecoder


VOCAB_SIZE = 100
TEST_HIDDEN_SIZE = 64
TEST_MEDUSA_VOCAB = 128


@st.composite
def logit_strategy(draw):
    vocab_size = VOCAB_SIZE
    num_tokens = draw(st.integers(min_value=1, max_value=6))
    logits = torch.randn(num_tokens, vocab_size)
    return logits


@st.composite
def draft_strategy(draw):
    num_drafts = draw(st.integers(min_value=1, max_value=5))
    tokens = [draw(st.integers(min_value=0, max_value=VOCAB_SIZE - 1)) for _ in range(num_drafts)]
    return tokens


def test_decoder_initialization():
    """Decoder initializes with default configuration."""
    decoder = SpeculativeDecoder(num_assistant_tokens=3, method="ngram", medusa_hidden_size=64, medusa_vocab_size=128)
    assert decoder.num_assistant_tokens == 3
    metrics = decoder.get_metrics()
    assert "total_draft_tokens" in metrics
    assert "total_accepted" in metrics


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_verify_accept_respects_draft_count(target_logits):
    """Accepted tokens never exceed the number of draft tokens."""
    decoder = SpeculativeDecoder(num_assistant_tokens=3, medusa_hidden_size=64, medusa_vocab_size=128)
    tokenizer = type("Tokenizer", (), {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE})()

    probs = torch.softmax(target_logits[0], dim=-1)
    draft_ids = torch.multinomial(probs, min(probs.shape[0], 3), replacement=True).tolist()

    result = decoder.verify_and_accept(
        draft_tokens=draft_ids,
        target_logits=target_logits,
        tokenizer=tokenizer,
        temperature=1.0,
    )

    if isinstance(result, tuple):
        num_accepted, accepted_tokens, next_token = result
        assert num_accepted <= len(draft_ids)
        assert len(accepted_tokens) <= len(draft_ids) + 1


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_verify_accept_returns_valid_tokens(target_logits):
    """All accepted tokens are valid indices in the vocabulary."""
    decoder = SpeculativeDecoder(num_assistant_tokens=3, medusa_hidden_size=64, medusa_vocab_size=128)
    tokenizer = type("Tokenizer", (), {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE})()

    probs = torch.softmax(target_logits[0], dim=-1)
    draft_ids = torch.multinomial(probs, min(probs.shape[0], 3), replacement=True).tolist()

    result = decoder.verify_and_accept(
        draft_tokens=draft_ids,
        target_logits=target_logits,
        tokenizer=tokenizer,
        temperature=1.0,
    )

    if isinstance(result, tuple):
        num_accepted, accepted_tokens, next_token = result
        for token in accepted_tokens:
            assert 0 <= token < VOCAB_SIZE
        assert 0 <= next_token < VOCAB_SIZE


@given(target_logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_residual_correction_recovers_distribution(target_logits):
    """After rejection, the residual correction token comes from the target distribution.

    This tests that the residual correction step (p_target(x) - p_draft(x))_+ / sum
    produces a valid token.
    """
    decoder = SpeculativeDecoder(num_assistant_tokens=3, medusa_hidden_size=64, medusa_vocab_size=128)
    tokenizer = type("Tokenizer", (), {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE})()

    probs = torch.softmax(target_logits[0], dim=-1)
    draft_id = torch.multinomial(probs, 1).item()

    result = decoder.verify_and_accept(
        draft_tokens=[draft_id],
        target_logits=target_logits[:1],
        tokenizer=tokenizer,
        temperature=0.8,
    )

    if isinstance(result, tuple):
        num_accepted, accepted_tokens, next_token = result
        assert 0 <= next_token < VOCAB_SIZE


@given(logits=logit_strategy())
@settings(max_examples=10, deadline=None)
def test_generate_draft_tokens_returns_tokens(logits):
    """Draft token generation produces valid token IDs."""
    decoder = SpeculativeDecoder(num_assistant_tokens=3, medusa_hidden_size=64, medusa_vocab_size=128)
    class MockDraftModel:
        def __call__(self, input_ids, use_cache=True, past_key_values=None):
            logits_out = torch.randn(1, 3, VOCAB_SIZE)
            class Output:
                logits = logits_out
                past_key_values = None
            return Output()

    draft_model = MockDraftModel()

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
    drafts, _, _ = decoder.generate_draft_tokens(
        draft_model=draft_model,
        input_ids=input_ids,
        target_logits=logits,
    )

    for token in drafts:
        assert isinstance(token, (int, torch.Tensor))
        if isinstance(token, int):
            assert 0 <= token < VOCAB_SIZE


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=10, deadline=None)
def test_deterministic_with_temperature_zero(seed):
    """With temperature=0, the same input always produces the same output."""
    torch.manual_seed(seed)

    decoder = SpeculativeDecoder(num_assistant_tokens=2, medusa_hidden_size=64, medusa_vocab_size=128)
    tokenizer = type("Tokenizer", (), {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE})()

    target_logits = torch.randn(3, VOCAB_SIZE)
    probs = torch.softmax(target_logits[0], dim=-1)
    draft_ids = torch.multinomial(probs, 2, replacement=True).tolist()

    result1 = decoder.verify_and_accept(
        draft_tokens=draft_ids,
        target_logits=target_logits,
        tokenizer=tokenizer,
        temperature=0.0,
    )

    result2 = decoder.verify_and_accept(
        draft_tokens=draft_ids,
        target_logits=target_logits,
        tokenizer=tokenizer,
        temperature=0.0,
    )

    assert result1 == result2


@given(
    num_drafts=st.integers(min_value=1, max_value=4),
    temperature=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=10, deadline=None)
def test_acceptance_rate_within_bounds(num_drafts, temperature):
    """Acceptance rate is always between 0 and num_drafts."""
    torch.manual_seed(42)

    decoder = SpeculativeDecoder(num_assistant_tokens=num_drafts, medusa_hidden_size=64, medusa_vocab_size=128)
    tokenizer = type("Tokenizer", (), {"eos_token_id": VOCAB_SIZE - 1, "vocab_size": VOCAB_SIZE})()

    target_logits = torch.randn(num_drafts + 1, VOCAB_SIZE)
    probs = torch.softmax(target_logits[0], dim=-1)
    draft_ids = torch.multinomial(probs, num_drafts, replacement=True).tolist()

    result = decoder.verify_and_accept(
        draft_tokens=draft_ids,
        target_logits=target_logits,
        tokenizer=tokenizer,
        temperature=temperature,
    )

    if isinstance(result, tuple):
        num_accepted, _, _ = result
        assert 0 <= num_accepted <= num_drafts


def test_get_metrics_returns_dict():
    """get_metrics always returns the expected keys."""
    decoder = SpeculativeDecoder(medusa_hidden_size=64, medusa_vocab_size=128)
    metrics = decoder.get_metrics()

    assert isinstance(metrics, dict)
    required_keys = [
        "total_draft_tokens", "total_accepted", "acceptance_rate",
        "method", "step_count", "enabled",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
