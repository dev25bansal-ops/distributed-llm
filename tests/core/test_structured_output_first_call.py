"""Regression tests for the C3 release-blocker.

On the first constrained-generation call, ``_build_token_index`` returned
``{}`` (it spawned a background thread and returned an empty index), so
``get_logits_mask`` produced an all-tokens-blocked mask and generation
terminated immediately on EOS with empty/invalid JSON.  The index is now
built synchronously, and a degenerate tokenizer falls back to an
unconstrained (all-allowed) mask.
"""

import pytest
import torch

from distllm.core.structured_output import JSONSchemaConstraint


class _FakeTokenizer:
    """Deterministic tokenizer: token i decodes to chars[i]."""
    name_or_path = "fake"
    vocab_size = 32
    eos_token_id = 1

    _CHARS = (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        '{', '}', '"', ':', '[', ']', 't', 'r', 'u', 'e', '0', '1', '2', '3', '4', '5',
    )

    def decode(self, token_ids):
        return self._CHARS[token_ids[0] % len(self._CHARS)]


class _EmptyTokenizer:
    """Degenerate tokenizer: every token decodes to an empty string."""
    name_or_path = "empty"
    vocab_size = 8
    eos_token_id = 1

    def decode(self, token_ids):
        return ''


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


def test_first_call_builds_index_synchronously(fake_tokenizer):
    c = JSONSchemaConstraint(schema={})
    mask = c.get_logits_mask(fake_tokenizer.vocab_size, fake_tokenizer, device="cpu")

    # The index must be fully built synchronously on the first call (not {}).
    assert c._token_first_chars is not None
    assert len(c._token_first_chars) == fake_tokenizer.vocab_size

    # The first-call mask must NOT be all-blocked: more than just EOS allowed.
    assert mask.sum().item() > 1
    # EOS is blocked mid-document (F-098) so generation cannot terminate
    # with truncated JSON; valid JSON-opening tokens are allowed instead.
    assert mask[fake_tokenizer.eos_token_id].item() is False
    assert mask.sum().item() > 1


def test_first_call_mask_is_deterministic(fake_tokenizer):
    c = JSONSchemaConstraint(schema={})
    m1 = c.get_logits_mask(fake_tokenizer.vocab_size, fake_tokenizer, device="cpu")
    m2 = c.get_logits_mask(fake_tokenizer.vocab_size, fake_tokenizer, device="cpu")
    assert torch.equal(m1, m2)


def test_restart_is_forever_usable(fake_tokenizer):
    """A second constraint instance (fresh process sim) also works first call."""
    c2 = JSONSchemaConstraint(schema={})
    mask = c2.get_logits_mask(fake_tokenizer.vocab_size, fake_tokenizer, device="cpu")
    assert mask.sum().item() > 1


def test_degenerate_tokenizer_falls_back_to_noop_mask():
    """No usable first chars -> unconstrained (all allowed), not all blocked."""
    c = JSONSchemaConstraint(schema={})
    tok = _EmptyTokenizer()
    mask = c.get_logits_mask(tok.vocab_size, tok, device="cpu")
    assert mask.all().item() is True