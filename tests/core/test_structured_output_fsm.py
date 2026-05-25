"""Tests for the backward-compatible JSONSchemaConstraint wrapper.

JSONSchemaConstraint (from distllm.core.structured_output) is re-exported
from constrained_decoder.py as a backward-compatible wrapper that wraps
the byte-level ConstrainedConstraint / JSONSchemaFSM.

Tests cover the public API: from_response_format, get_logits_mask,
update, is_complete, generated_text.
"""

from unittest.mock import MagicMock
import pytest
import torch

from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.constrained_decoder import SchemaConstrainedDecoder


# ─── Tokenizer Fixture ─────────────────────────────────────────────────────

@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.vocab_size = 256
    tok.eos_token_id = 1
    tok.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
    return tok


# ─── Construction ──────────────────────────────────────────────────────────

class TestConstruction:
    """Tests for JSONSchemaConstraint constructor."""

    def test_construct_with_schema(self):
        """Constructor accepts optional schema."""
        constraint = JSONSchemaConstraint(schema={"type": "object"})
        assert constraint._schema == {"type": "object"}

    def test_construct_without_schema(self):
        """Constructor works without schema."""
        constraint = JSONSchemaConstraint()
        assert constraint._schema is None
        assert constraint._constraint is None

    def test_from_response_format_json_object(self):
        """json_object type without tokenizer returns None."""
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}
        )
        assert constraint is None

    def test_from_response_format_json_object_with_tokenizer(self, tokenizer):
        """json_object type with tokenizer returns constraint."""
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tokenizer
        )
        assert constraint is not None

    def test_from_response_format_json_schema(self):
        """json_schema type without tokenizer returns None."""
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_schema", "schema": {"type": "object"}}
        )
        assert constraint is None

    def test_from_response_format_grammar_no_tokenizer(self):
        """grammar without tokenizer returns None."""
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "grammar", "grammar": 'root ::= "a"'}
        )
        assert constraint is None

    def test_from_response_format_unknown(self):
        """Unknown type returns None."""
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "unknown"}
        )
        assert constraint is None

    def test_from_response_format_empty(self):
        """Empty response_format returns None."""
        constraint = SchemaConstrainedDecoder.from_response_format({})
        assert constraint is None


# ─── get_logits_mask ──────────────────────────────────────────────────────

class TestGetLogitsMask:
    """Tests for get_logits_mask."""

    def test_mask_returns_boolean_tensor(self, tokenizer):
        """get_logits_mask returns a boolean tensor of correct shape."""
        c = JSONSchemaConstraint()
        mask = c.get_logits_mask(256, tokenizer)
        assert mask.shape == (256,)
        assert mask.dtype == torch.bool

    def test_mask_has_some_allowed(self, tokenizer):
        """Initial mask allows some tokens."""
        c = JSONSchemaConstraint()
        mask = c.get_logits_mask(256, tokenizer)
        assert mask.sum() > 0

    def test_lazy_creation(self, tokenizer):
        """Internal constraint is lazily created on first get_logits_mask."""
        c = JSONSchemaConstraint()
        assert c._constraint is None
        c.get_logits_mask(256, tokenizer)
        assert c._constraint is not None

    def test_eos_allowed_in_complete_state(self, tokenizer):
        """EOS token allowed when JSON is complete."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)  # trigger lazy creation
        c.update('{"a": 1}')
        mask = c.get_logits_mask(256, tokenizer)
        eos_id = tokenizer.eos_token_id
        assert mask[eos_id].item() is True

    def test_eos_not_allowed_in_incomplete_state(self, tokenizer):
        """EOS token not allowed when JSON is incomplete."""
        c = JSONSchemaConstraint()
        mask = c.get_logits_mask(256, tokenizer)
        eos_id = tokenizer.eos_token_id
        assert mask[eos_id].item() is False

    def test_mask_evolves_after_update(self, tokenizer):
        """Mask changes after FSM advances."""
        c = JSONSchemaConstraint()
        mask_before = c.get_logits_mask(256, tokenizer)
        c.update("{")
        mask_after = c.get_logits_mask(256, tokenizer)
        assert not torch.equal(mask_before, mask_after)

    def test_mask_requires_tokenizer(self):
        """get_logits_mask needs a valid tokenizer."""
        c = JSONSchemaConstraint()
        with pytest.raises(Exception):
            c.get_logits_mask(256, None)


# ─── update / is_complete ─────────────────────────────────────────────────

class TestUpdateAndComplete:
    """Tests for update() and is_complete()."""

    def test_not_complete_initial(self):
        """Not complete at initial state."""
        c = JSONSchemaConstraint()
        assert not c.is_complete()

    def test_complete_after_json(self, tokenizer):
        """Complete JSON object."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)  # trigger lazy creation
        c.update('{"a": 1}')
        assert c.is_complete()

    def test_complete_empty_object(self, tokenizer):
        """Empty object is complete."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update("{}")
        assert c.is_complete()

    def test_not_complete_unclosed(self, tokenizer):
        """Unterminated string is not complete."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"a": "unfinish')
        assert not c.is_complete()

    def test_update_without_get_logits_mask(self):
        """update() silently does nothing if constraint not yet created."""
        c = JSONSchemaConstraint()
        c.update('{"a": 1}')
        assert c.generated_text == ""
        assert not c.is_complete()

    def test_is_complete_after_internal_update(self, tokenizer):
        """is_complete after multiple updates."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"a": ')
        assert not c.is_complete()
        c.update('1}')
        assert c.is_complete()


# ─── generated_text ───────────────────────────────────────────────────────

class TestGeneratedText:
    """Tests for generated_text property."""

    def test_initial_empty(self):
        """Initial generated text is empty."""
        c = JSONSchemaConstraint()
        assert c.generated_text == ""

    def test_after_updates(self, tokenizer):
        """generated_text accumulates all updates."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"key": ')
        c.update('"value"}')
        assert c.generated_text == '{"key": "value"}'

    def test_without_get_logits_mask(self):
        """Without lazy creation, generated_text is empty."""
        c = JSONSchemaConstraint()
        c.update('{"a": 1}')
        assert c.generated_text == ""


# ─── End-to-end ───────────────────────────────────────────────────────────

class TestEndToEnd:
    """Full integration tests."""

    def test_complete_json_object(self, tokenizer):
        """Complete JSON object produced step by step."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"name": "world", "value": 42}')
        assert c.is_complete()
        assert c.generated_text == '{"name": "world", "value": 42}'

    def test_json_with_array(self, tokenizer):
        """JSON with array."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('[1, 2, 3]')
        assert c.is_complete()

    def test_literals(self, tokenizer):
        """true, false, null values."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"a": true, "b": false, "c": null}')
        assert c.is_complete()

    def test_nested(self, tokenizer):
        """Nested object (using byte-level FSM that handles nesting)."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"a": {"b": 1}}')
        assert c.is_complete()

    def test_escape_sequence(self, tokenizer):
        """String with escape sequence."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"msg": "hello\\nworld"}')
        assert c.is_complete()

    def test_numbers(self, tokenizer):
        """Various number formats."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        c.update('{"a": 42, "b": -5, "c": 3.14, "d": 1e10}')
        assert c.is_complete()

    def test_mask_allows_progress(self, tokenizer):
        """At each step, mask allows at least some tokens."""
        c = JSONSchemaConstraint()
        c.get_logits_mask(256, tokenizer)
        texts = [
            "",
            "{",
            '{"k"',
            '{"k":',
            '{"k": "v"',
        ]
        for text in texts:
            if text:
                c.update(text)
            mask = c.get_logits_mask(256, tokenizer)
            assert mask.sum() > 0, f"No tokens allowed after '{text}'"
