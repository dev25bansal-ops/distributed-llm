"""Constrained decoding: token-level constraints for structured output.

Provides SchemaConstrainedDecoder (backed by JSONSchemaFSM) for grammar,
regex, and JSON schema constraints on generated tokens.
"""

import torch

from distllm.core.structured_output import JSONSchemaConstraint


class ConstrainedConstraint:
    """Base class for token-level constraints during generation.

    Subclasses must implement ``get_logits_mask`` and may implement
    ``update`` to track generated text.
    """

    def get_logits_mask(self, vocab_size: int) -> torch.Tensor:
        """Return boolean mask: True for allowed tokens."""
        raise NotImplementedError

    def update(self, token_str: str) -> None:
        """Advance constraint state based on the generated token text."""


class SchemaConstrainedDecoder(ConstrainedConstraint):
    """Token-level constraint backed by JSONSchemaFSM.

    Delegates to JSONSchemaConstraint for the actual state machine
    and adds convenience classmethod ``from_response_format``.
    """

    def __init__(self, constraint: JSONSchemaConstraint):
        self._constraint = constraint
        self._tokenizer = None

    @classmethod
    def from_response_format(cls, response_format: dict, tokenizer=None) -> "SchemaConstrainedDecoder | None":
        """Create a constraint from an OpenAI response_format dict.

        Supports: json_object, json_schema, grammar, regex.

        Args:
            response_format: Dict with 'type' key.
            tokenizer: Tokenizer for mask generation.

        Returns:
            SchemaConstrainedDecoder or None if type unknown.
        """
        fmt_type = response_format.get("type", "")
        constraint = JSONSchemaConstraint.from_response_format(response_format, tokenizer)
        if constraint is None or constraint.schema is None and fmt_type not in ("json_object",):
            if fmt_type not in ("json_object", "json_schema", "grammar", "regex"):
                return None
        decoder = cls(constraint=constraint)
        decoder._tokenizer = tokenizer
        return decoder

    def get_logits_mask(self, vocab_size: int) -> torch.Tensor:
        """Get allowed token mask from the underlying constraint.

        Blocks EOS when the FSM is not in an accepting (done) state.
        """
        mask = self._constraint.get_logits_mask(vocab_size, self._tokenizer).clone()
        if self._constraint._state != "done" and self._tokenizer is not None:
            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_id is not None and eos_id < len(mask):
                mask[eos_id] = False
        return mask

    def update(self, token_str: str) -> None:
        """Advance the FSM by feeding the generated token text."""
        self._constraint.update(token_str)
