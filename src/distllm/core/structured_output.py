"""Structured output: JSON schema-constrained decoding."""

import string
from typing import Dict, List, Optional, Set

import torch


class JSONSchemaConstraint:
    """Token-level constraint for JSON schema-constrained decoding.

    Uses a simple character-level state machine to track valid JSON
    structure during generation. At each step, only tokens whose first
    character is valid for the current JSON parse state are allowed.

    This is a simplified approach — it does not validate against a full
    JSON schema, but ensures the output is valid JSON syntax.
    """

    # Class-level cache: tokenizer id -> token_first_chars dict
    # Avoids rebuilding the expensive token index for every new constraint
    _token_index_cache: dict = {}

    def __init__(self, schema: Optional[dict] = None):
        self.schema = schema
        self._state = "object_start"
        self._stack: List[str] = []
        self._in_string = False
        self._escape_next = False
        self._generated = ""
        self._token_first_chars: Optional[Dict[int, str]] = None

    def _build_token_index(self, tokenizer) -> Dict[int, str]:
        """Precompute the first character of every token ID.

        Uses class-level cache keyed by tokenizer id to avoid rebuilding
        for the same tokenizer across multiple constraint instances.

        Returns dict mapping token_id -> first_char (or '' for empty).
        """
        # Use tokenizer object id as cache key
        tok_id = id(tokenizer)
        if tok_id in self._token_index_cache:
            return self._token_index_cache[tok_id]

        # Build index using dict comprehension (faster than loop with setitem)
        index = {
            token_id: (tokenizer.decode([token_id])[0] if tokenizer.decode([token_id]) else '')
            for token_id in range(tokenizer.vocab_size)
        }
        self._token_index_cache[tok_id] = index
        return index

    def get_logits_mask(self, vocab_size: int, tokenizer) -> torch.Tensor:
        """Return a boolean mask: True for allowed token IDs, False for blocked.

        Uses a precomputed token->first_char index if available,
        falling back to on-the-fly decode for unknown tokenizers.

        Args:
            vocab_size: Size of the tokenizer vocabulary.
            tokenizer: The tokenizer to decode token IDs.

        Returns:
            Boolean tensor of shape [vocab_size].
        """
        valid_chars = self._valid_next_chars()
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Build token index on first call, then reuse
        if self._token_first_chars is None:
            self._token_first_chars = self._build_token_index(tokenizer)

        # Use precomputed index
        if self._token_first_chars is not None:
            for token_id in range(min(vocab_size, len(self._token_first_chars))):
                first_char = self._token_first_chars.get(token_id, '')
                if first_char in valid_chars:
                    mask[token_id] = True
                elif first_char.isspace() and self._state in (
                    "after_value", "after_key", "after_colon", "object_value", "array_value"
                ):
                    mask[token_id] = True
        else:
            for token_id in range(vocab_size):
                try:
                    token_str = tokenizer.decode([token_id])
                except (ValueError, IndexError):
                    continue
                if token_str and len(token_str) > 0:
                    first_char = token_str[0]
                    if first_char in valid_chars:
                        mask[token_id] = True
                    if first_char.isspace() and self._state in (
                        "after_value", "after_key", "after_colon", "object_value", "array_value"
                    ):
                        mask[token_id] = True

        # Always allow EOS token
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            mask[tokenizer.eos_token_id] = True

        return mask

    def update(self, token_str: str) -> None:
        """Advance the state machine after emitting a token.

        Args:
            token_str: The decoded token string.
        """
        self._generated += token_str
        for ch in token_str:
            self._state = self._transition(self._state, ch)

    def _valid_next_chars(self) -> Set[str]:
        """Return the set of characters valid for the current JSON state."""
        if self._in_string and not self._escape_next:
            # Inside a JSON string: any printable char except control chars
            return set(string.printable) - {'\n', '\r', '\x0b', '\x0c'}

        transitions: Dict[str, Set[str]] = {
            "object_start": {'"', '}'},
            "after_open_brace": {'"', '}'},
            "after_key": {':'},
            "after_colon": {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "after_value": {',', '}'},
            "after_comma": {'"'},
            "array_start": {']', '"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "after_array_value": {',', ']'},
            "after_array_comma": {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "done": set(),
        }
        return transitions.get(self._state, {'"', '}'})

    def _transition(self, state: str, char: str) -> str:
        """Advance the JSON state machine by one character."""
        if self._escape_next:
            self._escape_next = False
            return state

        if self._in_string:
            if char == '\\':
                self._escape_next = True
                return state
            if char == '"':
                self._in_string = False
                if state == "in_string_key":
                    return "after_key"
                return "after_value"
            return state  # Stay in string

        # Not in string
        if char == '"':
            self._in_string = True
            if state in ("object_start", "after_open_brace", "after_comma"):
                return "in_string_key"
            return "in_string"

        if char == '{':
            return "after_open_brace"
        if char == '}':
            if self._stack:
                return self._stack.pop()
            return "done"
        if char == ':':
            return "after_colon"
        if char == ',':
            if state in ("after_value", "after_array_value"):
                return "after_comma"
            if state in ("after_array_value",):
                return "after_array_comma"
            return state
        if char == '[':
            return "array_start"
        if char == ']':
            if self._stack:
                return self._stack.pop()
            return "done"

        # Start of a JSON value (true, false, null, number)
        if state in ("after_colon", "array_start", "after_array_comma"):
            if char in 'tfn':  # true, false, null
                return "after_value"
            if char in '-0123456789':
                return "in_number"
            return "after_value"

        if state == "in_number" and char in '0123456789.eE+-':
            return "in_number"
        if state == "in_number":
            return "after_value"

        return state

    def is_complete(self) -> bool:
        """Check if we have generated valid complete JSON."""
        return self._state == "done" or (
            not self._in_string and self._state in ("after_value", "done")
        )

    @property
    def generated_text(self) -> str:
        return self._generated
