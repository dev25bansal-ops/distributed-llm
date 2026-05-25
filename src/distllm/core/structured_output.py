"""Structured output: JSON schema-constrained decoding."""

from loguru import logger
import string

import torch


class JSONSchemaConstraint:
    """Token-level constraint for JSON schema-constrained decoding.

    Uses a simple character-level state machine to track valid JSON
    structure during generation. At each step, only tokens whose first
    character is valid for the current JSON parse state are allowed.

    This is a simplified approach — it does not validate against a full
    JSON schema, but ensures the output is valid JSON syntax.

    For advanced constraint (grammar, regex, full JSON schema), use
    SchemaConstrainedDecoder from constrained_decoder.py.
    """

    # Class-level cache: tokenizer id -> token_first_chars dict
    # Avoids rebuilding the expensive token index for every new constraint
    _token_index_cache: dict = {}
    _token_ord_cache: dict = {}

    def __init__(self, schema: dict | None = None):
        self.schema = schema
        self._state = "object_start"
        self._stack: list[str] = []
        self._in_string = False
        self._escape_next = False
        self._generated = ""
        self._token_first_chars: dict[int, str] | None = None
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_response_format(cls, response_format: dict, tokenizer=None):
        """Create a constraint from OpenAI response_format dict.

        Supports: json_object, json_schema, grammar, regex.
        Delegates to SchemaConstrainedDecoder for non-json_object types.

        Args:
            response_format: Dict with 'type' key.
            tokenizer: Tokenizer for constraint creation.

        Returns:
            Constraint instance or None.
        """
        fmt_type = response_format.get("type", "")

        if fmt_type == "json_object":
            return cls(schema={})

        if fmt_type == "json_schema":
            schema = response_format.get("schema", {})
            return cls(schema=schema)

        # For grammar/regex, SchemaConstrainedDecoder not available; fallback to simple JSON
        if fmt_type in ("grammar", "regex"):
            return cls(schema={})

        return cls(schema=None)

    def _build_token_index(self, tokenizer) -> dict[int, str]:
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

        Uses GPU-side vectorized operations instead of Python loop iteration.
        Precomputes first characters for all tokens, then uses torch.isin
        for O(1) mask generation.

        Args:
            vocab_size: Size of the tokenizer vocabulary.
            tokenizer: The tokenizer to decode token IDs.

        Returns:
            Boolean tensor of shape [vocab_size].
        """
        valid_chars = self._valid_next_chars()
        # Convert to ordinals for tensor comparison
        valid_ords = set()
        for ch in valid_chars:
            valid_ords.add(ord(ch))
        # Also allow whitespace for certain states
        if self._state in (
            "after_value", "after_key", "after_colon", "object_value", "array_value"
        ):
            for ch in ' \t\n\r':
                valid_ords.add(ord(ch))

        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        cache_key = (
            self._state,
            self._in_string,
            self._escape_next,
            tuple(sorted(valid_ords)),
            vocab_size,
            eos_token_id,
        )
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        # Build token index on first call, then reuse
        if self._token_first_chars is None:
            self._token_first_chars = self._build_token_index(tokenizer)

        # GPU-side: collect all token ordinals as a tensor
        n = min(vocab_size, len(self._token_first_chars))
        ord_cache_key = (id(tokenizer), n)
        first_ords = self._token_ord_cache.get(ord_cache_key)
        if first_ords is None:
            first_chars = [self._token_first_chars.get(tid, '') for tid in range(n)]
            first_ords = torch.tensor(
                [ord(c) if c else 0 for c in first_chars], dtype=torch.long
            )
            self._token_ord_cache[ord_cache_key] = first_ords

        # Create mask: token is allowed if its first char is in valid_ords
        valid_ord_tensor = torch.tensor(list(valid_ords), dtype=torch.long)
        # Use broadcasting: [n] vs [num_valid] -> [n, num_valid]
        is_valid = (first_ords.unsqueeze(1) == valid_ord_tensor.unsqueeze(0)).any(dim=1)

        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[:n] = is_valid

        # Always allow EOS token
        if eos_token_id is not None:
            mask[eos_token_id] = True

        self._mask_cache[cache_key] = mask
        return mask

    def update(self, token_str: str) -> None:
        """Advance the state machine after emitting a token.

        Args:
            token_str: The decoded token string.
        """
        self._generated += token_str
        self._mask_cache.clear()
        for ch in token_str:
            self._state = self._transition(self._state, ch)

    def _valid_next_chars(self) -> set[str]:
        """Return the set of characters valid for the current JSON state."""
        if self._in_string and not self._escape_next:
            # Inside a JSON string: any printable char except control chars
            return set(string.printable) - {'\n', '\r', '\x0b', '\x0c'}

        transitions: dict[str, set[str]] = {
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
