"""Structured output package for constrained generation."""

import string
import threading

import torch
from loguru import logger

from distllm.core.structured_output.engine import StructuredOutputEngine, GenerationResult
from distllm.core.structured_output.config import StructuredOutputConfig
from distllm.core.structured_output.validator import SchemaValidator, OutputRepairer, ValidationResult, RepairResult
from distllm.core.structured_output.streaming import BufferedAccumulator, PartialJSONParser, StructuredStreamHandler, PartialResult


class JSONSchemaConstraint:
    """Token-level constraint for JSON schema-constrained decoding.

    Uses a character-level state machine to track valid JSON structure
    during generation. At each step, only tokens whose first character is
    valid for the current JSON parse state are allowed.

    Nesting is tracked with an explicit container stack: ``{`` / ``[``
    push onto the stack, and a close character only pops the innermost
    matching open. ``done`` is signaled exclusively when the OUTERMOST
    container closes (or a root scalar/string finishes), so nested
    documents such as ``{"a": {"b": [1, 2]}}`` no longer terminate early.

    This enforces valid *JSON syntax* — it does not validate against a
    full JSON schema (the schema layer in :mod:`.validator` does that).
    Known simplifications, consistent with a first-character token mask:
    number grammar is loose (any sequence of digits/.eE+- is accepted
    inside a number) and escape-sequence spelling after ``\\`` is not
    validated.

    For advanced constraint (grammar, regex, full JSON schema), use
    SchemaConstrainedDecoder from constrained_decoder.py.
    """

    # Insignificant whitespace allowed between JSON tokens.
    _WS: frozenset[str] = frozenset(' \t\n\r')
    # First characters of any JSON value.
    _VALUE_START: frozenset[str] = frozenset('"[{tfn-0123456789')

    _token_index_cache: dict = {}
    _token_ord_cache: dict = {}
    _build_lock: threading.Lock = threading.Lock()
    _valid_ord_sets: dict[str, tuple[int, ...]] = {}

    def __init__(self, schema: dict | None = None):
        self.schema = schema
        self._state = "object_start"
        self._stack: list[str] = []  # open '{' / '[' markers, innermost last
        self._in_string = False
        self._escape_next = False
        self._literal_remaining = ""  # rest of true/false/null after 1st char
        self._generated = ""
        self._token_first_chars: dict[int, str] | None = None
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        # Precomputed valid-ord tensor per valid-char set (lazily built)
        self._valid_ord_tensors: dict[frozenset, torch.Tensor] = {}

    @classmethod
    def from_response_format(cls, response_format: dict, tokenizer=None):
        """Create a constraint from OpenAI response_format dict."""
        fmt_type = response_format.get("type", "")

        if fmt_type == "json_object":
            return cls(schema={})

        if fmt_type == "json_schema":
            schema = response_format.get("schema", {})
            return cls(schema=schema)

        if fmt_type in ("grammar", "regex"):
            return cls(schema={})

        return cls(schema=None)

    def _build_token_index(self, tokenizer) -> dict[int, str]:
        """Precompute the first character of every token ID.

        Built synchronously on first use and cached so ``get_logits_mask`` is
        never computed from an empty index.  Previously this spawned a daemon
        thread and returned ``{}`` immediately, which made the first
        constrained generation block every token (only EOS survives) and
        terminate with empty/invalid JSON.
        """
        tok_key = getattr(tokenizer, 'name_or_path', None) or str(id(tokenizer))

        cached = self._token_index_cache.get(tok_key)
        if cached is not None:
            return cached

        with self._build_lock:
            cached = self._token_index_cache.get(tok_key)
            if cached is not None:
                return cached
            vocab_size = getattr(tokenizer, 'vocab_size', 32000)
            index: dict[int, str] = {}
            for token_id in range(vocab_size):
                decoded = tokenizer.decode([token_id])
                index[token_id] = decoded[0] if decoded else ''
            self._token_index_cache[tok_key] = index
            logger.debug(f"Token index built: {vocab_size} tokens for {tok_key}")
            return index

    def _get_valid_ord_tensor(self, valid_chars: frozenset[str]) -> torch.Tensor:
        """Return a precomputed tensor of valid character ordinals.

        Caches the result per valid-character set so repeated calls in the
        same state avoid the Python set→tensor conversion.
        """
        cached = self._valid_ord_tensors.get(valid_chars)
        if cached is not None:
            return cached

        tensor = torch.tensor(sorted(ord(ch) for ch in valid_chars), dtype=torch.long)
        self._valid_ord_tensors[valid_chars] = tensor
        return tensor

    def get_logits_mask(self, vocab_size: int, tokenizer, device: str | torch.device | None = None) -> torch.Tensor:
        """Return a boolean mask: True for allowed token IDs, False for blocked.

        Args:
            vocab_size: Vocabulary size.
            tokenizer: Tokenizer for decoding tokens.
            device: Target device for the mask tensor. If None, uses CPU.
        """
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        cache_key = (
            self._state,
            self._in_string,
            self._escape_next,
            len(self._stack),
            self._stack[-1] if self._stack else None,
            self._literal_remaining,
            vocab_size,
            eos_token_id,
            str(device),
        )
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._token_first_chars is None:
            self._token_first_chars = self._build_token_index(tokenizer)

        if not self._token_first_chars:
            # No token→first-char map available (e.g. empty/odd tokenizer):
            # return a no-op mask (all tokens allowed) rather than one that
            # blocks everything and forces immediate EOS.
            target = device or "cpu"
            return torch.ones(vocab_size, dtype=torch.bool, device=target)

        n = min(vocab_size, len(self._token_first_chars))
        ord_cache_key = (id(tokenizer), n)
        first_ords = self._token_ord_cache.get(ord_cache_key)
        if first_ords is None:
            first_chars = [self._token_first_chars.get(tid, '') for tid in range(n)]
            first_ords = torch.tensor(
                [ord(c) if c else 0 for c in first_chars], dtype=torch.long
            )
            self._token_ord_cache[ord_cache_key] = first_ords

        if not bool(first_ords.any()):
            # No usable first characters (empty/odd tokenizer): return a
            # no-op mask (all tokens allowed) rather than blocking everything.
            target = device or "cpu"
            return torch.ones(vocab_size, dtype=torch.bool, device=target)

        valid_ord_tensor = self._get_valid_ord_tensor(frozenset(self._valid_next_chars()))

        # Move tensors to target device for GPU-accelerated comparison
        target = device or "cpu"
        first_ords_dev = first_ords.to(target)
        valid_ords_dev = valid_ord_tensor.to(target)
        is_valid = (first_ords_dev.unsqueeze(1) == valid_ords_dev.unsqueeze(0)).any(dim=1)

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=target)
        mask[:n] = is_valid

        # Once the OUTERMOST close has been emitted, only EOS may follow, so
        # generation cannot append trailing garbage after valid JSON (F-098).
        # This takes precedence over the escape hatch below ('done' has an
        # empty valid-char set, which would otherwise look like a dead state).
        if self._state == "done" and eos_token_id is not None and eos_token_id < vocab_size:
            mask = torch.zeros_like(mask)
            mask[eos_token_id] = True
        elif not bool(mask.any()):
            # Escape hatch: if the FSM is in a dead state (no valid token can
            # advance it) or the tokenizer is degenerate (nothing decodes),
            # fall back to an unconstrained mask so the request can still
            # terminate instead of generating forever.
            mask = torch.ones(vocab_size, dtype=torch.bool, device=target)

        self._mask_cache[cache_key] = mask
        return mask

    def update(self, token_str: str) -> None:
        """Advance the state machine after emitting a token."""
        self._generated += token_str
        self._mask_cache.clear()
        self._valid_ord_tensors.clear()
        for ch in token_str:
            self._state = self._transition(self._state, ch)

    def _valid_next_chars(self) -> set[str]:
        """Return the set of characters valid for the current JSON state."""
        if self._in_string:
            # Inside a string (including right after a backslash): any
            # printable char may follow; raw line breaks are invalid.
            return set(string.printable) - {'\n', '\r', '\x0b', '\x0c'}

        ws = self._WS
        value_start = self._VALUE_START
        # A number may be terminated by whichever delimiter closes the
        # current context (C3 follow-up: offering the *other* bracket here
        # let mismatched closes slip through mid-number).
        if self._stack and self._stack[-1] == '[':
            number_terminators = {',', ']'}
        elif self._stack:
            number_terminators = {',', '}'}
        else:
            number_terminators = set()  # root number: only whitespace ends it
        transitions: dict[str, set[str]] = {
            # Root value: any JSON value may start here.
            "object_start": value_start | ws,
            "after_open_brace": {'"', '}'} | ws,
            "after_key": {':'} | ws,
            "after_colon": value_start | ws,
            # Value just completed inside an OBJECT (stack top is '{').
            "after_value": {',', '}'} | ws,
            "array_start": value_start | {']'} | ws,
            # Value just completed inside an ARRAY (stack top is '[').
            "after_array_value": {',', ']'} | ws,
            "after_comma": {'"'} | ws,
            "after_array_comma": value_start | ws,  # no trailing commas
            # A number may continue with digits / exponent / sign, and may
            # also be terminated by , } ] or whitespace (F-045).
            "in_number": set('0123456789.eE+-') | number_terminators | ws,
            # Remaining characters of true/false/null: next literal char only.
            "in_literal": {self._literal_remaining[0]} if self._literal_remaining else set(),
            "done": set(),
        }
        return set(transitions.get(self._state, value_start | ws))

    def _enclosing_after_value(self) -> str:
        """State after a VALUE completes or a container closes.

        Depends on the enclosing container: inside an object the next
        token is `,` or `}`; inside an array `,` or `]`; with no open
        container the document is complete.
        """
        if not self._stack:
            return "done"
        return "after_value" if self._stack[-1] == '{' else "after_array_value"

    def _transition(self, state: str, char: str) -> str:
        """Advance the JSON state machine by one character."""
        if state == "done":
            # Absorbing: nothing may follow the outermost close.
            return "done"

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
                return self._enclosing_after_value()
            return state

        if self._literal_remaining:
            # Continuation of true/false/null.  The mask only ever offers
            # the exact next literal character; anything else leaves the
            # state unchanged so direct update() calls cannot corrupt it.
            if char == self._literal_remaining[0]:
                self._literal_remaining = self._literal_remaining[1:]
                if not self._literal_remaining:
                    return self._enclosing_after_value()
            return state

        if char in ' \t\n\r':
            # Insignificant whitespace between tokens; after a number it
            # terminates the number ("1 }" must parse like "1}").
            if state == "in_number":
                return self._enclosing_after_value()
            return state

        if char == '"':
            self._in_string = True
            if state in ("after_open_brace", "after_comma"):
                return "in_string_key"  # object key position
            return "in_string"

        if char == '{':
            self._stack.append('{')
            return "after_open_brace"

        if char == '}':
            # Only pops when an object is actually open (C3: the first '}' at
            # ANY depth used to terminate the whole document).
            if self._stack and self._stack[-1] == '{':
                self._stack.pop()
                return self._enclosing_after_value()
            return state

        if char == ':':
            return "after_colon"

        if char == ',':
            # Route by enclosing container: array elements continue with a
            # value, object entries continue with a key (C3: array commas
            # used to demand a quoted key next).
            if self._stack and self._stack[-1] == '[':
                return "after_array_comma"
            if self._stack and self._stack[-1] == '{':
                return "after_comma"
            return state

        if char == '[':
            self._stack.append('[')
            return "array_start"

        if char == ']':
            if self._stack and self._stack[-1] == '[':
                self._stack.pop()
                return self._enclosing_after_value()
            return state

        if state in ("object_start", "after_colon", "array_start", "after_array_comma"):
            if char == 't':
                self._literal_remaining = "rue"
                return "in_literal"
            if char == 'f':
                self._literal_remaining = "alse"
                return "in_literal"
            if char == 'n':
                self._literal_remaining = "ull"
                return "in_literal"
            if char in '-0123456789':
                return "in_number"
            return state

        if state == "in_number" and char in '0123456789.eE+-':
            return "in_number"

        return state

    def is_complete(self) -> bool:
        """Check if we have generated valid complete JSON.

        True once the OUTERMOST container has closed, or for root-level
        scalars/strings (a bare ``"hi"``, ``true``, or an in-progress root
        number such as ``42`` — a number's end cannot be detected without a
        following delimiter).  Note the EOS gate in ``get_logits_mask`` is
        stricter: it opens only on the outermost close, so multi-digit root
        numbers can still be continued digit-by-digit.
        """
        return self._state == "done" or (
            not self._in_string and not self._stack and self._state == "in_number"
        )

    @property
    def generated_text(self) -> str:
        return self._generated


def validate_structured_output(text: str, schema: dict | None = None) -> dict | None:
    """Validate generated text against a JSON schema."""
    import json

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not schema:
        return obj

    try:
        import jsonschema
        jsonschema.validate(obj, schema)
        return obj
    except ImportError:
        logger.warning(
            "jsonschema not installed — falling back to basic type checking. "
            "Install with: pip install jsonschema"
        )
    except jsonschema.ValidationError:
        return None

    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(obj, dict):
        return None
    if expected_type == "array" and not isinstance(obj, list):
        return None
    if expected_type == "string" and not isinstance(obj, str):
        return None
    if expected_type == "number" and not isinstance(obj, (int, float)):
        return None
    if expected_type == "integer" and not isinstance(obj, int):
        return None
    if expected_type == "boolean" and not isinstance(obj, bool):
        return None

    if expected_type == "object" and isinstance(obj, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in obj:
                return None

    return obj


__all__ = [
    "JSONSchemaConstraint",
    "validate_structured_output",
    "StructuredOutputEngine",
    "GenerationResult",
    "StructuredOutputConfig",
    "SchemaValidator",
    "OutputRepairer",
    "ValidationResult",
    "RepairResult",
    "BufferedAccumulator",
    "PartialJSONParser",
    "StructuredStreamHandler",
    "PartialResult",
]
