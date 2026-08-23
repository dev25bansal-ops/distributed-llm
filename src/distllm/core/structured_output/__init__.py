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

    Uses a simple character-level state machine to track valid JSON
    structure during generation. At each step, only tokens whose first
    character is valid for the current JSON parse state are allowed.

    This is a simplified approach — it does not validate against a full
    JSON schema, but ensures the output is valid JSON syntax.

    For advanced constraint (grammar, regex, full JSON schema), use
    SchemaConstrainedDecoder from constrained_decoder.py.
    """

    _token_index_cache: dict = {}
    _token_ord_cache: dict = {}
    _build_lock: threading.Lock = threading.Lock()
    _valid_ord_sets: dict[str, tuple[int, ...]] = {}

    def __init__(self, schema: dict | None = None):
        self.schema = schema
        self._state = "object_start"
        self._stack: list[str] = []
        self._in_string = False
        self._escape_next = False
        self._generated = ""
        self._token_first_chars: dict[int, str] | None = None
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        # Precomputed valid-ord tensor for each state (lazily built)
        self._valid_ord_tensors: dict[str, torch.Tensor] = {}

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

    def _get_valid_ord_tensor(self, state: str, in_string: bool) -> torch.Tensor:
        """Return a precomputed tensor of valid character ordinals for the given state.

        Caches the result per (state, in_string) key so repeated calls
        avoid the Python set→tensor conversion.
        """
        cache_key = (state, in_string)
        cached = self._valid_ord_tensors.get(cache_key)
        if cached is not None:
            return cached

        valid_chars = self._valid_next_chars()
        valid_ords = {ord(ch) for ch in valid_chars}
        if state in (
            "after_value", "after_key", "after_colon", "object_value", "array_value"
        ):
            valid_ords.update(ord(ch) for ch in ' \t\n\r')

        tensor = torch.tensor(sorted(valid_ords), dtype=torch.long)
        self._valid_ord_tensors[cache_key] = tensor
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

        valid_ord_tensor = self._get_valid_ord_tensor(self._state, self._in_string)

        # Move tensors to target device for GPU-accelerated comparison
        target = device or "cpu"
        first_ords_dev = first_ords.to(target)
        valid_ords_dev = valid_ord_tensor.to(target)
        is_valid = (first_ords_dev.unsqueeze(1) == valid_ords_dev.unsqueeze(0)).any(dim=1)

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=target)
        mask[:n] = is_valid

        # Escape hatch: if the FSM is in a dead state (no valid token can
        # advance it) or the tokenizer is degenerate (nothing decodes),
        # fall back to an unconstrained mask so the request can still
        # terminate instead of generating forever.
        if not bool(mask.any()):
            mask = torch.ones(vocab_size, dtype=torch.bool, device=target)
        # Allow EOS when the generated JSON is complete so normal
        # termination is possible; block it mid-document so generation
        # cannot end with truncated/invalid JSON (F-098).  An all-allowed
        # escape mask also keeps EOS enabled.
        elif eos_token_id is not None and self.is_complete():
            mask[eos_token_id] = True

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
        if self._in_string and not self._escape_next:
            return set(string.printable) - {'\n', '\r', '\x0b', '\x0c'}

        transitions: dict[str, set[str]] = {
            "object_start": {'"', '{', '['},
            "after_open_brace": {'"', '}', '{', '['},
            "after_key": {':'},
            "after_colon": {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "after_value": {',', '}'},
            "after_comma": {'"'},
            "array_start": {']', '"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "after_array_value": {',', ']'},
            "after_array_comma": {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            # A number may continue with digits / exponent / sign, and may also
            # be terminated by , } ] or whitespace (F-045: previously the FSM
            # had no in_number entry, so multi-digit numbers were truncated).
            "in_number": set('0123456789.eE+-') | {',', '}', ']', ' ', '\t', '\n', '\r'},
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
            return state

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
            if state in ("after_value", "after_array_value", "in_number"):
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

        if state in ("after_colon", "array_start", "after_array_comma"):
            if char in 'tfn':
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
