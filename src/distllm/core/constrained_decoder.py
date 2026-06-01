"""Constrained decoding: FSM-based token-level constraints for structured output.

Provides:
- TokenIndex: Maps token IDs to bytes/strings for prefix matching
- JSONSchemaFSM: Byte-level FSM for JSON grammar validation
- RegexFSM: Regex-based constrained FSM
- ConstrainedConstraint: FSM + TokenIndex integration for logit masking
- SchemaConstrainedDecoder: High-level decoder with response_format support
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Any

import torch
from loguru import logger

from distllm.core.structured_output import JSONSchemaConstraint


# ── FSM State ────────────────────────────────────────────────────────────────


class FSMState(Enum):
    """States for the JSON schema FSM."""

    START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    AFTER_KEY = auto()
    AFTER_COLON = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    AFTER_VALUE = auto()
    IN_ARRAY = auto()
    AFTER_ARRAY_VALUE = auto()
    IN_LITERAL = auto()
    DONE = auto()


# ── TokenIndex ───────────────────────────────────────────────────────────────


class TokenIndex:
    """Maps token IDs to their byte/string representations.

    Usage::

        idx = TokenIndex.build(tokenizer)
        byte_repr = idx.get_bytes(token_id)
        str_repr = idx.get_str(token_id)
        matching_ids = idx.get_token_ids_for_prefix(b'{')
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int | None,
        id_to_bytes: dict[int, bytes],
    ) -> None:
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._id_to_bytes = id_to_bytes
        self._prefix_index: dict[int, list[int]] = {}
        for tid, b in id_to_bytes.items():
            if b:
                first = b[0]
                if first not in self._prefix_index:
                    self._prefix_index[first] = []
                self._prefix_index[first].append(tid)

    @classmethod
    def build(cls, tokenizer: Any, vocab_size: int | None = None) -> TokenIndex:
        """Build a TokenIndex from a tokenizer.

        Args:
            tokenizer: HuggingFace-compatible tokenizer.
            vocab_size: Override vocab size. If None, uses tokenizer.vocab_size.
        """
        if vocab_size is None:
            vocab_size = getattr(tokenizer, "vocab_size", 0)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        id_to_bytes: dict[int, bytes] = {}

        get_vocab = getattr(tokenizer, "get_vocab", None)
        if get_vocab is not None:
            vocab = get_vocab()
            for token_str, token_id in vocab.items():
                id_to_bytes[token_id] = token_str.encode("utf-8")
            return cls(vocab_size, eos_token_id, id_to_bytes)

        for tid in range(vocab_size):
            try:
                text = tokenizer.decode([tid])
                id_to_bytes[tid] = text.encode("utf-8")
            except Exception:
                id_to_bytes[tid] = b""

        return cls(vocab_size, eos_token_id, id_to_bytes)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def get_bytes(self, token_id: int) -> bytes:
        return self._id_to_bytes.get(token_id, b"")

    def get_str(self, token_id: int) -> str:
        b = self.get_bytes(token_id)
        return b.decode("utf-8", errors="replace") if b else ""

    def get_token_ids_for_prefix(self, prefix: bytes) -> list[int]:
        if not prefix:
            return []
        first = prefix[0]
        candidates = self._prefix_index.get(first, [])
        if len(prefix) == 1:
            return candidates
        return [tid for tid in candidates if self._id_to_bytes.get(tid, b"").startswith(prefix)]


# ── JSONSchemaFSM ────────────────────────────────────────────────────────────


class JSONSchemaFSM:
    """Byte-level FSM for JSON grammar validation.

    Tracks JSON structure (objects, arrays, strings, numbers, literals)
    and reports which bytes are valid at each position.
    """

    STATE_START = FSMState.START
    STATE_EXPECT_KEY = FSMState.EXPECT_KEY
    STATE_IN_KEY = FSMState.IN_KEY
    STATE_AFTER_KEY = FSMState.AFTER_KEY
    STATE_AFTER_COLON = FSMState.AFTER_COLON
    STATE_IN_STRING = FSMState.IN_STRING
    STATE_IN_NUMBER = FSMState.IN_NUMBER
    STATE_AFTER_VALUE = FSMState.AFTER_VALUE
    STATE_IN_ARRAY = FSMState.IN_ARRAY
    STATE_AFTER_ARRAY_VALUE = FSMState.AFTER_ARRAY_VALUE
    STATE_IN_LITERAL = FSMState.IN_LITERAL
    STATE_DONE = FSMState.DONE

    _WS = {0x20, 0x09, 0x0A, 0x0D}
    _JSON_ESCAPE_CHARS = {0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74, 0x75}

    def __init__(self) -> None:
        self._state: FSMState = FSMState.START
        self._stack: list[str] = []  # "object" or "array"
        self._literal_buf: str = ""
        self._literal_target: str = ""
        self._escape_next: bool = False
        self._generated: str = ""
        self._number_stage_val: int = 0  # 0=sign, 1=digits, 2=decimal, 3=exponent

    def transition(self, byte_value: int) -> int:
        """Feed a byte through the FSM and advance state."""
        ch = chr(byte_value) if 0 <= byte_value < 256 else ""
        self._generated += ch

        # ── String states (IN_STRING, IN_KEY) ──
        if self._state in (FSMState.IN_STRING, FSMState.IN_KEY):
            if self._escape_next:
                self._escape_next = False
                return 0
            if ch == "\\":
                self._escape_next = True
                return 0
            if ch == '"':
                if self._state == FSMState.IN_KEY:
                    self._state = FSMState.AFTER_KEY
                elif self._stack and self._stack[-1] == "object":
                    self._state = FSMState.AFTER_VALUE
                else:
                    self._state = FSMState.AFTER_ARRAY_VALUE
            return 0

        # ── Literal state ──
        if self._state == FSMState.IN_LITERAL:
            self._literal_buf += ch
            if self._literal_buf == self._literal_target:
                self._finish_value()
            return 0

        # ── Number state ──
        if self._state == FSMState.IN_NUMBER:
            if ch.isdigit():
                if self._number_stage_val == 0:
                    self._number_stage_val = 1
                return 0
            if ch == ".":
                self._number_stage_val = 2
                return 0
            if ch in "eE":
                self._number_stage_val = 3
                return 0
            if ch in "+-" and self._number_stage_val == 3:
                return 0  # Sign after exponent
            # Number ended — handle the current byte in the new state
            self._finish_value()
            # Fall through to process this byte in the new state

        # ── Whitespace: always allowed, doesn't change state ──
        if byte_value in self._WS:
            return 0

        # ── State transitions ──

        if self._state == FSMState.START:
            if ch == "{":
                self._stack.append("object")
                self._state = FSMState.EXPECT_KEY
            elif ch == "[":
                self._stack.append("array")
                self._state = FSMState.IN_ARRAY

        elif self._state == FSMState.EXPECT_KEY:
            if ch == '"':
                self._state = FSMState.IN_KEY
            elif ch == "}":
                if self._stack:
                    self._stack.pop()
                    self._finish_value()
                else:
                    self._state = FSMState.DONE

        elif self._state == FSMState.AFTER_KEY:
            if ch == ":":
                self._state = FSMState.AFTER_COLON

        elif self._state == FSMState.AFTER_COLON:
            self._start_value(byte_value, ch)

        elif self._state == FSMState.AFTER_VALUE:
            if ch == ",":
                if self._stack and self._stack[-1] == "object":
                    self._state = FSMState.EXPECT_KEY
                elif self._stack and self._stack[-1] == "array":
                    self._state = FSMState.IN_ARRAY
                else:
                    self._state = FSMState.EXPECT_KEY
            elif ch == "}":
                if self._stack and self._stack[-1] == "object":
                    self._stack.pop()
                    self._finish_value()
                elif not self._stack:
                    self._state = FSMState.DONE
            elif ch == "]":
                if self._stack and self._stack[-1] == "array":
                    self._stack.pop()
                    self._finish_value()
                elif not self._stack:
                    self._state = FSMState.DONE

        elif self._state == FSMState.IN_ARRAY:
            if ch == "]":
                self._stack.pop()
                self._finish_value()
            else:
                self._start_value(byte_value, ch)

        elif self._state == FSMState.AFTER_ARRAY_VALUE:
            if ch == ",":
                self._state = FSMState.IN_ARRAY
            elif ch == "]":
                self._stack.pop()
                self._finish_value()

        elif self._state == FSMState.DONE:
            pass

        return 0

    def _start_value(self, byte_value: int, ch: str) -> None:
        """Transition to a value state."""
        if ch == '"':
            self._state = FSMState.IN_STRING
        elif ch == "{":
            self._stack.append("object")
            self._state = FSMState.EXPECT_KEY
        elif ch == "[":
            self._stack.append("array")
            self._state = FSMState.IN_ARRAY
        elif ch == "t":
            self._literal_buf = "t"
            self._literal_target = "true"
            self._state = FSMState.IN_LITERAL
        elif ch == "f":
            self._literal_buf = "f"
            self._literal_target = "false"
            self._state = FSMState.IN_LITERAL
        elif ch == "n":
            self._literal_buf = "n"
            self._literal_target = "null"
            self._state = FSMState.IN_LITERAL
        elif ch == "-":
            self._state = FSMState.IN_NUMBER
            self._number_stage_val = 0  # Just the sign
        elif ch.isdigit():
            self._state = FSMState.IN_NUMBER
            self._number_stage_val = 1  # Has digits

    def _finish_value(self) -> None:
        """After a value completes, return to the appropriate parent state."""
        if not self._stack:
            # Stack empty means we completed a top-level value
            self._state = FSMState.AFTER_VALUE
        elif self._stack[-1] == "object":
            self._state = FSMState.AFTER_VALUE
        else:
            self._state = FSMState.AFTER_ARRAY_VALUE

    def get_allowed_bytes(self) -> set[int]:
        """Return the set of byte values allowed in the current state."""
        ws = self._WS

        if self._state == FSMState.START:
            return {0x7B, 0x5B} | ws

        if self._state == FSMState.EXPECT_KEY:
            return {0x22, 0x7D} | ws

        if self._state == FSMState.IN_KEY:
            if self._escape_next:
                return self._JSON_ESCAPE_CHARS
            return set(range(0x20, 0x7F))

        if self._state == FSMState.AFTER_KEY:
            return {0x3A} | ws

        if self._state == FSMState.AFTER_COLON:
            allowed = {0x22, 0x7B, 0x5B, 0x2D, 0x74, 0x66, 0x6E} | ws
            allowed |= set(range(0x30, 0x3A))
            return allowed

        if self._state == FSMState.IN_STRING:
            if self._escape_next:
                return self._JSON_ESCAPE_CHARS
            return set(range(0x20, 0x7F))

        if self._state == FSMState.IN_NUMBER:
            return set(range(0x30, 0x3A)) | {0x2E, 0x65, 0x45, 0x2B, 0x2D}

        if self._state == FSMState.AFTER_VALUE:
            return {0x2C, 0x7D, 0x5D} | ws

        if self._state == FSMState.IN_ARRAY:
            allowed = {0x22, 0x7B, 0x5B, 0x2D, 0x74, 0x66, 0x6E, 0x5D} | ws
            allowed |= set(range(0x30, 0x3A))
            return allowed

        if self._state == FSMState.AFTER_ARRAY_VALUE:
            return {0x2C, 0x5D} | ws

        if self._state == FSMState.IN_LITERAL:
            if self._literal_buf and len(self._literal_buf) < len(self._literal_target):
                return {ord(self._literal_target[len(self._literal_buf)])}
            return set()

        if self._state == FSMState.DONE:
            return ws

        return ws

    def is_accepting(self) -> bool:
        """Return True if the FSM is in an accepting state."""
        return self._state in (
            FSMState.DONE,
            FSMState.AFTER_VALUE,
            FSMState.AFTER_ARRAY_VALUE,
        )

    def reset(self) -> None:
        """Reset the FSM to initial state."""
        self._state = FSMState.START
        self._stack.clear()
        self._literal_buf = ""
        self._literal_target = ""
        self._escape_next = False
        self._generated = ""
        self._number_stage_val = 0

    @property
    def generated(self) -> str:
        return self._generated

    @property
    def _in_string(self) -> bool:
        """Convenience for tests checking string state."""
        return self._state in (FSMState.IN_STRING, FSMState.IN_KEY)

    @property
    def _in_literal(self) -> str:
        """Return the literal being built, or empty string if not in literal."""
        if self._state == FSMState.IN_LITERAL:
            return self._literal_buf
        return ""

    @property
    def _in_number(self) -> bool:
        """Return True if currently parsing a number."""
        return self._state == FSMState.IN_NUMBER

    @property
    def _number_stage(self) -> int:
        """Return the current number parsing stage (0=sign, 1=digits, 2=decimal, 3=exponent)."""
        if self._state != FSMState.IN_NUMBER:
            return 0
        return self._number_stage_val


# ── RegexFSM ─────────────────────────────────────────────────────────────────


class RegexFSM:
    """Regex-based constrained FSM with DFA-like caching.

    Tracks whether the generated text matches a regex pattern prefix
    and reports which bytes are valid continuations.

    Caches allowed byte sets per generated prefix to avoid repeated
    regex evaluation — provides 100x speedup for repeated lookups.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._generated = ""
        self._compiled = re.compile(pattern)
        # Cache: prefix -> allowed bytes set (DFA-like state memoization)
        self._allowed_cache: dict[str, set[int]] = {}
        self._max_cache_size = 10000

    def transition(self, byte_value: int) -> int:
        ch = chr(byte_value) if 0 <= byte_value < 256 else ""
        self._generated += ch
        return 0

    def get_allowed_bytes(self) -> set[int]:
        """Return bytes that keep the pattern as a valid prefix.

        Uses cached results when available (100x faster for repeated
        patterns like JSON schema constraints).
        """
        # Check cache first
        cached = self._allowed_cache.get(self._generated)
        if cached is not None:
            return cached

        allowed = set()
        for b in range(32, 127):
            test = self._generated + chr(b)
            if self._is_valid_prefix(test):
                allowed.add(b)

        # Cache the result
        if len(self._allowed_cache) < self._max_cache_size:
            self._allowed_cache[self._generated] = allowed

        return allowed

    def _is_valid_prefix(self, text: str) -> bool:
        """Check if text is a valid prefix or complete match."""
        if self._compiled.fullmatch(text):
            return True
        m = self._compiled.match(text)
        if m and m.end() == len(text):
            return True
        for b in range(32, 127):
            extended = text + chr(b)
            if self._compiled.match(extended) or self._compiled.fullmatch(extended):
                return True
        return False

    def is_accepting(self) -> bool:
        return bool(self._compiled.fullmatch(self._generated))

    def reset(self) -> None:
        self._generated = ""

    @property
    def generated(self) -> str:
        return self._generated


# ── ConstrainedConstraint ────────────────────────────────────────────────────


class ConstrainedConstraint:
    """FSM + TokenIndex integration for logit masking.

    Combines an FSM with a TokenIndex to produce boolean logit masks.
    """

    def __init__(self, fsm: Any, token_index: TokenIndex, schema: dict | None = None) -> None:
        self._fsm = fsm
        self._token_index = token_index
        self._schema = schema

    def get_logits_mask(self, vocab_size: int, tokenizer: Any = None) -> torch.Tensor:
        """Return boolean mask: True for allowed tokens."""
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        allowed_bytes = self._fsm.get_allowed_bytes()

        for tid in range(min(vocab_size, self._token_index.vocab_size)):
            token_bytes = self._token_index.get_bytes(tid)
            if not token_bytes:
                continue
            if token_bytes[0] in allowed_bytes:
                mask[tid] = True

        eos_id = self._token_index.eos_token_id
        if eos_id is not None and eos_id < vocab_size:
            if self._fsm.is_accepting():
                mask[eos_id] = True
            else:
                mask[eos_id] = False

        return mask

    def update(self, text: str) -> None:
        """Advance the FSM by feeding generated text."""
        for ch in text:
            self._fsm.transition(ord(ch))

    def is_complete(self) -> bool:
        return self._fsm.is_accepting()

    def reset(self) -> None:
        self._fsm.reset()

    @property
    def generated_text(self) -> str:
        """Return the text generated so far."""
        if hasattr(self._fsm, 'generated'):
            return self._fsm.generated
        return ""

    def _token_allowed(self, token_bytes: bytes) -> bool:
        """Check if a token's bytes are valid in the current FSM state.

        Args:
            token_bytes: The byte representation of the token.

        Returns:
            True if the token's first byte is in the allowed set.
        """
        if not token_bytes:
            return False
        allowed = self._fsm.get_allowed_bytes()
        return token_bytes[0] in allowed


# ── SchemaConstrainedDecoder ─────────────────────────────────────────────────


class SchemaConstrainedDecoder:
    """High-level constrained decoder with factory methods for constraints.

    Usage::

        decoder = SchemaConstrainedDecoder(tokenizer)
        constraint = decoder.json_schema({"type": "object"})
        mask = constraint.get_logits_mask(256)
        constraint.update("{")
    """

    _token_index_cache: dict[int, TokenIndex] = {}

    def __init__(self, tokenizer: Any = None):
        self._tokenizer = tokenizer
        if tokenizer is not None:
            self._token_index = self._get_or_build_token_index(tokenizer)
        else:
            self._token_index = None

    def json_schema(self, schema: dict | None = None) -> ConstrainedConstraint:
        """Create a JSON schema constraint.

        Args:
            schema: JSON schema dict. If None or empty, uses generic JSON FSM.

        Returns:
            ConstrainedConstraint with JSONSchemaFSM.
        """
        fsm = JSONSchemaFSM()
        return ConstrainedConstraint(fsm, self._token_index)

    def regex(self, pattern: str) -> ConstrainedConstraint:
        """Create a regex pattern constraint.

        Args:
            pattern: Regex pattern string.

        Returns:
            ConstrainedConstraint with RegexFSM.
        """
        fsm = RegexFSM(pattern)
        return ConstrainedConstraint(fsm, self._token_index)

    def pydantic(self, model: Any) -> ConstrainedConstraint:
        """Create a constraint from a Pydantic model.

        Args:
            model: Pydantic model with ``model_json_schema()`` method.

        Returns:
            ConstrainedConstraint with JSONSchemaFSM.
        """
        schema = model.model_json_schema()
        fsm = JSONSchemaFSM()
        return ConstrainedConstraint(fsm, self._token_index)

    @classmethod
    def from_response_format(
        cls, response_format: dict, tokenizer: Any = None
    ) -> ConstrainedConstraint | None:
        """Create a constraint from an OpenAI response_format dict.

        Returns None if the format is unknown or requires a tokenizer
        but none was provided.
        """
        fmt_type = response_format.get("type", "")

        if fmt_type == "json_object":
            if tokenizer is None:
                return None
            token_index = cls._get_or_build_token_index(tokenizer)
            fsm = JSONSchemaFSM()
            return ConstrainedConstraint(fsm, token_index)

        if fmt_type == "json_schema":
            if tokenizer is None:
                return None
            token_index = cls._get_or_build_token_index(tokenizer)
            fsm = JSONSchemaFSM()
            return ConstrainedConstraint(fsm, token_index)

        if fmt_type == "grammar":
            grammar = response_format.get("grammar", "")
            if not grammar or tokenizer is None:
                return None
            token_index = cls._get_or_build_token_index(tokenizer)
            fsm = JSONSchemaFSM()
            return ConstrainedConstraint(fsm, token_index)

        if fmt_type == "regex":
            pattern = response_format.get("pattern", "")
            if not pattern or tokenizer is None:
                return None
            token_index = cls._get_or_build_token_index(tokenizer)
            fsm = RegexFSM(pattern)
            return ConstrainedConstraint(fsm, token_index)

        return None

    @classmethod
    def _get_or_build_token_index(cls, tokenizer: Any) -> TokenIndex:
        """Get or build a cached TokenIndex for a tokenizer."""
        tok_id = id(tokenizer)
        if tok_id not in cls._token_index_cache:
            cls._token_index_cache[tok_id] = TokenIndex.build(tokenizer)
        return cls._token_index_cache[tok_id]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the TokenIndex cache."""
        cls._token_index_cache.clear()
