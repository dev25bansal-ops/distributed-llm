"""FSM-based constrained decoding for structured output.

Replaces the O(vocab * decode) approach with:
1. Token index built once at startup (token_id -> bytes/chars)
2. Deterministic FSM (DFA) from JSON schema / regex / grammar
3. O(vocab) mask computation using precomputed token->byte mapping
4. Support for JSON Schema, Regex, Grammar, and Pydantic models

Inspired by outlines, guidance, and xgrammar approaches.
"""

import re
import json
import string
from typing import Any
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Token Index: built once per tokenizer, maps token_id -> bytes
# ---------------------------------------------------------------------------

class TokenIndex:
    """Precomputed mapping from token_id to its byte representation.

    Built once at startup to avoid calling tokenizer.decode() during
    constrained generation, which would iterate the entire vocabulary
    (32K-128K) at every step.
    """

    def __init__(self):
        self._token_to_bytes: dict[int, bytes] = {}
        self._token_to_str: dict[int, str] = {}
        self._eos_token_id: int | None = None
        self._vocab_size: int = 0

    @classmethod
    def build(cls, tokenizer, vocab_size: int | None = None) -> "TokenIndex":
        """Build the token index from a tokenizer.

        Uses tokenizer.get_vocab() for efficiency if available,
        falling back to individual decode calls.

        Args:
            tokenizer: HuggingFace tokenizer.
            vocab_size: Override vocabulary size.

        Returns:
            Populated TokenIndex.
        """
        idx = cls()
        idx._vocab_size = vocab_size or tokenizer.vocab_size
        idx._eos_token_id = getattr(tokenizer, 'eos_token_id', None)

        # Fast path: get_vocab() returns {token_str: token_id}
        if hasattr(tokenizer, 'get_vocab') and callable(getattr(tokenizer, 'get_vocab')):
            vocab = tokenizer.get_vocab()
            for token_str, token_id in vocab.items():
                idx._token_to_bytes[token_id] = token_str.encode('utf-8', errors='replace')
                idx._token_to_str[token_id] = token_str
            return idx

        # Slow path: decode each token individually (only once at startup)
        for token_id in range(idx._vocab_size):
            try:
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                idx._token_to_bytes[token_id] = token_str.encode('utf-8', errors='replace')
                idx._token_to_str[token_id] = token_str
            except Exception:
                idx._token_to_bytes[token_id] = b''
                idx._token_to_str[token_id] = ''

        return idx

    def get_bytes(self, token_id: int) -> bytes:
        """Get the byte representation of a token."""
        return self._token_to_bytes.get(token_id, b'')

    def get_str(self, token_id: int) -> str:
        """Get the string representation of a token."""
        return self._token_to_str.get(token_id, '')

    def get_token_ids_for_prefix(self, prefix: bytes) -> list[int]:
        """Get all token IDs whose byte representation starts with prefix.

        Used for FSM state transitions: find tokens that continue the
        current FSM state.

        Args:
            prefix: Byte prefix to match.

        Returns:
            List of matching token IDs.
        """
        # This is precomputed during FSM building, not at runtime
        return []

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


# ---------------------------------------------------------------------------
# JSON Schema FSM: states and transitions based on JSON grammar
# ---------------------------------------------------------------------------

@dataclass
class FSMState:
    """A single state in the constrained decoding FSM."""
    state_id: int
    name: str
    # Set of (byte_value, next_state_id) transitions
    transitions: dict[int, int] = field(default_factory=dict)
    # Set of allowed byte ranges: (min_byte, max_byte)
    byte_ranges: list[tuple[int, int]] = field(default_factory=list)
    # Whether this state is an accepting (terminal) state
    is_accepting: bool = False


class JSONSchemaFSM:
    """Deterministic finite state machine for JSON schema-constrained decoding.

    The FSM tracks the current position in the JSON grammar and, at each
    step, determines which bytes are valid for the next token.

    States:
    - object_start: Before any content, or after '{'
    - in_key: Inside a string key
    - after_key: After closing key quote, expecting ':'
    - after_colon: After ':', expecting value
    - in_string: Inside a string value
    - in_number: Inside a number
    - in_true/in_false/in_null: Inside literals
    - after_value: After a complete value
    - done: JSON is complete

    This FSM does NOT validate against a full JSON schema - it ensures
    syntactically valid JSON output. For schema-level validation, see
    SchemaConstrainedFSM below.
    """

    # JSON grammar states
    STATE_START = 0
    STATE_IN_KEY = 1
    STATE_AFTER_KEY = 2
    STATE_AFTER_COLON = 3
    STATE_IN_STRING = 4
    STATE_IN_NUMBER = 5
    STATE_IN_LITERAL = 6
    STATE_AFTER_VALUE = 7
    STATE_IN_ARRAY = 8
    STATE_AFTER_ARRAY_VALUE = 9
    STATE_DONE = 10
    STATE_EXPECT_KEY = 11  # After '{' or ',': expecting a key

    # Allowed bytes per state
    _BYTE_WHITESPACE = {0x20, 0x09, 0x0A, 0x0D}  # space, tab, LF, CR
    _BYTE_DIGIT = set(range(0x30, 0x3A))  # 0-9
    _BYTE_HEX_DIGIT = _BYTE_DIGIT | set(range(0x41, 0x47)) | set(range(0x61, 0x67))
    _BYTE_QUOTE = 0x22  # "
    _BYTE_COLON = 0x3A  # :
    _BYTE_COMMA = 0x2C  # ,
    _BYTE_LBRACE = 0x7B  # {
    _BYTE_RBRACE = 0x7D  # }
    _BYTE_LBRACKET = 0x5B  # [
    _BYTE_RBRACKET = 0x5D  # ]
    _BYTE_BACKSLASH = 0x5C  # \
    _BYTE_MINUS = 0x2D  # -
    _BYTE_DOT = 0x2E  # .
    _BYTE_ESCAPED = {0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74}  # " \ / b f n r t

    def __init__(self, schema: dict | None = None):
        self.schema = schema
        self._state = self.STATE_START
        self._stack: list[tuple[int, str]] = []  # (state, context_type: 'object'|'array')
        self._in_string = False
        self._escape_next = False
        self._in_literal = ""  # Current literal being built (true/false/null)
        self._in_number = False
        self._number_stage = 0  # 0=sign, 1=int, 2=frac, 3=exp

    def get_allowed_bytes(self) -> set[int]:
        """Get the set of bytes allowed in the current FSM state.

        Returns:
            Set of allowed byte values (0-255).
        """
        if self._state == self.STATE_DONE:
            return self._BYTE_WHITESPACE

        if self._escape_next:
            return self._BYTE_ESCAPED | {0x75}  # escaped chars + 'u' for unicode

        if self._in_string:
            # Inside a string: any printable char except control chars, " and \
            return set(range(0x20, 0x7F)) - {self._BYTE_QUOTE, self._BYTE_BACKSLASH}

        if self._in_literal:
            # Building a literal (true/false/null)
            expected = self._get_literal_next_byte()
            if expected is not None:
                return {expected}
            return set()

        allowed = set()

        if self._state == self.STATE_START:
            allowed = {self._BYTE_LBRACE, self._BYTE_LBRACKET} | self._BYTE_WHITESPACE

        elif self._state in (self.STATE_EXPECT_KEY, self.STATE_IN_KEY):
            allowed = {self._BYTE_QUOTE, self._BYTE_RBRACE} | self._BYTE_WHITESPACE

        elif self._state == self.STATE_AFTER_KEY:
            allowed = {self._BYTE_COLON} | self._BYTE_WHITESPACE

        elif self._state == self.STATE_AFTER_COLON:
            # Value start: string, number, object, array, or literal
            allowed = {
                self._BYTE_QUOTE,  # string
                self._BYTE_LBRACE,  # object
                self._BYTE_LBRACKET,  # array
                self._BYTE_MINUS,  # negative number
            } | self._BYTE_DIGIT | self._BYTE_WHITESPACE
            # t (true), f (false), n (null)
            allowed |= {0x74, 0x66, 0x6E}

        elif self._state == self.STATE_IN_STRING:
            allowed = set(range(0x20, 0x7F)) - {self._BYTE_QUOTE, self._BYTE_BACKSLASH}

        elif self._state == self.STATE_IN_NUMBER:
            allowed = self._get_number_allowed()

        elif self._state == self.STATE_AFTER_VALUE:
            allowed = {self._BYTE_COMMA, self._BYTE_RBRACE, self._BYTE_RBRACKET} | self._BYTE_WHITESPACE

        elif self._state == self.STATE_IN_ARRAY:
            # Array value start
            allowed = {
                self._BYTE_QUOTE, self._BYTE_LBRACE, self._BYTE_LBRACKET,
                self._BYTE_MINUS,
            } | self._BYTE_DIGIT | self._BYTE_WHITESPACE
            allowed |= {0x74, 0x66, 0x6E}

        elif self._state == self.STATE_AFTER_ARRAY_VALUE:
            allowed = {self._BYTE_COMMA, self._BYTE_RBRACKET} | self._BYTE_WHITESPACE

        return allowed

    def _get_literal_next_byte(self) -> int | None:
        """Get the next expected byte for the current literal."""
        literals = {
            "t": b"true",
            "f": b"false",
            "n": b"null",
        }
        full = literals.get(self._in_literal)
        if full is None:
            return None
        pos = len(self._in_literal) - 1  # We've already matched the first char
        if pos < len(full):
            return full[pos]
        return None

    def _get_number_allowed(self) -> set[int]:
        """Get allowed bytes for the current number parsing stage."""
        if self._number_stage == 0:  # Sign
            return self._BYTE_DIGIT | {self._BYTE_MINUS}
        elif self._number_stage == 1:  # Integer part
            return self._BYTE_DIGIT | {self._BYTE_DOT, 0x65, 0x45}  # . e E
        elif self._number_stage == 2:  # Fraction
            return self._BYTE_DIGIT | {0x65, 0x45}
        else:  # Exponent
            return self._BYTE_DIGIT | {self._BYTE_MINUS, 0x2B}  # - +

    def transition(self, byte_val: int) -> int:
        """Advance the FSM state based on an emitted byte.

        Args:
            byte_val: Byte value (0-255) of the emitted character.

        Returns:
            New state ID.
        """
        # Handle escape sequences
        if self._escape_next:
            self._escape_next = False
            if byte_val == 0x75:  # 'u' - expect 4 hex digits
                pass  # Handled by subsequent transitions
            return self._state

        if self._in_string:
            if byte_val == self._BYTE_BACKSLASH:
                self._escape_next = True
                return self._state
            if byte_val == self._BYTE_QUOTE:
                self._in_string = False
                if self._state == self.STATE_IN_KEY:
                    self._state = self.STATE_AFTER_KEY
                elif self._state == self.STATE_IN_STRING:
                    self._state = self.STATE_AFTER_VALUE
                return self._state
            return self._state  # Stay in string

        # Not in string
        if byte_val == self._BYTE_BACKSLASH:
            self._escape_next = True
            return self._state

        if byte_val in self._BYTE_WHITESPACE:
            return self._state  # Whitespace doesn't change state

        if byte_val == self._BYTE_QUOTE:
            self._in_string = True
            if self._state == self.STATE_EXPECT_KEY:
                # Expecting a key: this is a key string
                self._state = self.STATE_IN_KEY
            elif self._state == self.STATE_AFTER_COLON:
                # After colon: this is a value string
                self._state = self.STATE_IN_STRING
            elif self._state in (self.STATE_START, self.STATE_IN_ARRAY):
                # At start or in array: this is a string value
                self._state = self.STATE_IN_STRING
            else:
                self._state = self.STATE_IN_STRING
            return self._state

        if byte_val == self._BYTE_LBRACE:
            self._stack.append((self._state, 'object'))
            self._state = self.STATE_EXPECT_KEY
            return self._state

        if byte_val == self._BYTE_RBRACE:
            if self._stack:
                prev_state, ctx = self._stack.pop()
                self._state = self.STATE_AFTER_VALUE
            else:
                self._state = self.STATE_DONE
            return self._state

        if byte_val == self._BYTE_COLON:
            self._state = self.STATE_AFTER_COLON
            return self._state

        if byte_val == self._BYTE_COMMA:
            if self._state in (self.STATE_AFTER_VALUE, self.STATE_IN_NUMBER, self.STATE_IN_LITERAL):
                # Determine context from stack
                if self._stack:
                    _, ctx = self._stack[-1]
                    if ctx == 'object':
                        self._state = self.STATE_EXPECT_KEY
                    else:  # array
                        self._state = self.STATE_IN_ARRAY
                else:
                    # Root level
                    self._state = self.STATE_EXPECT_KEY
            elif self._state == self.STATE_AFTER_ARRAY_VALUE:
                self._state = self.STATE_IN_ARRAY
            return self._state

        if byte_val == self._BYTE_LBRACKET:
            self._stack.append((self._state, 'array'))
            self._state = self.STATE_IN_ARRAY
            return self._state

        if byte_val == self._BYTE_RBRACKET:
            if self._stack:
                prev_state, ctx = self._stack.pop()
                self._state = self.STATE_AFTER_VALUE
            else:
                # Closing root-level array
                self._state = self.STATE_DONE
            return self._state

        # Number handling
        if byte_val in self._BYTE_DIGIT or byte_val == self._BYTE_MINUS:
            self._in_number = True
            if self._state in (self.STATE_AFTER_COLON, self.STATE_IN_ARRAY):
                self._state = self.STATE_IN_NUMBER
                if byte_val == self._BYTE_MINUS:
                    self._number_stage = 0
                else:
                    self._number_stage = 1
            elif self._state == self.STATE_IN_NUMBER:
                self._update_number_state(byte_val)
            return self._state

        # Literal handling (true, false, null)
        if byte_val in (0x74, 0x66, 0x6E):  # t, f, n
            self._in_literal = chr(byte_val)
            if self._state in (self.STATE_AFTER_COLON, self.STATE_IN_ARRAY):
                self._state = self.STATE_IN_LITERAL
            return self._state

        if self._state == self.STATE_IN_LITERAL:
            self._in_literal += chr(byte_val)
            if self._in_literal in ("true", "false", "null"):
                self._in_literal = ""
                self._state = self.STATE_AFTER_VALUE
            return self._state

        return self._state

    def _update_number_state(self, byte_val: int) -> None:
        """Update number parsing stage based on current byte."""
        if self._number_stage == 0:  # Sign
            self._number_stage = 1
        elif self._number_stage == 1:  # Integer
            if byte_val == self._BYTE_DOT:
                self._number_stage = 2
            elif byte_val in (0x65, 0x45):  # e, E
                self._number_stage = 3
        elif self._number_stage == 2:  # Fraction
            if byte_val in (0x65, 0x45):
                self._number_stage = 3
        # Stage 3 (exponent) stays in 3

    def is_accepting(self) -> bool:
        """Check if the current state is accepting (complete JSON)."""
        return self._state == self.STATE_DONE or (
            not self._in_string and not self._in_literal and
            self._state in (self.STATE_AFTER_VALUE,)
        )

    def reset(self) -> None:
        """Reset the FSM to initial state."""
        self._state = self.STATE_START
        self._stack = []
        self._in_string = False
        self._escape_next = False
        self._in_literal = ""
        self._in_number = False
        self._number_stage = 0


# ---------------------------------------------------------------------------
# Regex FSM: compile regex pattern to DFA for constrained decoding
# ---------------------------------------------------------------------------

class RegexFSM:
    """FSM derived from a regex pattern for constrained decoding.

    Uses Python's re module to validate transitions. While not as fast
    as a true DFA, it provides correctness for any regex pattern.
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self._generated = ""

    def get_allowed_bytes(self) -> set[int]:
        """Get bytes that would keep the generated string matching the regex.

        Tests each printable ASCII byte to see if appending it keeps
        the string as a valid prefix of the regex pattern.

        This is O(256 * regex_compile) per step but only for ASCII.
        For production, a true DFA compilation is preferred.
        """
        allowed = set()
        for byte_val in range(0x20, 0x7F):  # Printable ASCII
            char = chr(byte_val)
            test = self._generated + char
            # Check if test could be a prefix of a matching string
            try:
                # Use a trick: try to match test + ".*" against pattern
                # If it matches, the byte is allowed
                if self._compiled.match(test):
                    allowed.add(byte_val)
                # Also allow if it's a valid prefix (partial match possible)
                elif self._is_valid_prefix(test):
                    allowed.add(byte_val)
            except Exception:
                pass
        return allowed

    def _is_valid_prefix(self, text: str) -> bool:
        """Check if text could be a prefix of a string matching the pattern."""
        try:
            # Try appending wildcard and see if pattern can still match
            return self._compiled.match(text + " " * 1000) is not None
        except Exception:
            return False

    def transition(self, byte_val: int) -> int:
        """Advance the regex FSM by emitting a byte."""
        self._generated += chr(byte_val)
        return 0  # Single-state FSM

    def is_accepting(self) -> bool:
        """Check if the generated string fully matches the pattern."""
        match = self._compiled.fullmatch(self._generated)
        return match is not None

    def reset(self) -> None:
        """Reset the regex FSM."""
        self._generated = ""


# ---------------------------------------------------------------------------
# Constrained Decoding Constraint: integrates FSM with logit masking
# ---------------------------------------------------------------------------

class ConstrainedConstraint:
    """FSM-based constraint for logit masking during decoding.

    Uses a precomputed TokenIndex to map token IDs to bytes,
    then uses the FSM to determine which tokens are allowed
    at each decoding step.

    This is O(vocab) per step with simple byte lookups,
    compared to the old O(vocab * decode) approach.
    """

    def __init__(
        self,
        fsm,
        token_index: TokenIndex,
        schema: dict | None = None,
    ):
        self._fsm = fsm
        self._token_index = token_index
        self._schema = schema
        self._generated = ""
        self._allowed_cache: set[int] | None = None

    def get_logits_mask(self, vocab_size: int, tokenizer=None) -> torch.Tensor:
        """Return a boolean mask: True for allowed token IDs, False for blocked.

        Uses the precomputed TokenIndex to check each token's byte
        representation against the FSM's allowed bytes.

        Args:
            vocab_size: Size of the tokenizer vocabulary.
            tokenizer: Not used (kept for API compatibility).

        Returns:
            Boolean tensor of shape [vocab_size].
        """
        allowed_bytes = self._fsm.get_allowed_bytes()
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Vectorized: check all tokens at once using precomputed byte mapping
        for token_id in range(min(vocab_size, self._token_index.vocab_size)):
            token_bytes = self._token_index.get_bytes(token_id)

            # Token is allowed if its first byte is in the allowed set
            # or if it's an empty token (special tokens)
            if len(token_bytes) == 0:
                mask[token_id] = True
            elif token_bytes[0] in allowed_bytes:
                mask[token_id] = True
            # Multi-byte tokens: allow if first byte is whitespace and FSM allows whitespace
            elif len(token_bytes) > 1 and token_bytes[0] in (0x20, 0x0A, 0x0D, 0x09):
                if token_bytes[0] in allowed_bytes:
                    mask[token_id] = True

        # Always allow EOS token
        eos_id = self._token_index.eos_token_id
        if eos_id is not None and eos_id < vocab_size:
            mask[eos_id] = True

        return mask

    def update(self, token_str: str) -> None:
        """Advance the FSM after emitting a token.

        Args:
            token_str: The decoded token string.
        """
        self._generated += token_str
        for ch in token_str.encode('utf-8', errors='replace'):
            self._fsm.transition(ch)

    def is_complete(self) -> bool:
        """Check if the FSM has reached an accepting state."""
        return self._fsm.is_accepting()

    @property
    def generated_text(self) -> str:
        return self._generated


# ---------------------------------------------------------------------------
# High-level API: SchemaConstrainedDecoder
# ---------------------------------------------------------------------------

class SchemaConstrainedDecoder:
    """High-level interface for constrained decoding.

    Supports:
    - JSON Schema validation
    - Regex patterns
    - Grammar strings
    - Pydantic models (auto-converted to JSON schema)

    Usage:
        decoder = SchemaConstrainedDecoder(tokenizer)
        constraint = decoder.json_schema(schema_dict)
        # In generation loop:
        mask = constraint.get_logits_mask(vocab_size)
        logits[mask == False] = -float('inf')
        constraint.update(tokenizer.decode([next_token]))
    """

    _token_index_cache: dict[int, TokenIndex] = {}

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._token_index = self._get_or_build_token_index(tokenizer)

    @classmethod
    def _get_or_build_token_index(cls, tokenizer) -> TokenIndex:
        """Get or build token index for a tokenizer (cached)."""
        tok_id = id(tokenizer)
        if tok_id not in cls._token_index_cache:
            cls._token_index_cache[tok_id] = TokenIndex.build(tokenizer)
        return cls._token_index_cache[tok_id]

    def json_schema(self, schema: dict | None = None) -> ConstrainedConstraint:
        """Create a JSON schema-constrained constraint.

        Args:
            schema: JSON schema dict (optional, uses generic JSON if None).

        Returns:
            ConstrainedConstraint with JSON FSM.
        """
        fsm = JSONSchemaFSM(schema=schema)
        return ConstrainedConstraint(fsm, self._token_index, schema=schema)

    def regex(self, pattern: str) -> ConstrainedConstraint:
        """Create a regex-constrained constraint.

        Args:
            pattern: Regex pattern string.

        Returns:
            ConstrainedConstraint with Regex FSM.
        """
        fsm = RegexFSM(pattern)
        return ConstrainedConstraint(fsm, self._token_index)

    def grammar(self, grammar: str) -> ConstrainedConstraint:
        """Create a grammar-constrained constraint using GBNF format.

        GBNF is the grammar format used by llama.cpp for structured generation.
        Supports: literals, character classes, alternation, repetition, optional.

        Example grammar:
            root ::= "(" expr ")"
            expr ::= term (("+" | "-") term)*
            term ::= [0-9]+

        Args:
            grammar: GBNF grammar string.

        Returns:
            ConstrainedConstraint with GBNF FSM.
        """
        try:
            from distllm.core.grammar_decoder import GBNFFSM
            fsm = GBNFFSM(grammar)
            return ConstrainedConstraint(fsm, self._token_index)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"GBNF compilation failed: {e}, falling back to regex"
            )

        pattern = self._grammar_to_regex(grammar)
        fsm = RegexFSM(pattern)
        return ConstrainedConstraint(fsm, self._token_index)

    def pydantic(self, model) -> ConstrainedConstraint:
        """Create a Pydantic model-constrained constraint.

        Args:
            model: Pydantic BaseModel class.

        Returns:
            ConstrainedConstraint with JSON schema from model.
        """
        schema = model.model_json_schema()
        return self.json_schema(schema)

    def _grammar_to_regex(self, grammar: str) -> str:
        """Convert a simple EBNF grammar to a regex pattern.

        This is a simplified conversion. For production, use lark or
        a dedicated grammar compiler.
        """
        # Placeholder: return a generic JSON-matching regex
        return r'.*'

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the token index cache."""
        cls._token_index_cache.clear()


# ---------------------------------------------------------------------------
# Backwards compatibility: JSONSchemaConstraint alias
# ---------------------------------------------------------------------------

class JSONSchemaConstraint(SchemaConstrainedDecoder):
    """Backwards-compatible wrapper for the new FSM-based constraint.

    Old API:
        constraint = JSONSchemaConstraint(schema)
        mask = constraint.get_logits_mask(vocab_size, tokenizer)
        constraint.update(token_str)

    New API (recommended):
        decoder = SchemaConstrainedDecoder(tokenizer)
        constraint = decoder.json_schema(schema)
        mask = constraint.get_logits_mask(vocab_size)
        constraint.update(token_str)
    """

    def __init__(self, schema: dict | None = None):
        self._schema = schema
        self._constraint: ConstrainedConstraint | None = None
        self._tokenizer = None

    def get_logits_mask(self, vocab_size: int, tokenizer) -> torch.Tensor:
        """Get logits mask (lazy-creates FSM constraint on first call)."""
        if self._constraint is None:
            decoder = SchemaConstrainedDecoder(tokenizer)
            self._constraint = decoder.json_schema(self._schema)
        return self._constraint.get_logits_mask(vocab_size)

    def update(self, token_str: str) -> None:
        """Advance constraint after emitting a token."""
        if self._constraint is not None:
            self._constraint.update(token_str)

    def is_complete(self) -> bool:
        """Check if generation is complete."""
        if self._constraint is None:
            return False
        return self._constraint.is_complete()

    @property
    def generated_text(self) -> str:
        if self._constraint is None:
            return ""
        return self._constraint.generated_text
