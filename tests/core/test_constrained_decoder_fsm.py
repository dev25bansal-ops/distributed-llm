"""Tests for FSM-level constrained decoding in constrained_decoder.py.

Covers:
- TokenIndex: building, caching, prefix lookup
- JSONSchemaFSM: byte-level state transitions, nesting, literals, numbers
- RegexFSM: prefix matching, allowed bytes, accepting states
- ConstrainedConstraint: logit masking, token validation, EOS handling
"""

import time
from unittest.mock import MagicMock, patch
import pytest
import torch

from distllm.core.constrained_decoder import (
    ConstrainedConstraint,
    JSONSchemaFSM,
    RegexFSM,
    TokenIndex,
)


# ===========================================================================
# TokenIndex
# ===========================================================================

class TestTokenIndex:
    """Tests for the TokenIndex class (token_id → bytes/str mapping)."""

    def _make_mock_tokenizer(self, vocab_size: int = 128):
        """Create a mock tokenizer with get_vocab() fast path."""
        tok = MagicMock()
        tok.vocab_size = vocab_size
        tok.eos_token_id = 1
        # Fast path: get_vocab returns {token_str: token_id}
        tok.get_vocab.return_value = {chr(i): i for i in range(32, vocab_size)} if vocab_size > 32 else {}
        tok.decode.return_value = ""
        return tok

    def _make_slow_tokenizer(self, vocab_size: int = 50):
        """Create a mock tokenizer without get_vocab (slow path)."""
        tok = MagicMock(spec=['vocab_size', 'eos_token_id', 'decode'])
        tok.vocab_size = vocab_size
        tok.eos_token_id = 0
        # decode returns chr(id) for single token
        def decode_fn(ids, **kw):
            if isinstance(ids, list) and len(ids) == 1:
                i = ids[0]
                if 32 <= i < 127:
                    return chr(i)
                return f"<tok{i}>"
            return ""
        tok.decode.side_effect = decode_fn
        return tok

    def test_build_fast_path(self):
        """TokenIndex.build uses get_vocab() when available."""
        tok = self._make_mock_tokenizer(128)
        idx = TokenIndex.build(tok)
        assert idx.vocab_size == 128
        assert idx.eos_token_id == 1
        # 'A' (chr(65)) should map to b'A'
        assert idx.get_bytes(65) == b'A'
        assert idx.get_str(65) == 'A'

    def test_build_slow_path(self):
        """TokenIndex.build falls back to decode() without get_vocab()."""
        tok = self._make_slow_tokenizer(50)
        idx = TokenIndex.build(tok)
        assert idx.vocab_size == 50
        assert idx.eos_token_id == 0

    def test_get_bytes_unknown_token(self):
        """get_bytes returns empty bytes for unknown token ID."""
        tok = self._make_mock_tokenizer(100)
        idx = TokenIndex.build(tok)
        assert idx.get_bytes(999) == b''

    def test_get_str_unknown_token(self):
        """get_str returns empty string for unknown token ID."""
        tok = self._make_mock_tokenizer(100)
        idx = TokenIndex.build(tok)
        assert idx.get_str(999) == ''

    def test_get_token_ids_for_prefix(self):
        """get_token_ids_for_prefix finds tokens by byte prefix."""
        tok = self._make_mock_tokenizer(128)
        idx = TokenIndex.build(tok)
        # Tokens for 'A' through 'F' (chr(65)-chr(70)) start with b'A' (65)
        # But actually token 'A' is at id 65 with bytes b'A'
        results = idx.get_token_ids_for_prefix(b'A')
        assert 65 in results
        # Token chr(65) = 'A', so b'A' prefix matches b'A'

    def test_get_token_ids_for_prefix_empty(self):
        """get_token_ids_for_prefix returns empty list for no match."""
        tok = self._make_mock_tokenizer(50)
        idx = TokenIndex.build(tok)
        # Token 32 (' ') has byte b' '
        results = idx.get_token_ids_for_prefix(b'XYZ')
        assert results == []

    def test_vocab_size_property(self):
        """vocab_size returns the configured vocabulary size."""
        tok = self._make_mock_tokenizer(256)
        idx = TokenIndex.build(tok)
        assert idx.vocab_size == 256

    def test_eos_token_id_none(self):
        """eos_token_id is None when tokenizer has none."""
        tok = self._make_mock_tokenizer(100)
        tok.eos_token_id = None
        idx = TokenIndex.build(tok)
        assert idx.eos_token_id is None

    def test_utf8_multi_byte_tokens(self):
        """Tokens with multi-byte UTF-8 are stored correctly."""
        tok = MagicMock()
        tok.vocab_size = 3
        tok.eos_token_id = 0
        tok.get_vocab.return_value = {
            "hello": 0,
            "\u00e9": 1,  # é (2 bytes in UTF-8)
            "\U0001f600": 2,  # 😀 (4 bytes in UTF-8)
        }
        idx = TokenIndex.build(tok)
        assert idx.get_bytes(0) == b'hello'
        assert idx.get_bytes(1) == b'\xc3\xa9'
        assert idx.get_bytes(2) == b'\xf0\x9f\x98\x80'
        assert idx.get_str(2) == '\U0001f600'

    def test_get_vocab_exception_propagates(self):
        """Tokenizer that raises in get_vocab propagates the error."""
        tok = MagicMock()
        tok.vocab_size = 5
        tok.eos_token_id = 0
        tok.get_vocab.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError, match="fail"):
            TokenIndex.build(tok)

    def test_decode_exception_returns_empty(self):
        """If tokenizer.decode() raises, the token maps to empty bytes."""
        tok = MagicMock()
        tok.vocab_size = 3
        tok.eos_token_id = 0
        # Remove get_vocab to force slow path
        def raise_fn(ids, **kw):
            raise ValueError("bad token")
        tok.decode.side_effect = raise_fn
        idx = TokenIndex.build(tok)
        assert idx.get_bytes(0) == b''
        assert idx.get_str(0) == ''


# ===========================================================================
# JSONSchemaFSM
# ===========================================================================

class TestJSONSchemaFSM:
    """Tests for the byte-level JSON grammar FSM."""

    # ─── Helpers ──────────────────────────────────────────────────────────

    def fsm(self):
        return JSONSchemaFSM()

    def run_bytes(self, fsm, text: str):
        """Feed bytes from a text string through the FSM."""
        for ch in text:
            fsm.transition(ord(ch))

    def assert_bytes_allowed(self, fsm, expected: set[int], msg=""):
        """Assert that get_allowed_bytes matches expected exactly."""
        allowed = fsm.get_allowed_bytes()
        assert allowed == expected, (
            f"{msg}: expected {sorted(expected)}, got {sorted(allowed)}"
        )

    def assert_chars_allowed(self, fsm, expected: set[str], msg=""):
        """Assert that get_allowed_bytes matches expected characters."""
        allowed = fsm.get_allowed_bytes()
        expected_bytes = {ord(c) for c in expected}
        assert allowed == expected_bytes, (
            f"{msg}: expected {sorted(expected)}, got {sorted(chr(b) for b in allowed)}"
        )

    # ─── Initial State ────────────────────────────────────────────────────

    def test_initial_state_start(self):
        """Initial state: allows '{', '[', and whitespace."""
        f = self.fsm()
        assert f._state == f.STATE_START
        ws = {0x20, 0x09, 0x0A, 0x0D}
        assert f.get_allowed_bytes() == {0x7B, 0x5B} | ws

    def test_initial_state_is_not_accepting(self):
        """FSM does not start in an accepting state."""
        assert not self.fsm().is_accepting()

    # ─── Object: Open/Close ───────────────────────────────────────────────

    def test_open_brace_transitions_to_expect_key(self):
        """'{' sets state to EXPECT_KEY."""
        f = self.fsm()
        f.transition(0x7B)  # {
        assert f._state == f.STATE_EXPECT_KEY
        # EXPECT_KEY: allows '"' (key start), '}' (empty object), whitespace
        allowed = f.get_allowed_bytes()
        assert 0x22 in allowed  # "
        assert 0x7D in allowed  # }

    def test_empty_object(self):
        """'{}' produces complete JSON."""
        f = self.fsm()
        self.run_bytes(f, "{}")
        assert f._state == f.STATE_AFTER_VALUE
        assert f.is_accepting()

    def test_close_brace_empty_stack_becomes_done(self):
        """'}' with empty stack transitions to DONE."""
        f = self.fsm()
        self.run_bytes(f, "{}")
        # After '{}', state is AFTER_VALUE due to stack pop
        # Now manually set stack empty and test
        f._stack = []
        f._state = f.STATE_EXPECT_KEY
        f.transition(0x7D)  # }
        assert f._state == f.STATE_DONE
        self.assert_chars_allowed(f, {' ', '\t', '\n', '\r'})

    # ─── Object: Simple Key-Value ─────────────────────────────────────────

    def test_simple_object(self):
        """FSM accepts a complete simple object."""
        f = self.fsm()
        f.transition(0x7B)  # {
        f.transition(0x22)  # " → IN_KEY
        for ch in "key":
            f.transition(ord(ch))
        f.transition(0x22)  # " → AFTER_KEY
        f.transition(0x3A)  # : → AFTER_COLON
        f.transition(0x22)  # " → IN_STRING
        for ch in "val":
            f.transition(ord(ch))
        f.transition(0x22)  # " → AFTER_VALUE
        f.transition(0x7D)  # } → pop stack → AFTER_VALUE
        assert f.is_accepting()
        assert f._state == f.STATE_AFTER_VALUE

    def test_after_colon_allows_value_chars(self):
        """AFTER_COLON allows string, number, object, array, literals."""
        f = self.fsm()
        self.run_bytes(f, '{"k":')
        # Now in AFTER_COLON: should allow value-starting bytes
        allowed = f.get_allowed_bytes()
        # Must allow:
        assert 0x22 in allowed  # "
        assert 0x7B in allowed  # {
        assert 0x5B in allowed  # [
        assert 0x2D in allowed  # -
        # Digits 0-9
        for d in range(0x30, 0x3A):
            assert d in allowed, f"digit {chr(d)} should be allowed"
        # Literal starters
        assert 0x74 in allowed  # t
        assert 0x66 in allowed  # f
        assert 0x6E in allowed  # n
        # Whitespace
        for ws in [0x20, 0x09, 0x0A, 0x0D]:
            assert ws in allowed

    # ─── Object: Comma ────────────────────────────────────────────────────

    def test_comma_after_value_in_object(self):
        "',' after a value in object context goes to EXPECT_KEY."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "v"')  # state = AFTER_VALUE
        f.transition(0x2C)  # ,
        assert f._state == f.STATE_EXPECT_KEY
        allowed = f.get_allowed_bytes()
        assert 0x22 in allowed  # "
        assert 0x7D in allowed  # }

    def test_comma_with_number_value(self):
        "',' after a number transitions correctly."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1')
        # state = IN_NUMBER
        f.transition(0x2C)  # ,
        assert f._state == f.STATE_EXPECT_KEY

    def test_comma_in_array(self):
        "',' after array value goes to IN_ARRAY."""
        f = self.fsm()
        self.run_bytes(f, '[1')
        f.transition(0x2C)  # ,
        assert f._state == f.STATE_IN_ARRAY

    def test_after_value_allows_comma_brace_bracket(self):
        """AFTER_VALUE allows comma, close brace, close bracket, whitespace."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1')
        # state is IN_NUMBER, need to advance to AFTER_VALUE
        # Hard to reach AFTER_VALUE from IN_NUMBER directly without the next char
        # Let's use AFTER_ARRAY_VALUE instead: after ']' in nested
        f2 = self.fsm()
        self.run_bytes(f2, '{"a": 1}')
        # after '}', state is AFTER_VALUE from stack pop
        assert f2._state == f.STATE_AFTER_VALUE
        # AFTER_VALUE: comma, }, ], whitespace
        allowed = f2.get_allowed_bytes()
        assert 0x2C in allowed  # ,
        assert 0x7D in allowed  # }
        assert 0x5D in allowed  # ]
        for ws in [0x20, 0x09, 0x0A, 0x0D]:
            assert ws in allowed

    # ─── Nested Objects ───────────────────────────────────────────────────

    def test_nested_object(self):
        """Nested objects with stack tracking."""
        f = self.fsm()
        self.run_bytes(f, '{"a": {')
        assert f._state == f.STATE_EXPECT_KEY
        # Stack should have 2 entries
        assert len(f._stack) == 2
        self.run_bytes(f, '"b": 1')
        # Close inner }
        f.transition(0x7D)
        assert f._state == f.STATE_AFTER_VALUE
        # Close outer }
        f.transition(0x7D)
        assert f._state == f.STATE_AFTER_VALUE
        assert f.is_accepting()

    def test_deeply_nested_objects(self):
        """FSM handles deeply nested objects."""
        f = self.fsm()
        self.run_bytes(f, '{"a": {"b": {"c": {"d": "e"')
        assert len(f._stack) == 4
        # Close all 4 levels
        for _ in range(4):
            f.transition(0x7D)
        assert f.is_accepting()

    # ─── Arrays ───────────────────────────────────────────────────────────

    def test_array_open(self):
        "'[' transitions to IN_ARRAY."""
        f = self.fsm()
        f.transition(0x5B)  # [
        assert f._state == f.STATE_IN_ARRAY

    def test_simple_array(self):
        """FSM accepts a complete array."""
        f = self.fsm()
        self.run_bytes(f, '[1, 2, 3]')
        assert f.is_accepting()

    def test_array_of_strings(self):
        """Array of string values."""
        f = self.fsm()
        self.run_bytes(f, '["a", "b"]')
        assert f.is_accepting()

    def test_empty_array(self):
        """Empty array '[]' is valid."""
        f = self.fsm()
        self.run_bytes(f, '[]')
        assert f.is_accepting()

    def test_mixed_array(self):
        """Array with different value types."""
        f = self.fsm()
        self.run_bytes(f, '[1, "two", true, null, false]')
        assert f.is_accepting()

    def test_nested_arrays(self):
        """Nested arrays."""
        f = self.fsm()
        self.run_bytes(f, '[[1, 2], [3, [4, 5]]]')
        assert f.is_accepting()

    def test_array_comma_after_value(self):
        """Comma in array context."""
        f = self.fsm()
        self.run_bytes(f, '[1')
        f.transition(0x2C)  # ,
        assert f._state == f.STATE_IN_ARRAY

    # ─── String Handling ──────────────────────────────────────────────────

    def test_string_allowed_chars(self):
        """Inside a string: all printable ASCII (including " and \)."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "')
        allowed = f.get_allowed_bytes()
        # All printable ASCII (0x20-0x7E) allowed
        for b in range(0x20, 0x7F):
            assert b in allowed, f"byte 0x{b:02X} ({chr(b)}) should be allowed in string"
        assert 0x22 in allowed  # " (valid transition to close string)
        assert 0x5C in allowed  # \ (valid escape start)

    def test_string_escape_next(self):
        """After backslash, allowed bytes are the JSON escape set + 'u'."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "')
        f.transition(0x5C)  # \ → escape_next = True
        allowed = f.get_allowed_bytes()
        # " \ / b f n r t u
        expected = {0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74, 0x75}
        assert allowed == expected

    def test_string_escape_sequence(self):
        """Valid escape sequence in string."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "hello\\nworld\\t\\"')
        # After escape sequences, still in string
        assert f._in_string
        f.transition(0x22)  # closing "
        assert not f._in_string
        assert f._state == f.STATE_AFTER_VALUE

    def test_string_backslash_dash_u_unicode(self):
        """Backslash-u (\\uXXXX) escape in string."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "\\u0041')
        # After \\u, we consumed 'u', which just clears escape_next
        assert f._in_string
        # Next chars are the hex digits (0-9, A-F, a-f) as normal string chars
        # The hex digits after \u are just part of the string content

    # ─── Number Handling ──────────────────────────────────────────────────

    def test_number_integer(self):
        """Integer number value."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 42')
        assert f._state == f.STATE_IN_NUMBER
        assert f._number_stage == 1

    def test_number_negative(self):
        """Negative number."""
        f = self.fsm()
        self.run_bytes(f, '{"a": -')
        assert f._state == f.STATE_IN_NUMBER
        assert f._number_stage == 0

    def test_number_float(self):
        """Floating-point number."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 3.14')
        assert f._state == f.STATE_IN_NUMBER
        assert f._number_stage == 2

    def test_number_exponential(self):
        """Scientific notation number."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1e10')
        assert f._state == f.STATE_IN_NUMBER
        assert f._number_stage == 3

    def test_number_negative_exponent(self):
        """Number with negative exponent."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1e-5')
        assert f._state == f.STATE_IN_NUMBER
        assert f._number_stage == 3

    def test_number_positive_exponent(self):
        """Number with positive exponent sign."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1E+2')
        assert f._state == f.STATE_IN_NUMBER

    def test_number_allowed_bytes(self):
        """get_allowed_bytes during number parsing."""
        f = self.fsm()
        self.run_bytes(f, '{"a": ')
        # AFTER_COLON: start a number
        f.transition(0x33)  # '3' → IN_NUMBER, stage=1
        allowed = f.get_allowed_bytes()
        # Stage 1: digits, '.', 'e', 'E'
        for d in range(0x30, 0x3A):
            assert d in allowed, f"digit 0x{d:X} should be allowed in number"
        assert 0x2E in allowed  # .
        assert 0x65 in allowed  # e
        assert 0x45 in allowed  # E

    def test_zero_number(self):
        """Single zero digit in number."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 0')
        assert f._state == f.STATE_IN_NUMBER

    # ─── Literals (true, false, null) ─────────────────────────────────────

    def test_literal_true(self):
        """'true' literal."""
        f = self.fsm()
        self.run_bytes(f, '{"a": true')
        assert not f._in_literal
        assert f._state == f.STATE_AFTER_VALUE

    def test_literal_false(self):
        """'false' literal."""
        f = self.fsm()
        self.run_bytes(f, '{"a": false')
        assert f._state == f.STATE_AFTER_VALUE

    def test_literal_null(self):
        """'null' literal."""
        f = self.fsm()
        self.run_bytes(f, '{"a": null')
        assert f._state == f.STATE_AFTER_VALUE

    def test_literal_allowed_bytes_step_by_step(self):
        """Each step of literal tracking allows only the next correct byte."""
        f = self.fsm()
        self.run_bytes(f, '{"a": t')
        assert f._state == f.STATE_IN_LITERAL
        assert f._in_literal == "t"

        # Next should be 'r' only
        allowed = f.get_allowed_bytes()
        assert allowed == {0x72}  # 'r'

        f.transition(0x72)  # r
        allowed = f.get_allowed_bytes()
        assert allowed == {0x75}  # 'u'

        f.transition(0x75)  # u
        allowed = f.get_allowed_bytes()
        assert allowed == {0x65}  # 'e'

        f.transition(0x65)  # e
        assert f._state == f.STATE_AFTER_VALUE
        assert f._in_literal == ""

    def test_literal_false_step_by_step(self):
        """Each byte of 'false' is tracked correctly."""
        f = self.fsm()
        self.run_bytes(f, '{"a": f')
        assert f._in_literal == "f"
        for expected in "alse":
            allowed = f.get_allowed_bytes()
            assert allowed == {ord(expected)}, f"expected '{expected}', got {[chr(b) for b in allowed]}"
            f.transition(ord(expected))
        assert f._state == f.STATE_AFTER_VALUE

    def test_literal_null_step_by_step(self):
        """Each byte of 'null' is tracked correctly."""
        f = self.fsm()
        self.run_bytes(f, '{"a": n')
        expected_chars = "ull"
        for expected in expected_chars:
            allowed = f.get_allowed_bytes()
            assert allowed == {ord(expected)}, f"expected '{expected}', got {[chr(b) for b in allowed]}"
            f.transition(ord(expected))
        assert f._state == f.STATE_AFTER_VALUE

    # ─── Accepting / Complete ─────────────────────────────────────────────

    def test_is_accepting_complete_object(self):
        """is_accepting returns True for complete top-level object."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1}')
        assert f.is_accepting()

    def test_is_accepting_complete_array(self):
        """is_accepting returns True for complete top-level array."""
        f = self.fsm()
        self.run_bytes(f, '[1, 2, 3]')
        assert f.is_accepting()

    def test_is_accepting_incomplete(self):
        """is_accepting returns False for incomplete JSON."""
        f = self.fsm()
        self.run_bytes(f, '{"a')
        assert not f.is_accepting()

    def test_is_accepting_empty(self):
        """is_accepting returns False on empty FSM."""
        assert not self.fsm().is_accepting()

    def test_is_accepting_string_no_end(self):
        """Unterminated string is not accepting."""
        f = self.fsm()
        self.run_bytes(f, '{"a": "hello')
        assert not f.is_accepting()

    def test_is_accepting_literal_no_end(self):
        """Unterminated literal is not accepting."""
        f = self.fsm()
        self.run_bytes(f, '{"a": tr')
        assert not f.is_accepting()

    # ─── Reset ────────────────────────────────────────────────────────────

    def test_reset(self):
        """reset() returns FSM to initial state."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1}')
        assert f.is_accepting()
        f.reset()
        assert f._state == f.STATE_START
        assert f._stack == []
        assert not f._in_string
        assert not f._escape_next
        assert f._in_literal == ""
        assert not f._in_number
        assert f._number_stage == 0
        assert not f.is_accepting()

    # ─── Whitespace ───────────────────────────────────────────────────────

    def test_whitespace_preserves_state(self):
        """Whitespace bytes don't change the FSM state."""
        f = self.fsm()
        initial = f._state
        for ws in [0x20, 0x09, 0x0A, 0x0D]:
            f.transition(ws)
            assert f._state == initial, f"whitespace 0x{ws:02X} changed state"

    def test_whitespace_allowed_in_many_states(self):
        """Whitespace is allowed in all active states."""
        states_to_test = [
            '{"a"',    # AFTER_KEY
            '{"a":',   # AFTER_COLON
            '{"a": "v"',  # AFTER_VALUE  
            '[',       # IN_ARRAY
        ]
        for prefix in states_to_test:
            f = self.fsm()
            self.run_bytes(f, prefix)
            allowed = f.get_allowed_bytes()
            for ws in [0x20, 0x09, 0x0A, 0x0D]:
                assert ws in allowed, f"whitespace not allowed after '{prefix}'"

    # ─── Edge Cases ───────────────────────────────────────────────────────

    def test_done_state_only_whitespace(self):
        """DONE state only allows whitespace bytes."""
        f = self.fsm()
        f._state = f.STATE_DONE
        expected = {0x20, 0x09, 0x0A, 0x0D}
        assert f.get_allowed_bytes() == expected

    def test_allowed_bytes_when_escape_next(self):
        """When escape_next is set, only JSON escape chars + 'u' are allowed."""
        f = self.fsm()
        self.run_bytes(f, '{"k": "test\\')
        assert f._escape_next
        allowed = f.get_allowed_bytes()
        expected = {0x22, 0x5C, 0x2F, 0x62, 0x66, 0x6E, 0x72, 0x74, 0x75}
        assert allowed == expected

    def test_allowed_bytes_after_value(self):
        """AFTER_VALUE allows comma, close brace/bracket, whitespace."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1}')
        # After '}', stack pops → AFTER_VALUE
        allowed = f.get_allowed_bytes()
        assert 0x2C in allowed  # ,
        assert 0x7D in allowed  # }
        assert 0x5D in allowed  # ]
        for ws in [0x20, 0x09, 0x0A, 0x0D]:
            assert ws in allowed

    def test_stack_used_for_nesting(self):
        """Stack tracks nesting correctly for mixed objects/arrays."""
        f = self.fsm()
        self.run_bytes(f, '{"a": [1, {"b": 2')
        # Object→Array→Object nesting
        assert len(f._stack) == 3
        # Close inner object, array, outer object
        f.transition(0x7D)  # } → pop to AFTER_VALUE
        assert len(f._stack) == 2
        f.transition(0x5D)  # ] → pop to AFTER_VALUE
        assert len(f._stack) == 1
        f.transition(0x7D)  # } → pop to AFTER_VALUE
        assert len(f._stack) == 0
        assert f.is_accepting()

    def test_many_values_in_object(self):
        """Multiple key-value pairs in an object."""
        f = self.fsm()
        self.run_bytes(f, '{"a": 1, "b": "two", "c": true, "d": null}')
        assert f.is_accepting()

    def test_get_allowed_bytes_return_set_not_none(self):
        """get_allowed_bytes always returns a set."""
        f = self.fsm()
        assert isinstance(f.get_allowed_bytes(), set)


# ===========================================================================
# RegexFSM
# ===========================================================================

class TestRegexFSM:
    """Tests for regex-based constrained FSM."""

    def test_simple_pattern_accepting(self):
        """RegexFSM accepts when full pattern matched."""
        fsm = RegexFSM(r"hello")
        for ch in "hello":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_simple_pattern_not_accepting(self):
        """RegexFSM not accepting before full match."""
        fsm = RegexFSM(r"hello")
        for ch in "hel":
            fsm.transition(ord(ch))
        assert not fsm.is_accepting()

    def test_pattern_with_alternation(self):
        """Alternation pattern matching."""
        fsm = RegexFSM(r"(yes|no)")
        for ch in "yes":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_pattern_with_repetition(self):
        """Repetition pattern."""
        fsm = RegexFSM(r"[a-z]+")
        for ch in "abc":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_get_allowed_bytes(self):
        """get_allowed_bytes returns printable ASCII bytes that keep prefix valid."""
        # Use a pattern with optional/repetition for better prefix matching
        fsm = RegexFSM(r"[a-z]+")
        allowed = fsm.get_allowed_bytes()
        assert isinstance(allowed, set)
        # Multiple chars should be allowed
        assert ord('a') in allowed
        assert ord('z') in allowed

    def test_get_allowed_bytes_empty(self):
        """get_allowed_bytes after full match allows EOS."""
        fsm = RegexFSM(r"ab?")
        fsm.transition(ord('a'))
        fsm.transition(ord('b'))
        assert fsm.is_accepting()
        # Even though accepting, get_allowed_bytes may still return chars
        # that could extend a partial match

    def test_reset(self):
        """reset() clears generated text."""
        fsm = RegexFSM(r"test")
        for ch in "test":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()
        fsm.reset()
        assert fsm._generated == ""
        assert not fsm.is_accepting()

    def test_pattern_with_group(self):
        """Pattern with groups."""
        fsm = RegexFSM(r"a(b)c")
        for ch in "abc":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_pattern_with_quantifier(self):
        """Pattern with '?' quantifier."""
        fsm = RegexFSM(r"colou?r")
        for ch in "color":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_pattern_number_match(self):
        """Pattern matching a number."""
        fsm = RegexFSM(r"\d+")
        for ch in "123":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_whitespace_in_pattern(self):
        """Pattern with whitespace."""
        fsm = RegexFSM(r"\s+")
        fsm.transition(ord(' '))
        assert fsm.is_accepting()

    def test_allowed_bytes_forwards_compatible(self):
        """get_allowed_bytes returns a set (may be empty)."""
        fsm = RegexFSM(r"[0-9]+")
        allowed = fsm.get_allowed_bytes()
        assert isinstance(allowed, set)

    def test_transition_returns_zero(self):
        """transition always returns 0 (single state)."""
        fsm = RegexFSM(r"a")
        result = fsm.transition(ord('a'))
        assert result == 0

    def test_is_accepting_false_for_empty_generated(self):
        """Empty generated string does not match unless pattern is optional."""
        fsm = RegexFSM(r"hello")
        assert not fsm.is_accepting()

    def test_prefix_matching_allows_partial(self):
        """get_allowed_bytes allows bytes that continue a prefix match."""
        fsm = RegexFSM(r"ab?")
        # After 'a', 'b' should be allowed (continues toward "ab")
        fsm.transition(ord('a'))
        allowed = fsm.get_allowed_bytes()
        assert ord('b') in allowed


# ===========================================================================
# ConstrainedConstraint
# ===========================================================================
#
# Note: These tests use JSONSchemaFSM since it's the most featureful FSM.

def _make_tokenizer():
    """Create a minimal mock tokenizer for ConstrainedConstraint tests."""
    tok = MagicMock()
    tok.vocab_size = 256
    tok.eos_token_id = 1
    # Use get_vocab fast path
    tok.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
    tok.decode.return_value = ""
    return tok


class TestConstrainedConstraint:
    """Tests for ConstrainedConstraint (FSM + TokenIndex integration)."""

    @pytest.fixture
    def token_index(self):
        tok = _make_tokenizer()
        return TokenIndex.build(tok)

    @pytest.fixture
    def json_fsm(self):
        return JSONSchemaFSM()

    def test_get_logits_mask_shape(self, token_index, json_fsm):
        """get_logits_mask returns a boolean tensor of correct shape."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)
        assert mask.dtype == torch.bool

    def test_get_logits_mask_some_allowed(self, token_index, json_fsm):
        """At least some tokens are allowed in initial state."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        mask = constraint.get_logits_mask(256)
        assert mask.sum() > 0

    def test_get_logits_mask_evolves_after_brace(self, token_index, json_fsm):
        """Mask changes after '{' is emitted."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        mask_before = constraint.get_logits_mask(256)
        b_before = mask_before.sum()
        constraint.update("{")
        mask_after = constraint.get_logits_mask(256)
        b_after = mask_after.sum()
        # Mask should change after transitioning state
        # (may be more or fewer tokens allowed)
        assert b_before != b_after or b_after > 0

    def test_eos_not_allowed_in_non_accepting_state(self, token_index, json_fsm):
        """EOS token should not be allowed when FSM is not accepting."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        mask = constraint.get_logits_mask(256)
        eos_id = token_index.eos_token_id
        assert eos_id is not None
        assert eos_id < 256
        assert not mask[eos_id].item(), "EOS should not be allowed in non-accepting state"

    def test_eos_allowed_in_accepting_state(self, token_index):
        """EOS token should be allowed when FSM is accepting."""
        fsm = JSONSchemaFSM()
        # Drive to accepting state
        for ch in '{"a": 1}':
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

        constraint = ConstrainedConstraint(fsm, token_index)
        mask = constraint.get_logits_mask(256)
        eos_id = token_index.eos_token_id
        assert mask[eos_id].item(), "EOS should be allowed in accepting state"

    def test_update_propagates_to_fsm(self, token_index, json_fsm):
        """update() advances the underlying FSM."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        assert not constraint.is_complete()
        constraint.update('{"a": 1}')
        assert constraint.is_complete()

    def test_is_complete_delegates_to_fsm(self, token_index, json_fsm):
        """is_complete returns the FSM's accepting state."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        assert constraint.is_complete() == json_fsm.is_accepting()

    def test_generated_text(self, token_index, json_fsm):
        """generated_text accumulates updates."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        assert constraint.generated_text == ""
        constraint.update("{")
        assert constraint.generated_text == "{"
        constraint.update('"a": 1}')
        assert constraint.generated_text == '{"a": 1}'

    def test_token_allowed_valid_token(self, token_index, json_fsm):
        """_token_allowed returns True for tokens that keep FSM valid."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        # After '{', a '"' token should be valid
        json_fsm.transition(0x7B)  # {
        result = constraint._token_allowed(b'"key": "val"')
        assert result, 'Token with first byte " should be valid after {'

    def test_token_allowed_invalid_token(self, token_index, json_fsm):
        """_token_allowed returns False for tokens that break FSM."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        # After ':', a value must follow — '}' token alone is invalid here
        json_fsm.transition(0x7B)  # {
        json_fsm.transition(0x22)  # "
        for ch in "k":
            json_fsm.transition(ord(ch))
        json_fsm.transition(0x22)  # "
        json_fsm.transition(0x3A)  # :
        # Now state = AFTER_COLON, trying '}' is invalid at this position
        result = constraint._token_allowed(b'}')
        assert not result

    def test_vocab_size_mismatch(self, token_index, json_fsm):
        """get_logits_mask handles different vocab_size gracefully."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        larger = constraint.get_logits_mask(500)
        assert larger.shape == (500,)
        smaller = constraint.get_logits_mask(50)
        assert smaller.shape == (50,)

    def test_constraint_with_schema(self, token_index, json_fsm):
        """Constraint stores schema when provided."""
        schema = {"type": "object"}
        constraint = ConstrainedConstraint(json_fsm, token_index, schema=schema)
        assert constraint._schema == schema

    def test_constraint_reset_via_update(self, token_index):
        """After update to accepting state, a new update continues."""
        fsm = JSONSchemaFSM()
        constraint = ConstrainedConstraint(fsm, token_index)
        constraint.update('{"a": 1}')
        assert constraint.is_complete()
        # After complete JSON, state is AFTER_VALUE
        # = accepting: can accept EOS or more content via comma/bracket
        # The FSM should allow more content (',' → new key/value in object)
        assert constraint.is_complete()

    def test_get_logits_mask_timing(self, token_index, json_fsm):
        """Constrained get_logits_mask should not be excessively slow."""
        constraint = ConstrainedConstraint(json_fsm, token_index)
        # Baseline: torch.zeros allocation
        baseline_start = time.perf_counter()
        for _ in range(50):
            _ = torch.zeros(256, dtype=torch.bool)
        baseline_elapsed = time.perf_counter() - baseline_start

        # Constrained: get_logits_mask with FSM
        constrained_start = time.perf_counter()
        for _ in range(50):
            _ = constraint.get_logits_mask(256)
        constrained_elapsed = time.perf_counter() - constrained_start

        # Constrained should not be more than 100x slower than baseline
        # (accounts for FSM traversal + token index lookups)
        ratio = constrained_elapsed / max(baseline_elapsed, 1e-9)
        assert ratio < 100, f"get_logits_mask {ratio:.1f}x slower than baseline"


# ===========================================================================
# Integration: SchemaConstrainedDecoder with FSM
# ===========================================================================

class TestSchemaConstrainedDecoderFSM:
    """Integration tests connecting SchemaConstrainedDecoder to the FSM layers."""

    def _make_tokenizer(self):
        tok = MagicMock()
        tok.vocab_size = 256
        tok.eos_token_id = 1
        tok.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
        return tok

    def test_json_object_uses_json_schema_fsm(self):
        """json_object mode creates constraint with JSONSchemaFSM."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder, JSONSchemaFSM
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        constraint = decoder.json_schema()
        assert isinstance(constraint._fsm, JSONSchemaFSM)

    def test_regex_uses_regex_fsm(self):
        """regex mode creates constraint with RegexFSM."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder, RegexFSM
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        constraint = decoder.regex(r"[a-z]+")
        assert isinstance(constraint._fsm, RegexFSM)

    def test_e2e_simple_json(self):
        """End-to-end: build constraint, step through JSON generation."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        constraint = decoder.json_schema({})

        # Step 1: initial mask (should allow '{' and '[')
        assert not constraint.is_complete()
        mask1 = constraint.get_logits_mask(256)
        assert mask1[0x7B].item()  # '{' should be allowed

        # Step 2: emit '{'
        constraint.update("{")
        mask2 = constraint.get_logits_mask(256)
        assert mask2[0x22].item()  # '"' should be allowed

        # Step 3: emit a key
        constraint.update('"a"')
        mask3 = constraint.get_logits_mask(256)
        assert mask3[0x3A].item()  # ':' should be allowed

        # Step 4: emit ':'
        constraint.update(":")
        mask4 = constraint.get_logits_mask(256)
        assert mask4[0x22].item()  # '"' for string value
        assert mask4[0x7B].item()  # '{' for nested object

        # Step 5: complete
        constraint.update(' "b"')
        constraint.update("}")
        assert constraint.is_complete()

    def test_eos_behavior(self):
        """EOS only allowed when constraint is accepting."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        constraint = decoder.json_schema({})

        # Before completion, EOS should be blocked
        mask = constraint.get_logits_mask(256)
        eos_id = tok.eos_token_id
        assert not mask[eos_id].item()

        # Complete JSON
        constraint.update('{"a": 1}')
        mask = constraint.get_logits_mask(256)
        assert mask[eos_id].item()

    def test_pydantic_model(self):
        """pydantic mode converts model to JSON schema."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        from unittest.mock import MagicMock
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)

        model = MagicMock()
        model.model_json_schema.return_value = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        constraint = decoder.pydantic(model)
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)

    def test_from_response_format_json_object(self):
        """from_response_format with json_object returns a constraint."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tok
        )
        assert constraint is not None
        assert not constraint.is_complete()

    def test_from_response_format_grammar(self):
        """from_response_format with grammar returns a constraint."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "grammar", "grammar": 'root ::= [a-z]+'},
            tokenizer=tok,
        )
        assert constraint is not None

    def test_token_index_cache(self):
        """TokenIndex is cached across instances."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        decoder1 = SchemaConstrainedDecoder(tok)
        decoder2 = SchemaConstrainedDecoder(tok)
        assert decoder1._token_index is decoder2._token_index

    def test_clear_cache(self):
        """clear_cache() resets the token index cache."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        SchemaConstrainedDecoder.clear_cache()
        assert SchemaConstrainedDecoder._token_index_cache == {}

    def test_repeated_schema_shares_token_index(self):
        """Multiple json_schema calls share the cached TokenIndex."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        SchemaConstrainedDecoder.clear_cache()
        tok = self._make_tokenizer()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        decoder1 = SchemaConstrainedDecoder(tok)
        c1 = decoder1.json_schema(schema)
        c2 = decoder1.json_schema(schema)
        assert c1._token_index is c2._token_index

    def test_repeated_schema_creates_separate_fsms(self):
        """Each json_schema call creates a fresh FSM (no schema-level FSM cache)."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        SchemaConstrainedDecoder.clear_cache()
        tok = self._make_tokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        c1 = decoder.json_schema({})
        c2 = decoder.json_schema({})
        assert c1._fsm is not c2._fsm

    def test_from_response_format_no_tokenizer_returns_none(self):
        """from_response_format returns None when no tokenizer provided for constraint types."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        assert SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}
        ) is None

    def test_from_response_format_grammar_empty(self):
        """from_response_format with empty grammar returns None."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = self._make_tokenizer()
        assert SchemaConstrainedDecoder.from_response_format(
            {"type": "grammar", "grammar": ""}, tokenizer=tok
        ) is None
