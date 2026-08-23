"""Regression: JSONSchema FSM must allow multi-digit number continuation.

F-045: `_valid_next_chars()` had no `in_number` entry, so once a number started
the next token's first character was restricted to the default `{'"', '}'}` —
every multi-digit number was truncated to a single digit.
"""

from __future__ import annotations

from distllm.core.structured_output import JSONSchemaConstraint


class TestMultiDigitNumbers:
    def test_number_continuation_allowed(self):
        c = JSONSchemaConstraint()
        c.update('{"n":1')  # drive into a number value
        assert c._state == "in_number"
        valid = c._valid_next_chars()
        # A second digit must be allowed (continue the number).
        assert '2' in valid
        assert '0' in valid
        assert '9' in valid

    def test_full_multi_digit_number_is_valid_json(self):
        c = JSONSchemaConstraint()
        c.update('{"n":123')  # multi-digit number
        assert c._state == "in_number"
        valid = c._valid_next_chars()
        # The number may be terminated by , or }.
        assert ',' in valid
        assert '}' in valid

    def test_comma_after_number_transitions_to_after_comma(self):
        c = JSONSchemaConstraint()
        c.update('{"n":42')
        assert c._state == "in_number"
        c.update(',')
        assert c._state == "after_comma"

    def test_decimal_and_exponent_continuation(self):
        c = JSONSchemaConstraint()
        c.update('{"n":3.14')
        assert c._state == "in_number"
        c.update('e')
        assert c._state == "in_number"
        c.update('5')
        assert c._state == "in_number"
