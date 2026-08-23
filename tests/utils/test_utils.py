"""Tests for distllm/utils modules — GBNFGrammar and scheduling helpers.

These modules are pure utility functions with no external dependencies,
so no mocks are needed.
"""

from __future__ import annotations

import pytest

from tests.utils.conftest import (
    GBNFGrammar,
    _extra_rules,
    _primitive_example,
    _primitive_rule,
    _type_to_rule,
    generate_gbnf_for_json_schema,
    group_by_length,
    json_schema_to_gbnf,
)

# ============================================================================
# GBNFGrammar: __str__
# ============================================================================


class TestGBNFGrammarStr:
    def test_empty_rules(self):
        g = GBNFGrammar()
        assert str(g) == ""

    def test_single_rule(self):
        g = GBNFGrammar(rules=['root ::= "hello"'])
        assert str(g) == 'root ::= "hello"'

    def test_multiple_rules(self):
        g = GBNFGrammar(rules=['root ::= "a"', 'foo ::= "b"'])
        assert str(g) == 'root ::= "a"\nfoo ::= "b"'

    def test_preserves_rule_order(self):
        rules = [f"rule{i} ::= {i}" for i in range(10)]
        g = GBNFGrammar(rules=rules)
        lines = str(g).split("\n")
        assert len(lines) == 10
        for i, line in enumerate(lines):
            assert line == f"rule{i} ::= {i}"


# ============================================================================
# json_schema_to_gbnf
# ============================================================================


class TestJsonSchemaToGbnf:
    def test_simple_string_schema(self):
        schema = json_schema_to_gbnf('{"type": "string"}')
        assert isinstance(schema, GBNFGrammar)
        assert str(schema) == 'root ::= "str"'

    def test_simple_integer_schema(self):
        schema = json_schema_to_gbnf('{"type": "integer"}')
        assert str(schema) == 'root ::= "0"'

    def test_simple_number_schema(self):
        schema = json_schema_to_gbnf('{"type": "number"}')
        assert str(schema) == 'root ::= "0.0"'

    def test_simple_boolean_schema(self):
        schema = json_schema_to_gbnf('{"type": "boolean"}')
        assert str(schema) == 'root ::= "true"'

    def test_null_default_type(self):
        """Missing type field defaults to 'object' in _convert_schema."""
        schema = json_schema_to_gbnf('{"properties": {}}')
        assert 'root ::= "{" ws' in str(schema)
        assert '"}" ws' in str(schema)

    def test_unknown_type(self):
        schema = json_schema_to_gbnf('{"type": "array"}')
        assert str(schema) == 'root ::= "null"'

    def test_object_no_properties(self):
        schema = json_schema_to_gbnf('{"type": "object", "properties": {}}')
        text = str(schema)
        assert 'root ::= "{" ws' in text
        assert '"}" ws' in text

    def test_object_single_string_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"name": {"type": "string"}}}'
        )
        text = str(schema)
        assert 'root ::= "{" ws' in text
        assert '"name"' in text
        assert '([^"' in text  # string rule component

    def test_object_single_integer_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"count": {"type": "integer"}}}'
        )
        text = str(schema)
        assert '  "count" ws ":" ws ("-"? [0-9]+)' in text

    def test_object_multiple_properties(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}'
        )
        text = str(schema)
        # Both properties should appear
        assert '"a" ws ":" ws' in text
        assert '"b" ws ":" ws' in text
        # Separator between properties
        assert '"," ws' in text

    def test_object_boolean_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"flag": {"type": "boolean"}}}'
        )
        text = str(schema)
        assert '("true" | "false")' in text

    def test_object_number_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"price": {"type": "number"}}}'
        )
        text = str(schema)
        assert '"-"? ([0-9]+ "."[0-9]+ | [0-9]+)' in text

    def test_object_single_property_no_trailing_comma(self):
        """Last property in an object should not have a trailing comma rule."""
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"x": {"type": "integer"}}}'
        )
        text = str(schema)
        # With one property there's no iteration so no comma rule injected
        assert '"," ws' not in text

    def test_object_two_properties_trailing_comma_on_first(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}'
        )
        text = str(schema)
        # First property should have a comma separator after it
        assert '"," ws' in text

    def test_object_with_enum_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}}}'
        )
        text = str(schema)
        assert "red" in text
        assert "green" in text
        assert "blue" in text

    def test_object_with_oneof_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"value": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}'
        )
        text = str(schema)
        assert "string" not in text  # types are expanded to rules, not named
        assert "[0-9]" in text  # integer rule present

    def test_object_with_ref_property(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"user": {"$ref": "#/$defs/User"}}}'
        )
        text = str(schema)
        assert "User ::= value" in text

    def test_object_with_defs(self):
        schema = json_schema_to_gbnf(
            '{"type": "object", "properties": {"name": {"type": "string"}}, "$defs": {"User": {"type": "object", "properties": {"id": {"type": "integer"}}}}}'
        )
        text = str(schema)
        assert "User ::= value" in text
        assert "$defs" not in text  # $defs key itself shouldn't appear in grammar

    def test_empty_properties_object(self):
        schema = json_schema_to_gbnf('{"type": "object", "properties": {}}')
        text = str(schema)
        assert 'root ::= "{" ws' in text
        assert '"}" ws' in text

    def test_dict_input_vs_string_input(self):
        """Both str and dict inputs should produce the same result."""
        from_str = json_schema_to_gbnf('{"type": "string"}')
        from_dict = json_schema_to_gbnf({"type": "string"})
        assert str(from_str) == str(from_dict)


# ============================================================================
# generate_gbnf_for_json_schema
# ============================================================================


class TestGenerateGbnfForJsonSchema:
    def test_returns_string(self):
        result = generate_gbnf_for_json_schema({"type": "string"})
        assert isinstance(result, str)
        assert result == 'root ::= "str"'

    def test_object_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = generate_gbnf_for_json_schema(schema)
        assert 'root ::= "{" ws' in result
        assert '"x" ws ":" ws ("-"? [0-9]+)' in result
        assert '"}" ws' in result

    def test_strict_flag_accepted(self):
        """strict parameter is accepted though not currently used."""
        result = generate_gbnf_for_json_schema({"type": "boolean"}, strict=True)
        assert result == 'root ::= "true"'

    def test_strict_flag_false_default(self):
        result = generate_gbnf_for_json_schema({"type": "boolean"}, strict=False)
        assert result == 'root ::= "true"'


# ============================================================================
# _type_to_rule  (internal helper)
# ============================================================================


class TestTypeToRule:
    def test_enum(self):
        rule = _type_to_rule({"type": "string", "enum": ["a", "b", "c"]})
        # Spaces inside parens are part of the generated format
        assert '( "a" | "b" | "c" )' == rule

    def test_enum_single_value(self):
        rule = _type_to_rule({"type": "string", "enum": ["only"]})
        assert '( "only" )' == rule

    def test_oneof(self):
        rule = _type_to_rule(
            {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        )
        # oneOf should produce alternatives delimited by |
        assert "(" in rule
        assert ")" in rule
        assert "|" in rule
        parts = rule.strip("() ").split(" | ")
        # The string rule itself contains | so splitting gives more than 2 parts
        assert len(parts) >= 2

    def test_ref(self):
        rule = _type_to_rule({"$ref": "#/$defs/Foo"})
        assert rule == "Foo ::= value"

    def test_default_type_string(self):
        """When type is missing, defaults to string."""
        rule = _type_to_rule({})
        assert "\\\\" in rule  # string rule has backslash-escaped quotes

    def test_primitive_string(self):
        rule = _type_to_rule({"type": "string"})
        assert "\\" in rule

    def test_primitive_integer(self):
        rule = _type_to_rule({"type": "integer"})
        assert "[0-9]" in rule

    def test_primitive_number(self):
        rule = _type_to_rule({"type": "number"})
        assert "[0-9]" in rule
        assert '"."' in rule

    def test_primitive_boolean(self):
        rule = _type_to_rule({"type": "boolean"})
        assert "true" in rule
        assert "false" in rule

    def test_null_type(self):
        rule = _type_to_rule({"type": "null"})
        assert rule == '"null"'

    def test_name_arg_ignored(self):
        """The name parameter is accepted but not used in the current impl."""
        with_name = _type_to_rule({"type": "integer"}, name="count")
        without_name = _type_to_rule({"type": "integer"})
        assert with_name == without_name


# ============================================================================
# _primitive_rule  (internal helper)
# ============================================================================


class TestPrimitiveRule:
    def test_string_rule(self):
        rule = _primitive_rule("string")
        # Should match double-quoted strings
        assert '\\\\"' in rule
        assert "(" in rule
        assert ")" in rule

    def test_integer_rule(self):
        rule = _primitive_rule("integer")
        assert rule == '("-"? [0-9]+)'

    def test_number_rule(self):
        rule = _primitive_rule("number")
        assert rule == '("-"? ([0-9]+ "."[0-9]+ | [0-9]+))'

    def test_boolean_rule(self):
        rule = _primitive_rule("boolean")
        assert rule == '("true" | "false")'

    def test_unknown_type_falls_back_to_null(self):
        rule = _primitive_rule("array")
        assert rule == '"null"'

    def test_empty_string_falls_back_to_null(self):
        rule = _primitive_rule("")
        assert rule == '"null"'


# ============================================================================
# _primitive_example  (internal helper)
# ============================================================================


class TestPrimitiveExample:
    def test_string(self):
        assert _primitive_example("string") == "str"

    def test_integer(self):
        assert _primitive_example("integer") == "0"

    def test_number(self):
        assert _primitive_example("number") == "0.0"

    def test_boolean(self):
        assert _primitive_example("boolean") == "true"

    def test_unknown_type(self):
        assert _primitive_example("array") == "null"

    def test_empty_string(self):
        assert _primitive_example("") == "null"


# ============================================================================
# _extra_rules  (internal helper)
# ============================================================================


class TestExtraRules:
    def test_no_defs(self):
        assert _extra_rules({}) == []

    def test_single_def(self):
        rules = _extra_rules({"$defs": {"Foo": {"type": "object"}}})
        assert rules == ["Foo ::= value"]

    def test_multiple_defs(self):
        rules = _extra_rules(
            {
                "$defs": {
                    "Foo": {"type": "object"},
                    "Bar": {"type": "string"},
                }
            }
        )
        assert len(rules) == 2
        assert "Foo ::= value" in rules
        assert "Bar ::= value" in rules

    def test_defs_with_nested_schema(self):
        """Def schema structure is not inspected; just emits name ::= value."""
        rules = _extra_rules(
            {"$defs": {"ComplexType": {"type": "object", "properties": {"x": {"type": "integer"}}}}}
        )
        assert rules == ["ComplexType ::= value"]

    def test_unknown_key_inside_defs(self):
        """_extra_rules only looks at the $defs key, others are ignored."""
        rules = _extra_rules({"$defs": {}, "other": "stuff"})
        assert rules == []


# ============================================================================
# group_by_length
# ============================================================================


class FakeSeq:
    """Minimal sequence-like object with a ``total_len`` attribute."""

    def __init__(self, total_len: int, label: str = "") -> None:
        self.total_len = total_len
        self._label = label

    def __repr__(self) -> str:
        return f"FakeSeq({self.total_len}, {self._label!r})"


class TestGroupByLengthEmpty:
    def test_no_sequences(self):
        result = group_by_length([])
        assert isinstance(result, dict)
        assert all(isinstance(k, int) for k in result)
        assert all(v == [] for v in result.values())

    def test_num_buckets_default_is_four(self):
        result = group_by_length([])
        assert len(result) == 4
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_custom_num_buckets(self):
        result = group_by_length([], num_buckets=1)
        assert len(result) == 1
        assert 0 in result

    def test_custom_num_buckets_zero(self):
        """Zero buckets returns an empty dict (no range to iterate)."""
        result = group_by_length([], num_buckets=0)
        assert result == {}


class TestGroupByLengthSingle:
    def test_single_sequence(self):
        seq = FakeSeq(100)
        result = group_by_length([seq])
        # Single item goes in bucket 0
        assert result[0] == [seq]
        assert sum(len(v) for v in result.values()) == 1

    def test_single_min_length(self):
        seq = FakeSeq(1)
        result = group_by_length([seq])
        assert result[0] == [seq]


class TestGroupByLengthUniform:
    def test_all_same_length(self):
        seqs = [FakeSeq(50) for _ in range(5)]
        result = group_by_length(seqs)
        # All in bucket 0 when min_len == max_len
        assert len(result[0]) == 5
        for i in range(1, 4):
            assert result[i] == []

    def test_same_length_different_objects(self):
        a = FakeSeq(100, "a")
        b = FakeSeq(100, "b")
        result = group_by_length([a, b])
        assert set(result[0]) == {a, b}


class TestGroupByLengthDistribution:
    def test_two_distinct_lengths(self):
        seqs = [FakeSeq(10), FakeSeq(1000)]
        result = group_by_length(seqs, num_buckets=4)
        items = [item for bucket in result.values() for item in bucket]
        assert len(items) == 2

    def test_length_progressive(self):
        """Sequences of increasing lengths should distribute across buckets."""
        lengths = [1, 10, 100, 1000, 10000]
        seqs = [FakeSeq(l) for l in lengths]
        result = group_by_length(seqs, num_buckets=4)
        all_items = [item for bucket in result.values() for item in bucket]
        assert len(all_items) == len(lengths)

    def test_log_scale_bucketing(self):
        """Very short and very long sequences should not all land in bucket 0."""
        lengths = [1, 1, 1, 5000, 10000, 20000]
        seqs = [FakeSeq(l) for l in lengths]
        result = group_by_length(seqs, num_buckets=4)
        # The short ones should be in lower buckets
        # The long ones should be in higher buckets
        bucket_assignments = {}
        for bucket_idx, bucket_seqs in result.items():
            for s in bucket_seqs:
                bucket_assignments[s.total_len] = bucket_idx
        # Long sequences should be in higher buckets than short ones
        assert bucket_assignments[20000] >= bucket_assignments[5000]
        assert bucket_assignments[5000] >= bucket_assignments[1]

    def test_small_range_distribution(self):
        """Sequences with small differences should still distribute."""
        seqs = [FakeSeq(l) for l in [1, 2, 3, 4, 5]]
        result = group_by_length(seqs, num_buckets=4)
        total = sum(len(v) for v in result.values())
        assert total == 5


class TestGroupByLengthEdgeCases:
    def test_zero_length_sequence(self):
        seqs = [FakeSeq(0)]
        result = group_by_length(seqs)
        # total_len of 0 should be handled (max with 1)
        assert len(result[0]) == 1

    def test_mixed_zero_and_positive(self):
        seqs = [FakeSeq(0), FakeSeq(100), FakeSeq(1000)]
        result = group_by_length(seqs, num_buckets=4)
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_negative_total_len(self):
        """Log of a negative number would be problematic, test defensive behavior."""
        seqs = [FakeSeq(-1)]
        result = group_by_length(seqs)
        # Should not crash; max with 1 is used
        assert len(result[0]) == 1

    def test_large_length_values(self):
        seqs = [FakeSeq(10**6), FakeSeq(10**9)]
        result = group_by_length(seqs, num_buckets=4)
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_bucket_indices_are_valid(self):
        """All bucket indices must be in 0..num_buckets-1."""
        lengths = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
        seqs = [FakeSeq(l) for l in lengths]
        result = group_by_length(seqs, num_buckets=4)
        for bucket_idx in result:
            assert 0 <= bucket_idx <= 3

    def test_preserves_identity(self):
        """Sequences should not be modified and should preserve identity."""
        seqs = [FakeSeq(10), FakeSeq(100)]
        orig_ids = {id(s) for s in seqs}
        result = group_by_length(seqs, num_buckets=4)
        result_ids = {id(s) for bucket in result.values() for s in bucket}
        assert result_ids == orig_ids
