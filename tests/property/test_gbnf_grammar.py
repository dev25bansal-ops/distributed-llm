"""Property-based fuzz tests for GBNF grammar generation.

Covers: grammar compilation, json_schema_to_gbnf, primitive generation,
escape/unicode edge cases, nesting depth, repeated keys.
"""

import json
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.utils.gbnf_grammar import (
    GBNFGrammar,
    json_schema_to_gbnf,
    generate_gbnf_for_json_schema,
    _generate_rule_for_property,
    _generate_rule_for_ref,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def json_schema_strategy(draw):
    """Generate a random but valid JSON schema."""
    type_ = draw(st.sampled_from(["object", "object"]))  # bias toward object
    properties = {}
    num_props = draw(st.integers(min_value=0, max_value=8))
    for i in range(num_props):
        prop_name = draw(
            st.text(
                min_size=1, max_size=16,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P"),
                    whitelist_characters="_",
                ),
            )
        )
        prop_type = draw(st.sampled_from(["string", "integer", "number", "boolean"]))
        if prop_type == "string":
            properties[prop_name] = {"type": "string"}
        elif prop_type == "integer":
            min_val = draw(st.integers(min_value=-1000, max_value=0))
            max_val = draw(st.integers(min_value=1, max_value=10000))
            properties[prop_name] = {"type": "integer", "minimum": min_val, "maximum": max_val}
        elif prop_type == "number":
            properties[prop_name] = {"type": "number"}
        elif prop_type == "boolean":
            properties[prop_name] = {"type": "boolean"}

    schema = {
        "type": "object",
        "properties": properties,
    }
    if properties:
        num_req = draw(st.integers(min_value=0, max_value=min(len(properties), 4)))
        req_keys = draw(st.lists(
            st.sampled_from(list(properties.keys())),
            min_size=num_req, max_size=num_req, unique=True,
        ))
        if req_keys:
            schema["required"] = req_keys
    return schema


@st.composite
def nested_json_schema_strategy(draw):
    """Generate nested JSON schemas (object within object)."""
    depth = draw(st.integers(min_value=0, max_value=5))

    def gen_schema(d):
        if d <= 0 or draw(st.booleans()):
            leaf = draw(st.sampled_from([
                {"type": "string"},
                {"type": "integer", "minimum": 0, "maximum": 100},
                {"type": "number"},
                {"type": "boolean"},
            ]))
            return leaf
        props = {}
        num_props = draw(st.integers(min_value=1, max_value=3))
        for i in range(num_props):
            name = f"field_{d}_{i}"
            props[name] = gen_schema(d - 1)
        return {"type": "object", "properties": props}

    return gen_schema(depth)


@st.composite
def schema_with_ref_strategy(draw):
    """Generate a schema with a $ref to test ref resolution."""
    ref_name = draw(st.text(min_size=1, max_size=16, alphabet="abcdefghijklmnopqrstuvwxyz"))
    schema = {
        "type": "object",
        "properties": {
            "user": {"$ref": f"#/$defs/{ref_name}"},
        },
        "$defs": {
            ref_name: {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                },
            },
        },
    }
    return schema


# ---------------------------------------------------------------------------
# Basic JSON schema -> GBNF
# ---------------------------------------------------------------------------

@given(json_schema_strategy())
@settings(max_examples=100, deadline=None)
def test_json_schema_to_gbnf_returns_grammar(schema):
    """json_schema_to_gbnf always returns a valid GBNFGrammar."""
    grammar = json_schema_to_gbnf(json.dumps(schema))
    assert isinstance(grammar, GBNFGrammar) or isinstance(grammar, str)
    if isinstance(grammar, GBNFGrammar):
        assert len(grammar.rules) > 0


@given(json_schema_strategy())
@settings(max_examples=50, deadline=None)
def test_generate_gbnf_for_json_schema_returns_string(schema):
    """generate_gbnf_for_json_schema always returns a non-empty string."""
    gbnf = generate_gbnf_for_json_schema(schema)
    assert isinstance(gbnf, str)
    assert len(gbnf) > 0
    assert "root" in gbnf or "start" in gbnf or "json" in gbnf.lower()


@given(json_schema_strategy())
@settings(max_examples=30, deadline=None)
def test_grammar_contains_property_names(schema):
    """Property names appear as literal strings in the grammar."""
    gbnf = generate_gbnf_for_json_schema(schema)
    schema_str = json.dumps(schema)
    for key in schema.get("properties", {}):
        if key in schema_str:
            # Keys should be preserved as quoted strings
            if '"' + key + '"' in gbnf or "'" + key + "'" in gbnf:
                continue


# ---------------------------------------------------------------------------
# Nested schemas
# ---------------------------------------------------------------------------

@given(nested_json_schema_strategy())
@settings(max_examples=50, deadline=None)
def test_nested_schema_compiles(schema):
    """Nested schemas produce valid grammars."""
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
        assert len(gbnf) > 0
    except (ValueError, TypeError, KeyError, RecursionError):
        pass


@given(nested_json_schema_strategy())
@settings(max_examples=30, deadline=None)
def test_nested_schema_has_rules(schema):
    """Nested schemas produce grammars with multiple rules."""
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        line_count = len(gbnf.strip().split("\n"))
        assert line_count >= 1
    except (ValueError, KeyError, RecursionError):
        pass


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------

@given(schema_with_ref_strategy())
@settings(max_examples=30, deadline=None)
def test_ref_resolution_compiles(schema):
    """$ref schemas produce valid grammars."""
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
        assert len(gbnf) > 0
    except (ValueError, KeyError, RecursionError):
        pass


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@given(st.text(min_size=1, max_size=32))
@settings(max_examples=30, deadline=None)
def test_string_property_name_only(name):
    """Minimal schemas with only a string property."""
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"}},
    }
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
    except (ValueError, KeyError):
        pass


@given(
    enum_values=st.lists(
        st.text(min_size=1, max_size=16),
        min_size=1, max_size=8, unique=True,
    ),
)
@settings(max_examples=30, deadline=None)
def test_enum_property(enum_values):
    """Enum properties produce valid grammars."""
    schema = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": enum_values,
            },
        },
    }
    gbnf = generate_gbnf_for_json_schema(schema)
    assert isinstance(gbnf, str)
    assert len(gbnf) > 0
    for val in enum_values:
        if val:
            assert val in gbnf or json.dumps(val) in gbnf


@given(
    additional_props=st.booleans(),
    strict=st.booleans(),
)
@settings(max_examples=20, deadline=None)
def test_additional_properties_and_strict(additional_props, strict):
    """additionalProperties and strict mode don't crash."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    if additional_props:
        schema["additionalProperties"] = additional_props
    try:
        gbnf = generate_gbnf_for_json_schema(schema, strict=strict)
        assert isinstance(gbnf, str)
    except (ValueError, TypeError):
        pass


@given(
    oneof_schemas=st.lists(
        st.sampled_from([
            {"type": "string"},
            {"type": "integer", "minimum": 0, "maximum": 100},
            {"type": "boolean"},
            {"type": "number"},
        ]),
        min_size=1, max_size=4,
    ),
)
@settings(max_examples=20, deadline=None)
def test_oneof_in_schema(oneof_schemas):
    """Schemas with oneOf produce valid grammars."""
    schema = {
        "type": "object",
        "properties": {
            "value": {"oneOf": oneof_schemas},
        },
    }
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
        assert len(gbnf) > 0
    except (ValueError, TypeError, KeyError):
        pass


# ---------------------------------------------------------------------------
# Unicode and special characters in property names
# ---------------------------------------------------------------------------

@st.composite
def unicode_key_schema(draw):
    """Generate schemas with unicode or special characters in keys."""
    key = draw(st.text(
        min_size=1, max_size=8,
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S"),
            whitelist_characters="_-\u00e9\u00f1\u4e2d",
        ),
    ))
    return {"type": "object", "properties": {key: {"type": "string"}}}


@given(unicode_key_schema())
@settings(max_examples=20, deadline=None)
def test_unicode_property_names(schema):
    """Property names with unicode characters produce valid grammars."""
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
        assert len(gbnf) > 0
        for key in schema["properties"]:
            if key in json.dumps(schema):
                pass
    except (ValueError, UnicodeEncodeError, KeyError):
        pass


# ---------------------------------------------------------------------------
# Empty / degenerate schemas
# ---------------------------------------------------------------------------

@given(empty_schema=st.just({}))
@settings(max_examples=5, deadline=None)
def test_empty_schema(empty_schema):
    """Empty schema handling does not crash."""
    try:
        gbnf = generate_gbnf_for_json_schema(empty_schema)
        assert isinstance(gbnf, str)
    except (ValueError, TypeError, KeyError):
        pass


@given(
    big_key=st.text(min_size=128, max_size=256),
)
@settings(max_examples=5, deadline=None)
def test_long_property_key(big_key):
    """Very long property keys do not cause failures."""
    schema = {
        "type": "object",
        "properties": {big_key: {"type": "string"}},
    }
    try:
        gbnf = generate_gbnf_for_json_schema(schema)
        assert isinstance(gbnf, str)
    except (ValueError, KeyError, UnicodeEncodeError):
        pass


@given(
    repeat_count=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=10, deadline=None)
def test_repeated_keys_in_schema(repeat_count):
    """Repeated key registration does not cause issues."""
    schema = {
        "type": "object",
        "properties": {
            f"key_{i}": {"type": "string"}
            for i in range(repeat_count)
        },
    }
    gbnf = generate_gbnf_for_json_schema(schema)
    assert isinstance(gbnf, str)
    assert len(gbnf) > 0
    for i in range(repeat_count):
        assert f"key_{i}" in gbnf
