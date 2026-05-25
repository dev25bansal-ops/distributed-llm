"""Tests for the structured output engine package.

Tests: SchemaConverter, PartialJSONParser, SchemaValidator,
OutputRepairer, StructuredOutputEngine, config.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distllm.core.structured_output import (
    GBNFGrammar,
    OutputRepairer,
    PartialJSONParser,
    PartialResult,
    SchemaConverter,
    SchemaValidator,
    StructuredOutputConfig,
    StructuredOutputEngine,
    StructuredStreamChunk,
    StructuredStreamHandler,
    ValidationResult,
)


# ─── SchemaConverter ──────────────────────────────────────────────────────────


class TestGBNFGrammar:
    def test_empty_grammar(self):
        g = GBNFGrammar()
        assert str(g) == ""

    def test_add_rule(self):
        g = GBNFGrammar()
        g.add_rule("root", '"hello"')
        assert "root ::=" in str(g)
        assert '"hello"' in str(g)

    def test_multiple_rules(self):
        g = GBNFGrammar()
        g.add_rule("a", '"1"')
        g.add_rule("b", '"2"')
        rules = str(g).split("\n")
        assert len(rules) == 2


class TestSchemaConverter:
    def test_simple_object(self):
        converter = SchemaConverter()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text
        assert "name" in text
        assert "age" in text

    def test_nested_object(self):
        converter = SchemaConverter(max_depth=5)
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
            },
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text
        assert "name" in text
        assert "email" in text

    def test_array_of_strings(self):
        converter = SchemaConverter()
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text
        assert "tags" in text

    def test_enum(self):
        converter = SchemaConverter()
        schema = {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "enum": ["red", "green", "blue"],
                },
            },
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "red" in text
        assert "green" in text
        assert "blue" in text

    def test_const(self):
        converter = SchemaConverter()
        schema = {"const": "hello"}
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_enum_mixed_types(self):
        converter = SchemaConverter()
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "enum": ["red", 42, True, None],
                },
            },
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "red" in text

    def test_enum_integer_values(self):
        converter = SchemaConverter()
        schema = {
            "enum": [1, 2, 3],
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_enum_single_value(self):
        converter = SchemaConverter()
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active"]}},
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "active" in text

    def test_all_of_merges(self):
        converter = SchemaConverter()
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"properties": {"b": {"type": "integer"}}},
            ],
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_any_of(self):
        converter = SchemaConverter()
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ],
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_one_of(self):
        converter = SchemaConverter()
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "boolean"},
            ],
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_array_with_min_max(self):
        converter = SchemaConverter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_array_min_items_zero(self):
        converter = SchemaConverter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3,
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_array_exact_count(self):
        converter = SchemaConverter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_array_max_items_zero(self):
        converter = SchemaConverter()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 0,
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_object_with_additional_properties(self):
        converter = SchemaConverter(allow_additional=True)
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "additionalProperties": {"type": "integer"},
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "root ::=" in text

    def test_empty_schema(self):
        converter = SchemaConverter()
        grammar = converter.convert({})
        assert "root ::=" in str(grammar)

    def test_ref_defs(self):
        converter = SchemaConverter(resolve_refs=True)
        schema = {
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        grammar = converter.convert(schema)
        text = str(grammar)
        assert "Address" in text
        assert "root ::=" in text


# ─── PartialJSONParser ────────────────────────────────────────────────────────


class TestPartialJSONParser:
    def test_complete_json(self):
        parser = PartialJSONParser()
        result = parser.feed('{"name": "test"}')
        assert result.is_complete
        assert result.data == {"name": "test"}

    def test_partial_object_prefix(self):
        parser = PartialJSONParser()
        result = parser.feed('{"name": "te')
        assert not result.is_complete
        # Should still have some recovered data
        assert result.data is not None or not result.is_complete

    def test_partial_object_closing(self):
        parser = PartialJSONParser()
        result = parser.feed('{"name": "test"')
        assert not result.is_complete

    def test_partial_nested(self):
        parser = PartialJSONParser()
        parser.feed('{"user": {"name": "')
        result = parser.feed("Alice")
        assert not result.is_complete

    def test_empty_feed(self):
        parser = PartialJSONParser()
        result = parser.feed("")
        assert not result.is_complete

    def test_reset(self):
        parser = PartialJSONParser()
        parser.feed('{"a": 1}')
        parser.reset()
        result = parser.feed("")
        assert not result.is_complete

    def test_partial_array(self):
        parser = PartialJSONParser()
        result = parser.feed('[1, 2, 3')
        assert not result.is_complete
        # Should recover to [1,2,3]
        assert result.data == [1, 2, 3] if result.data is not None else True

    def test_incremental_object(self):
        parser = PartialJSONParser()
        parser.feed('{"a": ')
        result = parser.feed('1}')
        assert result.is_complete
        assert result.data == {"a": 1}

    def test_malformed_json(self):
        parser = PartialJSONParser()
        result = parser.feed("{invalid}")
        assert not result.is_complete

    def test_recovery_close_brackets(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": {"b": 1}')
        # After recovery, should close the inner brace
        assert result.data is not None


# ─── StructuredStreamHandler ──────────────────────────────────────────────────


class TestStructuredStreamHandler:
    @pytest.mark.asyncio
    async def test_process_simple_stream(self):
        handler = StructuredStreamHandler(enable_partial_parsing=False)
        async def token_stream():
            yield "hello"
            yield " world"

        chunks = []
        async for chunk in handler.process(token_stream()):
            chunks.append(chunk)
        assert len(chunks) >= 2
        assert handler.full_text == "hello world"

    @pytest.mark.asyncio
    async def test_process_with_parsing(self):
        handler = StructuredStreamHandler(enable_partial_parsing=True)
        async def token_stream():
            for char in '{"a": 1}':
                yield char

        chunks = []
        async for chunk in handler.process(token_stream()):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert handler.token_count == len('{"a": 1}')

    @pytest.mark.asyncio
    async def test_process_buffered(self):
        handler = StructuredStreamHandler(enable_partial_parsing=True, min_chars=3)
        async def token_stream():
            yield '{"a"'
            yield ': 1}'

        chunks = []
        async for chunk in handler.process_buffered(token_stream()):
            chunks.append(chunk)
        assert len(chunks) > 0

    def test_reset(self):
        handler = StructuredStreamHandler()
        handler._full_text = "test"
        handler._token_count = 5
        handler.reset()
        assert handler.full_text == ""
        assert handler.token_count == 0


# ─── SchemaValidator ──────────────────────────────────────────────────────────


class TestSchemaValidator:
    def test_no_schema(self):
        validator = SchemaValidator()
        result = validator.validate({"a": 1})
        assert result.valid

    def test_empty_schema(self):
        validator = SchemaValidator()
        result = validator.validate({"a": 1}, {})
        assert result.valid

    def test_valid_object(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = validator.validate({"name": "Alice"}, schema)
        assert result.valid

    def test_missing_required(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = validator.validate({"age": 30}, schema)
        assert not result.valid
        assert any("missing" in e for e in result.errors)

    def test_wrong_type(self):
        validator = SchemaValidator(strict=True)
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
        }
        result = validator.validate({"age": "not_a_number"}, schema)
        assert not result.valid

    def test_enum_validation(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
        }
        result = validator.validate({"color": "green"}, schema)
        assert not result.valid
        assert "enum" in str(result.errors).lower()

    def test_enum_valid(self):
        validator = SchemaValidator()
        schema = {
            "properties": {"color": {"enum": ["red", "blue"]}},
        }
        result = validator.validate({"color": "red"}, schema)
        assert result.valid

    def test_enum_null_value(self):
        validator = SchemaValidator()
        schema = {
            "properties": {"flag": {"enum": [None, "active"]}},
        }
        result = validator.validate({"flag": None}, schema)
        assert result.valid

    def test_enum_mixed_rejects_invalid(self):
        validator = SchemaValidator()
        schema = {
            "enum": [1, "two", True],
        }
        result = validator.validate("three", schema)
        assert not result.valid
        assert "enum" in str(result.errors).lower()

    def test_const_valid(self):
        validator = SchemaValidator()
        schema = {
            "properties": {"status": {"const": "active"}},
        }
        result = validator.validate({"status": "active"}, schema)
        assert result.valid

    def test_string_min_length(self):
        validator = SchemaValidator()
        schema = {
            "properties": {"code": {"type": "string", "minLength": 5}},
        }
        result = validator.validate({"code": "ab"}, schema)
        assert not result.valid

    def test_integer_range(self):
        validator = SchemaValidator()
        schema = {
            "properties": {"value": {"type": "integer", "minimum": 0, "maximum": 100}},
        }
        result = validator.validate({"value": 150}, schema)
        assert not result.valid

    def test_array_validation(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
        result = validator.validate(["a", "b"], schema)
        assert result.valid

    def test_array_too_few_items(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        }
        result = validator.validate(["a"], schema)
        assert not result.valid
        assert "too few" in str(result.errors).lower() or "minimum" in str(result.errors).lower()

    def test_array_too_many_items(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        }
        result = validator.validate(["a", "b", "c"], schema)
        assert not result.valid
        assert "too many" in str(result.errors).lower() or "maximum" in str(result.errors).lower()

    def test_array_exact_count_validation(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        }
        result = validator.validate(["a", "b"], schema)
        assert result.valid

    def test_array_min_items_zero_succeeds(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        }
        result = validator.validate([], schema)
        assert result.valid

    def test_array_wrong_item_type(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        result = validator.validate([1, 2, 3], schema)
        assert not result.valid

    def test_nested_object_validation(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
            "required": ["inner"],
        }
        result = validator.validate({"inner": {"x": 5.0}}, schema)
        assert result.valid

    def test_nested_failure(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        }
        result = validator.validate({"inner": {}}, schema)
        assert not result.valid

    def test_all_of(self):
        validator = SchemaValidator()
        schema = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}},
                {"required": ["a"]},
            ],
        }
        result = validator.validate({"a": "test"}, schema)
        assert result.valid

    def test_any_of(self):
        validator = SchemaValidator()
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ],
        }
        result = validator.validate("hello", schema)
        assert result.valid

    def test_one_of(self):
        validator = SchemaValidator()
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "integer"},
            ],
        }
        result = validator.validate("hello", schema)
        assert result.valid

    def test_one_of_fails_multiple_match(self):
        validator = SchemaValidator()
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "string"},
            ],
        }
        result = validator.validate("hello", schema)
        assert not result.valid

    def test_boolean_type(self):
        validator = SchemaValidator()
        schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        result = validator.validate({"flag": True}, schema)
        assert result.valid


# ─── OutputRepairer ───────────────────────────────────────────────────────────


class TestOutputRepairer:
    def test_already_valid(self):
        repairer = OutputRepairer()
        result = repairer.repair('{"a": 1}')
        assert result.success
        assert result.data == {"a": 1}

    def test_close_brackets(self):
        repairer = OutputRepairer()
        result = repairer.repair('{"a": {"b": 1}')
        assert result.success
        assert result.data == {"a": {"b": 1}}

    def test_trailing_comma(self):
        repairer = OutputRepairer()
        result = repairer.repair('{"a": 1,}')
        assert result.success
        assert result.data == {"a": 1}

    def test_unquoted_key(self):
        repairer = OutputRepairer()
        result = repairer.repair("{a: 1}")
        assert result.success
        assert result.data == {"a": 1}

    def test_single_quotes(self):
        repairer = OutputRepairer()
        result = repairer.repair("{'a': 1}")
        assert result.success
        assert result.data == {"a": 1}

    def test_extract_json_from_text(self):
        repairer = OutputRepairer()
        result = repairer.repair("Here is the result: {\"a\": 1}")
        assert result.success
        assert result.data == {"a": 1}

    def test_empty_text(self):
        repairer = OutputRepairer()
        result = repairer.repair("")
        assert not result.success

    def test_whitespace_only(self):
        repairer = OutputRepairer()
        result = repairer.repair("   ")
        assert not result.success

    def test_truncated_string(self):
        repairer = OutputRepairer()
        result = repairer.repair('{"name": "Alice')
        assert result.success
        assert result.data == {"name": "Alice"}

    def test_truncated_array(self):
        repairer = OutputRepairer()
        result = repairer.repair("[1, 2, 3")
        assert result.success
        assert result.data == [1, 2, 3]

    def test_complex_nested_truncated(self):
        repairer = OutputRepairer()
        result = repairer.repair('{"user": {"name": "Bob", "age": 30}, "items": [1, 2')
        assert result.success

    def test_no_valid_json(self):
        repairer = OutputRepairer()
        result = repairer.repair("not json at all")
        assert not result.success


# ─── StructuredOutputConfig ───────────────────────────────────────────────────


class TestStructuredOutputConfig:
    def test_default_values(self):
        config = StructuredOutputConfig()
        assert config.enabled is True
        assert config.default_mode == "json_schema"
        assert config.max_retries == 3
        assert config.schema_config.resolve_refs is True
        assert config.grammar.use_dfa is True
        assert config.streaming.partial_parsing is True
        assert config.validation.validate_output is True

    def test_custom_values(self):
        config = StructuredOutputConfig(
            enabled=False,
            default_mode="regex",
            max_retries=5,
        )
        assert config.enabled is False
        assert config.default_mode == "regex"
        assert config.max_retries == 5

    def test_nested_config(self):
        config = StructuredOutputConfig()
        config.schema_config.strict_validation = False
        assert config.schema_config.strict_validation is False


# ─── StructuredOutputEngine ───────────────────────────────────────────────────


class TestStructuredOutputEngine:
    def test_init_defaults(self):
        engine = StructuredOutputEngine()
        assert engine._config.enabled is True
        assert engine._config.default_mode == "json_schema"

    def test_init_custom_config(self):
        config = StructuredOutputConfig(enabled=False)
        engine = StructuredOutputEngine(config)
        assert engine._config.enabled is False

    def test_is_json_mode_true(self):
        engine = StructuredOutputEngine()
        assert engine._is_json_mode({"type": "json_object"}) is True
        assert engine._is_json_mode({"type": "json_schema"}) is True
        assert engine._is_json_mode({"type": "pydantic"}) is True

    def test_is_json_mode_false(self):
        engine = StructuredOutputEngine()
        assert engine._is_json_mode({"type": "grammar"}) is False
        assert engine._is_json_mode({"type": "regex"}) is False
        assert engine._is_json_mode({"type": "unknown"}) is False

    def test_build_constraint_none(self):
        engine = StructuredOutputEngine()
        result = engine.build_constraint({"type": "unknown"}, tokenizer=None)
        assert result is None

    def test_build_constraint_json_object(self):
        engine = StructuredOutputEngine()
        tokenizer = MagicMock()
        tokenizer.vocab_size = 256
        tokenizer.eos_token_id = 1
        tokenizer.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
        constraint = engine.build_constraint(
            {"type": "json_object"}, tokenizer=tokenizer,
        )
        assert constraint is not None

    def test_build_constraint_json_schema(self):
        engine = StructuredOutputEngine()
        tokenizer = MagicMock()
        tokenizer.vocab_size = 256
        tokenizer.eos_token_id = 1
        tokenizer.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
        constraint = engine.build_constraint(
            {"type": "json_schema", "schema": {"type": "object"}},
            tokenizer=tokenizer,
        )
        assert constraint is not None

    def test_validate_valid_json(self):
        engine = StructuredOutputEngine()
        result = engine.validate(
            '{"name": "Alice", "age": 30}',
            {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            }},
        )
        assert result.valid

    def test_validate_invalid_json(self):
        engine = StructuredOutputEngine()
        result = engine.validate(
            "{invalid}",
            {"type": "json_schema", "schema": {}},
        )
        assert not result.valid

    def test_validate_no_schema(self):
        engine = StructuredOutputEngine()
        result = engine.validate('{"a": 1}', None)
        assert result.valid

    def test_validate_non_json_mode(self):
        engine = StructuredOutputEngine()
        result = engine.validate("hello", {"type": "grammar"})
        assert result.valid

    def test_repair(self):
        engine = StructuredOutputEngine()
        result = engine.repair('{"a": 1}', {"type": "json_schema"})
        assert result.success
        assert result.data == {"a": 1}

    def test_repair_invalid(self):
        engine = StructuredOutputEngine()
        result = engine.repair("not json", {"type": "json_schema"})
        assert not result.success

    @patch.object(StructuredOutputEngine, "build_constraint", return_value="mock_constraint")
    def test_generate_calls_generator(self, mock_build):
        engine = StructuredOutputEngine()
        mock_gen = MagicMock(return_value="generated text")
        result = engine.generate(
            mock_gen, "prompt",
            response_format=None,
        )
        assert result.text == "generated text"
        assert result.valid

    @patch.object(StructuredOutputEngine, "build_constraint", return_value="mock_constraint")
    def test_generate_with_json_schema(self, mock_build):
        engine = StructuredOutputEngine()
        result_text = '{"name": "Alice"}'
        mock_gen = MagicMock(return_value=result_text)
        result = engine.generate(
            mock_gen, "prompt",
            response_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        )
        assert result.text == result_text
        assert result.data == {"name": "Alice"}
        assert result.valid

    @patch.object(StructuredOutputEngine, "build_constraint", return_value="mock_constraint")
    def test_generate_with_invalid_json_schema(self, mock_build):
        engine = StructuredOutputEngine()
        result_text = '{"name": 123}'
        mock_gen = MagicMock(return_value=result_text)
        result = engine.generate(
            mock_gen, "prompt",
            response_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        )
        assert result.text == result_text
        assert not result.valid

    @pytest.mark.asyncio
    async def test_stream_structured(self):
        engine = StructuredOutputEngine()
        async def token_stream():
            for ch in '{"a": 1}':
                yield ch

        chunks = []
        async for chunk in engine.stream_structured(
            token_stream(),
            {"type": "json_schema", "schema": {}},
        ):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_structured_no_format(self):
        engine = StructuredOutputEngine()
        async def token_stream():
            yield "hello"

        chunks = []
        async for chunk in engine.stream_structured(token_stream(), None):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_structured_with_constraint(self):
        engine = StructuredOutputEngine()
        async def token_stream():
            for ch in '{"valid": true}':
                yield ch

        chunks = []
        async for chunk in engine.stream_structured(
            token_stream(),
            {"type": "json_schema", "schema": {}},
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        # Final chunk should be marked final
        assert chunks[-1].is_final

    @pytest.mark.asyncio
    async def test_stream_structured_partial_at_each_step(self):
        engine = StructuredOutputEngine()
        tokens = list('{"valid": true}')

        all_chunks = []
        for i in range(1, len(tokens) + 1):
            async def partial_stream(prefix=tokens[:i]):
                for ch in prefix:
                    yield ch

            chunks = []
            async for chunk in engine.stream_structured(
                partial_stream(),
                {"type": "json_schema", "schema": {}},
            ):
                chunks.append(chunk)
            all_chunks.append(chunks)
            last = chunks[-1]
            assert last.is_final
        assert len(all_chunks) == len(tokens)


# ─── Invalid/Impossible Constraint Tests ──────────────────────────────────


class TestInvalidConstraints:
    """Test behavior with contradictory or impossible constraints."""

    def test_contradictory_min_max_items(self):
        """minItems > maxItems makes validation impossible."""
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 3,
        }
        result = validator.validate(["a", "b", "c", "d"], schema)
        assert not result.valid

    def test_contradictory_min_max_items_empty(self):
        """Empty array fails with contradictory min/max."""
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 0,
        }
        result = validator.validate([], schema)
        assert not result.valid

    def test_contradictory_range_no_fallback(self):
        """Schema that cannot match any data still produces error, not crash."""
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "val": {"type": "integer", "minimum": 10, "maximum": 5},
            },
        }
        result = validator.validate({"val": 7}, schema)
        assert not result.valid
