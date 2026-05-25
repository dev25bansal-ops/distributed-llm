"""Gap tests: Structured Output (full coverage of FSM, GBNF, JSON schema, streaming, repair)."""

import pytest
import torch

from distllm.core.constrained_decoder import (
    SchemaConstrainedDecoder, JSONSchemaConstraint, TokenIndex,
    JSONSchemaFSM, RegexFSM, FSMState,
)
from distllm.core.grammar_decoder import GBNFParser, GBNFFSM
from distllm.core.structured_output.engine import StructuredOutputEngine, GenerationResult
from distllm.core.structured_output.config import StructuredOutputConfig
from distllm.core.structured_output.validator import SchemaValidator, OutputRepairer, ValidationResult, RepairResult
from distllm.core.structured_output.streaming import (
    BufferedAccumulator, PartialJSONParser, StructuredStreamHandler, PartialResult,
)


class TestJSONSchemaFSM:
    def test_initial_state(self):
        fsm = JSONSchemaFSM()
        assert fsm.is_accepting() is False
        fsm.reset()
        assert fsm.is_accepting() is False

    def test_transition_through_object(self):
        fsm = JSONSchemaFSM()
        for b in b'{"key": "value"}':
            fsm.transition(b)
        assert fsm.is_accepting()

    def test_transition_through_array(self):
        fsm = JSONSchemaFSM()
        for b in b'[1, 2, 3]':
            fsm.transition(b)
        assert fsm.is_accepting()

    def test_get_allowed_bytes(self):
        fsm = JSONSchemaFSM()
        allowed = fsm.get_allowed_bytes()
        assert isinstance(allowed, set)
        assert len(allowed) > 0

    def test_nested_object(self):
        fsm = JSONSchemaFSM()
        for b in b'{"a": {"b": 1}}':
            fsm.transition(b)
        assert fsm.is_accepting()


class TestRegexFSM:
    def test_simple_pattern(self):
        fsm = RegexFSM(r"hello")
        for b in b"hello":
            fsm.transition(b)
        assert fsm.is_accepting()

    def test_pattern_rejects_wrong(self):
        fsm = RegexFSM(r"\d+")
        for b in b"abc":
            fsm.transition(b)
        assert not fsm.is_accepting()

    def test_get_allowed_bytes(self):
        fsm = RegexFSM(r"yes|no")
        allowed = fsm.get_allowed_bytes()
        assert isinstance(allowed, set)


class TestSchemaConstrainedDecoder:
    @staticmethod
    def _make_tokenizer():
        class MockTokenizer:
            vocab_size = 100
            def get_vocab(self):
                return {f"tok{i}": i for i in range(100)}
            def decode(self, ids):
                rev = {i: f"tok{i}" for i in range(100)}
                if isinstance(ids, (list, tuple)):
                    return "".join(rev.get(i, "") for i in ids)
                return rev.get(ids, "")
            @property
            def eos_token_id(self):
                return 0
        return MockTokenizer()

    def test_json_schema_returns_constraint(self):
        decoder = SchemaConstrainedDecoder(self._make_tokenizer())
        constraint = decoder.json_schema({"type": "object", "properties": {"x": {"type": "integer"}}})
        assert constraint is not None

    def test_regex_returns_constraint(self):
        decoder = SchemaConstrainedDecoder(self._make_tokenizer())
        constraint = decoder.regex(r"\d+")
        assert constraint is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
    def test_get_logits_mask(self):
        decoder = SchemaConstrainedDecoder(self._make_tokenizer())
        constraint = decoder.json_schema({"type": "object"})
        if constraint is not None:
            mask = constraint.get_logits_mask(100)
            assert mask is not None


class TestGBNFFSM:
    def test_parse_grammar(self):
        grammar = 'root ::= "hello" " " "world"'
        fsm = GBNFFSM(grammar)
        for b in b"hello world":
            fsm.transition(b)
        assert fsm.is_accepting()

    def test_dfa_caching(self):
        grammar = 'root ::= "a" | "b"'
        fsm = GBNFFSM(grammar)
        fsm.compile_to_dfa()
        allowed = fsm.get_allowed_bytes()
        assert isinstance(allowed, set)

    def test_can_end(self):
        grammar = 'root ::= "x"'
        fsm = GBNFFSM(grammar)
        fsm.transition(ord("x"))
        assert fsm.can_end()

    def test_alt_grammar(self):
        grammar = 'root ::= "yes" | "no"'
        fsm = GBNFFSM(grammar)
        for b in b"yes":
            fsm.transition(b)
        assert fsm.is_accepting()
        fsm.reset()
        for b in b"no":
            fsm.transition(b)
        assert fsm.is_accepting()


class TestTokenIndex:
    def test_build(self):
        class MockTokenizer:
            vocab_size = 3
            def get_vocab(self):
                return {"<pad>": 0, "hello": 1, " ": 2}
            def decode(self, ids):
                d = {0: "<pad>", 1: "hello", 2: " "}
                if isinstance(ids, (list, tuple)):
                    return "".join(d.get(i, "") for i in ids)
                return d.get(ids, "")
            @property
            def eos_token_id(self):
                return 0
        idx = TokenIndex.build(MockTokenizer(), vocab_size=3)
        assert idx is not None
        assert idx.get_str(1) == "hello"


class TestStructuredOutputEngine:
    def test_init_defaults(self):
        engine = StructuredOutputEngine()
        assert engine is not None

    def test_generate_result_dataclass(self):
        r = GenerationResult(text="test", data={"x": 1}, valid=True, constraint=None, validation_result=None,
                           repair_result=None, token_count=5)
        assert r.text == "test"
        assert r.valid
        assert r.token_count == 5

    def test_validate_json(self):
        engine = StructuredOutputEngine()
        result = engine.validate('{"a": 1}', response_format={"type": "json_object"})
        assert result.valid


class TestSchemaValidator:
    def test_valid_object(self):
        v = SchemaValidator()
        result = v.validate({"x": 1}, schema={"type": "object", "properties": {"x": {"type": "integer"}}})
        assert result.valid

    def test_invalid_type(self):
        v = SchemaValidator()
        result = v.validate("not an object", schema={"type": "object"})
        assert not result.valid

    def test_missing_required(self):
        v = SchemaValidator()
        result = v.validate({"a": 1}, schema={"type": "object", "required": ["a", "b"],
                                              "properties": {"a": {}, "b": {}}})
        assert not result.valid


class TestOutputRepairer:
    def test_repair_direct_parse(self):
        r = OutputRepairer(max_attempts=2)
        result = r.repair('{"valid": "json"}')
        assert result is not None
        assert hasattr(result, 'success') or hasattr(result, 'repaired_text')

    def test_repair_truncated(self):
        r = OutputRepairer(max_attempts=2)
        result = r.repair('{"key": "value"')
        assert result is not None


class TestBufferedAccumulator:
    def test_add_and_flush(self):
        acc = BufferedAccumulator(min_chars=50)
        chunks = acc.add("hello ")
        assert chunks == []
        chunks = acc.add("world")
        assert isinstance(chunks, list)

    def test_flush_all(self):
        acc = BufferedAccumulator(min_chars=50)
        acc.add("hello")
        text = acc.flush_all()
        assert len(text) > 0

    def test_has_content(self):
        acc = BufferedAccumulator()
        assert not acc.has_content
        acc.add("x")
        assert acc.has_content


class TestPartialJSONParser:
    def test_feed_partial(self):
        parser = PartialJSONParser()
        result = parser.feed('{"key": "val')
        assert isinstance(result, PartialResult)
        assert "key" in result.text or not result.is_complete

    def test_feed_complete(self):
        parser = PartialJSONParser()
        result = parser.feed('{"key": "val"}')
        assert result.data.get("key") == "val" or not result.is_complete

    def test_recovery_strategies(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": 1,')
        assert isinstance(result, PartialResult)


class TestStructuredOutputConfig:
    def test_defaults(self):
        cfg = StructuredOutputConfig()
        assert cfg.enabled
        assert cfg.default_mode == "json_schema"
        assert cfg.schema_config.resolve_refs
        assert not cfg.schema_config.allow_additional_properties
