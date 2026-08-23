"""Tests for GBNF grammar parser and FSM.

Converted from script-style standalone test to proper pytest format.
"""

from __future__ import annotations

import torch

from distllm.core.grammar_decoder import (
    GBNFFSM,
    GBNFParser,
    LiteralNode,
    SeqNode,
    AltNode,
)


class TestGBNFParser:
    """GBNF grammar parsing."""

    def test_parse_literal(self):
        grammar = 'root ::= "hello"'
        parser = GBNFParser(grammar)
        rules = parser.parse()
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "hello"

    def test_parse_alternation(self):
        grammar = 'root ::= "yes" | "no"'
        parser = GBNFParser(grammar)
        rules = parser.parse()
        assert isinstance(rules["root"], (AltNode, SeqNode, LiteralNode))


class TestGBNFFSM:
    """GBNF finite state machine."""

    def test_literal_first_byte(self):
        fsm = GBNFFSM('root ::= "hello"')
        allowed = fsm.get_allowed_bytes()
        assert 0x68 in allowed  # 'h'

    def test_literal_transitions(self):
        fsm = GBNFFSM('root ::= "hello"')
        fsm.transition(0x68)  # h
        allowed = fsm.get_allowed_bytes()
        assert 0x65 in allowed  # 'e'
        assert not fsm.is_accepting()

        fsm.transition(0x65)  # e
        fsm.transition(0x6C)  # l
        fsm.transition(0x6C)  # l
        fsm.transition(0x6F)  # o
        assert fsm.is_accepting()

    def test_char_class_digits(self):
        fsm = GBNFFSM('root ::= [0-9]+')
        allowed = fsm.get_allowed_bytes()
        assert 0x30 in allowed  # 0
        assert 0x39 in allowed  # 9

        fsm.transition(0x35)  # 5
        allowed = fsm.get_allowed_bytes()
        assert 0x30 in allowed  # digits still allowed after digit

    def test_alternation(self):
        fsm = GBNFFSM('root ::= "yes" | "no"')
        allowed = fsm.get_allowed_bytes()
        assert 0x79 in allowed  # y
        assert 0x6E in allowed  # n

    def test_grammar_constrained_decoder(self):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder

        class FakeTokenizer:
            vocab_size = 100
            eos_token_id = 1

            def get_vocab(self):
                return {chr(i): i for i in range(32, 96)} | {f"t{i}": i for i in range(96, 100)}

        tok = FakeTokenizer()
        decoder = SchemaConstrainedDecoder(tok)
        constraint = decoder.grammar('root ::= [a-zA-Z]+')
        mask = constraint.get_logits_mask(100)
        assert mask.sum().item() > 0, "grammar should allow some tokens"
