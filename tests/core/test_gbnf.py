"""Tests for GBNF grammar parser and FSM.

Tests the current grammar_decoder API (not the old AST-node-based version).
"""

from __future__ import annotations

import pytest

from distllm.core.grammar_decoder import GBNFFSM, GBNFParser


class TestGBNFParser:
    """GBNF grammar parsing (new API: returns dict[str, list[list[str]]])."""

    def test_parse_literal(self):
        grammar = 'root ::= "hello"'
        parser = GBNFParser()
        rules = parser.parse(grammar)
        assert "root" in rules
        assert rules["root"] == [["hello"]]

    def test_parse_alternation(self):
        grammar = 'root ::= "yes" | "no"'
        parser = GBNFParser()
        rules = parser.parse(grammar)
        assert "root" in rules
        assert ["yes"] in rules["root"]
        assert ["no"] in rules["root"]

    def test_parse_empty(self):
        parser = GBNFParser()
        rules = parser.parse("")
        assert rules == {}


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

    def test_alternation(self):
        fsm = GBNFFSM('root ::= "yes" | "no"')
        allowed = fsm.get_allowed_bytes()
        assert 0x79 in allowed  # y
        assert 0x6E in allowed  # n

    def test_is_accepting_initial(self):
        fsm = GBNFFSM('root ::= "hello"')
        assert not fsm.is_accepting()

    def test_is_accepting_complete(self):
        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(ord("a"))
        assert fsm.is_accepting()

    def test_can_end(self):
        fsm = GBNFFSM('root ::= "a"?')
        assert fsm.can_end() == fsm.is_accepting()

    def test_reset(self):
        fsm = GBNFFSM('root ::= "hi"')
        fsm.transition(ord("h"))
        fsm.transition(ord("i"))
        assert fsm.is_accepting()
        fsm.reset()
        assert not fsm.is_accepting()

    def test_compile_to_dfa(self):
        fsm = GBNFFSM('root ::= "abc"')
        fsm.compile_to_dfa()
        assert fsm.get_allowed_bytes() == {ord("a")}
        fsm.transition(ord("a"))
        assert fsm.get_allowed_bytes() == {ord("b")}

    def test_get_logits_mask(self):
        class FakeTokenizer:
            vocab_size = 100
            eos_token_id = 1

            def decode(self, token_ids):
                return chr(token_ids[0]) if token_ids and token_ids[0] < 128 else ""

            def get_vocab(self):
                return {chr(i): i for i in range(32, 96)} | {f"t{i}": i for i in range(96, 100)}

        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(ord("a"))
        mask = fsm.get_logits_mask(100, FakeTokenizer())
        assert mask.sum().item() > 0

    def test_invalid_byte(self):
        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(256)
        assert fsm.get_allowed_bytes() == set()
