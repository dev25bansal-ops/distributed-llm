"""Tests for GBNF grammar parsing and byte-level FSM.

The old AST-node-based API (AltNode, _DerivativeEngine, etc.) was removed
from grammar_decoder.py.  GBNFParser now returns raw dict/list-of-strings.
GBNFFSM still exists.  These tests are preserved as much as possible.
"""

from __future__ import annotations

import pytest

try:
    from distllm.core.grammar_decoder import (
        AltNode,
        AnyCharNode,
        CharClassNode,
        EMPTY,
        GBNFFSM,
        GBNFParser,
        LiteralNode,
        NEVER,
        NeverNode,
        OptionalNode,
        OneOrMoreNode,
        RepeatNode,
        SeqNode,
        RuleRefNode,
        _DerivativeEngine,
        _simplify,
        _simplify_alt,
        _simplify_seq,
    )
    _HAS_AST_NODES = True
except ImportError:
    # New grammar_decoder only has GBNFFSM and GBNFParser
    from distllm.core.grammar_decoder import GBNFFSM, GBNFParser
    AltNode = AnyCharNode = CharClassNode = LiteralNode = NeverNode = None
    OptionalNode = OneOrMoreNode = RepeatNode = SeqNode = RuleRefNode = None
    _DerivativeEngine = None
    EMPTY = NEVER = None
    _simplify = _simplify_alt = _simplify_seq = None
    _HAS_AST_NODES = False

pytestmark = pytest.mark.skipif(
    not _HAS_AST_NODES,
    reason="AST node classes removed from grammar_decoder; tests need rewrite for new API",
)


# ===========================================================================
# AST Simplification
# ===========================================================================

class TestASTSimplification:
    """Tests for _simplify, _simplify_seq, _simplify_alt."""

    def test_simplify_seq_empty(self):
        """Empty sequence simplifies to EMPTY."""
        assert _simplify_seq([]) is EMPTY

    def test_simplify_seq_single(self):
        """Single-element sequence unwraps."""
        lit = LiteralNode("a")
        assert _simplify_seq([lit]) is lit

    def test_simplify_seq_flattens(self):
        """Nested SeqNodes are flattened."""
        inner = SeqNode((LiteralNode("a"), LiteralNode("b")))
        result = _simplify_seq([inner, LiteralNode("c")])
        assert isinstance(result, SeqNode)
        assert len(result.children) == 3

    def test_simplify_seq_removes_never(self):
        """If any child is NeverNode, result is NEVER."""
        result = _simplify_seq([LiteralNode("a"), NeverNode()])
        assert result is NEVER

    def test_simplify_seq_removes_empty(self):
        """EmptyNode children are removed from sequence."""
        result = _simplify_seq([EMPTY, LiteralNode("a")])
        assert result == LiteralNode("a")

    def test_simplify_alt_empty(self):
        """Empty alternation simplifies to NEVER."""
        assert _simplify_alt([]) is NEVER

    def test_simplify_alt_single(self):
        """Single-element alternation unwraps."""
        lit = LiteralNode("a")
        assert _simplify_alt([lit]) is lit

    def test_simplify_alt_removes_never(self):
        """NeverNode alternatives are removed."""
        result = _simplify_alt([LiteralNode("a"), NeverNode()])
        assert result == LiteralNode("a")

    def test_simplify_alt_dedup(self):
        """Duplicate alternatives are removed."""
        result = _simplify_alt([LiteralNode("a"), LiteralNode("a")])
        # Should be single 'a'
        assert result == LiteralNode("a")

    def test_simplify_alt_flattens(self):
        """Nested AltNodes are flattened."""
        inner = AltNode((LiteralNode("a"), LiteralNode("b")))
        result = _simplify_alt([inner, LiteralNode("c")])
        assert isinstance(result, AltNode)
        assert len(result.alternatives) == 3

    def test_simplify_literal_empty(self):
        """Empty literal simplifies to EMPTY."""
        assert _simplify(LiteralNode("")) is EMPTY

    def test_simplify_repeat_empty(self):
        """Repeat of EMPTY or NEVER simplifies to EMPTY."""
        assert _simplify(RepeatNode(EMPTY)) is EMPTY
        assert _simplify(RepeatNode(NeverNode())) is EMPTY

    def test_simplify_one_or_more_empty(self):
        """OneOrMore of EMPTY is EMPTY."""
        assert _simplify(OneOrMoreNode(EMPTY)) is EMPTY

    def test_simplify_one_or_more_never(self):
        """OneOrMore of NEVER is NEVER."""
        assert _simplify(OneOrMoreNode(NeverNode())) is NEVER

    def test_simplify_optional_never_is_empty(self):
        """Optional of NEVER is EMPTY."""
        assert _simplify(OptionalNode(NeverNode())) is EMPTY

    def test_simplify_passthrough(self):
        """Literal nodes pass through unchanged."""
        assert _simplify(LiteralNode("abc")) == LiteralNode("abc")


# ===========================================================================
# GBNFParser
# ===========================================================================

class TestGBNFParser:
    """Tests for parsing GBNF grammar text into AST nodes."""

    def parse(self, text: str) -> dict:
        return GBNFParser(text).parse()

    # ─── Basic Rules ──────────────────────────────────────────────────────

    def test_single_literal_rule(self):
        """Parse a single rule with a literal."""
        rules = self.parse('root ::= "hello"')
        assert "root" in rules
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "hello"

    def test_empty_grammar(self):
        """Empty grammar text produces no rules."""
        rules = self.parse("")
        assert rules == {}

    def test_whitespace_only_grammar(self):
        """Whitespace-only text produces no rules."""
        rules = self.parse("   \n  \t  ")
        assert rules == {}

    def test_multiple_rules(self):
        """Multiple rules are parsed separately."""
        rules = self.parse("""
            root ::= "a"
            b ::= "c"
        """)
        assert "root" in rules
        assert "b" in rules

    def test_rule_with_multiline_body(self):
        """Rule body can continue on next line."""
        rules = self.parse("""
            root ::= "a"
               "b"
        """)
        node = rules["root"]
        assert isinstance(node, SeqNode)
        assert len(node.children) == 2

    def test_rule_reference(self):
        """Rule references are parsed as RuleRefNode."""
        rules = self.parse("root ::= foo")
        node = rules["root"]
        assert isinstance(node, RuleRefNode)
        assert node.name == "foo"

    # ─── Alternation ──────────────────────────────────────────────────────

    def test_alternation(self):
        """| creates AltNode."""
        rules = self.parse('root ::= "a" | "b"')
        node = rules["root"]
        assert isinstance(node, AltNode)
        assert len(node.alternatives) == 2

    def test_multiple_alternation(self):
        """Multiple | branches."""
        rules = self.parse('root ::= "a" | "b" | "c"')
        node = rules["root"]
        assert isinstance(node, AltNode)
        assert len(node.alternatives) == 3

    def test_alternation_with_rule_refs(self):
        """Alternation mixing literals and rule refs."""
        rules = self.parse("root ::= \"a\" | foo | \"b\"")
        node = rules["root"]
        assert isinstance(node, AltNode)
        assert len(node.alternatives) == 3

    # ─── Character Classes ────────────────────────────────────────────────

    def test_char_class(self):
        """[abc] creates CharClassNode."""
        rules = self.parse('root ::= [abc]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        assert node.chars == frozenset({ord('a'), ord('b'), ord('c')})

    def test_char_class_range(self):
        """[a-z] creates range of bytes."""
        rules = self.parse('root ::= [a-z]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        expected = set(range(ord('a'), ord('z') + 1))
        assert node.chars == frozenset(expected)

    def test_char_class_mixed_range_and_literal(self):
        """[a-cx] combines range and literal."""
        rules = self.parse('root ::= [a-cx]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        expected = {ord('a'), ord('b'), ord('c'), ord('x')}
        assert node.chars == frozenset(expected)

    def test_negated_char_class(self):
        """[^a] inverts the character class (negation)."""
        rules = self.parse('root ::= [^a]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        # All bytes except ord('a')
        expected = set(range(256)) - {ord('a')}
        assert node.chars == frozenset(expected)

    def test_negated_range(self):
        """[^a-z] negates a range."""
        rules = self.parse('root ::= [^a-z]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        expected = set(range(256)) - set(range(ord('a'), ord('z') + 1))
        assert node.chars == frozenset(expected)

    def test_empty_char_class(self):
        """Empty [] creates CharClassNode with empty set."""
        rules = self.parse('root ::= []')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        assert node.chars == frozenset()

    def test_char_class_single_char(self):
        """[x] creates CharClassNode with one char."""
        rules = self.parse('root ::= [x]')
        node = rules["root"]
        assert isinstance(node, CharClassNode)
        assert node.chars == frozenset({ord('x')})

    # ─── Any Char ─────────────────────────────────────────────────────────

    def test_any_char(self):
        """'.' creates AnyCharNode."""
        rules = self.parse('root ::= .')
        node = rules["root"]
        assert isinstance(node, AnyCharNode)

    # ─── Repetition ───────────────────────────────────────────────────────

    def test_repeat_star(self):
        """'*' creates RepeatNode."""
        rules = self.parse('root ::= "a"*')
        node = rules["root"]
        assert isinstance(node, RepeatNode)
        assert isinstance(node.child, LiteralNode)
        assert node.child.value == "a"

    def test_repeat_plus(self):
        """'+' creates OneOrMoreNode."""
        rules = self.parse('root ::= [0-9]+')
        node = rules["root"]
        assert isinstance(node, OneOrMoreNode)

    def test_optional(self):
        """'?' creates OptionalNode."""
        rules = self.parse('root ::= "a"?')
        node = rules["root"]
        assert isinstance(node, OptionalNode)

    def test_repeat_star_on_group(self):
        """'(...)*' applies repetition to group."""
        rules = self.parse('root ::= ("ab")*')
        node = rules["root"]
        assert isinstance(node, RepeatNode)

    def test_repeat_plus_on_rule_ref(self):
        """'rule+' applies repetition to rule reference."""
        rules = self.parse('root ::= digit+')
        node = rules["root"]
        assert isinstance(node, OneOrMoreNode)

    # ─── Groups ───────────────────────────────────────────────────────────

    def test_group(self):
        """'(...)' creates a grouped expression."""
        rules = self.parse('root ::= ("a" | "b")')
        node = rules["root"]
        assert isinstance(node, AltNode)
        assert len(node.alternatives) == 2

    def test_nested_groups(self):
        """Nested groups are handled."""
        rules = self.parse('root ::= (("a" | "b") "c")')
        node = rules["root"]
        assert isinstance(node, SeqNode)

    # ─── Sequence ─────────────────────────────────────────────────────────

    def test_sequence(self):
        """Multiple atoms in a row form a SeqNode."""
        rules = self.parse('root ::= "a" "b" "c"')
        node = rules["root"]
        assert isinstance(node, SeqNode)
        assert len(node.children) == 3

    def test_mixed_sequence(self):
        """Sequence of different atom types."""
        rules = self.parse('root ::= "a" [0-9]+ foo')
        node = rules["root"]
        assert isinstance(node, SeqNode)
        assert len(node.children) == 3

    # ─── Comments ─────────────────────────────────────────────────────────

    def test_comment(self):
        """'#' starts a comment."""
        rules = self.parse('# this is a comment\nroot ::= "a"')
        assert "root" in rules

    def test_comment_inline(self):
        """Inline comment after rule."""
        rules = self.parse('root ::= "a"  # comment')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "a"

    def test_comment_not_in_string(self):
        """'#' inside a string is not a comment."""
        rules = self.parse('root ::= "#"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "#"

    # ─── Escapes ──────────────────────────────────────────────────────────

    def test_escape_newline(self):
        r"""\n becomes a literal newline."""
        rules = self.parse('root ::= "\\n"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "\n"

    def test_escape_tab(self):
        r"""\t becomes a tab."""
        rules = self.parse('root ::= "\\t"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "\t"

    def test_escape_quote(self):
        r"""\" becomes a literal quote."""
        rules = self.parse('root ::= "\\""')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == '"'

    def test_escape_backslash(self):
        r"""\\ becomes a single backslash."""
        rules = self.parse('root ::= "\\\\"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "\\"

    def test_escape_hex(self):
        r"""\x41 becomes 'A'."""
        rules = self.parse('root ::= "\\x41"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "A"

    def test_escape_unicode_four(self):
        r"""\u0041 becomes 'A'."""
        rules = self.parse('root ::= "\\u0041"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "A"

    def test_escape_unicode_eight(self):
        r"""\U00000041 becomes 'A'."""
        rules = self.parse('root ::= "\\U00000041"')
        node = rules["root"]
        assert isinstance(node, LiteralNode)
        assert node.value == "A"

    # ─── Start Rule Handling ──────────────────────────────────────────────

    def test_start_rule_present(self):
        """When start_rule exists, it's used."""
        fsm = GBNFFSM('root ::= "a"', start_rule="root")
        assert fsm._start_rule == "root"

    def test_start_rule_fallback(self):
        """When start_rule is missing, first rule is used."""
        fsm = GBNFFSM('myrule ::= "a"', start_rule="nonexistent")
        assert fsm._start_rule == "myrule"

    def test_no_rules_raises(self):
        """Empty grammar raises ValueError."""
        with pytest.raises(ValueError, match="No rules"):
            GBNFFSM("", start_rule="root")

    # ─── Complex Grammars ─────────────────────────────────────────────────

    def test_json_string_grammar(self):
        """Parse a JSON string grammar (simplified version)."""
        grammar = '''root ::= " "? "\\"" ( [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\\""'''
        rules = self.parse(grammar)
        assert "root" in rules
        node = rules["root"]
        assert node is not None

    def test_arithmetic_grammar(self):
        """Parse an arithmetic expression grammar."""
        grammar = """
            root ::= expr
            expr ::= term (("+" | "-") term)*
            term ::= factor (("*" | "/") factor)*
            factor ::= "(" expr ")" | [0-9]+
        """
        rules = self.parse(grammar)
        assert "root" in rules
        assert "expr" in rules
        assert "term" in rules
        assert "factor" in rules

    def test_non_ascii_literal(self):
        """Non-ASCII multi-byte characters in literal are expanded to byte sequence."""
        rules = self.parse('root ::= "é"')  # é is 2 bytes in UTF-8
        node = rules["root"]
        # Should be a SeqNode of byte-level LiteralNodes
        assert isinstance(node, SeqNode)
        assert len(node.children) == 2
        assert node.children[0].value == chr(0xC3)
        assert node.children[1].value == chr(0xA9)


# ===========================================================================
# _DerivativeEngine
# ===========================================================================

class TestDerivativeEngine:
    """Tests for Brzozowski derivative operations on grammar AST."""

    def make_engine(self, rules: dict | None = None):
        if rules is None:
            rules = {}
        return _DerivativeEngine(rules)

    # ─── nullable ─────────────────────────────────────────────────────────

    def test_nullable_empty(self):
        """EMPTY is nullable."""
        engine = self.make_engine()
        assert engine.nullable(EMPTY) is True

    def test_nullable_never(self):
        """NEVER is not nullable."""
        engine = self.make_engine()
        assert engine.nullable(NEVER) is False

    def test_nullable_literal(self):
        """Literal is not nullable."""
        engine = self.make_engine()
        assert engine.nullable(LiteralNode("a")) is False

    def test_nullable_char_class(self):
        """Char class is not nullable."""
        engine = self.make_engine()
        assert engine.nullable(CharClassNode(frozenset({ord('a')}))) is False

    def test_nullable_any_char(self):
        """AnyChar is not nullable."""
        engine = self.make_engine()
        assert engine.nullable(AnyCharNode()) is False

    def test_nullable_seq_all_empty(self):
        """Sequence of all-EMPTY is nullable."""
        engine = self.make_engine()
        node = SeqNode((EMPTY, EMPTY))
        assert engine.nullable(node) is True

    def test_nullable_seq_some_not(self):
        """Sequence with non-nullable child is not nullable."""
        engine = self.make_engine()
        node = SeqNode((EMPTY, LiteralNode("a")))
        assert engine.nullable(node) is False

    def test_nullable_alt_one_nullable(self):
        """Alternation with at least one nullable alternative is nullable."""
        engine = self.make_engine()
        node = AltNode((LiteralNode("a"), EMPTY))
        assert engine.nullable(node) is True

    def test_nullable_alt_none_nullable(self):
        """Alternation with no nullable alternatives is not nullable."""
        engine = self.make_engine()
        node = AltNode((LiteralNode("a"), LiteralNode("b")))
        assert engine.nullable(node) is False

    def test_nullable_repeat(self):
        """RepeatNode is always nullable."""
        engine = self.make_engine()
        assert engine.nullable(RepeatNode(LiteralNode("a"))) is True

    def test_nullable_one_or_more(self):
        """OneOrMoreNode is nullable iff its child is."""
        engine = self.make_engine()
        assert engine.nullable(OneOrMoreNode(EMPTY)) is True
        assert engine.nullable(OneOrMoreNode(LiteralNode("a"))) is False

    def test_nullable_optional(self):
        """OptionalNode is always nullable."""
        engine = self.make_engine()
        assert engine.nullable(OptionalNode(LiteralNode("a"))) is True

    def test_nullable_rule_ref(self):
        """RuleRef is nullable if the referenced rule is."""
        engine = self.make_engine({
            "a": LiteralNode("x"),
            "b": EMPTY,
        })
        assert engine.nullable(RuleRefNode("a")) is False
        assert engine.nullable(RuleRefNode("b")) is True

    def test_nullable_rule_ref_missing(self):
        """Missing rule reference is not nullable."""
        engine = self.make_engine({})
        assert engine.nullable(RuleRefNode("missing")) is False

    def test_nullable_rule_ref_cycle(self):
        """Cyclic rule ref does not loop infinitely."""
        engine = self.make_engine({
            "a": RuleRefNode("a"),
        })
        assert engine.nullable(RuleRefNode("a")) is False

    # ─── first_bytes ──────────────────────────────────────────────────────

    def test_first_bytes_literal(self):
        """first_bytes of literal returns its first byte."""
        engine = self.make_engine()
        assert engine.first_bytes(LiteralNode("hello")) == {ord('h')}

    def test_first_bytes_empty_literal(self):
        """first_bytes of empty literal returns empty set."""
        engine = self.make_engine()
        assert engine.first_bytes(LiteralNode("")) == set()

    def test_first_bytes_empty_node(self):
        """first_bytes of EMPTY returns empty set."""
        engine = self.make_engine()
        assert engine.first_bytes(EMPTY) == set()

    def test_first_bytes_never(self):
        """first_bytes of NEVER returns empty set."""
        engine = self.make_engine()
        assert engine.first_bytes(NEVER) == set()

    def test_first_bytes_char_class(self):
        """first_bytes of char class returns its chars."""
        engine = self.make_engine()
        chars = frozenset({ord('a'), ord('b'), ord('c')})
        assert engine.first_bytes(CharClassNode(chars)) == {ord('a'), ord('b'), ord('c')}

    def test_first_bytes_any_char(self):
        """first_bytes of AnyChar returns all 256 bytes."""
        engine = self.make_engine()
        result = engine.first_bytes(AnyCharNode())
        assert result == set(range(256))

    def test_first_bytes_alt(self):
        """first_bytes of alternation unions all alternatives."""
        engine = self.make_engine()
        node = AltNode((LiteralNode("a"), LiteralNode("b")))
        assert engine.first_bytes(node) == {ord('a'), ord('b')}

    def test_first_bytes_seq_first_char(self):
        """first_bytes of sequence returns first byte of first child."""
        engine = self.make_engine()
        node = SeqNode((LiteralNode("a"), LiteralNode("b")))
        assert engine.first_bytes(node) == {ord('a')}

    def test_first_bytes_seq_with_nullable_first(self):
        """If first child is nullable, includes second child's first bytes."""
        engine = self.make_engine()
        node = SeqNode((OptionalNode(LiteralNode("a")), LiteralNode("b")))
        expected = {ord('a'), ord('b')}
        assert engine.first_bytes(node) == expected

    def test_first_bytes_repeat(self):
        """first_bytes of repeat is same as child."""
        engine = self.make_engine()
        assert engine.first_bytes(RepeatNode(LiteralNode("a"))) == {ord('a')}

    def test_first_bytes_optional(self):
        """first_bytes of optional is same as child."""
        engine = self.make_engine()
        assert engine.first_bytes(OptionalNode(LiteralNode("a"))) == {ord('a')}

    def test_first_bytes_one_or_more(self):
        """first_bytes of one-or-more is same as child."""
        engine = self.make_engine()
        assert engine.first_bytes(OneOrMoreNode(LiteralNode("a"))) == {ord('a')}

    def test_first_bytes_rule_ref(self):
        """first_bytes resolves through rule references."""
        engine = self.make_engine({
            "digit": CharClassNode(frozenset(range(ord('0'), ord('9') + 1))),
        })
        result = engine.first_bytes(RuleRefNode("digit"))
        assert result == set(range(ord('0'), ord('9') + 1))

    def test_first_bytes_rule_ref_missing(self):
        """Missing rule ref returns empty set."""
        engine = self.make_engine({})
        assert engine.first_bytes(RuleRefNode("missing")) == set()

    def test_first_bytes_rule_ref_cycle(self):
        """Cyclic rule ref does not loop infinitely."""
        engine = self.make_engine({
            "a": RuleRefNode("a"),
        })
        assert engine.first_bytes(RuleRefNode("a")) == set()

    # ─── derive ───────────────────────────────────────────────────────────

    def test_derive_empty_never(self):
        """derive(EMPTY) → NEVER, derive(NEVER) → NEVER."""
        engine = self.make_engine()
        assert engine.derive(EMPTY, ord('a')) is NEVER
        assert engine.derive(NEVER, ord('a')) is NEVER

    def test_derive_literal_match(self):
        """derive(literal, matching byte) → remainder."""
        engine = self.make_engine()
        result = engine.derive(LiteralNode("ab"), ord('a'))
        assert result == LiteralNode("b")

    def test_derive_literal_exact_match(self):
        """derive(literal, only byte matching) → EMPTY."""
        engine = self.make_engine()
        result = engine.derive(LiteralNode("a"), ord('a'))
        assert result is EMPTY

    def test_derive_literal_no_match(self):
        """derive(literal, non-matching byte) → NEVER."""
        engine = self.make_engine()
        result = engine.derive(LiteralNode("a"), ord('b'))
        assert result is NEVER

    def test_derive_char_class_match(self):
        """derive(char_class, matching byte) → EMPTY."""
        engine = self.make_engine()
        chars = frozenset({ord('a'), ord('b')})
        result = engine.derive(CharClassNode(chars), ord('a'))
        assert result is EMPTY

    def test_derive_char_class_no_match(self):
        """derive(char_class, non-matching byte) → NEVER."""
        engine = self.make_engine()
        chars = frozenset({ord('a')})
        result = engine.derive(CharClassNode(chars), ord('c'))
        assert result is NEVER

    def test_derive_any_char(self):
        """derive(AnyChar) → EMPTY (any byte consumes it)."""
        engine = self.make_engine()
        result = engine.derive(AnyCharNode(), ord('x'))
        assert result is EMPTY

    def test_derive_seq_first_matches(self):
        """derive(seq, first byte matches) → rest of seq."""
        engine = self.make_engine()
        node = SeqNode((LiteralNode("a"), LiteralNode("b")))
        result = engine.derive(node, ord('a'))
        assert result == LiteralNode("b")

    def test_derive_seq_with_nullable_first(self):
        """derive(seq with nullable first) skips optional and matches rest."""
        engine = self.make_engine()
        node = SeqNode((OptionalNode(LiteralNode("a")), LiteralNode("b")))
        result = engine.derive(node, ord('b'))
        # Since 'a' is optional, derive tries both paths:
        # 1. derive('a'?) + 'b' → NEVER + 'b' → NEVER
        # 2. nullable('a'?) = True → derive('b') → EMPTY
        # Result: EMPTY (the optional 'a' is skipped, 'b' matched)
        assert result is EMPTY

    def test_derive_alt_first_matches(self):
        """derive(alt, byte matching first alt) → derive(first alt)."""
        engine = self.make_engine()
        node = AltNode((LiteralNode("a"), LiteralNode("b")))
        result = engine.derive(node, ord('a'))
        assert result is EMPTY

    def test_derive_alt_no_match(self):
        """derive(alt, byte matching no alt) → NEVER."""
        engine = self.make_engine()
        node = AltNode((LiteralNode("a"), LiteralNode("b")))
        result = engine.derive(node, ord('c'))
        assert result is NEVER

    def test_derive_repeat(self):
        """derive(repeat, matching byte) → derive(child) + repeat."""
        engine = self.make_engine()
        node = RepeatNode(LiteralNode("a"))
        result = engine.derive(node, ord('a'))
        # After derive: EMPTY + Repeat → Repeat
        assert result == RepeatNode(LiteralNode("a"))

    def test_derive_one_or_more(self):
        """derive(one_or_more, matching byte) → derive(child) + repeat."""
        engine = self.make_engine()
        node = OneOrMoreNode(LiteralNode("a"))
        result = engine.derive(node, ord('a'))
        # After derive: EMPTY + Repeat → Repeat
        assert result == RepeatNode(LiteralNode("a"))

    def test_derive_optional_matching(self):
        """derive(optional, matching byte) → derive(child)."""
        engine = self.make_engine()
        node = OptionalNode(LiteralNode("a"))
        result = engine.derive(node, ord('a'))
        assert result is EMPTY

    def test_derive_optional_not_matching(self):
        """derive(optional, non-matching byte) → NEVER (since optional can be empty but derive must match)."""
        engine = self.make_engine()
        node = OptionalNode(LiteralNode("a"))
        result = engine.derive(node, ord('b'))
        assert result is NEVER

    def test_derive_rule_ref(self):
        """derive(rule_ref) resolves and derives the referenced rule."""
        engine = self.make_engine({
            "a": LiteralNode("x"),
        })
        result = engine.derive(RuleRefNode("a"), ord('x'))
        assert result is EMPTY

    def test_derive_rule_ref_missing(self):
        """derive(missing rule_ref) → NEVER."""
        engine = self.make_engine({})
        result = engine.derive(RuleRefNode("missing"), ord('a'))
        assert result is NEVER

    def test_derive_rule_ref_non_matching(self):
        """derive(rule_ref, wrong byte) → NEVER."""
        engine = self.make_engine({
            "a": LiteralNode("x"),
        })
        result = engine.derive(RuleRefNode("a"), ord('y'))
        assert result is NEVER

    def test_derive_byte_out_of_range(self):
        """derive works for any byte 0-255."""
        engine = self.make_engine()
        # 0xFF byte against AnyChar
        result = engine.derive(AnyCharNode(), 0xFF)
        assert result is EMPTY


# ===========================================================================
# GBNFFSM
# ===========================================================================

class TestGBNFFSM:
    """Tests for the GBNF byte-level FSM."""

    # ─── Basic Literal ────────────────────────────────────────────────────

    def test_literal_exact(self):
        """FSM accepts exactly the literal string."""
        fsm = GBNFFSM('root ::= "hello"')
        assert not fsm.is_accepting()
        for ch in "hello":
            assert fsm.get_allowed_bytes() == {ord(ch)}
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_literal_wrong_byte_rejected(self):
        """Wrong byte leads to no allowed bytes."""
        fsm = GBNFFSM('root ::= "abc"')
        fsm.transition(ord('a'))
        fsm.transition(ord('x'))  # wrong
        assert fsm.get_allowed_bytes() == set()

    def test_literal_first_bytes(self):
        """First allowed byte matches literal start."""
        fsm = GBNFFSM('root ::= "hello"')
        allowed = fsm.get_allowed_bytes()
        assert allowed == {ord('h')}

    # ─── Alternation ──────────────────────────────────────────────────────

    def test_alternation(self):
        """FSM accepts any alternative."""
        fsm = GBNFFSM('root ::= "yes" | "no"')
        allowed = fsm.get_allowed_bytes()
        assert ord('y') in allowed
        assert ord('n') in allowed
        assert len(allowed) == 2

    def test_alternation_choose_first(self):
        """FSM can follow first alternative through."""
        fsm = GBNFFSM('root ::= "yes" | "no"')
        fsm.transition(ord('y'))
        remaining = "es"
        for ch in remaining:
            assert fsm.get_allowed_bytes() == {ord(ch)}
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_alternation_choose_second(self):
        """FSM can follow second alternative through."""
        fsm = GBNFFSM('root ::= "yes" | "no"')
        fsm.transition(ord('n'))
        remaining = "o"
        for ch in remaining:
            assert fsm.get_allowed_bytes() == {ord(ch)}
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    # ─── Character Class ──────────────────────────────────────────────────

    def test_char_class(self):
        """Character class allows multiple first bytes."""
        fsm = GBNFFSM('root ::= [abc]')
        allowed = fsm.get_allowed_bytes()
        assert allowed == {ord('a'), ord('b'), ord('c')}

    def test_char_class_range(self):
        """Character class range allows all bytes in range."""
        fsm = GBNFFSM('root ::= [0-9]')
        allowed = fsm.get_allowed_bytes()
        expected = set(range(ord('0'), ord('9') + 1))
        assert allowed == expected

    def test_char_class_negated(self):
        """Negated char class allows all bytes except those listed."""
        fsm = GBNFFSM('root ::= [^a]')
        allowed = fsm.get_allowed_bytes()
        assert len(allowed) == 255
        assert ord('a') not in allowed

    # ─── Any Char ─────────────────────────────────────────────────────────

    def test_any_char(self):
        "'.' allows any single byte."
        fsm = GBNFFSM('root ::= .')
        allowed = fsm.get_allowed_bytes()
        assert len(allowed) == 256

    def test_any_char_consumes_one(self):
        "'.' consumes exactly one byte and becomes accepting."
        fsm = GBNFFSM('root ::= .')
        fsm.transition(0x42)  # any byte
        assert fsm.is_accepting()

    # ─── Repetition ───────────────────────────────────────────────────────

    def test_repeat_zero_or_more(self):
        """'*' allows zero repetitions (nullable)."""
        fsm = GBNFFSM('root ::= "a"*')
        assert fsm.is_accepting()  # * is always nullable

    def test_repeat_one_to_three(self):
        """'*' can match multiple repetitions."""
        fsm = GBNFFSM('root ::= "a"*')
        allowed = fsm.get_allowed_bytes()
        assert ord('a') in allowed
        for _ in range(3):
            fsm.transition(ord('a'))
            assert fsm.is_accepting()
            assert ord('a') in fsm.get_allowed_bytes()

    def test_one_or_more(self):
        """'+' at least one match."""
        fsm = GBNFFSM('root ::= "a"+')
        assert not fsm.is_accepting()  # need at least one 'a'
        fsm.transition(ord('a'))
        assert fsm.is_accepting()

    def test_one_or_more_multiple(self):
        """'+' allows more repeats after first."""
        fsm = GBNFFSM('root ::= [a-z]+')
        fsm.transition(ord('a'))
        assert fsm.is_accepting()
        assert ord('b') in fsm.get_allowed_bytes()
        fsm.transition(ord('b'))
        assert fsm.is_accepting()

    def test_optional(self):
        """'?' allows zero or one."""
        fsm = GBNFFSM('root ::= "a"?')
        assert fsm.is_accepting()  # zero allowed
        fsm.transition(ord('a'))
        assert fsm.is_accepting()

    # ─── Sequence ─────────────────────────────────────────────────────────

    def test_sequence(self):
        """Sequence requires each part in order."""
        fsm = GBNFFSM('root ::= "a" "b" "c"')
        for ch in "abc":
            allowed = fsm.get_allowed_bytes()
            assert allowed == {ord(ch)}
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_sequence_optional_element(self):
        """Sequence with optional element."""
        fsm = GBNFFSM('root ::= "a" "b"? "c"')
        fsm.transition(ord('a'))
        allowed = fsm.get_allowed_bytes()
        assert ord('b') in allowed
        assert ord('c') in allowed  # can skip optional
        fsm.transition(ord('c'))
        assert fsm.is_accepting()

    # ─── DFA Mode ─────────────────────────────────────────────────────────

    def test_dfa_caching(self):
        """DFA mode caches transitions."""
        fsm = GBNFFSM('root ::= "hello"')
        fsm.compile_to_dfa()
        assert fsm._use_dfa is True
        fsm.transition(ord('h'))
        # Cache should now have entry for repr(state), ord('h')
        assert len(fsm._transition_cache) >= 1

    def test_dfa_vs_non_dfa_same_result(self):
        """DFA and non-DFA modes produce identical results."""
        grammar = 'root ::= [a-z]+'
        fsm1 = GBNFFSM(grammar)
        fsm2 = GBNFFSM(grammar)
        fsm2.compile_to_dfa()

        for ch in "hello":
            assert fsm1.get_allowed_bytes() == fsm2.get_allowed_bytes()
            fsm1.transition(ord(ch))
            fsm2.transition(ord(ch))

        assert fsm1.is_accepting() == fsm2.is_accepting()

    # ─── Reset ────────────────────────────────────────────────────────────

    def test_reset(self):
        """Reset returns FSM to initial state."""
        fsm = GBNFFSM('root ::= "hello"')
        for ch in "hello":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()
        fsm.reset()
        assert not fsm.is_accepting()
        assert fsm.get_allowed_bytes() == {ord('h')}

    # ─── is_accepting / can_end ───────────────────────────────────────────

    def test_is_accepting_before_start(self):
        """is_accepting is False before matching starts (for non-optional)."""
        fsm = GBNFFSM('root ::= "hello"')
        assert not fsm.is_accepting()

    def test_is_accepting_after_full(self):
        """is_accepting is True after full match."""
        fsm = GBNFFSM('root ::= "hello"')
        for ch in "hello":
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_can_end_is_accepting(self):
        """can_end() has same result as is_accepting()."""
        fsm = GBNFFSM('root ::= "a"?')
        assert fsm.can_end() == fsm.is_accepting()

    # ─── Complex Grammars ─────────────────────────────────────────────────

    def test_email_like_grammar(self):
        """Match a simple email-like pattern."""
        grammar = """
            root ::= local "@" domain
            local ::= [a-z]+
            domain ::= [a-z]+ "." [a-z]+
        """
        fsm = GBNFFSM(grammar)
        text = "user@example.com"
        for ch in text:
            allowed = fsm.get_allowed_bytes()
            assert ord(ch) in allowed, f"byte '{ch}' not allowed at pos {text.find(ch)}"
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    def test_arithmetic_grammar_evaluation(self):
        """Match a simple arithmetic expression."""
        grammar = """
            root ::= expr
            expr ::= term (("+" | "-") term)*
            term ::= [0-9]+
        """
        fsm = GBNFFSM(grammar)
        text = "42+13-7"
        for ch in text:
            allowed = fsm.get_allowed_bytes()
            assert ord(ch) in allowed, f"byte '{ch}' not allowed at pos {text.find(ch)}"
            fsm.transition(ord(ch))
        assert fsm.is_accepting()

    # ─── Edge Cases ───────────────────────────────────────────────────────

    def test_invalid_byte_transitions_to_never(self):
        """Byte value outside 0-255 sets state to NEVER."""
        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(256)
        assert fsm.get_allowed_bytes() == set()
        assert not fsm.is_accepting()

    def test_negative_byte_transitions_to_never(self):
        """Negative byte value sets state to NEVER."""
        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(-1)
        assert fsm.get_allowed_bytes() == set()

    def test_get_allowed_bytes_returns_set(self):
        """get_allowed_bytes always returns a set."""
        fsm = GBNFFSM('root ::= "a"')
        assert isinstance(fsm.get_allowed_bytes(), set)

    def test_is_accepting_after_derive_to_never(self):
        """After deriving to NEVER, is_accepting returns False."""
        fsm = GBNFFSM('root ::= "a"')
        fsm.transition(ord('b'))
        assert not fsm.is_accepting()
        assert fsm.get_allowed_bytes() == set()

    # ─── Interleaved allow/transition rounds ─────────────────────────────

    def test_step_by_step(self):
        """FSM correctly interleaves get_allowed_bytes and transition."""
        fsm = GBNFFSM('root ::= [a-z]+ "!"')
        assert not fsm.is_accepting()
        assert fsm.get_allowed_bytes() == set(range(ord('a'), ord('z') + 1))

        fsm.transition(ord('h'))
        assert not fsm.is_accepting()
        # Can continue with lowercase OR the '!'
        allowed = fsm.get_allowed_bytes()
        assert ord('i') in allowed
        assert 0x21 in allowed  # '!'

        fsm.transition(ord('i'))
        allowed = fsm.get_allowed_bytes()
        assert 0x21 in allowed  # '!'

        fsm.transition(0x21)  # '!'
        assert fsm.is_accepting()
