"""GBNF (GGML BNF) grammar parser and NFA-based position tracker.

GBNF is the grammar format used by llama.cpp for structured generation.
Uses a position-set approach (like regex NFA simulation) to track all
possible locations within the grammar after each byte.

Grammar constructs:
  "literal"    - Match exact string
  [charclass]  - Match character class
  rule_name    - Reference another rule
  (group)      - Grouping
  expr1 | expr2 - Alternation
  expr*        - Zero or more
  expr+        - One or more
  expr?        - Optional (zero or one)
  .            - Any single byte
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# AST Node Types
# ---------------------------------------------------------------------------

class GBNFNode:
    pass


@dataclass
class LiteralNode(GBNFNode):
    value: str


@dataclass
class CharClassNode(GBNFNode):
    chars: set[int]


class AnyCharNode(GBNFNode):
    pass


@dataclass
class RuleRefNode(GBNFNode):
    name: str


@dataclass
class SeqNode(GBNFNode):
    children: list[GBNFNode] = field(default_factory=list)


@dataclass
class AltNode(GBNFNode):
    alternatives: list[GBNFNode] = field(default_factory=list)


@dataclass
class RepeatNode(GBNFNode):
    child: GBNFNode | None = None


@dataclass
class OneOrMoreNode(GBNFNode):
    child: GBNFNode | None = None


@dataclass
class OptionalNode(GBNFNode):
    child: GBNFNode | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GBNFParser:
    """Parse GBNF grammar string into AST."""

    def __init__(self, grammar_text: str):
        self._text = grammar_text
        self._rules: dict[str, GBNFNode] = {}

    def parse(self) -> dict[str, GBNFNode]:
        current_name: str | None = None
        current_expr: list[str] = []
        for line in self._text.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if '::=' in stripped:
                if current_name is not None and current_expr:
                    self._rules[current_name] = self._parse_expr('\n'.join(current_expr))
                parts = stripped.split('::=', 1)
                current_name = parts[0].strip()
                current_expr = [parts[1].strip()]
            else:
                if current_name is not None:
                    current_expr.append(stripped)
        if current_name is not None and current_expr:
            self._rules[current_name] = self._parse_expr('\n'.join(current_expr))
        return self._rules

    def _parse_expr(self, text: str) -> GBNFNode:
        return _ExprParser(text).parse_expr()


class _ExprParser:
    def __init__(self, text: str):
        self._t = text
        self._p = 0

    def parse_expr(self) -> GBNFNode:
        alts = [self._parse_seq()]
        while self._peek() == '|':
            self._adv()
            self._skip()
            alts.append(self._parse_seq())
        return alts[0] if len(alts) == 1 else AltNode(alts)

    def _parse_seq(self) -> GBNFNode:
        kids: list[GBNFNode] = []
        while self._p < len(self._t):
            self._skip()
            ch = self._peek()
            if ch is None or ch in ')|\n\r':
                break
            atom = self._parse_atom()
            if atom is None:
                break
            atom = self._suffix(atom)
            kids.append(atom)
        return kids[0] if len(kids) == 1 else SeqNode(kids)

    def _parse_atom(self) -> GBNFNode | None:
        self._skip()
        ch = self._peek()
        if ch is None or ch in ')|\n\r':
            return None
        if ch == '"':
            return self._lit()
        if ch == '[':
            return self._cc()
        if ch == '(':
            self._adv()
            node = self.parse_expr()
            self._skip()
            if self._peek() == ')':
                self._adv()
            return node
        if ch == '.':
            self._adv()
            return AnyCharNode()
        if ch.isalpha() or ch == '_':
            return RuleRefNode(self._name())
        return None

    def _suffix(self, node: GBNFNode) -> GBNFNode:
        ch = self._peek()
        if ch == '*':
            self._adv()
            return RepeatNode(node)
        if ch == '+':
            self._adv()
            return OneOrMoreNode(node)
        if ch == '?':
            self._adv()
            return OptionalNode(node)
        return node

    def _lit(self) -> LiteralNode:
        self._adv()
        start = self._p
        while self._p < len(self._t) and self._t[self._p] != '"':
            if self._t[self._p] == '\\':
                self._p += 1
            self._p += 1
        val = self._t[start:self._p]
        if self._p < len(self._t):
            self._adv()
        return LiteralNode(val)

    def _cc(self) -> CharClassNode:
        self._adv()
        chars: set[int] = set()
        while self._p < len(self._t) and self._t[self._p] != ']':
            if (self._p + 2 < len(self._t)
                    and self._t[self._p + 1] == '-'
                    and self._t[self._p + 2] != ']'):
                for c in range(ord(self._t[self._p]), ord(self._t[self._p + 2]) + 1):
                    chars.add(c)
                self._p += 3
            else:
                chars.add(ord(self._t[self._p]))
                self._p += 1
        if self._p < len(self._t):
            self._adv()
        return CharClassNode(chars)

    def _name(self) -> str:
        start = self._p
        while self._p < len(self._t) and (self._t[self._p].isalnum() or self._t[self._p] == '_'):
            self._p += 1
        return self._t[start:self._p]

    def _skip(self) -> None:
        while self._p < len(self._t) and self._t[self._p] in ' \t\r\n':
            self._p += 1

    def _peek(self) -> str | None:
        return None if self._p >= len(self._t) else self._t[self._p]

    def _adv(self) -> None:
        if self._p < len(self._t):
            self._p += 1


# ---------------------------------------------------------------------------
# NFA Position-Set Tracker
# ---------------------------------------------------------------------------

class Position:
    """A position in the grammar: which node and how far into its children."""
    def __init__(self, node: GBNFNode, child_idx: int = 0, literal_pos: int = 0):
        self.node = node
        self.child_idx = child_idx
        self.literal_pos = literal_pos

    def copy(self) -> 'Position':
        return Position(self.node, self.child_idx, self.literal_pos)

    def __hash__(self) -> int:
        return id(self.node) ^ (self.child_idx << 16) ^ self.literal_pos

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return False
        return (self.node is other.node
                and self.child_idx == other.child_idx
                and self.literal_pos == other.literal_pos)


class GBNFFSM:
    """NFA-based grammar FSM using position-set simulation.

    Tracks a SET of current positions within the grammar. At each byte,
    all positions attempt to advance. If any position reaches acceptance,
    the FSM is accepting.
    """

    def __init__(self, grammar_text: str, start_rule: str = "root"):
        parser = GBNFParser(grammar_text)
        self._rules = parser.parse()
        if start_rule not in self._rules:
            if self._rules:
                start_rule = next(iter(self._rules))
            else:
                raise ValueError("No rules in grammar")
        self._start_node = self._rules[start_rule]
        self._positions: set[Position] = set()
        self.reset()

    def reset(self) -> None:
        self._positions = set()
        if isinstance(self._start_node, SeqNode):
            self._positions.add(Position(self._start_node, 0))
        else:
            wrapper = SeqNode(children=[self._start_node])
            self._positions.add(Position(wrapper, 0))
            self._start_node = wrapper
        self._expand_empty()

    def get_allowed_bytes(self) -> set[int]:
        result: set[int] = set()
        for pos in list(self._positions):
            if pos.child_idx >= self._child_count(pos.node):
                continue
            child = self._get_child(pos)
            self._add_first_bytes(child, result)
        return result

    def transition(self, byte_val: int) -> None:
        next_positions: set[Position] = set()
        for pos in self._positions:
            if pos.child_idx >= self._child_count(pos.node):
                continue
            child = self._get_child(pos)
            new_positions = self._advance_node(child, byte_val)
            for np in new_positions:
                if np is None:
                    # Child fully consumed, advance sequence
                    advanced = Position(pos.node, pos.child_idx + 1)
                    next_positions.add(advanced)
                else:
                    next_positions.add(np)
        self._positions = next_positions
        self._expand_empty()

    def _expand_empty(self) -> None:
        """Expand positions that can match empty (Optionals, Repeats with 0)."""
        changed = True
        while changed:
            changed = False
            for pos in list(self._positions):
                if pos.child_idx >= self._child_count(pos.node):
                    continue
                child = self._get_child(pos)
                empty_next = self._get_empty_transitions(child)
                for ep in empty_next:
                    if isinstance(pos.node, SeqNode):
                        new_pos = Position(pos.node, pos.child_idx + 1)
                    else:
                        new_pos = Position(pos.node, pos.child_idx + 1)
                    if new_pos not in self._positions:
                        self._positions.add(new_pos)
                        changed = True

    def _get_child(self, pos: Position) -> GBNFNode | None:
        if isinstance(pos.node, SeqNode):
            if pos.child_idx < len(pos.node.children):
                return pos.node.children[pos.child_idx]
        return pos.node

    def _child_count(self, node: GBNFNode) -> int:
        if isinstance(node, SeqNode):
            return len(node.children)
        return 1

    def _add_first_bytes(self, node: GBNFNode | None, result: set[int],
                         visited: set[str] | None = None) -> None:
        if node is None:
            return
        if visited is None:
            visited = set()

        if isinstance(node, LiteralNode):
            if node.value:
                result.add(ord(node.value[0]))
        elif isinstance(node, CharClassNode):
            result.update(node.chars)
        elif isinstance(node, AnyCharNode):
            result.update(range(256))
        elif isinstance(node, RuleRefNode):
            if node.name not in visited and node.name in self._rules:
                visited.add(node.name)
                self._add_first_bytes(self._rules[node.name], result, visited)
        elif isinstance(node, SeqNode):
            for child in node.children:
                before = len(result)
                self._add_first_bytes(child, result, visited)
                if len(result) > before:
                    break
        elif isinstance(node, AltNode):
            for alt in node.alternatives:
                self._add_first_bytes(alt, result, visited)
        elif isinstance(node, (RepeatNode, OneOrMoreNode, OptionalNode)):
            if node.child is not None:
                self._add_first_bytes(node.child, result, visited)

    def _advance_node(self, node: GBNFNode | None, byte_val: int) -> list[Position | None]:
        if node is None:
            return [None]

        if isinstance(node, LiteralNode):
            if node.value and ord(node.value[0]) == byte_val:
                remaining = LiteralNode(node.value[1:]) if len(node.value) > 1 else None
                if remaining:
                    return [Position(remaining)]
                return [None]
            return []

        if isinstance(node, CharClassNode):
            if byte_val in node.chars:
                return [None]
            return []

        if isinstance(node, AnyCharNode):
            return [None]

        if isinstance(node, RuleRefNode):
            if node.name in self._rules:
                return self._advance_node(self._rules[node.name], byte_val)
            return []

        if isinstance(node, SeqNode):
            if not node.children:
                return [None]
            child = node.children[0]
            results = self._advance_node(child, byte_val)
            final: list[Position | None] = []
            for r in results:
                if r is None:
                    advanced = Position(node, 1)
                    final.append(advanced)
                else:
                    advanced = Position(node, 0)
                    final.append(advanced)
            return final

        if isinstance(node, AltNode):
            for alt in node.alternatives:
                results = self._advance_node(alt, byte_val)
                if results:
                    return results
            return []

        if isinstance(node, RepeatNode):
            if node.child is not None:
                results = self._advance_node(node.child, byte_val)
                if results:
                    return [Position(node)]  # stay in repeat
            return []  # zero repetitions -> no match

        if isinstance(node, OneOrMoreNode):
            if node.child is not None:
                results = self._advance_node(node.child, byte_val)
                if results:
                    return [Position(node)]  # stay in one-or-more
            return []

        if isinstance(node, OptionalNode):
            if node.child is not None:
                results = self._advance_node(node.child, byte_val)
                if results:
                    return results
            return [None]

        return []

    def _get_empty_transitions(self, node: GBNFNode | None) -> list[bool]:
        if node is None:
            return [True]
        if isinstance(node, OptionalNode):
            return [True]
        if isinstance(node, RepeatNode):
            return [True]
        if isinstance(node, AltNode):
            for alt in node.alternatives:
                if self._get_empty_transitions(alt):
                    return [True]
            return []
        if isinstance(node, SeqNode):
            if all(self._get_empty_transitions(c) for c in node.children):
                return [True]
            return []
        return []

    def is_accepting(self) -> bool:
        for pos in self._positions:
            if isinstance(pos.node, SeqNode):
                if pos.child_idx >= len(pos.node.children):
                    return True
            elif pos.child_idx >= 1:
                return True
        return False

    def can_end(self) -> bool:
        return self.is_accepting()
