"""GBNF grammar parser and byte-level FSM for constrained decoding.

GBNF is the GGML/llama.cpp grammar format used for structured generation.
This module parses a practical subset of GBNF and runs it as a byte-level
finite-state machine using Brzozowski derivatives: after each emitted byte,
the current state becomes the residual grammar that remains to be matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class GBNFNode:
    """Base class for grammar AST nodes."""


@dataclass(frozen=True)
class EmptyNode(GBNFNode):
    """Matches the empty string."""


@dataclass(frozen=True)
class NeverNode(GBNFNode):
    """Matches no strings."""


@dataclass(frozen=True)
class LiteralNode(GBNFNode):
    value: str


@dataclass(frozen=True)
class CharClassNode(GBNFNode):
    chars: frozenset[int]


@dataclass(frozen=True)
class AnyCharNode(GBNFNode):
    pass


@dataclass(frozen=True)
class RuleRefNode(GBNFNode):
    name: str


@dataclass(frozen=True)
class SeqNode(GBNFNode):
    children: tuple[GBNFNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AltNode(GBNFNode):
    alternatives: tuple[GBNFNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RepeatNode(GBNFNode):
    child: GBNFNode


@dataclass(frozen=True)
class OneOrMoreNode(GBNFNode):
    child: GBNFNode


@dataclass(frozen=True)
class OptionalNode(GBNFNode):
    child: GBNFNode


EMPTY = EmptyNode()
NEVER = NeverNode()


class GBNFParser:
    """Parse GBNF grammar text into AST nodes."""

    def __init__(self, grammar_text: str):
        self._text = grammar_text
        self._rules: dict[str, GBNFNode] = {}

    def parse(self) -> dict[str, GBNFNode]:
        current_name: str | None = None
        current_expr: list[str] = []

        for raw_line in self._text.splitlines():
            stripped = self._strip_comment(raw_line).strip()
            if not stripped:
                continue

            if "::=" in stripped:
                if current_name is not None:
                    self._rules[current_name] = self._parse_expr(" ".join(current_expr))
                current_name, expr = stripped.split("::=", 1)
                current_name = current_name.strip()
                current_expr = [expr.strip()]
            elif current_name is not None:
                current_expr.append(stripped)

        if current_name is not None:
            self._rules[current_name] = self._parse_expr(" ".join(current_expr))

        return self._rules

    def _parse_expr(self, text: str) -> GBNFNode:
        return _ExprParser(text).parse_expr()

    @staticmethod
    def _strip_comment(line: str) -> str:
        in_string = False
        bracket_depth = 0
        escaped = False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
            if not in_string:
                if ch == '[':
                    bracket_depth += 1
                elif ch == ']':
                    bracket_depth -= 1
            if ch == "#" and not in_string and bracket_depth == 0:
                return line[:i]
        return line


class _ExprParser:
    def __init__(self, text: str):
        self._t = text
        self._p = 0

    def parse_expr(self) -> GBNFNode:
        alts = [self._parse_seq()]
        while True:
            self._skip()
            if self._peek() != "|":
                break
            self._adv()
            alts.append(self._parse_seq())
        return _simplify_alt(alts)

    def _parse_seq(self) -> GBNFNode:
        children: list[GBNFNode] = []
        while self._p < len(self._t):
            self._skip()
            ch = self._peek()
            if ch is None or ch in ")|":
                break
            atom = self._parse_atom()
            if atom is None:
                raise ValueError(f"Unexpected grammar token at offset {self._p}: {self._t[self._p:self._p + 16]!r}")
            children.append(self._suffix(atom))
        return _simplify_seq(children)

    def _parse_atom(self) -> GBNFNode | None:
        self._skip()
        ch = self._peek()
        if ch is None or ch in ")|":
            return None
        if ch == '"':
            return self._literal()
        if ch == "[":
            return self._char_class()
        if ch == "(":
            self._adv()
            node = self.parse_expr()
            self._skip()
            if self._peek() != ")":
                raise ValueError("Unclosed grammar group")
            self._adv()
            return node
        if ch == ".":
            self._adv()
            return AnyCharNode()
        if ch.isalpha() or ch == "_":
            return RuleRefNode(self._name())
        return None

    def _suffix(self, node: GBNFNode) -> GBNFNode:
        self._skip()
        ch = self._peek()
        if ch == "*":
            self._adv()
            return RepeatNode(node)
        if ch == "+":
            self._adv()
            return OneOrMoreNode(node)
        if ch == "?":
            self._adv()
            return OptionalNode(node)
        return node

    def _literal(self) -> GBNFNode:
        self._adv()
        chars: list[str] = []
        while self._p < len(self._t):
            ch = self._t[self._p]
            self._adv()
            if ch == '"':
                literal_str = "".join(chars)
                return self._to_byte_literal(literal_str)
            if ch == "\\" and self._p < len(self._t):
                chars.append(self._decode_escape())
            else:
                chars.append(ch)
        raise ValueError("Unclosed grammar literal")

    @staticmethod
    def _to_byte_literal(text: str) -> GBNFNode:
        """Convert a string literal to a byte-level sequence.

        ASCII characters map directly. Non-ASCII multi-byte characters
        are expanded to their UTF-8 byte sequences so the derivative
        engine can match individual bytes (0-255).
        """
        if all(ord(c) < 128 for c in text):
            return LiteralNode(text)
        seq_nodes = []
        for c in text:
            encoded = c.encode('utf-8')
            seq_nodes.extend(LiteralNode(chr(b)) for b in encoded)
        if len(seq_nodes) == 1:
            return seq_nodes[0]
        return SeqNode(tuple(seq_nodes))

    def _decode_escape(self) -> str:
        single = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            '"': '"',
            "\\": "\\",
        }
        ch = self._peek()
        if ch is None:
            return "\\"
        self._adv()
        if ch in single:
            return single[ch]
        if ch == "x":
            return self._decode_hex_escape()
        if ch == "u":
            return self._decode_unicode_escape(4)
        if ch == "U":
            return self._decode_unicode_escape(8)
        return ch

    def _decode_hex_escape(self) -> str:
        hex_str = ""
        for _ in range(2):
            h = self._peek()
            if h is None or h not in "0123456789abcdefABCDEF":
                break
            hex_str += h
            self._adv()
        if len(hex_str) == 2:
            return chr(int(hex_str, 16))
        return "\\x" + hex_str

    def _decode_unicode_escape(self, digits: int) -> str:
        hex_str = ""
        for _ in range(digits):
            h = self._peek()
            if h is None or h not in "0123456789abcdefABCDEF":
                break
            hex_str += h
            self._adv()
        if len(hex_str) == digits:
            return chr(int(hex_str, 16))
        return "\\u" + hex_str if digits == 4 else "\\U" + hex_str

    def _char_class(self) -> GBNFNode:
        self._adv()
        single_bytes: set[int] = set()
        multi_byte_seqs: list[GBNFNode] = []
        negate = False
        if self._peek() == "^":
            negate = True
            self._adv()

        while self._p < len(self._t) and self._peek() != "]":
            ch = self._read_class_char()
            if self._peek() == "-" and self._p + 1 < len(self._t) and self._t[self._p + 1] != "]":
                self._adv()
                end_ch = self._read_class_char()
                lo, hi = sorted((ord(ch), ord(end_ch)))
                for cp in range(lo, hi + 1):
                    encoded = chr(cp).encode('utf-8')
                    if len(encoded) == 1:
                        single_bytes.add(encoded[0])
                    else:
                        seq_nodes = [LiteralNode(chr(b)) for b in encoded]
                        multi_byte_seqs.append(SeqNode(tuple(seq_nodes)))
            else:
                encoded = ch.encode('utf-8')
                if len(encoded) == 1:
                    single_bytes.add(encoded[0])
                else:
                    seq_nodes = [LiteralNode(chr(b)) for b in encoded]
                    multi_byte_seqs.append(SeqNode(tuple(seq_nodes)))

        if self._peek() != "]":
            raise ValueError("Unclosed grammar character class")
        self._adv()

        alternatives: list[GBNFNode] = []
        if single_bytes:
            if negate:
                single_bytes = set(range(256)) - single_bytes
            alternatives.append(CharClassNode(frozenset(single_bytes)))
        if multi_byte_seqs:
            if negate:
                alternatives.append(AnyCharNode())
            else:
                alternatives.extend(multi_byte_seqs)

        if not alternatives:
            return CharClassNode(frozenset())
        if len(alternatives) == 1:
            return alternatives[0]
        return AltNode(tuple(alternatives))

    def _read_class_char(self) -> str:
        ch = self._peek()
        if ch is None:
            raise ValueError("Unexpected end of character class")
        self._adv()
        if ch == "\\":
            ch = self._decode_escape()
        return ch

    def _name(self) -> str:
        start = self._p
        while self._p < len(self._t):
            ch = self._t[self._p]
            if ch.isalnum() or ch in "_-":
                self._p += 1
            else:
                break
        return self._t[start:self._p]

    def _skip(self) -> None:
        while self._p < len(self._t) and self._t[self._p] in " \t\r\n":
            self._p += 1

    def _peek(self) -> str | None:
        return None if self._p >= len(self._t) else self._t[self._p]

    def _adv(self) -> None:
        self._p += 1


def _simplify_seq(children: list[GBNFNode] | tuple[GBNFNode, ...]) -> GBNFNode:
    flattened: list[GBNFNode] = []
    for child in children:
        child = _simplify(child)
        if isinstance(child, NeverNode):
            return NEVER
        if isinstance(child, EmptyNode):
            continue
        if isinstance(child, SeqNode):
            flattened.extend(child.children)
        else:
            flattened.append(child)
    if not flattened:
        return EMPTY
    if len(flattened) == 1:
        return flattened[0]
    return SeqNode(tuple(flattened))


def _simplify_alt(alternatives: list[GBNFNode] | tuple[GBNFNode, ...]) -> GBNFNode:
    flattened: list[GBNFNode] = []
    seen: set[str] = set()
    for alt in alternatives:
        alt = _simplify(alt)
        if isinstance(alt, NeverNode):
            continue
        if isinstance(alt, AltNode):
            candidates = alt.alternatives
        else:
            candidates = (alt,)
        for candidate in candidates:
            key = repr(candidate)
            if key not in seen:
                seen.add(key)
                flattened.append(candidate)
    if not flattened:
        return NEVER
    if len(flattened) == 1:
        return flattened[0]
    return AltNode(tuple(flattened))


def _simplify(node: GBNFNode) -> GBNFNode:
    if isinstance(node, LiteralNode) and node.value == "":
        return EMPTY
    if isinstance(node, SeqNode):
        return _simplify_seq(node.children)
    if isinstance(node, AltNode):
        return _simplify_alt(node.alternatives)
    if isinstance(node, RepeatNode):
        child = _simplify(node.child)
        if isinstance(child, (EmptyNode, NeverNode)):
            return EMPTY
        return RepeatNode(child)
    if isinstance(node, OneOrMoreNode):
        child = _simplify(node.child)
        if isinstance(child, EmptyNode):
            return EMPTY
        if isinstance(child, NeverNode):
            return NEVER
        return OneOrMoreNode(child)
    if isinstance(node, OptionalNode):
        child = _simplify(node.child)
        if isinstance(child, NeverNode):
            return EMPTY
        return OptionalNode(child)
    return node


class _DerivativeEngine:
    def __init__(self, rules: dict[str, GBNFNode]):
        self.rules = rules

    def nullable(self, node: GBNFNode, visiting: set[str] | None = None) -> bool:
        node = _simplify(node)
        if isinstance(node, EmptyNode):
            return True
        if isinstance(node, (NeverNode, LiteralNode, CharClassNode, AnyCharNode)):
            return False
        if isinstance(node, SeqNode):
            return all(self.nullable(child, visiting) for child in node.children)
        if isinstance(node, AltNode):
            return any(self.nullable(alt, visiting) for alt in node.alternatives)
        if isinstance(node, RepeatNode):
            return True
        if isinstance(node, OneOrMoreNode):
            return self.nullable(node.child, visiting)
        if isinstance(node, OptionalNode):
            return True
        if isinstance(node, RuleRefNode):
            if visiting is None:
                visiting = set()
            if node.name in visiting or node.name not in self.rules:
                return False
            visiting.add(node.name)
            return self.nullable(self.rules[node.name], visiting)
        return False

    def first_bytes(self, node: GBNFNode, visiting: set[str] | None = None) -> set[int]:
        node = _simplify(node)
        if isinstance(node, (EmptyNode, NeverNode)):
            return set()
        if isinstance(node, LiteralNode):
            return {ord(node.value[0])} if node.value else set()
        if isinstance(node, CharClassNode):
            return set(node.chars)
        if isinstance(node, AnyCharNode):
            return set(range(256))
        if isinstance(node, AltNode):
            allowed: set[int] = set()
            for alt in node.alternatives:
                allowed.update(self.first_bytes(alt, visiting))
            return allowed
        if isinstance(node, SeqNode):
            allowed: set[int] = set()
            for child in node.children:
                allowed.update(self.first_bytes(child, visiting))
                if not self.nullable(child):
                    break
            return allowed
        if isinstance(node, (RepeatNode, OneOrMoreNode, OptionalNode)):
            return self.first_bytes(node.child, visiting)
        if isinstance(node, RuleRefNode):
            if visiting is None:
                visiting = set()
            if node.name in visiting or node.name not in self.rules:
                return set()
            visiting.add(node.name)
            return self.first_bytes(self.rules[node.name], visiting)
        return set()

    def derive(self, node: GBNFNode, byte_val: int) -> GBNFNode:
        node = _simplify(node)
        if isinstance(node, (EmptyNode, NeverNode)):
            return NEVER
        if isinstance(node, LiteralNode):
            if node.value and ord(node.value[0]) == byte_val:
                return LiteralNode(node.value[1:]) if len(node.value) > 1 else EMPTY
            return NEVER
        if isinstance(node, CharClassNode):
            return EMPTY if byte_val in node.chars else NEVER
        if isinstance(node, AnyCharNode):
            return EMPTY
        if isinstance(node, RuleRefNode):
            target = self.rules.get(node.name)
            return self.derive(target, byte_val) if target is not None else NEVER
        if isinstance(node, AltNode):
            return _simplify_alt([self.derive(alt, byte_val) for alt in node.alternatives])
        if isinstance(node, SeqNode):
            if not node.children:
                return NEVER
            first, rest = node.children[0], list(node.children[1:])
            alternatives = [_simplify_seq([self.derive(first, byte_val), *rest])]
            if self.nullable(first):
                alternatives.append(self.derive(_simplify_seq(rest), byte_val))
            return _simplify_alt(alternatives)
        if isinstance(node, RepeatNode):
            return _simplify_seq([self.derive(node.child, byte_val), RepeatNode(node.child)])
        if isinstance(node, OneOrMoreNode):
            return _simplify_seq([self.derive(node.child, byte_val), RepeatNode(node.child)])
        if isinstance(node, OptionalNode):
            return self.derive(node.child, byte_val)
        return NEVER


class GBNFFSM:
    """Byte-level FSM for a parsed GBNF grammar."""

    def __init__(self, grammar_text: str, start_rule: str = "root"):
        self._rules = GBNFParser(grammar_text).parse()
        if not self._rules:
            raise ValueError("No rules in grammar")
        if start_rule not in self._rules:
            start_rule = next(iter(self._rules))
        self._start_rule = start_rule
        self._engine = _DerivativeEngine(self._rules)
        self._transition_cache: dict[tuple[str, int], GBNFNode] = {}
        self._use_dfa = False
        self.reset()

    def reset(self) -> None:
        self._state = _simplify(self._rules[self._start_rule])

    def compile_to_dfa(self) -> None:
        """Enable cached deterministic transitions.

        The DFA is built lazily to avoid eagerly exploring grammars with loops
        or large character classes. Transitions are still deterministic and are
        memoized per residual grammar state.
        """
        self._use_dfa = True

    def get_allowed_bytes(self) -> set[int]:
        return self._engine.first_bytes(self._state)

    def transition(self, byte_val: int) -> None:
        if byte_val < 0 or byte_val > 255:
            self._state = NEVER
            return

        if self._use_dfa:
            key = (repr(self._state), byte_val)
            next_state = self._transition_cache.get(key)
            if next_state is None:
                next_state = _simplify(self._engine.derive(self._state, byte_val))
                self._transition_cache[key] = next_state
            self._state = next_state
        else:
            self._state = _simplify(self._engine.derive(self._state, byte_val))

    def is_accepting(self) -> bool:
        return self._engine.nullable(self._state)

    def can_end(self) -> bool:
        return self.is_accepting()
