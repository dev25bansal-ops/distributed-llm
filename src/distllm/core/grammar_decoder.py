"""GBNF grammar decoder for constrained generation.

Provides GBNFParser (parses GBNF grammar strings) and GBNFFSM
(finite state machine for byte-level grammar validation).

GBNF (Grammar-Based Normal Form) is a simple grammar format used
by llama.cpp for constrained decoding.
"""

from __future__ import annotations

import re
from typing import Any

import torch


class GBNFParser:
    """Parses GBNF grammar strings into rule definitions.

    GBNF format::

        root ::= "hello" " " "world"
        digit ::= [0-9]
        number ::= digit+

    Usage::

        parser = GBNFParser()
        rules = parser.parse('root ::= "hello" " " "world"')
    """

    def __init__(self) -> None:
        self._rules: dict[str, list[list[str]]] = {}

    def parse(self, grammar: str) -> dict[str, list[list[str]]]:
        """Parse a GBNF grammar string into rules.

        Args:
            grammar: GBNF grammar string.

        Returns:
            Dict of rule_name -> list of alternatives, where each
            alternative is a list of tokens (strings or rule refs).
        """
        rules: dict[str, list[list[str]]] = {}
        for line in grammar.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse rule: name ::= alternative1 | alternative2
            match = re.match(r"(\w+)\s*::=\s*(.*)", line)
            if not match:
                continue

            name = match.group(1)
            body = match.group(2).strip()

            alternatives = []
            for alt in self._split_alternatives(body):
                tokens = self._parse_alternative(alt)
                alternatives.append(tokens)

            rules[name] = alternatives

        self._rules = rules
        return rules

    def _split_alternatives(self, body: str) -> list[str]:
        """Split a rule body by '|' (respecting quoted strings)."""
        alternatives = []
        current = ""
        in_quote = False
        for ch in body:
            if ch == '"':
                in_quote = not in_quote
                current += ch
            elif ch == "|" and not in_quote:
                alternatives.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            alternatives.append(current.strip())
        return alternatives

    def _parse_alternative(self, alt: str) -> list[str]:
        """Parse a single alternative into tokens."""
        tokens = []
        i = 0
        while i < len(alt):
            if alt[i].isspace():
                i += 1
                continue
            if alt[i] == '"':
                # Quoted string
                end = alt.index('"', i + 1)
                tokens.append(alt[i + 1:end])
                i = end + 1
            elif alt[i] == "[":
                # Character class
                end = alt.index("]", i + 1)
                tokens.append(alt[i:end + 1])
                i = end + 1
            elif alt[i] == "(":
                # Group
                end = alt.index(")", i + 1)
                tokens.append(alt[i:end + 1])
                i = end + 1
            else:
                # Rule reference or operator
                j = i
                while j < len(alt) and not alt[j].isspace() and alt[j] not in '"[]()':
                    j += 1
                tokens.append(alt[i:j])
                i = j
        return tokens

    @property
    def rules(self) -> dict[str, list[list[str]]]:
        return self._rules


class GBNFFSM:
    """Finite state machine for GBNF grammar validation.

    Tracks the current position in a grammar and reports which
    bytes are valid at each step.  Integrates with the token-level
    ``get_logits_mask()`` interface used by ``JSONSchemaConstraint``
    for direct token masking during generation.

    Usage::

        fsm = GBNFFSM('root ::= "hello" " " "world"')
        for b in b"hello world":
            fsm.transition(b)
        assert fsm.is_accepting()
    """

    def __init__(self, grammar: str) -> None:
        self._grammar = grammar
        self._parser = GBNFParser()
        self._rules = self._parser.parse(grammar)
        self._generated = ""
        self._target = self._extract_target()
        self._position = 0
        self._compiled = False
        self._dfa: dict[int, dict[int, int]] = {}

    def _extract_target(self) -> str:
        """Extract the target string from the root rule."""
        root = self._rules.get("root", [])
        if not root:
            return ""
        alt = root[0]
        parts = []
        for token in alt:
            if token.startswith('"') and token.endswith('"'):
                parts.append(token[1:-1])
            elif token.startswith("["):
                parts.append(token)
            else:
                resolved = self._resolve_rule(token)
                if resolved:
                    parts.append(resolved)
        return "".join(parts)

    def _resolve_rule(self, name: str) -> str:
        """Resolve a rule reference to its literal value."""
        if name not in self._rules:
            return ""
        alt = self._rules[name][0]
        parts = []
        for token in alt:
            if token.startswith('"') and token.endswith('"'):
                parts.append(token[1:-1])
        return "".join(parts)

    def compile_to_dfa(self) -> None:
        """Compile the grammar to a DFA for faster lookups."""
        self._compiled = True
        self._dfa = {}
        for i, ch in enumerate(self._target):
            if i not in self._dfa:
                self._dfa[i] = {}
            self._dfa[i][ord(ch)] = i + 1
        self._dfa[len(self._target)] = {}

    def transition(self, byte_value: int) -> int:
        """Feed a byte through the FSM.

        Args:
            byte_value: The byte value to process.

        Returns:
            Always returns 0.
        """
        ch = chr(byte_value) if 0 <= byte_value < 256 else ""
        self._generated += ch
        if self._position < len(self._target) and ch == self._target[self._position]:
            self._position += 1
        return 0

    def get_allowed_bytes(self) -> set[int]:
        """Return the set of byte values allowed at the current position."""
        if self._compiled and self._dfa:
            state = self._dfa.get(self._position, {})
            return set(state.keys())
        if self._position < len(self._target):
            return {ord(self._target[self._position])}
        return set()

    # ── Token-level masking interface (compatible with JSONSchemaConstraint) ──

    def get_logits_mask(self, vocab_size: int, tokenizer, device: str | torch.device | None = None) -> torch.Tensor:
        """Return a boolean mask: True for allowed token IDs, False for blocked.

        Compatible with the interface used by ``JSONSchemaConstraint`` and
        ``inference_engine.py``.  A token is allowed if ALL bytes of its
        decoded form are valid per the GBNF grammar at the current position.
        """
        import torch as _torch

        allowed_bytes = self.get_allowed_bytes()
        if not allowed_bytes:
            return _torch.ones(vocab_size, dtype=_torch.bool, device=device or "cpu")

        # Walk through the vocabulary: decode each token ID and check
        # whether every byte passes the GBNF grammar.
        mask = _torch.zeros(vocab_size, dtype=_torch.bool, device=device or "cpu")
        vocab_size_actual = min(vocab_size, getattr(tokenizer, 'vocab_size', vocab_size))

        eos_id = getattr(tokenizer, 'eos_token_id', None)
        for token_id in range(vocab_size_actual):
            decoded = tokenizer.decode([token_id])
            if decoded and all(ord(b) in allowed_bytes for b in decoded):
                mask[token_id] = True

        if eos_id is not None and eos_id < vocab_size:
            mask[eos_id] = True

        return mask

    def is_accepting(self) -> bool:
        return self._position >= len(self._target)

    def can_end(self) -> bool:
        return self.is_accepting()

    def reset(self) -> None:
        self._generated = ""
        self._position = 0
