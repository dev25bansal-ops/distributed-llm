"""Regression tests for Critical bug C3: JSONSchemaConstraint nesting tracking.

The old FSM popped ``_stack`` but never pushed it, so the FIRST ``}`` or ``]``
at ANY depth signalled ``done``; ``after_value`` omitted ``]``; array commas
routed to ``after_comma`` (quoted-key-only).  Any ``response_format={"type":
"json_object"}`` generation of ``[1,2]``, ``["a"]`` or ``{"a":{"b":1}}`` had
digits / closers masked out mid-array and terminated early on the first inner
close.

These tests assert, for valid documents:
  * every character prefix is accepted by the FSM (next char ∈ _valid_next_chars),
  * ``is_complete()`` flips exactly at the end (outermost close / root scalar),
and for invalid mutations:
  * mismatched closers, trailing commas and stray delimiters are masked.

A seeded property/fuzz suite generates random JSON documents (stdlib ``json``
serialization: compact, spaced and indented) and checks the same invariants,
plus cross-container close probes (when ``}`` is required, ``]`` must be
masked, and vice versa).
"""

from __future__ import annotations

import json
import random
import string as _string

import pytest
import torch

from distllm.core.structured_output import JSONSchemaConstraint


# ─── Helpers ────────────────────────────────────────────────────────────────

def _walk_prefixes(doc: str):
    """Feed *doc* one char at a time; yield (index, valid_set_before) pairs."""
    c = JSONSchemaConstraint()
    for i, ch in enumerate(doc):
        valid = c._valid_next_chars()
        yield i, ch, valid, c
        c.update(ch)


def _assert_valid_document_walks(doc: str, *, expect_premature_free: bool = True) -> None:
    """Every prefix accepts its next char; completion happens exactly at the end."""
    completions = []
    for i, ch, valid, c in _walk_prefixes(doc):
        assert ch in valid, (
            f"prefix {doc[:i]!r} (state={c._state!r}) rejects next char {ch!r}"
        )
        completions.append(c.is_complete())
    c_final = JSONSchemaConstraint()
    c_final.update(doc)
    assert c_final.is_complete(), f"{doc!r} not complete at end (state={c_final._state!r})"
    if expect_premature_free:
        early = [i for i, done in enumerate(completions[:-1]) if done]
        assert not early, f"{doc!r} completed prematurely at indices {early}"


def _structural_positions(doc: str) -> list[bool]:
    """Per-character flags: True when the char sits OUTSIDE any string."""
    flags = []
    in_str = False
    escaped = False
    for ch in doc:
        flags.append(not in_str)
        if escaped:
            escaped = False
        elif in_str:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    return flags


# A fake tokenizer whose EOS decodes to '' so it can never collide with a
# valid JSON first character; every other printable ASCII char has a token.
_FAKE_CHARS = [""] + [chr(c) for c in range(32, 127)] + ["\t", "\n"]


class _FullASCIITokenizer:
    name_or_path = "fsm-nesting-test"
    vocab_size = len(_FAKE_CHARS)
    eos_token_id = 0

    def decode(self, token_ids):
        return _FAKE_CHARS[token_ids[0]]


# ─── Hand-written valid documents ───────────────────────────────────────────

HANDWRITTEN_DOCS = [
    # The exact repro shapes from the C3 finding:
    "[1,2]",
    '["a","b"]',
    '{"a":{"b":[1,{"c":null}]},"d":true}',
    "[]",
    "{}",
    '"a}{["',           # braces/brackets inside a string
    '{"a":""}',         # empty string value
    # Additional structure / edge coverage:
    '""',
    '"hi"',
    "true",
    "false",
    "null",
    "[true,false,null]",
    "[[]]",
    "[[[]],[1],[1,2]]",
    '{"a":{"b":{"c":{"d":1}}}}',
    '{"a":[1,2,{"b":["x",null,false]}]}',
    '{"nested":{"deep":{"deeper":[1,2,[3,{"z":false}]]}}}',
    '{"k": "v"}',       # space after colon
    "[1, 2, 3]",        # spaces after commas
    '{\n  "a": [\n    1,\n    -2.5e3\n  ]\n}',  # pretty-printed
    '{"n":123}',
    '{"n":-0.5e+10}',
    '{"empty_str":"","num":0,"neg":-1}',
    '{"esc":"a\\"}{[]\\\\b\\n"}',  # escaped quote/backslash + structural chars in string
    '{"tab":"\\t\\u0041"}',
]


class TestHandwrittenValidDocuments:
    @pytest.mark.parametrize(
        "doc",
        HANDWRITTEN_DOCS,
        ids=[f"case{i}:{d[:18]!r}" for i, d in enumerate(HANDWRITTEN_DOCS)],
    )
    def test_every_prefix_accepted_and_done_exactly_at_end(self, doc):
        # Guard the guard: each fixture must itself be valid JSON.
        json.loads(doc)
        _assert_valid_document_walks(doc)

    def test_update_in_chunks_matches_char_wise(self):
        """update() with realistic multi-char tokens equals char-by-char."""
        doc = '{"a":{"b":[1,{"c":null}]},"d":true}'
        chunks = [doc[i:i + 3] for i in range(0, len(doc), 3)]
        c = JSONSchemaConstraint()
        for chunk in chunks:
            c.update(chunk)
        assert c.is_complete()
        assert c.generated_text == doc


class TestArrayScalarsAndCommas:
    """C3 core repro: digits / closers / commas inside arrays."""

    def test_digits_not_masked_mid_array(self):
        c = JSONSchemaConstraint()
        c.update("[")
        valid = c._valid_next_chars()
        assert {"0", "1", "2", "-", "9"} <= valid, "digits masked at array start"
        c.update("1")
        assert {"2", "0", "9"} <= c._valid_next_chars(), "second digit masked"

    def test_string_elements_allowed_after_array_comma(self):
        c = JSONSchemaConstraint()
        c.update('["a",')
        valid = c._valid_next_chars()
        assert '"' in valid, "string element blocked after array comma"
        assert "]" not in valid, "trailing-comma close must be masked"

    def test_array_commas_do_not_demand_object_keys(self):
        """Old bug: array comma routed to after_comma ({'"'} only)."""
        c = JSONSchemaConstraint()
        c.update("[1,")
        valid = c._valid_next_chars()
        assert {"0", "-", "{", "[", "t", "f", "n"} & valid, "non-string elements blocked"

    def test_closing_array_after_scalar_value(self):
        c = JSONSchemaConstraint()
        c.update("[1")
        assert "]" in c._valid_next_chars(), "array cannot close after scalar (old after_value bug)"
        assert "}" not in c._valid_next_chars()

    def test_nested_close_does_not_finish_document(self):
        c = JSONSchemaConstraint()
        c.update('{"a":{"b":1}')
        assert not c.is_complete(), "inner close finished the document (C3)"
        assert c._stack, "inner close did not leave outer object open"
        c.update("}")
        assert c.is_complete()

    def test_deeply_nested_close_sequence(self):
        c = JSONSchemaConstraint()
        c.update('{"a":[1,{"b":[2]}]')
        assert not c.is_complete()
        c.update("}")
        assert c.is_complete()


class TestObjectStructure:
    def test_empty_object_immediately_completes(self):
        c = JSONSchemaConstraint()
        c.update("{}")
        assert c.is_complete()
        assert c._state == "done"

    def test_second_entry_requires_quoted_key(self):
        c = JSONSchemaConstraint()
        c.update('{"a":1,')
        valid = c._valid_next_chars()
        assert '"' in valid
        assert "}" not in valid, "trailing-comma close must be masked in objects"
        assert "0" not in valid and ":" not in valid


class TestLiteralsNumbersWhitespace:
    def test_true_spelled_character_by_character(self):
        c = JSONSchemaConstraint()
        c.update("tru")
        assert c._valid_next_chars() == {"e"}, "literal continuation chars were masked"
        assert not c.is_complete()
        c.update("e")
        assert c.is_complete() and c._state == "done"

    def test_false_and_null(self):
        for lit in ("false", "null"):
            c = JSONSchemaConstraint()
            c.update(lit[:-1])
            assert c._valid_next_chars() == {lit[-1]}
            c.update(lit[-1])
            assert c.is_complete()

    def test_whitespace_terminates_number_inside_object(self):
        c = JSONSchemaConstraint()
        c.update('{"a": 1 ')
        assert c._state != "in_number"
        assert "}" in c._valid_next_chars()
        c.update("}")
        assert c.is_complete()

    def test_mismatched_close_masked_mid_number(self):
        """C3 follow-up: number terminators must respect the container."""
        c = JSONSchemaConstraint()
        c.update('{"a": 1')
        assert "]" not in c._valid_next_chars(), "] offered to terminate number in object"
        assert {"}", ","} <= c._valid_next_chars()
        c2 = JSONSchemaConstraint()
        c2.update("[1")
        assert "}" not in c2._valid_next_chars(), "} offered to terminate number in array"
        assert "]" in c2._valid_next_chars()

    def test_root_number_is_complete_and_extensible(self):
        c = JSONSchemaConstraint()
        c.update("42")
        assert c.is_complete(), "bare root number should count as complete JSON"
        assert {"4", "0", "."} & c._valid_next_chars(), "digits masked mid root-number"


class TestInvalidMutationsMasked:
    """INVALID inputs must be rejected by _valid_next_chars()."""

    @staticmethod
    def _valid_after(prefix: str) -> set[str]:
        c = JSONSchemaConstraint()
        c.update(prefix)
        return c._valid_next_chars()

    def test_wrong_closer_for_object(self):
        assert "]" not in self._valid_after('{"a": 1')

    def test_wrong_closer_for_array(self):
        assert "}" not in self._valid_after('[1, 2')

    def test_trailing_comma_array_close(self):
        assert "]" not in self._valid_after("[1,2,")

    def test_trailing_comma_object_close(self):
        assert "}" not in self._valid_after('{"a":1,')

    def test_comma_right_after_colon(self):
        assert "," not in self._valid_after('{"a":')

    def test_colon_outside_object_context(self):
        assert ":" not in self._valid_after('["a"')

    def test_unclosed_document_not_complete(self):
        for prefix in ('{"a": 1', "[1, 2", '{"a": [1', '{"a"', "[", "{", '"unterminated'):
            c = JSONSchemaConstraint()
            c.update(prefix)
            assert not c.is_complete(), f"{prefix!r} considered complete"

    def test_done_is_absorbing_and_masks_everything(self):
        c = JSONSchemaConstraint()
        c.update("{}")
        assert c.is_complete()
        c.update(', "more": [1, 2], garbage{[')
        assert c._state == "done"
        assert c.is_complete()
        assert c._valid_next_chars() == set()

    def test_partial_literal_rejects_other_letters(self):
        c = JSONSchemaConstraint()
        c.update("tru")
        valid = c._valid_next_chars()
        assert valid == {"e"}
        assert not ({"r", "u", "n", "f", "t"} & valid)


# ─── Property / fuzz suite ──────────────────────────────────────────────────

_STRING_ALPHABET = 'ab{}[]"\':,012 \\-/x\t'


class _Gen:
    """Seeded random JSON document generator (bounded depth/size)."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def string(self) -> str:
        n = self.rng.randint(0, 6)
        return "".join(self.rng.choice(_STRING_ALPHABET) for _ in range(n))

    def scalar(self) -> object:
        return self.rng.choice([
            self.string(),
            self.rng.randint(-999, 999),
            round(self.rng.uniform(-1000, 1000), 3),
            self.rng.choice([True, False]),
            None,
        ])

    def value(self, depth: int) -> object:
        if depth <= 0 or self.rng.random() < 0.35:
            return self.scalar()
        if self.rng.random() < 0.5:
            return [self.value(depth - 1) for _ in range(self.rng.randint(0, 4))]
        return {
            self.string() or f"k{i}": self.value(depth - 1)
            for i in range(self.rng.randint(0, 4))
        }

    def document(self) -> object:
        """Roots: object/array/string/bool/null. Bare numbers are excluded at
        the root because any digit-prefix of a root number is itself complete
        JSON, making 'completion exactly at the end' ill-defined."""
        kind = self.rng.choice(["obj", "arr", "str", "bool", "null"])
        if kind == "obj":
            return {
                self.string() or f"k{i}": self.value(3)
                for i in range(self.rng.randint(0, 5))
            }
        if kind == "arr":
            return [self.value(3) for _ in range(self.rng.randint(0, 5))]
        if kind == "str":
            return self.string()
        if kind == "bool":
            return self.rng.choice([True, False])
        return None

    @staticmethod
    def serialize(obj: object, style: str) -> str:
        if style == "compact":
            return json.dumps(obj, separators=(",", ":"))
        if style == "spaced":
            return json.dumps(obj)  # ", " / ": "
        return json.dumps(obj, indent=int(style.split("=")[1]))


class TestFuzzValidDocuments:
    def test_random_documents_accept_all_prefixes_and_complete_at_end(self):
        rng = random.Random(20260824)
        gen = _Gen(rng)
        styles = ["compact", "spaced", "indent=1", "indent=2"]
        checked = 0
        for iteration in range(120):
            obj = gen.document()
            style = styles[iteration % len(styles)]
            doc = gen.serialize(obj, style)
            assert json.loads(doc) == obj  # serializer sanity
            _assert_valid_document_walks(doc)
            checked += 1
        assert checked == 120

    def test_random_documents_cross_container_probe(self):
        """When the true next char is '}', ']' must be masked (and vice versa);
        when it is ':', ',' must be masked. Only structural positions probed."""
        rng = random.Random(987654321)
        gen = _Gen(rng)
        probes = 0
        for _ in range(80):
            doc = gen.serialize(gen.document(), "compact")
            flags = _structural_positions(doc)
            for i, ch, valid, _c in _walk_prefixes(doc):
                assert ch in valid
                if not flags[i]:
                    continue
                if ch == "}":
                    assert "]" not in valid, f"probe failed at {doc[:i]!r}"
                    probes += 1
                elif ch == "]":
                    assert "}" not in valid, f"probe failed at {doc[:i]!r}"
                    probes += 1
                elif ch == ":":
                    assert "," not in valid, f"probe failed at {doc[:i]!r}"
                    probes += 1
        assert probes > 500, f"too few probes ran ({probes}); generator degenerate?"

    def test_random_mutations_are_masked(self):
        """At a structural position, substituting the opposite delimiter must
        be rejected by the FSM (this is the exact failure mode of C3)."""
        rng = random.Random(55555)
        gen = _Gen(rng)
        wrong_for = {"}": "]", "]": "}", ":": ",", ",": ":"}
        mutations = 0
        for _ in range(80):
            doc = gen.serialize(gen.document(), "compact")
            flags = _structural_positions(doc)
            candidates = [i for i in range(len(doc)) if flags[i] and doc[i] in wrong_for]
            rng.shuffle(candidates)
            for i in candidates[:4]:
                wrong = wrong_for[doc[i]]
                c = JSONSchemaConstraint()
                c.update(doc[:i])
                assert wrong not in c._valid_next_chars(), (
                    f"mutation {doc[:i]!r} + {wrong!r} not masked "
                    f"(original {doc!r}, state={c._state!r}, stack={c._stack})"
                )
                mutations += 1
        assert mutations > 60, f"too few mutations checked ({mutations})"


# ─── Mask-level integration (get_logits_mask) ───────────────────────────────

class TestLogitsMaskIntegration:
    @pytest.fixture
    def tok(self):
        return _FullASCIITokenizer()

    def test_c3_repro_array_of_integers_never_blocked(self, tok):
        """response_format=json_object generating '[1,2]': every needed char
        must survive masking mid-document, and EOS must be gated."""
        c = JSONSchemaConstraint.from_response_format({"type": "json_object"})
        assert isinstance(c, JSONSchemaConstraint)
        for ch in "[1,2]":
            m = c.get_logits_mask(tok.vocab_size, tok)
            assert m.dtype == torch.bool and m.shape == (tok.vocab_size,)
            assert not m[tok.eos_token_id].item(), f"EOS allowed before {ch!r}"
            matching = [i for i in range(tok.vocab_size) if _FAKE_CHARS[i] == ch]
            assert any(m[i].item() for i in matching), f"mask blocked {ch!r} mid-array"
            c.update(ch)
        m = c.get_logits_mask(tok.vocab_size, tok)
        assert m[tok.eos_token_id].item(), "EOS blocked after completed document"
        assert m.sum().item() == 1, "non-EOS tokens allowed after completed document"

    def test_nested_document_eos_gated_until_outermost_close(self, tok):
        doc = '{"a":{"b":[1,{"c":null}]},"d":true}'
        c = JSONSchemaConstraint(schema={})
        for idx, ch in enumerate(doc):
            m = c.get_logits_mask(tok.vocab_size, tok)
            # '<' etc. aside, EOS decodes to '' which never matches a valid ord;
            # mid-document it must never be offered.
            assert not m[tok.eos_token_id].item(), f"EOS allowed at prefix {doc[:idx]!r}"
            matching = [i for i in range(tok.vocab_size) if _FAKE_CHARS[i] == ch]
            assert any(m[i].item() for i in matching), f"mask blocked {ch!r} at {doc[:idx]!r}"
            c.update(ch)
        m = c.get_logits_mask(tok.vocab_size, tok)
        assert m.sum().item() == 1 and m[tok.eos_token_id].item()

    def test_mask_deterministic_per_state_and_evolves(self, tok):
        c = JSONSchemaConstraint()
        m1 = c.get_logits_mask(tok.vocab_size, tok)
        m2 = c.get_logits_mask(tok.vocab_size, tok)
        assert torch.equal(m1, m2), "same state must yield identical masks"
        c.update("{")
        m3 = c.get_logits_mask(tok.vocab_size, tok)
        assert not torch.equal(m1, m3), "mask did not evolve after '{'"

    def test_mask_fuzz_never_blocks_needed_char(self, tok):
        rng = random.Random(424242)
        gen = _Gen(rng)
        for _ in range(40):
            doc = gen.serialize(gen.document(), "compact")
            c = JSONSchemaConstraint()
            for ch in doc:
                m = c.get_logits_mask(tok.vocab_size, tok)
                matching = [i for i in range(tok.vocab_size) if _FAKE_CHARS[i] == ch]
                assert any(m[i].item() for i in matching), (
                    f"mask blocked {ch!r} at prefix {doc!r} (state={c._state!r})"
                )
                c.update(ch)


# ─── Public API stability ───────────────────────────────────────────────────

class TestPublicApiStable:
    def test_valid_next_chars_signature_and_type(self):
        c = JSONSchemaConstraint()
        valid = c._valid_next_chars()
        assert isinstance(valid, set)
        assert {'"', "{", "["} <= valid

    def test_from_response_format_types(self):
        assert isinstance(
            JSONSchemaConstraint.from_response_format({"type": "json_object"}).schema, dict
        )
        c = JSONSchemaConstraint.from_response_format(
            {"type": "json_schema", "schema": {"type": "object"}}
        )
        assert c.schema == {"type": "object"}
        assert JSONSchemaConstraint.from_response_format({"type": "??"}).schema is None

    def test_generated_text_accumulates(self):
        c = JSONSchemaConstraint()
        c.update('{"key": ')
        c.update('"value"}')
        assert c.generated_text == '{"key": "value"}'
        assert c.is_complete()
