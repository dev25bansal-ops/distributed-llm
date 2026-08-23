"""Grammar-constrained decoding with a FORMAL validity guarantee.

This module integrates ``outlines`` / ``outlines_core`` so that generated
output is **guaranteed** valid with respect to a JSON schema, a regex, or a
GBNF grammar.  At every decode step only grammar-valid tokens are selectable
(via a boolean logits mask derived from the outlines finite-state machine),
so the produced token stream is valid *by construction* — the
``OutputRepairer`` post-hoc repair path becomes unnecessary for the
constrained path.

``outlines`` is an **optional** dependency.  Every outlines import is guarded
by a ``try/except ImportError`` so this module imports without it.  When
outlines is unavailable (or the caller opts out), the existing GBNF +
``OutputRepairer`` fallback path is used instead — see
:meth:`GrammarConstrainedGenerator.create` and
:func:`grammar_constrained_or_fallback`.

The validity guarantee is proven model-free (no LLM required) by exercising
the pure outlines FSM machinery: for any sequence built by always picking a
token from the allowed-token set at each step, ``Guide.accepts_tokens`` /
``Index.is_final_state`` confirm the sequence is accepted by the grammar.
See ``tests/regression_high/test_a2_grammar_guaranteed.py``.

Usage::

    gen = GrammarConstrainedGenerator.create(schema={"type": "object", ...})
    mask = gen.get_logits_mask(vocab_size, tokenizer)   # True = allowed
    # ... sample only from allowed tokens, then:
    gen.advance(token_id)
    if gen.is_finished():
        break
    # produced stream is guaranteed grammar-valid
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Optional outlines dependency (guarded) ─────────────────────────────────
try:  # pragma: no cover - exercised only when outlines is installed
    from outlines_core import Guide, Index, Vocabulary  # type: ignore
    from outlines_core.json_schema import build_regex_from_schema  # type: ignore

    OUTLINES_AVAILABLE = True
except Exception:  # ImportError or any transitive failure
    Guide = Index = Vocabulary = None  # type: ignore
    build_regex_from_schema = None  # type: ignore
    OUTLINES_AVAILABLE = False


class OutlinesUnavailableError(RuntimeError):
    """Raised when an outlines-backed feature is used without ``outlines``."""


def gbnf_to_regex(grammar: str, start_rule: str = "root") -> str:
    """Translate a (subset of) GBNF grammar into an equivalent regex string.

    Handles the constructs emitted by ``distllm.utils.gbnf_grammar`` and the
    repo's GBNF tests:

    * quoted string literals ``"..."``
    * character classes ``[...]`` (with ranges, negation ``[^...]``)
    * alternation ``a | b``
    * groups ``( ... )``
    * repetition ``*`` ``+`` ``?``
    * rule references (inlined by expansion)

    ``ws`` whitespace rules are treated as ``[ \\t\\n\\r]*``.

    Raises ``ValueError`` if the grammar cannot be translated (e.g. unsupported
    construct) so callers can fall back to the GBNF + repairer path.
    """
    rules: dict[str, str] = {}
    for line in str(grammar).strip().split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "::=" in stripped:
            name, _, body = stripped.partition("::=")
            name = name.strip()
            body = body.strip()
            # A body may continue on subsequent (non-rule) lines; start the
            # accumulation and join continuations with spaces below.
            rules[name] = body
        elif rules:
            # Continuation line of the most-recently-defined rule.
            last = next(reversed(rules))
            rules[last] = (rules[last] + " " + stripped).strip()

    if not rules:
        raise ValueError("GBNF grammar contains no rules")

    # Expand a body into a regex.  ``ws`` -> optional whitespace, rule refs are
    # inlined, and top-level ``|`` becomes regex alternation.  A depth guard
    # avoids infinite recursion on cyclic grammars.
    def expand(body: str, depth: int = 0) -> str:
        if depth > 32:
            raise ValueError("GBNF rule expansion exceeded depth limit (cycle?)")
        alts = _split_top_level(body)
        return "|".join(_expand_alt(a, depth) for a in alts)

    def _expand_alt(body: str, depth: int) -> str:
        out: list[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch.isspace():
                i += 1
                continue
            if ch == '"':
                end = body.index('"', i + 1)
                lit = body[i + 1 : end]
                out.append(re.escape(lit))
                i = end + 1
            elif ch == "[":
                end = body.index("]", i + 1)
                cls = body[i : end + 1]
                out.append(_gbnf_char_class_to_regex(cls))
                i = end + 1
            elif ch == "(":
                depth_p = 1
                j = i + 1
                while j < n and depth_p > 0:
                    if body[j] == "(":
                        depth_p += 1
                    elif body[j] == ")":
                        depth_p -= 1
                    j += 1
                inner = expand(body[i + 1 : j - 1], depth + 1)
                out.append("(?:%s)" % inner)
                i = j
            elif ch in "*+?":
                if out:
                    out[-1] += ch
                i += 1
            else:
                j = i
                while j < n and (not body[j].isspace()) and body[j] not in '"[]()*+?|':
                    j += 1
                ref = body[i:j]
                if ref == "ws":
                    out.append(r"[ \t\n\r]*")
                elif ref in rules:
                    out.append(expand(rules[ref], depth + 1))
                else:
                    out.append(re.escape(ref))
                i = j
        return "".join(out)

    return expand(rules.get(start_rule, next(iter(rules.values()))))


def _split_top_level(body: str) -> list[str]:
    """Split a rule body on top-level ``|`` (ignoring ``|`` inside quotes/[])."""
    parts: list[str] = []
    cur = ""
    in_q = False
    depth = 0
    for ch in body:
        if ch == '"':
            in_q = not in_q
            cur += ch
        elif ch == "[" and not in_q:
            depth += 1
            cur += ch
        elif ch == "]" and not in_q:
            depth -= 1
            cur += ch
        elif ch == "(" and not in_q:
            depth += 1
            cur += ch
        elif ch == ")" and not in_q:
            depth -= 1
            cur += ch
        elif ch == "|" and not in_q and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _gbnf_char_class_to_regex(cls: str) -> str:
    """Convert a GBNF character class ``[...]`` into a regex character class."""
    inner = cls[1:-1]
    if inner.startswith("^"):
        return "[^" + re.escape(inner[1:]) + "]"
    return "[" + re.escape(inner) + "]"


def _grammar_text_from_source(
    schema: dict | str | None,
    regex: str | None,
    grammar: str | None,
) -> tuple[str, str]:
    """Return ``(outlines_index_input, kind)``.

    ``kind`` is one of ``"json"`` / ``"regex"`` and tells how to feed the
    outlines ``Index``.
    """
    if schema is not None:
        if isinstance(schema, str):
            schema_dict = json.loads(schema)
        else:
            schema_dict = schema
        # Validate the schema is a real JSON schema by building the regex.
        if build_regex_from_schema is None:  # pragma: no cover
            raise OutlinesUnavailableError("outlines_core not installed")
        regex_text = build_regex_from_schema(json.dumps(schema_dict))
        return regex_text, "regex"

    if regex is not None:
        return str(regex), "regex"

    if grammar is not None:
        # GBNF grammar -> regex translation (raises ValueError if unsupported).
        regex_text = gbnf_to_regex(grammar)
        return regex_text, "regex"

    raise ValueError("one of schema / regex / grammar must be provided")


class GrammarConstrainedGenerator:
    """Outlines-backed grammar-constrained generator with a validity guarantee.

    The generator wraps an outlines ``Index`` (the compiled FSM) and a
    ``Guide`` (per-generation state).  At every step
    :meth:`get_allowed_token_ids` returns exactly the token ids that keep the
    stream grammar-valid, so any sampling restricted to that set yields a
    *guaranteed-valid* sequence.  The OutputRepairer is therefore not needed
    on this path.

    The outlines dependency is optional.  Use the :meth:`create` factory to
    obtain an instance, or ``None`` when outlines is unavailable / fallback is
    requested.
    """

    def __init__(
        self,
        schema: dict | str | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        eos_token_id: int | None = None,
        vocabulary: dict[str, list[int]] | None = None,
    ) -> None:
        if not OUTLINES_AVAILABLE:
            raise OutlinesUnavailableError(
                "outlines_core is not installed; install with "
                "`uv pip install outlines` or use the GBNF + OutputRepairer fallback."
            )

        self._eos_token_id = eos_token_id
        self._custom_vocab = vocabulary
        self._index_input, self._kind = _grammar_text_from_source(schema, regex, grammar)
        self._regex_text = self._index_input  # for inspection

        # The vocabulary is lazily built from a tokenizer on first use; if a
        # custom symbol vocabulary was supplied we build the Index now.
        self._vocabulary: Any = None
        self._index: Any = None
        self._guide: Any = None
        if vocabulary is not None:
            self._build_index(vocabulary)

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        schema: dict | str | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        eos_token_id: int | None = None,
        vocabulary: dict[str, list[int]] | None = None,
        force_fallback: bool = False,
    ) -> "GrammarConstrainedGenerator | None":
        """Build an outlines-backed generator, or ``None`` to signal fallback.

        Args:
            force_fallback: When True (or outlines unavailable), return ``None``
                so the caller falls back to the GBNF + OutputRepairer path.

        Returns:
            A ``GrammarConstrainedGenerator`` or ``None`` (fallback needed).
        """
        if force_fallback or not OUTLINES_AVAILABLE:
            return None
        try:
            return cls(
                schema=schema,
                regex=regex,
                grammar=grammar,
                eos_token_id=eos_token_id,
                vocabulary=vocabulary,
            )
        except (ValueError, OutlinesUnavailableError):
            # Unsupported grammar / schema -> let caller use fallback path.
            return None

    # ── outlines machinery ──────────────────────────────────────────────────

    def _build_vocabulary_from_tokenizer(self, tokenizer: Any) -> Any:
        """Build an outlines ``Vocabulary`` from a HuggingFace-style tokenizer."""
        eos_id = self._eos_token_id
        if eos_id is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_token = getattr(tokenizer, "eos_token", None)

        formatted: dict[str, list[int]] = {}
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if get_vocab is not None:
            for token_str, token_id in get_vocab().items():
                formatted.setdefault(token_str, []).append(int(token_id))
        else:
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            for tid in range(vocab_size):
                try:
                    token_str = tokenizer.decode([tid])
                except Exception:
                    continue
                if token_str:
                    formatted.setdefault(token_str, []).append(int(tid))
        # outlines forbids EOS being present in the vocabulary
        if eos_token is not None and eos_token in formatted:
            formatted.pop(eos_token, None)
        return Vocabulary(eos_id if eos_id is not None else -1, formatted)

    def _build_index(self, vocabulary: Any) -> None:
        if not isinstance(vocabulary, Vocabulary):
            # outlines forbids a negative eos token id; when none is
            # supplied use a sentinel id that is NOT part of the grammar
            # vocabulary, so it never appears in the allowed-token set.
            # Termination is driven by is_finished()/final-state, not eos.
            eos = self._eos_token_id
            if eos is None or eos < 0:
                all_ids = [i for ids in vocabulary.values() for i in ids]
                eos = (max(all_ids) + 1) if all_ids else 0
            vocabulary = Vocabulary(eos, vocabulary)
        self._vocabulary = vocabulary
        self._index = Index(self._regex_text, vocabulary)
        self._guide = Guide(self._index)

    def _ensure_index(self, tokenizer: Any = None) -> None:
        if self._index is not None:
            return
        if self._custom_vocab is not None:
            self._build_index(self._custom_vocab)
            return
        if tokenizer is None:
            raise ValueError(
                "a tokenizer is required to build the outlines vocabulary "
                "(or pass `vocabulary=` with a custom symbol vocabulary)"
            )
        vocab = self._build_vocabulary_from_tokenizer(tokenizer)
        self._build_index(vocab)

    # ── Model-free token-level interface (the heart of the guarantee) ────────

    def reset(self) -> None:
        """Reset the per-generation FSM state."""
        if self._guide is not None:
            self._guide.reset()

    def get_allowed_token_ids(self) -> list[int]:
        """Return the token ids that keep the stream grammar-valid *now*.

        Picking any id from this set and advancing is guaranteed to keep the
        produced sequence accepted by the grammar.  This is the formal
        guarantee: restricting sampling to this set makes every outcome valid.
        """
        if self._index is None:
            raise RuntimeError("index not built; call _ensure_index first")
        state = self._index.get_initial_state() if self._guide is None else self._guide.get_state()
        return list(self._index.get_allowed_tokens(state))

    def advance(self, token_id: int) -> None:
        """Advance the FSM after committing ``token_id``."""
        if self._guide is None:
            raise RuntimeError("guide not built; call _ensure_index first")
        self._guide.advance(int(token_id))

    def is_finished(self) -> bool:
        """Return True once the FSM reached a final/accepting state."""
        if self._guide is None:
            return False
        return bool(self._guide.is_finished())

    def accepts_tokens(self, token_sequence: list[int]) -> bool:
        """Return True iff the given token sequence is accepted by the grammar.

        Model-free verification of a produced stream.
        """
        if self._index is None:
            raise RuntimeError("index not built; call _ensure_index first")
        guide = Guide(self._index)
        return bool(guide.accepts_tokens(list(token_sequence)))

    def generate_guaranteed(
        self,
        choose: Any = None,
        max_steps: int = 100_000,
        rng: Any = None,
    ) -> tuple[list[int], bool]:
        """Model-free reference sampler: always picks an allowed token.

        Used by tests to *prove* the guarantee without an LLM.  At each step
        it consults the allowed-token set (the token mask) and commits one
        allowed token (first by default, or chosen by ``choose(allowed)`` /
        ``rng``).  The resulting sequence is guaranteed accepted by the grammar.

        Returns:
            ``(token_sequence, finished)`` where ``finished`` indicates a final
            state was reached.
        """
        if self._index is None:
            raise RuntimeError("index not built; call _ensure_index first")
        state = self._index.get_initial_state()
        out: list[int] = []
        finished = False
        for _ in range(max_steps):
            allowed = sorted(self._index.get_allowed_tokens(state))
            if not allowed:
                break
            if choose is not None:
                tok = int(choose(allowed))
            elif rng is not None:
                tok = int(rng.choice(allowed))
            else:
                tok = int(allowed[0])
            out.append(tok)
            state = self._index.get_next_state(state, tok)
            if self._index.is_final_state(state):
                finished = True
                break
        return out, finished

    # ── LLM integration: logits mask ────────────────────────────────────────

    def get_logits_mask(
        self,
        vocab_size: int,
        tokenizer: Any = None,
        device: str | Any | None = None,
    ) -> Any:
        """Return a boolean mask over token ids (True = grammar-valid).

        LLM integration hook mirroring the interface of
        ``SchemaConstrainedDecoder`` / ``JSONSchemaConstraint``.  Allowed ids
        are taken directly from the outlines ``Index`` (token-level, no decode
        needed) and EOS is permitted only in an accepting state.
        """
        self._ensure_index(tokenizer)
        idx = self._index
        state = self._guide.get_state()

        import torch

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device or "cpu")
        allowed = idx.get_allowed_tokens(state)
        for tid in allowed:
            if 0 <= tid < vocab_size:
                mask[tid] = True
        # Permit EOS only when we could validly finish here.
        eos_id = self._eos_token_id
        if eos_id is None and tokenizer is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and 0 <= eos_id < vocab_size:
            if idx.is_final_state(state):
                mask[eos_id] = True
            else:
                mask[eos_id] = False
        return mask

    @property
    def regex_text(self) -> str:
        """The outlines regex the FSM was compiled from (for inspection)."""
        return self._regex_text

    @property
    def outlines_available(self) -> bool:
        return OUTLINES_AVAILABLE


def grammar_constrained_or_fallback(
    schema: dict | str | None = None,
    regex: str | None = None,
    grammar: str | None = None,
    eos_token_id: int | None = None,
    vocabulary: dict[str, list[int]] | None = None,
    tokenizer: Any = None,
    force_fallback: bool = False,
) -> tuple["GrammarConstrainedGenerator | None", bool]:
    """Return ``(generator, used_fallback)``.

    ``generator`` is an outlines-backed ``GrammarConstrainedGenerator`` when
    available and supported; otherwise ``None`` and the caller should use the
    existing GBNF + ``OutputRepairer`` path.  ``used_fallback`` tells the
    caller which path was taken.

    This is the single integration point for the rest of the codebase: call it,
    and branch on ``used_fallback`` to decide whether to run the repair path.
    """
    gen = GrammarConstrainedGenerator.create(
        schema=schema,
        regex=regex,
        grammar=grammar,
        eos_token_id=eos_token_id,
        vocabulary=vocabulary,
        force_fallback=force_fallback,
    )
    if gen is None:
        return None, True
    return gen, False


# ── Hook into the existing SchemaConstrainedDecoder (opt-in, backward compat) ──
def _patch_schema_constrained_decoder() -> None:
    """Add a ``grammar_constrained`` factory method to ``SchemaConstrainedDecoder``.

    Imported lazily to avoid a hard dependency.  No-op if the class is absent.
    The method returns a ``GrammarConstrainedGenerator`` when outlines is
    available, else ``None`` (caller falls back to GBNF + OutputRepairer).
    """
    try:
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
    except Exception:
        return

    if hasattr(SchemaConstrainedDecoder, "grammar_constrained"):
        return

    def grammar_constrained(self, schema=None, regex=None, grammar=None, force_fallback=False):
        gen, _ = grammar_constrained_or_fallback(
            schema=schema,
            regex=regex,
            grammar=grammar,
            eos_token_id=getattr(self._token_index, "eos_token_id", None)
            if getattr(self, "_token_index", None) is not None
            else None,
            force_fallback=force_fallback,
        )
        return gen

    SchemaConstrainedDecoder.grammar_constrained = grammar_constrained  # type: ignore
