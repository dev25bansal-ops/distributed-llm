"""Strategy module extracted from inference_engine.py (M-05 decomposition).

The InferenceEngine god class previously welded all five execution
strategies to itself via bidirectional references. Each strategy now lives
in its own module and receives the engine through a narrow context seam.
inference_engine.py re-imports these for backward compatibility, so all
existing call sites and tests continue to work unchanged.
"""

from __future__ import annotations

import torch
import uuid
from typing import Protocol

from distllm.core.spec_verify import accept_token


class EngineContext(Protocol):
    """Narrow seam the strategies may depend on.

    Any object providing these attributes can host the strategies,
    which makes them testable without a full InferenceEngine.
    """

    tokenizer: object
    local_partitioner: object | None
    _pipeline: object | None
    _stream_pool: object
    _node_order_property: list | None
    _draft_model_fn: object | None
    _draft_model_fns: list | None
    _draft_fleet: object | None
    _draft_adaptive: bool
    _draft_min_candidates: int
    _draft_max_candidates: int
    _spec_num_candidates: int
    _remote_draft_endpoint: str | None
    _remote_draft_api_key: str
    _remote_draft_model: str
    _remote_draft_transport: str
    _remote_draft_prompt_format: str
    _remote_draft_num_candidates: int
    _draft_min_candidates: int

    def _generate_local(self, prompt, max_new_tokens, temperature, top_p, top_k, **kw): ...
    def _iter_local_tokens(self, input_ids, max_new_tokens, **kw): ...
    def _generate_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k, request_id=None): ...
    def _generate_distributed_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k): ...


class PromptLookupStrategy:
    """Prompt-lookup speculative decoding — reuse matched n-grams from the
    generated prefix as draft tokens, requiring no separate draft model.

    When generating, the current suffix (last N tokens) is searched in the
    prefix.  If a match is found, the tokens *following* that match are used
    as draft candidates and verified in a single cached forward pass.

    The sequence is prefilled ONCE with ``use_cache=True``; every subsequent
    verification / fallback pass forwards only the new tokens with
    ``past_key_values`` threaded through.  Previously this strategy
    re-forwarded the entire sequence on every round (O(n²) total work), which
    made it dramatically slower than plain generation.
    """

    def __init__(self, engine: "InferenceEngine") -> None:
        self._engine = engine
        # How many recent tokens to use as the lookup key (n-gram size).
        self._ngram_size: int = 6
        # Max draft tokens to propose from matched prefix.
        self._max_draft: int = 10
        # Bug H-03 fix: incremental suffix index. Previously _find_match
        # scanned backward over the whole sequence allocating a fresh
        # min_match-length slice at every start position, EVERY round —
        # O(n) slice-allocations per lookup and O(n^2) across a generation
        # (~2M list ops per 512 tokens at 4K context). This dict maps each
        # observed n-gram (as a tuple) to the list of positions where it
        # occurred; the most recent position is tried first, giving the
        # same nearest-match semantics as the old backward scan in O(1)
        # amortized per lookup. Cleared whenever the engine resets.
        self._ngram_index: dict[tuple[int, ...], list[int]] = {}

    def _index_update(self, seq: list[int], min_match: int) -> None:
        """Record every complete n-gram position in ``seq`` into the index.

        Call after extending ``seq`` so new n-grams ending at the tail are
        added incrementally. Idempotent for already-indexed prefixes: the
        per-tuple position lists are append-only and each call only walks
        the NEW n-grams introduced by the extension.
        """
        n = len(seq)
        # n-grams are recorded once their last token exists, so n-grams
        # ending at index i (0-based) become available when len == i+1.
        # Track how far we have indexed to avoid re-walking.
        last_indexed = getattr(self, "_indexed_up_to", min_match - 1)
        if n <= last_indexed:
            return
        start = max(min_match, last_indexed + 1)
        for end in range(start, n + 1):
            key = tuple(seq[end - min_match:end])
            self._ngram_index.setdefault(key, []).append(end)
        self._indexed_up_to = n

    def _find_match(
        self, seq: list[int], min_match: int = 4,
    ) -> tuple[int, int] | None:
        """Find the nearest suffix match of *seq* in its own prefix.

        ``seq`` is a CPU-side mirror of the generated ids (kept incrementally
        so no GPU->CPU sync is needed per lookup round).

        Returns ``(match_end, num_draft)`` or ``None``.

        Bug H-03: this previously sliced-and-compared backward over the whole
        sequence every round (O(n^2) across a generation). Now consults the
        incremental ``_ngram_index`` and tries candidate positions from most
        recent backwards — identical semantics, O(1) amortized lookup.
        """
        n = len(seq)
        if n < min_match + 1:
            return None
        # Make sure everything indexed so far covers the current seq.
        self._index_update(seq, min_match)
        suffix_key = tuple(seq[-min_match:])
        positions = self._ngram_index.get(suffix_key)
        if not positions:
            return None
        # Most recent occurrence first (nearest-match semantics), but never
        # the suffix itself: the last recorded position is where the key's
        # final token sits == n, which IS the query suffix. The second-most-
        # recent is the true in-prefix match.
        for match_end in reversed(positions[:-1] if positions and positions[-1] == n else positions):
            available = n - match_end
            if available > 0:
                return match_end, min(available, self._max_draft)
        return None

    def _accept_drafts(
        self,
        prev_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_ids: torch.Tensor,
        temperature: float,
    ) -> int:
        """Number of leading draft tokens accepted.

        Textbook assisted-generation alignment: a causal LM's logits row at
        absolute position ``p`` predicts the token at position ``p + 1``.
        Draft token ``i`` occupies absolute position ``prefix_len + i``, so
        the model's opinion of it lives one row *earlier*: draft 0 is scored
        against ``prev_logits`` (the pending prediction that was available
        before the verify pass), and draft ``i >= 1`` against verify-pass
        row ``i - 1``.

        Greedy mode is fully batched (one GPU->CPU sync for all k positions);
        sampled mode keeps a per-position draw order of one ``torch.rand`` +
        ``multinomial``-free comparison per position.
        """
        k = draft_ids.shape[1]
        # Candidate predictions for positions prefix_len .. prefix_len+k-1:
        # the pending row followed by verify rows 0..k-2.  Only the first k
        # rows are consumed.
        predicted = torch.cat(
            [prev_logits.unsqueeze(1), verify_logits], dim=1,
        )[0][:k]
        accepted = 0
        if temperature == 0:
            predicted = predicted.argmax(dim=-1)  # [k]
            matches = (predicted == draft_ids[0]).tolist()  # one sync
            for ok in matches:
                if not ok:
                    break
                accepted += 1
            return accepted

        # Sampled mode: delegate to the shared verification kernel so this
        # loop uses true rejection sampling (min(1, p/q)) instead of the
        # raw p comparison that was here before (bug C-03). spec_verify is
        # the single source of truth for acceptance math.
        for i in range(k):
            if not accept_token(
                predicted.unsqueeze(0),  # (1, k, vocab) -> row addressable
                i,
                int(draft_ids[0, i].item()),
                draft_prob=None,  # n-gram draft: uniform fallback
                temperature=temperature,
                vocab_size=predicted.shape[-1],
            ):
                break
            accepted += 1
        return accepted

    def _generate_tokens(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        engine = self._engine
        model = engine.local_partitioner.full_model
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        # CPU mirror of the generated ids.  Maintained incrementally instead
        # of calling .tolist() on the device tensor every round (a full
        # GPU->CPU sync per lookup round in the old implementation).
        gen_cpu: list[int] = input_ids[0].tolist()

        with torch.no_grad():
            # Prefill once; thread past_key_values through every later pass.
            outputs = model(input_ids, use_cache=True)
            past = getattr(outputs, "past_key_values", None)
            # Invariant: pending_logits holds the logits at absolute position
            # cache_len - 1, i.e. the prediction for whatever token comes next.
            pending_logits = outputs.logits[:, -1, :]

            while len(gen_cpu) < target_len:
                # Tokens appended to gen_cpu THIS round; every one of them
                # must be yielded so streamed/collected text matches the
                # true token sequence (multi-token accept rounds included).
                new_tokens: list[int] = []

                # 1. Find a matching n-gram suffix in the prefix
                match = self._find_match(gen_cpu, min_match=4)
                draft_list: list[int] | None = None
                if match is not None:
                    match_end, num_draft = match
                    draft_list = gen_cpu[match_end:match_end + num_draft]

                if draft_list:
                    # 2. Verify draft tokens with a single cached forward
                    # pass over just the drafts.  Row i of the verify logits
                    # covers absolute position prefix_len + i and PREDICTS
                    # the token at prefix_len + i + 1 — so the opinion on
                    # draft i comes from one row earlier (see
                    # _accept_drafts), with draft 0 scored against the
                    # pre-verify pending prediction.
                    draft_ids = torch.tensor(
                        [draft_list], dtype=torch.long, device=device,
                    )
                    v_out = model(
                        draft_ids, use_cache=True, past_key_values=past,
                    )
                    verify_logits = v_out.logits
                    prefix_len = len(gen_cpu)

                    accepted = self._accept_drafts(
                        pending_logits, verify_logits, draft_ids, temperature,
                    )
                    # Respect the caller's token budget: drafts verified
                    # beyond the remaining allowance are treated as rejected
                    # (the crop below discards their KV entries), so the
                    # strategy never emits more than max_new_tokens.
                    accepted = min(accepted, target_len - prefix_len)

                    past = getattr(v_out, "past_key_values", None)
                    if accepted == len(draft_list):
                        # All drafts verified: keep the appended KV entries.
                        gen_cpu.extend(draft_list)
                        new_tokens.extend(draft_list)
                        pending_logits = verify_logits[:, -1, :]
                    else:
                        # Drop rejected tail's KV entries; keep accepted ones.
                        # ``accepted`` counts drafts whose prediction (one
                        # row earlier) matched, so the cache must retain
                        # exactly prefix_len + accepted entries and the
                        # correction row for the NEXT token is verify row
                        # accepted - 1 (it predicts position
                        # prefix_len + accepted).
                        if past is not None:
                            if hasattr(past, "crop"):
                                past.crop(prefix_len + accepted)
                            else:
                                # Bug C-04 fix: legacy tuple-of-tensors caches
                                # have no .crop(). Previously the guard above
                                # silently skipped truncation, so the KV cache
                                # retained the rejected tokens' entries while
                                # gen_cpu dropped them -> the model attended
                                # over rejected tokens and produced garbled
                                # text with no error. Slice explicitly.
                                keep = prefix_len + accepted
                                past = tuple(
                                    tuple(
                                        t[:, :, :keep, :] if t.dim() == 4 else t
                                        for t in layer
                                    )
                                    for layer in past
                                )
                        gen_cpu.extend(draft_list[:accepted])
                        new_tokens.extend(draft_list[:accepted])
                        if len(gen_cpu) < target_len:
                            # Budget remains for the replacement token.
                            if accepted > 0:
                                correction = verify_logits[:, accepted - 1, :]
                            else:
                                # Nothing accepted: prediction for position
                                # prefix_len still lives in the previous
                                # state's last-row logits.
                                correction = pending_logits
                            next_token = engine._token_gen.sample(
                                correction, temperature=temperature,
                                top_p=top_p, top_k=top_k,
                            )[0]
                            if next_token.dim() == 0:
                                next_token = next_token.unsqueeze(0)
                            if next_token.dim() == 1:
                                next_token = next_token.unsqueeze(-1)
                            tok_id = int(next_token.item())
                            gen_cpu.append(tok_id)
                            new_tokens.append(tok_id)
                            # Append the correction token to the KV cache.
                            c_out = model(
                                next_token, use_cache=True,
                                past_key_values=past,
                            )
                            past = getattr(c_out, "past_key_values", None)
                            pending_logits = c_out.logits[:, -1, :]
                else:
                    # 3. No draft match — standard single-token cached step.
                    next_token = engine._token_gen.sample(
                        pending_logits, temperature=temperature, top_p=top_p,
                        top_k=top_k,
                    )[0]
                    if next_token.dim() == 0:
                        next_token = next_token.unsqueeze(0)
                    if next_token.dim() == 1:
                        next_token = next_token.unsqueeze(-1)
                    tok_id = int(next_token.item())
                    gen_cpu.append(tok_id)
                    new_tokens.append(tok_id)
                    s_out = model(
                        next_token, use_cache=True, past_key_values=past,
                    )
                    past = getattr(s_out, "past_key_values", None)
                    pending_logits = s_out.logits[:, -1, :]

                for tok_id in new_tokens:
                    yield engine.tokenizer.decode(
                        [tok_id], skip_special_tokens=True,
                    )
                    if tok_id == engine.tokenizer.eos_token_id:
                        return

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Return the joined generation string (non-streaming contract).

        The per-token loop lives in ``_generate_tokens`` (a generator); this
        method collects it into a single string so the strategy honors the
        ``GenerationStrategy.generate -> str`` protocol (F-007 surfaced this
        mismatch — generate previously leaked a generator object).
        """
        return "".join(
            self._generate_tokens(
                prompt, max_new_tokens, temperature, top_p, top_k, **kwargs
            )
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        yield from self._generate_tokens(
            prompt, max_new_tokens, temperature, top_p, top_k, **kwargs
        )


# Backward-compatible alias (tests import the underscore name)
_PromptLookupStrategy = PromptLookupStrategy
