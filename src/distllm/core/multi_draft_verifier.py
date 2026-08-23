"""Multi-draft speculative decoding — verify multiple draft models simultaneously.

Sends candidates from N different draft models to the target model
in a single forward pass, accepting the first matching token from
any draft. This increases the acceptance rate by diversifying drafts.

Usage::

    verifier = MultiDraftVerifier(
        target_forward=target_fn,
        draft_forwards=[draft_a, draft_b, draft_c],
    )
    output = verifier.generate(input_ids, max_new_tokens=256)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger


class MultiDraftVerifier:
    """Speculative decoding with multiple draft models.

    Each draft model proposes candidates independently. The target
    model verifies all candidates in a single forward pass. The
    token with the highest target probability is accepted.

    This is useful when no single draft model dominates — different
    drafts excel at different token positions.
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_forwards: list[Callable],
        num_candidates_per_draft: int = 3,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        self._target = target_forward
        self._drafts = draft_forwards
        self._num_candidates = num_candidates_per_draft
        self._temperature = temperature
        self._device = torch.device(device)

        self._stats = {
            "target_calls": 0,
            "draft_calls": [0] * len(draft_forwards),
            "accepted_by_draft": [0] * len(draft_forwards),
            "total_proposed": 0,
        }

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        total_accepted = sum(s["accepted_by_draft"])
        s["total_accepted"] = total_accepted
        s["acceptance_rate"] = total_accepted / max(s["total_proposed"], 1)
        return s

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using multi-draft speculative decoding.

        Each draft model proposes an independent autoregressive chain of
        candidates. Each chain is verified against the target model in a
        separate forward pass with proper rejection sampling (min(1, p/q)).
        The chain with the most accepted tokens is chosen.

        This avoids the mathematically incorrect flattening of independent
        autoregressive chains into a single sequence for verification.
        """
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            if remaining <= 0:
                break

            num_draft = min(self._num_candidates, remaining)
            if num_draft <= 0:
                break

            # Phase 1: Generate an independent autoregressive chain from each draft model
            draft_chains: list[torch.Tensor] = []  # each is (1, num_draft)
            for draft_idx, draft_fn in enumerate(self._drafts):
                chain = self._draft_generate(draft_fn, generated, num_draft, **kwargs)
                draft_chains.append(chain)
                self._stats["draft_calls"][draft_idx] += 1

            self._stats["total_proposed"] += num_draft * len(self._drafts)

            # Phase 2: Verify each draft chain independently with the target model.
            # Each chain is conditioned on [generated, chain_tokens_so_far],
            # which is the correct autoregressive context for that chain.
            best_accepted = 0
            best_draft_idx = -1
            best_chain: torch.Tensor | None = None

            for draft_idx, chain in enumerate(draft_chains):
                # Build the full input: [prefix, candidate_chain]
                full_input = torch.cat([generated, chain], dim=1)
                target_logits = self._target(full_input, **kwargs)
                self._stats["target_calls"] += 1

                # Rejection-sample each token in the chain.
                # Draft token i occupies position prefix_len+i (prefix_len =
                # generated.shape[1]) and is predicted by logits at prefix_len+i-1
                # (logits[k] -> token[k+1]). F-046: prefix_len must be -1 here so
                # pos = prefix_len + i lands on the correct prediction.
                prefix_len = generated.shape[1] - 1
                accepted = 0
                for i in range(num_draft):
                    pos = prefix_len + i
                    if pos >= target_logits.shape[1]:
                        break

                    target_probs = F.softmax(
                        target_logits[:, pos, :] / self._temperature, dim=-1
                    )
                    token_id = chain[0, i].item()
                    p = target_probs[0, token_id].item()

                    if self._temperature == 0:
                        # Greedy: accept iff token equals target argmax
                        if target_logits[:, pos, :].argmax(dim=-1).item() != token_id:
                            break
                    else:
                        # Proper rejection sampling: compute draft probability q
                        # and accept with probability min(1, p/q).
                        # Get the draft logits at this position from the draft chain.
                        draft_input = torch.cat([generated, chain[:, :i + 1]], dim=1)
                        draft_logits = self._drafts[draft_idx](draft_input, **kwargs)
                        draft_logits = draft_logits[:, -1, :] if draft_logits.dim() > 2 else draft_logits
                        draft_probs = F.softmax(draft_logits / self._temperature, dim=-1)
                        q = draft_probs[0, token_id].item()

                        if q <= 0:
                            break  # Per spec decoding theory: reject when q=0
                        if torch.rand(1).item() >= p / q:
                            break  # Standard rejection sampling

                    accepted += 1

                if accepted > best_accepted:
                    best_accepted = accepted
                    best_draft_idx = draft_idx
                    best_chain = chain

            # Phase 3: Apply the best chain result
            if best_accepted > 0 and best_chain is not None:
                accepted_tokens = best_chain[:, :best_accepted]
                generated = torch.cat([generated, accepted_tokens], dim=1)
                if best_draft_idx >= 0:
                    self._stats["accepted_by_draft"][best_draft_idx] += best_accepted

            # Sample correction token from target (use the best chain's target logits)
            if best_draft_idx >= 0:
                full_input = torch.cat([generated[:, :-best_accepted] if best_accepted > 0 else generated,
                                        draft_chains[best_draft_idx]], dim=1)
                target_logits = self._target(full_input, **kwargs)
                if best_accepted < num_draft:
                    next_logits = target_logits[:, generated.shape[1] - 1, :]
                else:
                    next_logits = target_logits[:, -1, :]
            else:
                # No draft accepted; fallback to target-only sample
                target_logits = self._target(generated, **kwargs)
                next_logits = target_logits[:, -1, :]

            next_token = self._sample(next_logits)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _draft_generate(
        self,
        draft_fn: Callable,
        prefix: torch.Tensor,
        num_tokens: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate draft tokens from a single draft model."""
        draft_tokens = []
        current = prefix

        for _ in range(num_tokens):
            logits = draft_fn(current, **kwargs)
            next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
            token = self._sample(next_logits)
            draft_tokens.append(token)
            current = torch.cat([current, token], dim=1)

        return torch.cat(draft_tokens, dim=1) if draft_tokens else torch.empty(1, 0, dtype=torch.long)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class TreeMultiDraftVerifier:
    """Multi-draft verifier with tree-structured verification.

    Combines multiple draft models with tree-based candidate generation.
    Each draft model generates a tree of candidates (not just a chain).
    The target model verifies all trees in a single forward pass,
    accepting the longest matching prefix across all trees.

    This provides higher acceptance rates than chain-based verification
    because the tree explores multiple candidate paths simultaneously.

    Usage::

        verifier = TreeMultiDraftVerifier(
            target_forward=target_fn,
            draft_forwards=[draft_a, draft_b],
            branching_factor=3,
            depth=4,
        )
        output = verifier.generate(input_ids, max_new_tokens=256)
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_forwards: list[Callable],
        branching_factor: int = 3,
        depth: int = 4,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        from distllm.core.draft_tree import DraftTree

        self._target = target_forward
        self._trees = [
            DraftTree(
                draft_forward=fn,
                branching_factor=branching_factor,
                depth=depth,
                temperature=temperature,
                device=device,
            )
            for fn in draft_forwards
        ]
        self._temperature = temperature
        self._device = torch.device(device)

        self._stats = {
            "target_calls": 0,
            "draft_trees": len(draft_forwards),
            "total_accepted": 0,
            "total_proposed": 0,
        }

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        s["acceptance_rate"] = s["total_accepted"] / max(s["total_proposed"], 1)
        return s

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using tree-based multi-draft verification."""
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            if remaining <= 0:
                break

            # Phase 1: Generate draft trees from all draft models
            all_roots = []
            for tree in self._trees:
                root = tree.generate_tree(generated)
                all_roots.append(root)

            # Collect all candidate paths from all trees
            all_paths = []
            for root in all_roots:
                all_paths.extend(root.flatten())

            if not all_paths:
                break

            # Find the longest path for target verification
            longest_path = max(all_paths, key=len)
            self._stats["total_proposed"] += len(longest_path)

            # Phase 2: Verify with target model
            candidate_tensor = torch.tensor(
                [longest_path], dtype=generated.dtype, device=generated.device
            )
            full_input = torch.cat([generated, candidate_tensor], dim=1)
            target_logits = self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1

            # Phase 3: Verify each tree and find best result
            best_accepted = 0
            best_tokens: list[int] = []

            for root in all_roots:
                result = root.verify_tree(generated, root, target_logits)
                if result.accepted_count > best_accepted:
                    best_accepted = result.accepted_count
                    best_tokens = result.accepted_tokens

            if best_accepted > 0:
                accepted_tensor = torch.tensor(
                    [best_tokens], dtype=generated.dtype, device=generated.device
                )
                generated = torch.cat([generated, accepted_tensor], dim=1)
                self._stats["total_accepted"] += best_accepted

            # Sample correction token from target
            if best_accepted < len(longest_path):
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
