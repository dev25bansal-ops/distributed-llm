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
        """Generate tokens using multi-draft speculative decoding."""
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            if remaining <= 0:
                break

            # Phase 1: Generate candidates from all draft models
            all_candidates = []
            draft_sources = []  # Which draft model each candidate came from

            for draft_idx, draft_fn in enumerate(self._drafts):
                num_draft = min(self._num_candidates, remaining)
                if num_draft <= 0:
                    break

                draft_tokens = self._draft_generate(
                    draft_fn, generated, num_draft, **kwargs
                )
                self._stats["draft_calls"][draft_idx] += 1

                for i in range(draft_tokens.shape[1]):
                    all_candidates.append(draft_tokens[0, i].item())
                    draft_sources.append(draft_idx)

            if not all_candidates:
                break

            self._stats["total_proposed"] += len(all_candidates)

            # Phase 2: Verify all candidates with target model
            # Build input with the first candidate path
            candidate_tensor = torch.tensor(
                [all_candidates], dtype=generated.dtype, device=generated.device
            )
            full_input = torch.cat([generated, candidate_tensor], dim=1)

            target_logits = self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1

            # Phase 3: Find the best matching candidate
            accepted_count = 0
            best_draft_idx = -1

            prefix_len = generated.shape[1] - 1
            for i, (candidate_token, draft_idx) in enumerate(
                zip(all_candidates, draft_sources)
            ):
                if prefix_len + i >= target_logits.shape[1]:
                    break

                target_probs = F.softmax(
                    target_logits[:, prefix_len + i, :] / self._temperature, dim=-1
                )
                target_prob = target_probs[0, candidate_token].item()

                if target_prob > 0.3:  # Accept threshold
                    accepted_count += 1
                    best_draft_idx = draft_idx
                else:
                    break

            if accepted_count > 0:
                accepted_tokens = candidate_tensor[:, :accepted_count]
                generated = torch.cat([generated, accepted_tokens], dim=1)
                if best_draft_idx >= 0:
                    self._stats["accepted_by_draft"][best_draft_idx] += accepted_count

            # Sample correction token from target
            if accepted_count < len(all_candidates):
                next_logits = target_logits[:, generated.shape[1] - 1, :]
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
