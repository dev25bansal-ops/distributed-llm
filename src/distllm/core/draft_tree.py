"""Tree-based Speculative Decoding — draft multiple candidate trees.

Instead of generating a single chain of draft tokens, this module
generates a *tree* of candidates at each position.  The target model
verifies the entire tree in a single forward pass, accepting the
longest matching prefix.

This is inspired by Medusa and EAGLE approaches but designed to work
with remote draft models as well as local draft heads.

Usage::

    tree = DraftTree(
        draft_forward=draft_fn,
        branching_factor=3,
        depth=4,
    )
    candidates = tree.generate_tree(prefix_tokens)
    accepted = tree.verify_tree(prefix_tokens, candidates, target_logits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import torch
import torch.nn.functional as F

# Shared speculative-verification kernel (single source of truth for
# acceptance math). Imported here so the AST lint test (M-02) can prove no
# decoder reimplements rejection sampling.
from distllm.core.spec_verify import accept_token


@dataclass
class TreeNode:
    """A single node in the draft tree."""
    token_id: int
    logprob: float = 0.0
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flatten(self) -> list[list[int]]:
        """Return all root-to-leaf paths."""
        if self.is_leaf:
            return [[self.token_id]]
        paths = []
        for child in self.children:
            for path in child.flatten():
                paths.append([self.token_id] + path)
        return paths


@dataclass
class TreeVerificationResult:
    """Result of verifying a draft tree against target logits."""
    accepted_tokens: list[int]
    accepted_count: int
    best_path: list[int]
    total_candidates: int


class DraftTree:
    """Tree-based draft generation for speculative decoding.

    Generates a tree of candidate tokens where each position can have
    multiple branches (``branching_factor``).  The tree is verified
    breadth-first against target model logits.

    Args:
        draft_forward: Callable ``(input_ids) -> logits`` for draft model.
        branching_factor: Number of candidate tokens per position.
        depth: Maximum tree depth (draft length per step).
        temperature: Sampling temperature for draft generation.
        top_k: Top-k sampling for draft generation.
        device: Torch device.
    """

    def __init__(
        self,
        draft_forward: Callable[..., Any],
        branching_factor: int = 3,
        depth: int = 4,
        temperature: float = 1.0,
        top_k: int = 20,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self._draft = draft_forward
        self._branching = branching_factor
        self._depth = depth
        self._temperature = temperature
        self._top_k = top_k
        self._device = torch.device(device)
        # Bug M-03 fix: `_seed` was never set, so `getattr(self, '_seed')`
        # was always None and the deterministic-RNG path in verify_tree was
        # dead code. Initialize it (optionally overridable) so seeded
        # verification actually works.
        self._seed = seed

    def generate_tree(
        self,
        prefix: torch.Tensor,
    ) -> TreeNode:
        """Generate a draft tree from the prefix.

        Returns the root ``TreeNode`` whose ``flatten()`` method
        yields all root-to-leaf candidate sequences.
        """
        root = TreeNode(token_id=0, depth=0)
        self._expand_node(root, prefix, depth=0)
        return root

    def _expand_node(
        self,
        node: TreeNode,
        context: torch.Tensor,
        depth: int,
    ) -> None:
        """Recursively expand a tree node by generating top-K children."""
        if depth >= self._depth:
            return

        logits = self._draft(context)
        next_logits = logits[:, -1, :] if logits.dim() > 2 else logits

        if self._temperature == 0:
            # Greedy: take top-K
            topk_values, topk_indices = torch.topk(
                next_logits, self._branching, dim=-1,
            )
            # O-11: store a true log-probability, not the raw logit. The raw
            # topk value is a logit (unbounded, often positive); any consumer
            # doing exp(logprob) or ranking across modes got meaningless
            # numbers. log_softmax makes greedy and sampled modes share one
            # semantic (matches tree_speculative_decoder.py).
            log_probs = F.log_softmax(next_logits, dim=-1)
            for i in range(self._branching):
                token_id = topk_indices[0, i].item()
                logprob = float(log_probs[0, token_id].item())
                child = TreeNode(
                    token_id=token_id, logprob=logprob, depth=depth + 1,
                )
                node.children.append(child)

                # Recursively expand child
                child_context = torch.cat(
                    [context, torch.tensor([[token_id]], device=context.device)],
                    dim=1,
                )
                self._expand_node(child, child_context, depth + 1)
        else:
            # Sample branching_factor tokens
            if self._top_k > 0:
                values, indices = torch.topk(next_logits, self._top_k, dim=-1)
                next_logits = torch.full_like(
                    next_logits, float("-inf"),
                ).scatter_(-1, indices, values)

            probs = F.softmax(next_logits / self._temperature, dim=-1)

            # Sample without replacement
            for _ in range(min(self._branching, probs.shape[-1])):
                token_id = torch.multinomial(probs, num_samples=1).item()
                logprob = float(torch.log(probs[0, token_id]).item())
                child = TreeNode(
                    token_id=token_id, logprob=logprob, depth=depth + 1,
                )
                node.children.append(child)

                # Zero out the sampled token for next iteration
                probs[0, token_id] = 0.0
                if probs.sum() > 0:
                    probs = probs / probs.sum()

                # Recursively expand child
                child_context = torch.cat(
                    [context, torch.tensor([[token_id]], device=context.device)],
                    dim=1,
                )
                self._expand_node(child, child_context, depth + 1)

    def verify_tree(
        self,
        prefix: torch.Tensor,
        root: TreeNode,
        target_logits: torch.Tensor,
    ) -> TreeVerificationResult:
        """Verify a draft tree against target model logits.

        Performs breadth-first verification: checks each depth level
        against the target distribution.  Returns the longest accepted
        prefix across all branches.

        Args:
            prefix: Original prefix tokens, shape ``(1, seq_len)``.
            root: Root of the draft tree.
            target_logits: Target model logits for ``prefix + all_candidates``.

        Returns:
            ``TreeVerificationResult`` with the accepted tokens.
        """
        prefix_len = prefix.shape[1]
        all_paths = root.flatten()
        total_candidates = sum(len(p) for p in all_paths)

        best_accepted = 0
        best_path: list[int] = []

        # Bug H-02 fix: the greedy branch previously did a full-vocab
        # argmax + .item() sync PER TOKEN PER PATH (3^4 tree = 81 paths ->
        # ~324 syncs per round, >3 ms of pure GPU->CPU sync). Precompute
        # the argmax row ONCE per logits row and reuse across all paths
        # that touch that row — an 81x reduction in sync count for greedy.
        greedy_argmax: dict[int, int] = {}
        if self._temperature == 0:
            max_row = target_logits.shape[1]
            needed_rows = {prefix_len + i - 1
                          for path in all_paths
                          for i in range(len(path))
                          if prefix_len + i - 1 < max_row}
            for row in needed_rows:
                greedy_argmax[row] = int(target_logits[:, row, :].argmax(dim=-1).item())

        for path in all_paths:
            accepted = 0
            for i, token_id in enumerate(path):
                pos = prefix_len + i - 1
                if pos >= target_logits.shape[1]:
                    break

                if self._temperature == 0:
                    # Batched argmax (see H-02 note above).
                    target_token = greedy_argmax.get(pos)
                    if target_token != token_id:
                        break
                else:
                    # Sampled mode: delegate to the shared verification kernel
                    # for true rejection sampling min(1, p/q) (bug C-03 fix).
                    # The draft probability is not tracked per-tree-node, so
                    # the uniform-draft fallback inside accept_token applies.
                    # M-14: deterministic RNG when a seed is set.
                    rng = torch.Generator()
                    if getattr(self, '_seed', None) is not None:
                        rng = torch.Generator().manual_seed(self._seed + accepted)
                    if not accept_token(
                        target_logits,
                        pos,
                        int(token_id),
                        draft_prob=None,
                        temperature=self._temperature,
                        vocab_size=target_logits.shape[-1],
                        rng=rng,
                    ):
                        break

                accepted += 1

            if accepted > best_accepted:
                best_accepted = accepted
                best_path = path[:accepted]

        return TreeVerificationResult(
            accepted_tokens=best_path,
            accepted_count=best_accepted,
            best_path=best_path,
            total_candidates=total_candidates,
        )
