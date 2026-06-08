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
from typing import Any, Callable

import torch
import torch.nn.functional as F


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
    ) -> None:
        self._draft = draft_forward
        self._branching = branching_factor
        self._depth = depth
        self._temperature = temperature
        self._top_k = top_k
        self._device = torch.device(device)

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
            for i in range(self._branching):
                token_id = topk_indices[0, i].item()
                logprob = float(topk_values[0, i].item())
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

        for path in all_paths:
            accepted = 0
            for i, token_id in enumerate(path):
                pos = prefix_len + i - 1
                if pos >= target_logits.shape[1]:
                    break

                if self._temperature == 0:
                    target_token = target_logits[:, pos, :].argmax(dim=-1).item()
                    if target_token != token_id:
                        break
                else:
                    target_probs = F.softmax(
                        target_logits[:, pos, :] / self._temperature, dim=-1,
                    )
                    p = target_probs[0, token_id].item()
                    # M-14: Use deterministic RNG when seed is set for reproducibility
                    rng = torch.Generator()
                    if hasattr(self, '_seed') and self._seed is not None:
                        rng = torch.Generator().manual_seed(self._seed + accepted)
                    if torch.rand(1, generator=rng).item() >= p:
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
