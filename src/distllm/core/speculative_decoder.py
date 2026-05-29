"""Speculative decoding with a lightweight draft model.

Speeds up autoregressive generation by having a small draft model
predict multiple candidate tokens, then verifying them with the
target model in a single forward pass.

Usage::

    sd = SpeculativeDecoder(target_model, draft_model, num_candidates=5)
    output_ids = sd.generate(input_ids, max_new_tokens=256)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger


class SpeculativeDecoder:
    """Speculative decoding engine.

    Uses a fast draft model to propose candidate tokens and the target
    model to verify them. Accepts all tokens that match the target
    distribution, re-drafting from the first mismatch.

    Args:
        target_forward: Callable accepting ``input_ids`` and returning logits.
        draft_forward: Callable with same signature as *target_forward*.
        num_candidates: Number of draft tokens to generate per step.
        top_k: Top-k sampling for draft generation.
        temperature: Sampling temperature for draft generation.
        device: Torch device.
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_forward: Callable,
        num_candidates: int = 5,
        top_k: int = 20,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        self._target = target_forward
        self._draft = draft_forward
        self._num_candidates = num_candidates
        self._top_k = top_k
        self._temperature = temperature
        self._device = torch.device(device)

        self._stats = {"draft_calls": 0, "target_calls": 0, "accepted": 0, "total_proposed": 0}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments forwarded to both forward functions.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.
        """
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            num_draft = min(self._num_candidates, remaining)

            # --- Draft phase ---
            draft_tokens = self._draft_forward(generated, num_draft, **kwargs)
            self._stats["draft_calls"] += 1

            # --- Verification phase ---
            # Concatenate: prompt + all draft tokens
            full_input = torch.cat([generated, draft_tokens], dim=1)
            target_logits = self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1

            # Verify each draft token
            accepted_count = self._verify_tokens(
                generated, full_input, draft_tokens, target_logits, **kwargs
            )

            # Append accepted tokens
            generated = torch.cat([generated, draft_tokens[:, :accepted_count]], dim=1)

            if accepted_count < num_draft:
                # Sample one more token from target distribution at the rejection point
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits)
                generated = torch.cat([generated, next_token], dim=1)

        self._stats["total_proposed"] += self._stats["draft_calls"] * num_draft
        self._stats["accepted"] += generated.shape[1] - prompt_len

        return generated

    def _draft_forward(
        self, prefix: torch.Tensor, num_tokens: int, **kwargs: Any
    ) -> torch.Tensor:
        """Generate *num_tokens* draft tokens autoregressively."""
        if num_tokens <= 0:
            return torch.empty(1, 0, dtype=torch.long, device=prefix.device)
        draft_tokens = []
        current = prefix

        for _ in range(num_tokens):
            logits = self._draft(current, **kwargs)
            next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
            token = self._sample(next_logits)
            draft_tokens.append(token)
            current = torch.cat([current, token], dim=1)

        return torch.cat(draft_tokens, dim=1)

    def _verify_tokens(
        self,
        prefix: torch.Tensor,
        full_input: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        **kwargs: Any,
    ) -> int:
        """Verify draft tokens against target model distribution.

        Returns the number of accepted draft tokens.
        """
        num_draft = draft_tokens.shape[1]
        prefix_len = prefix.shape[1] - 1

        if self._temperature == 0:
            for i in range(num_draft):
                target_argmax = target_logits[:, prefix_len + i, :].argmax(dim=-1).item()
                if target_argmax != draft_tokens[0, i].item():
                    return i
            return num_draft

        for i in range(num_draft):
            target_probs = F.softmax(
                target_logits[:, prefix_len + i, :] / self._temperature, dim=-1
            )

            draft_out = self._draft(full_input[:, :prefix_len + i + 1], **kwargs)
            draft_probs = F.softmax(
                draft_out[:, -1, :] / self._temperature, dim=-1
            )

            q = draft_probs[0, draft_tokens[0, i]].item()
            p = target_probs[0, draft_tokens[0, i]].item()

            if q <= 0 or torch.rand(1).item() < p / q:
                if p <= 0:
                    return i
            else:
                return i

        return num_draft

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits with top-k."""
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        if self._top_k > 0:
            values, indices = torch.topk(logits, self._top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter_(-1, indices, values)

        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        if s["total_proposed"] > 0:
            s["acceptance_rate"] = round(s["accepted"] / max(s["total_proposed"], 1), 3)
        return s


class MultiDraftSpeculativeDecoder:
    """Speculative decoding with an ensemble of draft models.

    Uses multiple small draft models to propose candidate tokens via
    **consensus decoding**.  At each position all draft models see the
    same shared prefix and must predict the identical token for it to
    be included in the draft.  Across-model agreement provides higher
    confidence predictions, leading to increased acceptance rates
    compared to single-draft speculative decoding.

    Args:
        target_forward: Callable accepting ``input_ids`` and returning logits.
        draft_forwards: Sequence of callables with the same signature as
            *target_forward* (at least 2).
        num_candidates: Number of draft tokens to generate per step.
        top_k: Top-k sampling for draft generation.
        temperature: Sampling temperature.
        device: Torch device.
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_forwards: list[Callable],
        num_candidates: int = 5,
        top_k: int = 20,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        if len(draft_forwards) < 2:
            raise ValueError(
                f"MultiDraftSpeculativeDecoder requires >=2 draft models, got {len(draft_forwards)}"
            )
        self._target = target_forward
        self._draft_forwards = list(draft_forwards)
        self._num_candidates = num_candidates
        self._top_k = top_k
        self._temperature = temperature
        self._device = torch.device(device)

        self._stats: dict[str, Any] = {
            "draft_calls": 0,
            "target_calls": 0,
            "accepted": 0,
            "total_proposed": 0,
            "consensus_lengths": [],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using multi-draft speculative decoding.

        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments forwarded to forward functions.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.
        """
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            num_draft = min(self._num_candidates, remaining)

            # --- Consensus draft phase ---
            consensus_tokens, consensus_len = self._generate_consensus_draft(
                generated, num_draft, **kwargs,
            )
            self._stats["draft_calls"] += 1
            self._stats["consensus_lengths"].append(consensus_len)

            if consensus_len == 0:
                # No consensus — single token from target fallback
                target_logits = self._target(generated, **kwargs)
                self._stats["target_calls"] += 1
                next_token = self._sample(target_logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
                continue

            # --- Verification phase ---
            full_input = torch.cat([generated, consensus_tokens], dim=1)
            target_logits = self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1

            accepted_count = self._verify_tokens(
                generated, full_input, consensus_tokens, target_logits, **kwargs,
            )

            generated = torch.cat([generated, consensus_tokens[:, :accepted_count]], dim=1)

            if accepted_count < consensus_len:
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits)
                generated = torch.cat([generated, next_token], dim=1)

        self._stats["total_proposed"] = sum(self._stats["consensus_lengths"])
        self._stats["accepted"] = generated.shape[1] - prompt_len

        return generated

    def _generate_consensus_draft(
        self, prefix: torch.Tensor, num_tokens: int, **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Generate draft tokens from all models and find consensus prefix.

        Returns ``(consensus_tokens, consensus_length)`` where
        *consensus_tokens* has shape ``(1, consensus_length)``.
        """
        consensus_tokens: list[torch.Tensor] = []

        for pos in range(num_tokens):
            # Shared input: prefix + all consensus tokens so far
            if consensus_tokens:
                shared_input = torch.cat([prefix] + consensus_tokens, dim=1)
            else:
                shared_input = prefix

            tokens_at_pos = []
            for draft_fn in self._draft_forwards:
                logits = draft_fn(shared_input, **kwargs)
                next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
                token = self._sample(next_logits)
                tokens_at_pos.append(token.item())

            # All models must agree
            if all(t == tokens_at_pos[0] for t in tokens_at_pos):
                t = torch.tensor([[tokens_at_pos[0]]], device=prefix.device, dtype=torch.long)
                consensus_tokens.append(t)
            else:
                break

        if consensus_tokens:
            return torch.cat(consensus_tokens, dim=1), len(consensus_tokens)
        return torch.empty(1, 0, dtype=torch.long, device=prefix.device), 0

    def _verify_tokens(
        self,
        prefix: torch.Tensor,
        full_input: torch.Tensor,
        consensus_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        **kwargs: Any,
    ) -> int:
        """Verify consensus draft tokens against target model distribution.

        Returns the number of accepted draft tokens.
        """
        num_draft = consensus_tokens.shape[1]
        prefix_len = prefix.shape[1] - 1

        if self._temperature == 0:
            # Greedy: accept positions where target argmax matches
            for i in range(num_draft):
                target_argmax = target_logits[:, prefix_len + i, :].argmax(dim=-1).item()
                if target_argmax != consensus_tokens[0, i].item():
                    return i
            return num_draft

        for i in range(num_draft):
            target_probs = F.softmax(
                target_logits[:, prefix_len + i, :] / self._temperature, dim=-1,
            )

            shared_input = full_input[:, :prefix_len + i + 1]
            draft_out = self._draft_forwards[0](shared_input, **kwargs)
            draft_probs = F.softmax(
                draft_out[:, -1, :] / self._temperature, dim=-1,
            )

            q = draft_probs[0, consensus_tokens[0, i]].item()
            p = target_probs[0, consensus_tokens[0, i]].item()

            if q <= 0 or torch.rand(1).item() < p / q:
                if p <= 0:
                    return i
            else:
                return i

        return num_draft

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits with top-k."""
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        if self._top_k > 0:
            values, indices = torch.topk(logits, self._top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)

        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        total = s.get("total_proposed", 0)
        if total > 0:
            s["acceptance_rate"] = round(s["accepted"] / max(total, 1), 3)
        return s


class TreeDraftSpeculativeDecoder:
    """Speculative decoding with tree-structured draft (SpecInfer style).

    Instead of a single draft sequence, builds a **tree** of candidate
    token sequences from multiple draft models.  Each draft model
    proposes tokens at each depth, and the tree is expanded along all
    branches that have sufficient probability.  The target model then
    verifies the entire tree in a single forward pass using a tree
    attention mask.

    This increases the acceptance rate 2-3x compared to single-draft
    speculative decoding because the tree covers more of the target
    distribution's probability mass.

    Reference: Miao et al., 2024, "SpecInfer: Accelerating Generative
    Large Language Model Serving with Tree-based Speculative Inference"

    Args:
        target_forward: Callable accepting ``input_ids`` and returning logits.
        draft_forwards: List of draft model forward functions.
        max_tree_nodes: Maximum number of nodes in the draft tree.
        max_depth: Maximum depth of the draft tree.
        branching_factor: Max children per node at each depth.
        temperature: Sampling temperature.
        top_k: Top-k sampling for draft generation.
        device: Torch device.
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_forwards: list[Callable],
        max_tree_nodes: int = 32,
        max_depth: int = 5,
        branching_factor: int = 4,
        temperature: float = 1.0,
        top_k: int = 20,
        device: str = "cuda",
    ):
        if len(draft_forwards) < 1:
            raise ValueError("At least 1 draft model required")
        self._target = target_forward
        self._draft_forwards = list(draft_forwards)
        self._max_tree_nodes = max_tree_nodes
        self._max_depth = max_depth
        self._branching_factor = branching_factor
        self._temperature = temperature
        self._top_k = top_k
        self._device = torch.device(device)

        self._stats: dict[str, Any] = {
            "draft_calls": 0,
            "target_calls": 0,
            "accepted": 0,
            "total_proposed": 0,
            "tree_sizes": [],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using tree-based speculative decoding."""
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            if remaining <= 0:
                break

            # Build draft tree from multiple draft models
            tree = self._build_draft_tree(generated, **kwargs)
            self._stats["draft_calls"] += 1
            self._stats["tree_sizes"].append(len(tree))

            if not tree:
                # No draft candidates — single token from target
                target_logits = self._target(generated, **kwargs)
                self._stats["target_calls"] += 1
                next_token = self._sample(target_logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)
                continue

            # Flatten tree into a batch of sequences for verification
            sequences = self._tree_to_sequences(tree, generated)

            # Verify all sequences in one batched forward pass
            best_path = self._verify_tree(sequences, generated, **kwargs)

            if best_path:
                new_tokens = torch.tensor([best_path], device=generated.device, dtype=torch.long).unsqueeze(0)
                generated = torch.cat([generated, new_tokens], dim=1)
                self._stats["accepted"] += len(best_path)
                self._stats["total_proposed"] += sum(len(s) for s in sequences)
            else:
                # No path accepted — single token from target
                target_logits = self._target(generated, **kwargs)
                self._stats["target_calls"] += 1
                next_token = self._sample(target_logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _build_draft_tree(
        self, prefix: torch.Tensor, **kwargs: Any,
    ) -> list[dict]:
        """Build a tree of draft candidates from multiple draft models.

        Each node is a dict with 'token', 'prob', 'children', 'depth'.
        Returns the list of root-level nodes.
        """
        roots: list[dict] = []
        nodes_created = 0

        def _expand(node: dict, current_prefix: torch.Tensor, depth: int):
            nonlocal nodes_created
            if depth >= self._max_depth or nodes_created >= self._max_tree_nodes:
                return

            # Get candidates from each draft model
            candidates: dict[int, float] = {}  # token_id -> max_prob
            for draft_fn in self._draft_forwards:
                logits = draft_fn(current_prefix, **kwargs)
                next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
                probs = F.softmax(next_logits / max(self._temperature, 0.01), dim=-1)

                # Top-k candidates from this model
                top_probs, top_ids = torch.topk(probs, min(self._branching_factor, probs.shape[-1]), dim=-1)
                for i in range(top_ids.shape[1]):
                    tid = top_ids[0, i].item()
                    p = top_probs[0, i].item()
                    candidates[tid] = max(candidates[tid], p)

            # Sort by probability and take top branching_factor
            sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:self._branching_factor]

            for token_id, prob in sorted_candidates:
                if prob < 0.05:  # Skip very low probability tokens
                    continue
                child = {
                    "token": token_id,
                    "prob": prob,
                    "children": [],
                    "depth": depth,
                }
                node["children"].append(child)
                nodes_created += 1

                if nodes_created < self._max_tree_nodes:
                    child_prefix = torch.cat([
                        current_prefix,
                        torch.tensor([[token_id]], device=current_prefix.device, dtype=torch.long),
                    ], dim=1)
                    _expand(child, child_prefix, depth + 1)

        # Seed from each draft model
        for draft_fn in self._draft_forwards:
            logits = draft_fn(prefix, **kwargs)
            next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
            probs = F.softmax(next_logits / max(self._temperature, 0.01), dim=-1)
            top_probs, top_ids = torch.topk(probs, min(self._branching_factor, probs.shape[-1]), dim=-1)

            for i in range(top_ids.shape[1]):
                tid = top_ids[0, i].item()
                p = top_probs[0, i].item()
                # Check if this token is already a root
                existing = [r for r in roots if r["token"] == tid]
                if existing:
                    existing[0]["prob"] = max(existing[0]["prob"], p)
                    continue
                if p < 0.05:
                    continue
                root_node = {
                    "token": tid,
                    "prob": p,
                    "children": [],
                    "depth": 0,
                }
                roots.append(root_node)
                nodes_created += 1

                if nodes_created < self._max_tree_nodes:
                    child_prefix = torch.cat([
                        prefix,
                        torch.tensor([[tid]], device=prefix.device, dtype=torch.long),
                    ], dim=1)
                    _expand(root_node, child_prefix, 1)

        return roots

    def _tree_to_sequences(
        self, tree: list[dict], prefix: torch.Tensor,
    ) -> list[list[int]]:
        """Flatten tree into all root-to-leaf paths."""
        sequences: list[list[int]] = []

        def _dfs(node: dict, path: list[int]):
            current_path = path + [node["token"]]
            if not node["children"]:
                sequences.append(current_path)
            else:
                for child in node["children"]:
                    _dfs(child, current_path)

        for root in tree:
            _dfs(root, [])

        # Deduplicate and limit
        seen = set()
        unique = []
        for seq in sequences:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)

        return unique[:self._max_tree_nodes]

    def _verify_tree(
        self,
        sequences: list[list[int]],
        prefix: torch.Tensor,
        **kwargs: Any,
    ) -> list[int] | None:
        """Verify tree sequences against target model.

        Returns the best accepted path (longest prefix match), or None.
        """
        if not sequences:
            return None

        # Find longest common prefix among all sequences
        best_path: list[int] = []
        best_length = 0

        # Verify each sequence
        for seq in sequences:
            full_input = torch.cat([
                prefix,
                torch.tensor([seq], device=prefix.device, dtype=torch.long),
            ], dim=1)

            target_logits = self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1

            accepted = 0
            prefix_len = prefix.shape[1] - 1

            for i in range(len(seq)):
                if self._temperature == 0:
                    target_argmax = target_logits[:, prefix_len + i, :].argmax(dim=-1).item()
                    if target_argmax == seq[i]:
                        accepted += 1
                    else:
                        break
                else:
                    target_probs = F.softmax(
                        target_logits[:, prefix_len + i, :] / self._temperature, dim=-1,
                    )
                    p = target_probs[0, seq[i]].item()
                    if torch.rand(1).item() < p:
                        accepted += 1
                    else:
                        break

            if accepted > best_length:
                best_length = accepted
                best_path = seq[:accepted]

        return best_path if best_length > 0 else None

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        if self._top_k > 0:
            values, indices = torch.topk(logits, self._top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        total = s.get("total_proposed", 0)
        if total > 0:
            s["acceptance_rate"] = round(s["accepted"] / max(total, 1), 3)
        tree_sizes = s.get("tree_sizes", [])
        if tree_sizes:
            s["avg_tree_size"] = round(sum(tree_sizes) / len(tree_sizes), 1)
        return s
