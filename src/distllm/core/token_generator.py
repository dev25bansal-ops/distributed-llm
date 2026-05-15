"""Token generator for distributed LLM inference.

Handles token sampling, constraint application, and generation loops.
Extracted from the Coordinator class.
"""

from typing import Optional, Any, List

import torch

from distllm.core.structured_output import JSONSchemaConstraint


class TokenGenerator:
    """Handles token sampling and generation.

    Attributes:
        tokenizer: The tokenizer for encoding/decoding.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).
            temperature: Sampling temperature. 0 means argmax.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling. Only the top_k most likely tokens are considered. 0 means disabled.

        Returns:
            Sampled token IDs.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            # Apply top-k filtering before top-p
            if top_k > 0:
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                for i in range(probs.shape[0]):
                    top_k_indices = torch.topk(probs[i], top_k, dim=-1).indices
                    batch_indices = torch.arange(probs.shape[1], device=probs.device)
                    row_mask = ~torch.isin(batch_indices, top_k_indices)
                    indices_to_remove[i, row_mask] = True
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            return torch.multinomial(probs, 1).squeeze(0)
        else:
            return torch.argmax(logits, dim=-1)

    def sample_batch(
        self,
        logits: torch.Tensor,
        sequences: List[Any],
        tokenizer=None,
    ) -> torch.Tensor:
        """Sample next tokens for a batch, applying constraints per sequence.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).
            sequences: List of Sequence objects with optional constraints.
            tokenizer: Tokenizer for constraint mask generation.

        Returns:
            Stacked sampled token IDs.
        """
        tok = tokenizer or self.tokenizer
        batch_size = logits.shape[0]
        next_tokens_list = []

        for i, seq in enumerate(sequences):
            seq_logits = logits[i:i+1, :]

            # Apply constraint mask
            if seq.constraint is not None:
                mask = seq.constraint.get_logits_mask(seq_logits.shape[-1], tok)
                seq_logits = seq_logits.masked_fill(~mask, float('-inf'))

            token = self.sample(seq_logits, temperature=seq.temperature, top_p=seq.top_p, top_k=seq.top_k)
            next_tokens_list.append(token)

        return torch.stack(next_tokens_list).squeeze(-1)

    def apply_constraint(
        self,
        logits: torch.Tensor,
        constraint: Optional[JSONSchemaConstraint],
        tokenizer=None,
    ) -> torch.Tensor:
        """Apply a constraint mask to logits.

        Args:
            logits: Logits tensor.
            constraint: Optional constraint to apply.
            tokenizer: Tokenizer for mask generation.

        Returns:
            Constrained logits tensor.
        """
        if constraint is None:
            return logits

        tok = tokenizer or self.tokenizer
        mask = constraint.get_logits_mask(logits.shape[-1], tok)
        return logits.masked_fill(~mask, float('-inf'))
