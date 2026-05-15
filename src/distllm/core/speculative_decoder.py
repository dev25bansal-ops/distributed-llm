"""Speculative decoding: draft token generation and verification."""

import torch
from typing import List, Tuple, Optional
from loguru import logger


class SpeculativeDecoder:
    """Manages speculative decoding with draft token generation and verification."""

    def __init__(
        self,
        num_assistant_tokens: int = 5,
        min_acceptance_rate: float = 0.3,
        warmup_steps: int = 10,
    ):
        self.num_assistant_tokens = num_assistant_tokens
        self.min_acceptance_rate = min_acceptance_rate
        self.warmup_steps = warmup_steps

        # Acceptance rate tracking (EMA)
        self._acceptance_rate: float = 1.0
        self._total_draft_tokens: int = 0
        self._total_accepted: int = 0
        self._step_count: int = 0
        self._enabled: bool = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate

    def generate_draft_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
    ) -> Tuple[List[int], Optional[List]]:
        """Generate draft tokens using the draft model.

        Args:
            draft_model: The draft model (AutoModelForCausalLM or similar)
            input_ids: Current token IDs [batch, seq_len]
            past_key_values: Optional KV cache from previous steps

        Returns:
            (draft_tokens, updated_kv_cache)
        """
        with torch.no_grad():
            if past_key_values is not None:
                outputs = draft_model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = draft_model(input_ids, use_cache=True)

        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        draft_token = torch.argmax(logits, dim=-1).item()
        new_kv = outputs.past_key_values

        # Generate remaining tokens autoregressively
        draft_tokens = [draft_token]
        current_token = torch.tensor([[draft_token]], device=input_ids.device)

        for _ in range(self.num_assistant_tokens - 1):
            with torch.no_grad():
                outputs = draft_model(
                    current_token,
                    past_key_values=new_kv,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            current_token = torch.argmax(logits, dim=-1, keepdim=True)
            draft_tokens.append(current_token.item())
            new_kv = outputs.past_key_values

        return draft_tokens, new_kv

    def verify_and_accept(
        self,
        draft_tokens: List[int],
        target_logits: torch.Tensor,
        tokenizer,
    ) -> Tuple[int, List[int], int]:
        """Verify draft tokens against target model logits.

        Args:
            draft_tokens: List of draft token IDs to verify
            target_logits: Target model logits [batch, seq_len, vocab]
                or [batch, vocab] for single-step verification

        Returns:
            (accepted_count, accepted_tokens, next_token_id)
        """
        if not draft_tokens:
            # No draft tokens, just sample from target
            # Handle both [batch, vocab] and [batch, seq_len, vocab]
            if target_logits.dim() == 2:
                next_token = self._sample_token(target_logits)
            else:
                next_token = self._sample_token(target_logits[:, -1, :])
            return 0, [], next_token.item()

        accepted_count = 0
        accepted_tokens = []

        # Handle single-step case (target returns only next token logits)
        if target_logits.dim() == 2 or target_logits.shape[1] == 1:
            # Single token verification - use argmax for deterministic comparison
            if target_logits.dim() == 2:
                target_token = torch.argmax(target_logits, dim=-1)
            else:
                target_token = torch.argmax(target_logits[:, -1, :], dim=-1)
            if target_token.item() == draft_tokens[0]:
                accepted_count = 1
                accepted_tokens = [draft_tokens[0]]
                next_token = target_token.item()
            else:
                next_token = target_token.item()
            return accepted_count, accepted_tokens, next_token

        # Multi-token verification: target_logits [batch, seq_len, vocab]
        for i, draft_token in enumerate(draft_tokens):
            if i >= target_logits.shape[1]:
                break

            target_logits_at_pos = target_logits[:, i, :]  # [batch, vocab]
            target_token = torch.argmax(target_logits_at_pos, dim=-1)

            if target_token.item() == draft_token:
                accepted_count += 1
                accepted_tokens.append(draft_token)
            else:
                # Mismatch: use target's token instead
                accepted_tokens.append(target_token.item())
                break

        # Determine next token after accepted prefix
        if accepted_count < len(draft_tokens):
            # Rejection happened; next token is already in accepted_tokens
            next_token = accepted_tokens[-1] if accepted_tokens else self._sample_token(target_logits[:, -1, :]).item()
        else:
            # All draft tokens accepted; need target model to generate next
            last_accepted_pos = len(draft_tokens) - 1
            if last_accepted_pos + 1 < target_logits.shape[1]:
                next_token = torch.argmax(target_logits[:, last_accepted_pos + 1, :], dim=-1).item()
            else:
                # Need another forward pass; return sentinel
                next_token = -1

        # Update acceptance rate tracking
        self._record_acceptance(len(draft_tokens), accepted_count)

        return accepted_count, accepted_tokens, next_token

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample token from logits."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _record_acceptance(self, total: int, accepted: int):
        """Update acceptance rate tracking."""
        self._total_draft_tokens += total
        self._total_accepted += accepted
        self._step_count += 1

        if self._step_count >= self.warmup_steps:
            # Running average
            self._acceptance_rate = self._total_accepted / max(self._total_draft_tokens, 1)

            # Auto-disable if acceptance rate too low
            if self._acceptance_rate < self.min_acceptance_rate:
                logger.warning(
                    f"Speculative decoding auto-disabled: "
                    f"acceptance rate {self._acceptance_rate:.2f} < {self.min_acceptance_rate}"
                )
                self._enabled = False

    def get_metrics(self) -> dict:
        """Get speculative decoding metrics."""
        return {
            "acceptance_rate": self._acceptance_rate,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "step_count": self._step_count,
            "enabled": self._enabled,
        }

    def reset(self):
        """Reset acceptance rate tracking."""
        self._acceptance_rate = 1.0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._step_count = 0
        self._enabled = True

    def generate_batch_draft_tokens(
        self,
        draft_model,
        input_ids_list: List[torch.Tensor],
        past_key_values_list: Optional[List] = None,
    ) -> Tuple[List[List[int]], Optional[List]]:
        """Generate draft tokens for multiple sequences in parallel.

        Generates draft tokens autoregressively for each sequence using
        the draft model with KV cache reuse.

        Args:
            draft_model: The draft model.
            input_ids_list: List of input tensors, one per sequence.
            past_key_values_list: Optional list of KV caches, one per sequence.

        Returns:
            (draft_tokens_per_seq, kv_caches_per_seq)
        """
        all_draft_tokens = []
        all_new_kvs = []

        for i, input_ids in enumerate(input_ids_list):
            past_kv = past_key_values_list[i] if past_key_values_list else None
            draft_tokens, new_kv = self.generate_draft_tokens(
                draft_model, input_ids, past_key_values=past_kv
            )
            all_draft_tokens.append(draft_tokens)
            all_new_kvs.append(new_kv)

        return all_draft_tokens, all_new_kvs

    def verify_batch(
        self,
        draft_tokens_list: List[List[int]],
        target_logits_list: List[torch.Tensor],
        tokenizer,
    ) -> List[Tuple[int, List[int], int]]:
        """Verify draft tokens for multiple sequences in parallel.

        Args:
            draft_tokens_list: List of draft token lists, one per sequence.
            target_logits_list: List of target logits, one per sequence.
            tokenizer: Tokenizer for decoding.

        Returns:
            List of (accepted_count, accepted_tokens, next_token) per sequence.
        """
        results = []
        for draft_tokens, target_logits in zip(draft_tokens_list, target_logits_list):
            result = self.verify_and_accept(draft_tokens, target_logits, tokenizer)
            results.append(result)
        return results
