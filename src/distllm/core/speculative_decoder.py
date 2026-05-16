"""Speculative decoding: draft token generation and verification.

Supports three speculation methods:
1. draft_model: Standard draft model (smaller model generates draft tokens)
2. medusa: Multi-head speculation (multiple prediction heads on target model)
3. ngram: N-gram matching from generated text (free, no extra model needed)
"""

import torch
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from loguru import logger


class NgramMatcher:
    """N-gram based speculative decoding (free draft tokens, no extra model).

    Matches n-grams from the generated text to predict likely next tokens
    based on previously seen sequences. Works well for repetitive text,
    code, and structured outputs.
    """

    def __init__(self, min_match: int = 4, max_match: int = 10):
        self.min_match = min_match
        self.max_match = max_match
        # Maps n-gram tuple -> list of next tokens seen
        self._ngram_index: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self._total_tokens_seen = 0

    def update(self, token_ids: List[int]) -> None:
        """Index newly generated tokens for future matching."""
        for n in range(self.min_match, min(self.max_match + 1, len(token_ids) + 1)):
            for i in range(len(token_ids) - n):
                ngram = tuple(token_ids[i : i + n])
                next_token = token_ids[i + n]
                self._ngram_index[ngram].append(next_token)
        self._total_tokens_seen += len(token_ids)

    def predict(self, context: List[int], max_drafts: int = 5) -> List[int]:
        """Predict draft tokens based on n-gram matching.

        Uses the longest matching n-gram from the end of context
        to find the most likely next tokens.

        Args:
            context: Recent token IDs to match against.
            max_drafts: Maximum number of draft tokens to generate.

        Returns:
            List of predicted draft token IDs.
        """
        if self._total_tokens_seen == 0:
            return []

        drafts = []
        current_context = list(context)

        for _ in range(max_drafts):
            best_match = []
            best_n = 0

            # Try to find longest matching n-gram
            for n in range(min(self.max_match, len(current_context)), self.min_match - 1, -1):
                ngram = tuple(current_context[-n:])
                if ngram in self._ngram_index:
                    candidates = self._ngram_index[ngram]
                    # Most common next token
                    next_token = max(set(candidates), key=candidates.count)
                    best_match = [next_token]
                    best_n = n
                    break

            if not best_match:
                break

            drafts.extend(best_match)
            current_context.extend(best_match)

        return drafts[:max_drafts]

    def stats(self) -> dict:
        return {
            "total_tokens_indexed": self._total_tokens_seen,
            "unique_ngrams": len(self._ngram_index),
        }


class MedusaHeads:
    """Medusa-style multi-head speculation.

    Adds multiple prediction heads on top of the target model's hidden states.
    Each head predicts one future token, allowing parallel draft generation
    without a separate draft model.

    In production, these would be trained LoRA heads. Here we implement
    a lightweight approximation using the target model's own logits
    with different temperature/scaling per head.
    """

    def __init__(self, num_heads: int = 4, num_tokens_per_head: int = 3):
        self.num_heads = num_heads
        self.num_tokens_per_head = num_tokens_per_head

    def generate_draft_tokens(
        self,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Generate draft tokens from each Medusa head.

        Each head predicts a sequence of tokens autoregressively.
        Without trained heads, uses scaled/diversified sampling from
        target model logits.

        Args:
            logits: Target model logits [batch, seq_len, vocab].
            hidden_states: Optional hidden states for head projection.

        Returns:
            List of draft token sequences, one per head.
        """
        # Get last position logits [batch, vocab]
        last_logits = logits[:, -1, :]  # [batch, vocab]
        batch_size = last_logits.shape[0]

        all_drafts = []
        for head_idx in range(self.num_heads):
            # Diversify each head with different temperature scaling
            temperature = 0.5 + head_idx * 0.3  # 0.5, 0.8, 1.1, 1.4
            drafts = self._autoregressive_draft(
                last_logits,
                num_tokens=self.num_tokens_per_head,
                temperature=temperature,
            )
            all_drafts.append(drafts)

        return all_drafts

    def _autoregressive_draft(
        self,
        logits: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
    ) -> List[int]:
        """Generate draft tokens autoregressively from logits."""
        drafts = []
        current_logits = logits.clone()

        for _ in range(num_tokens):
            probs = torch.softmax(current_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_id = next_token.item()
            drafts.append(token_id)

            # Simulate next step: shift logits slightly (approximation)
            # In real Medusa, this would be a forward through the head
            current_logits = current_logits * 0.9 + torch.randn_like(current_logits) * 0.1

        return drafts

    def merge_heads(self, head_drafts: List[List[int]]) -> List[int]:
        """Merge draft sequences from multiple heads into one sequence.

        Uses majority voting / consensus across heads.
        """
        if not head_drafts:
            return []

        max_len = max(len(d) for d in head_drafts)
        merged = []

        for pos in range(max_len):
            votes = []
            for head_draft in head_drafts:
                if pos < len(head_draft):
                    votes.append(head_draft[pos])

            if not votes:
                break

            # Take most common token at each position
            next_token = max(set(votes), key=votes.count)
            merged.append(next_token)

        return merged


class SpeculativeDecoder:
    """Manages speculative decoding with multiple speculation methods.

    Supports:
    - draft_model: Standard draft model approach
    - medusa: Multi-head speculation on target model
    - ngram: N-gram matching from generated text (free)
    - auto: Automatically selects best method based on context
    """

    def __init__(
        self,
        num_assistant_tokens: int = 5,
        min_acceptance_rate: float = 0.3,
        warmup_steps: int = 10,
        method: str = "draft_model",
        medusa_num_heads: int = 4,
        medusa_num_tokens_per_head: int = 3,
        ngram_min_match: int = 4,
    ):
        self.num_assistant_tokens = num_assistant_tokens
        self.min_acceptance_rate = min_acceptance_rate
        self.warmup_steps = warmup_steps
        self.method = method

        # Acceptance rate tracking (EMA)
        self._acceptance_rate: float = 1.0
        self._total_draft_tokens: int = 0
        self._total_accepted: int = 0
        self._step_count: int = 0
        self._enabled: bool = True

        # Method-specific components
        self._medusa_heads = MedusaHeads(
            num_heads=medusa_num_heads,
            num_tokens_per_head=medusa_num_tokens_per_head,
        )
        self._ngram_matcher = NgramMatcher(
            min_match=ngram_min_match,
        )

        # Method selection for "auto" mode
        self._auto_method = "draft_model"  # default fallback

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate

    def get_active_method(self, draft_model=None) -> str:
        """Determine which speculation method to use."""
        if self.method != "auto":
            return self.method

        # Auto-selection logic
        if draft_model is not None:
            return "draft_model"
        if self._ngram_matcher._total_tokens_seen > 100:
            return "ngram"
        return "medusa"  # Default: use medusa heads on target model

    def generate_draft_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        target_logits: Optional[torch.Tensor] = None,
        generated_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], Optional[List]]:
        """Generate draft tokens using the active speculation method.

        Args:
            draft_model: The draft model (for draft_model method).
            input_ids: Current token IDs [batch, seq_len].
            past_key_values: Optional KV cache from previous steps.
            target_logits: Target model logits (for medusa method).
            generated_ids: Previously generated token IDs (for ngram method).

        Returns:
            (draft_tokens, updated_kv_cache)
        """
        active_method = self.get_active_method(draft_model)

        if active_method == "ngram":
            return self._generate_ngram_drafts(generated_ids)
        elif active_method == "medusa":
            return self._generate_medusa_drafts(target_logits)
        else:
            return self._generate_draft_model_tokens(draft_model, input_ids, past_key_values)

    def _generate_draft_model_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
    ) -> Tuple[List[int], Optional[List]]:
        """Generate draft tokens using a draft model."""
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

    def _generate_medusa_drafts(
        self,
        target_logits: Optional[torch.Tensor],
    ) -> Tuple[List[int], None]:
        """Generate draft tokens using Medusa multi-head speculation."""
        if target_logits is None:
            return [], None

        head_drafts = self._medusa_heads.generate_draft_tokens(target_logits)
        merged = self._medusa_heads.merge_heads(head_drafts)
        return merged[: self.num_assistant_tokens], None

    def _generate_ngram_drafts(
        self,
        generated_ids: Optional[List[int]],
    ) -> Tuple[List[int], None]:
        """Generate draft tokens using n-gram matching."""
        if generated_ids is None or len(generated_ids) < self._ngram_matcher.min_match:
            return [], None

        drafts = self._ngram_matcher.predict(generated_ids, max_drafts=self.num_assistant_tokens)
        return drafts, None

    def record_generated_tokens(self, token_ids: List[int]) -> None:
        """Record generated tokens for n-gram indexing."""
        self._ngram_matcher.update(token_ids)

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
        metrics = {
            "acceptance_rate": self._acceptance_rate,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "step_count": self._step_count,
            "enabled": self._enabled,
            "method": self.get_active_method(),
        }
        if self.method == "ngram" or self.method == "auto":
            metrics["ngram_stats"] = self._ngram_matcher.stats()
        return metrics

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
        target_logits_list: Optional[List] = None,
        generated_ids_list: Optional[List[List[int]]] = None,
    ) -> Tuple[List[List[int]], Optional[List]]:
        """Generate draft tokens for multiple sequences in parallel."""
        all_draft_tokens = []
        all_new_kvs = []

        for i, input_ids in enumerate(input_ids_list):
            past_kv = past_key_values_list[i] if past_key_values_list else None
            target_logits = target_logits_list[i] if target_logits_list else None
            generated_ids = generated_ids_list[i] if generated_ids_list else None

            draft_tokens, new_kv = self.generate_draft_tokens(
                draft_model,
                input_ids,
                past_key_values=past_kv,
                target_logits=target_logits,
                generated_ids=generated_ids,
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
        """Verify draft tokens for multiple sequences in parallel."""
        results = []
        for draft_tokens, target_logits in zip(draft_tokens_list, target_logits_list):
            result = self.verify_and_accept(draft_tokens, target_logits, tokenizer)
            results.append(result)
        return results
