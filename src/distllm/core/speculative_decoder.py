"""Speculative decoding: draft token generation and verification.

Supports four speculation methods:
1. draft_model: Standard draft model (smaller model generates draft tokens)
2. medusa: Multi-head speculation (multiple prediction heads on target model)
3. eagle: Extrapolative generation with language embedding
4. ngram: N-gram matching from generated text (free, no extra model needed)
"""

import random
import torch
import torch.nn as nn
from typing import Tuple
from loguru import logger

from distllm.core.drafters import NgramMatcher, MedusaHeads, EAGLEGenerator


class SpeculativeDecoder:
    """Manages speculative decoding with multiple speculation methods.

    Supports:
    - draft_model: Standard draft model approach
    - medusa: Multi-head speculation on target model
    - eagle: Extrapolative generation with language embedding
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
        eagle_hidden_size: int = 4096,
        eagle_vocab_size: int = 32000,
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
        self._eagle_generator = EAGLEGenerator(
            hidden_size=eagle_hidden_size,
            vocab_size=eagle_vocab_size,
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

    @acceptance_rate.setter
    def acceptance_rate(self, value: float):
        self._acceptance_rate = max(0.0, min(1.0, value))

    def get_active_method(self, draft_model=None, hidden_states=None, target_logits=None) -> str:
        """Determine which speculation method to use."""
        if self.method != "auto":
            return self.method

        # Auto-selection logic
        if draft_model is not None:
            return "draft_model"
        if hidden_states is not None:
            return "eagle"
        if self._ngram_matcher._total_tokens_seen > 100:
            return "ngram"
        # Only use medusa if target logits are available
        if target_logits is not None:
            return "medusa"
        return "draft_model"  # Safe fallback

    def generate_draft_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
        target_logits: torch.Tensor | None = None,
        generated_ids: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
    ) -> Tuple[list[int], list | None, list[torch.Tensor] | None]:
        """Generate draft tokens using the active speculation method.

        Args:
            draft_model: The draft model (for draft_model method).
            input_ids: Current token IDs [batch, seq_len].
            past_key_values: Optional KV cache from previous steps.
            target_logits: Target model logits (for medusa method).
            generated_ids: Previously generated token IDs (for ngram method).
            hidden_states: Target model hidden states (for eagle method).
            lm_head: Language model head (for eagle method).

        Returns:
            (draft_tokens, updated_kv_cache, draft_logits_per_step)
        """
        active_method = self.get_active_method(draft_model, hidden_states, target_logits)

        if active_method == "ngram":
            if generated_ids is None:
                return [], None, None
            drafts, _ = self._generate_ngram_drafts(generated_ids)
            return drafts, None, None
        elif active_method == "eagle":
            if hidden_states is None or lm_head is None:
                return [], None, None
            drafts, _ = self._generate_eagle_drafts(hidden_states, lm_head)
            return drafts, None, None
        elif active_method == "medusa":
            if target_logits is None:
                return [], None, None
            drafts, _ = self._generate_medusa_drafts(target_logits, hidden_states)
            return drafts, None, None
        else:
            if draft_model is None:
                return [], None, None
            return self._generate_draft_model_tokens(draft_model, input_ids, past_key_values)

    def _generate_draft_model_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
    ) -> Tuple[list[int], list | None, list[torch.Tensor] | None]:
        """Generate draft tokens using a draft model.

        Returns:
            (draft_tokens, updated_kv_cache, draft_logits_per_step)
        """
        if draft_model is None:
            return [], None, None

        draft_logits_list = []

        # Ensure input is on the same device as the draft model
        model_device = next(draft_model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if past_key_values is not None:
            past_key_values = self._ensure_kv_device(past_key_values, model_device)

        with torch.no_grad():
            if past_key_values is not None:
                outputs = draft_model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = draft_model(input_ids, use_cache=True)

        if not hasattr(outputs, 'logits'):
            return [], None, None

        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        draft_logits_list.append(logits)
        draft_token = torch.argmax(logits, dim=-1).item()
        new_kv = outputs.past_key_values

        # Generate remaining tokens autoregressively
        draft_tokens = [draft_token]
        current_token = torch.tensor([[draft_token]], device=model_device)

        for _ in range(self.num_assistant_tokens - 1):
            with torch.no_grad():
                outputs = draft_model(
                    current_token,
                    past_key_values=new_kv,
                    use_cache=True,
                )
            if not hasattr(outputs, 'logits'):
                break
            logits = outputs.logits[:, -1, :]
            draft_logits_list.append(logits)
            current_token = torch.argmax(logits, dim=-1, keepdim=True)
            draft_tokens.append(current_token.item())
            new_kv = outputs.past_key_values

        return draft_tokens, new_kv, draft_logits_list

    @staticmethod
    def _ensure_kv_device(past_key_values, device):
        """Move KV cache tensors to the specified device."""
        if past_key_values is None:
            return None
        if isinstance(past_key_values, (list, tuple)):
            return tuple(
                tuple(t.to(device) if hasattr(t, 'to') else t for t in layer)
                for layer in past_key_values
            )
        return past_key_values

    def _generate_medusa_drafts(
        self,
        target_logits: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> Tuple[list[int], None]:
        """Generate draft tokens using Medusa multi-head speculation."""
        if hidden_states is None or target_logits is None:
            return [], None

        if not hasattr(self._medusa_heads, 'generate_draft_tokens'):
            return [], None
        if not hasattr(self._medusa_heads, 'merge_heads'):
            return [], None

        head_drafts = self._medusa_heads.generate_draft_tokens(logits=target_logits, hidden_states=hidden_states)
        if head_drafts is None:
            return [], None
        merged = self._medusa_heads.merge_heads(head_drafts)
        if not merged:
            return [], None
        return merged[: self.num_assistant_tokens], None

    def _generate_ngram_drafts(
        self,
        generated_ids: list[int] | None,
    ) -> Tuple[list[int], None]:
        """Generate draft tokens using n-gram matching."""
        if generated_ids is None or len(generated_ids) < self._ngram_matcher.min_match:
            return [], None

        drafts = self._ngram_matcher.predict(generated_ids, max_drafts=self.num_assistant_tokens)
        return drafts, None

    def _generate_eagle_drafts(
        self,
        hidden_states: torch.Tensor | None,
        lm_head: nn.Module | None,
    ) -> Tuple[list[int], None]:
        """Generate draft tokens using EAGLE hidden state extrapolation."""
        if hidden_states is None or lm_head is None:
            return [], None

        drafts = self._eagle_generator.generate_draft_tokens(
            hidden_states=hidden_states,
            lm_head=lm_head,
            num_drafts=self.num_assistant_tokens,
        )
        return drafts, None

    def record_generated_tokens(self, token_ids: list[int]) -> None:
        """Record generated tokens for n-gram indexing."""
        self._ngram_matcher.update(token_ids)

    def verify_and_accept(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        tokenizer,
        temperature: float = 1.0,
        draft_logits: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> Tuple[int, list[int], int]:
        """Verify draft tokens against target model logits.

        Uses standard speculative decoding rejection sampling
        (Leviathan et al., 2022) when draft_logits and temperature > 0 are
        provided; falls back to greedy argmax matching otherwise.

        Args:
            draft_tokens: Draft token IDs [num_drafts] or [1, num_drafts].
            target_logits: Target model logits [batch, seq_len, vocab].
            tokenizer: Tokenizer instance for EOS detection.
            temperature: Sampling temperature (0=greedy, >0=sampling).
            draft_logits: Draft model logits for rejection sampling.
                Can be a list of [1, vocab] tensors or a stacked [num_drafts, vocab]
                tensor. When None, falls back to greedy verification.

        Returns:
            Tuple of (num_accepted, accepted_tokens, next_token_id).
        """
        # Handle None/empty draft tokens
        if draft_tokens is None:
            draft_ids = []
        elif isinstance(draft_tokens, torch.Tensor):
            draft_ids = draft_tokens.flatten().tolist()
        elif isinstance(draft_tokens, list):
            draft_ids = draft_tokens
        else:
            try:
                draft_ids = list(draft_tokens)
            except TypeError:
                draft_ids = []

        if not draft_ids:
            next_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else -1
            self._record_acceptance(0, 0)
            return 0, [], next_token

        # Validate target_logits
        if target_logits is None:
            next_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else -1
            self._record_acceptance(len(draft_ids), 0)
            return 0, [], next_token

        # Get target token predictions for each verification position
        if target_logits.dim() == 3:
            batch_logits = target_logits[0]  # [seq_len, vocab]
        elif target_logits.dim() == 2:
            batch_logits = target_logits
        else:
            batch_logits = target_logits.unsqueeze(0) if target_logits.dim() == 1 else target_logits

        use_rejection = temperature > 0 and draft_logits is not None

        # Verify each draft token
        accepted = []
        for i, draft_id in enumerate(draft_ids):
            if i >= batch_logits.shape[0]:
                break

            target_logit_row = batch_logits[i]

            if use_rejection:
                # Standard rejection sampling (Leviathan et al., 2022)
                if isinstance(draft_logits, list):
                    draft_logit_row = draft_logits[i]
                else:
                    draft_logit_row = draft_logits[i]
                # Squeeze batch dimension if present
                while draft_logit_row.dim() > 1:
                    draft_logit_row = draft_logit_row.squeeze(0)

                target_probs = torch.softmax(target_logit_row / temperature, dim=-1)
                draft_probs = torch.softmax(draft_logit_row / temperature, dim=-1)

                p_target = target_probs[draft_id].item()
                p_draft = draft_probs[draft_id].item()

                accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
                if random.random() < accept_prob:
                    accepted.append(draft_id)
                else:
                    # Sample from corrected distribution (max(0, p_target - p_draft))
                    corrected = torch.clamp(target_probs - draft_probs, min=0)
                    corrected_sum = corrected.sum()
                    if corrected_sum > 0:
                        corrected /= corrected_sum
                        next_token = torch.multinomial(corrected, 1).item()
                    else:
                        next_token = target_logit_row.argmax(dim=-1).item()
                    num_accepted = i
                    accepted.append(next_token)
                    self._record_acceptance(len(draft_ids), num_accepted)
                    return num_accepted, accepted, next_token
            else:
                # Greedy verification (original behavior)
                target_token = target_logit_row.argmax(dim=-1).item()
                if target_token == draft_id:
                    accepted.append(draft_id)
                else:
                    accepted.append(target_token)
                    num_accepted = i
                    next_token = target_token
                    self._record_acceptance(len(draft_ids), num_accepted)
                    return num_accepted, accepted, next_token

        num_accepted = len(accepted)

        # Determine next token after accepted prefix
        next_pos = num_accepted
        if next_pos < batch_logits.shape[0]:
            if use_rejection:
                # Sample next token from target distribution
                logit_row = batch_logits[next_pos]
                probs = torch.softmax(logit_row / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = batch_logits[next_pos].argmax(dim=-1).item()
        elif accepted:
            next_token = accepted[-1]
        else:
            next_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else -1

        self._record_acceptance(len(draft_ids), num_accepted)

        return num_accepted, accepted, next_token

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
        input_ids_list: list[torch.Tensor],
        past_key_values_list: list | None = None,
        target_logits_list: list | None = None,
        generated_ids_list: list[list[int]] | None = None,
    ) -> Tuple[list[list[int]], list | None, list | None]:
        n = len(input_ids_list)
        if n == 0:
            return [], None, None

        all_draft_tokens: list[list[int] | None] = [None] * n
        all_new_kvs: list | None = [None] * n
        all_draft_logits: list | None = [None] * n

        for i in range(n):
            past_kv = past_key_values_list[i] if past_key_values_list else None
            target_logits = target_logits_list[i] if target_logits_list else None
            generated_ids = generated_ids_list[i] if generated_ids_list else None
            draft_tokens, new_kv, draft_logits = self.generate_draft_tokens(
                draft_model,
                input_ids_list[i],
                past_key_values=past_kv,
                target_logits=target_logits,
                generated_ids=generated_ids,
            )
            all_draft_tokens[i] = draft_tokens
            all_new_kvs[i] = new_kv
            all_draft_logits[i] = draft_logits

        return all_draft_tokens, all_new_kvs, all_draft_logits

    def verify_batch(
        self,
        draft_tokens_list: list,
        target_logits_list: list[torch.Tensor],
        tokenizer,
        draft_logits_list: list | None = None,
        temperature: float = 1.0,
    ) -> list[tuple[int, list[int], int]]:
        """Verify draft tokens for multiple sequences.

        Args:
            draft_tokens_list: List of draft token IDs per sequence.
            target_logits_list: List of target model logit tensors per sequence.
            tokenizer: Tokenizer instance for EOS detection.
            draft_logits_list: Optional list of draft logit tensors per sequence.
            temperature: Sampling temperature.

        Returns:
            List of (num_accepted, accepted_tokens, next_token_id) tuples.
        """
        results = []
        for i, (draft_tokens, target_logits) in enumerate(zip(draft_tokens_list, target_logits_list)):
            dl = draft_logits_list[i] if draft_logits_list else None
            result = self.verify_and_accept(
                draft_tokens, target_logits, tokenizer,
                temperature=temperature, draft_logits=dl,
            )
            results.append(result)
        return results
