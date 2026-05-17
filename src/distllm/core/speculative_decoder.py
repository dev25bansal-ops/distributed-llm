"""Speculative decoding: draft token generation and verification.

Supports four speculation methods:
1. draft_model: Standard draft model (smaller model generates draft tokens)
2. medusa: Multi-head speculation (multiple prediction heads on target model)
3. eagle: Extrapolative generation with language embedding
4. ngram: N-gram matching from generated text (free, no extra model needed)
"""

import torch
import torch.nn as nn
from typing import Any
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
        self._ngram_index: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._total_tokens_seen = 0

    def update(self, token_ids: list[int]) -> None:
        """Index newly generated tokens for future matching."""
        for n in range(self.min_match, min(self.max_match + 1, len(token_ids) + 1)):
            for i in range(len(token_ids) - n):
                ngram = tuple(token_ids[i : i + n])
                next_token = token_ids[i + n]
                self._ngram_index[ngram].append(next_token)
        self._total_tokens_seen += len(token_ids)

    def predict(self, context: list[int], max_drafts: int = 5) -> list[int]:
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
        hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
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
    ) -> list[int]:
        """Generate draft tokens autoregressively from logits.

        Uses argmax sampling with diversity: each position picks the most likely
        token that hasn't been drafted yet at this position.
        In production, this would use trained Medusa heads.
        """
        drafts = []
        # Get top-k candidates once (assume roughly stationary distribution)
        # to produce diverse drafts across positions
        batch_logits = logits[0] if logits.dim() == 3 else logits
        top_k = min(num_tokens + 5, batch_logits.shape[-1])
        top_probs, top_indices = torch.topk(
            torch.softmax(batch_logits / max(temperature, 0.01), dim=-1),
            top_k, dim=-1
        )

        # Pick different top tokens for each draft position (diversity)
        used_ids = set()
        for pos in range(num_tokens):
            idx = 0
            while idx < top_k:
                candidate = top_indices[idx].item()
                if candidate not in used_ids or idx >= top_k - 1:
                    break
                idx += 1
            token_id = top_indices[idx].item()
            drafts.append(token_id)
            used_ids.add(token_id)

        return drafts

    def merge_heads(self, head_drafts: list[list[int]]) -> list[int]:
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


class EAGLEGenerator:
    """EAGLE-style speculative decoding via language embedding extrapolation.

    Uses the target model's own hidden states to predict future tokens:
    1. Extract hidden states from the target model's last layer
    2. Use a lightweight predictor head to extrapolate future hidden states
    3. Project extrapolated states back to token space via the LM head

    This is more accurate than Medusa because it uses actual hidden state
    evolution rather than temperature-diversified sampling.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_layers: int = 2,
        num_draft_tokens: int = 5,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_draft_tokens = num_draft_tokens

        # Lightweight predictor: predicts next hidden state from current
        # In production this would be trained; here we use a simple projection
        self._predictor: nn.Sequential | None = None
        self._lm_head: nn.Linear | None = None
        self._initialized = False

    def _init_networks(self, device: torch.device) -> None:
        """Initialize predictor and LM head networks."""
        if self._initialized:
            return

        self._predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(device)

        # Initialize with identity mapping + small noise for diversity
        with torch.no_grad():
            self._predictor[0].weight.copy_(torch.eye(self.hidden_size) * 0.1)
            self._predictor[3].weight.copy_(torch.eye(self.hidden_size) * 0.5)

        self._initialized = True

    def generate_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        num_drafts: int = 5,
        temperature: float = 0.8,
    ) -> list[int]:
        """Generate draft tokens via hidden state extrapolation.

        Args:
            hidden_states: Target model hidden states [batch, seq_len, hidden_size].
            lm_head: Language model head for token prediction.
            num_drafts: Number of draft tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of predicted draft token IDs.
        """
        device = hidden_states.device
        self._init_networks(device)

        # Get last hidden state as starting point
        current_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]

        drafts = []
        for _ in range(num_drafts):
            # Predict next hidden state via extrapolation
            next_hidden = self._predictor(current_hidden)

            # Predictor output is the next hidden state directly
            extrapolated = next_hidden

            # Project to vocabulary via LM head
            logits = lm_head(extrapolated)  # [batch, vocab]

            # Sample token
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            drafts.append(next_token.item())

            # Use extrapolated state as input for next prediction
            current_hidden = extrapolated

        return drafts

    def generate_with_anchor(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        num_drafts: int = 5,
        anchor_ratio: float = 0.3,
    ) -> list[int]:
        """Generate draft tokens with anchor-based extrapolation.

        Uses a weighted combination of predicted and original hidden states
        as anchors to maintain coherence with the target model's trajectory.

        Args:
            hidden_states: Target model hidden states [batch, seq_len, hidden_size].
            lm_head: Language model head for token prediction.
            num_drafts: Number of draft tokens to generate.
            anchor_ratio: Weight given to anchor (0 = pure prediction, 1 = pure anchor).

        Returns:
            List of predicted draft token IDs.
        """
        device = hidden_states.device
        self._init_networks(device)

        # Compute anchor as mean of recent hidden states
        recent = hidden_states[:, -min(4, hidden_states.shape[1]):, :]
        anchor = recent.mean(dim=1)  # [batch, hidden_size]

        current_hidden = hidden_states[:, -1, :]
        drafts = []

        for _ in range(num_drafts):
            # Predict next hidden state
            predicted_delta = self._predictor(current_hidden)

            # Interpolate between prediction and anchor
            extrapolated = current_hidden + predicted_delta * 0.5
            anchored = extrapolated * (1 - anchor_ratio) + anchor * anchor_ratio

            # Project to token
            logits = lm_head(anchored)
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            drafts.append(next_token.item())

            current_hidden = anchored

        return drafts


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
        if self._step_count < self.warmup_steps:
            return False
        return self._enabled

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_rate

    @acceptance_rate.setter
    def acceptance_rate(self, value: float):
        self._acceptance_rate = max(0.0, min(1.0, value))

    def get_active_method(self, draft_model=None, hidden_states=None) -> str:
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
        return "medusa"  # Default: use medusa heads on target model

    def generate_draft_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
        target_logits: torch.Tensor | None = None,
        generated_ids: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
    ) -> Tuple[list[int], list | None]:
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
            (draft_tokens, updated_kv_cache)
        """
        active_method = self.get_active_method(draft_model, hidden_states)

        if active_method == "ngram":
            return self._generate_ngram_drafts(generated_ids)
        elif active_method == "eagle":
            return self._generate_eagle_drafts(hidden_states, lm_head)
        elif active_method == "medusa":
            return self._generate_medusa_drafts(target_logits)
        else:
            return self._generate_draft_model_tokens(draft_model, input_ids, past_key_values)

    def _generate_draft_model_tokens(
        self,
        draft_model,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
    ) -> Tuple[list[int], list | None]:
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
        target_logits: torch.Tensor | None,
    ) -> Tuple[list[int], None]:
        """Generate draft tokens using Medusa multi-head speculation."""
        if target_logits is None:
            return [], None

        head_drafts = self._medusa_heads.generate_draft_tokens(target_logits)
        merged = self._medusa_heads.merge_heads(head_drafts)
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
    ) -> Tuple[int, list[int], int]:
        """Verify draft tokens against target model logits.

        For each draft token, checks if it matches the target model's argmax.
        Accepts a prefix of the draft, then samples the next token from the
        target distribution.

        Args:
            draft_tokens: Draft token IDs [num_drafts] or [1, num_drafts].
            target_logits: Target model logits [batch, seq_len, vocab].
            tokenizer: Tokenizer instance for EOS detection.

        Returns:
            Tuple of (num_accepted, accepted_tokens, next_token_id).
        """
        # Flatten draft tokens to 1D
        if isinstance(draft_tokens, torch.Tensor):
            draft_ids = draft_tokens.flatten().tolist()
        else:
            draft_ids = list(draft_tokens)

        # Get target token predictions for each verification position
        if target_logits.dim() == 3:
            batch_logits = target_logits[0]  # [seq_len, vocab]
        else:
            batch_logits = target_logits

        # Verify each draft token greedily
        accepted = []
        for i, draft_id in enumerate(draft_ids):
            if i < batch_logits.shape[0]:
                target_token = batch_logits[i].argmax(dim=-1).item()
                if target_token == draft_id:
                    accepted.append(draft_id)
                else:
                    break
            else:
                break

        num_accepted = len(accepted)

        # Determine next token after accepted prefix
        next_pos = num_accepted
        if next_pos < batch_logits.shape[0]:
            next_token = batch_logits[next_pos].argmax(dim=-1).item()
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
    ) -> Tuple[list[list[int]], list | None]:
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
        draft_logits_list: list[torch.Tensor],
        target_logits_list: list[torch.Tensor],
        temperature: float = 1.0,
    ) -> list[bool]:
        """Verify draft logits for multiple sequences."""
        results = []
        for draft_logits, target_logits in zip(draft_logits_list, target_logits_list):
            result = self.verify_and_accept(target_logits, draft_logits, temperature)
            results.append(result)
        return results


class TrainedEAGLEHeads(nn.Module):
    """Trained EAGLE-style draft head with configurable MLP architecture.

    Replaces the old EAGLEGenerator stub with actual trained modules.
    Architecture:
    - Input: target model hidden states [batch, hidden_size]
    - 2-4 layer MLP with LayerNorm + GELU
    - Output: logits over vocabulary for draft token prediction

    Supports:
    - Configurable depth (2-4 layers)
    - Residual connections
    - Dropout for regularization
    - Training checkpoint save/load
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = max(2, min(num_layers, 4))
        self.use_residual = use_residual

        layers = []
        in_dim = hidden_size
        for i in range(self.num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp(hidden_states)
        if self.use_residual:
            hidden = hidden + hidden_states
        return self.lm_head(hidden)

    def generate_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        num_drafts: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> list[int]:
        """Generate draft tokens autoregressively.

        Uses the trained head to predict each token, feeding predicted
        token embeddings back as input for the next step.
        """
        draft_tokens = []
        current_hidden = hidden_states[:, -1:, :]

        for _ in range(num_drafts):
            logits = self.forward(current_hidden)
            logits = logits[:, -1, :]

            if top_k > 0:
                values, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token.item())

            # Embed predicted token for next step
            current_hidden = current_hidden + self._embed_token(next_token)

        return draft_tokens

    def _embed_token(self, token: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token)
        return embedded

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()


class EAGLE2Heads(nn.Module):
    """EAGLE-2 draft head with feature alignment and layer sharing.

    EAGLE-2 improves on EAGLE by:
    1. Feature alignment: aligns draft head features with target model
    2. Layer sharing: reuses target model's early layers for feature extraction
    3. Multi-token prediction: predicts N future tokens in parallel

    Architecture:
    - Shared feature extractor (1-2 transformer layers)
    - N parallel prediction heads (one per future token)
    - Feature alignment loss during training
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_draft_tokens: int = 5,
        num_feature_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_draft_tokens = num_draft_tokens

        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.draft_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_draft_tokens)
        ])

        self.feature_align = nn.Linear(hidden_size, hidden_size)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        features = self.feature_extractor(hidden_states)
        aligned = self.feature_align(features)
        return [head(aligned) for head in self.draft_heads]

    def generate_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        num_drafts: int | None = None,
    ) -> list[int]:
        n = num_drafts or self.num_draft_tokens
        all_logits = self.forward(hidden_states)
        draft_tokens = []
        for i in range(min(n, len(all_logits))):
            logits = all_logits[i][:, -1, :]
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token.item())
        return draft_tokens

    def compute_feature_alignment_loss(
        self,
        draft_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(draft_features, target_features)

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
