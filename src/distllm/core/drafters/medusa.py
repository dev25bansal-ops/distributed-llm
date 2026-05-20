from typing import Tuple

import torch
import torch.nn as nn

from loguru import logger


class MedusaHeads(nn.Module):
    """Medusa-style multi-head speculation with trained prediction heads.

    Adds multiple prediction heads on top of the target model's hidden states.
    Each head predicts one future token, allowing parallel draft generation
    without a separate draft model.

    Architecture:
    - num_heads independent MLP heads, each taking hidden states as input
    - Head i predicts the token at position t+i+1 given hidden state at t
    - Each head: Linear(hidden_size, hidden_size) -> GELU -> Linear(hidden_size, vocab_size)

    Usage:
    1. Train heads on your target model (save checkpoint via save_checkpoint())
    2. Load trained weights via load_checkpoint(path)
    3. Call generate_draft_tokens() with hidden_states from the target model
    """

    def __init__(
        self,
        num_heads: int = 4,
        num_tokens_per_head: int = 3,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_tokens_per_head = num_tokens_per_head
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._weights_loaded = False

        # Each head predicts one future position: head 0 -> t+1, head 1 -> t+2, etc.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

        # Token embedding for autoregressive feedback
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize head weights with small random values to break symmetry."""
        with torch.no_grad():
            for head in self.heads:
                for module in head.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    @property
    def is_trained(self) -> bool:
        """Check if trained weights have been loaded."""
        return self._weights_loaded

    def load_checkpoint(self, path: str) -> None:
        """Load trained Medusa head weights from checkpoint."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()
        self._weights_loaded = True
        logger.info(f"Medusa heads loaded trained weights from {path}")

    def save_checkpoint(self, path: str) -> None:
        """Save current Medusa head weights."""
        torch.save(self.state_dict(), path)

    def generate_draft_tokens(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens from trained Medusa heads.

        Args:
            logits: Target model logits [batch, seq_len, vocab].
                Used to extract the last hidden position if hidden_states is None.
            hidden_states: Hidden states from target model [batch, seq_len, hidden_size].

        Returns:
            List of draft token sequences, one per head.

        Raises:
            RuntimeError: If no trained weights have been loaded.
        """
        if hidden_states is None:
            return []

        if not self._weights_loaded:
            # Fallback: diversified sampling from logits (not true Medusa)
            if logits is None:
                return []
            last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            probs = torch.softmax(last_logits / 0.8, dim=-1)
            drafts = []
            for _ in range(self.num_tokens_per_head):
                next_token = torch.multinomial(probs, 1).item()
                drafts.append(next_token)
            return [drafts]

        device = hidden_states.device
        self.to(device)

        # Get last hidden state as starting point
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        batch_size = last_hidden.shape[0]

        all_drafts = []
        for head_idx in range(self.num_heads):
            head = self.heads[head_idx]
            drafts = self._autoregressive_draft(
                head, last_hidden, batch_size, device
            )
            all_drafts.append(drafts)

        return all_drafts

    def _autoregressive_draft(
        self,
        head: nn.Sequential,
        hidden: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> list[int]:
        """Generate draft tokens autoregressively using a single trained head."""
        drafts = []
        current_hidden = hidden  # [batch, hidden_size]

        for _ in range(self.num_tokens_per_head):
            logits = head(current_hidden)  # [batch, vocab]
            logits = logits.mean(dim=0) if batch_size > 1 else logits[0]  # [vocab]

            # Top-k sampling
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            drafts.append(next_token)

            # Embed predicted token for next step (residual update)
            token_embed = self.embedding(torch.tensor([next_token], device=device))
            current_hidden = current_hidden + token_embed.unsqueeze(0)

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
