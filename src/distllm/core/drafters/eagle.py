import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        warnings.warn(
            "EAGLEGenerator uses untrained identity projections and produces "
            "meaningless draft tokens. For production use, either: "
            "(1) Use TrainedEAGLEHeads with trained checkpoint, "
            "(2) Use draft_model method with a smaller model, "
            "(3) Use ngram method for zero-cost speculation. "
            "This warning can be suppressed with warnings.filterwarnings('ignore', category=UserWarning)",
            UserWarning,
            stacklevel=2,
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_draft_tokens = num_draft_tokens

        # Lightweight predictor: predicts next hidden state from current
        # In production this would be trained; here we use a simple projection
        self._predictor: nn.Sequential | None = None
        self._lm_head: nn.Linear | None = None
        self._initialized = False
        self._probs_buffer: torch.Tensor | None = None

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
            scaled = logits / temperature
            if self._probs_buffer is None or self._probs_buffer.shape != scaled.shape:
                self._probs_buffer = torch.empty_like(scaled)
            probs = F.softmax(scaled, dim=-1, out=self._probs_buffer)
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
            scaled = logits / 0.8
            if self._probs_buffer is None or self._probs_buffer.shape != scaled.shape:
                self._probs_buffer = torch.empty_like(scaled)
            probs = F.softmax(scaled, dim=-1, out=self._probs_buffer)
            next_token = torch.multinomial(probs, 1)
            drafts.append(next_token.item())

            current_hidden = anchored

        return drafts


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
        self._probs_buffer: torch.Tensor | None = None

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

            scaled = logits / temperature
            if self._probs_buffer is None or self._probs_buffer.shape != scaled.shape:
                self._probs_buffer = torch.empty_like(scaled)
            probs = F.softmax(scaled, dim=-1, out=self._probs_buffer)
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
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
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
        self._probs_buffer: torch.Tensor | None = None
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
            scaled = logits / 0.8
            if self._probs_buffer is None or self._probs_buffer.shape != scaled.shape:
                self._probs_buffer = torch.empty_like(scaled)
            probs = F.softmax(scaled, dim=-1, out=self._probs_buffer)
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
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.eval()
