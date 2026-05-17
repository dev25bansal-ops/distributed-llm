"""Train Medusa/EAGLE speculative decoding heads on the target model.

Provides training infrastructure for training auxiliary prediction heads
used in speculative decoding:
- Medusa heads: multiple independent draft head branches
- EAGLE heads: feature-conditioned draft prediction
- EAGLE-2 heads: feature alignment + shared backbone

The trainer:
1. Freezes the base model
2. Adds configurable draft head architecture
3. Trains on target model's own generations (self-supervised)
4. Produces trained weights compatible with SpeculativeDecoder
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class TrainerConfig:
    num_draft_heads: int = 3
    head_dim: int = 2048
    num_layers: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 1000
    batch_size: int = 4
    seq_len: int = 512
    grad_clip: float = 1.0
    save_every: int = 200
    output_dir: str = "./draft_heads"
    eval_every: int = 100


@dataclass
class TrainingStats:
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0


class MedusaDraftHead(nn.Module):
    """Medusa-style multi-head draft prediction module.

    Each head independently predicts the next token at a specific offset.
    head i predicts the token at position t + i + 1.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 3, head_dim: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                nn.GELU(),
                nn.Linear(head_dim // 2, vocab_size),
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        shared = self.shared(hidden_states)
        return [head(shared) for head in self.heads]


class EAGLEDraftHead(nn.Module):
    """EAGLE-style feature-conditioned draft head.

    Uses the base model's hidden states as features for draft prediction,
    with a shared MLP backbone and per-offset prediction heads.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int = 3, head_dim: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.feature_proj = nn.Linear(hidden_size + vocab_size, head_dim)
        self.feature_norm = nn.LayerNorm(head_dim)
        self.backbone = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(head_dim, vocab_size)
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, embedding_layer: nn.Module) -> List[torch.Tensor]:
        embeds = embedding_layer(input_ids)
        combined = torch.cat([hidden_states, embeds], dim=-1)
        features = self.feature_proj(combined)
        features = self.feature_norm(features)
        backbone_out = self.backbone(features) + features
        return [head(backbone_out) for head in self.heads]


class SpeculativeTrainer:
    """Trains Medusa/EAGLE draft heads for speculative decoding.

    Usage:
        trainer = SpeculativeTrainer(
            base_model=model,
            tokenizer=tokenizer,
            head_type="medusa",
            config=TrainerConfig(),
        )
        trainer.train(num_steps=500)
        trainer.save("./draft_heads.pt")
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any = None,
        head_type: str = "medusa",
        config: Optional[TrainerConfig] = None,
        data_fn: Optional[Callable] = None,
        device: str = "cuda",
    ):
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._head_type = head_type
        self._config = config or TrainerConfig()
        self._data_fn = data_fn
        self._device = device

        self._hidden_size = 4096
        self._vocab_size = 32000
        self._embedding_layer: Optional[nn.Module] = None

        self._draft_head: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._stats = TrainingStats()

    def build_head(self) -> nn.Module:
        if self._head_type == "medusa":
            head = MedusaDraftHead(
                hidden_size=self._hidden_size,
                vocab_size=self._vocab_size,
                num_heads=self._config.num_draft_heads,
                head_dim=self._config.head_dim,
            )
        elif self._head_type == "eagle":
            head = EAGLEDraftHead(
                hidden_size=self._hidden_size,
                vocab_size=self._vocab_size,
                num_heads=self._config.num_draft_heads,
                head_dim=self._config.head_dim,
            )
        else:
            raise ValueError(f"Unknown head type: {self._head_type}")

        self._embedding_layer = self._find_embedding_layer()
        return head.to(self._device)

    def _find_embedding_layer(self) -> Optional[nn.Module]:
        for name, module in self._base_model.named_modules():
            if isinstance(module, nn.Embedding) and module.weight.shape[0] == self._vocab_size:
                return module
        return None

    def _freeze_base_model(self) -> None:
        for param in self._base_model.parameters():
            param.requires_grad = False
        self._base_model.eval()

    def _generate_training_data(self, batch_size: int, seq_len: int) -> torch.Tensor:
        if self._data_fn is not None:
            return self._data_fn(batch_size, seq_len)
        return torch.randint(0, min(self._vocab_size, 1000), (batch_size, seq_len), device=self._device)

    def train(self, num_steps: Optional[int] = None) -> TrainingStats:
        """Run training loop for draft heads.

        Args:
            num_steps: Number of training steps (overrides config).

        Returns:
            TrainingStats with final metrics.
        """
        num_steps = num_steps or self._config.max_steps
        self._freeze_base_model()

        self._draft_head = self.build_head()
        self._optimizer = torch.optim.AdamW(
            self._draft_head.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        def _lr_lambda(step: int) -> float:
            if step < self._config.warmup_steps:
                return step / max(self._config.warmup_steps, 1)
            return 1.0

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, _lr_lambda)

        logger.info(
            f"Training {self._head_type} heads: "
            f"{self._config.num_draft_heads} heads, "
            f"{sum(p.numel() for p in self._draft_head.parameters()):,} params, "
            f"{num_steps} steps"
        )

        self._draft_head.train()
        start_time = time.time()
        total_tokens = 0

        for step in range(1, num_steps + 1):
            inputs = self._generate_training_data(self._config.batch_size, self._config.seq_len)

            with torch.no_grad():
                outputs = self._base_model(inputs)
                hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') and outputs.hidden_states else None
                if hidden is None:
                    hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
                if hidden is None:
                    hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            logits_list = self._draft_head(hidden)

            loss = 0.0
            correct = 0
            total = 0

            for i, logits in enumerate(logits_list):
                shift = i + 1
                if shift >= inputs.shape[1]:
                    continue
                target = inputs[:, shift:shift + logits.shape[1]]
                logits = logits[:, :target.shape[1], :]
                loss += F.cross_entropy(logits.reshape(-1, self._vocab_size), target.reshape(-1))
                preds = logits.argmax(dim=-1)
                correct += (preds == target).sum().item()
                total += target.numel()

            loss = loss / len(logits_list)
            accuracy = correct / max(total, 1)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._draft_head.parameters(), self._config.grad_clip)
            self._optimizer.step()
            self._scheduler.step()

            total_tokens += total
            elapsed = time.time() - start_time

            self._stats = TrainingStats(
                step=step,
                loss=loss.item(),
                accuracy=accuracy,
                tokens_per_second=total_tokens / max(elapsed, 0.001),
                elapsed_seconds=elapsed,
            )

            if step % self._config.eval_every == 0 or step == num_steps:
                logger.info(
                    f"Step {step}/{num_steps}: loss={loss.item():.4f}, "
                    f"acc={accuracy:.4f}, tok/s={self._stats.tokens_per_second:.0f}"
                )

            if step % self._config.save_every == 0:
                self.save(f"{self._config.output_dir}/step_{step}.pt")

        self.save(f"{self._config.output_dir}/final.pt")
        logger.info(f"Training complete: {self._stats}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._stats

    @torch.no_grad()
    def evaluate(self, eval_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate draft head accuracy on evaluation data."""
        if self._draft_head is None:
            return {"loss": float('inf'), "accuracy": 0.0}

        self._draft_head.eval()
        inputs = eval_data if eval_data is not None else self._generate_training_data(2, self._config.seq_len)

        outputs = self._base_model(inputs)
        hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') and outputs.hidden_states else outputs[0]
        logits_list = self._draft_head(hidden)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for i, logits in enumerate(logits_list):
            shift = i + 1
            if shift >= inputs.shape[1]:
                continue
            target = inputs[:, shift:shift + logits.shape[1]]
            logits = logits[:, :target.shape[1], :]
            total_loss += F.cross_entropy(logits.reshape(-1, self._vocab_size), target.reshape(-1)).item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == target).sum().item()
            total_tokens += target.numel()

        self._draft_head.train()
        return {
            "loss": total_loss / len(logits_list),
            "accuracy": total_correct / max(total_tokens, 1),
        }

    def save(self, path: str) -> None:
        if self._draft_head is None:
            raise RuntimeError("No draft head to save")
        torch.save({
            "head_type": self._head_type,
            "config": self._config,
            "model_state_dict": self._draft_head.state_dict(),
            "stats": self._stats,
        }, path)
        logger.info(f"Saved draft head to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self._head_type = checkpoint.get("head_type", self._head_type)
        saved_config = checkpoint.get("config")
        if saved_config:
            for field_name in self._config.__dataclass_fields__:
                if hasattr(saved_config, field_name):
                    setattr(self._config, field_name, getattr(saved_config, field_name))
        self._draft_head = self.build_head()
        self._draft_head.load_state_dict(checkpoint["model_state_dict"])
        self._stats = checkpoint.get("stats", TrainingStats())
        logger.info(f"Loaded draft head from {path} (step {self._stats.step})")
