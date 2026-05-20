"""Quality SLA enforcement for precision-aware serving.

Defines per-request quality budgets that map to minimum precision
requirements, preventing INT4 serving for high-quality requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class QualityTier(str, Enum):
    """Quality tiers for request SLA enforcement."""
    HIGH = "high"       # BF16 required (creative writing, code generation)
    MEDIUM = "medium"   # FP16 acceptable (general Q&A, summarization)
    LOW = "low"         # INT4/INT8 acceptable (classification, extraction)


@dataclass
class SLAPolicy:
    """SLA policy for a quality tier."""
    tier: QualityTier
    min_precision: str  # torch.dtype name
    max_quality_loss: float  # Maximum acceptable KL-divergence
    request_class: str = ""

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "SLAPolicy":
        """Get the default SLA policy for a quality tier."""
        policies = {
            QualityTier.HIGH: cls(
                tier=tier,
                min_precision="bfloat16",
                max_quality_loss=0.01,
                request_class="creative",
            ),
            QualityTier.MEDIUM: cls(
                tier=tier,
                min_precision="float16",
                max_quality_loss=0.05,
                request_class="general",
            ),
            QualityTier.LOW: cls(
                tier=tier,
                min_precision="int8",
                max_quality_loss=0.2,
                request_class="classification",
            ),
        }
        return policies[tier]


PRECISION_RANK = {
    "int4": 0,
    "int8": 1,
    "float8_e4m3fn": 2,
    "float8_e5m2": 2,
    "float16": 3,
    "bfloat16": 4,
    "float32": 5,
}


class QualitySLA:
    """Enforces quality SLA for precision-aware request routing.

    Maps request quality requirements to minimum precision levels,
    and provides KL-divergence-based quality evaluation.
    """

    @staticmethod
    def select_precision_for_request(
        quality_tier: QualityTier = QualityTier.MEDIUM,
    ) -> torch.dtype:
        """Select the minimum torch.dtype for a quality tier."""
        policy = SLAPolicy.for_tier(quality_tier)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int8": torch.int8,
        }
        return dtype_map.get(policy.min_precision, torch.float16)

    @staticmethod
    def check_precision_meets_sla(
        actual_precision: str,
        required_tier: QualityTier,
    ) -> bool:
        """Check if the actual precision meets the SLA requirement."""
        policy = SLAPolicy.for_tier(required_tier)
        actual_rank = PRECISION_RANK.get(actual_precision.lower(), 0)
        required_rank = PRECISION_RANK.get(policy.min_precision.lower(), 0)
        return actual_rank >= required_rank

    @staticmethod
    def evaluate_quality(
        gold_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> float:
        """Evaluate quality loss via KL-divergence between distributions.

        Args:
            gold_logits: Reference model logits [seq_len, vocab].
            candidate_logits: Candidate model logits [seq_len, vocab].

        Returns:
            KL-divergence value (lower = better quality match).
        """
        if gold_logits.shape != candidate_logits.shape:
            raise ValueError("Logit shapes must match")

        gold_probs = torch.softmax(gold_logits.float(), dim=-1)
        candidate_probs = torch.softmax(candidate_logits.float(), dim=-1)

        # Add epsilon to avoid log(0)
        eps = 1e-8
        kl_div = torch.sum(gold_probs * torch.log((gold_probs + eps) / (candidate_probs + eps)), dim=-1)

        return kl_div.mean().item()

    @staticmethod
    def infer_quality_tier_from_request(prompt: str) -> QualityTier:
        """Infer the quality tier from the request prompt.

        Heuristic-based: code generation and creative writing -> HIGH,
        general Q&A -> MEDIUM, classification/extraction -> LOW.
        """
        prompt_lower = prompt.lower()

        # High quality indicators
        high_keywords = ["write a story", "write code", "create", "generate", "compose",
                         "def ", "class ", "function "]
        if any(kw in prompt_lower for kw in high_keywords):
            return QualityTier.HIGH

        # Low quality indicators
        low_keywords = ["classify", "category", "label", "yes or no", "true or false",
                        "extract", "sentiment"]
        if any(kw in prompt_lower for kw in low_keywords):
            return QualityTier.LOW

        return QualityTier.MEDIUM
