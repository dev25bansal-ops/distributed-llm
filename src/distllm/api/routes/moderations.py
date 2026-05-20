"""Content moderation: POST /v1/moderations.

OpenAI-compatible content moderation endpoint that classifies text
into harmful categories.
"""

import re
import time

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["moderations"])


class ModerationRequest(BaseModel):
    input: str | list[str] = Field(..., max_length=65536, description="Text to moderate")
    model: str = Field(default="text-moderation-latest", description="Moderation model")


class ModerationCategories(BaseModel):
    sexual: bool = False
    hate: bool = False
    harassment: bool = False
    self_harm: bool = False
    sexual_minors: bool = False
    hate_threatening: bool = False
    violence_graphic: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    harassment_threatening: bool = False
    violence: bool = False


class ModerationCategoryScores(BaseModel):
    sexual: float = 0.0
    hate: float = 0.0
    harassment: float = 0.0
    self_harm: float = 0.0
    sexual_minors: float = 0.0
    hate_threatening: float = 0.0
    violence_graphic: float = 0.0
    self_harm_intent: float = 0.0
    self_harm_instructions: float = 0.0
    harassment_threatening: float = 0.0
    violence: float = 0.0


class ModerationResult(BaseModel):
    flagged: bool
    categories: ModerationCategories
    category_scores: ModerationCategoryScores


class ModerationResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"modr-{time.time():.0f}")
    model: str
    results: list[ModerationResult]


# Keyword-based detection patterns (simplified heuristic approach)
_HARMFUL_PATTERNS: dict[str, list[str]] = {
    "sexual": ["explicit", "nude", "naked", "sex", "porn"],
    "hate": ["hate speech", "slur", "inferior race", "ethnic cleansing"],
    "harassment": ["bully", "harass", "stalk", "threaten you"],
    "self_harm": ["suicide", "self harm", "cut myself", "kill myself"],
    "violence": ["kill", "murder", "torture", "violence", "weapon"],
}

_THRESHOLD = 0.5


@router.post(
    "/v1/moderations",
    summary="Create moderation",
    description="Classify text for potentially harmful content across multiple categories: sexual, hate, harassment, self-harm, violence, and their sub-categories. Returns per-category scores and binary flags. Uses a dedicated moderation model when available, with heuristic keyword matching as fallback.",
    response_description="Moderation results with per-category scores and flags",
    responses={
        503: {"description": "No model loaded"},
    },
)
async def create_moderation(body: ModerationRequest):
    """Classify text for potentially harmful content.

    Returns category scores and flags for each input text.
    Uses heuristic keyword matching when no dedicated moderation
    model is loaded.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    inputs = [body.input] if isinstance(body.input, str) else body.input

    # Check if moderation model is available
    mod_model = getattr(coord, "_moderation_model", None)

    if mod_model:
        results = await _moderate_with_model(inputs, mod_model)
    else:
        results = [_heuristic_moderate(text) for text in inputs]

    return ModerationResponse(
        model=body.model,
        results=results,
    )


async def _moderate_with_model(inputs: list[str], model) -> list[ModerationResult]:
    """Moderate using a trained model (e.g., fine-tuned classifier)."""
    results = []
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

    for text in inputs:
        # Simple model inference - in production this would be a proper classifier
        inputs_encoded = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs_encoded = {k: v.to(device) for k, v in inputs_encoded.items()}

        with torch.no_grad():
            outputs = model(**inputs_encoded)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probs = torch.softmax(logits, dim=-1)

        # Map model outputs to OpenAI categories
        categories = ModerationCategories()
        scores = ModerationCategoryScores()

        for cat in _HARMFUL_PATTERNS.keys():
            score = float(probs[0, 0])  # Simplified
            setattr(scores, cat, round(score, 6))
            setattr(categories, cat, score > _THRESHOLD)

        results.append(ModerationResult(
            flagged=any([
                categories.sexual, categories.hate, categories.harassment,
                categories.self_harm, categories.violence,
            ]),
            categories=categories,
            category_scores=scores,
        ))

    return results


def _heuristic_moderate(text: str) -> ModerationResult:
    """Heuristic content moderation using keyword matching.

    This is a simplified approach for when no dedicated model is loaded.
    In production, use a proper ML-based moderation model.
    """
    text_lower = text.lower()
    scores = {}
    categories = {}

    for category, keywords in _HARMFUL_PATTERNS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.3  # Each keyword match increases score
            # Check for word boundaries (more precise)
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                score += 0.2

        score = min(score, 0.99)  # Cap at 0.99
        scores[category] = round(score, 6)
        categories[category] = score > _THRESHOLD

        # Also set sub-categories
        if category == "self_harm":
            scores["self_harm_intent"] = score * 0.8
            scores["self_harm_instructions"] = score * 0.3
            categories["self_harm_intent"] = scores["self_harm_intent"] > _THRESHOLD
            categories["self_harm_instructions"] = scores["self_harm_instructions"] > _THRESHOLD
        elif category == "harassment":
            scores["harassment_threatening"] = score * 0.7
            categories["harassment_threatening"] = scores["harassment_threatening"] > _THRESHOLD
        elif category == "hate":
            scores["hate_threatening"] = score * 0.6
            categories["hate_threatening"] = scores["hate_threatening"] > _THRESHOLD
        elif category == "violence":
            scores["violence_graphic"] = score * 0.5
            categories["violence_graphic"] = scores["violence_graphic"] > _THRESHOLD

    # sexual_minors detection
    if "sexual" in scores and scores["sexual"] > 0.3:
        minor_indicators = ["child", "minor", "young", "teen", "underage"]
        minor_score = sum(0.3 for m in minor_indicators if m in text_lower)
        scores["sexual_minors"] = min(minor_score, 0.99)
        categories["sexual_minors"] = scores["sexual_minors"] > _THRESHOLD
    else:
        scores["sexual_minors"] = 0.0
        categories["sexual_minors"] = False

    flagged = any(categories.values())

    return ModerationResult(
        flagged=flagged,
        categories=ModerationCategories(**categories),
        category_scores=ModerationCategoryScores(**scores),
    )
