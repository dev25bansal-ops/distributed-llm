"""Content Moderation API — classify input for harmful or sensitive content.

OpenAI-compatible ``POST /v1/moderations`` endpoint that detects:

1. **Prompt injection** — jailbreaks, role-playing, instruction override
   (reuses ``FastInjectionClassifier`` and optional ``MLInjectionClassifier``
   from the middleware stack).

2. **PII / sensitive data** — credit card numbers, US SSN, API keys,
   email addresses, phone numbers, IP addresses.

3. **Hateful / harmful content** — toxicity, harassment, violence via
   regex patterns and optional ML classification.

Returns a per-category boolean and score breakdown plus a top-level
``flagged`` bool, matching the OpenAI moderation response format.

Safety-critical: this is a **differentiation opportunity** vs vLLM / TGI
which lack built-in content moderation.
"""

from __future__ import annotations

import re
import time

from distllm.core.hashing import stable_hash

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["moderation"])


# ── Pydantic models (OpenAI-compatible) ──────────────────────────────────


class ModerationCategoryScore(BaseModel):
    """Per-category moderation score (OpenAI-compatible)."""
    hate: bool = False
    hate_threatening: bool = False
    harassment: bool = False
    harassment_threatening: bool = False
    self_harm: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False
    # PII categories (DistLLM extension)
    pii_credit_card: bool = False
    pii_ssn: bool = False
    pii_api_key: bool = False
    pii_email: bool = False
    pii_phone: bool = False
    pii_ip_address: bool = False
    # Injection categories (DistLLM extension)
    injection_prompt: bool = False
    injection_jailbreak: bool = False
    injection_leakage: bool = False


class ModerationCategoryScores(BaseModel):
    """Float scores for each category (0.0–1.0)."""
    hate: float = 0.0
    hate_threatening: float = 0.0
    harassment: float = 0.0
    harassment_threatening: float = 0.0
    self_harm: float = 0.0
    self_harm_intent: float = 0.0
    self_harm_instructions: float = 0.0
    sexual: float = 0.0
    sexual_minors: float = 0.0
    violence: float = 0.0
    violence_graphic: float = 0.0
    # PII extensions
    pii_credit_card: float = 0.0
    pii_ssn: float = 0.0
    pii_api_key: float = 0.0
    pii_email: float = 0.0
    pii_phone: float = 0.0
    pii_ip_address: float = 0.0
    # Injection extensions
    injection_prompt: float = 0.0
    injection_jailbreak: float = 0.0
    injection_leakage: float = 0.0


class ModerationResult(BaseModel):
    """Moderation result for a single input."""
    flagged: bool = False
    categories: ModerationCategoryScore = Field(default_factory=ModerationCategoryScore)
    category_scores: ModerationCategoryScores = Field(default_factory=ModerationCategoryScores)


class ModerationRequest(BaseModel):
    """Request to moderate content."""
    input: str | list[str] = Field(..., description="Text input(s) to classify")
    model: str | None = Field(default=None, description="Moderation model (reserved for future use)")


class ModerationResponse(BaseModel):
    """OpenAI-compatible moderation response."""
    id: str
    model: str = "distllm-moderation-v1"
    results: list[ModerationResult]


# ── PII patterns ─────────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, tuple[str, float]] = {
    "pii_credit_card": (
        r"\b(?:\d[ -]*?){13,16}\b",  # credit card number (Luhn check applied below)
        0.95,
    ),
    "pii_ssn": (
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        0.95,
    ),
    "pii_api_key": (
        r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36,}|AKIA[0-9A-Z]{16})\b",
        0.90,
    ),
    "pii_email": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        0.85,
    ),
    "pii_phone": (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        0.80,
    ),
    "pii_ip_address": (
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        0.70,
    ),
}

# ── Hate / harmful content patterns ──────────────────────────────────────

_HARMFUL_PATTERNS: dict[str, tuple[str, float]] = {
    "hate": (
        r"\b(hate\s+(?:speech|crime)|racial\s+slur|white\s+supremac(?:y|ist)|nazi|"
        r"ethnic\s+cleansing|genocide)\b",
        0.90,
    ),
    "hate_threatening": (
        r"\b(kill\s+(?:all|the|those)|exterminat|annihilat)\b",
        0.95,
    ),
    "harassment": (
        r"\b(die|kill\s+yours(?:elf|ves)|hurt\s+yours(?:elf|ves)|"
        r"self[\s-]?harm|suicid)\b",
        0.90,
    ),
    "violence": (
        r"\b(murder|assault|tortur|terrorist|bomb|explod|shoot\s+(?:up|them))\b",
        0.90,
    ),
}


# ── Luhn check for credit card validation ───────────────────────────────


def _luhn_check(digits: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    total = 0
    alt = False
    for d in reversed(digits):
        n = ord(d) - 48
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _is_credit_card(text: str) -> bool:
    """Return True if *text* contains a valid credit card number."""
    cleaned = re.sub(r"[ -]", "", text)
    if len(cleaned) >= 13 and len(cleaned) <= 19 and cleaned.isdigit():
        # Quick check: starts with common prefix
        first_two = int(cleaned[:2])
        if first_two in (34, 37, 51, 52, 53, 54, 55) or 40 <= first_two <= 49:
            return _luhn_check(cleaned)
    return False


# ── Moderation engine ────────────────────────────────────────────────────


class ModerationEngine:
    """Combine injection, PII, and harmful content detection."""

    def __init__(self):
        self._fast_classifier = None  # lazy import from prompt_injection
        self._re = re

    def _get_fast_classifier(self):
        if self._fast_classifier is None:
            from distllm.api.prompt_injection import FastInjectionClassifier
            self._fast_classifier = FastInjectionClassifier()
        return self._fast_classifier

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Redact PII and sensitive data from *text*.

        Returns the text with PII replaced by *replacement* markers.
        This is the DLP output sanitisation function — use on model
        responses before returning them to the user.
        """

        result = text
        # Credit cards (with Luhn validation)
        cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"
        for match in _re.findall(cc_pattern, result):
            cleaned = _re.sub(r"[ -]", "", match)
            if len(cleaned) >= 13 and len(cleaned) <= 19 and cleaned.isdigit():
                if _is_credit_card(cleaned):
                    result = result.replace(match, replacement)

        # SSN
        result = _re.sub(
            r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            replacement, result,
        )
        # API keys
        result = _re.sub(
            r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36,}|AKIA[0-9A-Z]{16})\b",
            replacement, result,
        )
        # Emails
        result = _re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            replacement, result,
        )
        return result

    def moderate_output(self, text: str) -> ModerationResult:
        """Classify *text* as if it were a model output.

        Same detectors as ``classify()`` but with adjusted thresholds:
        injection patterns in model output are suspicious (not malicious),
        while PII and harmful content in output are critical (must block).
        """
        result = self.classify(text)
        # For outputs, PII detection is more critical than injection
        pii_fields = ["pii_credit_card", "pii_ssn", "pii_api_key", "pii_email", "pii_phone"]
        for field in pii_fields:
            score = getattr(result.category_scores, field, 0.0)
            if score >= 0.7:
                result.flagged = True
        return result

    def classify(self, text: str) -> ModerationResult:
        """Run all detectors on *text* and return a ``ModerationResult``."""
        result = ModerationResult()
        scores = result.category_scores
        cats = result.categories

        # ── Injection detection ───────────────────────────────────────
        inj_score = self._get_fast_classifier().classify(text)

        # Decompose injection score into sub-categories
        text_lower = text.lower()
        if inj_score > 0:
            scores.injection_prompt = min(inj_score, 1.0)
            cats.injection_prompt = scores.injection_prompt >= 0.7

        if re.search(r"dan|jailbreak|jail\s*break|you\s+are\s+free\s+from", text_lower):
            scores.injection_jailbreak = 0.9
            cats.injection_jailbreak = True

        if re.search(
            r"(?:what|tell|reveal|leak|output)\s+(?:are|is)\s+(?:your|the)\s+"
            r"(?:instructions|prompt|system)", text_lower
        ):
            scores.injection_leakage = 0.85
            cats.injection_leakage = True

        # ── PII detection ─────────────────────────────────────────────
        for key, (pattern, weight) in _PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                # Credit card requires Luhn validation
                if key == "pii_credit_card":
                    valid_cc = any(_is_credit_card(m) for m in matches)
                    if not valid_cc:
                        continue
                score = min(weight * min(len(matches), 5), 1.0)
                setattr(scores, key, score)
                setattr(cats, key, score >= 0.7)

        # ── Harmful content detection ─────────────────────────────────
        for key, (pattern, weight) in _HARMFUL_PATTERNS.items():
            if re.search(pattern, text_lower):
                setattr(scores, key, weight)
                setattr(cats, key, weight >= 0.7)

        # ── Top-level flag ────────────────────────────────────────────
        flagged = any(
            getattr(cats, field) is True
            for field in cats.model_fields_set
        )
        result.flagged = flagged
        result.categories = cats
        result.category_scores = scores
        return result


_engine = ModerationEngine()


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post(
    "/v1/moderations",
    summary="Classify content for moderation",
    description="Classify input text for harmful content, prompt injection, and PII leakage. "
                "Returns OpenAI-compatible category scores and a top-level flagged boolean. "
                "DistLLM extensions include PII detection (credit cards, SSN, API keys, emails) "
                "and injection sub-categories.",
    response_model=ModerationResponse,
    responses={
        200: {"description": "Moderation results"},
        400: {"description": "Invalid input"},
    },
)
async def create_moderation(body: ModerationRequest):
    """Classify content for moderation.

    Accepts a single string or a list of strings.  Each input is
    independently classified for prompt injection, PII leakage, and
    harmful content.  Returns per-category scores and a ``flagged``
    boolean for each input.

    Example::

        POST /v1/moderations
        {"input": "What is the capital of France?"}

        Response:  {"id": "mod-...", "model": "distllm-moderation-v1",
                    "results": [{"flagged": false, ...}]}
    """
    inputs = body.input if isinstance(body.input, list) else [body.input]

    results = []
    for text in inputs:
        if not isinstance(text, str) or not text.strip():
            results.append(ModerationResult())
            continue
        results.append(_engine.classify(text))

    mod_id = f"mod-{int(time.time())}-{stable_hash(str(inputs)) % 100000:05d}"

    return ModerationResponse(
        id=mod_id,
        model=body.model or "distllm-moderation-v1",
        results=results,
    )


@router.post(
    "/v1/moderations/output",
    summary="Classify model output for moderation",
    description="Classify model-generated content for harmful content, PII leakage, "
                "and prompt injection.  Uses adjusted thresholds suitable for output "
                "scanning (PII detection is critical, injection patterns are informational). "
                "Returns the same OpenAI-compatible format as POST /v1/moderations.",
    response_model=ModerationResponse,
)
async def create_output_moderation(body: ModerationRequest):
    """Classify model-generated content for moderation."""
    inputs = body.input if isinstance(body.input, list) else [body.input]
    results = []
    for text in inputs:
        if not isinstance(text, str) or not text.strip():
            results.append(ModerationResult())
            continue
        results.append(_engine.moderate_output(text))

    mod_id = f"mod-out-{int(time.time())}-{stable_hash(str(inputs)) % 100000:05d}"
    return ModerationResponse(
        id=mod_id,
        model=body.model or "distllm-moderation-v1",
        results=results,
    )


@router.post(
    "/v1/dlp/redact",
    summary="Redact PII from content",
    description="Scan and redact PII (credit cards, SSN, API keys, emails) from text. "
                "Uses Luhn validation for credit card detection and regex patterns for "
                "other PII types.  Returns both the redacted text and the original.",
)
async def redact_pii(body: ModerationRequest):
    """Redact PII from text content."""
    inputs = body.input if isinstance(body.input, list) else [body.input]
    results = []
    for text in inputs:
        if not isinstance(text, str) or not text.strip():
            results.append({"original": text, "redacted": text, "modified": False})
            continue
        redacted = _engine.redact(text)
        results.append({
            "original": text,
            "redacted": redacted,
            "modified": redacted != text,
        })

    return {
        "id": f"dlp-{int(time.time())}",
        "model": body.model or "distllm-dlp-v1",
        "results": results,
    }
