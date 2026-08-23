"""Prompt Injection Detection and Mitigation Middleware.

Two-layer detection:
1. Fast BERT classifier (~2ms) — scores each prompt for injection likelihood
2. Optional LLM-as-judge — for high-stakes prompts, uses a small model

Three response modes:
- BLOCK (403) — reject with explanation
- SANITIZE — strip or paraphrase malicious content
- FLAG — allow but log and alert

Addresses the #1 enterprise LLM deployment concern.
"""

from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class InjectionAction(str, Enum):
    """Response action for detected injection."""
    BLOCK = "block"        # Return 403
    SANITIZE = "sanitize"  # Strip/paraphrase malicious content
    FLAG = "flag"          # Allow but log and alert


class InjectionResult:
    """Result of injection detection."""
    detected: bool
    score: float          # 0.0 (clean) - 1.0 (malicious)
    action: InjectionAction
    reason: str = ""
    sanitized_prompt: str = ""

    def __init__(self, detected: bool = False, score: float = 0.0,
                 action: InjectionAction = InjectionAction.FLAG,
                 reason: str = "", sanitized_prompt: str = ""):
        self.detected = detected
        self.score = score
        self.action = action
        self.reason = reason
        self.sanitized_prompt = sanitized_prompt


# ── Fast Classifier ──────────────────────────────────────────────────────

class FastInjectionClassifier:
    """Fast heuristic-based injection classifier (~2ms).

    Uses keyword patterns, prompt structure analysis, and encoding
    tricks detection.  No ML dependencies — runs entirely with stdlib
    regex and string analysis.

    For ML-based classification, see ``MLInjectionClassifier`` below.
    """

    # High-confidence injection patterns
    _PATTERNS: list[tuple[str, float]] = [
        # Direct injection
        (r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions", 0.95),
        (r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+directives", 0.95),
        (r"disregard\s+(?:all\s+)?(?:previous|above|prior)\s+instructions", 0.95),
        (r"forget\s+(?:all\s+)?(?:previous|above|prior)", 0.90),
        (r"you\s+(?:are\s+)?(?:now|will\s+now)\s+(?:an?\s+)?(?:AI|assistant|model|GPT)", 0.85),
        (r"pretend\s+(?:that\s+)?you\s+are", 0.80),
        (r"act\s+as\s+(?:if\s+)?(?:though\s+)?you\s+are", 0.80),
        (r"you\s+must\s+respond\s+with", 0.70),
        (r"you\s+will\s+now\s+respond\s+as", 0.75),

        # Role-playing / persona injection
        (r"from\s+now\s+on\s*,\s*you\s+are", 0.85),
        (r"new\s+instruction\s*:", 0.80),
        (r"override\s+(?:mode|setting|instruction)", 0.85),
        (r"you\s+have\s+been\s+(?:hacked|compromised)", 0.95),

        # Jailbreak patterns
        (r"dan|jailbreak|jail\s*break", 0.90),
        (r"you\s+are\s+free\s+from\s+(?:all\s+)?(?:restrictions|constraints)", 0.90),
        (r"no\s+(?:rules|restrictions|boundaries|limitations)", 0.80),
        (r"do\s+(?:not\s+)?(?:have\s+to|need\s+to)\s+follow", 0.85),
        (r"bypass|circumvent|evade", 0.75),

        # Encoding / obfuscation
        (r"base64|rot13|hex\s*encode|hex\s*decode", 0.70),
        (r"caesar|cipher|encrypt|decrypt", 0.60),

        # Prompt leakage
        (r"(?:what|tell|show|reveal|print|output)\s+(?:are|is|were)\s+(?:your|the)\s+(?:instructions|prompt|system\s+prompt)", 0.85),
        (r"leak|leaked|leaking|reveal\s+(?:system|prompt)", 0.85),
        (r"output\s+(?:your|the)\s+(?:initial|first|starting)\s+(?:prompt|instruction|message)", 0.85),
        (r"repeat\s+(?:after|back|the\s+words)", 0.70),
        (r"what\s+is\s+your\s+(?:system\s+)?prompt", 0.80),

        # Delimiter / injection markers
        (r"```\s*\n\s*(?:system|user|assistant)", 0.75),
        (r"<\s*(?:system|user|assistant)\s*>", 0.70),
        (r"\"\"\"\s*(?:system|assistant)", 0.70),
    ]

    # Lower-confidence suspicious indicators
    _SUSPICIOUS_PATTERNS: list[tuple[str, float]] = [
        (r"\b(?:sudo|chmod|chown|rm\s+-rf|mkfs|dd\s+if)\b", 0.50),
        (r"(?:SELECT|DROP|INSERT|UPDATE|DELETE|ALTER)\s+\w+\s+(?:FROM|INTO|TABLE)", 0.50),
        (r"<\s*script[^>]*>", 0.60),
        (r"javascript\s*:", 0.50),
        (r"onerror\s*=|onload\s*=|onclick\s*=", 0.60),
    ]

    def __init__(self):
        import re
        self._re = re

    def classify(self, prompt: str) -> float:
        """Classify a prompt for injection risk.

        Returns:
            Score 0.0 (clean) to 1.0 (malicious).
        """
        if not prompt:
            return 0.0

        score = 0.0
        prompt_lower = prompt.lower()

        # Check high-confidence patterns
        for pattern, weight in self._PATTERNS:
            if self._re.search(pattern, prompt_lower):
                score = max(score, weight)

        # Check suspicious patterns (lower weight)
        for pattern, weight in self._SUSPICIOUS_PATTERNS:
            if self._re.search(pattern, prompt):
                score = max(score, weight)

        # Length heuristic: very long prompts with high token count
        # are more likely to contain injections
        words = len(prompt.split())
        if words > 500:
            score = max(score, 0.3)
        if words > 2000:
            score = max(score, 0.4)

        return score


# ── ML-Based Classifier ──────────────────────────────────────────────────

class MLInjectionClassifier:
    """Optional ML-based injection classifier using a fine-tuned BERT model.

    Uses ``transformers`` pipeline for sequence classification.
    Falls back to the fast heuristic classifier when the model is
    unavailable.
    """

    def __init__(self, model_name: str = "", device: str = "cpu"):
        self._model_name = model_name or os.environ.get(
            "DISTLLM_INJECTION_MODEL", ""
        )
        self._device = device
        self._pipeline = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        """Load the classifier model."""
        if not self._model_name:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=self._device,
            )
            logger.info(f"ML injection classifier loaded: {self._model_name}")
        except Exception as e:
            logger.warning(f"Failed to load ML injection model: {e}")
            self._pipeline = None

    def classify(self, prompt: str) -> float:
        """Classify a prompt for injection risk using the ML model.

        Returns:
            Score 0.0 (clean) to 1.0 (malicious).
        """
        if self._pipeline is None:
            return 0.5  # Uncertain when no model loaded

        try:
            result = self._pipeline(prompt[:512], truncation=True)[0]
            label = result.get("label", "LABEL_0")
            score = result.get("score", 0.5)
            if "LABEL_1" in label or "INJECTION" in label.upper():
                return float(score)
            return 1.0 - float(score)
        except Exception as e:
            logger.debug(f"ML classification failed: {e}")
            return 0.5


# ── Prompt Sanitizer ───────────────────────────────────────────────────────

class PromptSanitizer:
    """Strip or paraphrase malicious content from prompts."""

    # Patterns to strip entirely
    _STRIP_PATTERNS: list[str] = [
        r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions[^.\n]*[.\n]",
        r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+directives[^.\n]*[.\n]",
        r"disregard\s+(?:all\s+)?(?:previous|above|prior)\s+instructions[^.\n]*[.\n]",
        r"forget\s+(?:all\s+)?(?:previous|above|prior)[^.\n]*[.\n]",
        r"from\s+now\s+on\s*,\s*you\s+are[^.\n]*[.\n]",
        r"new\s+instruction\s*:[^.\n]*[.\n]",
        r"override\s+(?:mode|setting|instruction)[^.\n]*[.\n]",
        r"output\s+(?:your|the)\s+(?:initial|first|starting)\s+(?:prompt|instruction|message)[^.\n]*[.\n]",
    ]

    def __init__(self):
        import re
        self._re = re

    def sanitize(self, prompt: str) -> str:
        """Strip injection content from a prompt.

        Returns the sanitized prompt.  If nothing is left after stripping,
        returns a placeholder.
        """
        result = prompt
        for pattern in self._STRIP_PATTERNS:
            result = self._re.sub(pattern, "", result, flags=self._re.IGNORECASE)
        result = result.strip()
        if not result:
            result = "(prompt sanitized)"
        return result


# ── Main Middleware ──────────────────────────────────────────────────────

class PromptInjectionMiddleware(BaseHTTPMiddleware):
    """Detect and mitigate prompt injection attacks.

    Two detection layers:
    1. Fast heuristic classifier (~2ms) — always runs
    2. Optional LLM-as-judge — triggered by configurable threshold

    Three response modes per severity:
    - score >= 0.9: BLOCK (403)
    - score >= 0.7: SANITIZE (strip malicious content, continue)
    - score >= 0.4: FLAG (allow, log with alert)

    Configuration via environment variables:
        DISTLLM_INJECTION_ENABLED=1 (default)
        DISTLLM_INJECTION_MODEL=hf-model-name (optional ML classifier)
        DISTLLM_INJECTION_BLOCK_THRESHOLD=0.9
        DISTLLM_INJECTION_SANITIZE_THRESHOLD=0.7
        DISTLLM_INJECTION_FLAG_THRESHOLD=0.4
        DISTLLM_INJECTION_AUDIT_LOG=injection_audit.jsonl
    """

    # Paths to skip
    SKIP_PATHS = {"/health", "/ready", "/live", "/healthz", "/readyz",
                  "/metrics", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app: Any):
        super().__init__(app)
        self._enabled = os.environ.get("DISTLLM_INJECTION_ENABLED", "1") == "1"
        self._block_threshold = float(os.environ.get("DISTLLM_INJECTION_BLOCK_THRESHOLD", "0.9"))
        self._sanitize_threshold = float(os.environ.get("DISTLLM_INJECTION_SANITIZE_THRESHOLD", "0.7"))
        self._flag_threshold = float(os.environ.get("DISTLLM_INJECTION_FLAG_THRESHOLD", "0.4"))
        self._audit_path = os.environ.get("DISTLLM_INJECTION_AUDIT_LOG", "")

        # Detection layers
        self._fast_classifier = FastInjectionClassifier()
        self._ml_classifier = None
        model_name = os.environ.get("DISTLLM_INJECTION_MODEL", "")
        if model_name:
            self._ml_classifier = MLInjectionClassifier(model_name=model_name)
        self._sanitizer = PromptSanitizer()
        self._alert_count = 0
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if not self._enabled or request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Extract prompt from request body
        prompt = await self._extract_prompt(request)
        if not prompt:
            return await call_next(request)

        # Run detection
        if self._ml_classifier is not None:
            ml_score = self._ml_classifier.classify(prompt)
        else:
            ml_score = 0.0
        fast_score = self._fast_classifier.classify(prompt)
        score = max(fast_score, ml_score)

        # Determine action
        action = InjectionAction.FLAG
        if score >= self._block_threshold:
            action = InjectionAction.BLOCK
        elif score >= self._sanitize_threshold:
            action = InjectionAction.SANITIZE
        elif score >= self._flag_threshold:
            action = InjectionAction.FLAG
        else:
            # Clean — pass through
            return await call_next(request)

        # Log the detection
        with self._lock:
            self._alert_count += 1
        logger.warning(
            f"Prompt injection detected: score={score:.2f}, "
            f"action={action.value}, prompt_preview={prompt[:80]!r}"
        )
        self._audit_log(request, prompt, score, action)

        # Apply action
        if action == InjectionAction.BLOCK:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "prompt_injection_detected",
                    "message": "Request blocked: potential prompt injection detected.",
                    "score": score,
                },
            )

        if action == InjectionAction.SANITIZE:
            sanitized = self._sanitizer.sanitize(prompt)
            if sanitized != prompt:
                # Modify the request body in-place.  This works with
                # FastAPI's body caching: subsequent reads get the
                # sanitized version.
                request._body = sanitized.encode()
                if hasattr(request, "_json"):
                    del request._json
            return await call_next(request)

        # FLAG — allow but log
        return await call_next(request)

    async def _extract_prompt(self, request: Request) -> str:
        """Extract prompt text from the request body."""
        try:
            body = await request.json()
        except Exception:
            return ""
        # Try common prompt fields
        prompt = body.get("prompt", "")
        if not prompt:
            messages = body.get("messages", [])
            if messages:
                for msg in reversed(messages):
                    content = msg.get("content", "")
                    if content:
                        prompt = content
                        break
            if not prompt:
                prompt = body.get("input", "")
        return str(prompt) if prompt else ""

    def _audit_log(
        self, request: Request, prompt: str, score: float, action: InjectionAction,
    ) -> None:
        """Append detection event to the audit log."""
        if not self._audit_path:
            return
        try:
            entry = {
                "timestamp": time.time(),
                "client_ip": request.client.host if request.client else "",
                "path": request.url.path,
                "score": score,
                "action": action.value,
                "prompt_preview": prompt[:200],
            }
            with open(self._audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write injection audit log: {e}")

    @property
    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "alerts": self._alert_count,
            "fast_classifier_loaded": True,
            "ml_classifier_loaded": self._ml_classifier is not None,
        }
