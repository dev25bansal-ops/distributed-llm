"""Log redaction — strip PII from log messages before writing.

Integrates with the existing PIIInspector from request_auditor.py
to provide safe logging wrappers and exception redaction.

Usage:
    from distllm.security.log_redaction import redact, redact_exception

    logger.info(redact(f"User prompt: {prompt}"))  # PII stripped
    logger.error(f"Request failed: {redact_exception(e)}")  # Safe error
"""

from __future__ import annotations

import re
from typing import Any


# PII patterns (consistent with request_auditor.PII_PATTERNS)
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "api_key": re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|api[-_]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])"),
    "aws_key": re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
}


class LogRedactor:
    """Strips PII from strings before logging.

    Applies regex patterns to detect and replace sensitive data.
    Thread-safe (pure function, no state).
    """

    REDACTION_TOKEN = "[REDACTED]"

    @classmethod
    def redact(cls, text: str) -> str:
        """Replace all PII matches with [REDACTED] token."""
        result = text
        for name, pattern in _PII_PATTERNS.items():
            result = pattern.sub(cls.REDACTION_TOKEN, result)
        return result

    @classmethod
    def redact_exception(cls, exc: BaseException) -> str:
        """Return a redacted error message from an exception.

        Extracts the string representation of the exception,
        strips PII, and returns the safe result.
        """
        msg = str(exc)
        return cls.redact(msg)

    @classmethod
    def contains_pii(cls, text: str) -> bool:
        """Check if text contains any PII patterns."""
        for pattern in _PII_PATTERNS.values():
            if pattern.search(text):
                return True
        return False


# Convenience module-level functions
redact = LogRedactor.redact
redact_exception = LogRedactor.redact_exception
contains_pii = LogRedactor.contains_pii
