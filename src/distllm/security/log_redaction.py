"""Log redaction — strip PII from log messages before writing.

Integrates with the existing PIIInspector from request_auditor.py
to provide safe logging wrappers and exception redaction.

Usage:
    from distllm.security.log_redaction import redact, redact_exception

    logger.info(redact(f"User prompt: {prompt}"))  # PII stripped
    logger.error(f"Request failed: {redact_exception(e)}")  # Safe error
"""

from __future__ import annotations

import logging
import os
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
    # High-entropy base64url tokens — e.g. the ApiKeyStore auto-generates keys
    # via secrets.token_urlsafe(48) with NO 'sk-' prefix and no 'api_key='
    # context, so the patterns above would let them leak into logs. A 40+ char
    # contiguous base64url run is a strong token signal.
    "long_token": re.compile(r"\b[A-Za-z0-9_-]{40,}\b"),
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


# ---------------------------------------------------------------------------
# Global log-redaction sink (logging.Filter on the root logger's handlers)
# ---------------------------------------------------------------------------
#
# The per-call ``LogRedactor.redact(...)`` wrapper only protects messages that
# are explicitly wrapped.  A stray ``logger.info(f"token={raw}")`` elsewhere in
# the codebase (or in a third-party library) still leaks.  ``RedactingFilter``
# is installed on the *handlers* of the root logger so that EVERY emitted
# record -- regardless of which logger produced it, including child / third
# party loggers that propagate to root -- is redacted before it reaches any
# handler/output.
#
# Why a handler filter and not a logger filter?
#   ``logging.Filter`` instances attached to a Logger only run for records
#   emitted *by that exact logger*.  Filters attached to a Handler run for every
#   record the handler emits, and child loggers propagate their records up to
#   the root logger's handlers.  So a filter on a root handler is the only
#   reliable "global sink" that also covers third-party libraries.

# Secret-shaped patterns layered on top of the PII set above.
_SECRET_PATTERNS: dict[str, re.Pattern] = {
    # OpenAI-style keys (sk-...) and explicit api_key="..." assignments
    "api_key": re.compile(
        r"(?i)(sk-[a-zA-Z0-9]{20,}|api[-_]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])"
    ),
    # AWS access key ids
    "aws_key": re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
    # OAuth / bearer tokens
    "bearer": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"),
    # Authorization / Proxy-Authorization header style values
    "authorization": re.compile(
        r"(?i)(authorization|proxy-authorization)\s*[:=]\s*['\"]?[A-Za-z0-9\-._~+/]+=*"
    ),
    # password / secret / token style assignments
    "credential_assignment": re.compile(
        r"(?i)(password|passwd|pwd|secret|client[_-]?secret|token|"
        r"api[_-]?token|access[_-]?token|refresh[_-]?token)\s*[:=]\s*"
        r"['\"]?[^\s'\"]{3,}['\"]?"
    ),
    # Long hexadecimal tokens (>= 32 chars): hashes, key material, org ids.
    "long_hex_token": re.compile(r"\b[a-fA-F0-9]{32,}\b"),
    # Long base64/url-safe tokens (>= 40 chars).
    "long_b64_token": re.compile(r"\b[A-Za-z0-9_\-+/]{40,}={0,2}\b"),
}

# The complete default pattern set: PII + explicit secrets.
DEFAULT_PATTERNS: dict[str, re.Pattern] = {
    **_PII_PATTERNS,
    **_SECRET_PATTERNS,
}

# Environment variable keys.
ENV_REDACTION_ENABLED = "DISTLLM_LOG_REDACTION"  # "off"/"0"/"false"/"no" disables
ENV_REDACTION_PATTERNS = "DISTLLM_REDACT_PATTERNS"  # extra ";"-separated regexes


class RedactingFilter(logging.Filter):
    """A :class:`logging.Filter` that redacts secrets from every record.

    Attach an instance to a handler (typically on the root logger) so that all
    records -- including those propagated from child / third-party loggers --
    have their formatted message scrubbed before being emitted.

    The record's ``msg`` is replaced with the redacted text and ``args`` is
    cleared, so downstream formatters never re-apply ``%``-formatting and the
    redacted string is what ultimately reaches the output.
    """

    REDACTION_TOKEN = "[REDACTED]"

    def __init__(
        self,
        name: str = "",
        *,
        patterns: dict[str, re.Pattern] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name)
        self.enabled = enabled
        self.patterns: list[re.Pattern] = list((patterns or DEFAULT_PATTERNS).values())

    @classmethod
    def from_env(cls, name: str = "") -> "RedactingFilter":
        """Build a filter honouring ``DISTLLM_LOG_REDACTION`` / ``DISTLLM_REDACT_PATTERNS``."""
        enabled = (
            os.environ.get(ENV_REDACTION_ENABLED, "on").strip().lower()
            not in ("off", "0", "false", "no", "")
        )
        patterns: dict[str, re.Pattern] = dict(DEFAULT_PATTERNS)
        extra = os.environ.get(ENV_REDACTION_PATTERNS)
        if extra:
            for i, raw in enumerate(p.strip() for p in extra.split(";") if p.strip()):
                try:
                    patterns[f"custom_{i}"] = re.compile(raw)
                except re.error:
                    # Ignore malformed custom pattern rather than crash logging.
                    continue
        return cls(name=name, patterns=patterns, enabled=enabled)

    def redact_message(self, text: str) -> str:
        """Redact every configured secret/PII pattern from ``text``."""
        for pattern in self.patterns:
            text = pattern.sub(self.REDACTION_TOKEN, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """Mutate the record in place to carry a redacted message.

        Always returns ``True`` so the record is never dropped -- redaction is
        supposed to mask, not to suppress.  If redaction fails for any reason we
        leave the record untouched rather than risk dropping or corrupting logs.
        """
        if not self.enabled:
            return True
        try:
            message = record.getMessage()
            record.msg = self.redact_message(message)
            record.args = ()
        except Exception:
            pass
        return True


def install_global_redaction(
    logger: logging.Logger | None = None,
    *,
    force: bool = False,
) -> RedactingFilter:
    """Install a :class:`RedactingFilter` globally.

    Attaches the filter to every handler currently on the root logger (or the
    supplied logger).  Because child loggers propagate their records to the root
    logger's handlers, this covers *all* loggers in the process -- including
    third-party libraries -- not just ``distllm``'s own loggers.

    Idempotent: if a ``RedactingFilter`` is already present on a handler it is
    left in place unless ``force=True``.

    Returns the filter instance that was installed (or the first existing one).
    """
    if logger is None:
        logger = logging.getLogger()

    redactor = RedactingFilter.from_env()

    # If redaction is explicitly disabled via env, do not install anything.
    if not redactor.enabled and not force:
        return redactor

    # 1) Attach to the logger itself.  This only runs for records the logger
    #    emits directly (e.g. third-party code calling ``logging.info`` which
    #    uses the root logger), but it is cheap and closes that gap.
    if not any(isinstance(f, RedactingFilter) for f in logger.filters) or force:
        logger.addFilter(redactor)

    # 2) Attach to every handler on the logger.  Child loggers propagate their
    #    records up to these handlers, so this is the real "global sink" that
    #    also covers third-party libraries.
    for handler in logger.handlers:
        if any(isinstance(f, RedactingFilter) for f in handler.filters):
            if not force:
                continue
        handler.addFilter(redactor)

    return redactor


# Convenience module-level functions
redact = LogRedactor.redact
redact_exception = LogRedactor.redact_exception
contains_pii = LogRedactor.contains_pii
