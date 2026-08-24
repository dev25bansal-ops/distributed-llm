"""Log redaction — strip PII/secrets from log messages before writing.

Two integrations, installed together by :func:`install_global_redaction`:

1. **loguru** (the primary path): the DistLLM codebase emits through loguru,
   so a *core-level* ``logger.configure(patcher=...)`` is installed.  Loguru
   applies the core patcher to every record dict after %-formatting and
   before any sink writes, which means ALL existing ``logger.*`` calls across
   the codebase -- and any future ones -- get redacted transparently,
   regardless of which sinks are configured (default stderr, JSON sink,
   ``enqueue=True`` queues, Loki, ...).  No call site needs to change.

2. **stdlib logging**: a ``RedactingFilter`` on the root logger and its
   handlers, covering libraries that bridge into stdlib logging.

Per-call helpers remain available::

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


# ---------------------------------------------------------------------------
# Loguru global redaction (core-level patcher)
# ---------------------------------------------------------------------------
#
# 794 source files emit through ``loguru.logger``; none of them will ever be
# rewritten to call ``redact()`` by hand.  A *core-level* patcher installed via
# ``logger.configure(patcher=...)`` is applied by loguru to every record dict,
# process-wide, after %-format interpolation and before any sink writes.  This
# covers every existing and future ``logger.*`` call transparently, whatever
# sinks are configured (default stderr, JSON stdout, enqueue'd queues, Loki).
#
# Why a core patcher and not a per-sink filter?
#   - Sink filters only cover sinks added *after* installation; loguru's
#     default stderr handler exists before startup code runs, and callers may
#     add their own sinks at any time.
#   - The patcher mutates the record once; ``Handler.emit`` then detects that
#     the pre-colorized message no longer matches ``record["message"]`` and
#     re-derives output from the mutated record, so colorized sinks are safe.
#   - Patching happens before ``enqueue=True`` queueing, so queued/worker-side
#     writes see the redacted message too.


def _loguru_patcher(record: dict) -> None:
    """Loguru core patcher: redact ``record["message"]`` in place.

    Runs on every loguru record before handler emission.  Failures are
    swallowed: redaction must never break logging itself.
    """
    try:
        if not _patcher_enabled():
            return
        message = record.get("message")
        if isinstance(message, str) and message:
            # Use a module-level shared filter instance for its pattern list;
            # building one per record would be wasteful.
            global _SHARED_REDACTOR
            if _SHARED_REDACTOR is None:
                _SHARED_REDACTOR = RedactingFilter.from_env()
            record["message"] = _SHARED_REDACTOR.redact_message(message)
    except Exception:
        pass


# Marker so idempotency checks can recognise *our* patcher even if other
# code touches ``core.patcher`` too (loguru supports only one core patcher;
# installing ours necessarily replaces any previously-configured one).
_loguru_patcher._distllm_redaction = True  # type: ignore[attr-defined]

_SHARED_REDACTOR: "RedactingFilter | None" = None


def _patcher_enabled() -> bool:
    """Honour ``DISTLLM_LOG_REDACTION`` at emit time (cheap env read)."""
    enabled = os.environ.get(ENV_REDACTION_ENABLED, "on").strip().lower()
    return enabled not in ("off", "0", "false", "no", "")


def install_loguru_redaction(*, force: bool = False) -> bool:
    """Install the loguru core-level redaction patcher.

    Idempotent: a no-op when our patcher is already active on loguru's core
    (unless ``force=True``, which reinstalls unconditionally).  Returns
    ``True`` if the patcher is active after the call, ``False`` when
    redaction is disabled via ``DISTLLM_LOG_REDACTION`` (unless forced).
    """
    from loguru import logger as _loguru_logger

    if not _patcher_enabled() and not force:
        return False

    core = _loguru_logger._core  # noqa: SLF001 -- documented extension point
    if not force and getattr(core.patcher, "_distllm_redaction", False):
        return True

    _loguru_logger.configure(patcher=_loguru_patcher)
    return True


def uninstall_loguru_redaction() -> None:
    """Remove the loguru redaction patcher (restores raw output).

    Used by tests to verify uninstall restores pass-through behaviour.
    """
    from loguru import logger as _loguru_logger

    core = _loguru_logger._core  # noqa: SLF001 -- documented extension point
    if getattr(core.patcher, "_distllm_redaction", False):
        with core.lock:
            core.patcher = None


def install_global_redaction(
    logger: logging.Logger | None = None,
    *,
    force: bool = False,
) -> RedactingFilter:
    """Install redaction globally across BOTH logging frameworks.

    **loguru** (primary): installs a core-level patcher so that all ~794
    modules emitting through ``loguru.logger`` are redacted transparently.
    This happens only when called with no ``logger`` argument -- i.e. the
    process-wide activation form used by server/CLI startup.  Passing an
    explicit ``logger`` performs a *scoped* stdlib-only installation (used by
    tests), leaving the global loguru pipeline untouched.

    **stdlib** (secondary): attaches a :class:`RedactingFilter` to the given
    logger (default: the root logger) and each of its handlers, covering
    libraries that log via stdlib (including those bridged into it).

    Idempotent in both dimensions: an already-installed patcher or filter is
    left in place unless ``force=True``.  Honours ``DISTLLM_LOG_REDACTION``
    (set to off/0/false/no to skip installation entirely).

    Returns the stdlib filter instance (installed or pre-existing).
    """
    scoped = logger is not None
    if not scoped:
        # Process-wide activation: cover the loguru pipeline too.
        install_loguru_redaction(force=force)
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
