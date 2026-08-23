"""Content moderation pipeline — toxicity, PII, jailbreak, and topic filtering.

Provides five detector classes and a coordinating pipeline:

* ``ToxicityDetector`` — detect toxic language (transformer, ONNX, or keywords)
* ``PIIRedactor`` — detect and redact personally identifiable information
* ``JailbreakDetector`` — detect prompt injection and jailbreak attempts
* ``TopicFilter`` — filter content against configurable topic policies
* ``ContentModerationPipeline`` — orchestrate all detectors in a single pass

Each detector also exposes an ``async_check()`` / ``async_redact()`` variant
that runs the heavy work in a thread pool.
"""

from distllm.security.content_moderation.base import (
    _Missing,
    _MISSING,
    _lazy_import,
    ToxicResult,
    PIEEntity,
    PIIResult,
    JailbreakResult,
    TopicFilterResult,
    ModerationResult,
)
from distllm.security.content_moderation.toxicity import ToxicityDetector
from distllm.security.content_moderation.pii import PIIRedactor
from distllm.security.content_moderation.jailbreak import JailbreakDetector
from distllm.security.content_moderation.topics import TopicFilter
from distllm.security.content_moderation.pipeline import ContentModerationPipeline

__all__ = [
    "ToxicityDetector",
    "PIIRedactor",
    "JailbreakDetector",
    "TopicFilter",
    "ContentModerationPipeline",
    "ToxicResult",
    "PIEEntity",
    "PIIResult",
    "JailbreakResult",
    "TopicFilterResult",
    "ModerationResult",
]
