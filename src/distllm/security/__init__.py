"""Security utilities and E2E encryption for tensor transport.

Provides:
  - ``safe_urlopen``, ``validate_http_url``, ``hf_revision`` — SSRF/DNS
    rebinding protection and model revision pinning
  - ``E2EEncryption`` — NaCl/libsodium-based end-to-end encryption for
    tensor bytes with per-session X25519 key exchange
  - ``SessionManager`` — manages per-node-pair encryption sessions
  - Content moderation — ``ToxicityDetector``, ``PIIRedactor``,
    ``JailbreakDetector``, ``TopicFilter``, ``ContentModerationPipeline``
"""

from .content_moderation import (
    ContentModerationPipeline,
    JailbreakDetector,
    JailbreakResult,
    ModerationResult,
    PIEEntity,
    PIIRedactor,
    PIIResult,
    TopicFilter,
    TopicFilterResult,
    ToxicityDetector,
    ToxicResult,
)
from .e2e import (
    HAS_NACL,
    E2EEncryption,
    SessionKeys,
    decrypt_tensor_payload,
    encrypt_tensor_payload,
)
from .utils import hf_revision, safe_urlopen, validate_http_url
from .watermark import (
    ModelWatermark,
    WatermarkConfig,
    run_cli,
)

__all__ = [
    "hf_revision",
    "safe_urlopen",
    "validate_http_url",
    "E2EEncryption",
    "SessionKeys",
    "encrypt_tensor_payload",
    "decrypt_tensor_payload",
    "HAS_NACL",
    "ModelWatermark",
    "WatermarkConfig",
    "run_cli",
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
