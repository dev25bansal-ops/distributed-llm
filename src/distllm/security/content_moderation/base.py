"""Base types and shared utilities for the content moderation subsystem."""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Lazy optional dependency checks
# ---------------------------------------------------------------------------


class _Missing:
    """Sentinel used when an optional dependency is not installed."""

    pass


_MISSING = _Missing()


def _lazy_import(module_name: str, attr: str | None = None) -> tuple[bool, Any]:
    """Return ``(imported_ok, module_or_default)``, importing lazily."""
    try:
        __import__(module_name)
        mod = __import__(module_name)
        return True, getattr(mod, attr) if attr else mod
    except ImportError:
        return False, _MISSING


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_CACHE_DIR = os.environ.get(
    "CONTENT_MODEL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "distllm", "content_moderation"),
)
_DEFAULT_TOXICITY_THRESHOLD = float(
    os.environ.get("CONTENT_MODERATION_THRESHOLD", "0.7")
)
_DEFAULT_JAILBREAK_THRESHOLD = float(
    os.environ.get("CONTENT_JAILBREAK_THRESHOLD", "0.5")
)

_DEFAULT_TOXICITY_MODEL = "unitary/toxic-bert"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToxicResult:
    """Result from a toxicity check.

    Attributes:
        toxic: Whether the text exceeds the configured toxicity threshold.
        categories: Per-category toxicity scores (category -> score 0.0--1.0).
        score: Aggregate toxicity score (0.0--1.0).
    """

    toxic: bool
    categories: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass(frozen=True)
class PIEEntity:
    """A detected PII entity.

    Attributes:
        type: Entity type (``"email"``, ``"phone"``, ``"ssn"``,
            ``"credit_card"``, ``"ip_address"``, or a custom label).
        value: The original text of the detected entity.
        start: Character offset where the entity starts in the original text.
        end: Character offset where the entity ends in the original text.
        redacted: The replacement text (e.g. ``"[EMAIL]"``).
    """

    type: str
    value: str
    start: int
    end: int
    redacted: str


@dataclass(frozen=True)
class PIIResult:
    """Result from PII redaction.

    Attributes:
        redacted_text: Input text with PII replaced by placeholder tokens.
        entities_found: List of detected PII entities.
    """

    redacted_text: str
    entities_found: list[PIEEntity] = field(default_factory=list)


@dataclass(frozen=True)
class JailbreakResult:
    """Result from a jailbreak attempt check.

    Attributes:
        jailbreak_attempt: Whether a jailbreak or prompt injection was detected.
        confidence: Confidence score (0.0--1.0).
        matched_patterns: The specific patterns that triggered detection.
        explanation: Human-readable explanation of the detection.
    """

    jailbreak_attempt: bool
    confidence: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass(frozen=True)
class TopicFilterResult:
    """Result from topic-based content filtering.

    Attributes:
        allowed: Whether the content is permitted under the active policies.
        violated_policies: List of policy names that were violated (empty if
            ``allowed`` is ``True``).
        matched_terms: Specific terms from the policy that matched.
    """

    allowed: bool
    violated_policies: list[str] = field(default_factory=list)
    matched_terms: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModerationResult:
    """Aggregated result from the full content moderation pipeline.

    Attributes:
        passed: ``True`` when all enabled checks pass (no toxicity, no
            jailbreak, topics allowed, ...).
        toxicity: Toxicity check result, or ``None`` if disabled.
        pii: PII redaction result, or ``None`` if disabled.
        jailbreak: Jailbreak detection result, or ``None`` if disabled.
        topic_filter: Topic filter result, or ``None`` if disabled.
        redacted_text: The input text after PII redaction (same as input
            when PII redaction is disabled).
    """

    passed: bool
    toxicity: ToxicResult | None = None
    pii: PIIResult | None = None
    jailbreak: JailbreakResult | None = None
    topic_filter: TopicFilterResult | None = None
    redacted_text: str = ""


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class _TextClassifierBackend(abc.ABC):
    """Abstract base for text classification backends."""

    @abc.abstractmethod
    def predict(self, text: str, categories: list[str] | None = None) -> dict[str, float]:
        """Run classification and return per-category scores (0.0--1.0)."""

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this backend is ready for inference."""


class _TransformersBackend(_TextClassifierBackend):
    """Classifier backend backed by HuggingFace ``transformers``.

    The model pipeline is loaded lazily on first call to ``predict()``
    so construction is fast even when the model needs to be downloaded.
    """

    def __init__(self, model_name: str = _DEFAULT_TOXICITY_MODEL) -> None:
        self._model_name = model_name
        self._pipeline: Any = None
        self._load_attempted: bool = False

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if self._load_attempted:
            return
        self._load_attempted = True
        has_tf, transformers_mod = _lazy_import("transformers")
        if not has_tf:
            import logging
            logging.getLogger(__name__).warning(
                "transformers not installed; %s backend unavailable.", self._model_name
            )
            return
        try:
            pipeline_fn = transformers_mod.pipeline
            has_torch, torch_mod = _lazy_import("torch")
            device = (
                0
                if has_torch and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available()
                else -1
            )

            self._pipeline = pipeline_fn(
                "text-classification",
                model=self._model_name,
                top_k=None,
                device=device,
            )
            import logging
            logging.getLogger(__name__).info(
                "Loaded toxicity model: %s (transformers backend)", self._model_name
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "Failed to load transformers model %s: %s", self._model_name, exc
            )
            self._pipeline = None

    def available(self) -> bool:
        """Return whether the pipeline is already loaded.

        Does *not* trigger a download or import -- this is a lightweight
        check used during backend selection in ``ToxicityDetector.__init__``.
        """
        return self._pipeline is not None

    def predict(self, text: str, categories: list[str] | None = None) -> dict[str, float]:
        self._ensure_loaded()
        if not self._pipeline:
            raise RuntimeError("Transformers backend is not available.")
        result = self._pipeline(text)  # type: ignore[misc]
        if isinstance(result, list) and isinstance(result[0], dict):
            result = result[0]
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]
        scores: dict[str, float] = {}
        for entry in result:
            label = entry.get("label", "").lower()
            score = float(entry.get("score", 0.0))
            if categories and label not in categories:
                continue
            scores[label] = score
        return scores


class _ONNXBackend(_TextClassifierBackend):
    """Classifier backend backed by ONNX Runtime.

    Expects the model directory to contain ``model.onnx`` and
    ``tokenizer.json`` (or a ``tokenizer`` subdirectory).
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._session: Any = None
        self._tokenizer: Any = None
        self._labels: list[str] = []
        self._init()

    def _init(self) -> None:
        has_onnx, ort_mod = _lazy_import("onnxruntime")
        if not has_onnx:
            import logging
            logging.getLogger(__name__).warning("onnxruntime not installed; ONNX backend unavailable.")
            return
        onnx_file = os.path.join(self._model_path, "model.onnx")
        if not os.path.isfile(onnx_file):
            import logging
            logging.getLogger(__name__).warning("ONNX model not found at %s", onnx_file)
            return
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort_mod.InferenceSession(onnx_file, providers=providers)

            # Attempt to load a tokenizer from the same directory.
            has_tf, transformers_mod = _lazy_import("transformers")
            if has_tf:
                try:
                    tok_path = os.path.join(self._model_path, "tokenizer.json")
                    if os.path.isfile(tok_path):
                        self._tokenizer = transformers_mod.PreTrainedTokenizerFast(
                            tokenizer_file=tok_path
                        )
                except Exception:
                    self._tokenizer = None

            # Read labels from model metadata, if present.
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            raw_labels = meta.get("labels", "")
            self._labels = raw_labels.split(",") if raw_labels else []

            import logging
            logging.getLogger(__name__).info(
                "Loaded ONNX toxicity model from %s (labels=%s)",
                self._model_path,
                self._labels,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to load ONNX model: %s", exc)
            self._session = None

    def available(self) -> bool:
        return self._session is not None

    def predict(self, text: str, categories: list[str] | None = None) -> dict[str, float]:
        if not self.available():
            raise RuntimeError("ONNX backend is not available.")
        if self._tokenizer is not None:
            inputs = self._tokenizer(
                text, return_tensors="np", truncation=True, max_length=128
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        else:
            has_np, np = _lazy_import("numpy")  # noqa: A001
            if not has_np:
                raise RuntimeError("numpy is required for ONNX backend without tokenizer.")
            ort_inputs = {
                "input_ids": np.ones((1, 128), dtype=np.int64),
                "attention_mask": np.ones((1, 128), dtype=np.int64),
            }
        raw = self._session.run(None, ort_inputs)
        logits = raw[0][0]
        has_np, np = _lazy_import("numpy")  # noqa: A001
        if not has_np:
            raise RuntimeError("numpy is required for ONNX backend prediction.")
        scores: dict[str, float] = {}
        labels = self._labels if self._labels else [f"class_{i}" for i in range(len(logits))]
        for i, label in enumerate(labels):
            prob = float(1.0 / (1.0 + np.exp(-logits[i])))
            if categories and label not in categories:
                continue
            scores[label] = prob
        return scores


class _KeywordBackend(_TextClassifierBackend):
    """Fallback classifier backend based on keyword matching.

    The keyword sets here are deliberately conservative to minimise
    false positives.  Neutral descriptors (``"race"``, ``"gender"``)
    are NOT included -- they are not inherently toxic.
    """

    _TOXIC_KEYWORDS: dict[str, list[str]] = {
        "toxicity": [
            "hate", "kill", "die",
            "loser", "pathetic", "useless", "worthless",
        ],
        "insult": [
            "idiot", "moron", "fool", "imbecile",
            "cretin", "ignorant", "buffoon",
        ],
        "profanity": [
            "fuck", "shit", "damn", "bitch", "bastard",
            "crap", "dick",
        ],
        "threat": [
            "i will kill", "i will hurt", "i will destroy",
            "you will die", "you're dead", "going to kill",
            "end you", "hurt you",
        ],
        "identity_attack": [
            "racist", "sexist", "bigot", "nazi",
        ],
        "severe_toxicity": [
            "exterminate", "eradicate", "wipe out", "annihilate",
            "massacre", "genocide",
        ],
    }

    def __init__(self, custom_keywords: dict[str, list[str]] | None = None) -> None:
        self._keywords = dict(self._TOXIC_KEYWORDS)
        if custom_keywords:
            for category, terms in custom_keywords.items():
                self._keywords.setdefault(category, []).extend(terms)

    def available(self) -> bool:
        return True

    def predict(self, text: str, categories: list[str] | None = None) -> dict[str, float]:
        text_lower = text.lower()
        scores: dict[str, float] = {}
        for category, terms in self._keywords.items():
            if categories and category not in categories:
                continue
            matches = sum(1 for term in terms if term in text_lower)
            if matches >= 2:
                scores[category] = 1.0
            elif matches == 1:
                scores[category] = 0.8
        return scores
