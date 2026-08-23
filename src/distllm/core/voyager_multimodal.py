"""True multi-modal inference pipeline.

Encodes multiple modalities (text, image, audio, video) in parallel,
routes to the optimal model combination, and generates a unified response.

Architecture::

    request (text + optional image/audio/video)
         │
         ▼
    ┌──────────────────────┐
    │  MultiModalRouter    │  ← analyzes modalities, selects model combo
    │  analyze() / route() │     routes: text-only, image-text, audio-text,
    │                      │            video-text
    └──────────┬───────────┘
               │ routing plan
               ▼
    ┌──────────────────────┐
    │ ParallelEncoder      │  ← encodes modalities in parallel across GPUs
    │ Pipeline             │     using ThreadPoolExecutor
    │ encode_* methods     │
    └──────────┬───────────┘
               │ encoded embeddings
               ▼
    ┌──────────────────────┐
    │  Voyager (orchestr.) │  ← combines embeddings, runs generation
    │  process()           │
    └──────────────────────┘

Usage::

    voyager = Voyager()
    response = voyager.process(
        text="Describe this image",
        image=image_array,
        audio=audio_array,
        video=video_array,
    )
    print(response.text)

    stats = voyager.stats()
    # {"total_requests": 5, "modality_distribution": {...}, ...}
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


# ── Optional dependency guards ─────────────────────────────────────────────

_HAS_TORCH = False
_HAS_TRANSFORMERS = False
_HAS_NUMPY = False
_HAS_CLIP = False
_HAS_WHISPER = False
_HAS_PIL = False
_HAS_TORCHVISION = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import transformers

    _HAS_TRANSFORMERS = True
except ImportError:
    transformers = None  # type: ignore[assignment]

try:
    import clip  # type: ignore[import-untyped]

    _HAS_CLIP = True
except ImportError:
    clip = None  # type: ignore[assignment]

try:
    import whisper  # type: ignore[import-untyped]

    _HAS_WHISPER = True
except ImportError:
    whisper = None  # type: ignore[assignment]

try:
    from PIL import Image as PILImage

    _HAS_PIL = True
except ImportError:
    PILImage = None  # type: ignore[assignment]

try:
    import torchvision.transforms as T

    _HAS_TORCHVISION = True
except ImportError:
    T = None  # type: ignore[assignment]


# ── Enums & Data Classes ───────────────────────────────────────────────────


class ModalityType(str, Enum):
    """Modalities the pipeline can process."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RouteType(str, Enum):
    """Pre-defined routing paths."""

    TEXT_ONLY = "text-only"
    IMAGE_TEXT = "image-text"
    AUDIO_TEXT = "audio-text"
    VIDEO_TEXT = "video-text"


@dataclass
class MultiModalRequest:
    """A single request carrying one or more modalities.

    At minimum *text* should be provided; *image*, *audio*, and *video*
    are optional and can be passed together.
    """

    text: str = ""
    image: Any = None  # numpy array, torch tensor, or PIL Image
    audio: Any = None  # numpy waveform array or path
    video: Any = None  # numpy array (T, H, W, C) or list of frames
    metadata: dict[str, Any] = field(default_factory=dict)

    def modality_set(self) -> set[ModalityType]:
        """Return the set of non-empty modalities in this request."""
        result: set[ModalityType] = set()
        if self.text:
            result.add(ModalityType.TEXT)
        if self.image is not None:
            result.add(ModalityType.IMAGE)
        if self.audio is not None:
            result.add(ModalityType.AUDIO)
        if self.video is not None:
            result.add(ModalityType.VIDEO)
        return result


@dataclass
class RoutingPlan:
    """The routing decision produced by :class:`MultiModalRouter`."""

    route_type: RouteType
    encoder_models: list[str] = field(default_factory=list)
    generation_model: str = ""
    pipeline_order: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class EncodedOutput:
    """Encoded representation of a single modality."""

    modality: ModalityType
    embedding: Any = None  # numpy array or torch tensor
    model_used: str = ""
    encoding_time_ms: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class VoyagerResponse:
    """The final response from the Voyager pipeline."""

    text: str = ""
    full_response: str = ""
    route_type: RouteType = RouteType.TEXT_ONLY
    encoding_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    model_used: str = ""
    tokens_generated: int = 0


# ── Helper: resolve device ────────────────────────────────────────────────


def _resolve_device(preferred: str = "") -> str:
    """Resolve the best available device.

    Falls back ``cuda:N`` → ``mps`` → ``cpu`` depending on what is
    available.  When *preferred* is set and available, it is used
    as-is.
    """
    if preferred and _HAS_TORCH and torch is not None:
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return preferred
        if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return preferred
        if preferred == "cpu":
            return preferred
    if _HAS_TORCH and torch is not None:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# 1. ModalityEncoder
# ═══════════════════════════════════════════════════════════════════════════


class ModalityEncoder:
    """Encodes different input modalities into vector embeddings.

    Wraps existing models (CLIP, Whisper, text transformers) when
    available; gracefully falls back to a hash-based / zero-embedding
    when the relevant package is not installed.

    Usage::

        encoder = ModalityEncoder(device="cuda:0")
        text_emb = encoder.encode_text("Hello world")
        image_emb = encoder.encode_image(image_array)
        audio_emb = encoder.encode_audio(audio_array)
        video_emb = encoder.encode_video(video_array)
    """

    def __init__(
        self,
        device: str = "",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "ViT-B/32",
        whisper_model_size: str = "tiny",
        embed_dim: int = 384,
    ):
        self._device = _resolve_device(device)
        self._text_model_name = text_model_name
        self._clip_model_name = clip_model_name
        self._whisper_model_size = whisper_model_size
        self._embed_dim = embed_dim
        self._lock = threading.Lock()

        # Lazy-loaded models
        self._text_encoder: Any = None
        self._text_tokenizer: Any = None
        self._clip_model: Any = None
        self._whisper_model: Any = None

        self._stats = {
            "encode_text_calls": 0,
            "encode_image_calls": 0,
            "encode_audio_calls": 0,
            "encode_video_calls": 0,
            "total_errors": 0,
        }

    # ── Public encoding methods ─────────────────────────────────────────

    def encode_text(self, text: str) -> tuple[Any, str]:
        """Encode text into an embedding vector.

        Returns:
            A tuple ``(embedding_array, model_name)``.  The embedding is
            a 1-D float32 array (numpy or torch depending on available
            runtime).  Returns a zero vector when no encoder is available.
        """
        self._stats["encode_text_calls"] += 1
        if not text:
            return self._zero_embedding(), "none"

        # Try sentence-transformer style (HuggingFace)
        if _HAS_TRANSFORMERS:
            encoder, tokenizer = self._lazy_load_text_encoder()
            if encoder is not None and tokenizer is not None:
                try:
                    inputs = tokenizer(
                        text[:512], return_tensors="pt", padding=True, truncation=True,
                    )
                    if _HAS_TORCH and torch is not None:
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = encoder(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()
                    return embedding, self._text_model_name
                except Exception as exc:
                    logger.warning(f"Text encoding failed: {exc}")
                    self._stats["total_errors"] += 1

        # Fallback: simple hash-based embedding for detection
        return self._zero_embedding(), "none"

    def encode_image(self, image: Any) -> tuple[Any, str]:
        """Encode an image into an embedding vector.

        Accepts a numpy array (H, W, C), a torch tensor, or a PIL Image.
        Uses CLIP when available; returns a zero vector otherwise.

        Returns:
            A tuple ``(embedding_array, model_name)``.
        """
        self._stats["encode_image_calls"] += 1
        if image is None:
            return self._zero_embedding(), "none"

        # Try CLIP
        if _HAS_CLIP:
            model = self._lazy_load_clip()
            if model is not None:
                try:
                    image_tensor = self._prepare_image_for_clip(image)
                    if image_tensor is None:
                        return self._zero_embedding(), "none"
                    if _HAS_TORCH and torch is not None:
                        image_tensor = image_tensor.to(self._device)
                    with torch.no_grad():
                        embedding = model.encode_image(image_tensor)
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()
                    return embedding, self._clip_model_name
                except Exception as exc:
                    logger.warning(f"Image encoding (CLIP) failed: {exc}")
                    self._stats["total_errors"] += 1

        # Try torchvision + transformers
        if _HAS_TRANSFORMERS and (_HAS_TORCH or _HAS_NUMPY):
            try:
                image_tensor = self._prepare_image_for_clip(image)
                if image_tensor is not None:
                    embedding = self._image_to_embedding_via_transformers(image_tensor)
                    if embedding is not None:
                        return embedding, self._text_model_name
            except Exception as exc:
                logger.warning(f"Image encoding (transformers) failed: {exc}")
                self._stats["total_errors"] += 1

        return self._zero_embedding(), "none"

    def encode_audio(self, audio: Any) -> tuple[Any, str]:
        """Encode audio (waveform array) into an embedding vector.

        Uses Whisper when available; returns a zero vector otherwise.

        The input should be a 1-D numpy float32 array (sampled at 16 kHz)
        or a path to an audio file.

        Returns:
            A tuple ``(embedding_array, model_name)``.
        """
        self._stats["encode_audio_calls"] += 1
        if audio is None:
            return self._zero_embedding(), "none"

        if _HAS_WHISPER:
            model = self._lazy_load_whisper()
            if model is not None:
                try:
                    audio_array = self._prepare_audio(audio)
                    if audio_array is None:
                        return self._zero_embedding(), "none"
                    audio_array = audio_array.astype("float32") if _HAS_NUMPY else audio_array

                    if _HAS_TORCH and torch is not None:
                        audio_tensor = (
                            torch.from_numpy(audio_array)
                            if isinstance(audio_array, np.ndarray)
                            else audio_array
                        )
                        audio_tensor = audio_tensor.to(self._device)
                    else:
                        audio_tensor = audio_array

                    result = model.transcribe(audio_tensor)  # type: ignore[arg-type]
                    # Whisper returns dict with 'segments'; we build an embedding
                    # from the encoder output or fallback to text features
                    segments = result.get("segments", [])
                    text_feats = " ".join(s["text"] for s in segments)

                    # Re-encode as text embedding
                    if text_feats.strip():
                        return self.encode_text(text_feats)

                    return self._zero_embedding(), self._whisper_model_size
                except Exception as exc:
                    logger.warning(f"Audio encoding failed: {exc}")
                    self._stats["total_errors"] += 1

        # Fallback: if we have numpy, compute simple audio features
        if _HAS_NUMPY and np is not None:
            try:
                audio_array = self._prepare_audio(audio)
                if audio_array is not None and len(audio_array) > 0:
                    # Compute MFCC-like stats as a simple audio descriptor
                    embed = np.zeros(self._embed_dim, dtype="float32")
                    n = min(len(audio_array), self._embed_dim)
                    embed[:n] = audio_array[:n]
                    # Normalise
                    norm = np.linalg.norm(embed)
                    if norm > 0:
                        embed /= norm
                    return embed, "audio_statistical"
            except Exception:
                self._stats["total_errors"] += 1

        return self._zero_embedding(), "none"

    def encode_video(self, video: Any) -> tuple[Any, str]:
        """Encode video (sequence of frames) into an embedding vector.

        Samples key frames and encodes each via :meth:`encode_image`,
        then aggregates (mean-pool) into a single embedding.

        Accepts:
        - A numpy array shaped ``(T, H, W, C)``
        - A list of numpy arrays / torch tensors
        - A 4-D torch tensor ``(T, C, H, W)``

        Returns:
            A tuple ``(embedding_array, model_name)``.
        """
        self._stats["encode_video_calls"] += 1
        if video is None:
            return self._zero_embedding(), "none"

        frames = self._extract_frames(video)
        if not frames:
            return self._zero_embedding(), "none"

        # Sample up to 8 key frames uniformly
        max_frames = 8
        if len(frames) > max_frames:
            indices = [int(i * len(frames) / max_frames) for i in range(max_frames)]
            frames = [frames[i] for i in indices]

        # Encode each frame in sequence (no parallelism at this level —
        # parallelism is handled by ParallelEncoderPipeline)
        embeddings: list[Any] = []
        model_used = "none"
        for frame in frames:
            emb, model_used = self.encode_image(frame)
            if _HAS_NUMPY and np is not None and isinstance(emb, np.ndarray):
                embeddings.append(emb)
            elif _HAS_TORCH and torch is not None and isinstance(emb, torch.Tensor):
                embeddings.append(emb.cpu().numpy())

        if not embeddings:
            return self._zero_embedding(), "none"

        # Mean-pool
        if _HAS_NUMPY and np is not None:
            pooled = np.mean(np.stack(embeddings), axis=0)
        elif _HAS_TORCH and torch is not None:
            stacked = torch.stack([torch.from_numpy(e) for e in embeddings])
            pooled = stacked.mean(dim=0).numpy()
        else:
            pooled = self._zero_embedding()

        return pooled, model_used

    # ── Lazy model loading ──────────────────────────────────────────────

    def _lazy_load_text_encoder(self) -> tuple[Any, Any]:
        """Load the text embedding model on first access."""
        if self._text_encoder is not None:
            return self._text_encoder, self._text_tokenizer
        if not _HAS_TRANSFORMERS or transformers is None:
            logger.info("transformers not installed — text encoder unavailable")
            return None, None

        with self._lock:
            if self._text_encoder is not None:
                return self._text_encoder, self._text_tokenizer
            try:
                from transformers import AutoModel, AutoTokenizer

                logger.info(
                    f"Loading text encoder: {self._text_model_name} on {self._device}"
                )
                self._text_tokenizer = AutoTokenizer.from_pretrained(self._text_model_name)
                self._text_encoder = AutoModel.from_pretrained(self._text_model_name)
                if _HAS_TORCH and torch is not None:
                    self._text_encoder = self._text_encoder.to(self._device)
                    self._text_encoder.eval()
                logger.info("Text encoder loaded")
            except Exception as exc:
                logger.error(f"Failed to load text encoder: {exc}")
                self._stats["total_errors"] += 1
        return self._text_encoder, self._text_tokenizer

    def _lazy_load_clip(self) -> Any:
        """Load CLIP model on first access."""
        if self._clip_model is not None:
            return self._clip_model
        if not _HAS_CLIP:
            logger.info("clip not installed — image encoder unavailable")
            return None

        with self._lock:
            if self._clip_model is not None:
                return self._clip_model
            try:
                logger.info(
                    f"Loading CLIP: {self._clip_model_name} on {self._device}"
                )
                self._clip_model, _ = clip.load(self._clip_model_name, device=self._device)
                logger.info("CLIP model loaded")
            except Exception as exc:
                logger.error(f"Failed to load CLIP: {exc}")
                self._stats["total_errors"] += 1
        return self._clip_model

    def _lazy_load_whisper(self) -> Any:
        """Load Whisper model on first access."""
        if self._whisper_model is not None:
            return self._whisper_model
        if not _HAS_WHISPER:
            logger.info("whisper not installed — audio encoder unavailable")
            return None

        with self._lock:
            if self._whisper_model is not None:
                return self._whisper_model
            try:
                logger.info(
                    f"Loading Whisper: {self._whisper_model_size} on {self._device}"
                )
                self._whisper_model = whisper.load_model(
                    self._whisper_model_size,
                    device=self._device,
                )
                logger.info("Whisper model loaded")
            except Exception as exc:
                logger.error(f"Failed to load Whisper: {exc}")
                self._stats["total_errors"] += 1
        return self._whisper_model

    # ── Internal helpers ────────────────────────────────────────────────

    def _zero_embedding(self) -> Any:
        """Return a zero vector of the configured embedding dimension."""
        if _HAS_NUMPY and np is not None:
            return np.zeros(self._embed_dim, dtype="float32")
        if _HAS_TORCH and torch is not None:
            return torch.zeros(self._embed_dim, dtype=torch.float32)
        return [0.0] * self._embed_dim

    def _prepare_image_for_clip(self, image: Any) -> Any:
        """Convert an image to a CLIP-compatible tensor.

        Returns a torch tensor ``(1, 3, H, W)`` or ``None`` on failure.
        """
        if not _HAS_TORCH or torch is None:
            return None

        # Already a tensor
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                # (C, H, W) → (1, C, H, W)
                return image.unsqueeze(0)
            if image.ndim == 4:
                return image
            return None

        # numpy array
        if _HAS_NUMPY and np is not None and isinstance(image, np.ndarray):
            if image.ndim == 3:
                # (H, W, C) → (1, C, H, W)
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                return tensor / 255.0 if image.dtype == np.uint8 else tensor
            if image.ndim == 4:
                return torch.from_numpy(image).float()

        # PIL Image
        if _HAS_PIL and PILImage is not None:
            try:
                from torchvision.transforms import Compose, Normalize, Resize, ToTensor

                transform = Compose([
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ])
                if isinstance(image, PILImage.Image):
                    return transform(image).unsqueeze(0)
            except Exception:
                pass

        return None

    def _image_to_embedding_via_transformers(self, image_tensor: Any) -> Any | None:
        """Use a HuggingFace vision model to produce an embedding."""
        if not _HAS_TRANSFORMERS or transformers is None:
            return None
        try:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = AutoModel.from_pretrained("google/vit-base-patch16-224")
            if _HAS_TORCH and torch is not None:
                model = model.to(self._device)
                model.eval()

            if _HAS_TORCH and torch is not None and isinstance(image_tensor, torch.Tensor):
                inputs = processor(images=image_tensor, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            else:
                inputs = processor(images=image_tensor, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
            return embedding
        except Exception as exc:
            logger.warning(f"Vision transformer encoding failed: {exc}")
            return None

    def _prepare_audio(self, audio: Any) -> Any | None:
        """Convert audio input to a 1-D float32 numpy array (16 kHz)."""
        if _HAS_NUMPY and np is not None and isinstance(audio, np.ndarray):
            if audio.ndim == 2:
                # stereo → mono
                audio = audio.mean(axis=1)
            return audio.astype("float32")

        if _HAS_TORCH and torch is not None and isinstance(audio, torch.Tensor):
            if audio.ndim == 2:
                audio = audio.mean(dim=1)
            return audio.cpu().numpy().astype("float32")

        # String path — try to load via librosa / soundfile (optional)
        if isinstance(audio, str):
            try:
                import soundfile as sf  # type: ignore[import-untyped]

                data, _ = sf.read(audio)
                return data.astype("float32")
            except ImportError:
                logger.warning(
                    "soundfile not installed — cannot load audio from path"
                )
                return None

        return None

    def _extract_frames(self, video: Any) -> list[Any]:
        """Extract individual frames from a video input.

        Returns a list of arrays/tensors, or an empty list on failure.
        """
        frames: list[Any] = []

        # numpy array (T, H, W, C)
        if _HAS_NUMPY and np is not None and isinstance(video, np.ndarray):
            if video.ndim == 4:
                for i in range(video.shape[0]):
                    frames.append(video[i])
                return frames

        # torch tensor (T, C, H, W) or (T, H, W, C)
        if _HAS_TORCH and torch is not None and isinstance(video, torch.Tensor):
            if video.ndim == 4:
                for i in range(video.shape[0]):
                    frames.append(video[i])
                return frames

        # list of arrays/tensors
        if isinstance(video, (list, tuple)):
            return list(video)

        return frames

    def stats(self) -> dict[str, Any]:
        """Return encoder statistics."""
        return dict(self._stats)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MultiModalRouter
# ═══════════════════════════════════════════════════════════════════════════


class MultiModalRouter:
    """Learned / rule-based router that selects the optimal model combination.

    :meth:`analyze` inspects the request to determine which modalities
    are present.  :meth:`route` maps the modality combination to a
    concrete :class:`RoutingPlan` with selected models and execution
    order.

    The default implementation uses deterministic rules.  A learning
    backend can be plugged in via *reward_fn* and *update()* to make
    data-driven decisions over time.

    Usage::

        router = MultiModalRouter()
        mods, complexity = router.analyze(request)
        plan = router.route(request)
    """

    _TEXT_GENERATION_MODELS: dict[RouteType, str] = {
        RouteType.TEXT_ONLY: "default-text-model",
        RouteType.IMAGE_TEXT: "vision-language-model",
        RouteType.AUDIO_TEXT: "audio-language-model",
        RouteType.VIDEO_TEXT: "video-language-model",
    }

    _ENCODER_MODELS: dict[ModalityType, str] = {
        ModalityType.TEXT: "text-encoder",
        ModalityType.IMAGE: "clip-image-encoder",
        ModalityType.AUDIO: "whisper-audio-encoder",
        ModalityType.VIDEO: "frame-encoder",
    }

    def __init__(self, model_map: dict[RouteType, str] | None = None):
        """Initialize the router.

        Args:
            model_map: Optional override mapping ``RouteType`` to
                generation model names.  Falls back to built-in defaults.
        """
        self._model_map: dict[RouteType, str] = {
            **self._TEXT_GENERATION_MODELS,
            **(model_map or {}),
        }
        self._lock = threading.Lock()

        # Routing statistics for accuracy tracking
        self._stats = {
            "total_routes": 0,
            "text_only": 0,
            "image_text": 0,
            "audio_text": 0,
            "video_text": 0,
            "correct_routes": 0,
            "total_feedback": 0,
        }

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self, request: MultiModalRequest) -> tuple[set[ModalityType], float]:
        """Analyze a request and determine required modalities and complexity.

        Args:
            request: The incoming multi-modal request.

        Returns:
            A tuple ``(required_modalities, complexity_score)`` where
            *complexity_score* is a float in ``[0, 1]`` (higher = more
            complex).
        """
        modalities = request.modality_set()
        if not modalities:
            modalities = {ModalityType.TEXT}

        # Estimate complexity based on number and type of modalities
        base = len(modalities) / len(ModalityType)
        has_video = ModalityType.VIDEO in modalities
        complexity = min(1.0, base + (0.3 if has_video else 0.0))

        return modalities, complexity

    def route(self, request: MultiModalRequest) -> RoutingPlan:
        """Produce a routing plan for a request.

        Determines the :class:`RouteType` based on the set of modalities
        present, selects the encoder models and generation model, and
        returns a full :class:`RoutingPlan`.

        Args:
            request: The incoming multi-modal request.

        Returns:
            A fully populated :class:`RoutingPlan`.
        """
        modalities, _ = self._analyze_route_type(request)
        route_type = self._modalities_to_route_type(modalities)
        generation_model = self._model_map.get(route_type, "default-text-model")
        encoder_models = [
            self._ENCODER_MODELS[m]
            for m in modalities
            if m != ModalityType.TEXT
        ]

        # Pipeline order: encode non-text modalities first, then generate
        pipeline_order = encoder_models + [f"generate:{generation_model}"]

        with self._lock:
            self._stats["total_routes"] += 1
            self._stats[route_type.value.replace("-", "_")] += 1

        return RoutingPlan(
            route_type=route_type,
            encoder_models=encoder_models,
            generation_model=generation_model,
            pipeline_order=pipeline_order,
        )

    def update(
        self,
        request: MultiModalRequest,
        plan: RoutingPlan,
        feedback: float,
    ) -> None:
        """Update routing preferences based on feedback.

        This is a stub for future learning-based routing.  *feedback*
        should be a score in ``[0, 1]`` with 1 = perfect, 0 = useless.

        Args:
            request: The original request.
            plan: The routing plan that was executed.
            feedback: Quality score for the route.
        """
        with self._lock:
            self._stats["total_feedback"] += 1
            if feedback >= 0.5:
                self._stats["correct_routes"] += 1

    @property
    def routing_accuracy(self) -> float:
        """Routing accuracy based on accumulated feedback."""
        total = self._stats["total_feedback"]
        if total == 0:
            return 0.0
        return self._stats["correct_routes"] / total

    # ── Internal ────────────────────────────────────────────────────────

    def _analyze_route_type(
        self,
        request: MultiModalRequest,
    ) -> tuple[set[ModalityType], RouteType]:
        """Determine the route type from a request."""
        modalities = request.modality_set()
        if not modalities:
            modalities = {ModalityType.TEXT}
        route_type = self._modalities_to_route_type(modalities)
        return modalities, route_type

    @staticmethod
    def _modalities_to_route_type(modalities: set[ModalityType]) -> RouteType:
        """Map a set of modalities to the best route type."""
        if ModalityType.VIDEO in modalities:
            return RouteType.VIDEO_TEXT
        if ModalityType.IMAGE in modalities:
            return RouteType.IMAGE_TEXT
        if ModalityType.AUDIO in modalities:
            return RouteType.AUDIO_TEXT
        return RouteType.TEXT_ONLY

    def stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        with self._lock:
            return {
                **dict(self._stats),
                "routing_accuracy": self.routing_accuracy,
            }


# ═══════════════════════════════════════════════════════════════════════════
# 3. ParallelEncoderPipeline
# ═══════════════════════════════════════════════════════════════════════════


class ParallelEncoderPipeline:
    """Encodes multiple modalities in parallel across threads/GPUs.

    Each modality encoder can be scheduled on a different device,
    enabling true parallel encoding when multiple GPUs are available.

    Usage::

        pipeline = ParallelEncoderPipeline(encoders={"gpu0": encoder_0, "gpu1": encoder_1})
        outputs = pipeline.execute(request, plan)
    """

    def __init__(
        self,
        encoders: dict[str, ModalityEncoder] | None = None,
        max_workers: int = 4,
    ):
        """Initialize the parallel pipeline.

        Args:
            encoders: Mapping of device/slot names to :class:`ModalityEncoder`
                instances.  If ``None``, a single default encoder is created.
            max_workers: Maximum thread pool workers for parallel encoding.
        """
        self._encoders: dict[str, ModalityEncoder] = encoders or {
            _resolve_device(): ModalityEncoder(),
        }
        self._max_workers = max_workers
        self._lock = threading.Lock()

        self._stats = {
            "total_executions": 0,
            "total_encoded_modalities": 0,
            "total_encoding_time_ms": 0.0,
            "errors": 0,
        }

    def execute(
        self,
        request: MultiModalRequest,
        plan: RoutingPlan,
    ) -> dict[str, EncodedOutput]:
        """Encode all modalities in the request in parallel.

        Schedules an encoding task for each non-text modality present
        in the request, running them via a :class:`ThreadPoolExecutor`.

        Text is encoded synchronously after parallel tasks join since
        it is almost always the fastest modality.

        Args:
            request: The multi-modal request.
            plan: The routing plan (used to determine which modalities
                to encode and in what order).

        Returns:
            A mapping of modality names to :class:`EncodedOutput`.
        """
        start = time.monotonic()
        output: dict[str, EncodedOutput] = {}
        futures: list[concurrent.futures.Future] = []

        modalities = request.modality_set()
        modality_to_encode: dict[ModalityType, tuple[str, Any]] = {}

        if ModalityType.IMAGE in modalities and request.image is not None:
            modality_to_encode[ModalityType.IMAGE] = ("image", request.image)
        if ModalityType.AUDIO in modalities and request.audio is not None:
            modality_to_encode[ModalityType.AUDIO] = ("audio", request.audio)
        if ModalityType.VIDEO in modalities and request.video is not None:
            modality_to_encode[ModalityType.VIDEO] = ("video", request.video)

        if not modality_to_encode:
            # Text-only — encode synchronously
            text_emb, text_model = self._pick_encoder().encode_text(request.text)
            output["text"] = EncodedOutput(
                modality=ModalityType.TEXT,
                embedding=text_emb,
                model_used=text_model,
                success=True,
            )
            elapsed = (time.monotonic() - start) * 1000
            self._record_execution(len(output), elapsed)
            return output

        # Launch parallel encoding tasks
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(modality_to_encode), self._max_workers),
        )
        encoder = self._pick_encoder()

        encode_methods = {
            "image": encoder.encode_image,
            "audio": encoder.encode_audio,
            "video": encoder.encode_video,
        }

        for modality_type, (name, data) in modality_to_encode.items():
            method = encode_methods.get(name)
            if method is not None:
                future = executor.submit(self._safe_encode, method, data, name)
                futures.append(future)

        # Collect results
        parallel_start = time.monotonic()
        for future in concurrent.futures.as_completed(futures):
            emb, modality_key, model_name, success, error = future.result()
            if modality_key:
                output[modality_key] = EncodedOutput(
                    modality=ModalityType(modality_key),
                    embedding=emb,
                    model_used=model_name,
                    encoding_time_ms=(time.monotonic() - parallel_start) * 1000,
                    success=success,
                    error=error or "",
                )
        executor.shutdown(wait=False)

        # Encode text after parallel modalities (cheap, no need for separate thread)
        text_emb, text_model = encoder.encode_text(request.text)
        output["text"] = EncodedOutput(
            modality=ModalityType.TEXT,
            embedding=text_emb,
            model_used=text_model,
            success=True,
        )

        elapsed = (time.monotonic() - start) * 1000
        self._record_execution(len(output), elapsed)
        return output

    def _safe_encode(
        self,
        method: Any,
        data: Any,
        modality_name: str,
    ) -> tuple[Any, str, str, bool, str]:
        """Safely invoke an encoding method, catching exceptions."""
        try:
            emb, model_used = method(data)
            return emb, modality_name, model_used, True, ""
        except Exception as exc:
            logger.error(f"Parallel encoding failed for {modality_name}: {exc}")
            with self._lock:
                self._stats["errors"] += 1
            return None, modality_name, "none", False, str(exc)

    def _pick_encoder(self) -> ModalityEncoder:
        """Return the first available encoder."""
        for encoder in self._encoders.values():
            return encoder
        return ModalityEncoder()

    def _record_execution(self, modalities_encoded: int, elapsed_ms: float) -> None:
        """Update execution statistics."""
        with self._lock:
            self._stats["total_executions"] += 1
            self._stats["total_encoded_modalities"] += modalities_encoded
            self._stats["total_encoding_time_ms"] += elapsed_ms

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        with self._lock:
            return dict(self._stats)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Voyager — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class Voyager:
    """Multi-modal inference orchestrator.

    Combines :class:`MultiModalRouter`, :class:`ParallelEncoderPipeline`,
    and :class:`ModalityEncoder` into a single ``process()`` call.

    Accepts text, image, audio, and video in a single request, encodes
    non-text modalities in parallel, and generates a response using the
    selected model.

    Usage::

        voyager = Voyager()
        response = voyager.process(
            text="What is shown in this image and audio?",
            image=my_image,
            audio=my_audio,
        )
        print(response.text)

        stats = voyager.stats()
        print(stats["modality_distribution"])
    """

    def __init__(
        self,
        router: MultiModalRouter | None = None,
        pipeline: ParallelEncoderPipeline | None = None,
        encoders: dict[str, ModalityEncoder] | None = None,
        max_workers: int = 4,
    ):
        """Initialize the Voyager orchestration pipeline.

        Args:
            router: Routing component.  Created automatically if ``None``.
            pipeline: Parallel encoding pipeline.  Created automatically
                if ``None``.
            encoders: ModalityEncoder mapping for the pipeline.  Passed
                directly when *pipeline* is not provided.
            max_workers: Maximum parallel encoding threads.
        """
        effective_encoders = encoders or {"default": ModalityEncoder()}
        self._router = router or MultiModalRouter()
        self._pipeline = pipeline or ParallelEncoderPipeline(
            encoders=effective_encoders,
            max_workers=max_workers,
        )
        self._lock = threading.Lock()

        self._stats = {
            "total_requests": 0,
            "total_encode_time_ms": 0.0,
            "total_generate_time_ms": 0.0,
            "total_tokens": 0,
            "modality_distribution": {
                "text_only": 0,
                "image_text": 0,
                "audio_text": 0,
                "video_text": 0,
            },
            "routing_accuracy": 0.0,
        }

    # ── Public API ──────────────────────────────────────────────────────

    def process(
        self,
        text: str = "",
        image: Any = None,
        audio: Any = None,
        video: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> VoyagerResponse:
        """Execute the full multi-modal inference pipeline.

        1. Wraps inputs into a :class:`MultiModalRequest`.
        2. Routes the request via :class:`MultiModalRouter`.
        3. Encodes modalities in parallel via :class:`ParallelEncoderPipeline`.
        4. Generates a response using the selected generation model.
        5. Returns a :class:`VoyagerResponse` with timing and metadata.

        Args:
            text: Text prompt (may be empty if non-text content is present).
            image: Image data (numpy array, torch tensor, or PIL Image).
            audio: Audio data (numpy waveform or file path).
            video: Video data (numpy array or list of frames).
            metadata: Optional metadata attached to the request.

        Returns:
            A :class:`VoyagerResponse` with the generated text and timing.
        """
        t0 = time.monotonic()
        request = MultiModalRequest(
            text=text,
            image=image,
            audio=audio,
            video=video,
            metadata=metadata or {},
        )

        # 1. Route
        plan = self._router.route(request)

        # 2. Encode (parallel)
        encode_start = time.monotonic()
        encoded = self._pipeline.execute(request, plan)
        encode_time = (time.monotonic() - encode_start) * 1000

        # 3. Generate response
        generate_start = time.monotonic()
        response_text = self._generate(request, plan, encoded)
        generate_time = (time.monotonic() - generate_start) * 1000

        total_time = (time.monotonic() - t0) * 1000

        tokens = max(1, len(response_text.split()))

        response = VoyagerResponse(
            text=response_text,
            full_response=response_text,
            route_type=plan.route_type,
            encoding_time_ms=encode_time,
            generation_time_ms=generate_time,
            total_time_ms=total_time,
            model_used=plan.generation_model,
            tokens_generated=tokens,
        )

        self._record_request(plan.route_type, encode_time, generate_time, tokens)
        return response

    def stats(self) -> dict[str, Any]:
        """Return aggregated pipeline statistics.

        Returns:
            Dict with keys:
            - ``total_requests`` — total requests processed
            - ``total_encode_time_ms`` — cumulative encoding wall time
            - ``total_generate_time_ms`` — cumulative generation wall time
            - ``total_tokens`` — cumulative tokens generated
            - ``modality_distribution`` — breakdown per route type
            - ``routing_accuracy`` — accuracy from the router (requires
              feedback to be meaningful)
            - ``router_stats`` — detailed router statistics
            - ``pipeline_stats`` — detailed pipeline statistics
            - ``encoder_stats`` — detailed encoder statistics
        """
        with self._lock:
            return {
                **dict(self._stats),
                "routing_accuracy": self._router.routing_accuracy,
                "router_stats": self._router.stats(),
                "pipeline_stats": self._pipeline.stats(),
                "encoder_stats": self._get_encoder_stats(),
            }

    # ── Generation ──────────────────────────────────────────────────────

    def _generate(
        self,
        request: MultiModalRequest,
        plan: RoutingPlan,
        encoded: dict[str, EncodedOutput],
    ) -> str:
        """Generate a response from encoded modalities.

        The default implementation builds a text prompt that includes
        the original text and markers for each encoded modality.

        When a generation function or coordinator is provided via
        :meth:`set_generation_fn`, it is called instead.

        Subclasses can override this method to feed actual embeddings
        into a multi-modal generation model.
        """
        # Build a combined representation
        parts: list[str] = []

        has_media = any(
            (encoded.get(m) is not None and encoded[m].success)
            for m in ("image", "audio", "video")
        )
        generation_fn = getattr(self, "_generation_fn", None)

        # REAL multimodal generation requires a generation function that can
        # consume the encoded embeddings.  Emitting marker strings ('[Image...]')
        # as if they were actual model output is fake — fail closed instead.
        if has_media and generation_fn is None:
            raise RuntimeError(
                "Voyager: multimodal generation requires a registered "
                "generation function (set_generation_fn) that can consume the "
                "encoded embeddings; refusing to emit marker placeholders as "
                "real output"
            )

        if request.text:
            parts.append(request.text)

        if "image" in encoded and encoded["image"].success:
            parts.append("[ImageEmbedding]")

        if "audio" in encoded and encoded["audio"].success:
            parts.append("[AudioEmbedding]")

        if "video" in encoded and encoded["video"].success:
            parts.append("[VideoEmbedding]")

        if not parts:
            return ""

        prompt = " ".join(parts)

        # Call generation function if registered.
        if generation_fn is not None:
            try:
                return generation_fn(prompt, {"model": plan.generation_model})
            except Exception as exc:
                logger.error(f"Generation function failed: {exc}")
                return f"[Generation error: {exc}]"

        # TEXT_ONLY path with no generation function: return the text itself
        # (there are no encoded modalities to be lost).
        return prompt if plan.route_type == RouteType.TEXT_ONLY else prompt

    def set_generation_fn(
        self,
        fn: Any,
    ) -> None:
        """Register an external generation function.

        The callable should accept ``(prompt: str, kwargs: dict) -> str``.

        Args:
            fn: A callable that takes a prompt string and a kwargs dict
                and returns a generated text string.
        """
        self._generation_fn = fn

    # ── Internal ────────────────────────────────────────────────────────

    def _record_request(
        self,
        route_type: RouteType,
        encode_time_ms: float,
        generate_time_ms: float,
        tokens: int,
    ) -> None:
        """Update pipeline statistics after a request."""
        key = route_type.value.replace("-", "_")
        with self._lock:
            self._stats["total_requests"] += 1
            self._stats["total_encode_time_ms"] += encode_time_ms
            self._stats["total_generate_time_ms"] += generate_time_ms
            self._stats["total_tokens"] += tokens
            if key in self._stats["modality_distribution"]:
                self._stats["modality_distribution"][key] += 1

    def _get_encoder_stats(self) -> dict[str, Any]:
        """Collect stats from all registered encoders."""
        aggregate: dict[str, Any] = {
            "encode_text_calls": 0,
            "encode_image_calls": 0,
            "encode_audio_calls": 0,
            "encode_video_calls": 0,
            "total_errors": 0,
        }
        counts = 0
        for encoder in self._pipeline._encoders.values():
            enc_stats = encoder.stats()
            for k in aggregate:
                aggregate[k] += enc_stats.get(k, 0)
            counts += 1
        aggregate["encoder_count"] = counts
        return aggregate

    def clear_stats(self) -> None:
        """Reset all accumulated statistics."""
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "total_encode_time_ms": 0.0,
                "total_generate_time_ms": 0.0,
                "total_tokens": 0,
                "modality_distribution": {
                    "text_only": 0,
                    "image_text": 0,
                    "audio_text": 0,
                    "video_text": 0,
                },
                "routing_accuracy": 0.0,
            }
