"""Multi-Modal Pipeline Parallelism — vision + language model support.

Extends the distributed pipeline to handle multi-modal inputs (images + text)
by routing vision encoding and language generation through separate pipeline
stages on different devices.

Features:
- Automatic modality detection (text, image, mixed)
- Vision encoder pipeline stage (CLIP, SigLIP, etc.)
- Language model pipeline stage (standard transformer)
- Image pre-processing and feature extraction
- Multi-modal token interleaving
- Adaptive routing based on input type
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


class ModalityType(enum.Enum):
    """Input modality classification."""
    TEXT = "text"
    IMAGE = "image"
    MULTI_MODAL = "multi_modal"
    AUDIO = "audio"  # Future


@dataclass
class ModalityDetection:
    """Result of modality detection on input."""
    primary_modality: ModalityType = ModalityType.TEXT
    has_images: bool = False
    has_text: bool = True
    image_count: int = 0
    text_length: int = 0
    image_urls: list[str] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class VisionPipelineConfig:
    """Configuration for the vision encoder pipeline."""
    vision_model: str = "openai/clip-vit-large-patch14-336"
    image_size: tuple[int, int] = (336, 336)
    patch_size: int = 14
    vision_dim: int = 1024
    num_image_tokens: int = 576  # (336/14)^2
    dtype: str = "float16"
    device: str = "auto"


@dataclass
class MultiModalPipelineConfig:
    """Configuration for multi-modal pipeline parallelism."""
    enabled: bool = False
    vision: VisionPipelineConfig = field(default_factory=VisionPipelineConfig)
    # How to map image features into the language model's embedding space
    projection_type: str = "linear"  # linear, mlp, cross_attention
    # Max images per request
    max_images: int = 5
    # Whether vision encoder runs on a separate device
    separate_vision_device: bool = True


def detect_modality(messages: list[dict[str, Any]]) -> ModalityDetection:
    """Detect the modality of input messages.

    Analyzes the message content to determine if it contains text,
    images, or both.

    Args:
        messages: List of chat messages with content.

    Returns:
        ModalityDetection with detected modality info.
    """
    detection = ModalityDetection()

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                detection.has_text = True
                detection.text_length += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text.strip():
                            detection.has_text = True
                            detection.text_length += len(text)
                    elif item_type == "image_url":
                        detection.has_images = True
                        detection.image_count += 1
                        url_obj = item.get("image_url", {})
                        if isinstance(url_obj, dict):
                            detection.image_urls.append(url_obj.get("url", ""))
                        elif hasattr(url_obj, "url"):
                            detection.image_urls.append(url_obj.url)

    # Classify primary modality
    if detection.has_images and detection.has_text:
        detection.primary_modality = ModalityType.MULTI_MODAL
    elif detection.has_images:
        detection.primary_modality = ModalityType.IMAGE
    else:
        detection.primary_modality = ModalityType.TEXT

    return detection


class VisionEncoder:
    """Vision encoder for processing images into feature vectors.

    Supports CLIP, SigLIP, and similar vision models.
    Runs on a dedicated device for pipeline parallelism.
    """

    def __init__(self, config: VisionPipelineConfig | None = None):
        self._config = config or VisionPipelineConfig()
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._loaded = False

    def load(self) -> None:
        """Load the vision model."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading vision model: {self._config.vision_model}")
            self._processor = CLIPProcessor.from_pretrained(self._config.vision_model)
            self._model = CLIPModel.from_pretrained(
                self._config.vision_model,
                torch_dtype=torch.float16 if self._config.dtype == "float16" else torch.float32,
            )

            # Move to device
            if self._config.device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self._config.device

            self._model = self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Vision model loaded on {self._device}")

        except ImportError:
            logger.warning("transformers not available, vision encoding disabled")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def encode_images(self, images: list[Any]) -> torch.Tensor | None:
        """Encode images into feature vectors.

        Args:
            images: List of PIL Images or image tensors.

        Returns:
            Tensor of shape (num_images, num_patches, vision_dim)
            or None if encoder not loaded.
        """
        if not self._loaded or self._model is None:
            logger.warning("Vision encoder not loaded")
            return None

        try:
            inputs = self._processor(
                images=images, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                # Get vision features (last hidden state)
                vision_outputs = self._model.vision_model(**inputs)
                # Shape: (batch, num_patches, vision_dim)
                features = vision_outputs.last_hidden_state

            return features

        except Exception as e:
            logger.error(f"Vision encoding failed: {e}")
            return None

    def encode_single_image(self, image: Any) -> torch.Tensor | None:
        """Encode a single image."""
        return self.encode_images([image])


class ModalityRouter:
    """Routes requests through the appropriate pipeline stages.

    For text-only requests: standard language pipeline.
    For image requests: vision encoder → projection → language pipeline.
    For mixed requests: vision encoder + text embedding → merged → language pipeline.
    """

    def __init__(
        self,
        vision_config: VisionPipelineConfig | None = None,
        separate_device: bool = True,
    ):
        self._vision_config = vision_config or VisionPipelineConfig()
        self._separate_device = separate_device
        self._vision_encoder: VisionEncoder | None = None
        self._projection_layer: torch.nn.Module | None = None

    def initialize(self) -> None:
        """Initialize the vision pipeline."""
        self._vision_encoder = VisionEncoder(self._vision_config)
        self._vision_encoder.load()

    @property
    def has_vision(self) -> bool:
        return self._vision_encoder is not None and self._vision_encoder.is_loaded

    def route_request(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[ModalityDetection, torch.Tensor | None]:
        """Route a request through the appropriate pipeline.

        Args:
            messages: Input messages.

        Returns:
            (detection, vision_features) — vision_features is None for text-only.
        """
        detection = detect_modality(messages)

        if detection.has_images and self.has_vision:
            # Load and encode images
            images = self._load_images(detection.image_urls)
            if images:
                features = self._vision_encoder.encode_images(images)
                return detection, features

        return detection, None

    def _load_images(self, urls: list[str]) -> list[Any]:
        """Load images from URLs or base64 data URIs."""
        images = []
        for url in urls:
            try:
                if url.startswith("data:"):
                    # Base64 data URI
                    import base64
                    import io
                    from PIL import Image

                    header, data = url.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                elif url.startswith("http"):
                    # HTTP URL
                    import httpx
                    from PIL import Image
                    import io

                    resp = httpx.get(url, timeout=10, follow_redirects=True)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    images.append(img)
                else:
                    # Local file path
                    from PIL import Image
                    img = Image.open(url).convert("RGB")
                    images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {url[:50]}...: {e}")
        return images


def build_projection_layer(
    vision_dim: int,
    language_dim: int,
    projection_type: str = "linear",
) -> torch.nn.Module:
    """Build a projection layer to map vision features into language embedding space.

    Args:
        vision_dim: Dimension of vision features.
        language_dim: Dimension of language model embeddings.
        projection_type: Type of projection (linear, mlp, cross_attention).

    Returns:
        Projection module.
    """
    if projection_type == "linear":
        return torch.nn.Linear(vision_dim, language_dim)
    elif projection_type == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(vision_dim, language_dim),
            torch.nn.GELU(),
            torch.nn.Linear(language_dim, language_dim),
        )
    else:
        logger.warning(f"Unknown projection type '{projection_type}', using linear")
        return torch.nn.Linear(vision_dim, language_dim)


def interleave_multimodal_tokens(
    text_embeddings: torch.Tensor,
    image_features: torch.Tensor,
    image_positions: list[int],
    num_image_tokens: int,
) -> torch.Tensor:
    """Interleave image features into text embeddings at specified positions.

    For models like LLaVA where image tokens are inserted at specific
    positions in the text sequence.

    Args:
        text_embeddings: Shape (batch, seq_len, dim)
        image_features: Shape (batch, num_images, num_patches, dim)
        image_positions: Positions in the text sequence where images should be inserted.
        num_image_tokens: Number of tokens per image.

    Returns:
        Combined embeddings with images interleaved.
    """
    batch_size, text_len, dim = text_embeddings.shape
    total_image_tokens = len(image_positions) * num_image_tokens
    total_len = text_len + total_image_tokens

    # Create output tensor
    output = torch.zeros(batch_size, total_len, dim, device=text_embeddings.device, dtype=text_embeddings.dtype)

    text_idx = 0
    output_idx = 0
    img_idx = 0

    for pos in sorted(image_positions):
        # Copy text before this image position
        tokens_before = pos - text_idx
        if tokens_before > 0:
            output[:, output_idx:output_idx + tokens_before] = text_embeddings[:, text_idx:pos]
            output_idx += tokens_before

        # Insert image features
        if img_idx < image_features.shape[1]:
            img_feats = image_features[:, img_idx]  # (batch, num_patches, dim)
            output[:, output_idx:output_idx + num_image_tokens] = img_feats[:, :num_image_tokens]
            output_idx += num_image_tokens
            img_idx += 1

        text_idx = pos

    # Copy remaining text
    remaining = text_len - text_idx
    if remaining > 0:
        output[:, output_idx:output_idx + remaining] = text_embeddings[:, text_idx:]

    return output


# ── Module-level singleton ──────────────────────────────────────────────────

_router: ModalityRouter | None = None


def get_modality_router(
    config: MultiModalPipelineConfig | None = None,
) -> ModalityRouter:
    """Get or create the module-level ModalityRouter singleton."""
    global _router
    if _router is None:
        cfg = config or MultiModalPipelineConfig()
        _router = ModalityRouter(
            vision_config=cfg.vision,
            separate_device=cfg.separate_vision_device,
        )
    return _router
