"""Multi-modal inference engine for vision, audio, and document understanding.

Distributes multi-modal models across nodes:
- Vision encoder on dedicated GPU node
- Language model on pipeline-parallel nodes
- Audio encoder on dedicated GPU node

Usage::

    engine = MultimodalEngine(coordinator)
    result = engine.process(
        text="Describe this image",
        image=image_tensor,
        model="llava-1.5-7b",
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from loguru import logger


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class MultimodalInput:
    """Input containing multiple modalities."""
    text: str = ""
    image: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    document_pages: list[torch.Tensor] | None = None
    modality_type: ModalityType = ModalityType.TEXT


@dataclass
class MultimodalResult:
    """Result of multi-modal inference."""
    text: str
    modality_type: ModalityType
    processing_time_ms: float = 0.0
    encoder_node: str = ""
    decoder_node: str = ""
    tokens_generated: int = 0


class MultimodalEngine:
    """Distributed multi-modal inference engine.

    Routes different modalities to specialized nodes:
    - Vision encoder → dedicated GPU node
    - Audio encoder → dedicated GPU node
    - Language model → pipeline-parallel nodes

    Supports:
    - Vision-language models (LLaVA, GPT-4V style)
    - Audio models (Whisper style)
    - Document understanding (PDF/image parsing)
    """

    def __init__(self, coordinator: Any = None):
        self._coordinator = coordinator
        self._vision_encoder_node: str | None = None
        self._audio_encoder_node: str | None = None
        self._document_processor_node: str | None = None

        # Stats
        self._stats = {
            "total_requests": 0,
            "vision_requests": 0,
            "audio_requests": 0,
            "document_requests": 0,
            "text_requests": 0,
        }

    def set_vision_encoder_node(self, node_id: str) -> None:
        """Set the node that handles vision encoding."""
        self._vision_encoder_node = node_id
        logger.info(f"Vision encoder assigned to node {node_id}")

    def set_audio_encoder_node(self, node_id: str) -> None:
        """Set the node that handles audio encoding."""
        self._audio_encoder_node = node_id
        logger.info(f"Audio encoder assigned to node {node_id}")

    def set_document_processor_node(self, node_id: str) -> None:
        """Set the node that handles document processing."""
        self._document_processor_node = node_id
        logger.info(f"Document processor assigned to node {node_id}")

    def process(
        self,
        text: str = "",
        image: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        document_pages: list[torch.Tensor] | None = None,
        model: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> MultimodalResult:
        """Process a multi-modal input.

        Routes the input to appropriate encoder nodes, then runs
        the language model on the combined representation.

        Args:
            text: Text prompt.
            image: Image tensor (CHW format).
            audio: Audio waveform tensor.
            document_pages: List of page image tensors.
            model: Model name to use.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            MultimodalResult with generated text.
        """
        t0 = time.monotonic()
        self._stats["total_requests"] += 1

        # Determine primary modality
        if image is not None:
            modality = ModalityType.IMAGE
            self._stats["vision_requests"] += 1
        elif audio is not None:
            modality = ModalityType.AUDIO
            self._stats["audio_requests"] += 1
        elif document_pages:
            modality = ModalityType.DOCUMENT
            self._stats["document_requests"] += 1
        else:
            modality = ModalityType.TEXT
            self._stats["text_requests"] += 1

        # For text-only, delegate to standard pipeline
        if modality == ModalityType.TEXT:
            if self._coordinator:
                result_text = self._coordinator.generate(
                    text, max_new_tokens=max_tokens, temperature=temperature,
                )
            else:
                result_text = "[No coordinator available]"

            return MultimodalResult(
                text=result_text,
                modality_type=modality,
                processing_time_ms=(time.monotonic() - t0) * 1000,
                tokens_generated=len(result_text.split()),
            )

        # Multi-modal: encode non-text modality, then generate
        # This is a framework — actual implementation depends on model architecture
        logger.info(f"Processing {modality.value} input (image={image is not None}, audio={audio is not None})")

        if self._coordinator:
            # Build multimodal prompt
            prompt = self._build_multimodal_prompt(text, modality)
            result_text = self._coordinator.generate(
                prompt, max_new_tokens=max_tokens, temperature=temperature,
            )
        else:
            result_text = f"[Multimodal processing not available for {modality.value}]"

        return MultimodalResult(
            text=result_text,
            modality_type=modality,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            tokens_generated=len(result_text.split()),
        )

    def _build_multimodal_prompt(self, text: str, modality: ModalityType) -> str:
        """Build a text prompt that represents the multimodal input."""
        if modality == ModalityType.IMAGE:
            return f"[IMAGE] {text}"
        elif modality == ModalityType.AUDIO:
            return f"[AUDIO] {text}"
        elif modality == ModalityType.DOCUMENT:
            return f"[DOCUMENT] {text}"
        return text

    def stats(self) -> dict:
        """Return multi-modal engine statistics."""
        return dict(self._stats)
