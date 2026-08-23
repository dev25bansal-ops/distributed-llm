"""Tests for MultimodalEngine using real objects via load_module pattern."""

from __future__ import annotations

import torch
import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mm_mod = load_module("distllm/core/multimodal_engine.py")
MultimodalEngine = _mm_mod.MultimodalEngine
MultimodalInput = _mm_mod.MultimodalInput
MultimodalResult = _mm_mod.MultimodalResult
ModalityType = _mm_mod.ModalityType


class TestModalityType:
    """ModalityType enum values."""

    def test_values(self) -> None:
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.DOCUMENT.value == "document"


class TestMultimodalInput:
    """MultimodalInput dataclass construction."""

    def test_minimal(self) -> None:
        inp = MultimodalInput()
        assert inp.text == ""
        assert inp.image is None
        assert inp.audio is None
        assert inp.modality_type == ModalityType.TEXT

    def test_with_image(self) -> None:
        img = torch.zeros(3, 224, 224)
        inp = MultimodalInput(text="desc", image=img)
        assert inp.image is not None
        assert inp.image.shape == (3, 224, 224)

    def test_with_multiple(self) -> None:
        inp = MultimodalInput(
            text="test",
            image=torch.zeros(1, 1),
            audio=torch.zeros(16000),
        )
        assert inp.text == "test"
        assert inp.image is not None
        assert inp.audio is not None


class TestMultimodalResult:
    """MultimodalResult dataclass construction."""

    def test_minimal(self) -> None:
        result = MultimodalResult(text="hello", modality_type=ModalityType.TEXT)
        assert result.text == "hello"
        assert result.processing_time_ms == 0.0

    def test_full(self) -> None:
        result = MultimodalResult(
            text="ok",
            modality_type=ModalityType.IMAGE,
            processing_time_ms=123.4,
            encoder_node="node-1",
            tokens_generated=42,
        )
        assert result.encoder_node == "node-1"
        assert result.tokens_generated == 42


class TestMultimodalEngineConstruction:
    """MultimodalEngine construction and configuration."""

    def test_minimal_construction(self) -> None:
        engine = MultimodalEngine()
        assert engine._coordinator is None

    def test_with_coordinator(self) -> None:
        engine = MultimodalEngine(coordinator="coord")
        assert engine._coordinator == "coord"

    def test_initial_stats(self) -> None:
        engine = MultimodalEngine()
        stats = engine.stats()
        assert stats["total_requests"] == 0
        assert stats["vision_requests"] == 0
        assert stats["audio_requests"] == 0
        assert stats["document_requests"] == 0
        assert stats["text_requests"] == 0

    def test_set_vision_encoder_node(self) -> None:
        engine = MultimodalEngine()
        engine.set_vision_encoder_node("node-vision")
        assert engine._vision_encoder_node == "node-vision"

    def test_set_audio_encoder_node(self) -> None:
        engine = MultimodalEngine()
        engine.set_audio_encoder_node("node-audio")
        assert engine._audio_encoder_node == "node-audio"

    def test_set_document_processor_node(self) -> None:
        engine = MultimodalEngine()
        engine.set_document_processor_node("node-doc")
        assert engine._document_processor_node == "node-doc"


class TestMultimodalEngineProcessText:
    """MultimodalEngine.process with text-only input."""

    def test_text_only_no_coordinator(self) -> None:
        engine = MultimodalEngine()
        result = engine.process(text="Hello")
        assert isinstance(result, MultimodalResult)
        assert result.modality_type == ModalityType.TEXT
        assert "[No coordinator" in result.text

    def test_text_only_increments_stats(self) -> None:
        engine = MultimodalEngine()
        engine.process(text="Hello")
        stats = engine.stats()
        assert stats["total_requests"] == 1
        assert stats["text_requests"] == 1

    def test_text_only_with_mock_coordinator(self) -> None:
        class MockCoord:
            def generate(self, prompt, max_new_tokens=256, temperature=0.7):
                return f"Response to: {prompt}"

        engine = MultimodalEngine(coordinator=MockCoord())
        result = engine.process(text="Hi", max_tokens=50, temperature=0.5)
        assert "Response to: Hi" in result.text
        assert result.tokens_generated > 0


class TestMultimodalEngineProcessMultimodal:
    """MultimodalEngine.process with image/audio/document."""

    def test_image_input_no_coordinator(self) -> None:
        engine = MultimodalEngine()
        img = torch.zeros(3, 100, 100)
        result = engine.process(image=img, text="Describe")
        assert result.modality_type == ModalityType.IMAGE

    def test_audio_input_no_coordinator(self) -> None:
        engine = MultimodalEngine()
        audio = torch.zeros(16000)
        result = engine.process(audio=audio, text="Transcribe")
        assert result.modality_type == ModalityType.AUDIO

    def test_document_input_no_coordinator(self) -> None:
        engine = MultimodalEngine()
        pages = [torch.zeros(3, 100, 100)]
        result = engine.process(document_pages=pages, text="Summarize")
        assert result.modality_type == ModalityType.DOCUMENT

    def test_image_increments_vision_stats(self) -> None:
        engine = MultimodalEngine()
        img = torch.zeros(3, 100, 100)
        engine.process(image=img, text="Describe")
        stats = engine.stats()
        assert stats["vision_requests"] == 1

    def test_audio_increments_audio_stats(self) -> None:
        engine = MultimodalEngine()
        engine.process(audio=torch.zeros(16000), text="Transcribe")
        stats = engine.stats()
        assert stats["audio_requests"] == 1

    def test_document_increments_document_stats(self) -> None:
        engine = MultimodalEngine()
        engine.process(document_pages=[torch.zeros(1, 1)], text="Sum")
        stats = engine.stats()
        assert stats["document_requests"] == 1

    def test_processing_time_ms_nonzero(self) -> None:
        engine = MultimodalEngine()
        result = engine.process(text="test")
        assert result.processing_time_ms > 0

    def test_image_with_mock_coordinator(self) -> None:
        class MockCoord:
            def generate(self, prompt, max_new_tokens=256, temperature=0.7):
                return f"Generated: {prompt}"

        engine = MultimodalEngine(coordinator=MockCoord())
        img = torch.zeros(3, 100, 100)
        result = engine.process(image=img, text="Describe", model="llava")
        assert "[IMAGE]" in result.text


class TestMultimodalEngineBuildPrompt:
    """_build_multimodal_prompt formats prompts correctly."""

    def test_image_prompt_prefix(self) -> None:
        engine = MultimodalEngine()
        prompt = engine._build_multimodal_prompt("Describe", ModalityType.IMAGE)
        assert prompt == "[IMAGE] Describe"

    def test_audio_prompt_prefix(self) -> None:
        engine = MultimodalEngine()
        prompt = engine._build_multimodal_prompt("What?", ModalityType.AUDIO)
        assert prompt == "[AUDIO] What?"

    def test_document_prompt_prefix(self) -> None:
        engine = MultimodalEngine()
        prompt = engine._build_multimodal_prompt("Sum", ModalityType.DOCUMENT)
        assert prompt == "[DOCUMENT] Sum"

    def test_text_prompt_no_prefix(self) -> None:
        engine = MultimodalEngine()
        prompt = engine._build_multimodal_prompt("Hello", ModalityType.TEXT)
        assert prompt == "Hello"

    def test_processing_time_increases(self) -> None:
        engine = MultimodalEngine()
        r1 = engine.process(text="a")
        r2 = engine.process(text="b")
        assert r2.processing_time_ms > 0
