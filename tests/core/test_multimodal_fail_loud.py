"""Regression tests for multimodal_engine fail-loud (C-01).

The engine previously discarded image/audio/document_pages after enum
selection, formatted the prompt as a literal "[IMAGE] {text}" string,
and returned text generated from that synthetic prompt with no warning.
This was a silent correctness bug. C-01 fix raises NotImplementedError
on any non-text modality payload.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


def _load_module():
    sys.modules.pop("distllm.core.multimodal_engine", None)
    return importlib.import_module("distllm.core.multimodal_engine")


def test_text_only_still_works():
    mod = _load_module()
    engine = mod.MultimodalEngine.__new__(mod.MultimodalEngine)
    engine._stats = {
        "total_requests": 0,
        "text_requests": 0,
        "vision_requests": 0,
        "audio_requests": 0,
        "document_requests": 0,
        "unsupported_modality_requests": 0,
    }
    engine._coordinator = MagicMock()
    engine._coordinator.generate = MagicMock(return_value="hello world")

    result = engine.process(text="hi", image=None, audio=None, document_pages=None)

    assert result.text == "hello world"
    assert result.modality_type == mod.ModalityType.TEXT
    assert engine._stats["text_requests"] == 1


def test_image_payload_raises_not_implemented():
    mod = _load_module()
    engine = mod.MultimodalEngine.__new__(mod.MultimodalEngine)
    engine._stats = {
        "total_requests": 0,
        "text_requests": 0,
        "vision_requests": 0,
        "audio_requests": 0,
        "document_requests": 0,
        "unsupported_modality_requests": 0,
    }
    engine._coordinator = MagicMock()
    engine._coordinator.generate = MagicMock()

    image = torch.zeros(3, 8, 8)
    with pytest.raises(NotImplementedError) as exc:
        engine.process(text="what is this", image=image)
    assert "image" in str(exc.value).lower()
    assert engine._coordinator.generate.call_count == 0


def test_audio_payload_raises_not_implemented():
    mod = _load_module()
    engine = mod.MultimodalEngine.__new__(mod.MultimodalEngine)
    engine._stats = {
        "total_requests": 0,
        "text_requests": 0,
        "vision_requests": 0,
        "audio_requests": 0,
        "document_requests": 0,
        "unsupported_modality_requests": 0,
    }
    engine._coordinator = MagicMock()
    engine._coordinator.generate = MagicMock()

    audio = torch.zeros(16000)
    with pytest.raises(NotImplementedError) as exc:
        engine.process(text="transcribe", audio=audio)
    assert "audio" in str(exc.value).lower()


def test_document_pages_raises_not_implemented():
    mod = _load_module()
    engine = mod.MultimodalEngine.__new__(mod.MultimodalEngine)
    engine._stats = {
        "total_requests": 0,
        "text_requests": 0,
        "vision_requests": 0,
        "audio_requests": 0,
        "document_requests": 0,
        "unsupported_modality_requests": 0,
    }
    engine._coordinator = MagicMock()
    engine._coordinator.generate = MagicMock()

    pages = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]
    with pytest.raises(NotImplementedError) as exc:
        engine.process(text="summarize", document_pages=pages)
    assert "document" in str(exc.value).lower()


def test_image_never_calls_coordinator_generate():
    """The engine must NOT call coordinator.generate with a synthetic prompt.

    This is the core of the C-01 silent-bug fix: before, the engine would
    call coordinator.generate with "[IMAGE] {text}" and return the model's
    response as if vision worked. The test fails if generate was called
    with any non-text payload.
    """
    mod = _load_module()
    engine = mod.MultimodalEngine.__new__(mod.MultimodalEngine)
    engine._stats = {
        "total_requests": 0,
        "text_requests": 0,
        "vision_requests": 0,
        "audio_requests": 0,
        "document_requests": 0,
        "unsupported_modality_requests": 0,
    }
    engine._coordinator = MagicMock()
    generate_mock = MagicMock()
    engine._coordinator.generate = generate_mock

    image = torch.zeros(3, 8, 8)
    with pytest.raises(NotImplementedError):
        engine.process(text="describe", image=image)
    generate_mock.assert_not_called()
    assert engine._stats["unsupported_modality_requests"] == 1
