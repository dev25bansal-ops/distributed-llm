"""Regression tests for multimodal_engine honest non-text handling (C-01/O-09).

The engine originally discarded image/audio/document_pages after enum
selection and returned text generated from a synthetic "[IMAGE] {text}"
prompt with no warning — a silent correctness bug. The C-01 fix raised
NotImplementedError; the current contract routes the payload structurally
and returns a result TAGGED with the detected modality, with an explicit
"no encoder" notice instead of fabricated text. The core invariant is
unchanged and tested here: coordinator.generate is NEVER called with a
non-text payload.
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


def _engine():
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
    return mod, engine


def test_text_only_still_works():
    mod, engine = _engine()
    engine._coordinator.generate = MagicMock(return_value="hello world")

    result = engine.process(text="hi", image=None, audio=None, document_pages=None)

    assert result.text == "hello world"
    assert result.modality_type == mod.ModalityType.TEXT
    assert engine._stats["text_requests"] == 1


def test_image_payload_returns_tagged_result_no_encoder_notice():
    mod, engine = _engine()

    result = engine.process(text="what is this", image=torch.zeros(3, 8, 8))
    assert result.modality_type == mod.ModalityType.IMAGE
    assert "image" in result.text.lower()
    assert "encoder" in result.text.lower()  # explicit, not fabricated content


def test_audio_payload_returns_tagged_result_no_encoder_notice():
    mod, engine = _engine()

    result = engine.process(text="transcribe", audio=torch.zeros(16000))
    assert result.modality_type == mod.ModalityType.AUDIO
    assert "audio" in result.text.lower()
    assert "encoder" in result.text.lower()


def test_document_pages_returns_tagged_result_no_encoder_notice():
    mod, engine = _engine()

    pages = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]
    result = engine.process(text="summarize", document_pages=pages)
    assert result.modality_type == mod.ModalityType.DOCUMENT
    assert "document" in result.text.lower()
    assert "encoder" in result.text.lower()


def test_image_never_calls_coordinator_generate():
    """The engine must NOT call coordinator.generate with a synthetic prompt.

    This is the core of the C-01 silent-bug invariant: before, the engine
    would call coordinator.generate with "[IMAGE] {text}" and return the
    model's response as if vision worked. The test fails if generate was
    called with any non-text payload.
    """
    mod, engine = _engine()
    generate_mock = MagicMock()
    engine._coordinator.generate = generate_mock

    image = torch.zeros(3, 8, 8)
    result = engine.process(text="describe", image=image)
    generate_mock.assert_not_called()
    assert result.modality_type == mod.ModalityType.IMAGE
    assert engine._stats["unsupported_modality_requests"] == 1
