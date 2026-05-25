"""Audio transcription and TTS tests: /v1/audio/transcriptions, /v1/audio/speech."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


class TestTranscription:
    """POST /v1/audio/transcriptions."""
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._whisper_model = None
        coord._whisper_processor = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_transcribe_wav(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "received" in data["text"].lower()
        assert "bytes" in data["text"]

    def test_transcribe_unsupported_format(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp2", b"fake audio data", "audio/mp2")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert "unsupported" in err["message"].lower()
        assert ".mp2" in err["message"]

    def test_transcribe_verbose_json(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "transcribe"
        assert "language" in data
        assert "duration" in data
        assert "text" in data
        assert "received" in data["text"].lower()

    def test_transcribe_verbose_json_with_word_timestamps(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities": "word,segment",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["words"] is not None
        assert len(data["words"]) > 0
        assert data["segments"] is not None
        assert len(data["segments"]) > 0

    def test_transcribe_whisper_fallback(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "placeholder" in data["text"].lower() or "received" in data["text"].lower()

    def test_transcribe_too_large(self):
        resp = TestClient(app).post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"x" * (26 * 1024 * 1024), "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert "too large" in err["message"].lower()

    def test_transcribe_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"fake audio data", "audio/wav")},
                data={"model": "whisper-1"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestDurationEstimation:
    """_estimate_duration accuracy."""

    def test_duration_10kb(self):
        from distllm.api.routes.audio import _estimate_duration
        assert _estimate_duration(b"x" * 10240) == 1.0

    def test_duration_5kb(self):
        from distllm.api.routes.audio import _estimate_duration
        assert _estimate_duration(b"x" * 5120) == 0.5

    def test_duration_zero_bytes(self):
        from distllm.api.routes.audio import _estimate_duration
        assert _estimate_duration(b"") == 0.0

    def test_duration_100kb(self):
        from distllm.api.routes.audio import _estimate_duration
        assert _estimate_duration(b"x" * 102400) == 10.0


class TestTTS:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._tts_model = None
        coord._tts_processor = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_tts_default(self):
        resp = TestClient(app).post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"

    def test_tts_voice_selection(self):
        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            resp = TestClient(app).post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": voice},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/mpeg"

    def test_tts_fallback_sine_wave(self):
        resp = TestClient(app).post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert len(resp.content) > 0

    def test_tts_speed_control(self):
        resp_slow = TestClient(app).post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy", "speed": 0.5},
        )
        resp_fast = TestClient(app).post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy", "speed": 2.0},
        )
        assert resp_slow.status_code == 200
        assert resp_fast.status_code == 200
        assert resp_slow.content != resp_fast.content

    def test_tts_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": "alloy"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original
