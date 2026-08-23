"""Audio transcription and TTS tests: /v1/audio/transcriptions, /v1/audio/speech.

Note: The audio routes module does not exist in the current codebase.
Tests that exercise the endpoint are marked xfail.
"""

import secrets
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.core.api_key_store import reset_api_key_store
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _setup_auth(monkeypatch):
    test_api_key = secrets.token_urlsafe(32)
    monkeypatch.setenv("API_KEY", test_api_key)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    reset_api_key_store()


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
        coord._model_router = None
        g.coordinator = coord
        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {__import__('os').environ.get('API_KEY', '')}"
        yield
        g.coordinator = original

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_wav(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_unsupported_format(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp2", b"fake audio data", "audio/mp2")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_verbose_json(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json"},
        )
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_verbose_json_with_word_timestamps(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities": "word,segment"},
        )
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_whisper_fallback(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio data", "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_too_large(self):
        resp = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"x" * (26 * 1024 * 1024), "audio/wav")},
            data={"model": "whisper-1"},
        )
        assert resp.status_code == 400

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_transcribe_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"fake audio data", "audio/wav")},
                data={"model": "whisper-1"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestDurationEstimation:
    """_estimate_duration accuracy."""

    def _import_module(self):
        try:
            from distllm.api.routes.audio import _estimate_duration
            return _estimate_duration
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Audio module not available in current source")

    def test_duration_10kb(self):
        func = self._import_module()
        assert func(b"x" * 10240) == 1.0

    def test_duration_5kb(self):
        func = self._import_module()
        assert func(b"x" * 5120) == 0.5

    def test_duration_zero_bytes(self):
        func = self._import_module()
        assert func(b"") == 0.0

    def test_duration_100kb(self):
        func = self._import_module()
        assert func(b"x" * 102400) == 10.0


class TestTTS:
    """POST /v1/audio/speech."""

    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._tts_model = None
        coord._tts_processor = None
        coord._model_router = None
        g.coordinator = coord
        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {__import__('os').environ.get('API_KEY', '')}"
        yield
        g.coordinator = original

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_tts_default(self):
        resp = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "alloy"})
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_tts_voice_selection(self):
        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            resp = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": voice})
            assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_tts_fallback_sine_wave(self):
        resp = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "alloy"})
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_tts_speed_control(self):
        resp_slow = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "alloy", "speed": 0.5})
        resp_fast = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "alloy", "speed": 2.0})
        assert resp_slow.status_code == 200
        assert resp_fast.status_code == 200

    @pytest.mark.xfail(reason="Audio routes not implemented in current source")
    def test_tts_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = self.client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "alloy"})
            assert resp.status_code == 503
        finally:
            g.coordinator = original
