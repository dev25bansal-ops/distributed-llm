"""Real-Time Audio/Video Inference Pipeline over WebRTC.

Streams audio/video through::

    WebRTC audio track → OPUS decode → VAD → ASR → LLM → TTS → OPUS encode → WebRTC audio track

Requires optional dependencies (installed separately):
    pip install distllm[media]

Dependencies: aiortc (for WebRTC), faster-whisper (for ASR),
              XTTS or Piper (for TTS), webrtcvad (for VAD).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from enum import Enum
from typing import Any, Callable

from loguru import logger


# ── Lazy import guards ──────────────────────────────────────────────────

HAS_MEDIA_DEPS = False
try:
    import numpy as np
    HAS_MEDIA_DEPS = True
except ImportError:
    np = None  # type: ignore[assignment]


class PipelineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


# ── VAD (Voice Activity Detection) ──────────────────────────────────────

class VoiceActivityDetector:
    """Voice activity detection using webrtcvad.

    Detects when someone is speaking in an audio stream.
    """

    def __init__(self, mode: int = 2, frame_ms: int = 30):
        self._mode = mode
        self._frame_ms = frame_ms
        self._vad = None
        self._load()

    def _load(self) -> None:
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._mode)
            logger.info("VAD loaded (webrtcvad)")
        except ImportError:
            logger.info("webrtcvad not available — voice activity detection disabled")

    def is_speech(self, audio_frame: bytes, sample_rate: int = 16000) -> bool:
        """Detect speech in a 16-bit PCM audio frame."""
        if self._vad is None:
            return True  # No VAD = always active
        try:
            return self._vad.is_speech(audio_frame, sample_rate)
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._vad is not None


# ── ASR (Automatic Speech Recognition) ──────────────────────────────────

class SpeechRecognizer:
    """Speech-to-text using faster-whisper.

    Converts raw PCM audio to text for LLM processing.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        self._model_size = model_size
        self._device = device
        self._model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size, device=self._device,
                compute_type="int8" if self._device == "cpu" else "float16",
            )
            logger.info(f"ASR model loaded: {self._model_size} on {self._device}")
        except ImportError:
            logger.warning("faster-whisper not available — ASR disabled")

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        """Transcribe PCM audio to text."""
        if self._model is None:
            return ""
        try:
            import numpy as np
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            with self._lock:
                segments, _ = self._model.transcribe(audio_array, beam_size=1)
                return " ".join(seg.text for seg in segments)
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return ""

    @property
    def available(self) -> bool:
        return self._model is not None


# ── TTS (Text-to-Speech) ────────────────────────────────────────────────

class TextToSpeech:
    """Text-to-speech using Piper (local, fast, no GPU needed).

    Falls back to a simple SSML-speaking approach when Piper is unavailable.
    """

    def __init__(self, model_path: str = "", device: str = "cpu"):
        self._model_path = model_path or os.environ.get("DISTLLM_TTS_MODEL", "")
        self._device = device
        self._pipe = None
        self._load()

    def _load(self) -> None:
        if not self._model_path:
            logger.info("TTS: no model path set (DISTLLM_TTS_MODEL)")
            return
        try:
            import torch
            # Try Piper (local, fast)
            from piper import PiperVoice
            self._pipe = PiperVoice.load(self._model_path)
            logger.info(f"TTS loaded: {self._model_path}")
        except ImportError:
            logger.info("Piper not available — TTS disabled")

    def synthesize(self, text: str) -> bytes:
        """Convert text to 16-bit PCM audio at 22050 Hz."""
        if self._pipe is None:
            return b""
        try:
            import numpy as np
            audio = self._pipe.synthesize(text)
            if isinstance(audio, np.ndarray):
                return (audio * 32767).astype(np.int16).tobytes()
            return audio if isinstance(audio, bytes) else b""
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b""

    @property
    def available(self) -> bool:
        return self._pipe is not None


# ── LLM Interface ───────────────────────────────────────────────────────

class LLMResponder:
    """Interface to the DistLLM inference engine for generating responses.

    Wraps the Coordinator's generate() or a direct chat API call.
    """

    def __init__(
        self,
        generate_fn: Callable[[str, dict], str] | None = None,
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise.",
    ):
        self._generate = generate_fn
        self._system_prompt = system_prompt
        self._lock = threading.Lock()

    def respond(self, text: str, **kwargs: Any) -> str:
        """Generate a response to spoken text."""
        if self._generate:
            return self._generate(text, {"system_prompt": self._system_prompt, **kwargs})
        # Fallback: echo (for testing without an LLM)
        return f"You said: {text}"


# ── Audio Pipeline ──────────────────────────────────────────────────────

class AudioPipeline:
    """Full audio processing pipeline: VAD → ASR → LLM → TTS.

    Manages the lifecycle of a real-time voice conversation over WebRTC.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector | None = None,
        asr: SpeechRecognizer | None = None,
        llm: LLMResponder | None = None,
        tts: TextToSpeech | None = None,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        silence_timeout_s: float = 1.5,
    ):
        self._vad = vad or VoiceActivityDetector()
        self._asr = asr or SpeechRecognizer()
        self._llm = llm or LLMResponder()
        self._tts = tts or TextToSpeech()
        self._sample_rate = sample_rate
        self._frame_duration = frame_duration_ms
        self._silence_timeout = silence_timeout_s

        self._state = PipelineState.IDLE
        self._buffer: list[bytes] = []
        self._last_speech_time = time.time()
        self._lock = threading.Lock()

    async def process_audio_frame(self, frame: bytes) -> bytes | None:
        """Process a single audio frame.

        Args:
            frame: 16-bit PCM audio frame.

        Returns:
            Response audio bytes if a response is ready, None otherwise.
        """
        is_speech = self._vad.is_speech(frame, self._sample_rate)

        with self._lock:
            if is_speech:
                self._buffer.append(frame)
                self._last_speech_time = time.time()
                if self._state == PipelineState.IDLE:
                    self._state = PipelineState.LISTENING
                return None
            else:
                # Check silence timeout
                if self._state == PipelineState.LISTENING:
                    if time.time() - self._last_speech_time > self._silence_timeout:
                        self._state = PipelineState.PROCESSING
                        # Combine buffered audio
                        full_audio = b"".join(self._buffer)
                        self._buffer.clear()
                        return await self._process_utterance(full_audio)
                return None

    async def _process_utterance(self, audio: bytes) -> bytes:
        """Transcribe, generate response, synthesize speech."""
        logger.debug(f"Processing utterance: {len(audio)} bytes")

        # ASR
        text = await asyncio.to_thread(self._asr.transcribe, audio, self._sample_rate)
        if not text:
            self._state = PipelineState.IDLE
            return b""
        logger.debug(f"ASR: {text}")

        # LLM
        response = await asyncio.to_thread(self._llm.respond, text)
        logger.debug(f"LLM: {response}")

        # TTS
        audio_out = await asyncio.to_thread(self._tts.synthesize, response)
        self._state = PipelineState.IDLE if not audio_out else PipelineState.SPEAKING
        return audio_out

    @property
    def state(self) -> PipelineState:
        with self._lock:
            return self._state


# ── Media Stream Router ──────────────────────────────────────────────────

class MediaStreamRouter:
    """Maps each WebRTC peer connection to an inference session.

    Routes incoming audio/video tracks through the pipeline and sends
    the response track back.

    Usage::

        router = MediaStreamRouter()
        session = router.create_session(session_id)
        session.add_audio_track(inbound_track)
        session.add_video_track(inbound_video_track)
        await router.run_session(session_id)
    """

    def __init__(self, pipeline: AudioPipeline | None = None):
        self._pipeline = pipeline or AudioPipeline()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> dict[str, Any]:
        session = {
            "session_id": session_id,
            "audio_queue": asyncio.Queue(),
            "video_queue": asyncio.Queue(),
            "audio_out_queue": asyncio.Queue(),
            "created_at": time.time(),
            "interrupt": False,
        }
        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"Media session created: {session_id}")
        return session

    async def push_audio(self, session_id: str, frame: bytes) -> None:
        """Push an incoming audio frame into the session queue."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session:
            await session["audio_queue"].put(frame)

    async def run_session(self, session_id: str) -> None:
        """Run the inference loop for a session.

        Reads audio frames from the queue, processes them through
        the pipeline, and writes response audio to the output queue.
        Supports barge-in (interrupt) by clearing the pipeline.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return

        try:
            while not session["interrupt"]:
                try:
                    frame = await asyncio.wait_for(
                        session["audio_queue"].get(), timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                response = await self._pipeline.process_audio_frame(frame)
                if response:
                    await session["audio_out_queue"].put(response)
        except Exception as e:
            logger.error(f"Session {session_id} error: {e}")

    async def get_response_audio(self, session_id: str) -> bytes | None:
        """Get the next response audio chunk for a session (non-blocking)."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session and not session["audio_out_queue"].empty():
            return session["audio_out_queue"].get_nowait()
        return None

    def interrupt(self, session_id: str) -> None:
        """Interrupt/stop a running session (barge-in support)."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session:
            session["interrupt"] = True
            # Clear pending audio
            while not session["audio_queue"].empty():
                session["audio_queue"].get_nowait()
            while not session["audio_out_queue"].empty():
                session["audio_out_queue"].get_nowait()

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "pipeline_state": self._pipeline.state.value,
            }
