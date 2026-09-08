"""Regression: F-043 audio pipeline must not get stuck in SPEAKING.

The state machine once transitioned to SPEAKING after the first utterance's
TTS and never returned to LISTENING, so a second utterance was buffered but
never transcribed/answered.  Post-F-043 the pipeline returns to LISTENING
after TTS (SPEAKING is an explicit opt-in via mark_speaking() for barge-in
gating), and new speech re-arms listening from any state including SPEAKING.
"""

from __future__ import annotations

import time

import pytest

from distllm.core.media_pipeline import AudioPipeline, PipelineState


class _FakeVAD:
    def __init__(self):
        self._speech = False

    def set_speech(self, s: bool):
        self._speech = s

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self._speech


class _FakeASR:
    def transcribe(self, audio: bytes, sample_rate: int) -> str:
        return "hello there"


class _FakeLLM:
    def respond(self, text: str) -> str:
        return "hi back"


class _FakeTTS:
    def synthesize(self, text: str) -> bytes:
        return b"\x00\x01audio"


def _pipeline(silence_timeout_s: float = 0.1) -> tuple[AudioPipeline, _FakeVAD]:
    vad = _FakeVAD()
    p = AudioPipeline(vad=vad, asr=_FakeASR(), llm=_FakeLLM(), tts=_FakeTTS(),
                      silence_timeout_s=silence_timeout_s)
    return p, vad


async def _clap(p: AudioPipeline, vad: _FakeVAD, speak: bool, timeout: float = 0.12):
    vad.set_speech(speak)
    # Feed a frame; when speak ends, advance time beyond the silence timeout.
    out = await p.process_audio_frame(b"\x00" * 320)
    if not speak:
        p._last_speech_time = time.time() - timeout
        out = await p.process_audio_frame(b"\x00" * 320)
    return out


class TestAudioMultipleUtterances:
    def test_state_resumes_listening_after_speaking(self):
        import asyncio
        asyncio.run(self._run())

    async def _run(self):
        p, vad = _pipeline()
        # Utterance 1: speak, then silence -> processed, TTS answer returned.
        await _clap(p, vad, True)
        assert p.state == PipelineState.LISTENING
        out1 = await _clap(p, vad, False)
        assert out1 == b"\x00\x01audio"
        # F-043 fix: after TTS the pipeline returns to LISTENING (not
        # SPEAKING) so the silence-timeout path re-arms for the next
        # utterance. Callers that play the audio out loud hold SPEAKING
        # explicitly via mark_speaking().
        assert p.state == PipelineState.LISTENING, f"expected LISTENING, got {p.state}"

        # Utterance 2: must still be processed and answered — this is the
        # F-043 regression the test guards (pipeline used to get stuck after
        # the first utterance and never answer a second one).
        out2 = await _clap(p, vad, True)
        assert out2 is None  # speech frames are buffered, no answer yet
        assert p.state == PipelineState.LISTENING
        out3 = await _clap(p, vad, False)
        assert out3 == b"\x00\x01audio", "second utterance must be processed"
        assert p.state == PipelineState.LISTENING

        # Barge-in gate: while TTS playback is held in SPEAKING, new speech
        # must re-arm LISTENING (the original stuck-in-SPEAKING bug).
        p.mark_speaking()
        assert p.state == PipelineState.SPEAKING
        await _clap(p, vad, True)
        assert p.state == PipelineState.LISTENING, "must resume listening, got {p.state}"

    def test_speech_during_speaking_transitions_to_listening(self):
        import asyncio
        asyncio.run(self._run_speech_during())

    async def _run_speech_during(self):
        p, vad = _pipeline()
        # Manually drop to SPEAKING (post-response state).
        p._state = PipelineState.SPEAKING
        vad.set_speech(True)
        await p.process_audio_frame(b"\x00" * 320)
        assert p.state == PipelineState.LISTENING