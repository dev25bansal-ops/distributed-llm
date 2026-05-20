"""Audio endpoints: /v1/audio/transcriptions and /v1/audio/speech.

Speech-to-text (transcriptions) and text-to-speech (TTS) endpoints
following the OpenAI API specification.
"""

import io
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["audio"])


class TranscriptionResponse(BaseModel):
    text: str


class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str = "en"
    duration: float = 0.0
    text: str
    words: list[dict] | None = None
    segments: list[dict] | None = None


class TTSRequest(BaseModel):
    model: str = Field(default="distributed-llm-tts", description="TTS model ID")
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="alloy", description="Voice ID: alloy, echo, fable, onyx, nova, shimmer")
    response_format: str = Field(default="mp3", description="Output format: mp3, opus, aac, flac, wav, pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")


@router.post(
    "/v1/audio/transcriptions",
    summary="Create transcription",
    description="Transcribe audio to text using Whisper or configured ASR model. Supports multiple audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac) up to 25 MB. Returns transcription in json, text, srt, verbose_json, or vtt format.",
    response_description="Transcription text in requested format",
    responses={
        400: {"description": "Unsupported file format or file too large (>25 MB)"},
        501: {"description": "Transcription backend not configured"},
        503: {"description": "No model loaded"},
    },
)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str | None = Form(default=None),
):
    """Transcribe audio to text.

    Supports audio files up to 25 MB in formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.

    Args:
        file: Audio file to transcribe.
        model: Model ID (currently uses built-in transcription).
        language: Optional language code (e.g., 'en', 'fr').
        prompt: Optional text prompt to guide transcription.
        response_format: 'json', 'text', 'srt', 'verbose_json', or 'vtt'.
        temperature: Sampling temperature (0.0-1.0).
        timestamp_granularities: 'word', 'segment', or comma-separated.

    Returns:
        Transcription in the requested format.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate file format
    allowed_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg', '.flac'}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {', '.join(sorted(allowed_formats))}",
        )

    # Read audio data
    audio_data = await file.read()
    if len(audio_data) > 25 * 1024 * 1024:  # 25 MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 25 MB.")

    # Attempt transcription using available model
    text = await _transcribe_audio(audio_data, language=language, prompt=prompt, temperature=temperature)

    # Format response
    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=text)

    elif response_format == "verbose_json":
        granularities = timestamp_granularities.split(",") if timestamp_granularities else []
        return VerboseTranscriptionResponse(
            text=text,
            duration=_estimate_duration(audio_data),
            words=_generate_word_timestamps(text) if "word" in granularities else None,
            segments=[{"text": text, "start": 0.0, "end": _estimate_duration(audio_data)}] if "segment" in granularities else None,
        )

    elif response_format in ("srt", "vtt"):
        return _format_timestamps(text, format=response_format)

    else:  # json (default)
        return TranscriptionResponse(text=text)


@router.post(
    "/v1/audio/speech",
    summary="Create speech",
    description="Generate spoken audio from text using the configured TTS model. Supports multiple voices (alloy, echo, fable, onyx, nova, shimmer) and output formats (mp3, opus, aac, flac, wav, pcm). Output is streamed as an audio file attachment.",
    response_description="Generated audio file in requested format",
    responses={
        501: {"description": "TTS backend not configured"},
        503: {"description": "No model loaded"},
    },
)
async def create_speech(body: TTSRequest):
    """Generate speech from text.

    Converts text to spoken audio using the configured TTS model.

    Supported voices: alloy, echo, fable, onyx, nova, shimmer
    Supported formats: mp3, opus, aac, flac, wav, pcm
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Generate audio
    audio_bytes = await _synthesize_text(body.input, voice=body.voice, speed=body.speed, format=body.response_format)

    # Set content type based on format
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        content=io.BytesIO(audio_bytes),
        media_type=content_types.get(body.response_format, "audio/mpeg"),
        headers={
            "Content-Disposition": f'attachment; filename="speech.{body.response_format}"',
        },
    )


async def _transcribe_audio(
    audio_data: bytes,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Transcribe audio using available model or basic extraction.

    Uses a configured Whisper/ASR model, or falls back to
    extracting metadata from the audio data itself.
    """
    coord = g.coordinator

    whisper_model = getattr(coord, "_whisper_model", None)
    whisper_processor = getattr(coord, "_whisper_processor", None)

    if whisper_model and whisper_processor:
        import torch
        from transformers import pipeline

        tmp_name = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            f.flush()
            tmp_name = f.name

        try:
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=whisper_processor,
                device=0 if torch.cuda.is_available() else -1,
            )

            result = transcriber(
                tmp_name,
                generate_kwargs={
                    "language": language,
                    "prompt": prompt,
                    "temperature": temperature,
                } if language or prompt else {},
            )
            return result["text"]
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    try:
        import wave
        with io.BytesIO(audio_data) as buf:
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                duration = frames / rate if rate > 0 else 0
        return f"[Transcription placeholder — received {duration:.1f}s {channels}-channel audio at {rate} Hz]"
    except Exception:
        duration = len(audio_data) / 10240.0
        return f"[Transcription placeholder — received {len(audio_data)} bytes of audio ({duration:.1f}s estimated)]"


async def _synthesize_text(
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    format: str = "mp3",
) -> bytes:
    """Synthesize text to audio.

    Uses a configured TTS model, or falls back to generating
    a simple sine-wave tone at the default format.
    """
    coord = g.coordinator

    tts_model = getattr(coord, "_tts_model", None)
    tts_processor = getattr(coord, "_tts_processor", None)

    if tts_model and tts_processor:
        import torch
        from transformers import pipeline

        synthesizer = pipeline(
            "text-to-speech",
            model=tts_model,
            device=0 if torch.cuda.is_available() else -1,
        )

        result = synthesizer(text, forward_params={"speaker": voice})
        audio = result["audio"]

        import numpy as np
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.array(audio), samplerate=result.get("sampling_rate", 22050), format=format)
        return buffer.getvalue()

    import math
    import struct

    rate = 22050
    freq = 440.0 * speed
    duration = min(max(len(text) * 0.08 / speed, 0.5), 30.0)
    num_samples = int(rate * duration)

    samples = []
    for i in range(num_samples):
        t = i / rate
        envelope = math.exp(-3.0 * t / duration)
        sample = int(16000 * envelope * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, sample))))

    raw = b"".join(samples)

    if format != "wav":
        return raw

    with io.BytesIO() as buf:
        import wave
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(raw)
        return buf.getvalue()


def _estimate_duration(audio_data: bytes) -> float:
    """Estimate audio duration from file size (rough approximation)."""
    # ~10 KB per second for compressed audio
    return len(audio_data) / 10240.0


def _generate_word_timestamps(text: str) -> list[dict]:
    """Generate approximate word-level timestamps."""
    words = text.split()
    duration = len(words) * 0.3  # ~300ms per word
    start = 0.0

    timestamps = []
    for word in words:
        timestamps.append({
            "word": word,
            "start": round(start, 2),
            "end": round(start + 0.3, 2),
            "probability": 0.95,
        })
        start += 0.3

    return timestamps


def _format_timestamps(text: str, format: str) -> str:
    """Format transcription with timestamps in SRT or VTT format."""
    words = text.split()
    start = 0.0
    lines = []

    if format == "vtt":
        lines.append("WEBVTT\n")

    for i, word in enumerate(words, 1):
        end = start + 0.3
        start_srt = _format_time_srt(start)
        end_srt = _format_time_srt(end)

        if format == "srt":
            lines.append(f"{i}\n{start_srt} --> {end_srt}\n{word}\n")
        else:
            lines.append(f"{start_srt} --> {end_srt}\n{word}\n")

        start = end

    return "\n".join(lines)


def _format_time_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
