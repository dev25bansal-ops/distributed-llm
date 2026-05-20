"""Token-by-token SSE streaming for chat completions.

Implements Server-Sent Events (SSE) streaming for real-time token generation.
Supports OpenAI-compatible chat completion chunk format, with pluggable
token generators and cancellation support.

Key features:
- Standard SSE protocol (text/event-stream)
- OpenAI-compatible chat completion chunk format
- Multiple tokens per event (configurable chunk size)
- Cancellation via asyncio.Event
- Per-token timing metadata
- Integration with tokenizer for detokenization
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncGenerator, AsyncIterator, Callable

from loguru import logger


@dataclass
class StreamChunk:
    """A single streaming chunk in OpenAI chat completions format."""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[dict[str, Any]] = field(default_factory=lambda: [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }
    ])
    usage: dict[str, Any] | None = None

    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"

    @staticmethod
    def data_done() -> str:
        return "data: [DONE]\n\n"


@dataclass
class StreamingConfig:
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1024
    stream_chunk_size: int = 1     # Tokens per event
    include_usage: bool = False
    echo_prompt: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)


class StreamingGenerator:
    """Token-by-token SSE streaming generator.

    Usage:
        generator = StreamingGenerator(tokenizer=tokenizer)

        async for chunk in generator.generate(
            prompt="Hello, world!",
            generate_fn=my_model.generate,
        ):
            print(chunk.to_sse(), end="")
    """

    def __init__(
        self,
        tokenizer: Any = None,
        config: StreamingConfig | None = None,
        decode_fn: Callable | None = None,
    ):
        self._tokenizer = tokenizer
        self._config = config or StreamingConfig()
        self._decode_fn = decode_fn or self._default_decode

    def _default_decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    async def generate(
        self,
        prompt: str,
        generate_fn: Callable,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming tokens from a prompt.

        Args:
            prompt: Input prompt string.
            generate_fn: Async generator that yields (token_id, is_done) tuples.
            cancel_event: Optional event to signal cancellation.

        Yields:
            StreamChunk for each token/event.
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        config = self._config

        # Echo prompt if configured
        if config.echo_prompt:
            yield StreamChunk(
                id=request_id,
                created=created,
                model=config.model,
                choices=[{"index": 0, "delta": {"role": "assistant", "content": prompt}, "finish_reason": None}],
            )

        token_buffer: list[int] = []
        tokens_generated = 0
        first_token_time = 0.0

        try:
            async for token_data in generate_fn(prompt):
                if cancel_event and cancel_event.is_set():
                    logger.debug(f"Streaming cancelled for request {request_id}")
                    break

                if isinstance(token_data, tuple):
                    token_id, is_done = token_data
                else:
                    token_id = token_data
                    is_done = False

                token_buffer.append(token_id)
                tokens_generated += 1

                if first_token_time == 0:
                    first_token_time = time.time()

                # Emit chunk when buffer reaches chunk_size or on final token
                if len(token_buffer) >= config.stream_chunk_size or is_done:
                    text = self._decode_fn(token_buffer)
                    finish_reason = "stop" if is_done else None

                    yield StreamChunk(
                        id=request_id,
                        created=created,
                        model=config.model,
                        choices=[{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": finish_reason,
                        }],
                    )
                    token_buffer = []

        except asyncio.CancelledError:
            logger.debug(f"Streaming cancelled via CancelledError for {request_id}")
        except Exception as e:
            logger.error(f"Streaming error for {request_id}: {e}")
            yield StreamChunk(
                id=request_id,
                created=created,
                model=config.model,
                choices=[{"index": 0, "delta": {}, "finish_reason": "error"}],
            )

        # Yield final chunk with usage if configured
        if config.include_usage:
            total_time = time.time() - created
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": tokens_generated,
                "total_tokens": len(prompt.split()) + tokens_generated,
                "time_ms": round(total_time * 1000, 2),
                "tokens_per_second": round(tokens_generated / max(total_time, 0.001), 1),
            }
            yield StreamChunk(
                id=request_id,
                created=created,
                model=config.model,
                choices=[{"index": 0, "delta": {}, "finish_reason": None}],
                usage=usage,
            )

        yield StreamChunk.data_done()

    async def generate_text_stream(
        self,
        prompt: str,
        generate_fn: Callable,
    ) -> AsyncGenerator[str, None]:
        """Simpler interface: yields just the text strings."""
        async for chunk in self.generate(prompt, generate_fn):
            # Skip the data: [DONE] marker (returned as str)
            if isinstance(chunk, str):
                continue
            text = chunk.choices[0]["delta"].get("content", "")
            if text:
                yield text

    @staticmethod
    def format_sse_response(response: str) -> str:
        return f"data: {response}\n\n"

    @staticmethod
    def format_done() -> str:
        return "data: [DONE]\n\n"
