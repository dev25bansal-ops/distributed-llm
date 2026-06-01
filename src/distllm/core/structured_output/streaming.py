"""Streaming support for structured output generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PartialResult:
    """Result of feeding partial JSON to the parser."""

    text: str
    data: dict | None = None
    is_complete: bool = False
    errors: list[str] = field(default_factory=list)


class BufferedAccumulator:
    """Buffers text chunks and flushes when a minimum size is reached.

    Usage::

        acc = BufferedAccumulator(min_chars=50)
        chunks = acc.add("hello ")
        # chunks is [] (below threshold)
        chunks = acc.add("world! " * 10)
        # chunks contains the buffered text
    """

    def __init__(self, min_chars: int = 50) -> None:
        self._min_chars = min_chars
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        """Add text to the buffer.

        Returns a list of chunks if the buffer exceeds the threshold,
        otherwise returns an empty list.
        """
        self._buffer += text
        if len(self._buffer) >= self._min_chars:
            chunk = self._buffer
            self._buffer = ""
            return [chunk]
        return []

    def flush_all(self) -> str:
        """Flush all remaining buffered text."""
        text = self._buffer
        self._buffer = ""
        return text

    @property
    def has_content(self) -> bool:
        """Return True if the buffer has content."""
        return len(self._buffer) > 0

    @property
    def buffer_size(self) -> int:
        """Return the current buffer size."""
        return len(self._buffer)


class PartialJSONParser:
    """Parses partial JSON chunks and returns partial results.

    Usage::

        parser = PartialJSONParser()
        result = parser.feed('{"key": "val')
        assert not result.is_complete
        result = parser.feed('ue"}')
        assert result.is_complete
    """

    def __init__(self) -> None:
        self._accumulated = ""

    def feed(self, text: str) -> PartialResult:
        """Feed a text chunk and return the current parse state.

        Args:
            text: The text chunk to feed.

        Returns:
            PartialResult with the current state.
        """
        self._accumulated += text

        # Try to parse the accumulated text
        try:
            data = json.loads(self._accumulated)
            return PartialResult(
                text=self._accumulated,
                data=data,
                is_complete=True,
            )
        except json.JSONDecodeError:
            pass

        # Try to extract partial data
        data = self._try_extract_partial(self._accumulated)
        return PartialResult(
            text=self._accumulated,
            data=data,
            is_complete=False,
        )

    def _try_extract_partial(self, text: str) -> dict | None:
        """Try to extract partial data from incomplete JSON."""
        # Try closing unclosed structures
        for suffix in ['"', '"}', '"]', "}", "]"]:
            try:
                data = json.loads(text + suffix)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                continue
        return None

    def reset(self) -> None:
        """Reset the parser state."""
        self._accumulated = ""


class StructuredStreamHandler:
    """Handles streaming of structured output with validation.

    Usage::

        handler = StructuredStreamHandler(config)
        for chunk in stream:
            handler.process_chunk(chunk)
        result = handler.finalize()
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._accumulator = BufferedAccumulator(
            min_chars=getattr(config, "streaming_buffer_size", 50) if config else 50
        )
        self._parser = PartialJSONParser()
        self._chunks: list[str] = []
        self._is_complete = False

    def process_chunk(self, text: str) -> list[str]:
        """Process a text chunk.

        Returns a list of validated chunks to emit.
        """
        self._chunks.append(text)
        chunks = self._accumulator.add(text)
        if chunks:
            for chunk in chunks:
                result = self._parser.feed(chunk)
                if result.is_complete:
                    self._is_complete = True
        return chunks

    def finalize(self) -> PartialResult:
        """Finalize the stream and return the complete result."""
        remaining = self._accumulator.flush_all()
        if remaining:
            self._parser.feed(remaining)

        full_text = "".join(self._chunks)
        try:
            data = json.loads(full_text)
            return PartialResult(
                text=full_text,
                data=data,
                is_complete=True,
            )
        except json.JSONDecodeError:
            return PartialResult(
                text=full_text,
                data=None,
                is_complete=False,
                errors=["Invalid JSON in final output"],
            )
