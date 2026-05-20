"""Streaming structured output with partial JSON parsing.

Provides incremental parsing of JSON output during generation,
allowing clients to see partial results before the full output
is complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator



@dataclass
class PartialResult:
    """A partially parsed structured output chunk.

    Attributes:
        text: Raw text accumulated so far.
        data: Parsed JSON data (may be partial/incomplete).
        is_complete: Whether the output is fully formed.
        path: JSON pointer path to the current parsing position.
        error: Error message if parsing failed.
    """
    text: str = ""
    data: Any = None
    is_complete: bool = False
    path: str = ""
    error: str = ""


@dataclass
class StructuredStreamChunk:
    """A single chunk from the structured output stream.

    Attributes:
        token: The raw token text for this step.
        partial: Partial parsing result (if enabled).
        is_final: Whether this is the final chunk.
    """
    token: str = ""
    partial: PartialResult | None = None
    is_final: bool = False


class BufferedAccumulator:
    """Accumulates characters and flushes at boundaries.

    Groups incoming tokens into larger chunks for more meaningful
    partial parse attempts. Flushes on JSON structural boundaries
    (commas, colons, closing brackets) or after a minimum length.
    """

    def __init__(self, min_chars: int = 10):
        self._buffer = ""
        self._min_chars = min_chars

    def add(self, text: str) -> list[str]:
        """Add text and return any flushed chunks."""
        self._buffer += text
        if len(self._buffer) >= self._min_chars:
            return self._flush()
        return []

    def _flush(self) -> list[str]:
        """Flush buffer, splitting on structural boundaries."""
        if not self._buffer:
            return []

        result = []
        structural = {",", ":", "}", "]", "\n"}
        pos = 0
        for i, ch in enumerate(self._buffer):
            if ch in structural and i - pos >= self._min_chars // 2:
                chunk = self._buffer[pos:i+1]
                result.append(chunk)
                pos = i + 1

        remaining = self._buffer[pos:]
        self._buffer = remaining if remaining else ""
        return result

    def flush_all(self) -> str:
        remaining = self._buffer
        self._buffer = ""
        return remaining

    @property
    def has_content(self) -> bool:
        return len(self._buffer) > 0


class PartialJSONParser:
    """Parses JSON incrementally, extracting what's valid so far.

    Attempts to parse the full accumulated text as JSON at each step.
    On failure, attempts increasingly aggressive recovery strategies:
    1. Strip trailing incomplete tokens
    2. Close unclosed brackets/braces
    3. Remove trailing comma
    4. Complete partial string values
    """

    def __init__(self):
        self._buffer = ""
        self._last_valid: Any = None

    def feed(self, text: str) -> PartialResult:
        """Feed a new text chunk and return the parsing result.

        Args:
            text: New text to append.

        Returns:
            PartialResult with current parsing state.
        """
        self._buffer += text
        return self._parse()

    def reset(self) -> None:
        """Clear the parser state."""
        self._buffer = ""
        self._last_valid = None

    def _parse(self) -> PartialResult:
        """Attempt to parse the buffer as JSON."""
        text = self._buffer

        # Try direct parse first
        result = self._try_parse(text)
        if result is not None:
            self._last_valid = result
            return PartialResult(
                text=text,
                data=result,
                is_complete=True,
                path="",
            )

        # Try partial parse with recovery
        recovered, recovery_path = self._recover_parse(text)
        if recovered is not None:
            self._last_valid = recovered
            return PartialResult(
                text=text,
                data=recovered,
                is_complete=False,
                path=recovery_path,
            )

        # Return last valid state if available
        if self._last_valid is not None:
            return PartialResult(
                text=text,
                data=self._last_valid,
                is_complete=False,
                path="",
                error="No complete parse available",
            )

        # Cannot parse at all
        return PartialResult(
            text=text,
            data=None,
            is_complete=False,
            path="",
            error="Cannot parse as JSON",
        )

    def _try_parse(self, text: str) -> Any:
        """Try to parse text as JSON."""
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    def _recover_parse(self, text: str) -> tuple[Any, str]:
        """Attempt to recover partial JSON by closing structures.

        Returns:
            Tuple of (parsed_data, recovery_path) or (None, "").
        """
        stripped = text.strip()
        if not stripped:
            return None, ""

        # Strategy 1: try progressively shorter suffixes
        for cutoff in range(len(stripped), len(stripped) // 2, -1):
            candidate = stripped[:cutoff]
            try:
                result = json.loads(candidate)
                return result, f"truncated at {cutoff}"
            except json.JSONDecodeError:
                pass

        # Strategy 2: close unclosed braces/brackets
        closed = self._close_brackets(stripped)
        try:
            result = json.loads(closed)
            return result, "brackets closed"
        except json.JSONDecodeError:
            pass

        # Strategy 3: close brackets + strip trailing comma
        cleaned = self._remove_trailing_comma(closed)
        try:
            result = json.loads(cleaned)
            return result, "cleaned"
        except json.JSONDecodeError:
            pass

        # Strategy 4: try to complete partial string values
        completed = self._complete_string(cleaned)
        try:
            result = json.loads(completed)
            return result, "strings completed"
        except json.JSONDecodeError:
            pass

        return None, ""

    @staticmethod
    def _close_brackets(text: str) -> str:
        """Close unclosed brackets and braces."""
        stack = []
        in_string = False
        escaped = False

        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        closing = "".join("}" if b == "{" else "]" for b in reversed(stack))
        return text + closing

    @staticmethod
    def _remove_trailing_comma(text: str) -> str:
        """Remove trailing commas before closing brackets."""
        import re
        result = re.sub(r",\s*([}\]])", r"\1", text)
        return result

    @staticmethod
    def _complete_string(text: str) -> str:
        """Complete partial string values by adding closing quotes."""
        in_string = False
        escaped = False
        last_open = -1

        for i, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                if not in_string:
                    in_string = True
                    last_open = i
                else:
                    in_string = False

        if in_string:
            text = text + '"'

        return text


class StructuredStreamHandler:
    """Handles streaming structured output with partial parsing.

    Wraps a token generation stream and yields StructuredStreamChunks
    with partial JSON parsing at each step.

    Usage:
        handler = StructuredStreamHandler()
        async for chunk in handler.process(token_stream):
            if chunk.partial and chunk.partial.data:
                print(chunk.partial.data)
    """

    def __init__(
        self,
        enable_partial_parsing: bool = True,
        flush_complete_pairs: bool = True,
        min_chars: int = 10,
    ):
        self._enable_partial_parsing = enable_partial_parsing
        self._flush_complete_pairs = flush_complete_pairs
        self._parser = PartialJSONParser()
        self._accumulator = BufferedAccumulator(min_chars=min_chars)
        self._full_text = ""
        self._token_count = 0
        self._flushed_pairs: set[str] = set()

    async def process(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[StructuredStreamChunk]:
        """Process a token stream into structured chunks.

        Args:
            token_stream: Async iterator of token strings.

        Yields:
            StructuredStreamChunk for each step.
        """
        async for token in token_stream:
            self._full_text += token
            self._token_count += 1

            if self._enable_partial_parsing:
                partial = self._parser.feed(token)
                yield StructuredStreamChunk(
                    token=token,
                    partial=partial,
                    is_final=partial.is_complete,
                )
            else:
                yield StructuredStreamChunk(token=token)

        # Final flush
        final_result = self._parser.feed("")
        yield StructuredStreamChunk(
            token="",
            partial=final_result,
            is_final=True,
        )

    async def process_buffered(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[StructuredStreamChunk]:
        """Process with buffering for more meaningful parse attempts.

        Buffers tokens and flushes on structural boundaries (commas,
        colons, closing brackets) or when buffer reaches minimum size.
        """
        async for token in token_stream:
            self._full_text += token
            self._token_count += 1

            flushed = self._accumulator.add(token)
            for chunk in flushed:
                if self._enable_partial_parsing:
                    partial = self._parser.feed(chunk)
                    yield StructuredStreamChunk(
                        token=chunk,
                        partial=partial,
                        is_final=False,
                    )
                else:
                    yield StructuredStreamChunk(token=chunk)

        remaining = self._accumulator.flush_all()
        if remaining:
            if self._enable_partial_parsing:
                partial = self._parser.feed(remaining)
                yield StructuredStreamChunk(
                    token=remaining,
                    partial=partial,
                    is_final=True,
                )
            else:
                yield StructuredStreamChunk(token=remaining)

    @property
    def full_text(self) -> str:
        return self._full_text

    @property
    def token_count(self) -> int:
        return self._token_count

    def reset(self) -> None:
        self._full_text = ""
        self._token_count = 0
        self._parser.reset()
        self._accumulator = BufferedAccumulator()
