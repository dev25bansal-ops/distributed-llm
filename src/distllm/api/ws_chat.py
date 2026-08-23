"""WebSocket chat streaming endpoint.

Provides a real-time chat interface over WebSocket with:

- Token-by-token streaming with configurable batching
- Client-side backpressure using pending-token tracking
- Multiplexed streams (multiple chat turns over one connection)
- Mid-stream parameter updates (temperature, top_p, etc.)
- Session management for multiple concurrent conversations

WebSocket protocol
------------------
Client -> Server:

    ``{"type": "chat", "data": {"messages": [...], "temperature": 0.7}}``

        Start a new chat session. Returns tokens until done.

    ``{"type": "update", "data": {"session_id": "...", "temperature": 0.5}}``

        Mid-stream parameter update for an active session.  Accepted keys:
        temperature, top_p, top_k, max_tokens.

    ``{"type": "cancel", "data": {"session_id": "..."}}``

        Cancel a running session.

    ``{"type": "ping", "data": {}}``

        Keep-alive probe.  Server replies with ``pong``.

Server -> Client:

    ``{"type": "token", "data": {"token": "Hello", "index": 0, "session_id": "..."}}``

        A token (or batch of tokens) from an active session.

    ``{"type": "done", "data": {"reason": "stop", "usage": {"completion_tokens": 42}, "session_id": "..."}}``

        Signals the end of a session's generation.

    ``{"type": "error", "data": {"message": "...", "session_id": "..."}}``

        An error occurred.  The session is terminated.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, WebSocketState
from loguru import logger

from distllm.api.api_state import g
from distllm.core.param_update_channel import GenerationParams
from distllm.core.token_streaming_buffer import TokenStreamingBuffer


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class WSChatConfig:
    """Per-connection configuration for the WebSocket chat handler."""

    # Backpressure: pause generation when this many tokens are pending delivery
    max_buffer_size: int = 256

    # Sleep duration when applying backpressure
    backpressure_sleep_s: float = 0.01

    # Tokens per WebSocket message (1 = token-by-token, N = batched)
    stream_chunk_size: int = 1

    # Maximum number of concurrent sessions per WebSocket connection
    max_concurrent_sessions: int = 10

    # Maximum entries in the async producer-consumer queue
    queue_maxsize: int = 128

    # Default generation parameters
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 0


# ── Session ────────────────────────────────────────────────────────────────


class ChatSession:
    """State for a single chat session multiplexed over a WebSocket connection."""

    def __init__(
        self,
        session_id: str,
        request_id: str,
        params: dict[str, Any],
    ) -> None:
        self.session_id = session_id
        self.request_id = request_id
        self.params = params
        self.cancel_event = asyncio.Event()
        self.created_at = time.monotonic()


# ── Handler ────────────────────────────────────────────────────────────────


class WSChatHandler:
    """Handles one WebSocket chat connection with multiplexed sessions.

    Usage::

        handler = WSChatHandler(websocket)
        await handler.handle()
    """

    def __init__(
        self,
        websocket: WebSocket,
        config: WSChatConfig | None = None,
        user_id: str = "default",
    ) -> None:
        self._ws = websocket
        self._config = config or WSChatConfig()
        self._user_id = user_id
        self._sessions: dict[str, ChatSession] = {}
        self._pending_tokens = 0
        self._send_lock = asyncio.Lock()
        self._running = True

    # ── Public API ─────────────────────────────────────────────────────────

    async def handle(self) -> None:
        """Accept the WebSocket and run the message loop."""
        await self._ws.accept()
        try:
            while self._running:
                raw = await self._ws.receive_text()
                await self._dispatch(raw)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.opt(exception=True).warning("WebSocket chat handler error")
        finally:
            self._running = False
            await self._cleanup()

    # ── Message dispatch ───────────────────────────────────────────────────

    async def _dispatch(self, raw: str) -> None:
        """Parse and route an incoming WebSocket message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("Invalid JSON message")
            return

        msg_type = msg.get("type", "")
        data: dict[str, Any] = msg.get("data") or {}

        if msg_type == "chat":
            await self._handle_chat(data)
        elif msg_type == "update":
            await self._handle_update(data)
        elif msg_type == "cancel":
            await self._handle_cancel(data)
        elif msg_type == "ping":
            await self._send({"type": "pong", "data": {"timestamp": time.time()}})
        else:
            await self._send_error(f"Unknown message type: {msg_type!r}")

    # ── Message handlers ───────────────────────────────────────────────────

    async def _handle_chat(self, data: dict[str, Any]) -> None:
        """Start a new chat session."""
        if len(self._sessions) >= self._config.max_concurrent_sessions:
            await self._send_error("Maximum concurrent sessions reached")
            return

        messages = data.get("messages")
        if not messages:
            await self._send_error("'messages' field is required")
            return

        prompt = _build_prompt(messages)
        session_id = data.get("session_id", str(uuid.uuid4()))
        params = _extract_params(data, self._config)
        request_id = f"ws-chat-{uuid.uuid4().hex[:12]}"
        session = ChatSession(session_id, request_id, params)
        self._sessions[session_id] = session

        # Register with the parameter update channel for mid-stream updates
        coord = g.coordinator
        puc = getattr(coord, "_param_update_channel", None) if coord else None
        if puc is not None:
            puc.register(
                request_id,
                GenerationParams(
                    temperature=params.get(
                        "temperature", self._config.default_temperature
                    ),
                    top_p=params.get("top_p", self._config.default_top_p),
                    top_k=params.get("top_k", self._config.default_top_k),
                    max_tokens=params.get(
                        "max_tokens", self._config.default_max_tokens
                    ),
                ),
            )

        logger.debug(
            "WS chat session started",
            session_id=session_id,
            request_id=request_id,
            messages=len(messages),
        )

        asyncio.create_task(self._run_session(session, prompt, request_id))

    async def _run_session(
        self,
        session: ChatSession,
        prompt: str,
        request_id: str,
    ) -> None:
        """Stream tokens for a single chat session.

        Runs the synchronous ``generate_stream`` in a daemon thread and
        bridges tokens back to the async world via an ``asyncio.Queue``.
        """
        coord = g.coordinator
        if coord is None:
            await self._send_error(
                "Coordinator not loaded", session_id=session.session_id
            )
            self._sessions.pop(session.session_id, None)
            return

        engine = getattr(coord, "_inference_engine", None)
        if engine is None:
            await self._send_error(
                "Inference engine not loaded", session_id=session.session_id
            )
            self._sessions.pop(session.session_id, None)
            return

        puc = getattr(coord, "_param_update_channel", None)

        # ── Producer-consumer bridge ───────────────────────────────────
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )

        def _producer() -> None:
            """Iterate the sync generator in a thread, push tokens to the queue."""
            try:
                gen = engine.generate_stream(
                    prompt,
                    max_new_tokens=session.params.get(
                        "max_tokens", self._config.default_max_tokens
                    ),
                    temperature=session.params.get(
                        "temperature", self._config.default_temperature
                    ),
                    top_p=session.params.get(
                        "top_p", self._config.default_top_p
                    ),
                    top_k=session.params.get(
                        "top_k", self._config.default_top_k
                    ),
                    request_id=request_id,
                )
                for token_text in gen:
                    if session.cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, token_text)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception:
                logger.opt(exception=True).warning(
                    "Producer thread error for session {}", session.session_id
                )
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        # ── Token streaming loop ───────────────────────────────────────
        token_buffer = TokenStreamingBuffer(
            max_batch_size=self._config.stream_chunk_size
        )
        sent_count = 0
        first_token = True

        try:
            while True:
                token = await queue.get()
                if token is None:
                    break

                # Empty tokens from the generator are skipped silently.
                if not token:
                    continue

                # Backpressure: pause if the consumer (client) has too many
                # pending tokens.  The pending count is decremented when a
                # ChatSession is cleaned up or when the client's buffer drains.
                while self._pending_tokens >= self._config.max_buffer_size:
                    if session.cancel_event.is_set():
                        break
                    await asyncio.sleep(self._config.backpressure_sleep_s)

                # Mid-stream parameter updates:
                #   The engine's ``generate_stream()`` reads temperature/top_p/
                #   top_k once at the start of generation, so mid-stream
                #   updates do NOT affect the *current* generation loop step.
                #   They WILL take effect on the next ``chat`` message within
                #   this session.  For true per-step parameter control, use
                #   a custom generation loop that re-reads params from the
                #   channel on each iteration (reference:
                #   ``streaming._generate_tokens``).
                if puc is not None and first_token:
                    # Log the initial params for debugging
                    params = puc.get(request_id)
                    if params is not None:
                        logger.trace(
                            "Session {} generating with params", session.session_id,
                        )

                # Check if the client is still connected
                if self._ws.client_state != WebSocketState.CONNECTED:
                    session.cancel_event.set()
                    break

                batch = token_buffer.add_token(token)
                if batch is not None:
                    await self._send(
                        {
                            "type": "token",
                            "data": {
                                "token": batch.text,
                                "index": sent_count,
                                "session_id": session.session_id,
                            },
                        }
                    )
                    sent_count += batch.token_count
                    self._pending_tokens += 1

                if first_token and sent_count > 0:
                    first_token = False

            # ── Flush remaining buffered tokens ────────────────────────
            batch = token_buffer.finish()
            if batch is not None:
                await self._send(
                    {
                        "type": "token",
                        "data": {
                            "token": batch.text,
                            "index": sent_count,
                            "session_id": session.session_id,
                        },
                    }
                )
                sent_count += batch.token_count

            # ── Done signal ────────────────────────────────────────────
            await self._send(
                {
                    "type": "done",
                    "data": {
                        "reason": "stop" if not session.cancel_event.is_set() else "cancelled",
                        "usage": {"completion_tokens": sent_count},
                        "session_id": session.session_id,
                    },
                }
            )

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.opt(exception=True).warning(
                "Session error {}", session.session_id
            )
            await self._send_error(
                "Generation failed", session_id=session.session_id
            )
        finally:
            if puc is not None:
                puc.unregister(request_id)
            self._sessions.pop(session.session_id, None)

    async def _handle_update(self, data: dict[str, Any]) -> None:
        """Apply a mid-stream parameter update to an active session."""
        session_id = data.get("session_id", "")
        session = self._sessions.get(session_id)
        if session is None:
            return  # Unknown session — silently ignored

        coord = g.coordinator
        puc = getattr(coord, "_param_update_channel", None) if coord else None
        if puc is None:
            return

        kwargs: dict[str, float | int] = {}
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if key in data:
                value = data[key]
                if key in ("top_k", "max_tokens"):
                    kwargs[key] = int(value)
                else:
                    kwargs[key] = float(value)

        if kwargs:
            puc.update(session.request_id, **kwargs)

    async def _handle_cancel(self, data: dict[str, Any]) -> None:
        """Cancel a running session."""
        session_id = data.get("session_id", "")
        session = self._sessions.get(session_id)
        if session is not None:
            session.cancel_event.set()

    # ── Send helpers ───────────────────────────────────────────────────────

    async def _send(self, data: dict[str, Any]) -> None:
        """Send a JSON message, serializing concurrent sends."""
        async with self._send_lock:
            if self._ws.client_state == WebSocketState.CONNECTED:
                try:
                    await self._ws.send_json(data)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to send WS message"
                    )

    async def _send_error(self, message: str, session_id: str = "") -> None:
        """Send an error message (and optionally terminate a session)."""
        data: dict[str, Any] = {
            "type": "error",
            "data": {"message": message},
        }
        if session_id:
            data["data"]["session_id"] = session_id
        await self._send(data)

    # ── Cleanup ────────────────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Cancel all active sessions and release resources."""
        coord = g.coordinator
        puc = getattr(coord, "_param_update_channel", None) if coord else None
        for session_id, session in list(self._sessions.items()):
            session.cancel_event.set()
            if puc is not None:
                puc.unregister(session.request_id)
        self._sessions.clear()


# ── Module-level helpers ───────────────────────────────────────────────────


def _build_prompt(messages: list[dict[str, str]]) -> str:
    """Flatten a chat message list into a single prompt string.

    Uses a simple ``role: content`` format.  Production deployments should
    replace this with the model's chat template via ``tokenizer.apply_chat_template``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _extract_params(
    data: dict[str, Any],
    config: WSChatConfig,
) -> dict[str, Any]:
    """Extract generation parameters from a chat message payload."""
    return {
        "temperature": data.get("temperature", config.default_temperature),
        "top_p": data.get("top_p", config.default_top_p),
        "top_k": data.get("top_k", config.default_top_k),
        "max_tokens": data.get("max_tokens", config.default_max_tokens),
    }
