"""Token-by-token streaming chat with Rich Live display and prompt_toolkit.

Features
--------
* prompt_toolkit PromptSession with persistent history (``~/.distllm/chat_history``)
* Multi-line input: **Enter** inserts newline, **Meta+Enter** submits
* SSE streaming via httpx displayed token-by-token with ``rich.live.Live``
* Final response rendered as Markdown via ``rich.markdown.Markdown``
* Slash commands: ``/clear``, ``/model``, ``/temp``, ``/help``, ``/exit``
* Token and timing summary after each response
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# prompt_toolkit -- optional dependency, graceful fallback
# ---------------------------------------------------------------------------
HAS_PROMPT_TOOLKIT = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    PromptSession = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HISTORY_DIR = os.path.expanduser("~/.distllm")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "chat_history")
_DEFAULT_API_BASE = "http://localhost:8000"
_DEFAULT_MODEL = "distributed-llm"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 256

# ── SSE line-delimiter helpers ──────────────────────────────────────────
_SSE_DATA_PREFIX = "data: "
_SSE_DONE_MARKER = "[DONE]"


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
@dataclass
class ChatState:
    """Mutable state that persists across turns within one chat session."""

    model: str = _DEFAULT_MODEL
    temperature: float = _DEFAULT_TEMPERATURE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    api_base: str = _DEFAULT_API_BASE
    messages: list[dict[str, str]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------
def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single ``data: {...}`` SSE line, return the decoded JSON dict.

    Returns ``None`` for heartbeats (empty ``data:``), the done marker, or
    lines that are not ``data:`` prefixed.
    """
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    payload = line[len(_SSE_DATA_PREFIX) :].strip()
    if not payload or payload == _SSE_DONE_MARKER:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    """Extract the content delta from a chunk. Returns empty string on failure."""
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        if delta is None:
            return ""
        return delta.get("content", "") or ""
    except (IndexError, KeyError, TypeError):
        return ""


def _extract_finish_reason(chunk: dict[str, Any]) -> str | None:
    """Extract finish_reason from a SSE chunk, or None."""
    try:
        return chunk["choices"][0].get("finish_reason")
    except (IndexError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Slash command implementations
# ---------------------------------------------------------------------------
def _cmd_clear(state: ChatState, console: Console) -> None:
    """``/clear`` -- clear conversation history (keeps configuration)."""
    prev_count = len(state.messages)
    state.messages.clear()
    console.print(f"[dim]Conversation cleared ({prev_count} message(s) removed).[/dim]")


def _cmd_model(state: ChatState, args: list[str], console: Console) -> None:
    """``/model <name>`` -- switch model."""
    if not args:
        console.print(f"[yellow]Current model:[/yellow] {state.model}")
        console.print("[dim]Usage: /model <model-name>[/dim]")
        return
    state.model = args[0]
    console.print(f"[green]Model set to:[/green] {state.model}")


def _cmd_temp(state: ChatState, args: list[str], console: Console) -> None:
    """``/temp <value>`` -- set temperature (0.0-2.0)."""
    if not args:
        console.print(f"[yellow]Current temperature:[/yellow] {state.temperature}")
        console.print("[dim]Usage: /temp <0.0-2.0>[/dim]")
        return
    try:
        val = float(args[0])
        if val < 0.0 or val > 2.0:
            console.print("[red]Temperature must be between 0.0 and 2.0.[/red]")
            return
        state.temperature = val
        console.print(f"[green]Temperature set to:[/green] {state.temperature}")
    except ValueError:
        console.print(f"[red]Invalid temperature: {args[0]}[/red]")


def _cmd_help(state: ChatState, console: Console) -> None:
    """``/help`` -- show available slash commands."""
    table = Table(title="Slash Commands", show_header=False, border_style="dim")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/model [name]", "Show or change the model")
    table.add_row("/temp [value]", "Show or set temperature (0.0-2.0)")
    table.add_row("/help", "Show this help")
    table.add_row("/exit", "Exit the chat")
    console.print(table)


def _cmd_exit(state: ChatState) -> bool:
    """``/exit`` -- signal to stop the loop."""
    return True


def _handle_slash_command(line: str, state: ChatState, console: Console) -> bool | None:
    """Handle a slash command. Returns True if caller should exit."""
    parts = shlex.split(line)
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd == "/clear":
        _cmd_clear(state, console)
    elif cmd == "/model":
        _cmd_model(state, args, console)
    elif cmd == "/temp":
        _cmd_temp(state, args, console)
    elif cmd == "/help":
        _cmd_help(state, console)
    elif cmd == "/exit":
        return _cmd_exit(state)
    else:
        console.print(f"[red]Unknown command:[/red] {cmd}")
        console.print("[dim]Type /help for available commands.[/dim]")
    return None


# ---------------------------------------------------------------------------
# API streaming
# ---------------------------------------------------------------------------
async def _stream_chat(
    state: ChatState,
    *,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the chat completions SSE endpoint.

    Yields token delta strings as they arrive from the API, one per SSE
    ``data:`` chunk.  Raises ``httpx.HTTPError`` on non-streaming errors.
    """
    url = f"{state.api_base.rstrip('/')}/v1/chat/completions"
    body = {
        "model": state.model,
        "messages": state.messages,
        "temperature": state.temperature,
        "max_tokens": state.max_tokens,
        "stream": True,
    }
    if request_id:
        body["request_id"] = request_id

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                chunk = _parse_sse_line(raw_line)
                if chunk is None:
                    continue
                text = _extract_delta_text(chunk)
                if text:
                    yield text


async def _stream_chat_with_timing(
    state: ChatState,
    *,
    request_id: str,
) -> AsyncGenerator[tuple[str, float, int], None]:
    """Like ``_stream_chat`` but yields ``(token, elapsed_sec, token_index)``.

    This wrapper is used internally by the Live-display renderer.
    """
    start = time.monotonic()
    idx = 0
    async for token in _stream_chat(state, request_id=request_id):
        idx += 1
        yield token, time.monotonic() - start, idx


# ---------------------------------------------------------------------------
# Live-update rendering loop
# ---------------------------------------------------------------------------
async def _do_streaming_turn(
    state: ChatState,
    console: Console,
) -> str | None:
    """Send the last user message, stream tokens, render final Markdown.

    Returns the full assistant text, or ``None`` if the request failed.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # ── Stage 1: streaming accumulation + Live display ──────────────
    spinner = Spinner("dots", text="Streaming...")
    accumulated = ""
    last_render = ""
    ttft: float | None = None
    start_time = time.monotonic()
    token_count = 0

    with Live(spinner, console=console, refresh_per_second=15, transient=True) as live:
        try:
            async for token, elapsed, idx in _stream_chat_with_timing(
                state, request_id=request_id,
            ):
                token_count += 1
                if ttft is None:
                    ttft = elapsed
                accumulated += token

                # Only re-render when new content actually arrived
                if accumulated != last_render:
                    last_render = accumulated
                    # Build display: streaming text in a Panel with a spinner
                    display = Panel(
                        Text(accumulated, style="green"),
                        title="Assistant",
                        border_style="green",
                        subtitle=f"[dim]{token_count} token(s), {elapsed:.1f}s[/dim]",
                    )
                    live.update(display)
        except httpx.HTTPStatusError as e:
            live.stop()
            try:
                detail = e.response.text
            except Exception:
                detail = str(e)
            console.print(f"[red]API error ({e.response.status_code}):[/red] {detail}")
            return None
        except httpx.TimeoutException:
            live.stop()
            console.print("[red]Request timed out. Check server connectivity.[/red]")
            return None
        except httpx.ConnectError:
            live.stop()
            console.print(
                f"[red]Could not connect to {state.api_base}.[/red]\n"
                "  Ensure the API server is running (distllm system api)."
            )
            return None
        except httpx.HTTPError as e:
            live.stop()
            console.print(f"[red]HTTP error:[/red] {e}")
            return None

    # ── Stage 2: render final Markdown ──────────────────────────────
    elapsed = time.monotonic() - start_time

    if accumulated:
        md = Markdown(accumulated)
        console.print()
        console.print(Panel(md, title="Assistant", border_style="green"))

    # Timing summary
    if token_count > 0 and elapsed > 0:
        tps = token_count / elapsed
        ttft_str = f", TTFT: {ttft:.2f}s" if ttft is not None else ""
        console.print(
            f"\n[dim]{token_count} tokens in {elapsed:.1f}s "
            f"({tps:.1f} tok/s{ttft_str})[/dim]"
        )
    else:
        console.print("\n[dim]No tokens generated.[/dim]")

    return accumulated


# ---------------------------------------------------------------------------
# prompt_toolkit bindings: multi-line input
# ---------------------------------------------------------------------------
def _make_key_bindings() -> KeyBindings | None:
    """Create key bindings where **Meta+Enter** submits and **Enter** inserts
    a newline. Returns ``None`` if prompt_toolkit is unavailable."""
    if not HAS_PROMPT_TOOLKIT:
        return None

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: Any) -> None:
        """Insert newline on plain Enter."""
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _meta_enter(event: Any) -> None:
        """Submit on Meta+Enter (Escape then Enter)."""
        event.current_buffer.validate_and_handle()

    return kb


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def _ensure_history_dir() -> None:
    """Create ``~/.distllm/`` if it does not exist."""
    os.makedirs(_HISTORY_DIR, exist_ok=True)


def _get_prompt_session() -> PromptSession | None:
    """Return a configured ``PromptSession`` or ``None``."""
    if not HAS_PROMPT_TOOLKIT:
        return None
    _ensure_history_dir()
    kb = _make_key_bindings()
    return PromptSession(
        history=FileHistory(_HISTORY_FILE),
        key_bindings=kb,
        multiline=True,
        # Show a hint about Meta+Enter
        bottom_toolbar="  [ Meta+Enter to submit | Enter newline | /help for commands ]",
        style="",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_chat_stream(
    *,
    model: str = _DEFAULT_MODEL,
    host: str = "localhost",
    port: int = 8000,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    console: Console | None = None,
) -> None:
    """Run an interactive streaming chat session.

    Parameters
    ----------
    model:
        Model identifier sent in the API request.
    host:
        API server hostname.
    port:
        API server port.
    max_tokens:
        Maximum tokens to generate per response.
    temperature:
        Sampling temperature (0.0-2.0).
    console:
        A ``rich.console.Console`` instance. Created fresh if omitted.
    """
    if console is None:
        console = Console()

    api_base = f"http://{host}:{port}"

    state = ChatState(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
    )

    # ── Welcome ─────────────────────────────────────────────────────
    welcome = Panel.fit(
        "[bold blue]DistLLM Streaming Chat[/bold blue]\n"
        f"Model: [green]{state.model}[/green]\n"
        f"Server: [green]{state.api_base}[/green]\n\n"
        "[dim]Multi-line input: Enter = newline, Meta+Enter = submit[/dim]\n"
        "[dim]Type /help for commands, /exit to quit.[/dim]",
        border_style="blue",
    )
    console.print()
    console.print(welcome)
    console.print()

    # If prompt_toolkit is unavailable, show a one-time notice
    if not HAS_PROMPT_TOOLKIT:
        console.print(
            "[yellow]prompt_toolkit not installed.[/yellow]\n"
            "  Install with: [bold]pip install prompt_toolkit[/bold]\n"
            "  Falling back to plain input (single-line, no history).\n"
        )

    prompt_session = _get_prompt_session()

    # ── Main interaction loop ───────────────────────────────────────
    while True:
        # ── Read input ──────────────────────────────────────────────
        try:
            if prompt_session is not None:
                raw = prompt_session.prompt(
                    ">>> ",
                    style="class:prompt",
                )
            else:
                # Fallback: single-line input with a plain input()
                console.print("[bold]You:[/bold] ", end="")
                raw = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        line = raw.strip()

        # ── Skip empty ──────────────────────────────────────────────
        if not line:
            continue

        # ── Slash commands ──────────────────────────────────────────
        if line.startswith("/"):
            should_exit = _handle_slash_command(line, state, console)
            if should_exit:
                console.print("[yellow]Goodbye![/yellow]")
                break
            continue

        # ── Append user message ─────────────────────────────────────
        state.messages.append({"role": "user", "content": line})

        # ── Stream response ─────────────────────────────────────────
        assistant_text = asyncio.run(
            _do_streaming_turn(state, console),
        )

        if assistant_text is not None:
            state.messages.append({"role": "assistant", "content": assistant_text})
        else:
            # Pop the failed user message so the user can retry
            state.messages.pop()
