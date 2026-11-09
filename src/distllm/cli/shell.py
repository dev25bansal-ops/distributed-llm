"""Interactive shell for DistLLM — the `distllm shell` experience.

Modelled on opencode / Claude Code / Hermes: a persistent prompt with slash
commands, `@file` context attachment, saved sessions, and streaming output.

Deliberately built on ``prompt_toolkit`` (already required by the [cli] extra)
rather than adding a TUI framework: the package should not grow a dependency
for a nicer prompt, and prompt_toolkit already gives history, key bindings,
multi-line editing and completion.

Sessions are plain JSON under ``$DISTLLM_SHELL_DIR`` (default
``.distllm/sessions``) so a conversation can be resumed, inspected or diffed
with ordinary tools.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

DEFAULT_SESSION_DIR = Path(".distllm") / "sessions"

# Slash commands the shell understands, with the help text shown by /help.
COMMANDS: dict[str, str] = {
    "/help": "Show this help",
    "/model [name]": "Show or change the model",
    "/temp [value]": "Show or set temperature (0.0-2.0)",
    "/max [n]": "Show or set max tokens",
    "/system [text]": "Show or set the system prompt",
    "/file <path>": "Attach a file's contents to the next message",
    "/files": "List attached files",
    "/clear": "Clear conversation history",
    "/save [name]": "Save this session",
    "/load <name>": "Load a saved session",
    "/sessions": "List saved sessions",
    "/tokens": "Show token usage for this session",
    "/exit": "Leave the shell",
}

# Files larger than this are summarised rather than inlined, so a stray
# `@huge.log` does not blow the context window.
MAX_FILE_BYTES = 64 * 1024

# Matches @path tokens in a prompt. Deliberately conservative: a path, not an
# email address or a decorator. The optional drive prefix (C:\) is required on
# Windows, where a bare [\w./\\-]+ match silently skipped the whole token.
_AT_FILE = re.compile(r"(?<![\w@])@((?:[A-Za-z]:[\\/])?[\w./\\-]+)")


@dataclass
class Session:
    """One conversation, saveable and resumable."""

    name: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    attached: list[dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "attached": self.attached,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "created": self.created,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Session:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _session_dir() -> Path:
    path = Path(os.environ.get("DISTLLM_SHELL_DIR", DEFAULT_SESSION_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    """Turn a user-supplied session name into a filename.

    Dots are allowed inside a name but a name that is only dots (``..``) would
    be a traversal, and one containing ``..`` reads as a path. Strip them and
    fall back to a default rather than writing outside the sessions directory.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    # Collapse runs of dots and drop leading/trailing ones, which also removes
    # the ".." sequences that make a name look like a path.
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = cleaned.strip(".-")
    return cleaned or "session"


def save_session(session: Session, name: str = "") -> Path:
    """Write *session* to disk and return the path."""
    session.name = _safe_name(name or session.name or f"session-{int(session.created)}")
    path = _session_dir() / f"{session.name}.json"
    path.write_text(json.dumps(session.to_json(), indent=2), encoding="utf-8")
    return path


def load_session(name: str) -> Session:
    path = _session_dir() / f"{_safe_name(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved session named '{name}'")
    return Session.from_json(json.loads(path.read_text(encoding="utf-8")))


def list_sessions() -> list[tuple[str, float, int]]:
    """Return ``(name, created, message_count)`` for each saved session."""
    out: list[tuple[str, float, int]] = []
    for path in sorted(_session_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append((path.stem, float(data.get("created", 0)), len(data.get("messages") or [])))
    return out


def read_file_for_context(path_str: str) -> dict[str, str] | None:
    """Load a file to attach, truncating very large ones."""
    path = Path(path_str).expanduser()
    if not path.exists():
        console.print(f"[red]No such file:[/red] {path_str}")
        return None
    if path.is_dir():
        entries = sorted(p.name for p in path.iterdir())[:50]
        return {
            "path": str(path),
            "content": f"Directory {path}:\n" + "\n".join(f"  {e}" for e in entries),
        }
    try:
        raw = path.read_bytes()
    except OSError as exc:
        console.print(f"[red]Cannot read {path_str}:[/red] {exc}")
        return None

    truncated = len(raw) > MAX_FILE_BYTES
    text = raw[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[... truncated at {MAX_FILE_BYTES} bytes of {len(raw)} ...]"
        console.print(
            f"[yellow]{path.name} is {len(raw) // 1024} KB — attaching the first "
            f"{MAX_FILE_BYTES // 1024} KB[/yellow]"
        )
    return {"path": str(path), "content": text}


def expand_at_files(prompt: str, attached: list[dict[str, str]], console_: Console) -> str:
    """Replace ``@path`` tokens with the file's contents.

    Attachments collected via ``/file`` are prepended as a context block so the
    model sees them even when the prompt does not mention the path again.
    """
    for match in list(_AT_FILE.finditer(prompt)):
        token = match.group(1)
        loaded = read_file_for_context(token)
        if loaded:
            attached.append(loaded)
            prompt = prompt.replace(f"@{token}", f"[attached: {token}]", 1)

    if not attached:
        return prompt

    blocks = []
    for item in attached:
        blocks.append(f"--- {item['path']} ---\n{item['content']}")
    return "Context files:\n" + "\n\n".join(blocks) + "\n\n" + prompt


def build_messages(session: Session) -> list[dict[str, str]]:
    """Assemble the request messages for the next turn."""
    messages: list[dict[str, str]] = []
    if session.system_prompt:
        messages.append({"role": "system", "content": session.system_prompt})
    messages.extend(session.messages)
    return messages


def slash_command_names() -> list[str]:
    """Command names without arguments, for shell completion."""
    return sorted({c.split()[0] for c in COMMANDS})


def render_help() -> None:
    table = Table(title="Shell commands", show_header=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print(
        "\n[dim]Tip: @path attaches a file to your message. "
        "Ctrl+C cancels a line, Ctrl+D exits.[/dim]"
    )


def render_tokens(session: Session) -> None:
    total = session.prompt_tokens + session.completion_tokens
    console.print(
        f"[cyan]Prompt:[/cyan] {session.prompt_tokens}  "
        f"[cyan]Completion:[/cyan] {session.completion_tokens}  "
        f"[cyan]Total:[/cyan] {total}  "
        f"[cyan]Turns:[/cyan] {len(session.messages) // 2}"
    )


def render_sessions(sessions: list[tuple[str, float, int]]) -> None:
    if not sessions:
        console.print("[dim]No saved sessions yet. Use /save <name>.[/dim]")
        return
    table = Table(title="Saved sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Created")
    table.add_column("Messages", justify="right")
    for name, created, count in sessions:
        table.add_row(name, time.strftime("%Y-%m-%d %H:%M", time.localtime(created)), str(count))
    console.print(table)


def render_reply(text: str) -> None:
    """Render an assistant reply as Markdown, falling back to plain text."""
    if "\n" in text or any(m in text for m in ("```", "**", "# ", "- ")):
        console.print(Markdown(text))
    else:
        console.print(text)


def render_welcome(model: str, base_url: str, session: Session) -> None:
    console.print(
        Panel.fit(
            f"[bold]DistLLM shell[/bold]\n"
            f"model  [cyan]{model}[/cyan]\n"
            f"server [cyan]{base_url}[/cyan]\n\n"
            f"[dim]{len(COMMANDS)} commands — /help to list them, "
            f"@path to attach a file, Ctrl+D to exit[/dim]",
            border_style="cyan",
        )
    )
    if session.messages:
        console.print(
            f"[dim]Resuming session with {len(session.messages) // 2} previous turn(s).[/dim]"
        )


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------
async def stream_turn(session: Session, base_url: str, *, request_id: str = ""):
    """Stream one assistant reply as an async generator of text deltas.

    Uses the shared CLI auth headers: a bare client would 401 against any
    server started by `distllm cluster start`, which is the bug that made the
    older chat command unusable against a real cluster.
    """
    import httpx

    from distllm.cli.http_client import auth_headers

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": session.model,
        "messages": build_messages(session),
        "temperature": session.temperature,
        "max_tokens": session.max_tokens,
        "stream": True,
    }
    if request_id:
        body["request_id"] = request_id

    headers = {"Content-Type": "application/json", **auth_headers()}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream("POST", url, json=body, headers=headers) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")
            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                payload = raw_line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    session.prompt_tokens = usage.get("prompt_tokens", session.prompt_tokens)
                    session.completion_tokens = usage.get(
                        "completion_tokens", session.completion_tokens
                    )


def handle_slash(line: str, session: Session) -> bool:
    """Execute a slash command. Returns True when the shell should exit."""
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/exit":
        return True
    if cmd == "/help":
        render_help()
    elif cmd == "/model":
        if arg:
            session.model = arg
            console.print(f"[green]Model:[/green] {arg}")
        else:
            console.print(f"[cyan]Model:[/cyan] {session.model}")
    elif cmd == "/temp":
        if arg:
            try:
                session.temperature = max(0.0, min(2.0, float(arg)))
                console.print(f"[green]Temperature:[/green] {session.temperature}")
            except ValueError:
                console.print("[red]Temperature must be a number[/red]")
        else:
            console.print(f"[cyan]Temperature:[/cyan] {session.temperature}")
    elif cmd == "/max":
        if arg:
            try:
                session.max_tokens = max(1, int(arg))
                console.print(f"[green]Max tokens:[/green] {session.max_tokens}")
            except ValueError:
                console.print("[red]Max tokens must be an integer[/red]")
        else:
            console.print(f"[cyan]Max tokens:[/cyan] {session.max_tokens}")
    elif cmd == "/system":
        if arg:
            session.system_prompt = arg
            console.print("[green]System prompt set.[/green]")
        else:
            current = session.system_prompt or "[dim](none)[/dim]"
            console.print(f"[cyan]System prompt:[/cyan] {current}")
    elif cmd == "/file":
        loaded = read_file_for_context(arg) if arg else None
        if not arg:
            console.print("[dim]Usage: /file <path>[/dim]")
        elif loaded:
            session.attached.append(loaded)
            console.print(
                f"[green]Attached[/green] {loaded['path']} "
                f"({len(loaded['content'])} chars) — sent with your next message"
            )
    elif cmd == "/files":
        if not session.attached:
            console.print("[dim]Nothing attached. Use /file <path> or @path.[/dim]")
        for item in session.attached:
            console.print(f"  {item['path']} ({len(item['content'])} chars)")
    elif cmd == "/clear":
        session.messages.clear()
        session.attached.clear()
        console.print("[green]Conversation cleared.[/green]")
    elif cmd == "/save":
        path = save_session(session, arg)
        console.print(f"[green]Saved:[/green] {path}")
    elif cmd == "/load":
        if not arg:
            render_sessions(list_sessions())
        else:
            try:
                loaded_session = load_session(arg)
            except FileNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
            else:
                session.__dict__.update(loaded_session.__dict__)
                console.print(
                    f"[green]Loaded '{loaded_session.name}'[/green] — "
                    f"{len(session.messages) // 2} turn(s)"
                )
    elif cmd == "/sessions":
        render_sessions(list_sessions())
    elif cmd == "/tokens":
        render_tokens(session)
    else:
        console.print(f"[red]Unknown command:[/red] {cmd}")
        console.print("[dim]Type /help for the list.[/dim]")
    return False


def _make_prompt_session(history_path: Path, console_: Console) -> Any:
    """Build a prompt session, falling back when the terminal cannot support one.

    prompt_toolkit refuses to start under git-bash on Windows:

        Error: Found xterm-256color, while expecting a Windows console.
        Maybe try to run this program using "winpty" or run it in cmd.exe.

    because MSYS sets TERM=xterm-256color while the console is still a Windows
    one. Raising there would make `distllm shell` unusable in the environment a
    Windows developer is most likely to be in, so fall back to a plain input()
    prompt: no completion or multi-line editing, but every command, session and
    @file feature still works.
    """
    import os
    import sys

    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    is_msys = bool(os.environ.get("MSYSTEM")) or (
        sys.platform == "win32" and os.environ.get("TERM", "").startswith("xterm")
    )

    if is_msys:
        console_.print(
            "[dim]Simplified prompt (git-bash/MSYS terminal): no tab completion. "
            "Run in cmd.exe or Windows Terminal for the full experience.[/dim]"
        )
        return _PlainPrompt(history_path)

    from prompt_toolkit import PromptSession

    try:
        return PromptSession(
            history=FileHistory(str(history_path)),
            completer=WordCompleter(slash_command_names(), ignore_case=True),
            complete_while_typing=True,
        )
    except Exception as exc:  # terminal cannot host a full-screen prompt
        console_.print(f"[dim]Simplified prompt ({exc.__class__.__name__}).[/dim]")
        return _PlainPrompt(history_path)


class _PlainPrompt:
    """Minimal stand-in for ``PromptSession`` using ``input()``.

    Keeps arrow-key history by delegating to ``readline`` where it exists, and
    raises EOFError on Ctrl+D so the caller's exit path is unchanged.
    """

    def __init__(self, history_path: Path) -> None:
        self._history_path = history_path
        try:  # readline is stdlib on POSIX and pyreadline3 on Windows
            import readline

            try:
                readline.read_history_file(str(history_path))
            except OSError:
                pass
            self._readline = readline
        except ImportError:
            self._readline = None

    def prompt(self, message: str = "> ") -> str:
        line = input(message)
        if self._readline is not None:
            try:
                self._readline.write_history_file(str(self._history_path))
            except OSError:
                pass
        return line


def run_shell(
    *,
    model: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    session_name: str = "",
) -> int:
    """Interactive loop. Returns a process exit code."""
    import asyncio

    session = Session(
        name=session_name,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        import prompt_toolkit  # noqa: F401  (capability check only)
    except ImportError:
        console.print(
            "[red]The shell needs prompt_toolkit:[/red] "
            'pip install "distributed-llm[cli]"'
        )
        return 1

    history_path = _session_dir() / "history.txt"
    prompt_session = _make_prompt_session(history_path, console)

    render_welcome(model, base_url, session)

    while True:
        try:
            line = prompt_session.prompt("> ").strip()
        except KeyboardInterrupt:
            console.print("[dim]^C — /exit or Ctrl+D to leave[/dim]")
            continue
        except EOFError:
            break

        if not line:
            continue

        if line.startswith("/"):
            if handle_slash(line, session):
                break
            continue

        text = expand_at_files(line, session.attached, console)
        session.attached.clear()
        session.messages.append({"role": "user", "content": text})

        collected: list[str] = []

        async def _run() -> None:
            async for delta in stream_turn(session, base_url):
                collected.append(delta)
                console.print(delta, end="", markup=False, highlight=False)

        try:
            asyncio.run(_run())
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            session.messages.pop()
            continue
        except Exception as exc:  # surfaced, never swallowed
            console.print(f"\n[red]Error:[/red] {exc}")
            session.messages.pop()
            console.print(
                "[dim]Is the server running? Start one with "
                "[cyan]distllm cluster start --model <model>[/cyan][/dim]"
            )
            continue

        reply = "".join(collected)
        if reply:
            session.messages.append({"role": "assistant", "content": reply})

    # Save on the way out so an accidental Ctrl+D does not lose the work.
    if session.messages:
        try:
            path = save_session(session)
            console.print(f"[dim]Session saved to {path}[/dim]")
        except OSError:
            pass
    console.print("[dim]Goodbye.[/dim]")
    return 0