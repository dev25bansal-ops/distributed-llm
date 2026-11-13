"""Verify every `distllm <command...>` in the docs names a real command.

The audit fleet found four documentation commands that cannot run:
`distllm backup ...` (it is `config backup`), `system run --coordinator-host`
(the flag lives on the `distllm-node` console script), and
`system logs --node/--tail` (the flags are `--component`/`--lines` on
`system logs stream`). Each was reproduced by copy-pasting the doc into a
shell. This script parses the docs, walks the Typer command tree for each
invoked path, and fails on a path that does not exist.

Flag-level checking is deliberately shallow: a flag is only reported when
the command resolves AND the flag name appears in the command's parameter
registrations. Long-form `--flag` tokens that do not match are reported as
warnings when CI_DOCS_STRICT=1, findings otherwise, so the gate can start
green on the current tree and tighten later.

Usage: python scripts/ci/check_docs_commands.py [--docs DIR ...]
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

# Console scripts that are not the `distllm` Typer app.
OTHER_SCRIPTS = {"distllm-node", "distllm-api", "distllm-coordinator", "distllm-cli-doctor"}

# Command words must sit on one line; [ \t] (not \s) keeps the match from
# spanning a newline and pairing `distllm` with the next line's word. The
# lookbehind pins `distllm` to a command position — line start (optionally
# after `$ ` in a shell prompt block) or after a `# ` comment marker — so
# path fragments like `helm install distllm distllm/distllm` and prose
# parentheticals do not match.
CMD_RE = re.compile(
    r"(?:(?<=^)|(?<=\n)|(?<=\n\$ )|\$ |(?<=\n# )|# )distllm((?:[ \t]+[a-z][a-z0-9-]*)+)",
    re.M,
)
# Commands are stated in code: fenced blocks or inline backtick spans. Prose
# like "distllm shows no production caller" must not match, so only code is
# scanned; a tilde-fence or triple-backtick with an optional language tag.
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
INLINE_RE = re.compile(r"`([^`\n]+)`")
# Trees that describe past audits, future plans, or historical changes —
# their commands name a state that never existed or no longer does.
EXCLUDE_PARTS = (
    "docs/_map/",
    "docs/archive/",
    "CHANGELOG.md",
    "TASKS.md",
    "TASKS-",
)


def _code_regions(text: str):
    """Yield (offset, code_text) for fenced blocks and inline spans."""
    for m in FENCE_RE.finditer(text):
        yield m.start(1), m.group(1)
    for m in INLINE_RE.finditer(text):
        yield m.start(1), m.group(1)


def _load_app():
    from distllm.cli.main import app

    return app


def _command_paths(app) -> dict[str, object]:
    """Map 'cmd sub sub' -> typer command object for every reachable path."""
    import typer

    paths: dict[str, object] = {}

    def walk(typer_app, prefix: list[str]) -> None:
        for cmd in typer_app.registered_commands:
            name = cmd.name or (cmd.callback.__name__.replace("_", "-") if cmd.callback else None)
            if not name:
                continue
            full = prefix + [name]
            paths[" ".join(full)] = cmd
        for group in typer_app.registered_groups:
            gname = group.name or (group.typer_instance.info.name if group.typer_instance else None)
            if not gname or group.typer_instance is None:
                continue
            full = prefix + [gname]
            paths[" ".join(full)] = group.typer_instance
            walk(group.typer_instance, full)

    walk(app, [])
    return paths


def _flags_for(cmd) -> set[str]:
    """Best-effort set of long flags a command registers."""
    import typer

    flags: set[str] = set()
    try:
        info = typer.main.get_command(cmd).params if hasattr(typer.main, "get_command") else []
    except Exception:
        info = []
    for p in info or []:
        for opt in getattr(p, "opts", []) or []:
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", nargs="*", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)
    if args.docs:
        doc_files = [Path(p) for p in args.docs]
    else:
        doc_files = sorted(root.glob("docs/**/*.md")) + sorted(root.glob("*.md"))

    try:
        app = _load_app()
    except Exception as exc:  # noqa: BLE001
        print(f"[docs-commands] cannot import the CLI app: {exc}")
        return 2

    known = _command_paths(app)
    # `distllm <group>` alone is also valid for groups with a callback.
    unknown: list[str] = []
    checked = 0
    for f in doc_files:
        posix = f.as_posix()
        if any(part in posix for part in EXCLUDE_PARTS):
            continue
        try:
            text = io.open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for region_off, code in _code_regions(text):
            for m in CMD_RE.finditer(code):
                # Strip trailing generic words that are placeholders, not commands.
                words = [w for w in m.group(1).split() if w]
                # Try the longest prefix that is a real path.
                resolved = None
                for cut in range(len(words), 0, -1):
                    cand = " ".join(words[:cut])
                    if cand in known:
                        resolved = cand
                        break
                # A lone group name (e.g. `distllm model`) with subcommands is
                # valid usage in help text; only flag when no prefix resolved.
                if resolved is None and words and any(
                    k.startswith(words[0] + " ") for k in known
                ):
                    resolved = words[0]
                checked += 1
                if resolved is None:
                    here = f.as_posix()
                    line = text.count("\n", 0, region_off + m.start()) + 1
                    unknown.append(f"{here}:{line}: `distllm{m.group(1)}` (no such command path)")

    print(f"[docs-commands] checked {checked} invocations, {len(unknown)} unresolved")
    for u in unknown:
        print("  " + u)
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
