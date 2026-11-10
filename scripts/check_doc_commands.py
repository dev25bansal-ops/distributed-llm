"""Every CLI invocation the user-facing docs name must actually resolve.

The install line was wrong for months because nobody ran it. This checks the
same class of error one level down -- commands the docs tell a user to type --
by resolving each against the real typer app rather than grepping source.

Negations are skipped on purpose: a line saying "there is no `distllm cluster
restart`" is correct documentation *because* the command is absent, and must not
be reported as a defect. So are URL fragments like `distllm https://...`.

Run from the repo root:  python scripts/check_doc_commands.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOCS = [
    "README.md",
    "docs/QUICKSTART.md",
    "docs/self-hosted.md",
    "docs/TROUBLESHOOTING.md",
]

# `distllm <group> [<subcommand>]` -- two lowercase words at most.
CMD = re.compile(r"\bdistllm\s+((?:[a-z][a-z-]*)(?:\s+[a-z][a-z-]*)?)")
NEGATION = re.compile(r"\b(no|not|isn't|without)\b", re.IGNORECASE)
SKIP_WORDS = {"https", "distllm", "docs", "pip"}


def _is_negation_or_url(text: str, match_start: int) -> bool:
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_start)
    if line_end == -1:
        line_end = len(text)
    prefix = text[line_start:match_start]
    return bool(NEGATION.search(prefix))


def main() -> int:
    import typer.main

    from distllm.cli.main import app

    root = typer.main.get_command(app)
    groups = getattr(root, "commands", {})

    seen: dict[str, str] = {}
    for doc in DOCS:
        path = ROOT / doc
        if not path.exists():
            print(f"note: {doc} not found, skipped")
            continue
        text = path.read_text(encoding="utf-8")
        for m in CMD.finditer(text):
            head = m.group(1).split()[0]
            if head in SKIP_WORDS:
                continue
            if _is_negation_or_url(text, m.start()):
                continue
            seen.setdefault(m.group(1), doc)

    missing: list[tuple[str, str, str]] = []
    for invocation, doc in sorted(seen.items()):
        tokens = invocation.split()
        group = tokens[0]
        if group not in groups:
            missing.append((invocation, doc, f"no command group '{group}'"))
            continue
        if len(tokens) > 1:
            subs = getattr(groups[group], "commands", {})
            if tokens[1] not in subs:
                missing.append(
                    (invocation, doc, f"'{group}' has no '{tokens[1]}' (has {sorted(subs)})")
                )

    print(f"checked {len(seen)} documented invocations against the CLI")
    if missing:
        print("\nDOCUMENTED BUT NOT REAL:")
        for invocation, doc, why in missing:
            print(f"  distllm {invocation}  [{doc}]  {why}")
        return 1
    print("all documented commands resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
