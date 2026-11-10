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
# "there is no `distllm X`" / "no such `distllm X`" -- a negation IMMEDIATELY
# before the command. A looser test (any negation word earlier on the line)
# silently skipped real instructions: a table row
#   | `Model not found` | Run `distllm model load <model-name>` first |
# was skipped because "not" appears in "not found", so a broken command there
# would never have been reported. Verified by auditing every skipped line.
NEGATION_BEFORE = re.compile(
    r"\b(?:there\s+is\s+no|no\s+such|is\s+no|not\s+a|never\s+a)\s+[`\"']?\s*$",
    re.IGNORECASE,
)
SKIP_WORDS = {"https", "distllm", "docs", "pip"}


def _is_negation_or_url(text: str, match_start: int) -> bool:
    """True only when a negation phrase sits immediately before the command.

    Deliberately narrow: the previous version skipped any line containing
    "no"/"not" anywhere, which suppressed real instructions such as
    "| `Model not found` | Run `distllm model load ...` |".
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    prefix = text[line_start:match_start]
    return bool(NEGATION_BEFORE.search(prefix))


def _check_join_ports(docs: list[Path]) -> list[str]:
    """`cluster join --port` is the COORDINATOR's port, so every worker joining
    the same coordinator must use the same value.

    A QUICKSTART example had Node 2 on 50050 and Node 3 on 50051 -- 50051 is a
    worker's own --listen-port, so the second worker could never have connected.
    A resolution check cannot catch this: both commands are real.
    """
    problems = []
    join = re.compile(r"distllm\s+cluster\s+join[^\n]*--port\s+(\d+)")
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        ports = set(join.findall(text))
        if len(ports) > 1:
            problems.append(
                f"{doc.name}: cluster join uses differing coordinator ports {sorted(ports)}; "
                "--port is the coordinator's port and must match across workers"
            )
    return problems


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
    port_problems = _check_join_ports([ROOT / d for d in DOCS if (ROOT / d).exists()])

    ok = True
    if missing:
        ok = False
        print("\nDOCUMENTED BUT NOT REAL:")
        for invocation, doc, why in missing:
            print(f"  distllm {invocation}  [{doc}]  {why}")
    if port_problems:
        ok = False
        print("\nDOCUMENTED BUT WRONG:")
        for problem in port_problems:
            print(f"  {problem}")
    if not ok:
        return 1
    print("all documented commands resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
