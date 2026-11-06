"""Re-run the docs-stale-1 classifier: staleness of docs/*.md (first 45 alphabetically)."""
import pathlib
from pathlib import Path

RULES = """Work in D:\\distributed-llm\\.claude\\worktrees\\bubbly-pondering-whisper (cwd). READ-ONLY: do not edit/delete anything.
Evidence rule: every verdict cites file:line or a grep result you actually ran. No guessing; mark REVIEW when uncertain.
Check references before calling anything junk: grep the repo for the path/name. If CI, Makefile, pyproject, docs, or code references it, it is KEEP or FIX, never DELETE.
Return structured JSON per schema: items[] with path/verdict/reason/evidence/risk + summary."""

LENS = """Scope: docs/ root-level + docs/*.md staleness (first 45 files alphabetically: ACADEMIC_PARTNERSHIPS..api.md). For each doc: dead links (verify target exists), stale paths/commands that contradict current code (spot-check 2-3 commands per doc against real files), personal data (IPs, emails), TOC accuracy, duplicated content vs another doc. Verdict per doc: KEEP / FIX (what exactly) / DELETE (only if fully superseded — name the superseding doc with evidence). Read each file."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"},
            "verdict": {"type": "string", "enum": ["DELETE", "KEEP", "FIX", "REVIEW"]},
            "reason": {"type": "string"},
            "evidence": {"type": "string"},
            "risk": {"type": "string"},
        }, "required": ["path", "verdict", "reason", "evidence"]}},
        "summary": {"type": "string"},
    },
    "required": ["items", "summary"],
}

r = agent(RULES + "\n\n" + LENS, {"label": "docs-stale-1-retry", "phase": "Classify", "schema": SCHEMA})
pathlib.Path("docs_stale_1_result.json").write_text(
    __import__("json").dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
return r
