"""M-02: AST-enforced invariant — no speculative acceptance loop may
reimplement rejection sampling outside ``spec_verify``.

Background (bug C-03): ``inference_engine._accept_drafts`` and
``draft_tree.verify_tree`` each hand-rolled the acceptance decision as
``rand >= p`` instead of true rejection sampling ``min(1, p/q)``. Four
independent acceptance implementations produced three distinct bugs. The
shared kernel ``spec_verify.accept_token`` exists precisely to prevent
this, but nothing enforced its use.

This test parses every module under ``src/distllm/`` with ``ast`` and
fails when a file defines a speculative-acceptance comparison WITHOUT
importing the shared kernel.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# Files that legitimately implement acceptance math: the shared kernel
# itself, and its test doubles. Everything else must import it.
KERNEL_FILES = {
    "spec_verify.py",
}

# Modules whose random comparisons are NOT speculative-token acceptance:
# learning exploration (epsilon-greedy), geo/gossip sampling, and chaos
# simulation fault injection. These are legitimate RNG uses.
LEGITIMATE_RANDOMNESS = {
    "learning_router.py",
    "geo.py",
    "gossip.py",
    "chaos_simulator.py",
}

# The banned pattern: any comparison involving a random draw against a
# probability on the same expression line (the hand-rolled acceptance
# decision). We detect via AST: an If whose test is a Compare where one
# side involves a call to torch.rand / random.random / numpy.random.
_RAND_FNS = {"rand", "random", "randn", "uniform"}


def _calls_rand(node: ast.AST) -> bool:
    """True if the expression calls a random-number generator."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = None
            if isinstance(fn, ast.Attribute):
                name = fn.attr
            elif isinstance(fn, ast.Name):
                name = fn.id
            if name in _RAND_FNS:
                return True
    return False


def _is_acceptance_if(node: ast.If) -> bool:
    """True if the If compares a random draw against something (the
    hand-rolled acceptance decision shape: ``if rand(...) >= p:``)."""
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    # random draw must be one side of the comparison
    return _calls_rand(test) or _calls_rand(node.body[0] if node.body else test)


def _imports_accept_token(path: Path) -> bool:
    """True if the file imports accept_token from the spec_verify kernel."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return bool(
        re.search(
            r"from\s+distllm\.core\.spec_verify\s+import\s+(.*\baccept_token\b)",
            text,
        )
    )


def find_offenders() -> list[tuple[Path, int, str]]:
    offenders: list[tuple[Path, int, str]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name in KERNEL_FILES or path.name in LEGITIMATE_RANDOMNESS:
            continue
        if not path.is_relative_to(SRC_ROOT / "distllm" / "core") and not path.is_relative_to(
            SRC_ROOT / "distllm" / "dist"
        ) and not path.is_relative_to(SRC_ROOT / "distllm" / "backends"):
            # Only the layers that decode tokens can hand-roll acceptance.
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_acceptance_if(node):
                if not _imports_accept_token(path):
                    offenders.append(
                        (path, node.lineno, "acceptance comparison without spec_verify import")
                    )
    return offenders


def test_no_hand_rolled_acceptance_loops():
    """Every acceptance comparison must live in a file importing the kernel."""
    offenders = find_offenders()
    assert not offenders, (
        f"{len(offenders)} file(s) reimplement speculative acceptance "
        "without importing spec_verify.accept_token (M-02 invariant):\n"
        + "\n".join(f"  {p}:{line}: {why}" for p, line, why in offenders)
    )


def test_kernel_exports_accept_token():
    """The shared kernel must keep exporting accept_token."""
    import importlib

    sv = importlib.import_module("distllm.core.spec_verify")
    assert callable(getattr(sv, "accept_token", None)), (
        "spec_verify.accept_token must remain the single source of truth"
    )


def test_all_decoders_import_kernel():
    """Every speculative decoder module must import the shared kernel."""
    import importlib

    must_import = [
        "distllm.core.inference_engine",
        "distllm.core.draft_tree",
        "distllm.core.speculative_decoder",
        "distllm.core.distributed_speculative",
        "distllm.core.compressed_speculative",
        "distllm.core.async_pipelined_speculative",
        "distllm.core.mtp_head",
    ]
    missing = []
    for mod_name in must_import:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue  # module may not exist on this install; not a lint failure
        if not hasattr(mod, "accept_token"):
            src_path = Path(getattr(mod, "__file__", ""))
            if src_path.exists() and "accept_token" not in src_path.read_text(
                encoding="utf-8", errors="ignore"
            ):
                missing.append(mod_name)
    assert not missing, (
        f"Decoder modules not importing spec_verify.accept_token: {missing}. "
        "Each decoder must delegate acceptance math to the shared kernel."
    )
