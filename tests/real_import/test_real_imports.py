"""Real-import integration suite (audit finding C6).

These tests import the REAL ``distllm`` package graph — no
``tests/_import_helper`` fakes, no ``load_module()`` path-bypass.  They exist
to catch the class of regression the fake-import harness hides: broken
``__init__`` exports, missing-symbol imports, and circular chains in the
production package (e.g. the C1–C3 release-blockers shipped green because the
fake harness never imported the real package).

This directory deliberately has NO conftest that calls
``bootstrap_fake_packages()`` — the tests run from a clean interpreter with
``PYTHONPATH=src`` (see the CI job).

Run:  pytest tests/real_import/ -v
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# Production-critical entry points: the modules the platform cannot function
# without.  Value = a public symbol that must be exported by that module.
CRITICAL_IMPORTS: dict[str, str | None] = {
    "distllm": None,
    "distllm.api": None,
    "distllm.core.coordinator": "Coordinator",
    "distllm.core.inference_engine": "InferenceEngine",
    "distllm.core.batch_scheduler": "BatchScheduler",
    "distllm.core.kv_cache": "KVCache",
    "distllm.core.quantization_selector": "QuantizationSelector",
    "distllm.core.structured_output": "JSONSchemaConstraint",
    "distllm.core.async_pipelined_speculative": "PipelinedSpeculativeDecoder",
    "distllm.core.speculative_decoder": "SpeculativeDecoder",
    "distllm.core.model_router": "ModelRouter",
    "distllm.core.cost_tracker": "CostTracker",
    "distllm.core.usage_meter": "UsageMeter",
    "distllm.core.semantic_cache": "SemanticCache",
    "distllm.core.coordinator_subsystem": "SubsystemManager",
    "distllm.core.vectorstore.base": "VectorDBInterface",
    "distllm.dist.pipeline": None,
    "distllm.backends.registry": None,
}


def _load_checker():
    """Load tests/scripts/check_real_imports.py as a module (stdlib-only)."""
    path = REPO / "tests" / "scripts" / "check_real_imports.py"
    spec = importlib.util.spec_from_file_location("check_real_imports", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_real_imports"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_fake_package_bootstrap_present():
    """Guard: this suite must run against the REAL package, not the fakes."""
    import distllm.core as core

    assert (SRC / "distllm" / "core" / "__init__.py").resolve() == Path(core.__file__).resolve(), (
        "distllm.core resolved outside src/ — a fake/installed copy is shadowing it. "
        "Run with PYTHONPATH=src."
    )
    # The fake-bootstrap markers must not be installed for this module.
    assert "tests._import_helper" not in sys.modules


def test_top_level_packages_import():
    for mod in (
        "distllm",
        "distllm.core",
        "distllm.dist",
        "distllm.models",
        "distllm.config",
        "distllm.backends",
        "distllm.api",
        "distllm.core.structured_output",
    ):
        importlib.import_module(mod)


def test_production_critical_imports():
    """Every production-critical module must import and export its symbol."""
    for mod, sym in CRITICAL_IMPORTS.items():
        module = importlib.import_module(mod)
        if sym is not None:
            assert hasattr(module, sym), f"{mod} is missing public symbol {sym!r}"


@pytest.mark.timeout(600)
def test_every_core_module_imports():
    """Full sweep: every module under src/distllm/core must import cleanly."""
    checker = _load_checker()
    ok, broken = checker.sweep()
    assert not broken, (
        f"{len(broken)} core modules fail to import — this is exactly the class "
        f"of regression the fake-import harness hides:\n" + "\n".join(
            f"  {dotted}: {err}" for dotted, err in broken.items()
        )
    )
    assert len(ok) >= 280
