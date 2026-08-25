"""Regression guard for the W3-C1 dead-code deletion campaign (batch 1).

Proves the 9 modules deleted in batch 1 stay deleted (no accidental
restoration via stale site-packages copies or egg-info), and that every wired
core subsystem still imports cleanly afterwards.

Run:  pytest tests/core/test_dead_code_batch1.py -v   (PYTHONPATH=src)
"""

from __future__ import annotations

import importlib
import importlib.util


DEAD_MODULES_BATCH_1: tuple[str, ...] = (
    "distllm.core.kv_cache_paged",
    "distllm.core.batch_builder",
    "distllm.core.sentinel_qos",
    "distllm.core.gaia_cache",
    "distllm.core.autoq",
    "distllm.core.aria_autoscaler",
    "distllm.core.pulse_performance_model",
    "distllm.core.priority_heap",
    "distllm.core.stats_collector",
)


def test_deleted_modules_stay_dead() -> None:
    """None of the batch-1 dead modules may be resolvable again."""
    import distllm.core  # noqa: F401  (resolve real package first)

    from pathlib import Path

    core_dir = Path(distllm.core.__file__).parent
    assert "src" in core_dir.parts, f"unexpected core location: {core_dir}"

    for dotted in DEAD_MODULES_BATCH_1:
        assert importlib.util.find_spec(dotted) is None, (
            f"{dotted} is importable again — it was proven dead and deleted "
            "(W3-C1 batch 1). Do not restore without re-wiring review."
        )
        leaf = dotted.rsplit(".", 1)[1]
        assert not (core_dir / f"{leaf}.py").exists(), f"{leaf}.py reappeared"


def test_wired_core_subsystems_import() -> None:
    """The live request path must import cleanly after the deletions."""
    for mod in (
        "distllm",
        "distllm.core",
        "distllm.core.coordinator",
        "distllm.core.inference_engine",
        "distllm.core.batch_scheduler",
        "distllm.core.request_pipeline",
        "distllm.core.kv_cache",
        "distllm.core.cache_manager",
        "distllm.core.speculative_decoder",
        "distllm.core.structured_output",
        "distllm.core.model_router",
    ):
        importlib.import_module(mod)
