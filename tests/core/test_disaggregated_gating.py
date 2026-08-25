"""C12 regression: disaggregated P&D scheduler must be fenced behind an
explicit opt-in.

Root cause: the scheduler was registered unconditionally at coordinator
startup, and its KV transfer is fake — the gRPC fallback pickles the payload,
logs an estimate, sends nothing over the network and reports success;
``allocate_decode_blocks`` logs and returns True without reserving anything.

Fix: registration is gated behind ``DISTLLM_DISAGGREGATED_ENABLED=1``
(default OFF) with a loud WARNING when opted in; the stub paths themselves
warn that no transfer/allocation happens.
"""

from __future__ import annotations

from loguru import logger

from distllm.core.advanced_scheduling.disaggregated import (
    DisaggregatedBatchScheduler,
)
from distllm.core.coordinator import Coordinator
from distllm.core.coordinator_config import CoordinatorConfig


class _WarningSink:
    """Loguru sink that captures WARNING+ messages."""

    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message):
        if message.record["level"].name in ("WARNING", "ERROR", "CRITICAL"):
            self.messages.append(message.record["message"])


class TestRegistrationGate:
    """The subsystem must not register unless explicitly enabled."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_DISAGGREGATED_ENABLED", raising=False)
        coord = Coordinator(config=CoordinatorConfig(model_name="test"))

        coord._maybe_start_disaggregated()

        assert coord._disaggregated_scheduler is None
        assert coord._subsystem_health["disaggregated_scheduler"]["status"] == "disabled"

    def test_opt_in_registers_subsystem(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_DISAGGREGATED_ENABLED", "1")
        coord = Coordinator(config=CoordinatorConfig(model_name="test"))

        calls: list[tuple] = []
        monkeypatch.setattr(coord, "_start_subsystem", lambda *a, **k: calls.append(a))

        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            coord._maybe_start_disaggregated()
        finally:
            logger.remove(handler_id)

        assert len(calls) == 1
        # Registered under the same name/attr the request path reads.
        name, module, cls, attr = calls[0]
        assert name == "disaggregated_scheduler"
        assert "disaggregated" in module
        assert attr == "_disaggregated_scheduler"
        # Loud warning that KV transfer is not implemented.
        assert any(
            "NOT IMPLEMENTED" in m.upper() for m in sink.messages
        ), "opting in must emit a loud WARNING that KV transfer is a stub"

    def test_other_values_do_not_enable(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_DISAGGREGATED_ENABLED", "true")
        coord = Coordinator(config=CoordinatorConfig(model_name="test"))

        calls: list[tuple] = []
        monkeypatch.setattr(coord, "_start_subsystem", lambda *a, **k: calls.append(a))
        coord._maybe_start_disaggregated()

        assert calls == []  # only the literal "1" opts in


class TestStubPathsWarn:
    """The fenced stubs must warn instead of pretending to work silently."""

    def test_stream_kv_cache_fallback_warns_and_sends_nothing(self):
        sched = DisaggregatedBatchScheduler()
        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            ok, _ms = sched.stream_kv_cache(
                request_id="req-1",
                prefill_node="gpu-0",
                decode_node="gpu-2",
                kv_data={"k": b"payload"},
                transport=None,  # forces the gRPC fallback (the stub)
            )
        finally:
            logger.remove(handler_id)

        assert ok is True  # semantics unchanged (fence, not behavior change)
        assert any(
            "NOT TRANSFERRED" in m.upper() for m in sink.messages
        ), f"stub must warn loudly, got: {sink.messages}"

    def test_allocate_decode_blocks_warns(self):
        sched = DisaggregatedBatchScheduler()
        sink = _WarningSink()
        handler_id = logger.add(sink, level="WARNING")
        try:
            result = sched.allocate_decode_blocks("req-1", 1024, "gpu-2")
        finally:
            logger.remove(handler_id)

        assert result is True
        assert any(
            "NOT ALLOCATED" in m.upper() for m in sink.messages
        ), f"stub must warn loudly, got: {sink.messages}"
