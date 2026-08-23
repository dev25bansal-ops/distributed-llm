"""Regression test for the C1 release-blocker.

``Coordinator.__init__`` used to reference ``self._subsystem_mgr`` (when wiring
the StragglerDetector callback) before the attribute was assigned, so a bare
``Coordinator(...)`` raised ``AttributeError`` and the platform could not
start.  Tests only passed because ``tests/conftest.py`` injected a class-level
``MagicMock``.  This test constructs a real Coordinator without any mock and
asserts the real ``SubsystemManager`` is wired.
"""

import pytest

from distllm.core.coordinator import Coordinator
from distllm.core.coordinator_subsystem import SubsystemManager


class TestCoordinatorStartupOrder:
    def test_constructor_wires_real_subsystem_manager(self):
        """Coordinator must construct and wire a real SubsystemManager."""
        coord = Coordinator(
            model_name="test-model",
            dtype="float32",
            max_batch_size=1,
            max_tokens_per_batch=4096,
        )
        try:
            # The real instance manager, not a MagicMock injected by conftest.
            assert isinstance(coord._subsystem_mgr, SubsystemManager)
            assert coord._subsystem_mgr.coordinator is coord
        finally:
            if hasattr(coord, "close"):
                coord.close()

    def test_straggler_callback_bound_to_subsystem_manager(self):
        """The StragglerDetector callback must be the manager's method."""
        coord = Coordinator(
            model_name="test-model",
            dtype="float32",
            max_batch_size=1,
            max_tokens_per_batch=4096,
        )
        try:
            detector = coord._straggler_detector
            assert detector is not None
            # The callback was captured at construction time and is the
            # bound method on the real SubsystemManager instance.
            assert (
                detector._on_straggler == coord._subsystem_mgr._on_straggler_detected
            )
        finally:
            if hasattr(coord, "close"):
                coord.close()
