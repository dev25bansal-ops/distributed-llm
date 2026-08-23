"""Regression tests for audit finding F-052.

PipelineTransportController.send() gates against the congestion window and
buffers the excess in ``_send_buf`` — but nothing ever drained the buffer,
so every byte beyond the current cwnd (~14.6 KB initially) was silently
dropped, truncating large tensor transfers.

The fix drains ``_send_buf`` from ``on_ack()`` while the congestion window
allows. These tests pin that behaviour:

1. A payload larger than the window is only partially sent immediately;
   the remainder sits in ``_send_buf`` (no silent drop *into the void*).
2. Repeated ``on_ack()`` calls drain the buffer completely, and the
   concatenation of everything put on the wire equals the original input.
3. Payloads smaller than the window are unaffected (single-shot send).
"""

from __future__ import annotations

import struct

from distllm.dist.pipeline.bandwidth_controller import (
    PipelineTransportController,
)

PROBE_MAGIC = struct.pack("!I", 0xDEADBEEF)


class LoopbackWire:
    """Collects every frame the controller puts on the wire."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def send(self, data: bytes) -> None:
        if data:
            self.frames.append(data)

    def user_bytes(self) -> bytes:
        """Concatenation of all non-bandwidth-probe frames."""
        return b"".join(f for f in self.frames if not f.startswith(PROBE_MAGIC))

    def probe_count(self) -> int:
        return sum(1 for f in self.frames if f.startswith(PROBE_MAGIC))


def make_controller(wire: LoopbackWire) -> PipelineTransportController:
    return PipelineTransportController(send_fn=wire.send)


def drain(controller: PipelineTransportController, max_acks: int = 500) -> None:
    """Pump ACKs until the send buffer is empty (or the cap trips)."""
    for _ in range(max_acks):
        if controller.stats()["pipeline_bytes"] == 0:
            return
        controller.on_ack()
    raise AssertionError("send buffer did not drain within ACK budget")


# ---------------------------------------------------------------------------
# F-052 regression
# ---------------------------------------------------------------------------


class TestCongestionWindowBuffering:
    def test_large_payload_is_buffered_not_dropped(self) -> None:
        """Bytes beyond the cwnd must be buffered, never vanish."""
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        payload = b"A" * 100_000
        ctrl.send(payload)

        # Initial cwnd is 10 * 1460 = 14600 bytes: only the window may be
        # on the wire so far, the rest must be sitting in the buffer.
        assert len(wire.user_bytes()) == 14_600
        assert ctrl.stats()["pipeline_bytes"] == 100_000 - 14_600

    def test_full_transfer_roundtrip_after_acks(self) -> None:
        """After enough ACKs, the wire carries the entire payload intact."""
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        payload = bytes(range(256)) * 4096  # 1 MiB, per finding recommendation
        ctrl.send(payload)
        drain(ctrl)

        assert wire.user_bytes() == payload
        assert ctrl.stats()["pipeline_bytes"] == 0

    def test_each_ack_flushes_at_most_one_window(self) -> None:
        """A single ACK must not blast the whole buffer out at once."""
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        payload = b"B" * 100_000
        ctrl.send(payload)
        before = len(wire.user_bytes())

        ctrl.on_ack()  # slow start: cwnd += MSS (1460)

        flushed = len(wire.user_bytes()) - before
        assert flushed == ctrl._congestion.get_window()
        assert ctrl.stats()["pipeline_bytes"] == len(payload) - before - flushed

    def test_small_payload_sent_in_full_without_buffering(self) -> None:
        """Payloads under the window are unaffected by the fix."""
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        payload = b"C" * 5_000
        ctrl.send(payload)

        assert wire.user_bytes() == payload
        assert ctrl.stats()["pipeline_bytes"] == 0

    def test_probes_are_not_counted_as_user_data(self) -> None:
        """The bandwidth probe on first send() must not pollute the stream."""
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        ctrl.send(b"D" * 1_000)

        assert wire.probe_count() == 1
        assert wire.user_bytes() == b"D" * 1_000

    def test_reset_clears_pending_buffer(self) -> None:
        wire = LoopbackWire()
        ctrl = make_controller(wire)

        ctrl.send(b"E" * 50_000)
        assert ctrl.stats()["pipeline_bytes"] > 0

        ctrl.reset()
        assert ctrl.stats()["pipeline_bytes"] == 0
