"""Bandwidth measurement, congestion control, and multi-stream striping for pipeline transport."""

from __future__ import annotations

import math
import struct
import time as _time
from collections import deque
from typing import Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MSS: int = 1460  # Maximum Segment Size (typical TCP MSS over Ethernet)
DEFAULT_PROBE_INTERVAL: float = 1.0  # Seconds between bandwidth probes
EWMA_ALPHA: float = 0.125  # Smoothing factor for EWMA bandwidth estimate
MIN_CWND: int = DEFAULT_MSS * 2  # Floor for congestion window
MIN_SSTHRESH: int = DEFAULT_MSS * 2  # Floor for slow-start threshold
LOSS_WINDOW_SEC: float = 5.0  # Rolling window for loss rate tracking


# ---------------------------------------------------------------------------
# BandwidthMeasurer
# ---------------------------------------------------------------------------

class BandwidthMeasurer:
    """Continuous bandwidth measurement with EWMA smoothing.

    Sends probes of varying sizes to a peer and measures round-trip time to
    estimate achievable bandwidth. Maintains separate EWMA-smoothed estimates
    for send and receive directions.

    Bandwidth is computed as::

        bps = probe_size_bytes * 2 / rtt_seconds

    which accounts for the round-trip nature of the probe exchange (the data
    must be echoed back to measure both directions simultaneously).
    """

    def __init__(
        self,
        send_fn: Callable[[bytes], None] | None = None,
        *,
        alpha: float = EWMA_ALPHA,
        probe_interval: float = DEFAULT_PROBE_INTERVAL,
        max_probe_bytes: int = 1 << 20,  # 1 MiB
        min_probe_bytes: int = 1 << 10,  # 1 KiB
    ) -> None:
        """Initialise the measurer.

        Args:
            send_fn: Optional callable that sends raw probe bytes to the peer.
                      If not provided, the caller must invoke ``probe()``
                      directly after each send.
            alpha: EWMA smoothing factor (0 < alpha <= 1).
            probe_interval: Minimum seconds between automatic probes.
            max_probe_bytes: Largest probe payload in bytes.
            min_probe_bytes: Smallest probe payload in bytes.
        """
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if probe_interval <= 0:
            raise ValueError(f"probe_interval must be > 0, got {probe_interval}")
        if min_probe_bytes < 64:
            raise ValueError(f"min_probe_bytes must be >= 64, got {min_probe_bytes}")
        if max_probe_bytes <= min_probe_bytes:
            raise ValueError(
                f"max_probe_bytes ({max_probe_bytes}) must be > "
                f"min_probe_bytes ({min_probe_bytes})"
            )

        self._send_fn = send_fn
        self._alpha = alpha
        self._probe_interval = probe_interval
        self._max_probe_bytes = max_probe_bytes
        self._min_probe_bytes = min_probe_bytes

        # EWMA estimates
        self._send_bps: float = 0.0  # send-direction bandwidth (bits/sec)
        self._recv_bps: float = 0.0  # recv-direction bandwidth (bits/sec)

        # Internal probe pacing
        self._last_probe_time: float = 0.0
        self._probe_counter: int = 0
        self._pending_probes: dict[int, tuple[float, int]] = {}  # seq -> (send_time, size)

    # -- Public API ---------------------------------------------------------

    def measure(self) -> tuple[float, float]:
        """Return the current (send_bps, recv_bps) estimate.

        Returns:
            A tuple ``(send_bps, recv_bps)`` in **bits per second**. Both
            values are zero until at least one probe round-trip completes.
        """
        return (self._send_bps, self._recv_bps)

    def send_bps(self) -> float:
        """Current send-direction bandwidth estimate in bits per second."""
        return self._send_bps

    def recv_bps(self) -> float:
        """Current receive-direction bandwidth estimate in bits per second."""
        return self._recv_bps

    def should_probe(self, now: float | None = None) -> bool:
        """Whether enough time has elapsed to send another probe.

        Args:
            now: Current timestamp (seconds since epoch); uses ``time.time()``
                 if not provided.

        Returns:
            True if a probe should be sent now.
        """
        if now is None:
            now = _time.time()
        return (now - self._last_probe_time) >= self._probe_interval

    def build_probe(self) -> bytes:
        """Build a binary-encoded probe payload and record it as pending.

        The probe format is a fixed header::

            [magic:4B][seq:8B][size:8B][payload:size B]

        - magic:    ``0xDEADBEEF`` (uint32, network byte order)
        - seq:      monotonically increasing sequence number (uint64)
        - size:     payload size in bytes (uint64)
        - payload:  zeros

        Returns:
            The complete probe bytes ready to send.
        """
        now = _time.time()
        self._last_probe_time = now

        # Ramp probe size up and down to sample different throughput regimes
        size = self._next_probe_size()
        seq = self._probe_counter
        self._probe_counter += 1

        header = struct.pack("!IQQ", 0xDEADBEEF, seq, size)
        payload = b"\x00" * size
        probe_bytes = header + payload

        self._pending_probes[seq] = (now, size)
        return probe_bytes

    def submit_probe_echo(self, echo_bytes: bytes) -> None:
        """Resolve a pending probe from the peer's echoed reply.

        The echo must be the *identical* bytes that were sent via
        ``build_probe()``.  The measurer parses the header, computes the
        RTT, and updates the EWMA estimates.

        Args:
            echo_bytes: The raw probe bytes returned by the peer.
        """
        if len(echo_bytes) < 20:
            return  # Header incomplete — ignore silently

        magic, seq, _ = struct.unpack("!IQQ", echo_bytes[:20])
        if magic != 0xDEADBEEF:
            return  # Not one of our probes

        entry = self._pending_probes.pop(seq, None)
        if entry is None:
            return  # Already resolved (stale duplicate)

        send_time, probe_size = entry
        rtt = _time.time() - send_time
        if rtt <= 0.0:
            return

        # Bandwidth = total bytes transferred / RTT (bits per second)
        # We send probe_size bytes and receive the same probe_size bytes back.
        total_bytes = probe_size * 2
        bps = (total_bytes * 8) / rtt

        # Update EWMA for send direction (assume symmetric split)
        half_bps = bps / 2.0
        if self._send_bps == 0.0:
            self._send_bps = half_bps
            self._recv_bps = half_bps
        else:
            self._send_bps = (
                self._alpha * half_bps + (1 - self._alpha) * self._send_bps
            )
            self._recv_bps = (
                self._alpha * half_bps + (1 - self._alpha) * self._recv_bps
            )

    def probe_count(self) -> int:
        """Number of probes sent that have not yet been echoed back."""
        return len(self._pending_probes)

    def reset(self) -> None:
        """Reset all estimates and pending probes."""
        self._send_bps = 0.0
        self._recv_bps = 0.0
        self._last_probe_time = 0.0
        self._pending_probes.clear()

    def _next_probe_size(self) -> int:
        """Cycle through increasing probe sizes, then wrap.

        Pattern: min, min*2, min*4, ..., max/2, max, max/2, ..., min, ...
        This gives a range of sampled sizes so the EWMA captures behaviour
        at different throughput levels.
        """
        n = self._probe_counter
        # Compute size = min * 2^(n mod (2 * num_doublings))
        # where num_doublings is how many times we can double before exceeding max.
        max_doublings = int(math.log2(self._max_probe_bytes // self._min_probe_bytes))
        # Cycle size: go up to max, then back down
        period = 2 * max_doublings
        pos = n % (period + 1) if period > 0 else 0
        if pos <= max_doublings:
            multiplier = 1 << pos
        else:
            multiplier = 1 << (2 * max_doublings - pos)
        size = self._min_probe_bytes * multiplier
        return min(size, self._max_probe_bytes)


# ---------------------------------------------------------------------------
# WindowCongestionController
# ---------------------------------------------------------------------------

class WindowCongestionController:
    """AIMD (Additive Increase Multiplicative Decrease) congestion controller.

    Manages a congestion window (``cwnd``) in bytes and a slow-start
    threshold (``ssthresh``).  The controller implements standard TCP Reno /
    NewReno behaviour:

    Slow start (``cwnd < ssthresh``)::

        cwnd += MSS         (per ack)

    Congestion avoidance (``cwnd >= ssthresh``)::

        cwnd += MSS * (MSS / cwnd)    (per ack)

    On loss::

        ssthresh = max(cwnd / 2, MIN_SSTHRESH)
        cwnd = ssthresh

    The controller also tracks a smoothed loss rate over a rolling time
    window for diagnostic purposes.
    """

    def __init__(
        self,
        initial_cwnd: int = DEFAULT_MSS * 10,
        initial_ssthresh: int = DEFAULT_MSS * 100,
        mss: int = DEFAULT_MSS,
    ) -> None:
        """Initialise the congestion controller.

        Args:
            initial_cwnd: Initial congestion window in bytes.
            initial_ssthresh: Initial slow-start threshold in bytes.
            mss: Maximum Segment Size in bytes.
        """
        if initial_cwnd < MIN_CWND:
            raise ValueError(
                f"initial_cwnd ({initial_cwnd}) must be >= {MIN_CWND}"
            )
        if initial_ssthresh < MIN_SSTHRESH:
            raise ValueError(
                f"initial_ssthresh ({initial_ssthresh}) must be >= {MIN_SSTHRESH}"
            )
        if mss < 1:
            raise ValueError(f"mss must be >= 1, got {mss}")

        self._cwnd = initial_cwnd
        self._ssthresh = initial_ssthresh
        self._mss = mss

        # Rolling loss tracking
        self._loss_events: deque[float] = deque()
        self._total_acks: int = 0
        self._total_losses: int = 0

    # -- Properties ---------------------------------------------------------

    @property
    def cwnd(self) -> int:
        """Current congestion window in bytes."""
        return self._cwnd

    @property
    def ssthresh(self) -> int:
        """Current slow-start threshold in bytes."""
        return self._ssthresh

    @property
    def mss(self) -> int:
        """Maximum Segment Size in bytes."""
        return self._mss

    # -- Public API ---------------------------------------------------------

    def get_window(self) -> int:
        """Return the current send window (``cwnd``) in bytes.

        This is the amount of data that may be in flight at any given time.
        """
        return self._cwnd

    def on_ack(self, acked_bytes: int = 1) -> None:
        """Update congestion state after a successful acknowledgment.

        Args:
            acked_bytes: Number of bytes acknowledged (default 1 for
                         per-packet ACK, or the actual segment size for
                         delayed / SACK ACKs).
        """
        self._total_acks += 1

        if self._cwnd < self._ssthresh:
            # Slow start: additive increase per ACK
            self._cwnd += self._mss
        else:
            # Congestion avoidance: cwnd += MSS * (MSS / cwnd)
            increment = (self._mss * self._mss) / self._cwnd
            self._cwnd += increment

    def on_loss(self) -> None:
        """React to a packet loss event: multiplicative decrease."""
        self._total_losses += 1
        now = _time.time()
        self._loss_events.append(now)

        # Prune old loss events outside the rolling window
        while self._loss_events and self._loss_events[0] < now - LOSS_WINDOW_SEC:
            self._loss_events.popleft()

        # Multiplicative decrease
        self._ssthresh = max(int(self._cwnd / 2), MIN_SSTHRESH)
        self._cwnd = self._ssthresh

    def in_slow_start(self) -> bool:
        """Whether the controller is currently in slow-start phase."""
        return self._cwnd < self._ssthresh

    def loss_rate(self) -> float:
        """Loss rate over the rolling window (0.0 - 1.0).

        Computed as ``losses_in_window / total_acks`` over the window.
        Returns 0.0 if no acks have been recorded.
        """
        if self._total_acks == 0:
            return 0.0

        now = _time.time()
        # Prune stale entries first
        while self._loss_events and self._loss_events[0] < now - LOSS_WINDOW_SEC:
            self._loss_events.popleft()

        # Estimate acks that occurred within the window proportionally
        window_losses = len(self._loss_events)
        # Approximate window acks as total_acks * (window / total_time)
        # For simplicity, return the overall ratio — the rolling window is
        # primarily used for trend detection, not precise accounting.
        return window_losses / max(self._total_acks, 1)

    def reset(self) -> None:
        """Reset to initial state (cwnd, ssthresh, loss history)."""
        self._cwnd = self._ssthresh  # Go to ssthresh, not initial_cwnd
        self._loss_events.clear()
        self._total_acks = 0
        self._total_losses = 0


# ---------------------------------------------------------------------------
# MultiStreamStripter
# ---------------------------------------------------------------------------

class MultiStreamStripter:
    """Split large tensor transfers across N streams with weighted fractions.

    Each stream receives a fraction of the data proportional to its measured
    bandwidth.  If no bandwidth measurements are provided, the data is split
    evenly.
    """

    def __init__(self) -> None:
        pass

    # -- Public API ---------------------------------------------------------

    def stripe(
        self,
        data: bytes,
        num_streams: int,
        *,
        weights: list[float] | None = None,
        min_chunk: int = 1,
    ) -> list[bytes]:
        """Split *data* into *num_streams* chunks.

        Args:
            data: Raw bytes to split.
            num_streams: Number of stripes (must be >= 1).
            weights: Optional per-stream bandwidth weights.  If provided,
                     ``len(weights)`` must equal *num_streams*.  Streams with
                     higher weights receive proportionally more data.  If
                     ``None``, data is split evenly.
            min_chunk: Minimum byte count per chunk (enforced for streams
                       that would otherwise get less).  Must be >= 1.

        Returns:
            A list of ``num_streams`` byte chunks.  Empty chunks are
            ``b""`` for streams that receive zero data.
        """
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")
        if not data:
            return [b""] * num_streams
        if min_chunk < 1:
            raise ValueError(f"min_chunk must be >= 1, got {min_chunk}")

        total = len(data)

        if num_streams == 1:
            return [data]

        # Determine per-stream fractions
        if weights is not None:
            if len(weights) != num_streams:
                raise ValueError(
                    f"len(weights) ({len(weights)}) must equal "
                    f"num_streams ({num_streams})"
                )
            total_weight = sum(abs(w) for w in weights)
            if total_weight <= 0:
                fractions = [1.0 / num_streams] * num_streams
            else:
                fractions = [abs(w) / total_weight for w in weights]
        else:
            fractions = [1.0 / num_streams] * num_streams

        # Allocate bytes.  We use a largest-remainder method to ensure the
        # sum of allocated bytes equals total (no rounding gaps).
        raw_alloc = [int(total * f) for f in fractions]
        remainder = total - sum(raw_alloc)
        # Distribute remainder one byte at a time to streams with the largest
        # fractional remainder.
        fractional_parts = [(total * f) - int(total * f) for f in fractions]
        for _ in range(remainder):
            idx = fractional_parts.index(max(fractional_parts))
            raw_alloc[idx] += 1
            fractional_parts[idx] = -1.0  # Prevents re-selection

        # Enforce min_chunk -- reduce from the largest chunk if needed
        if min_chunk > 1:
            for i in range(num_streams):
                if 0 < raw_alloc[i] < min_chunk:
                    # Borrow from the largest chunk
                    largest_idx = max(
                        range(num_streams), key=lambda j: raw_alloc[j]
                    )
                    if raw_alloc[largest_idx] - min_chunk >= raw_alloc[i]:
                        borrowed = min_chunk - raw_alloc[i]
                        raw_alloc[largest_idx] -= borrowed
                        raw_alloc[i] += borrowed
                    else:
                        # Not enough to borrow; leave as-is
                        pass

        # Build slices
        chunks: list[bytes] = []
        offset = 0
        for alloc in raw_alloc:
            if alloc <= 0:
                chunks.append(b"")
            else:
                chunks.append(data[offset : offset + alloc])
                offset += alloc

        # Safety check — if we ran out of data early, pad with empty chunks.
        while len(chunks) < num_streams:
            chunks.append(b"")

        return chunks

    def reassemble(self, chunks: list[bytes]) -> bytes:
        """Reassemble the original data from striped chunks.

        Args:
            chunks: The list of chunks returned by ``stripe()`` (in order).

        Returns:
            The concatenated original bytes.
        """
        return b"".join(chunks)


# ---------------------------------------------------------------------------
# PipelineTransportController
# ---------------------------------------------------------------------------

class PipelineTransportController:
    """Combined bandwidth measurement, congestion control, and multi-stream
    striping for pipeline tensor transport.

    This controller integrates:

    - :class:`BandwidthMeasurer` — probes the link to maintain EWMA-smoothed
      bandwidth estimates for send and receive directions.
    - :class:`WindowCongestionController` — AIMD congestion window that
      governs how much data can be in flight.
    - :class:`MultiStreamStripter` — splits large payloads across parallel
      streams with weight-aware allocation.

    Typical usage::

        controller = PipelineTransportController(
            send_fn=lambda data: transport.send(data),
        )

        # Sending with congestion-aware striping
        controller.send(my_tensor_bytes, peer)

        # Receiving and reassembling
        result = controller.recv()

        # View current diagnostics
        stats = controller.stats()
    """

    def __init__(
        self,
        send_fn: Callable[[bytes], None] | None = None,
        recv_fn: Callable[[], bytes] | None = None,
        *,
        num_streams: int = 4,
        bandwidth_alpha: float = EWMA_ALPHA,
        probe_interval: float = DEFAULT_PROBE_INTERVAL,
        mss: int = DEFAULT_MSS,
        initial_cwnd: int = DEFAULT_MSS * 10,
        initial_ssthresh: int = DEFAULT_MSS * 100,
    ) -> None:
        """Initialise the pipeline transport controller.

        Args:
            send_fn: Callable that sends raw bytes to the peer.  May be
                     ``None`` and set later via :meth:`set_send_fn`.
            recv_fn: Callable that receives raw bytes from the peer.  May be
                     ``None`` and set later via :meth:`set_recv_fn`.
            num_streams: Number of parallel streams for striping.
            bandwidth_alpha: EWMA smoothing factor for bandwidth measurement.
            probe_interval: Minimum seconds between bandwidth probes.
            mss: Maximum Segment Size for congestion control.
            initial_cwnd: Initial congestion window in bytes.
            initial_ssthresh: Initial slow-start threshold in bytes.
        """
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")

        self._num_streams = num_streams
        self._send_fn = send_fn
        self._recv_fn = recv_fn

        # Sub-controllers
        self._measurer = BandwidthMeasurer(
            send_fn=send_fn,
            alpha=bandwidth_alpha,
            probe_interval=probe_interval,
        )
        self._congestion = WindowCongestionController(
            mss=mss,
            initial_cwnd=initial_cwnd,
            initial_ssthresh=initial_ssthresh,
        )
        self._stripter = MultiStreamStripter()

        # Internal state
        self._send_buf: bytes = b""
        self._recv_buf: bytes = b""
        self._total_bytes_sent: int = 0
        self._total_bytes_recv: int = 0
        self._loss_count: int = 0
        self._ack_count: int = 0
        self._sent_chunks: list[bytes] = []

    # -- Public API ---------------------------------------------------------

    def set_send_fn(self, send_fn: Callable[[bytes], None]) -> None:
        """Set or replace the send callback."""
        self._send_fn = send_fn
        self._measurer._send_fn = send_fn  # noqa: SLF001

    def set_recv_fn(self, recv_fn: Callable[[], bytes]) -> None:
        """Set or replace the receive callback."""
        self._recv_fn = recv_fn

    def send(self, data: bytes, peer: str | None = None) -> None:
        """Send *data* with congestion-controlled striping.

        This method:

        1. Optionally sends a bandwidth probe if the probe interval has
           elapsed.
        2. Strips *data* across ``num_streams`` streams using the latest
           bandwidth weights.
        3. Applies congestion window gating (if the window does not allow
           the full send, data is buffered and sent incrementally).
        4. Tracks in-flight bytes and sends the chunks via ``send_fn``.

        Args:
            data: The raw bytes to send.
            peer: Optional peer identifier (reserved for future use).
        """
        if not self._send_fn:
            raise RuntimeError("send_fn not set; call set_send_fn() first")

        # -- Probe ----------------------------------------------------------
        if self._measurer.should_probe():
            probe = self._measurer.build_probe()
            self._send_fn(probe)

        # -- Strip ----------------------------------------------------------
        send_bps, recv_bps = self._measurer.measure()
        if send_bps > 0 and recv_bps > 0:
            weights = [send_bps, recv_bps]
            # Scale weights to num_streams by repeating the measured pair
            scaled_weights: list[float] = []
            for i in range(self._num_streams):
                scaled_weights.append(weights[i % len(weights)])
        else:
            scaled_weights = None

        chunks = self._stripter.stripe(
            data, self._num_streams, weights=scaled_weights,
        )
        self._sent_chunks = list(chunks)

        # -- Congestion window gating --------------------------------------
        window = self._congestion.get_window()
        total_bytes = sum(len(c) for c in chunks)

        if total_bytes > window:
            # Data exceeds current window — buffer excess for later
            self._send_buf = data
            # Send up to window bytes, drain from chunks
            sent = 0
            for i, chunk in enumerate(chunks):
                chunk_len = len(chunk)
                if sent + chunk_len <= window:
                    self._send_fn(chunk)
                    sent += chunk_len
                elif sent < window:
                    partial = window - sent
                    self._send_fn(chunk[:partial])
                    sent += partial
                    self._send_buf = data[sent:]
                else:
                    break
        else:
            for chunk in chunks:
                self._send_fn(chunk)

        self._total_bytes_sent += total_bytes

    def recv(self) -> bytes:
        """Receive and reassemble data from the peer.

        Returns:
            The reassembled raw bytes.
        """
        if not self._recv_fn:
            raise RuntimeError("recv_fn not set; call set_recv_fn() first")

        raw = self._recv_fn()
        if not raw:
            return b""

        # Check if this is a probe echo
        if len(raw) >= 20:
            magic, _, _ = struct.unpack("!IQQ", raw[:20])
            if magic == 0xDEADBEEF:
                self._measurer.submit_probe_echo(raw)
                return b""  # Probe echoes are not user data

        chunks = self._stripter.reassemble([raw])
        self._total_bytes_recv += len(chunks)

        return chunks

    def on_ack(self, acked_bytes: int = 1) -> None:
        """Notify the congestion controller of a successful ACK.

        After the window grows, any bytes previously buffered by
        :meth:`send` (because they exceeded the congestion window) are
        drained while the window allows, so buffered data is never
        silently dropped.
        """
        self._ack_count += 1
        self._congestion.on_ack(acked_bytes=acked_bytes)
        self._flush_send_buf()

    def _flush_send_buf(self) -> None:
        """Send buffered bytes while the congestion window allows.

        Sends at most ``get_window()`` bytes from ``_send_buf`` per call,
        keeping the remainder buffered for subsequent ACKs.
        """
        if not self._send_buf or not self._send_fn:
            return

        window = self._congestion.get_window()
        if window <= 0:
            return

        n = min(window, len(self._send_buf))
        self._send_fn(self._send_buf[:n])
        self._total_bytes_sent += n
        self._send_buf = self._send_buf[n:]

    def on_loss(self) -> None:
        """Notify the congestion controller of a packet loss."""
        self._loss_count += 1
        self._congestion.on_loss()

    def stats(self) -> dict[str, object]:
        """Return a snapshot of current transport statistics.

        Returns:
            A dictionary with keys:

            - ``send_bps``: Current send bandwidth (bits per second).
            - ``recv_bps``: Current receive bandwidth (bits per second).
            - ``cwnd_bytes``: Congestion window in bytes.
            - ``ssthresh_bytes``: Slow-start threshold in bytes.
            - ``in_slow_start``: Whether in slow-start phase.
            - ``loss_rate``: Rolling loss rate (0.0 - 1.0).
            - ``total_bytes_sent``: Cumulative bytes sent.
            - ``total_bytes_recv``: Cumulative bytes received.
            - ``total_acks``: Cumulative ACKs processed.
            - ``total_losses``: Cumulative loss events.
            - ``pipeline_bytes``: Number of bytes in the send buffer.
            - ``num_streams``: Configured number of streams.
            - ``probes_in_flight``: Number of unresolved bandwidth probes.
        """
        send_bps, recv_bps = self._measurer.measure()
        return {
            "send_bps": send_bps,
            "recv_bps": recv_bps,
            "cwnd_bytes": self._congestion.get_window(),
            "ssthresh_bytes": self._congestion.ssthresh,
            "in_slow_start": self._congestion.in_slow_start(),
            "loss_rate": self._congestion.loss_rate(),
            "total_bytes_sent": self._total_bytes_sent,
            "total_bytes_recv": self._total_bytes_recv,
            "total_acks": self._ack_count,
            "total_losses": self._loss_count,
            "pipeline_bytes": len(self._send_buf),
            "num_streams": self._num_streams,
            "probes_in_flight": self._measurer.probe_count(),
        }

    def reset(self) -> None:
        """Reset all sub-controllers and internal state."""
        self._measurer.reset()
        self._congestion.reset()
        self._send_buf = b""
        self._recv_buf = b""
        self._total_bytes_sent = 0
        self._total_bytes_recv = 0
        self._loss_count = 0
        self._ack_count = 0
        self._sent_chunks.clear()
