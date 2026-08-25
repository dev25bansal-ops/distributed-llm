"""Regression tests: QUIC message framing must not truncate large payloads.

P0 finding: ``dist/p2p/quic_transport.py`` parsed every ``StreamDataReceived``
event as one complete message.  aioquic delivers stream bytes in packet-sized
events (~1195 bytes of payload capacity per datagram), so any message larger
than one packet was silently truncated to its first event's bytes.

The fix splits outgoing messages into self-delimiting frames::

    [1B priority][4B index][4B count][4B chunk length][chunk bytes]

and reassembles them byte-level per stream on receive (frames may be split
or coalesced across events arbitrarily).

Two test layers:

* ``TestFraming*`` exercise the pure ``_frame_chunks`` / ``_StreamAssembler``
  helpers — no aioquic required.
* ``TestQuicRoundTrip*`` run real loopback QUIC connections; they require
  aioquic and skip gracefully when it is absent (repo convention, see
  ``test_quic_verify.py``).
"""

from __future__ import annotations

import os
import socket

import pytest

import distllm.dist.p2p.quic_transport as qt
from distllm.dist.p2p.quic_transport import (
    _CHUNK_HEADER_SIZE,
    _MAX_FRAME_PAYLOAD,
    StreamPriority,
    _StreamAssembler,
    _frame_chunks,
)


# ---------------------------------------------------------------------------
# Pure framing / reassembly unit tests (no aioquic needed)
# ---------------------------------------------------------------------------
class TestFrameChunks:
    def test_small_message_is_single_frame(self):
        frames = _frame_chunks(StreamPriority.GOSSIP, b"x" * 100)
        assert len(frames) == 1
        assert len(frames[0]) == _CHUNK_HEADER_SIZE + 100

    def test_exact_boundary_1024(self):
        assert len(_frame_chunks(StreamPriority.DATA, b"x" * _MAX_FRAME_PAYLOAD)) == 1

    def test_one_past_boundary_1025(self):
        frames = _frame_chunks(StreamPriority.DATA, b"x" * (_MAX_FRAME_PAYLOAD + 1))
        assert len(frames) == 2

    def test_empty_message_produces_one_zero_length_frame(self):
        frames = _frame_chunks(StreamPriority.METADATA, b"")
        assert len(frames) == 1
        # header says chunk length 0
        prio, idx, count, clen = qt.struct.unpack(
            qt._CHUNK_HEADER_FMT, frames[0][: _CHUNK_HEADER_SIZE]
        )
        assert (prio, idx, count, clen) == (1, 0, 1, 0)

    @pytest.mark.parametrize("size", [0, 1, 1023, 1024, 1025, 5000, 65536])
    def test_frames_never_exceed_packet_capacity(self, size):
        # With max_datagram_size=1200, aioquic's per-packet stream payload
        # capacity is ~1195 bytes; a frame straddling packets is fine now
        # (self-delimiting), but keeping frames small bounds latency.
        for frame in _frame_chunks(StreamPriority.DATA, b"\xab" * size):
            assert len(frame) <= _CHUNK_HEADER_SIZE + _MAX_FRAME_PAYLOAD


class TestStreamAssembler:
    def _reassemble(self, payload: bytes, feed_size: int) -> list[tuple]:
        assembler = _StreamAssembler()
        blob = b"".join(_frame_chunks(StreamPriority.DATA, payload))
        messages = []
        for i in range(0, len(blob), feed_size):
            messages.extend(assembler.feed(blob[i : i + feed_size]))
        return messages

    @pytest.mark.parametrize("size", [0, 1, 100, 1023, 1024, 1025, 1195, 5000, 65536])
    def test_roundtrip_all_sizes_whole_feed(self, size):
        payload = os.urandom(size)
        msgs = self._reassemble(payload, len(payload) + 1)  # single feed
        assert len(msgs) == 1
        prio, out = msgs[0]
        assert prio is StreamPriority.DATA
        assert out == payload

    def test_roundtrip_packet_sized_events(self):
        # Simulate aioquic: ~1160-byte events with boundaries unrelated to
        # frame boundaries (the original truncation scenario).
        payload = os.urandom(5000)
        msgs = self._reassemble(payload, 1160)
        assert msgs == [(StreamPriority.DATA, payload)]

    def test_roundtrip_byte_dribble(self):
        payload = os.urandom(3000)
        assembler = _StreamAssembler()
        blob = b"".join(_frame_chunks(StreamPriority.GOSSIP, payload))
        messages = [m for byte in blob for m in assembler.feed(bytes([byte]))]
        assert messages == [(StreamPriority.GOSSIP, payload)]

    def test_two_messages_coalesced_in_one_event(self):
        d1, d2 = b"hello", b"world" * 400
        blob = (
            b"".join(_frame_chunks(StreamPriority.GOSSIP, d1))
            + b"".join(_frame_chunks(StreamPriority.METADATA, d2))
        )
        assembler = _StreamAssembler()
        messages = assembler.feed(blob)
        assert messages == [
            (StreamPriority.GOSSIP, d1),
            (StreamPriority.METADATA, d2),
        ]

    def test_separate_streams_are_independent(self):
        a1, a2 = _StreamAssembler(), _StreamAssembler()
        p1 = os.urandom(2500)
        p2 = os.urandom(700)
        f1 = _frame_chunks(StreamPriority.DATA, p1)
        f2 = _frame_chunks(StreamPriority.METADATA, p2)
        # Interleave frames across two "streams"
        out1, out2 = [], []
        out1.extend(a1.feed(f1[0]))
        out2.extend(a2.feed(f2[0]))
        out1.extend(a1.feed(b"".join(f1[1:])))
        out2.extend(a2.feed(b"".join(f2[1:])))
        assert out1 == [(StreamPriority.DATA, p1)]
        assert out2 == [(StreamPriority.METADATA, p2)]

    def test_malformed_index_discards_and_resyncs(self):
        assembler = _StreamAssembler()
        good = _frame_chunks(StreamPriority.DATA, b"payload")
        # Corrupt the first frame's index to 99 (count=1)
        bad = bytearray(good[0])
        bad[1:5] = (99).to_bytes(4, "big")
        messages = assembler.feed(bytes(bad))
        assert messages == []  # discarded, assembler reset
        # Assembler still usable afterwards
        messages = assembler.feed(good[0])
        assert messages == [(StreamPriority.DATA, b"payload")]

    def test_invalid_priority_clamped_to_data(self):
        assembler = _StreamAssembler()
        blob = _frame_chunks(StreamPriority.DATA, b"ok")[0]
        corrupted = bytes([200]) + blob[1:]  # priority byte 200 is invalid
        messages = assembler.feed(corrupted)
        assert messages == [(StreamPriority.DATA, b"ok")]


# ---------------------------------------------------------------------------
# Live loopback round-trips (require aioquic; skip like test_quic_verify.py)
# ---------------------------------------------------------------------------
aioquic = pytest.importorskip("aioquic")

from distllm.dist.p2p.quic_transport import QuicTransport  # noqa: E402


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _make_server_client():
    """Start a QUIC server + connected client on a free loopback port."""
    server = QuicTransport(node_id="srv")
    port = _free_udp_port()
    await server.listen("127.0.0.1", port)
    client = QuicTransport(node_id="cli")
    conn = await client.connect("127.0.0.1", port)
    return server, client, conn


@pytest.mark.parametrize("size", [100, 1195, 5000, 65536])
class TestQuicRoundTripSizes:
    async def test_send_to_recv_stream_global_queue(self, size):
        server, client, conn = await _make_server_client()
        try:
            payload = os.urandom(size)
            await conn.send(StreamPriority.DATA, payload)

            prio, data, peer = await server.recv_stream()
            assert prio is StreamPriority.DATA
            assert data == payload
            assert peer.startswith("127.0.0.1:")
        finally:
            await client.close()
            await server.close()

    async def test_reply_via_connection_recv(self, size):
        """Server->client direction exercises QuicConnection.recv()."""
        server, client, conn = await _make_server_client()
        try:
            payload = os.urandom(size)
            await conn.send(StreamPriority.DATA, payload)
            await server.recv_stream()  # drain

            reply = payload[::-1]
            srv_conn = next(iter(server._connections.values()))
            await srv_conn.send(StreamPriority.METADATA, reply)

            prio, data = await conn.recv(timeout=10)
            assert prio is StreamPriority.METADATA
            assert data == reply
        finally:
            await client.close()
            await server.close()


@pytest.mark.asyncio
async def test_multiple_sequential_large_messages():
    server, client, conn = await _make_server_client()
    try:
        payloads = [os.urandom(5000) for _ in range(5)]
        for p in payloads:
            await conn.send(StreamPriority.DATA, p)

        received = []
        for _ in payloads:
            _, data, _ = await server.recv_stream()
            received.append(data)
        # NOTE: every send() uses a fresh QUIC stream, and QUIC guarantees
        # ordering only *within* a stream — cross-stream delivery order is
        # unspecified by design.  Assert completeness/content, not order.
        assert sorted(received) == sorted(payloads)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_empty_payload_round_trips():
    server, client, conn = await _make_server_client()
    try:
        await conn.send(StreamPriority.GOSSIP, b"")
        prio, data, _ = await server.recv_stream()
        assert (prio, data) == (StreamPriority.GOSSIP, b"")
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_accept_path_delivers_inbound_connection():
    server, client, conn = await _make_server_client()
    try:
        inbound = await server.accept()
        assert inbound.peer_id.startswith("127.0.0.1:")
        await inbound.send(StreamPriority.GOSSIP, b"welcome")
        prio, data = await conn.recv(timeout=10)
        assert (prio, data) == (StreamPriority.GOSSIP, b"welcome")
    finally:
        await client.close()
        await server.close()
