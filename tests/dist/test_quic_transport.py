"""Tests for QUIC transport for wide-area inference."""

import importlib.util
import os
import struct
from typing import Any

import pytest


LENGTH_PREFIX_FORMAT = "!I"
LENGTH_PREFIX_SIZE = struct.calcsize(LENGTH_PREFIX_FORMAT)


def _get_module():
    import sys
    import types

    path = os.path.join("src", "distllm", "dist", "quic_transport.py")
    spec = importlib.util.spec_from_file_location("quic_transport", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quic_transport"] = mod

    mod.logger = types.ModuleType("logger")
    mod.logger.debug = lambda *a, **kw: None
    mod.logger.info = lambda *a, **kw: None
    mod.logger.error = lambda *a, **kw: None
    mod.logger.exception = lambda *a, **kw: None

    spec.loader.exec_module(mod)
    return mod


class TestMessageFraming:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()

    def test_frame_and_unframe(self):
        data = b"hello world"
        framed = self.mod._frame_message(data)
        assert len(framed) == LENGTH_PREFIX_SIZE + len(data)
        msg, remaining = self.mod._unframe_message(bytearray(framed))
        assert msg == data
        assert remaining == bytearray()

    def test_unframe_empty_buffer(self):
        msg, remaining = self.mod._unframe_message(bytearray())
        assert msg is None
        assert remaining == bytearray()

    def test_unframe_partial_header(self):
        buf = bytearray(b"\x00")
        msg, remaining = self.mod._unframe_message(buf)
        assert msg is None
        assert remaining == buf

    def test_unframe_partial_body(self):
        data = b"short body"
        framed = self.mod._frame_message(data)
        truncated = bytearray(framed[:-3])
        msg, remaining = self.mod._unframe_message(truncated)
        assert msg is None
        assert len(remaining) < len(framed)

    def test_unframe_multiple_messages(self):
        d1 = b"msg1"
        d2 = b"msg2"
        framed = self.mod._frame_message(d1) + self.mod._frame_message(d2)
        buf = bytearray(framed)
        m1, buf = self.mod._unframe_message(buf)
        assert m1 == d1
        m2, buf = self.mod._unframe_message(buf)
        assert m2 == d2
        assert buf == bytearray()

    def test_unframe_large_message(self):
        data = os.urandom(100_000)
        framed = self.mod._frame_message(data)
        msg, remaining = self.mod._unframe_message(bytearray(framed))
        assert msg == data
        assert remaining == bytearray()


class TestIsQuicAvailable:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()

    def test_quic_not_available_by_default(self):
        result = self.mod.is_quic_available()
        assert result is False


class TestQuicStreamHandler:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.QuicStreamHandler = cls.mod.QuicStreamHandler

    def test_default_handler_no_fn(self):
        handler = self.QuicStreamHandler()
        assert handler.forward_fn is None

    def test_set_forward_fn(self):
        handler = self.QuicStreamHandler()
        fn = lambda x: b"ok"
        handler.forward_fn = fn
        assert handler.forward_fn is fn

    def test_quic_stream_received_dispatches_to_fn(self):
        results = []

        def handler_fn(data):
            results.append(data)
            return b"response"

        handler = self.QuicStreamHandler(forward_fn=handler_fn)

        class FakeConnection:
            sent = []

            def send(self, stream_id, data, end_stream=False):
                self.sent.append((stream_id, data, end_stream))

        conn = FakeConnection()
        framed = self.mod._frame_message(b"request_data")
        handler.quic_stream_received(123, framed, True, conn)

        assert results == [b"request_data"]
        assert len(conn.sent) == 1
        sid, data, end = conn.sent[0]
        assert sid == 123
        assert end is True
        msg, _ = self.mod._unframe_message(bytearray(data))
        assert msg == b"response"

    def test_partial_data_buffered(self):
        results = []

        def handler_fn(data):
            results.append(data)
            return b"resp"

        handler = self.QuicStreamHandler(forward_fn=handler_fn)

        class FakeConnection:
            def send(self, stream_id, data, end_stream=False):
                pass

        full = self.mod._frame_message(b"hello")
        # Send only first 3 bytes
        handler.quic_stream_received(42, full[:3], False, FakeConnection())
        assert results == []
        # Send remaining bytes (including end_stream)
        handler.quic_stream_received(42, full[3:], True, FakeConnection())
        assert results == [b"hello"]

    def test_handler_exception_does_not_crash(self):
        def handler_fn(data):
            raise RuntimeError("test error")

        handler = self.QuicStreamHandler(forward_fn=handler_fn)

        class FakeConnection:
            sent = []

            def send(self, stream_id, data, end_stream=False):
                self.sent.append(data)

        conn = FakeConnection()
        framed = self.mod._frame_message(b"data")
        handler.quic_stream_received(1, framed, True, conn)
        assert len(conn.sent) == 1
        assert conn.sent[0] == b""

    def test_cleanup_removes_buffer(self):
        handler = self.QuicStreamHandler()
        handler._buffers[99] = bytearray(b"some data")
        handler.cleanup(99)
        assert 99 not in handler._buffers


class TestQuicTransportClient:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.QuicTransportClient = cls.mod.QuicTransportClient
        cls.QuicConfig = cls.mod.QuicConfig

    def test_not_connected_after_init(self):
        client = self.QuicTransportClient()
        assert client.is_connected is False

    def test_forward_pass_raises_when_not_connected(self):
        client = self.QuicTransportClient()
        import asyncio
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.forward_pass(b"test"))


class TestQuicTransportServer:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.QuicTransportServer = cls.mod.QuicTransportServer
        cls.QuicConfig = cls.mod.QuicConfig

    def test_not_serving_after_init(self):
        server = self.QuicTransportServer()
        assert server.is_serving is False
        assert server.handler is not None

    def test_shutdown_noop_when_not_started(self):
        import asyncio
        server = self.QuicTransportServer()
        asyncio.run(server.shutdown())


def test_module_exports():
    mod = _get_module()
    assert hasattr(mod, "QuicConfig")
    assert hasattr(mod, "QuicStreamHandler")
    assert hasattr(mod, "QuicTransportClient")
    assert hasattr(mod, "QuicTransportServer")
    assert hasattr(mod, "is_quic_available")
    assert hasattr(mod, "_frame_message")
    assert hasattr(mod, "_unframe_message")
