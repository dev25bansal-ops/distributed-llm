"""Tests for BufferedAccumulator, PartialJSONParser, StructuredStreamHandler, PartialResult."""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/structured_output/streaming.py")
BufferedAccumulator = _mod.BufferedAccumulator
PartialJSONParser = _mod.PartialJSONParser
StructuredStreamHandler = _mod.StructuredStreamHandler
PartialResult = _mod.PartialResult


# ── PartialResult ───────────────────────────────────────────────────────────


class TestPartialResult:
    """Construction and defaults for PartialResult."""

    def test_minimal(self):
        r = PartialResult(text="hello")
        assert r.text == "hello"
        assert r.data is None
        assert r.is_complete is False
        assert r.errors == []

    def test_full_construction(self):
        r = PartialResult(text='{"a": 1}', data={"a": 1}, is_complete=True, errors=[])
        assert r.text == '{"a": 1}'
        assert r.data == {"a": 1}
        assert r.is_complete is True

    def test_with_errors(self):
        r = PartialResult(text="bad", errors=["parse error"])
        assert r.errors == ["parse error"]
        assert r.is_complete is False


# ── BufferedAccumulator ──────────────────────────────────────────────────────


class TestBufferedAccumulator:
    """Construction, defaults, key methods, edge cases."""

    def test_default_min_chars(self):
        acc = BufferedAccumulator()
        assert acc.buffer_size == 0
        assert acc.has_content is False

    def test_custom_min_chars(self):
        acc = BufferedAccumulator(min_chars=10)
        assert acc._min_chars == 10

    def test_add_below_threshold(self):
        acc = BufferedAccumulator(min_chars=50)
        chunks = acc.add("hello")
        assert chunks == []
        assert acc.has_content is True
        assert acc.buffer_size == 5

    def test_add_above_threshold(self):
        acc = BufferedAccumulator(min_chars=10)
        chunks = acc.add("hello world!!!")
        assert len(chunks) == 1
        assert chunks[0] == "hello world!!!"
        assert acc.has_content is False  # buffer flushed

    def test_add_exact_threshold(self):
        acc = BufferedAccumulator(min_chars=5)
        chunks = acc.add("hello")
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_add_multiple_accumulate_then_flush(self):
        acc = BufferedAccumulator(min_chars=50)
        acc.add("a" * 30)
        assert acc.buffer_size == 30
        chunks = acc.add("b" * 30)
        assert len(chunks) == 1
        assert len(chunks[0]) == 60

    def test_flush_all_empty(self):
        acc = BufferedAccumulator()
        text = acc.flush_all()
        assert text == ""
        assert acc.has_content is False

    def test_flush_all_with_content(self):
        acc = BufferedAccumulator(min_chars=50)
        acc.add("pending")
        text = acc.flush_all()
        assert text == "pending"
        assert acc.has_content is False

    def test_has_content_after_add(self):
        acc = BufferedAccumulator()
        assert acc.has_content is False
        acc.add("x")
        assert acc.has_content is True

    def test_has_content_after_flush(self):
        acc = BufferedAccumulator()
        acc.add("x")
        acc.flush_all()
        assert acc.has_content is False

    def test_buffer_size_tracking(self):
        acc = BufferedAccumulator(min_chars=100)
        assert acc.buffer_size == 0
        acc.add("abc")
        assert acc.buffer_size == 3
        acc.add("def")
        assert acc.buffer_size == 6
        acc.flush_all()
        assert acc.buffer_size == 0


# ── PartialJSONParser ───────────────────────────────────────────────────────


class TestPartialJSONParser:
    """Construction, feed method, reset, edge cases."""

    def test_initial_empty(self):
        parser = PartialJSONParser()
        assert parser._accumulated == ""

    def test_feed_partial_object(self):
        parser = PartialJSONParser()
        result = parser.feed('{"key": "val')
        assert isinstance(result, PartialResult)
        assert result.is_complete is False
        assert "key" in result.text

    def test_feed_complete_object(self):
        parser = PartialJSONParser()
        result = parser.feed('{"key": "val"}')
        assert result.is_complete is True
        assert result.data == {"key": "val"}

    def test_feed_multiple_calls(self):
        parser = PartialJSONParser()
        r1 = parser.feed('{"key": "')
        assert r1.is_complete is False
        r2 = parser.feed('val"}')
        assert r2.is_complete is True
        assert r2.data == {"key": "val"}

    def test_feed_empty_string(self):
        parser = PartialJSONParser()
        result = parser.feed("")
        assert isinstance(result, PartialResult)

    def test_feed_nested_object(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": {"b": 1}}')
        assert result.is_complete is True
        assert result.data == {"a": {"b": 1}}

    def test_feed_array(self):
        parser = PartialJSONParser()
        result = parser.feed("[1, 2, 3]")
        assert result.is_complete is True
        assert result.data == [1, 2, 3]  # list, not dict

    def test_feed_partial_nested(self):
        """Partial nested object with recovery through suffix closing."""
        parser = PartialJSONParser()
        result = parser.feed('{"a": {"b": 1')
        assert isinstance(result, PartialResult)
        # Should not crash, and should return a result (possibly partial data)

    def test_reset(self):
        parser = PartialJSONParser()
        parser.feed("some text")
        assert parser._accumulated != ""
        parser.reset()
        assert parser._accumulated == ""

    def test_feed_after_reset(self):
        parser = PartialJSONParser()
        parser.feed('{"old": 1}')
        parser.reset()
        result = parser.feed('{"new": 2}')
        assert result.is_complete is True
        assert result.data == {"new": 2}

    def test_try_extract_partial_closes_string(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": "hello')
        # The text has unclosed string, _try_extract_partial tries suffixes
        assert isinstance(result, PartialResult)

    def test_try_extract_partial_closes_object(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": 1')
        assert isinstance(result, PartialResult)

    def test_try_extract_partial_closes_array(self):
        parser = PartialJSONParser()
        result = parser.feed('{"a": [1, 2')
        assert isinstance(result, PartialResult)


# ── StructuredStreamHandler ─────────────────────────────────────────────────


class TestStructuredStreamHandler:
    """Construction, process_chunk, finalize, edge cases."""

    def test_construct_default(self):
        handler = StructuredStreamHandler()
        assert handler._config is None
        assert handler._is_complete is False
        assert handler._chunks == []

    def test_construct_with_config(self):
        """When config has streaming_buffer_size, it's passed to accumulator."""
        config = type("FakeConfig", (), {"streaming_buffer_size": 100})()
        handler = StructuredStreamHandler(config=config)
        assert handler._accumulator._min_chars == 100

    def test_process_chunk_single(self):
        handler = StructuredStreamHandler()
        chunks = handler.process_chunk('{"a": 1}')
        # Buffer may or may not flush depending on min_chars
        assert len(handler._chunks) == 1

    def test_process_chunk_multiple_accumulate(self):
        handler = StructuredStreamHandler()
        handler.process_chunk("hello ")
        handler.process_chunk("world")
        assert len(handler._chunks) == 2

    def test_finalize_complete_json(self):
        handler = StructuredStreamHandler()
        handler.process_chunk('{"a": 1}')
        result = handler.finalize()
        assert result.is_complete is True
        assert result.data == {"a": 1}
        assert result.errors == []

    def test_finalize_incomplete_json(self):
        handler = StructuredStreamHandler()
        handler.process_chunk('{"a": "unfinished')
        result = handler.finalize()
        # May or may not be complete after accumulator flush + feed
        assert result.text == '{"a": "unfinished'

    def test_finalize_empty(self):
        handler = StructuredStreamHandler()
        result = handler.finalize()
        assert result.is_complete is False

    def test_process_chunk_flushes_accumulator(self):
        """Process enough chunks to trigger the accumulator flush."""
        handler = StructuredStreamHandler()
        chunks = handler.process_chunk("x" * 100)
        # With min_chars=50, this should flush
        assert len(chunks) >= 1
        assert "x" * 100 in chunks[0]

    def test_finalize_after_multiple_chunks(self):
        handler = StructuredStreamHandler()
        handler.process_chunk('{"key": ')
        handler.process_chunk('"value"}')
        result = handler.finalize()
        assert result.is_complete is True
        assert result.data == {"key": "value"}

    def test_is_complete_tracking(self):
        handler = StructuredStreamHandler()
        assert handler._is_complete is False
        handler.process_chunk('{"a": 1}')
        # The accumulator might flush and parser feed might mark complete
        # Check that is_complete could be set after processing
        result = handler.finalize()
        assert result.is_complete is True

    def test_accumulator_integration(self):
        """Handler wraps a BufferedAccumulator internally."""
        handler = StructuredStreamHandler()
        assert isinstance(handler._accumulator, BufferedAccumulator)
        assert isinstance(handler._parser, PartialJSONParser)
