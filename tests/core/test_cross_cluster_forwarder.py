"""Tests for CrossClusterForwarder: request → remote cluster → response."""

import json
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import pytest

from distllm.core.cross_cluster_forwarder import CrossClusterForwarder


class MockCoordinatorHandler(BaseHTTPRequestHandler):
    """HTTP handler that simulates a remote coordinator response."""

    responses: dict[str, tuple[int, dict]] = {}
    received_requests: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(body)
        MockCoordinatorHandler.received_requests.append(data)

        path = urlparse(self.path).path
        key = f"POST {path}"
        status, resp_data = self.responses.get(key, (200, {"choices": [{"text": "mock"}]}))

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        key = f"GET {path}"
        status, resp_data = self.responses.get(key, (200, {"peers": []}))

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def coordinator_server():
    MockCoordinatorHandler.responses = {}
    MockCoordinatorHandler.received_requests = []

    server = HTTPServer(("127.0.0.1", 0), MockCoordinatorHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield port, f"http://127.0.0.1:{port}"
    server.shutdown()


class TestForwardRequest:
    def test_basic_forward(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /v1/completions"] = (
            200,
            {"choices": [{"text": "hello from remote"}]},
        )

        forwarder = CrossClusterForwarder()
        result = forwarder.forward_request(url, {"prompt": "say hi"})
        assert result["choices"][0]["text"] == "hello from remote"

    def test_forward_sends_x_forwarded(self, coordinator_server):
        port, url = coordinator_server
        forwarder = CrossClusterForwarder()
        forwarder.forward_request(url, {"prompt": "test"})
        assert len(MockCoordinatorHandler.received_requests) >= 1

    def test_forward_custom_timeout(self, coordinator_server):
        port, url = coordinator_server
        forwarder = CrossClusterForwarder()
        result = forwarder.forward_request(url, {"prompt": "hi"}, timeout_s=5)
        assert "choices" in result

    def test_forward_server_error(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /v1/completions"] = (500, {"error": "fail"})

        forwarder = CrossClusterForwarder(max_retries=1, retry_delay_s=0.1)
        with pytest.raises(Exception):
            forwarder.forward_request(url, {"prompt": "hi"})

    def test_forward_wrong_url(self):
        forwarder = CrossClusterForwarder(max_retries=0)
        with pytest.raises(Exception):
            forwarder.forward_request("http://127.0.0.1:1", {"prompt": "hi"})

    def test_forward_multiple_attempts_then_succeed(self, coordinator_server):
        port, url = coordinator_server
        attempt_count = {"n": 0}

        original_handler = MockCoordinatorHandler.do_POST

        def failing_then_succeed(self_handler, *args, **kwargs):
            attempt_count["n"] += 1
            if attempt_count["n"] <= 2:
                self_handler.send_response(503)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.end_headers()
                self_handler.wfile.write(b'{"error":"not ready"}')
            else:
                original_handler(self_handler, *args, **kwargs)

        MockCoordinatorHandler.do_POST = failing_then_succeed

        forwarder = CrossClusterForwarder(max_retries=3, retry_delay_s=0.05)
        result = forwarder.forward_request(url, {"prompt": "retry test"})

        assert "choices" in result
        assert attempt_count["n"] >= 3

    def test_forward_with_different_models(self, coordinator_server):
        port, url = coordinator_server
        forwarder = CrossClusterForwarder()
        result1 = forwarder.forward_request(url, {"model": "gpt-4", "prompt": "hi"})
        result2 = forwarder.forward_request(url, {"model": "llama", "prompt": "hello"})
        assert "choices" in result1
        assert "choices" in result2

    def test_forward_sets_headers_correctly(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /v1/completions"] = (200, {"ok": True})

        forwarded_headers = {}

        original_handler = MockCoordinatorHandler.do_POST

        def capture_headers(self_handler, *args, **kwargs):
            forwarded_headers["content-type"] = self_handler.headers.get("Content-Type", "")
            forwarded_headers["x-forwarded-from"] = self_handler.headers.get("X-Forwarded-From", "")
            original_handler(self_handler, *args, **kwargs)

        MockCoordinatorHandler.do_POST = capture_headers

        forwarder = CrossClusterForwarder()
        forwarder.forward_request(url, {"prompt": "test"})

        assert "application/json" in forwarded_headers.get("content-type", "")
        assert forwarded_headers.get("x-forwarded-from") == "federated"


class TestForwardKV:
    def test_forward_kv_cache(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /api/v1/cache/warm"] = (200, {"ok": True})

        forwarder = CrossClusterForwarder()
        result = forwarder.forward_kv_cache(url, "abc123", {"key": "val"})
        assert result is True

    def test_forward_kv_cache_rejected(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /api/v1/cache/warm"] = (400, {"error": "bad"})

        forwarder = CrossClusterForwarder()
        result = forwarder.forward_kv_cache(url, "abc", {})
        assert result is False

    def test_replicate_kv_batch(self, coordinator_server):
        port, url = coordinator_server
        MockCoordinatorHandler.responses["POST /api/v1/cache/warm"] = (200, {"ok": True})

        forwarder = CrossClusterForwarder()
        entries = [
            {"prefix_hash": "h1", "kv_data": {"k": "v1"}},
            {"prefix_hash": "h2", "kv_data": {"k": "v2"}},
        ]
        count = forwarder.replicate_kv_batch(entries, [url])
        assert count == 2

    def test_replicate_kv_batch_partial_fail(self, coordinator_server):
        port, url = coordinator_server

        # Use separate endpoints — one valid, one unreachable
        forwarder = CrossClusterForwarder()
        entries = [{"prefix_hash": "h1", "kv_data": {}}, {"prefix_hash": "h2", "kv_data": {}}]

        # First entry goes to valid server, second hits a closed port
        count = forwarder.replicate_kv_batch(entries, [url, "http://127.0.0.1:1"])
        assert count == 2


class TestForwardStreaming:
    def test_streaming_fallback_on_error(self):
        forwarder = CrossClusterForwarder()
        import asyncio

        async def run():
            results = []
            async for chunk in forwarder.forward_streaming(
                "http://127.0.0.1:1", {"prompt": "hi"}
            ):
                results.append(chunk)
            return results

        results = asyncio.run(run())
        assert len(results) == 1
        parsed = json.loads(results[0])
        assert "error" in parsed
