"""Tests for CacheMigrator (cross-cluster KV cache transfer)."""

from unittest.mock import MagicMock, patch
import json

import pytest

from distllm.core.cache_migration import CacheMigrator


class TestCacheMigrator:
    def make_response(self, status=200, data=None):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = json.dumps(data or {}).encode()
        return mock_resp

    def test_migrate_no_transport_returns_false(self):
        m = CacheMigrator()
        result = m.migrate_cache("src", "dst", ["hash1"])
        assert result is False

    def test_migrate_all_success(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.return_value = {"data": "value"}
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            result = m.migrate_cache("http://src", "http://dst", ["hash1", "hash2"])

        assert result is True
        assert transport.request_kv_cache.call_count == 2
        assert mock_open.call_count == 2

    def test_migrate_source_missing_skips(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.return_value = None
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            result = m.migrate_cache("http://src", "http://dst", ["hash1"])

        assert result is False
        mock_open.assert_not_called()

    def test_migrate_partial_failure(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.return_value = {"data": "v"}
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(500)
            result = m.migrate_cache("http://src", "http://dst", ["hash1"])

        assert result is False

    def test_migrate_some_source_missing(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.side_effect = [{"data": "v"}, None]
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            result = m.migrate_cache("http://src", "http://dst", ["hash1", "hash2"])

        assert result is False
        assert mock_open.call_count == 1

    def test_warm_cache_all_success(self):
        m = CacheMigrator()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            result = m.warm_cache_on_cluster("http://cluster", ["hash1", "hash2"])

        assert result is True
        assert mock_open.call_count == 2

    def test_warm_cache_some_fail(self):
        m = CacheMigrator()
        with patch("urllib.request.urlopen") as mock_open:
            mock = mock_open.return_value.__enter__.return_value
            mock.status = 500
            result = m.warm_cache_on_cluster("http://cluster", ["hash1", "hash2"])

        assert result is False

    def test_warm_cache_empty_list(self):
        m = CacheMigrator()
        with patch("urllib.request.urlopen") as mock_open:
            result = m.warm_cache_on_cluster("http://cluster", [])

        assert result is True
        mock_open.assert_not_called()

    def test_migrate_sends_correct_payload(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.return_value = {"data": "kv_val"}
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            m.migrate_cache("http://src", "http://dst:8080", ["hash1"])

        call_args = mock_open.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.full_url == "http://dst:8080/api/v1/cache/warm"
        assert "prefix_hash" in req.data.decode()
        assert "kv_data" in req.data.decode()

    def test_warm_sends_correct_payload(self):
        m = CacheMigrator()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            m.warm_cache_on_cluster("http://cluster", ["hash_abc"])

        call_args = mock_open.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.full_url == "http://cluster/api/v1/cache/warm"
        assert "prefix_hash" in req.data.decode()
        assert "action" in req.data.decode()

    def test_transport_network_error(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.side_effect = ConnectionError("timeout")
        m.set_transport(transport)

        with patch("urllib.request.urlopen") as mock_open:
            result = m.migrate_cache("http://src", "http://dst", ["hash1"])

        assert result is False
        mock_open.assert_not_called()

    def test_warm_network_error(self):
        m = CacheMigrator()
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = ConnectionError("refused")
            result = m.warm_cache_on_cluster("http://cluster", ["hash1"])

        assert result is False

    def test_set_transport_and_migrate(self):
        m = CacheMigrator()
        transport = MagicMock()
        transport.request_kv_cache.return_value = {"data": "v"}
        m.set_transport(transport)
        assert m._transport is transport

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value = self.make_response(200)
            result = m.migrate_cache("http://src", "http://dst", ["hash1"])

        assert result is True
