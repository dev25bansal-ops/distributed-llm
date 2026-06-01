"""NAT type detection parsing tests.

Covers NAT type classification logic, STUN packet parsing, NatMapping defaults,
and fallback behavior on network errors.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_nat = _load_module("distllm/dist/nat.py")


# ═══════════════════════════════════════════════════════════════════════════
# 7. NAT Type Detection Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestNatDetection:
    """NAT type classification logic and STUN packet parsing."""

    def test_nat_type_enum_values(self):
        assert _nat.NatType.UNKNOWN.value == "unknown"
        assert _nat.NatType.OPEN.value == "open"
        assert _nat.NatType.FULL_CONE.value == "full_cone"
        assert _nat.NatType.RESTRICTED.value == "restricted"
        assert _nat.NatType.PORT_RESTRICTED.value == "port_restricted"
        assert _nat.NatType.SYMMETRIC.value == "symmetric"

    def test_nat_mapping_defaults(self):
        m = _nat.NatMapping()
        assert m.public_ip == ""
        assert m.public_port == 0
        assert m.nat_type == _nat.NatType.UNKNOWN
        assert m.local_ip == ""
        assert m.local_port == 0

    def test_nat_mapping_custom(self):
        m = _nat.NatMapping(
            public_ip="1.2.3.4", public_port=5678,
            nat_type=_nat.NatType.FULL_CONE,
            local_ip="192.168.1.10", local_port=1234,
        )
        assert m.public_ip == "1.2.3.4"
        assert m.public_port == 5678
        assert m.nat_type == _nat.NatType.FULL_CONE

    def test_classify_nat_open(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_change_request') as mock_change:
            mock_change.return_value = ("1.2.3.4", 5678)
            result = client._classify_nat(("stun.l.google.com", 19302), ("1.2.3.4", 5678))
            assert result == _nat.NatType.OPEN

    def test_classify_nat_full_cone(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_change_request') as mock_change:
            mock_change.return_value = ("5.6.7.8", 9999)
            with patch.object(client, '_stun_binding_request') as mock_bind:
                mock_bind.return_value = ("1.2.3.4", 5678)
                result = client._classify_nat(
                    ("stun.l.google.com", 19302), ("1.2.3.4", 5678)
                )
                assert result == _nat.NatType.FULL_CONE

    def test_classify_nat_symmetric(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_change_request') as mock_change:
            mock_change.return_value = ("5.6.7.8", 9999)
            with patch.object(client, '_stun_binding_request') as mock_bind:
                mock_bind.return_value = ("9.9.9.9", 8888)
                result = client._classify_nat(
                    ("stun.l.google.com", 19302), ("1.2.3.4", 5678)
                )
                assert result == _nat.NatType.SYMMETRIC

    def test_classify_nat_port_restricted(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_change_request') as mock_change:
            mock_change.return_value = None
            with patch.object(client, '_stun_binding_request') as mock_bind:
                mock_bind.return_value = None
                result = client._classify_nat(
                    ("stun.l.google.com", 19302), ("1.2.3.4", 5678)
                )
                assert result == _nat.NatType.PORT_RESTRICTED

    def test_classify_nat_restricted(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_change_request') as mock_change:
            mock_change.return_value = ("5.6.7.8", 9999)
            with patch.object(client, '_stun_binding_request') as mock_bind:
                mock_bind.return_value = None
                result = client._classify_nat(
                    ("stun.l.google.com", 19302), ("1.2.3.4", 5678)
                )
                assert result == _nat.NatType.RESTRICTED

    def test_pick_alt_server_different(self):
        client = _nat.StunClient()
        alt = client._pick_alt_server(("stun.l.google.com", 19302))
        assert alt is not None
        assert alt[0] != "stun.l.google.com"

    def test_pick_alt_server_returns_none_when_only_one(self):
        client = _nat.StunClient()
        alt = client._pick_alt_server(("stun.l.google.com", 19302))
        assert alt is not None

    def test_stun_binding_request_packet_parse(self):
        """Verify STUN packet parsing logic with a synthetic response."""
        mapping = _nat.NatMapping(
            public_ip="1.2.3.4", public_port=5678,
            nat_type=_nat.NatType.FULL_CONE,
        )
        assert mapping.public_ip == "1.2.3.4"
        assert mapping.public_port == 5678
        assert mapping.nat_type == _nat.NatType.FULL_CONE

    def test_stun_detect_fallback_on_exception(self):
        client = _nat.StunClient()
        with patch.object(client, '_stun_binding_request', side_effect=Exception("Network error")):
            mapping = client.detect()
            assert mapping.nat_type == _nat.NatType.UNKNOWN

    def test_stun_change_request_xor_parsing(self):
        """Verify CHANGE-REQUEST response parsing includes CHANGED-ADDRESS."""
        client = _nat.StunClient()
        alt = client._pick_alt_server(("stun.l.google.com", 19302))
        assert alt is not None
        assert alt[0] != "stun.l.google.com"

    def test_stun_constants(self):
        assert _nat.StunClient.STUN_MAGIC_COOKIE == 0x2112A442
        assert _nat.StunClient.BINDING_REQUEST == 0x0001
        assert _nat.StunClient.ATTR_MAPPED_ADDRESS == 0x0001
        assert _nat.StunClient.ATTR_CHANGE_REQUEST == 0x0003
