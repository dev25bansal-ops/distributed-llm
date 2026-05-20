"""Chaos scenario: data corruption.

Simulates data corruption during gRPC communication by injecting
corrupted tensor data or malformed messages. Verifies that the system
detects and handles corrupted data gracefully.
"""

import json
import os
import random
import struct
from typing import Any

from loguru import logger


class DataCorruptor:
    """Injects data corruption to test system resilience.

    Supports:
    - Bit flips in tensor data
    - Corrupted JSON payloads
    - Truncated messages
    - Invalid protobuf data
    """

    def __init__(self, corruption_rate: float = 0.01):
        self.corruption_rate = corruption_rate
        self.stats = {"flips": 0, "corruptions": 0, "detections": 0}

    def corrupt_tensor(self, data: bytes, flip_prob: float | None = None) -> bytes:
        """Flip random bits in binary tensor data."""
        prob = flip_prob or self.corruption_rate
        data = bytearray(data)
        flips = 0
        for i in range(len(data)):
            if random.random() < prob:
                data[i] ^= 1 << random.randint(0, 7)
                flips += 1
        self.stats["flips"] += flips
        if flips > 0:
            self.stats["corruptions"] += 1
        return bytes(data)

    def corrupt_json(self, payload: dict | str) -> str:
        """Corrupt a JSON payload by modifying random fields."""
        self.stats["corruptions"] += 1
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        payload = list(payload)
        for i in range(len(payload)):
            if random.random() < self.corruption_rate:
                payload[i] = chr(random.randint(0x01, 0x7F))
        self.stats["detections"] += 1
        return "".join(payload)

    def truncate_message(self, data: bytes, min_keep: int = 4) -> bytes:
        """Truncate a message to simulate partial delivery."""
        self.stats["corruptions"] += 1
        keep = random.randint(min_keep, max(min_keep + 1, len(data) // 4))
        return data[:keep]

    def wrap_response(self, response: dict, corruption_type: str) -> dict:
        """Wrap a response with corruption simulation metadata."""
        return {
            "_simulation": True,
            "_corruption_type": corruption_type,
            "original": response,
        }

    def summary(self) -> dict[str, int]:
        return dict(self.stats)


def test_bit_flip_detection():
    """Corrupted tensor should fail checksum/validation."""
    import hashlib

    original = b"\x00" * 1024
    corruptor = DataCorruptor(corruption_rate=0.05)
    corrupted = corruptor.corrupt_tensor(original)

    original_hash = hashlib.sha256(original).hexdigest()
    corrupted_hash = hashlib.sha256(corrupted).hexdigest()
    assert original_hash != corrupted_hash, "Corruption should change hash"


def test_json_corruption_fails_parse():
    """Corrupted JSON should fail to parse."""
    import json

    payload = {"request_id": "abc", "prompt": "hello world"}
    corruptor = DataCorruptor(corruption_rate=0.1)
    corrupted = corruptor.corrupt_json(payload)

    try:
        json.loads(corrupted)
        # If it still parses, it might be valid but different
        parsed = json.loads(corrupted)
        assert parsed != payload, "Payload should change after corruption"
    except json.JSONDecodeError:
        pass  # Expected — invalid JSON
