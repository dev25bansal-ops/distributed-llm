from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DisaggPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETE = "complete"


class PoolStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    DRAINING = "draining"


@dataclass
class PrefillRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_tokens: list[int] = field(default_factory=list)
    max_new_tokens: int = 256
    adapter_id: str | None = None
    priority: int = 2
    created_at: float = field(default_factory=time.time)


@dataclass
class PrefillResult:
    request_id: str
    kv_cache: Any
    prompt_len: int
    first_token: int | None
    prefill_time_ms: float
    prefill_node_id: str


@dataclass
class DecodeRequest:
    request_id: str
    input_token: int
    kv_cache: Any
    position: int
    adapter_id: str | None = None


@dataclass
class PoolNode:
    node_id: str
    host: str
    port: int
    capacity: int = 0
    current_load: int = 0
    status: PoolStatus = PoolStatus.ACTIVE
    metrics: dict[str, float] = field(default_factory=dict)
