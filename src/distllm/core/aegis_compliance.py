"""SOC 2 / HIPAA compliance and model watermarking.

Provides an integrated compliance framework with four main components:

  - **AuditTrail** -- Immutable audit log backed by SQLite or append-only JSONL.
    Entries are never modified after creation.

  - **ModelWatermark** -- Imperceptible model watermarking for provenance.
    Encodes a secret into the LSBs of a small fraction of float32 weights.
    Requires PyTorch (optional dependency).

  - **ComplianceRule** -- HIPAA (encrypt PHI, log PHI access, restrict
    retention) and SOC 2 (access controls, change management, monitoring)
    rule checking.

  - **Aegis** -- Unified facade combining all three.  Single entry point for
    compliance checks, audit logging, and status reporting.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import sqlite3
import struct
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Aegis",
    "AuditTrail",
    "ComplianceError",
    "ComplianceRule",
    "HIPAA_RULES",
    "ModelWatermark",
    "SOC2_RULES",
    "WatermarkError",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WATERMARK_SALT = b"distllm-aegis-watermark-v2"
_WATERMARK_ATTR = "_aegis_watermark_meta"
_DEFAULT_WATERMARK_FRACTION = 0.0001  # 0.01% of parameters
_DEFAULT_FLUSH_INTERVAL = 10.0
_DEFAULT_MAX_ENTRIES = 100_000


# ---------------------------------------------------------------------------
# Standard compliance rules
# ---------------------------------------------------------------------------

HIPAA_RULES: dict[str, str] = {
    "encrypt_phi": "Encrypt all Protected Health Information at rest and in transit",
    "log_phi_access": "Log every access to PHI with user, timestamp, and reason",
    "restrict_retention": "Restrict data retention to minimum necessary duration",
    "phi_acl": "Restrict PHI access to authorized personnel only",
    "phi_audit": "Maintain audit records of PHI access for minimum 6 years",
}

SOC2_RULES: dict[str, str] = {
    "access_control": "Implement least-privilege access controls with periodic review",
    "change_management": "Document and approve all system changes before deployment",
    "monitoring": "Continuous monitoring of system activity and security events",
    "encryption_in_transit": "Encrypt all data in transit using TLS 1.2+",
    "incident_response": "Documented incident response plan with defined escalation paths",
}

_STANDARD_RULES: dict[str, str] = {}
_STANDARD_RULES.update(HIPAA_RULES)
_STANDARD_RULES.update(SOC2_RULES)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ComplianceError(Exception):
    """Raised when a compliance operation fails."""


class WatermarkError(ComplianceError):
    """Raised when watermark embedding, extraction, or detection fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single immutable audit trail entry."""

    entry_id: str
    event_type: str
    user_id: str
    resource: str
    action: str
    result: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ae_event_type ON audit_entries(event_type);
CREATE INDEX IF NOT EXISTS idx_ae_user_id ON audit_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_ae_timestamp ON audit_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_ae_resource ON audit_entries(resource);
CREATE INDEX IF NOT EXISTS idx_ae_action ON audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_ae_result ON audit_entries(result);
"""


class AuditTrail:
    '''Immutable audit log backed by SQLite or append-only JSONL file.

    Thread-safe.  Entries are append-only -- once recorded they are never
    modified or deleted.
    '''

    def __init__(
        self,
        *,
        backend: Literal['sqlite', 'file'] = 'sqlite',
        path: str = 'aegis_audit.db',
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self._backend = backend
        self._path = path
        self._max_entries = max_entries
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._initialized = False

        if backend == 'sqlite':
            self._conn: sqlite3.Connection | None = None
            self._cache: list[AuditEntry] = []
        else:
            self._buffer: list[str] = []
            self._entries: list[AuditEntry] = []
            self._last_flush: float = 0.0

    def initialize(self) -> None:
        '''Initialise the audit trail backend.  Idempotent.'''
        if self._initialized:
            return

        if self._backend == 'sqlite':
            conn = sqlite3.connect(
                self._path, timeout=30, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.executescript(_SQLITE_SCHEMA)
            self._conn = conn
        else:
            log_dir = os.path.dirname(self._path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        self._initialized = True
        logger.info(
            'AuditTrail initialised (backend={}, path={})',
            self._backend,
            self._path,
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    def record(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str = '',
        timestamp: float | None = None,
        **metadata: Any,
    ) -> str:
        '''Record an immutable audit entry.'''
        self._ensure_initialized()

        entry_id = uuid.uuid4().hex[:16]
        ts = timestamp if timestamp is not None else time.time()
        meta_str = json.dumps(metadata, default=str)

        entry = AuditEntry(
            entry_id=entry_id,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            timestamp=ts,
            metadata=metadata,
        )

        with self._lock:
            if self._backend == 'sqlite':
                assert self._conn is not None
                self._conn.execute(
                    '''INSERT INTO audit_entries
                       (entry_id, event_type, user_id, resource, action,
                        result, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (entry_id, event_type, user_id, resource, action,
                     result, ts, meta_str),
                )
                self._conn.commit()
                self._cache.append(entry)
                if len(self._cache) > self._max_entries:
                    self._cache = self._cache[-self._max_entries:]
            else:
                self._entries.append(entry)
                self._buffer.append(json.dumps(entry.to_dict(), default=str))
                if len(self._entries) > self._max_entries:
                    self._entries = self._entries[-self._max_entries:]
                self._maybe_flush()

        logger.debug(
            'Audit entry {}: {} {} {} -> {}',
            entry_id, user_id, action, resource, result,
        )
        return entry_id

    def query(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        '''Query audit entries with optional filters.'''
        self._ensure_initialized()
        filters = filters or {}

        if self._backend == 'sqlite':
            return self._query_sqlite(filters, limit=limit, offset=offset)
        return self._query_memory(filters, limit=limit, offset=offset)

    def _query_sqlite(
        self, filters: dict[str, Any], *, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        clauses: list[str] = []
        params: list[Any] = []

        for key in ('event_type', 'user_id', 'resource', 'action', 'result'):
            if key in filters:
                clauses.append(f'{key} = ?')
                params.append(filters[key])

        if 'start_time' in filters:
            clauses.append('timestamp >= ?')
            params.append(filters['start_time'])
        if 'end_time' in filters:
            clauses.append('timestamp <= ?')
            params.append(filters['end_time'])

        where = ' AND '.join(clauses) if clauses else '1=1'
        sql = f'SELECT * FROM audit_entries WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d['metadata'] = json.loads(d.get('metadata', '{}'))
            d.pop('id', None)
            results.append(d)
        return results

    def _query_memory(
        self, filters: dict[str, Any], *, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        with self._lock:
            matched = list(self._entries)

        for key in ('event_type', 'user_id', 'resource', 'action', 'result'):
            if key in filters:
                val = filters[key]
                matched = [e for e in matched if getattr(e, key) == val]

        if 'start_time' in filters:
            matched = [e for e in matched if e.timestamp >= filters['start_time']]
        if 'end_time' in filters:
            matched = [e for e in matched if e.timestamp <= filters['end_time']]

        matched.sort(key=lambda e: e.timestamp, reverse=True)
        matched = matched[offset:offset + limit]
        return [e.to_dict() for e in matched]

    def export(self, fmt: str = 'json') -> str:
        '''Export the audit log as a JSON string.'''
        self._ensure_initialized()
        if fmt != 'json':
            raise ValueError(f'Unsupported export format: {fmt!r}')

        if self._backend == 'sqlite':
            return self._export_sqlite()
        return self._export_memory()

    def _export_sqlite(self) -> str:
        assert self._conn is not None
        with self._lock:
            rows = self._conn.execute(
                'SELECT * FROM audit_entries ORDER BY timestamp'
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d['metadata'] = json.loads(d.get('metadata', '{}'))
            d.pop('id', None)
            entries.append(d)
        return json.dumps(entries, default=str, indent=2)

    def _export_memory(self) -> str:
        with self._lock:
            entries = [e.to_dict() for e in self._entries]
        return json.dumps(entries, default=str, indent=2)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise ComplianceError(
                'AuditTrail is not initialised. Call initialize() first.'
            )

    def _maybe_flush(self) -> None:
        now = time.time()
        if now - self._last_flush < self._flush_interval and len(self._buffer) < 100:
            return
        self.flush()

    def flush(self) -> None:
        '''Flush buffered entries to the JSONL file on disk.'''
        if self._backend != 'file':
            return
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()
        if not lines:
            return
        try:
            with open(self._path, 'a', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + chr(10))
        except OSError as e:
            logger.warning('Failed to flush audit log: {}', e)

    def close(self) -> None:
        '''Close the audit trail and release resources.'''
        if self._backend == 'sqlite':
            with self._lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
        else:
            self.flush()
        self._initialized = False
        logger.info('AuditTrail closed')


# ---------------------------------------------------------------------------
# ModelWatermark
# ---------------------------------------------------------------------------


class ModelWatermark:
    '''Watermark PyTorch models by encoding provenance data into weight LSBs.

    Embeds a secret message into the least-significant bits of a small
    fraction of float32 model parameters.  The modification is
    imperceptible and enables ownership verification.

    .. note::

        Requires PyTorch.  Raises ComplianceError if torch is not installed.
    '''

    def __init__(self, target_fraction: float = _DEFAULT_WATERMARK_FRACTION) -> None:
        if not HAS_TORCH:
            raise ComplianceError(
                'ModelWatermark requires PyTorch. Install with: pip install torch'
            )
        self._target_frac = target_fraction

    def embed(self, module: Any, secret: str) -> Any:
        '''Embed *secret* into the LSBs of *module* float32 weights.

        Returns the same module (modified in-place).
        '''
        if not secret:
            raise WatermarkError('Cannot embed an empty secret')

        msg_bytes = secret.encode('utf-8')
        num_bits = (len(msg_bytes) + 32) * 8  # message + SHA-256 tag

        params = self._collect_params(module)
        total_elements = sum(sz for _, _, sz in params)

        if total_elements == 0:
            raise WatermarkError('Module has no float32 parameters to watermark')

        needed = max(num_bits, int(total_elements * self._target_frac))
        if needed > total_elements:
            raise WatermarkError(
                f'Need {needed} parameter elements but module only has {total_elements}'
            )

        prefix = self._build_prefix(params)
        indices = self._select_unique_indices(total_elements, needed)

        auth_tag = hashlib.sha256(secret.encode()).digest()
        payload = msg_bytes + auth_tag

        with torch.no_grad():
            for bit_idx, global_idx in enumerate(indices[:num_bits]):
                flat, local = self._resolve_index(params, prefix, global_idx)
                byte_idx = bit_idx // 8
                bit_pos = bit_idx % 8
                bit = (payload[byte_idx] >> bit_pos) & 1
                self._set_lsb(flat, local, bit)

        meta = json.dumps({'len': len(msg_bytes), 'frac': self._target_frac})
        object.__setattr__(module, _WATERMARK_ATTR, meta)

        logger.info(
            'Watermark embedded ({} bytes, {} params modified)',
            len(msg_bytes), num_bits,
        )
        return module

    def detect(self, model: Any, secret: str) -> bool:
        '''Check if *model* contains a watermark matching *secret*.'''
        try:
            msg = self.extract(model)
            return msg == secret
        except WatermarkError:
            return False

    def extract(self, model: Any) -> str:
        '''Extract the watermark message from *model*.'''
        raw_meta = getattr(model, _WATERMARK_ATTR, None)
        if raw_meta is None:
            raise WatermarkError(
                'No watermark metadata found on module. '
                'Was the model watermarked using ModelWatermark.embed()?'
            )

        meta = json.loads(raw_meta)
        msg_len = meta['len']
        total_bytes = msg_len + 32  # message + SHA-256
        num_bits = total_bytes * 8

        params = self._collect_params(model)
        total_elements = sum(sz for _, _, sz in params)

        if total_elements == 0:
            raise WatermarkError('Module has no float32 parameters')

        needed = max(num_bits, int(total_elements * self._target_frac))
        prefix = self._build_prefix(params)
        indices = self._select_unique_indices(total_elements, needed)

        recovered = bytearray(total_bytes)
        for bit_idx, global_idx in enumerate(indices[:num_bits]):
            flat, local = self._resolve_index(params, prefix, global_idx)
            byte_idx = bit_idx // 8
            bit_pos = bit_idx % 8
            recovered[byte_idx] |= self._get_lsb(flat, local) << bit_pos

        extracted_msg = bytes(recovered[:msg_len]).decode('utf-8', errors='replace')
        stored_tag = bytes(recovered[msg_len:])
        expected_tag = hashlib.sha256(extracted_msg.encode()).digest()

        if stored_tag != expected_tag:
            raise WatermarkError(
                'Watermark integrity check failed -- '
                'message may be corrupted or the model was modified'
            )

        logger.debug('Watermark extracted ({} bytes)', msg_len)
        return extracted_msg

    @staticmethod
    def _collect_params(module: Any) -> list[tuple[str, Any, int]]:
        '''Collect float32 grad params sorted by name.'''
        result: list[tuple[str, Any, int]] = []
        for name, param in sorted(module.named_parameters(), key=lambda x: x[0]):
            if param.requires_grad and param.dtype == torch.float32:
                result.append((name, param.data.flatten(), param.data.numel()))
        return result

    @staticmethod
    def _build_prefix(params: list[tuple[str, Any, int]]) -> list[int]:
        prefix: list[int] = []
        cum = 0
        for _, _, sz in params:
            cum += sz
            prefix.append(cum)
        return prefix

    @staticmethod
    def _resolve_index(
        params: list[tuple[str, Any, int]], prefix: list[int], global_idx: int
    ) -> tuple[Any, int]:
        tensor_idx = bisect.bisect_right(prefix, global_idx)
        local_offset = global_idx - (prefix[tensor_idx - 1] if tensor_idx > 0 else 0)
        return params[tensor_idx][1], local_offset

    @staticmethod
    def _select_unique_indices(total: int, count: int) -> list[int]:
        '''Select *count* unique deterministic indices from [0, total).'''
        indices: list[int] = []
        seen: set[int] = set()
        i = 0
        max_iter = count * 10
        while len(indices) < count:
            if i > max_iter:
                for j in range(total):
                    if j not in seen:
                        seen.add(j)
                        indices.append(j)
                        if len(indices) >= count:
                            break
                break
            h = hashlib.sha256(_WATERMARK_SALT + str(i).encode()).hexdigest()
            idx = int(h, 16) % total
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
            i += 1
        return indices

    @staticmethod
    def _set_lsb(flat_tensor: Any, index: int, bit: int) -> None:
        val = flat_tensor[index].item()
        as_int = struct.unpack('I', struct.pack('f', val))[0]
        as_int = (as_int & ~1) | bit
        flat_tensor[index] = struct.unpack('f', struct.pack('I', as_int))[0]

    @staticmethod
    def _get_lsb(flat_tensor: Any, index: int) -> int:
        val = flat_tensor[index].item()
        as_int = struct.unpack('I', struct.pack('f', val))[0]
        return as_int & 1


# ---------------------------------------------------------------------------
# ComplianceRule
# ---------------------------------------------------------------------------


class ComplianceRule:
    '''HIPAA and SOC 2 compliance rule checking.

    Maintains a set of compliance rules and provides a check()
    method that evaluates an operation against all enabled rules.
    '''

    _PHI_RESOURCES = frozenset({
        'phi', 'phi_data', 'patient_record', 'health_record', 'medical_data',
        'ehr', 'emr', 'protected_health', 'hipaa_data',
    })
    _SENSITIVE_RESOURCES = frozenset({
        'user_data', 'credentials', 'api_keys', 'tokens', 'secrets',
    })
    _WRITE_ACTIONS = frozenset({
        'write', 'update', 'delete', 'create', 'modify',
    })

    def __init__(self, rules: dict[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._rules: dict[str, str] = {}
        self._active: dict[str, bool] = {}

        if rules is not None:
            for rule_id, desc in rules.items():
                self.add_rule(rule_id, desc)

    def add_rule(self, rule_id: str, description: str) -> None:
        '''Add a compliance rule.'''
        with self._lock:
            self._rules[rule_id] = description
            self._active.setdefault(rule_id, True)

    def remove_rule(self, rule_id: str) -> None:
        '''Remove a compliance rule entirely.'''
        with self._lock:
            self._rules.pop(rule_id, None)
            self._active.pop(rule_id, None)

    def activate_rule(self, rule_id: str) -> None:
        with self._lock:
            if rule_id in self._rules:
                self._active[rule_id] = True

    def deactivate_rule(self, rule_id: str) -> None:
        with self._lock:
            if rule_id in self._rules:
                self._active[rule_id] = False

    @property
    def active_rules(self) -> dict[str, str]:
        with self._lock:
            return {
                rid: desc for rid, desc in self._rules.items()
                if self._active.get(rid, True)
            }

    def check(self, operation: dict[str, Any]) -> tuple[bool, list[str]]:
        '''Check an operation against all active compliance rules.

        Args:
            operation: Dict describing the operation.  Supported keys:
                action, resource, user_role, encryption, data_type,
                retention_days, audit_logged, change_approved,
                monitoring_enabled, tls_version, incident,
                incident_reported.

        Returns:
            Tuple of (compliant, violated_rules).
        '''
        action = operation.get('action', '')
        resource = operation.get('resource', '')
        violated: list[str] = []

        with self._lock:
            active = {
                rid: desc for rid, desc in self._rules.items()
                if self._active.get(rid, True)
            }

        is_phi = self._is_phi_resource(resource) or operation.get('data_type') == 'phi'
        is_sensitive = is_phi or resource in self._SENSITIVE_RESOURCES
        is_write = action.lower() in self._WRITE_ACTIONS

        # HIPAA rules
        if 'encrypt_phi' in active and is_phi:
            if not operation.get('encryption', False):
                violated.append(active['encrypt_phi'])

        if 'log_phi_access' in active and is_phi:
            if operation.get('audit_logged') is False:
                violated.append(active['log_phi_access'])

        if 'restrict_retention' in active and is_phi:
            retention = operation.get('retention_days')
            if retention is not None and retention > 365:
                violated.append(active['restrict_retention'])

        if 'phi_acl' in active and is_phi:
            role = operation.get('user_role', '')
            if role not in ('admin', 'practitioner', 'auditor'):
                violated.append(active['phi_acl'])

        if 'phi_audit' in active and is_phi:
            if operation.get('audit_logged') is False:
                violated.append(active['phi_audit'])

        # SOC 2 rules
        if 'access_control' in active and is_sensitive and is_write:
            role = operation.get('user_role', '')
            if role not in ('admin', 'owner'):
                violated.append(active['access_control'])

        if 'change_management' in active and is_write:
            if not operation.get('change_approved', False):
                violated.append(active['change_management'])

        if 'monitoring' in active:
            if operation.get('monitoring_enabled') is False:
                violated.append(active['monitoring'])

        if 'encryption_in_transit' in active and is_sensitive:
            tls = operation.get('tls_version')
            if tls is None or (isinstance(tls, str) and self._parse_tls(tls) < 1.2):
                violated.append(active['encryption_in_transit'])

        if 'incident_response' in active and operation.get('incident', False):
            if not operation.get('incident_reported', False):
                violated.append(active['incident_response'])

        return (len(violated) == 0, violated)

    @staticmethod
    def _is_phi_resource(resource: str) -> bool:
        resource_lower = resource.lower()
        return any(term in resource_lower for term in ComplianceRule._PHI_RESOURCES)

    @staticmethod
    def _parse_tls(version: str) -> float:
        try:
            return float(version.strip())
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# Aegis -- unified compliance facade
# ---------------------------------------------------------------------------


class Aegis:
    '''Unified compliance system combining audit, watermarking, and rule checking.

    Orchestrates AuditTrail, ModelWatermark, and ComplianceRule into a
    single facade.
    '''

    def __init__(
        self,
        audit_backend: Literal['sqlite', 'file'] = 'sqlite',
        audit_path: str = 'aegis_audit.db',
    ) -> None:
        self._audit = AuditTrail(backend=audit_backend, path=audit_path)
        self._watermark: ModelWatermark | None = (
            ModelWatermark() if HAS_TORCH else None
        )
        self._compliance = ComplianceRule(rules=dict(_STANDARD_RULES))
        self._started = False
        self._start_time: float = 0.0

    @property
    def audit(self) -> AuditTrail:
        return self._audit

    @property
    def watermark(self) -> ModelWatermark | None:
        return self._watermark

    @property
    def compliance(self) -> ComplianceRule:
        return self._compliance

    def start(self) -> None:
        '''Initialise the compliance system.'''
        if self._started:
            raise ComplianceError('Aegis is already started')

        self._audit.initialize()
        self._start_time = time.time()
        self._started = True

        logger.info(
            'Aegis compliance system started '
            '(audit={}, watermark={}, rules={})',
            self._audit.initialized,
            self._watermark is not None,
            len(self._compliance.active_rules),
        )

    def stop(self) -> None:
        '''Shut down the compliance system.'''
        if not self._started:
            return
        self._audit.close()
        self._started = False
        logger.info('Aegis compliance system stopped')

    def check_request(
        self,
        user: str,
        resource: str,
        action: str,
        *,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> tuple[bool, dict[str, Any]]:
        '''Check a user request against compliance rules.

        Evaluates the request against active compliance rules and
        records an audit trail entry regardless of outcome.

        Returns:
            Tuple of (allowed, audit_entry) where audit_entry is a
            dict containing the entry ID, user, resource, action,
            compliance result, and violations.
        '''
        if not self._started:
            raise ComplianceError('Aegis is not started. Call start() first.')

        op: dict[str, Any] = {'action': action, 'resource': resource}
        op.update(extra)

        compliant, violations = self._compliance.check(op)

        audit_meta: dict[str, Any] = {
            'violations': violations,
            'compliant': compliant,
        }
        if metadata:
            audit_meta.update(metadata)

        entry_id = self._audit.record(
            event_type='compliance_check',
            user_id=user,
            resource=resource,
            action=action,
            result='allowed' if compliant else 'denied',
            **audit_meta,
        )

        entry = {
            'entry_id': entry_id,
            'user': user,
            'resource': resource,
            'action': action,
            'allowed': compliant,
            'violations': violations,
        }
        return (compliant, entry)

    def stats(self) -> dict[str, Any]:
        '''Return compliance system status and statistics.'''
        if not self._started:
            return {'started': False}

        active = self._compliance.active_rules
        hipaa_active = sum(1 for rid in active if rid in HIPAA_RULES)
        soc2_active = sum(1 for rid in active if rid in SOC2_RULES)

        return {
            'started': True,
            'uptime_seconds': time.time() - self._start_time,
            'audit_backend': self._audit._backend,
            'audit_path': self._audit._path,
            'audit_initialized': self._audit.initialized,
            'compliance_rules': len(active),
            'hipaa_rules_active': hipaa_active,
            'soc2_rules_active': soc2_active,
            'watermark_available': self._watermark is not None,
            'total_rules': len(HIPAA_RULES) + len(SOC2_RULES),
        }
