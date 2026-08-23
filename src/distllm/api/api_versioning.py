"""API versioning: Accept-Version header negotiation, deprecation scheduling,
field-level compatibility mapping, and a middleware that enforces the
version contract at the HTTP layer.

Architecture
------------
The versioning system has four components that work together:

1. **APIVersion** (enum) — Single source of truth for supported API versions.
   Versions are date-based strings (e.g. ``"2026-01"``) following the
   ``YYYY-MM`` convention, not ordinal integers.

2. **VersionNegotiator** — Determines which API version a request targets by
   checking, in order:
   a. The ``Accept-Version`` HTTP header
   b. The URL path prefix (``/v1/``, ``/v2/``)
   c. A configured default version (fallback)
   Returns the resolved version plus optional deprecation warnings.

3. **DeprecationSchedule** — Tracks sunset dates per version and provides
   migration path descriptions so clients know what to change.

4. **VersionCompatibilityLayer** — Bidirectional field-level mapping between
   versioned request/response shapes.  Allows the server to accept old-format
   payloads and return old-format responses even when the internal handler
   has moved to a newer schema.

5. **VersionMiddleware** (Starlette ``BaseHTTPMiddleware``) — Wires the
   negotiator, schedule, and compatibility layer into the request pipeline:
   - Extracts the version from every incoming request and stores it on
     ``request.state.api_version``.
   - Adds ``X-API-Version``, ``Sunset``, and ``X-API-Deprecation`` headers
     to every response.
   - Returns **410 Gone** for requests targeting a fully deprecated version
     whose sunset date has passed.
   - Optionally transforms request/response bodies through the compatibility
     layer when the request version does not match the handler's internal
     version.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


# ── Constants ─────────────────────────────────────────────────────────────────

SUNSET_HEADER = "Sunset"
DEPRECATION_HEADER = "X-API-Deprecation"
VERSION_HEADER = "X-API-Version"
ACCEPT_VERSION_HEADER = "Accept-Version"

# Default version used when no version indication is present.
DEFAULT_API_VERSION = "2026-07"

# Warning threshold: a version is considered "near sunset" when fewer than
# this many days remain before its sunset date.
NEAR_SUNSET_DAYS = 60


# ── Version Enum ──────────────────────────────────────────────────────────────


class APIVersion(str, Enum):
    """Canonical API version identifiers (date-based).

    Values are ``YYYY-MM`` strings so that ordering is lexicographically
    equivalent to chronological ordering.

    Add a new member each time the API surface changes in a way that is not
    backward-compatible.
    """

    V1 = "2026-01"
    V2 = "2026-07"

    @classmethod
    def from_string(cls, value: str) -> APIVersion | None:
        """Parse a string into an ``APIVersion`` member, returning ``None`` on
        mismatch."""
        value = value.strip()
        for member in cls:
            if member.value == value:
                return member
        return None

    @classmethod
    def from_path_prefix(cls, path: str) -> APIVersion | None:
        """Extract an API version from a URL path like ``/v1/...`` or ``/v2/...``.

        Returns ``None`` when the path carries no version prefix (e.g. unversioned
        endpoints such as ``/health`` or ``/metrics``).
        """
        match = re.match(r"^/(v\d+)/", path)
        if match is None:
            return None
        prefix = match.group(1)
        mapping: dict[str, APIVersion] = {
            "v1": APIVersion.V1,
            "v2": APIVersion.V2,
        }
        return mapping.get(prefix)

    def to_path_prefix(self) -> str:
        """Return the URL path prefix for this version (e.g. ``/v1``)."""
        ordinal: dict[APIVersion, str] = {
            APIVersion.V1: "v1",
            APIVersion.V2: "v2",
        }
        return f"/{ordinal[self]}"


# ── Version Negotiation Result ────────────────────────────────────────────────


@dataclass(frozen=True)
class VersionNegotiationResult:
    """The outcome of version negotiation for a single request.

    Attributes
    ----------
    version:
        The resolved ``APIVersion``, or ``None`` when the request could not
        be matched (caller should reject).
    source:
        How the version was determined: ``"header"``, ``"path"``, or
        ``"default"``.
    warnings:
        Human-readable deprecation warnings to emit as response headers.
    """

    version: APIVersion | None
    source: str
    warnings: list[str] = field(default_factory=list)


# ── Version Negotiator ────────────────────────────────────────────────────────


class VersionNegotiator:
    """Resolves the API version for an incoming request.

    Negotiation order:

    1. **Accept-Version header** — exact match against ``APIVersion`` values.
    2. **URL path prefix** — ``/v1/``, ``/v2/``.
    3. **Default** — configured fallback version.

    Parameters
    ----------
    default_version:
        Fallback version when no version indication is present.
    """

    def __init__(self, default_version: str = DEFAULT_API_VERSION) -> None:
        self._default = APIVersion.from_string(default_version)
        if self._default is None:
            logger.warning(
                "Default version %r is not a known APIVersion; falling back to %s",
                default_version,
                DEFAULT_API_VERSION,
            )
            self._default = APIVersion.from_string(DEFAULT_API_VERSION)
            assert self._default is not None

    # ── Public API ────────────────────────────────────────────────────────

    def negotiate(self, request: Request) -> VersionNegotiationResult:
        """Determine the API version for *request*.

        Returns a frozen ``VersionNegotiationResult``.  When the result's
        ``version`` is ``None`` the version could not be determined and the
        caller should respond with a client error.
        """
        # 1. Accept-Version header
        header_val = request.headers.get(ACCEPT_VERSION_HEADER)
        if header_val:
            parsed = self._parse_accept_version(header_val)
            if parsed is not None:
                return VersionNegotiationResult(
                    version=parsed,
                    source="header",
                    warnings=[],
                )
            # Header present but unparseable — warn but continue to path.
            logger.debug("Unparseable Accept-Version header: %r", header_val)

        # 2. URL path prefix
        path = request.url.path
        path_version = APIVersion.from_path_prefix(path)
        if path_version is not None:
            return VersionNegotiationResult(
                version=path_version,
                source="path",
                warnings=[],
            )

        # 3. Default fallback
        return VersionNegotiationResult(
            version=self._default,
            source="default",
            warnings=[],
        )

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_accept_version(header_value: str) -> APIVersion | None:
        """Try to parse a raw ``Accept-Version`` header into an ``APIVersion``.

        The header may be a bare date (``2026-01``) or a qualified form
        (``api=2026-01``).  Returns ``None`` when no member matches.
        """
        cleaned = header_value.strip()
        # Accept \"api=2026-01\" or \"2026-01\"
        for sep in ("api=", "version=", ""):
            candidate = cleaned.removeprefix(sep)
            candidate = candidate.strip()
            parsed = APIVersion.from_string(candidate)
            if parsed is not None:
                return parsed
        return None


# ── Deprecation Schedule ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MigrationStep:
    """A single action a client must take to migrate from one version to another.

    Attributes
    ----------
    description:
        Human-readable instruction describing what changed and what the
        client should do.
    field:
        The request or response field affected (e.g. ``"request.model"``).
        Empty string when the step is not field-specific.
    breaking:
        Whether this step represents a breaking change.
    """

    description: str
    field: str = ""
    breaking: bool = True


class DeprecationSchedule:
    """Manages sunset dates and migration paths for API versions.

    Parameters
    ----------
    sunset_dates:
        Mapping from ``APIVersion`` to an optional ``datetime.date``.
        When the date is ``None`` the version has no planned sunset.
    migration_steps:
        Mapping from ``(from_version, to_version)`` to a list of
        ``MigrationStep`` instances describing what the client must change.
    """

    def __init__(
        self,
        sunset_dates: dict[APIVersion, datetime.date | None] | None = None,
        migration_steps: dict[tuple[APIVersion, APIVersion], list[MigrationStep]]
        | None = None,
    ) -> None:
        self._sunset_dates: dict[APIVersion, datetime.date | None] = (
            dict(sunset_dates) if sunset_dates else {}
        )
        self._migration_steps: dict[
            tuple[APIVersion, APIVersion], list[MigrationStep]
        ] = (dict(migration_steps) if migration_steps else {})

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def default_schedule(cls) -> DeprecationSchedule:
        """Return the built-in deprecation schedule.

        Current schedule (as of 2026-07):
          - V1 (2026-01): sunset 2026-10-01, migrate to V2.
          - V2 (2026-07): no sunset (latest stable).
        """
        sunset = {
            APIVersion.V1: datetime.date(2026, 10, 1),
            APIVersion.V2: None,  # latest stable — no sunset
        }

        migration: dict[tuple[APIVersion, APIVersion], list[MigrationStep]] = {
            (APIVersion.V1, APIVersion.V2): [
                MigrationStep(
                    description=(
                        "Response object type changed from "
                        "``chat.completion`` to ``chat.completion.v2``."
                    ),
                    field="response.object",
                    breaking=True,
                ),
                MigrationStep(
                    description=(
                        "Response includes ``system_fingerprint`` field."
                    ),
                    field="response.system_fingerprint",
                    breaking=False,
                ),
                MigrationStep(
                    description=(
                        "Request field ``user`` is now validated as a "
                        "non-empty string."
                    ),
                    field="request.user",
                    breaking=False,
                ),
                MigrationStep(
                    description=(
                        "Removed deprecated ``logit_bias`` field; use "
                        "``logprobs`` instead."
                    ),
                    field="request.logit_bias",
                    breaking=True,
                ),
            ],
        }

        return cls(sunset_dates=sunset, migration_steps=migration)

    def is_deprecated(self, version: APIVersion) -> bool:
        """Return ``True`` when *version* has a sunset date in the past."""
        sunset = self._sunset_dates.get(version)
        if sunset is None:
            return False
        return datetime.date.today() >= sunset

    def is_near_sunset(self, version: APIVersion, days: int = NEAR_SUNSET_DAYS) -> bool:
        """Return ``True`` when *version* will sunset within *days* days."""
        sunset = self._sunset_dates.get(version)
        if sunset is None:
            return False
        remaining = (sunset - datetime.date.today()).days
        return 0 <= remaining <= days

    def days_until_sunset(self, version: APIVersion) -> int | None:
        """Return the number of days until *version* is sunset.

        Returns ``None`` when the version has no sunset date or the sunset
        has already passed.
        """
        sunset = self._sunset_dates.get(version)
        if sunset is None:
            return None
        remaining = (sunset - datetime.date.today()).days
        return remaining if remaining >= 0 else None

    def get_sunset_date(self, version: APIVersion) -> datetime.date | None:
        """Return the sunset date for *version*, or ``None``."""
        return self._sunset_dates.get(version)

    def sunset_header_value(self, version: APIVersion) -> str | None:
        """Return an HTTP-date string suitable for the ``Sunset`` header.

        Returns ``None`` when the version has no sunset.
        """
        sunset = self._sunset_dates.get(version)
        if sunset is None:
            return None
        # RFC 7231 / HTTP-date: IMF-fixdate
        return sunset.strftime("%a, %d %b %Y %H:%M:%S GMT")

    def get_migration_path(
        self,
        current_version: APIVersion,
        target_version: APIVersion,
    ) -> list[MigrationStep]:
        """Return the ordered list of migration steps to go from
        *current_version* to *target_version*.

        If no explicit path exists, returns a single generic step advising the
        client to consult the changelog.
        """
        path = self._migration_steps.get((current_version, target_version))
        if path is not None:
            return list(path)

        # Generic fallback.
        return [
            MigrationStep(
                description=(
                    f"No explicit migration path documented for "
                    f"{current_version.value} -> {target_version.value}. "
                    f"See https://docs.distllm.dev/api-versions for details."
                ),
                breaking=True,
            ),
        ]

    def auto_deprecate(self, now: datetime.date | None = None) -> list[APIVersion]:
        """Check all versions and return those whose sunset date has passed.

        This is intended to be called periodically (e.g. via a background
        task or during server startup) so that the server can log or reject
        versions whose grace period has fully elapsed.

        Parameters
        ----------
        now:
            Reference date.  Defaults to ``datetime.date.today()``.
        """
        if now is None:
            now = datetime.date.today()
        expired: list[APIVersion] = []
        for version, sunset in self._sunset_dates.items():
            if sunset is not None and now >= sunset:
                expired.append(version)
        return expired


# ── Version Compatibility Layer ───────────────────────────────────────────────


@dataclass
class FieldMapping:
    """Describes how a field is represented in two different API versions.

    Attributes
    ----------
    source_field:
        Dot-separated path of the field in the source version
        (e.g. ``"response.choices[].finish_reason"``).
    target_field:
        Dot-separated path of the field in the target version.
    transform:
        Optional callable that converts the source value to the target
        representation.  When ``None`` the value is copied as-is.
    """

    source_field: str
    target_field: str
    transform: Callable[[Any], Any] | None = None


class VersionCompatibilityLayer:
    """Bidirectional field-level mapping between API versions.

    The layer stores a set of ``FieldMapping`` instances for each version
    pair and direction (request / response).  When transforming, it walks
    the source dict and applies every registered mapping.

    Usage::

        layer = VersionCompatibilityLayer()
        layer.add_mapping(
            from_version=APIVersion.V1,
            to_version=APIVersion.V2,
            field_maps=[
                FieldMapping("request.logit_bias", "request.logprobs"),
                FieldMapping(
                    "response.object",
                    "response.object",
                    transform=lambda _: "chat.completion.v2",
                ),
            ],
        )

        # Upgrade an old V1 request to V2 format:
        v2_data = layer.transform_request(v1_data, APIVersion.V1, APIVersion.V2)

        # Downgrade a V2 response back to V1 format:
        v1_data = layer.transform_response(v2_data, APIVersion.V2, APIVersion.V1)
    """

    def __init__(self) -> None:
        # _request_mappings: (from_ver, to_ver) -> list[FieldMapping]
        self._request_mappings: dict[
            tuple[APIVersion, APIVersion], list[FieldMapping]
        ] = {}
        # _response_mappings: (from_ver, to_ver) -> list[FieldMapping]
        self._response_mappings: dict[
            tuple[APIVersion, APIVersion], list[FieldMapping]
        ] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def add_mapping(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
        field_maps: list[FieldMapping],
        *,
        direction: str = "both",
    ) -> None:
        """Register field mappings between two versions.

        Parameters
        ----------
        from_version:
            The source version of the data.
        to_version:
            The target version to transform into.
        field_maps:
            One or more ``FieldMapping`` descriptors.
        direction:
            ``"request"``, ``"response"``, or ``"both"`` (default).
        """
        if direction in ("request", "both"):
            self._request_mappings.setdefault(
                (from_version, to_version), []
            ).extend(field_maps)
        if direction in ("response", "both"):
            self._response_mappings.setdefault(
                (from_version, to_version), []
            ).extend(field_maps)

    # ── Transformation ────────────────────────────────────────────────────

    def transform_request(
        self,
        data: dict[str, Any],
        from_version: APIVersion,
        to_version: APIVersion,
    ) -> dict[str, Any]:
        """Transform a request body from *from_version* schema to
        *to_version* schema.

        Returns a new dict; the original is not mutated.
        """
        mappings = self._request_mappings.get((from_version, to_version))
        if mappings is None:
            return dict(data)
        return self._apply_mappings(data, mappings)

    def transform_response(
        self,
        data: dict[str, Any],
        from_version: APIVersion,
        to_version: APIVersion,
    ) -> dict[str, Any]:
        """Transform a response body from *from_version* schema to
        *to_version* schema.

        Returns a new dict; the original is not mutated.
        """
        mappings = self._response_mappings.get((from_version, to_version))
        if mappings is None:
            return dict(data)
        return self._apply_mappings(data, mappings)

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_mappings(
        data: dict[str, Any],
        mappings: list[FieldMapping],
    ) -> dict[str, Any]:
        """Apply a list of ``FieldMapping`` entries to a nested dict.

        The implementation uses dot-separated paths for field access.
        Array indexes are represented by ``[]`` segments (e.g.
        ``"choices[].finish_reason"``), meaning the mapping applies to
        every element of the array.
        """
        result = dict(data)

        for mapping in mappings:
            source_value = _get_nested(result, mapping.source_field)
            if source_value is _MISSING:
                continue

            transformed = (
                mapping.transform(source_value)
                if mapping.transform is not None
                else source_value
            )

            result = _set_nested(result, mapping.target_field, transformed)

        return result


# ── Sentinel for missing values ───────────────────────────────────────────────


class _Missing:
    """Sentinel to distinguish ``None`` from absent."""

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


def _get_nested(d: dict[str, Any], path: str) -> Any:
    """Retrieve a value from a nested dict using a dot-separated *path*.

    Supports array segments with ``[]`` to apply a lookup across all
    elements of a list (returns the first matched value, since transforms
    are uniform across elements).
    """
    parts = _split_path(path)
    current: Any = d
    for part in parts:
        if part == "[]" and isinstance(current, list):
            # Apply the remainder of the path to every element; return
            # the value from the first element that has it.
            remainder = ".".join(parts[parts.index("[]") + 1 :])
            for item in current:
                if isinstance(item, dict):
                    val = _get_nested(item, remainder)
                    if val is not _MISSING:
                        return val
            return _MISSING
        if isinstance(current, dict):
            current = current.get(part, _MISSING)
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _set_nested(d: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Set a value in a nested dict, returning a new dict (shallow copy at
    each level).  Does not mutate *d*.

    Supports array segments with ``[]`` to set the value on every element
    of a list.
    """
    parts = _split_path(path)
    if not parts:
        return d

    result = dict(d)
    current = result

    for i, part in enumerate(parts):
        if part == "[]" and isinstance(current, list):
            remainder = ".".join(parts[i + 1 :])
            new_list = []
            for item in current:
                if isinstance(item, dict):
                    new_list.append(
                        _set_nested(item, remainder, value)
                        if remainder
                        else value
                    )
                else:
                    new_list.append(item)
            # Walk back to parent to set the modified list.
            # For simplicity, we mutate result directly at the known path.
            # Full immutable walk is expensive; we accept shallow mutation
            # of the list container since it is local to this function and
            # never exposed.
            _set_nested_impl(result, parts[:i], new_list)
            return result

        if i == len(parts) - 1:
            # Last part — set the value.
            if isinstance(current, dict):
                current[part] = value
        else:
            if part not in current:
                current[part] = {}
            next_current = current.get(part)
            if isinstance(next_current, dict):
                current = next_current
            elif isinstance(next_current, list):
                # Enter list context for the next iteration.
                current = next_current
            else:
                # Intermediate missing — create dict.
                current[part] = {}
                current = current[part]

    return result


def _set_nested_impl(d: dict[str, Any], path_parts: list[str], value: Any) -> None:
    """Mutate *d* at *path_parts* — used internally by ``_set_nested`` for
    the array-branch case where immutability is already guaranteed by the
    caller."""
    current = d
    for i, part in enumerate(path_parts):
        if i == len(path_parts) - 1:
            current[part] = value
        else:
            next_val = current.get(part)
            if isinstance(next_val, dict):
                current = next_val
            else:
                current[part] = {}
                current = current[part]


def _split_path(path: str) -> list[str]:
    """Split a dot-separated path into segments, preserving ``[]`` as a
    single token."""
    segments: list[str] = []
    for segment in path.split("."):
        if "[]" in segment:
            # Handle both "choices[]" and "choices.[]"
            parts = segment.split("[]")
            for p in parts:
                if p:
                    segments.append(p)
                segments.append("[]")
            # Pop the trailing [] if the path ended with it and we added it
            if segments and segments[-1] == "[]" and not path.endswith("[]"):
                segments.pop()
        else:
            segments.append(segment)
    return segments


# ── Version Middleware ────────────────────────────────────────────────────────


class VersionMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces the API versioning contract.

    What it does for every request:

    1. Calls ``VersionNegotiator.negotiate()`` to determine the target
       version from the ``Accept-Version`` header or URL path.
    2. Stores the resolved version on ``request.state.api_version``.
    3. If the resolved version is fully deprecated (past sunset), responds
       with **410 Gone**.
    4. If the resolved version is *near* sunset (within 60 days), adds a
       ``Warning`` response header in addition to the standard deprecation
       headers.
    5. Appends ``X-API-Version``, ``Sunset``, and ``X-API-Deprecation``
       headers to every response.
    6. Optionally transforms request bodies from the negotiated version to
       the server's internal version, and response bodies back.

    Parameters
    ----------
    negotiator:
        The ``VersionNegotiator`` instance.  Defaults to a fresh negotiator.
    schedule:
        The ``DeprecationSchedule`` instance.  Defaults to
        ``DeprecationSchedule.default_schedule()``.
    compatibility_layer:
        Optional ``VersionCompatibilityLayer`` for request/response body
        transformation.
    internal_version:
        The server's internal version.  Requests at other versions are
        transformed to this version before reaching the route handler, and
        responses are transformed back.  Defaults to ``APIVersion.V2``.
    reject_deprecated:
        When ``True`` (default), requests to fully deprecated versions
        receive a 410 response.  Set to ``False`` to only warn (useful
        during a migration grace period).
    """

    def __init__(
        self,
        app: Any,
        negotiator: VersionNegotiator | None = None,
        schedule: DeprecationSchedule | None = None,
        compatibility_layer: VersionCompatibilityLayer | None = None,
        internal_version: APIVersion = APIVersion.V2,
        reject_deprecated: bool = True,
    ) -> None:
        super().__init__(app)
        self._negotiator = negotiator or VersionNegotiator()
        self._schedule = schedule or DeprecationSchedule.default_schedule()
        self._compat = compatibility_layer
        self._internal_version = internal_version
        self._reject_deprecated = reject_deprecated

    # ── Middleware Dispatch ───────────────────────────────────────────────

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Negotiate version.
        result = self._negotiator.negotiate(request)

        if result.version is None:
            return Response(
                status_code=400,
                content=(
                    "Could not determine API version. "
                    "Set the Accept-Version header or use a versioned path "
                    "(e.g. /v1/...)."
                ),
                media_type="text/plain",
            )

        # 2. Store version on request state for downstream use.
        request.state.api_version = result.version
        request.state.api_version_source = result.source

        # 3. Check full deprecation (past sunset).
        if self._reject_deprecated and self._schedule.is_deprecated(result.version):
            sunset = self._schedule.get_sunset_date(result.version)
            logger.warning(
                "Rejected request to deprecated version %s (sunset %s) from %s",
                result.version.value,
                sunset,
                request.client.host if request.client else "unknown",
            )
            return Response(
                status_code=410,
                content=(
                    f"API version {result.version.value} is no longer supported. "
                    f"It was removed on {sunset}. "
                    f"Please migrate to a newer version. "
                    f"See https://docs.distllm.dev/api-versions for migration."
                ),
                media_type="text/plain",
                headers={
                    VERSION_HEADER: result.version.value,
                    SUNSET_HEADER: self._schedule.sunset_header_value(result.version)
                    or "",
                },
            )

        # 4. Transform request body if compatibility layer is configured and
        #    the request version differs from the internal version.
        if (
            self._compat is not None
            and result.version != self._internal_version
        ):
            request.state._original_api_version = result.version
            # Note: actual body transformation happens in a helper that must
            # be called from the route handler, because FastAPI/Starlette
            # does not expose the parsed body to middleware in a
            # framework-independent way.  We store the target version so
            # route handlers can opt in.
            request.state.api_compat_target = self._internal_version
        else:
            request.state.api_compat_target = None

        # 5. Process the request.
        response = await call_next(request)

        # 6. Add version headers to the response.
        response.headers[VERSION_HEADER] = result.version.value

        sunset_header = self._schedule.sunset_header_value(result.version)
        if sunset_header is not None:
            response.headers[SUNSET_HEADER] = sunset_header

        # 7. Add deprecation warning header for versions near sunset.
        if self._schedule.is_near_sunset(result.version):
            sunset_date = self._schedule.get_sunset_date(result.version)
            days_left = self._schedule.days_until_sunset(result.version)
            migration_steps = self._schedule.get_migration_path(
                result.version, self._internal_version
            )
            breaking_count = sum(1 for s in migration_steps if s.breaking)
            response.headers[DEPRECATION_HEADER] = (
                f"Version {result.version.value} will be removed on "
                f"{sunset_date} ({days_left} days remaining). "
                f"{breaking_count} breaking change(s) to address. "
                f"See https://docs.distllm.dev/api-versions for migration."
            )

        # 8. Transform response body back to the request version if the
        #    compatibility layer is active and the request was upgraded.
        original_version: APIVersion | None = getattr(
            request.state, "_original_api_version", None
        )
        if (
            self._compat is not None
            and original_version is not None
            and original_version != self._internal_version
        ):
            # The response body may already be streamed at this point.
            # Full body transformation for buffered responses only.
            if hasattr(response, "body") and callable(response.body):
                body_value = response.body  # type: ignore[union-attr]
                if isinstance(body_value, bytes):
                    import json

                    try:
                        body_dict = json.loads(body_value)
                        transformed = self._compat.transform_response(
                            body_dict,
                            from_version=self._internal_version,
                            to_version=original_version,
                        )
                        # Build a new response with the transformed body.
                        from starlette.responses import JSONResponse

                        new_response = JSONResponse(
                            content=transformed,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type,
                        )
                        return new_response
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.debug(
                            "Could not transform response body for version %s",
                            original_version.value,
                        )

        return response


# ── Convenience factory ───────────────────────────────────────────────────────


def create_version_middleware(
    *,
    default_version: str = DEFAULT_API_VERSION,
    internal_version: APIVersion = APIVersion.V2,
    reject_deprecated: bool = True,
    enable_compatibility: bool = True,
) -> VersionMiddleware:
    """Create a fully-configured ``VersionMiddleware`` with sane defaults.

    Parameters
    ----------
    default_version:
        Fallback version string (default ``"2026-07"``).
    internal_version:
        The server's internal API version (default V2).
    reject_deprecated:
        Whether to return 410 for past-sunset versions (default ``True``).
    enable_compatibility:
        Whether to register the built-in field mappings for request/response
        transformation (default ``True``).
    """
    schedule = DeprecationSchedule.default_schedule()
    negotiator = VersionNegotiator(default_version=default_version)

    compat: VersionCompatibilityLayer | None = None
    if enable_compatibility:
        compat = VersionCompatibilityLayer()
        _register_default_mappings(compat)

    return VersionMiddleware(
        app=None,  # type: ignore[arg-type]  # set when added via add_middleware
        negotiator=negotiator,
        schedule=schedule,
        compatibility_layer=compat,
        internal_version=internal_version,
        reject_deprecated=reject_deprecated,
    )


def _register_default_mappings(layer: VersionCompatibilityLayer) -> None:
    """Register the built-in field mappings for V1 <-> V2 compatibility."""
    layer.add_mapping(
        from_version=APIVersion.V1,
        to_version=APIVersion.V2,
        field_maps=[
            FieldMapping(
                source_field="logit_bias",
                target_field="logprobs",
                transform=lambda val: (
                    {"top_logprobs": len(val)} if isinstance(val, dict) else None
                ),
            ),
        ],
        direction="request",
    )
    layer.add_mapping(
        from_version=APIVersion.V2,
        to_version=APIVersion.V1,
        field_maps=[
            FieldMapping(
                source_field="response.object",
                target_field="response.object",
                transform=lambda _: "chat.completion",
            ),
            FieldMapping(
                source_field="response.system_fingerprint",
                target_field="response.system_fingerprint",
                transform=lambda _: None,
            ),
        ],
        direction="response",
    )
