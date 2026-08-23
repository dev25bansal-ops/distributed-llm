# API Versioning

DistLLM's public HTTP API is versioned by URL path prefix. This page is the
contract for what that means, what clients can rely on, and how deprecation
works. The machine-readable schedule lives in
`src/distllm/api/api_versioning.py` (`DeprecationSchedule.default_schedule()`).

## Current versions

| Version | Identifier | Path prefix | Status |
|---------|-----------|-------------|--------|
| v1 | `2026-01` | `/v1/*` | Stable — frozen (see promise below) |
| v2 | `2026-07` | `/v2/*` | Current development surface |

Every response to a versioned path carries an `X-API-Version` header with the
version identifier (e.g. `X-API-Version: 2026-01`), so clients can verify which
contract the server is speaking.

## The /v1 freeze promise

Once an endpoint ships under `/v1/`, we will **not**:

- remove or rename request/response fields,
- change field types or validation ranges,
- change status codes or the error envelope shape.

We **may**, without notice:

- add optional request fields with safe defaults,
- add new response fields (clients must ignore unknown fields),
- fix bugs so behavior matches this documentation,
- add new endpoints under `/v1/`.

Unversioned endpoints (`/health`, `/ready`, `/live`, `/metrics`, `/api/*`,
`/dashboard`, `/docs`) are operational surface — health probes, admin dashboards,
internal tooling. They are **not** covered by the freeze promise and may change
in any release.

## Deprecation policy

When a version must break compatibility, it is sunset on a published schedule:

1. **Announcement** at least 90 days before the sunset date. The announcement
   sets three response headers on every request to the deprecated prefix:
   - `Sunset: <HTTP-date>` — when the version stops working (RFC 7231 format).
   - `X-API-Deprecation: Version 2026-01 will be removed on ... N breaking
     change(s) to address. See https://docs.distllm.dev/api-versions` — human-
     readable migration pointer.
   - `Warning` — added automatically during the final 60 days
     (`NEAR_SUNSET_DAYS`).
2. **Sunset date passes** → requests to the deprecated version return
   **410 Gone** with a plain-text migration message. The version stops being
   served; it does not linger in a broken state.
3. **Removal**: code for a fully sunset version is deleted no earlier than one
   release after its 410 behavior goes live.

Current schedule: **v1 sunsets 2026-10-01; migrate to v2 before then.**

## Migrating v1 → v2

The documented breaking changes (see `_register_default_mappings` and
`DeprecationSchedule.default_schedule()` in `src/distllm/api/api_versioning.py`):

| Field | v1 | v2 | Breaking? |
|-------|----|----|-----------|
| `response.object` | `chat.completion` | `chat.completion.v2` | yes |
| `response.system_fingerprint` | absent | present | no (additive) |
| `request.logit_bias` | dict of token→bias | removed — use `logprobs` / `top_logprobs` | yes |
| `request.user` | any string | validated non-empty string | no |

## How /v2 rolls out

- `/v2/` endpoints are mounted alongside `/v1/` from day one — there is no
  flag-flip moment and no proxy rewrite. Clients opt in by changing their base
  URL.
- A v2 endpoint accepts the same request schema as its v1 counterpart unless a
  migration step above says otherwise, so migrating is mostly a URL change plus
  the field fixes in the table.
- New capabilities land on `/v2/` first; anything backward-compatible is then
  mirrored into `/v1/` per the freeze promise. Anything breaking stays v2-only.
- When `/v2/` itself needs a breaking change, `/v3/` is introduced under this
  same policy and v2 gets a sunset date — never mutated in place.

## Negotiation mechanics (reference)

Path prefix is the canonical version selector. The `VersionMiddleware` /
`VersionNegotiator` classes in `src/distllm/api/api_versioning.py` additionally
support an `Accept-Version: 2026-01` header (checked first, path second,
default last), optional request/response body transformation through
`VersionCompatibilityLayer`, and automatic 410 enforcement. These are available
for enablement but today the always-on behavior is: path-prefix selection +
`X-API-Version`/`Sunset`/`X-API-Deprecation` response headers via
`SecurityHeadersMiddleware` in `src/distllm/api/server.py`.
