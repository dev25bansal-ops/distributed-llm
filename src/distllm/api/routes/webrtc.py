"""WebRTC signaling API routes for browser-based inference.

.. warning::
    **EXPERIMENTAL**: WebRTC support depends on ``aiortc`` which is complex,
    OS-dependent, and has known issues on some platforms. Use at your own
    risk in non-production environments. The API surface may change without
    notice.

Exposes HTTP endpoints for SDP offer/answer and ICE candidate exchange,
enabling browsers to establish WebRTC data channels to the DistLLM cluster.

Endpoints:
    POST /v1/webrtc/offer  — Exchange SDP offer for answer
    POST /v1/webrtc/ice    — Exchange ICE candidates
    GET  /v1/webrtc/status — Get signaling server status
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from ..errors import error_openapi_entry

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry


router = APIRouter(tags=["webrtc"], prefix="/v1/webrtc")


# ── Request/Response Models ─────────────────────────────────────────────────

class SDPOfferRequest(BaseModel):
    sdp: str = Field(..., description="SDP offer from the browser")
    type: str = Field(default="offer", description="SDP type")
    session_id: str = Field(default="", description="Client-generated session ID")


class SDPAnswerResponse(BaseModel):
    sdp: str
    type: str = "answer"
    session_id: str


class ICECandidateRequest(BaseModel):
    session_id: str
    candidate: str
    sdp_mid: str = ""
    sdp_mline_index: int = 0


class WebRTCStatusResponse(BaseModel):
    active_sessions: int
    total_sessions: int
    uptime_seconds: float


# ── Session Manager ─────────────────────────────────────────────────────────

class WebRTCSessionManager:
    """Manages WebRTC signaling sessions for browser clients."""

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._start_time = time.time()

    def create_session(self, session_id: str, sdp: str) -> str:
        """Create a new signaling session and return SDP answer."""
        sid = session_id or f"ws-{uuid.uuid4().hex[:8]}"
        self._sessions[sid] = {
            "session_id": sid,
            "created_at": time.time(),
            "offer_sdp": sdp,
            "status": "signaling",
        }
        logger.info(f"WebRTC session created: {sid}")
        return sid

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_stats(self) -> dict[str, Any]:
        active = sum(1 for s in self._sessions.values() if s["status"] != "closed")
        return {
            "active_sessions": active,
            "total_sessions": len(self._sessions),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }


_session_mgr = WebRTCSessionManager()


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/offer",
    response_model=SDPAnswerResponse,
    summary="Exchange an SDP offer for the server's SDP answer",
    responses={
        422: ERR_ENTRY("Malformed SDP offer payload", type_="invalid_request_error", code="invalid_request"),
        500: ERR_ENTRY("WebRTC setup failed while answering the offer", type_="server_error", code="500"),
        503: ERR_ENTRY("WebRTC not available (aiortc not installed)", type_="service_unavailable", code="503"),
    },
)
async def webrtc_offer(req: SDPOfferRequest):
    """Exchange SDP offer/answer for WebRTC connection.

    Browser sends its SDP offer, server returns an SDP answer.
    This initiates the WebRTC handshake for a data channel connection
    to the DistLLM cluster.
    """
    try:
        from distllm.dist.webrtc import HAS_WEBRTC, WebRTCConfig, WebRTCTransport
    except ImportError:
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")

    if not HAS_WEBRTC:
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")

    session_id = _session_mgr.create_session(req.session_id, req.sdp)

    try:
        from aiortc import RTCSessionDescription
        offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
    except ImportError:
        _session_mgr.close_session(session_id)
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")
    except Exception as e:
        logger.warning(f"WebRTC offer rejected (invalid SDP): {e}")
        _session_mgr.close_session(session_id)
        raise HTTPException(status_code=422, detail=f"Invalid SDP offer: {e}")

    try:
        transport = WebRTCTransport(role="answerer", node_id=session_id)
        answer = await transport.accept_offer(offer)

        # Do NOT wait for the connection here. The client cannot apply the
        # answer — and therefore cannot start ICE — until this response is
        # returned, so awaiting wait_connected(timeout=30) before returning
        # blocked every real offer for the full 30s and then still reported
        # "connected" (or 500'd after close_session). Return the answer now;
        # the established connection is reported by the status/candidate
        # endpoints and the transport is kept on the session for them.
        session = _session_mgr.get_session(session_id)
        if session:
            session["status"] = "answering"
            session["transport"] = transport

        return SDPAnswerResponse(
            sdp=answer.sdp,
            type=answer.type,
            session_id=session_id,
        )
    except Exception as e:
        # Name the failing stage: the old single "WebRTC setup failed" hid
        # whether the SDP, aiortc, or connectivity was at fault.
        logger.error(f"WebRTC offer handling failed: {e}", exc_info=True)
        _session_mgr.close_session(session_id)
        raise HTTPException(status_code=500, detail=f"WebRTC setup failed: {e}")


@router.post(
    "/ice",
    summary="Exchange ICE candidates for WebRTC connection establishment",
    responses={
        404: ERR_ENTRY("Session ID not found", type_="invalid_request_error", code="not_found"),
        409: ERR_ENTRY("Session has no active WebRTC transport yet", type_="invalid_request_error", code="409"),
        422: ERR_ENTRY("Malformed ICE candidate payload", type_="invalid_request_error", code="invalid_request"),
        500: ERR_ENTRY("ICE candidate could not be applied", type_="server_error", code="500"),
        503: ERR_ENTRY("WebRTC not available (aiortc not installed)", type_="service_unavailable", code="503"),
    },
)
async def webrtc_ice(req: ICECandidateRequest):
    """Exchange ICE candidates for WebRTC connection establishment."""
    session = _session_mgr.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # An empty candidate string cannot be applied to the peer connection.
    if not req.candidate:
        raise HTTPException(
            status_code=422,
            detail="Malformed ICE candidate: 'candidate' must not be empty",
        )

    transport = session.get("transport")
    if transport is None:
        raise HTTPException(status_code=409, detail="Session has no active WebRTC transport")

    try:
        from aiortc import RTCIceCandidate
    except ImportError:
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")

    # Imported before the try that uses it: a name referenced in an ``except``
    # clause must exist even on the paths that raise before assignment.
    from distllm.dist.webrtc import WebRTCError

    try:
        # C-07: use the candidate identity from the request body. aiortc's
        # addIceCandidate requires sdpMid or sdpMLineIndex; constructing the
        # candidate without either made every apply attempt raise ValueError.
        sdp_mid = req.sdp_mid or None
        candidate = RTCIceCandidate(
            component=1,
            foundation="1",
            ip=req.candidate.split("/")[0],
            port=0,
            priority=0,
            protocol="UDP",
            type="host",
            sdpMid=sdp_mid,
            sdpMLineIndex=None if sdp_mid is not None else req.sdp_mline_index,
        )
        await transport.add_ice_candidate(candidate)
    except ValueError as e:
        # aiortc rejects the candidate shape: a client-side problem.
        logger.warning(f"ICE candidate rejected: {e}")
        raise HTTPException(status_code=422, detail=f"Malformed ICE candidate: {e}")
    except WebRTCError as e:
        logger.warning(f"ICE candidate not applied (transport): {e}")
        raise HTTPException(status_code=503, detail=f"WebRTC transport unavailable: {e}")
    except HTTPException:
        raise
    except Exception as e:
        # Previously this was a 200 {"status": "error"} -- a status-code-only
        # client believed the candidate was registered and connectivity then
        # failed silently.
        logger.warning(f"ICE candidate handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"ICE candidate handling failed: {e}")
    return {"status": "ok"}


@router.get(
    "/status",
    response_model=WebRTCStatusResponse,
    summary="Get WebRTC signaling server status",
)
async def webrtc_status():
    """Get WebRTC signaling server status."""
    stats = _session_mgr.get_stats()
    return WebRTCStatusResponse(**stats)


@router.delete(
    "/sessions/{session_id}",
    summary="Close a WebRTC signaling session",
    responses={
        404: ERR_ENTRY("Session ID not found", type_="invalid_request_error", code="not_found"),
    },
)
async def close_session(session_id: str):
    """Close a WebRTC signaling session."""
    session = _session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    transport = session.get("transport")
    if transport:
        try:
            await transport.close()
        except Exception:
            # CQ-08: best-effort cleanup: a transport that is already closed must not break session teardown.
            pass

    _session_mgr.close_session(session_id)
    return {"status": "closed", "session_id": session_id}
