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
from pydantic import BaseModel, Field

from loguru import logger


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

@router.post("/offer", response_model=SDPAnswerResponse)
async def webrtc_offer(req: SDPOfferRequest):
    """Exchange SDP offer/answer for WebRTC connection.

    Browser sends its SDP offer, server returns an SDP answer.
    This initiates the WebRTC handshake for a data channel connection
    to the DistLLM cluster.
    """
    try:
        from distllm.dist.webrtc import HAS_WEBRTC, WebRTCTransport, WebRTCConfig
    except ImportError:
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")

    if not HAS_WEBRTC:
        raise HTTPException(status_code=503, detail="WebRTC not available (aiortc not installed)")

    session_id = _session_mgr.create_session(req.session_id, req.sdp)

    try:
        from aiortc import RTCSessionDescription
        offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
        transport = WebRTCTransport(role="answerer", node_id=session_id)
        answer = await transport.accept_offer(offer)
        await transport.wait_connected(timeout=30.0)

        session = _session_mgr.get_session(session_id)
        if session:
            session["status"] = "connected"
            session["transport"] = transport

        return SDPAnswerResponse(
            sdp=answer.sdp,
            type=answer.type,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"WebRTC offer handling failed: {e}", exc_info=True)
        _session_mgr.close_session(session_id)
        raise HTTPException(status_code=500, detail="WebRTC setup failed")


@router.post("/ice")
async def webrtc_ice(req: ICECandidateRequest):
    """Exchange ICE candidates for WebRTC connection establishment."""
    session = _session_mgr.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from aiortc import RTCIceCandidate
        candidate = RTCIceCandidate(
            component=1,
            foundation="1",
            ip="0.0.0.0",
            port=0,
            priority=0,
            protocol="UDP",
            type="host",
        )
        transport = session.get("transport")
        if transport:
            await transport.add_ice_candidate(candidate)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"ICE candidate handling failed: {e}")
        return {"status": "error", "detail": str(e)}


@router.get("/status", response_model=WebRTCStatusResponse)
async def webrtc_status():
    """Get WebRTC signaling server status."""
    stats = _session_mgr.get_stats()
    return WebRTCStatusResponse(**stats)


@router.delete("/sessions/{session_id}")
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
            pass

    _session_mgr.close_session(session_id)
    return {"status": "closed", "session_id": session_id}
