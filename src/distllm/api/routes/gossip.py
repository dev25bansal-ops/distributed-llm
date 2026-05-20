"""Internal gossip endpoints for P2P cache discovery."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["gossip"])


class GossipFetchRequest(BaseModel):
    requester_id: str | None = None
    prefix_hashes: list[str] = Field(default_factory=list)


def _get_gossip_protocol():
    coord = g.coordinator
    protocol = getattr(coord, "_gossip_protocol", None) if coord is not None else None
    if protocol is None:
        raise HTTPException(status_code=503, detail="Gossip protocol is not enabled")
    return protocol


@router.post(
    "/api/v1/gossip/exchange",
    summary="Exchange gossip advertisement",
    description="Merge a peer node's advertisement into the local gossip state and return this node's advertisement. HMAC signature verification ensures message integrity. Compatible with unsigned peers during rollout.",
    response_description="This node's current advertisement",
    responses={
        403: {"description": "Invalid gossip message HMAC"},
        503: {"description": "Gossip protocol not enabled"},
    },
)
async def exchange_gossip_advertisement(peer_advertisement: dict, request: Request):
    """Merge a peer advertisement and return this node's advertisement.

    Verifies the HMAC signature for message authentication.
    Skips verification if the peer doesn't send an HMAC (backward compat
    during rollout).
    """
    protocol = _get_gossip_protocol()
    # Verify HMAC if present (backward-compatible: accept unsigned during rollout)
    if "_hmac" in peer_advertisement:
        if not protocol.verify_message(dict(peer_advertisement)):
            raise HTTPException(status_code=403, detail="Invalid gossip message HMAC")
    else:
        logger = request.app.state.logger if hasattr(request.app.state, 'logger') else None
    protocol.process_advertisement(peer_advertisement)
    return protocol.advertise()


@router.post(
    "/api/v1/gossip/fetch",
    summary="Fetch gossip cache entries",
    description="Return local cache entry references for the requested prefix hashes. Used by peer nodes to discover cached KV entries for distributed prefix caching.",
    response_description="Matching cache entries with metadata",
    responses={
        503: {"description": "Gossip protocol not enabled"},
    },
)
async def fetch_gossip_entries(body: GossipFetchRequest):
    """Return local cache entry references for requested prefix hashes."""
    protocol = _get_gossip_protocol()
    entries = {
        prefix_hash: entry_ref
        for prefix_hash, entry_ref in protocol.state.local_entries.items()
        if prefix_hash in body.prefix_hashes
    }
    return {
        "success": True,
        "cache_entries": entries,
        "entries_returned": len(entries),
    }
