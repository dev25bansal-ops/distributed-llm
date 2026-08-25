"""Internal gossip endpoints for P2P cache discovery."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["gossip"])


class GossipFetchRequest(BaseModel):
    """Peer request for local KV cache entries.

    ``signature`` carries the gossip HMAC signature of the request body.
    Declared explicitly so pydantic does not strip it from the request
    (undeclared fields are dropped by default), which previously made
    signature verification on this endpoint structurally impossible.
    NOTE: the wire name is ``_hmac``; a leading-underscore annotation
    would become a pydantic private attribute instead of a parsed field,
    so the alias form is required.
    """

    requester_id: str | None = None
    prefix_hashes: list[str] = Field(default_factory=list)
    signature: str | None = Field(default=None, alias="_hmac")


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
    # HMAC verification is mandatory — reject unsigned messages
    if "_hmac" not in peer_advertisement:
        raise HTTPException(status_code=403, detail="Gossip message must include HMAC signature")
    if not protocol.verify_message(dict(peer_advertisement)):
        raise HTTPException(status_code=403, detail="Invalid gossip message HMAC")
    protocol.process_advertisement(peer_advertisement)
    return protocol.advertise()


@router.post(
    "/api/v1/gossip/fetch",
    summary="Fetch gossip cache entries",
    description="Return local cache entry references for the requested prefix hashes. Used by peer nodes to discover cached KV entries for distributed prefix caching. Requests must carry a valid HMAC signature when a shared gossip key is configured; unsigned requests are rejected with 403.",
    response_description="Matching cache entries with metadata",
    responses={
        403: {"description": "Missing or invalid gossip fetch HMAC signature"},
        503: {"description": "Gossip protocol not enabled"},
    },
)
async def fetch_gossip_entries(body: GossipFetchRequest):
    """Return local cache entry references for requested prefix hashes.

    SECURITY: cache data is served only to authenticated peers.  When a
    shared gossip HMAC key is configured, the request signature is verified
    (fail closed) and the response is signed so the requester can verify it.
    Without a shared key (legacy dev/test mode) unsigned requests are still
    served for backward compatibility — the protocol layer logs a loud
    one-time warning, matching the kademlia_dht.py convention.
    """
    protocol = _get_gossip_protocol()

    # Rebuild the exact wire body for verification: the requester signed
    # {"requester_id": ..., "prefix_hashes": [...]} (the same shape
    # GossipTransport.request_kv_cache sends), so verify over that dict
    # rather than the full pydantic model dump.
    wire_body: dict = {
        "requester_id": body.requester_id,
        "prefix_hashes": body.prefix_hashes,
    }
    authorized, reason = protocol.authorize_fetch_request({
        **wire_body,
        "_hmac": body.signature,
    })
    if not authorized:
        raise HTTPException(
            status_code=403,
            detail=f"Gossip fetch request rejected: {reason}",
        )

    entries = {
        prefix_hash: entry_ref
        for prefix_hash, entry_ref in protocol.state.local_entries.items()
        if prefix_hash in body.prefix_hashes
    }
    response = {
        "success": True,
        "cache_entries": entries,
        "entries_returned": len(entries),
    }
    # Sign only under a shared key: node-local dev/test keys differ per
    # node, so an attached signature would be unverifiable by the peer.
    if getattr(protocol, "has_shared_hmac_key", False):
        response = protocol.sign_fetch_request(response)
    return response
