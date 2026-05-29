"""FastAPI dependencies for role-based access control.

Usage::

    from distllm.api.auth_deps import require_role

    router = APIRouter(dependencies=[Depends(require_role("admin"))])

    @router.get("/admin/v1/nodes")
    async def list_nodes(request: Request):
        ...
"""

from fastapi import HTTPException, Request

from distllm.core.api_key_store import role_satisfies


def require_role(*roles: str):
    """FastAPI dependency factory: require the request's API key to have one of *roles*.

    Usage::

        @router.get("/admin/v1/nodes", dependencies=[Depends(require_role("admin"))])
        async def list_nodes():
            ...

        # Multiple acceptable roles:
        Depends(require_role("admin", "inference-only"))
    """

    async def _check_role(request: Request) -> None:
        actual = getattr(request.state, "api_key_role", None)
        if actual is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
            )
        for required in roles:
            if role_satisfies(actual, required):
                return
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions: role '{actual}' cannot access this endpoint "
                   f"(requires one of: {', '.join(roles)})",
        )

    return _check_role
