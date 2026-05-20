"""Agent API routes for ReAct agent execution."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["agent"])


def _get_coordinator():
    """Get the coordinator instance from the app state."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    return coord


class ToolDefinition(BaseModel):
    name: str
    description: str
    handler: str  # Python callable path or inline code


class AgentRunRequest(BaseModel):
    goal: str
    tools: list[ToolDefinition] = Field(default_factory=list)
    max_iterations: int | None = None


class AgentRunResponse(BaseModel):
    status: str
    result: str
    iterations: int
    memory: list[dict]


class AgentStatusResponse(BaseModel):
    enabled: bool
    state: str | None
    memory: list[dict]


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Run agent",
    description="Execute the ReAct reasoning/acting agent with a specified goal and optional tool definitions. The agent iteratively reasons about the goal, selects and executes tools, and refines its approach until completion.",
    response_description="Agent execution result with iterations and memory",
    responses={
        503: {"description": "Coordinator not available or agent loop not initialized"},
    },
)
async def agent_run(request: AgentRunRequest):
    """Run the ReAct agent with a goal and optional tools."""
    coord = _get_coordinator()
    agent_loop = getattr(coord, "_agent_loop", None)
    if agent_loop is None:
        raise HTTPException(status_code=503, detail="Agent loop not initialized")

    # Build tool registry from request
    tools = {}
    for tool_def in request.tools:
        tools[tool_def.name] = {
            "description": tool_def.description,
            "handler": tool_def.handler,
        }

    result = agent_loop.run(
        goal=request.goal,
        tools=tools,
        max_iterations=request.max_iterations,
    )
    return AgentRunResponse(
        status="completed",
        result=result.get("result", ""),
        iterations=result.get("iterations", 0),
        memory=result.get("memory", []),
    )


@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Get agent status",
    description="Return the current status of the agent loop, including whether it is enabled, its current state, and accumulated memory from previous runs.",
    response_description="Agent loop status and state",
)
async def agent_status():
    """Return agent loop status and current state."""
    coord = _get_coordinator()
    agent_loop = getattr(coord, "_agent_loop", None)
    if agent_loop is None:
        return AgentStatusResponse(enabled=False, state=None, memory=[])

    state = agent_loop.get_state()
    return AgentStatusResponse(
        enabled=True,
        state=state.get("state"),
        memory=state.get("memory", []),
    )
