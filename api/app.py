"""
FastAPI application for SmartTrip Agent — MCP edition.

Endpoints:
  POST /plan     — start a trip planning session
  GET  /plan/{session_id} — get the itinerary for a session
  GET  /history/{session_id} — get conversation history
  DELETE /session/{session_id} — clear session
  GET  /tools    — list available MCP tools
  GET  /health   — health check (includes MCP server status)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import agent_graph
from agent.mcp_client import get_mcp_client, shutdown_mcp_client, discover_mcp_tools
from memory.manager import MemoryManager
from langchain_core.messages import HumanMessage
from loguru import logger


# ---- MCP lifecycle: start on app startup, stop on shutdown ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start MCP server on app startup, clean up on shutdown."""
    logger.info("[lifespan] Starting MCP client...")
    client = get_mcp_client()
    logger.info(f"[lifespan] MCP server running: {client.is_running()}")
    yield
    logger.info("[lifespan] Shutting down MCP client...")
    shutdown_mcp_client()


app = FastAPI(
    title="SmartTrip AI Travel Planner (MCP Edition)",
    version="0.2.0",
    lifespan=lifespan,
)
_memory = MemoryManager()


class PlanRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=100, description="e.g. 'Chengdu'")
    origin: str = Field(default="", description="Departure city, e.g. 'Beijing'")
    start_date: str = Field(default="", description="YYYY-MM-DD")
    end_date: str = Field(default="", description="YYYY-MM-DD")
    budget: int = Field(default=0, description="Total budget in CNY")
    preferences: str = Field(default="", description="Free-text preferences, e.g. 'food, nature, budget-friendly'")
    travelers: int = Field(default=1, ge=1, le=20)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PlanResponse(BaseModel):
    session_id: str
    itinerary: str
    steps: int
    tools_called: list[str]


@app.post("/plan", response_model=PlanResponse)
async def plan_trip(req: PlanRequest):
    """Plan a complete trip. The Agent calls tools through MCP protocol."""
    if not req.origin:
        raise HTTPException(status_code=400, detail="Please provide departure city (origin)")

    user_msg = (
        f"Plan a {req.end_date and (req.start_date + ' to ' + req.end_date) or req.start_date or 'upcoming'} "
        f"trip from {req.origin} to {req.destination}.\n"
        + (f"Budget: {req.budget} CNY total.\n" if req.budget else "")
        + (f"Preferences: {req.preferences}.\n" if req.preferences else "")
        + (f"Travelers: {req.travelers}.\n" if req.travelers > 1 else "")
        + "Please search flights, hotels, attractions, weather, and calculate the budget."
    )

    logger.info(f"[/plan] session={req.session_id} dest={req.destination} budget={req.budget}")

    state = {
        "messages": [HumanMessage(content=user_msg)],
        "session_id": req.session_id,
        "step_count": 0,
        "final_answer": None,
    }

    try:
        final = await agent_graph.ainvoke(state)
    except Exception as e:
        logger.error(f"[/plan] Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    answer = final["messages"][-1].content

    # Extract tool names called (from ToolMessage)
    tool_names = [
        m.name for m in final["messages"]
        if hasattr(m, "name") and m.name
    ]

    return PlanResponse(
        session_id=req.session_id,
        itinerary=answer if isinstance(answer, str) else str(answer),
        steps=final["step_count"],
        tools_called=tool_names,
    )


@app.get("/plan/{session_id}")
async def get_plan(session_id: str):
    messages = _memory.get_short_term_messages(session_id)
    itinerary = ""
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if len(content) > len(itinerary):
            itinerary = content
    return {"session_id": session_id, "itinerary": itinerary or "No itinerary found"}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    messages = _memory.get_short_term_messages(session_id)
    return {
        "session_id": session_id,
        "messages": [
            {"role": m.__class__.__name__, "content": m.content}
            for m in messages
        ],
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/tools")
async def list_tools():
    """List all tools available through MCP server."""
    try:
        tools = discover_mcp_tools()
        return {
            "count": len(tools),
            "protocol": "MCP",
            "tools": [{"name": t.name, "description": t.description} for t in tools],
        }
    except Exception as e:
        return {"count": 0, "protocol": "MCP", "error": str(e)}


@app.get("/health")
async def health():
    client = get_mcp_client()
    return {
        "status": "ok",
        "mcp_server": "running" if client.is_running() else "stopped",
        "version": "0.2.0 (MCP Edition)",
    }
