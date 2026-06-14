"""Tests for SmartTrip Agent."""

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_agent_returns_itinerary():
    from agent.graph import agent_graph

    state = {
        "messages": [HumanMessage(content="Plan a trip from Beijing to Chengdu on 2026-07-01 to 2026-07-03, budget 5000 CNY")],
        "session_id": "test-001",
        "step_count": 0,
        "final_answer": None,
    }
    result = await agent_graph.ainvoke(state)
    assert result["messages"], "No messages returned"
    last = result["messages"][-1]
    assert last.content, "Empty answer"


@pytest.mark.asyncio
async def test_step_bounded():
    from agent.graph import agent_graph, MAX_STEPS

    state = {
        "messages": [HumanMessage(content="Search everything about Tokyo.")],
        "session_id": "test-002",
        "step_count": 0,
        "final_answer": None,
    }
    result = await agent_graph.ainvoke(state)
    assert result["step_count"] <= MAX_STEPS


def test_flight_schema_validation():
    from pydantic import ValidationError
    from tools.registry import FlightSearchInput

    with pytest.raises(ValidationError):
        FlightSearchInput(origin="", destination="Chengdu", date="2026-07-01")

    valid = FlightSearchInput(origin="Beijing", destination="Chengdu", date="2026-07-01")
    assert valid.origin == "Beijing"
