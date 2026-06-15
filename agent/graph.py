"""
LangGraph ReAct agent for travel planning — MCP edition.

The agent discovers tools from the SmartTrip MCP Server (JSON-RPC over stdio)
instead of calling Python functions directly.

Flow: START -> planner -> tool_executor(cond) -> reflector -> END

The agent autonomously plans a trip by:
1. Searching flights between cities        (flight_search via MCP)
2. Searching hotels at the destination     (hotel_search via MCP)
3. Looking up attractions and restaurants  (attraction_search via MCP)
4. Checking weather for travel dates       (weather_check via MCP)
5. Running budget calculations in sandbox  (budget_calculator via MCP)
6. Synthesising a structured itinerary
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from agent.mcp_client import discover_mcp_tools, get_mcp_client, shutdown_mcp_client
from agent.prompts import SYSTEM_PROMPT
from memory.manager import MemoryManager


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    step_count: int
    final_answer: str | None


MAX_STEPS = 20
_memory_manager = MemoryManager()

# ---- MCP-powered tools (discovered from MCP server at startup) ----
_mcp_tools = None
_mcp_llm_with_tools = None


def _ensure_mcp_connected():
    """Lazy-init: connect to MCP server and build LangChain tool bindings."""
    global _mcp_tools, _mcp_llm_with_tools
    if _mcp_tools is not None:
        return

    _mcp_tools = discover_mcp_tools()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen3.5"),
        temperature=0,
        streaming=True,
        base_url=os.getenv("LLM_BASE_URL", "https://maas.nscc-cs.cn/external/api/v1"),
        api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
    )
    _mcp_llm_with_tools = llm.bind_tools(_mcp_tools)


def planner_node(state: AgentState) -> dict:
    """LLM planner — calls LLM with MCP tools bound, producing AIMessage with tool_calls."""
    _ensure_mcp_connected()

    session_id = state["session_id"]
    long_ctx = _memory_manager.get_long_term_summary(session_id)

    sys_msg = SYSTEM_PROMPT
    if long_ctx:
        sys_msg += f"\n\n[User preferences from past trips]\n{long_ctx}"

    messages = [SystemMessage(content=sys_msg)] + list(state["messages"])
    response: AIMessage = _mcp_llm_with_tools.invoke(messages)
    _memory_manager.add_message(session_id, response)

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def reflector_node(state: AgentState) -> dict:
    """Reflect on tool results and decide next action — same as planner."""
    return planner_node(state)


def _build_tool_node():
    """Build ToolNode with MCP-discovered tools (lazy)."""
    _ensure_mcp_connected()
    return ToolNode(_mcp_tools)


def should_continue(state: AgentState) -> str:
    """Decide whether to call more tools or end the loop."""
    last_msg = state["messages"][-1]
    if state["step_count"] >= MAX_STEPS:
        return END
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    """Build the LangGraph state graph for the ReAct agent."""
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", _build_tool_node())
    graph.add_node("reflector", reflector_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "reflector")
    graph.add_conditional_edges("reflector", should_continue, {"tools": "tools", END: END})

    return graph.compile()


# Global compiled agent graph
agent_graph = build_graph()
