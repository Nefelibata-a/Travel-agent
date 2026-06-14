"""
LangGraph ReAct agent for travel planning.

Flow: START -> planner -> tool_executor(cond) -> reflector -> END

The agent autonomously plans a trip by:
1. Searching flights between cities
2. Searching hotels at the destination
3. Looking up attractions and restaurants
4. Checking weather for travel dates
5. Running budget calculations in a sandbox
6. Synthesising a structured itinerary
"""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from tools.registry import get_all_tools
from memory.manager import MemoryManager
from agent.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    step_count: int
    final_answer: str | None


MAX_STEPS = 20
_memory_manager = MemoryManager()


def _build_llm_with_tools():
    import os
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen3.5"),
        temperature=0,
        streaming=True,
        base_url=os.getenv("LLM_BASE_URL", "https://maas.nscc-cs.cn/external/api/v1"),
        api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
    )
    tools = get_all_tools()
    return llm.bind_tools(tools), tools


_llm_with_tools, _tools = _build_llm_with_tools()
_tool_node = ToolNode(_tools)


def planner_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    long_ctx = _memory_manager.get_long_term_summary(session_id)

    sys_msg = SYSTEM_PROMPT
    if long_ctx:
        sys_msg += f"\n\n[User preferences from past trips]\n{long_ctx}"

    messages = [SystemMessage(content=sys_msg)] + list(state["messages"])
    response: AIMessage = _llm_with_tools.invoke(messages)
    _memory_manager.add_message(session_id, response)

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def reflector_node(state: AgentState) -> dict:
    return planner_node(state)


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if state["step_count"] >= MAX_STEPS:
        return END
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", _tool_node)
    graph.add_node("reflector", reflector_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "reflector")
    graph.add_conditional_edges("reflector", should_continue, {"tools": "tools", END: END})

    return graph.compile()


agent_graph = build_graph()
