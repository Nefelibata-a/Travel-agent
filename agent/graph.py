"""
LangGraph ReAct agent for travel planning — MCP edition.

The agent discovers tools from the SmartTrip MCP Server (JSON-RPC over stdio)
instead of calling Python functions directly.

Flow: START -> planner -> tool_executor(cond) -> reflector -> END
"""

from __future__ import annotations

import operator
import os
import json
import time
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from loguru import logger

from agent.mcp_client import discover_mcp_tools, get_mcp_client, shutdown_mcp_client
from agent.prompts import SYSTEM_PROMPT
from memory.manager import MemoryManager


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    step_count: int
    final_answer: str | None


MAX_STEPS = 10  # reduced from 20: batch tool calls need far fewer iterations
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
    _mcp_llm_with_tools = llm.bind_tools(_mcp_tools, parallel_tool_calls=True)


# ============================================================================
# Instrumented nodes — log step timing and tool calls
# ============================================================================

def _planner_impl(state: AgentState) -> dict:
    """LLM planner — calls LLM with MCP tools bound, producing AIMessage with tool_calls."""
    _ensure_mcp_connected()

    session_id = state["session_id"]
    long_ctx = _memory_manager.get_long_term_summary(session_id)

    sys_msg = SYSTEM_PROMPT
    if long_ctx:
        sys_msg += f"\n\n[User preferences from past trips]\n{long_ctx}"

    messages = [SystemMessage(content=sys_msg)] + list(state["messages"])

    # ── 诊断：LLM 当前看到的消息数量 + 最后一条 ToolMessage 内容 ──
    step = state["step_count"] + 1
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_msgs:
        last_tm = tool_msgs[-1]
        content_preview = (last_tm.content or "")[:300]
        logger.info(
            f"[step {step}] LLM 上下文: {len(state['messages'])} 条消息, "
            f"ToolMessage 数={len(tool_msgs)}, "
            f"最后 ToolMessage({last_tm.name})={content_preview!r}..."
        )

    t0 = time.perf_counter()
    response: AIMessage = _mcp_llm_with_tools.invoke(messages)
    llm_ms = (time.perf_counter() - t0) * 1000

    tc = response.tool_calls or []

    if tc:
        names = [t["name"] for t in tc]
        # LLM 在调用工具前的「思考」
        reasoning = (response.content or "")[:200]
        logger.info(
            f"[step {step}] LLM {llm_ms:.0f}ms → calls: {names}"
            + (f" | 思考: {reasoning!r}" if reasoning else "")
        )
    else:
        logger.info(
            f"[step {step}] LLM {llm_ms:.0f}ms → final answer "
            f"({len(response.content or '')} chars)"
        )

    _memory_manager.add_message(session_id, response)

    return {
        "messages": [response],
        "step_count": step,
    }


def planner_node(state: AgentState) -> dict:
    return _planner_impl(state)


def reflector_node(state: AgentState) -> dict:
    return _planner_impl(state)


def _build_tool_node():
    """Build an instrumented ToolNode that logs and smart-deduplicates.
    
    规则:
      成功 → 永不重调（去重跳过）
      失败 → 最多重试 2 次，超过也跳过
    """
    _ensure_mcp_connected()
    base = ToolNode(_mcp_tools)
    MAX_RETRIES = 2

    def instrumented(state: AgentState) -> dict:
        # ── 统计每个工具：成功? 重试了几次? ──
        tool_state = {}  # name -> {"success": bool, "retries": int}
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                entry = tool_state.get(m.name, {"success": False, "retries": 0})
                content = m.content or ""
                err = False
                try:
                    d = json.loads(content) if isinstance(content, str) else content
                    if isinstance(d, dict) and ("error" in d or d.get("isError")):
                        err = True
                except (json.JSONDecodeError, TypeError):
                    pass
                if err:
                    entry["retries"] += 1
                    entry["success"] = False
                else:
                    entry["success"] = True
                tool_state[m.name] = entry

        # ── 拦截：成功过或重试超限 → 跳过 ──
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            new_calls, skip, skip_reason = [], [], []
            for tc in last_msg.tool_calls:
                ts = tool_state.get(tc["name"], {"success": False, "retries": 0})
                if ts["success"]:
                    skip.append(tc["name"]); skip_reason.append("已成功")
                elif ts["retries"] >= MAX_RETRIES:
                    skip.append(tc["name"]); skip_reason.append(f"重试{ts['retries']}次已达上限")
                else:
                    new_calls.append(tc)
            if skip:
                logger.warning(f"[dedup] 跳过: {list(zip(skip, skip_reason))}")
            if not new_calls:
                return {"messages": []}
            last_msg.tool_calls = new_calls

        t0 = time.perf_counter()
        result = base.invoke(state)
        ms = (time.perf_counter() - t0) * 1000

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        names = [m.name for m in tool_msgs]
        logger.info(f"[tools] {ms:.0f}ms → executed: {names}")

        return result

    return instrumented


def should_continue(state: AgentState) -> str:
    """Decide whether to call more tools or end the loop."""
    last_msg = state["messages"][-1]
    if state["step_count"] >= MAX_STEPS:
        logger.warning(f"[flow] MAX_STEPS reached ({MAX_STEPS}) → END")
        return END
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    logger.info(f"[flow] no more tool calls → END (total steps: {state['step_count']})")
    return END


# ============================================================================
# Graph assembly
# ============================================================================

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
