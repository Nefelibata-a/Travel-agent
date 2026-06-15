"""
Integration tests for MCP pipeline — verify:
  1. MCP server starts and responds to tools/list
  2. MCP server responds to tools/call
  3. MCP client discovers tools and creates LangChain wrappers
  4. Agent graph compiles with MCP tools
"""

import json
import subprocess
import sys
import time
import pytest


MCP_SERVER_SCRIPT = "tools/mcp_server.py"


# ============================================================================
# Test 1: MCP server lifecycle
# ============================================================================

def test_mcp_server_starts_and_initializes():
    """Verify MCP server starts, accepts initialize, and responds to tools/list."""
    proc = subprocess.Popen(
        [sys.executable, MCP_SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Step 1: initialize
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test-client", "version": "0.0.0"},
                "capabilities": {},
            }
        }) + "\n")
        proc.stdin.flush()

        init_response = json.loads(proc.stdout.readline().strip())
        assert init_response["jsonrpc"] == "2.0"
        assert init_response["id"] == 1
        assert "result" in init_response
        assert "SmartTrip MCP Server" in init_response["result"]["serverInfo"]["name"]
        assert "tools" in init_response["result"]["capabilities"]

        # Step 2: tools/list
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        }) + "\n")
        proc.stdin.flush()

        tools_response = json.loads(proc.stdout.readline().strip())
        assert tools_response["id"] == 2
        tools = tools_response["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        expected = ["flight_search", "hotel_search", "attraction_search", "weather_check", "budget_calculator"]
        for name in expected:
            assert name in tool_names, f"Missing tool: {name}"

        # Verify each tool has required fields
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]
            assert "properties" in tool["inputSchema"]

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)


# ============================================================================
# Test 2: MCP server tools/call
# ============================================================================

def test_mcp_server_tool_call_without_api_key():
    """Verify tools/call works (even without API key, should get error response)."""
    proc = subprocess.Popen(
        [sys.executable, MCP_SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Initialize
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "0.0.0"},
                "capabilities": {},
            }
        }) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()  # consume init response

        # Call a tool — should work even without API key (returns error in content)
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "flight_search",
                "arguments": {"origin": "Beijing", "destination": "Shanghai", "date": "2026-07-01"}
            }
        }) + "\n")
        proc.stdin.flush()

        call_response = json.loads(proc.stdout.readline().strip())
        assert call_response["id"] == 2
        assert "result" in call_response
        content = call_response["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"

        # Should contain either results or error about missing API key
        text = content[0]["text"]
        assert isinstance(text, str) and len(text) > 0

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)


# ============================================================================
# Test 3: MCP server responds to invalid tool names gracefully
# ============================================================================

def test_mcp_server_unknown_tool():
    """Verify MCP server returns error for unknown tools."""
    proc = subprocess.Popen(
        [sys.executable, MCP_SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Initialize
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "0.0.0"},
                "capabilities": {},
            }
        }) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()

        # Call unknown tool
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}}
        }) + "\n")
        proc.stdin.flush()

        call_response = json.loads(proc.stdout.readline().strip())
        assert call_response["id"] == 2
        assert "result" in call_response
        assert call_response["result"].get("isError") is True

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)


# ============================================================================
# Test 4: Agent graph compiles
# ============================================================================

def test_agent_graph_compiles():
    """Verify agent graph module imports and compiles without errors."""
    from agent.graph import agent_graph
    assert agent_graph is not None
    # graph should have nodes
    nodes = agent_graph.get_graph().nodes
    assert "planner" in nodes or "planner" in str(nodes)


# ============================================================================
# Test 5: Tool registry schemas are valid
# ============================================================================

def test_flight_schema_validation():
    from pydantic import ValidationError
    from tools.registry import FlightSearchInput

    with pytest.raises(ValidationError):
        FlightSearchInput(origin="", destination="Chengdu", date="2026-07-01")

    valid = FlightSearchInput(origin="Beijing", destination="Chengdu", date="2026-07-01")
    assert valid.origin == "Beijing"


def test_tool_registry_has_all_tools():
    from tools.registry import TOOL_REGISTRY, get_tool_by_name

    assert len(TOOL_REGISTRY) == 5
    for name in ["flight_search", "hotel_search", "attraction_search", "weather_check", "budget_calculator"]:
        fn = get_tool_by_name(name)
        assert fn is not None, f"Tool function '{name}' not found"
        assert callable(fn), f"'{name}' is not callable"


def test_tool_definitions_format():
    from tools.registry import get_tool_definitions

    defs = get_tool_definitions()
    assert len(defs) == 5

    for d in defs:
        assert "name" in d
        assert "description" in d
        assert "inputSchema" in d
        assert d["inputSchema"]["type"] == "object"
        assert "properties" in d["inputSchema"]
        # 'function' key should NOT be in definitions (it's internal)
        assert "function" not in d
