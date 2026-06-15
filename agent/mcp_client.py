"""
MCP Client — connects to SmartTrip MCP Server and provides LangChain-compatible tools.

Architecture:
  Agent (LangGraph) → MCPToolWrapper (BaseTool) → MCPClientManager (subprocess stdio) → MCP Server

Usage:
  from agent.mcp_client import discover_mcp_tools
  tools = discover_mcp_tools()  # returns list[BaseTool]
  llm.bind_tools(tools)
"""

from __future__ import annotations

import subprocess
import json
import sys
import os
import threading
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model
from loguru import logger


# ============================================================================
# MCP Client Manager — manages the MCP server subprocess
# ============================================================================

class MCPClientManager:
    """
    Manages a single MCP server subprocess with JSON-RPC 2.0 over stdio.

    Thread-safe: uses a lock for write operations and response matching by request id.
    """

    def __init__(self, server_script: str | None = None):
        if server_script is None:
            server_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "tools", "mcp_server.py"
            )
        self._server_script = server_script
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._tools_cache: list[dict] | None = None

    # ---- Lifecycle ----

    def start(self):
        """Start the MCP server subprocess."""
        if self._process is not None:
            return

        logger.info(f"Starting MCP server: {self._server_script}")
        python = sys.executable

        self._process = subprocess.Popen(
            [python, self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Send initialize
        init_result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "SmartTrip Agent", "version": "1.0.0"},
            "capabilities": {},
        })
        logger.info(f"MCP server initialized: {init_result.get('serverInfo', {})}")

        # Discovery: list all available tools
        tools_result = self._send_request("tools/list", {})
        self._tools_cache = tools_result.get("tools", [])
        logger.info(f"Discovered {len(self._tools_cache)} MCP tools: "
                    f"{[t['name'] for t in self._tools_cache]}")

    def stop(self):
        """Stop the MCP server subprocess."""
        if self._process is None:
            return
        logger.info("Stopping MCP server...")
        try:
            self._process.stdin.close()
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
        finally:
            self._process = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ---- Tool Management ----

    def get_tool_definitions(self) -> list[dict]:
        """Return cached tool definitions from tools/list."""
        if self._tools_cache is None:
            self.start()
        return self._tools_cache or []

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call an MCP tool and return the text result.
        Raises RuntimeError if the server is not running or the call fails.
        """
        if not self.is_running():
            self.start()

        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        # Parse MCP tool result
        content = result.get("content", [])
        if not content:
            if result.get("isError"):
                raise RuntimeError(f"MCP tool '{name}' returned an error")
            return ""

        # Concatenate all text content blocks
        texts = [c["text"] for c in content if c.get("type") == "text"]
        combined = "\n".join(texts)

        if result.get("isError"):
            raise RuntimeError(f"MCP tool '{name}' error: {combined}")

        return combined

    # ---- JSON-RPC Communication ----

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response (blocking, thread-safe)."""
        with self._lock:
            self._request_id += 1
            req_id = self._request_id

            request = json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }, ensure_ascii=False)

            # Write request
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()

            # Read response (match by request id)
            timeout = 30  # seconds
            start = time.time()
            while time.time() - start < timeout:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"MCP server closed stdout. Stderr: {self._read_stderr()}"
                    )
                try:
                    response = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if response.get("id") == req_id:
                    if "error" in response:
                        err = response["error"]
                        raise RuntimeError(
                            f"MCP error [{err.get('code')}]: {err.get('message')}"
                        )
                    return response.get("result", {})

            raise TimeoutError(f"MCP request '{method}' timed out after {timeout}s")

    def _read_stderr(self) -> str:
        """Non-blocking read of stderr for diagnostics."""
        if self._process is None:
            return ""
        import select as _select
        try:
            lines = []
            while True:
                r, _, _ = _select.select([self._process.stderr], [], [], 0.1)
                if not r:
                    break
                line = self._process.stderr.readline()
                if not line:
                    break
                lines.append(line.strip())
            return "\n".join(lines[-10:])
        except Exception:
            return "(stderr read failed)"


# ============================================================================
# Singleton MCP client
# ============================================================================

_mcp_client: MCPClientManager | None = None


def get_mcp_client() -> MCPClientManager:
    """Get or create the singleton MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClientManager()
        _mcp_client.start()
    return _mcp_client


def shutdown_mcp_client():
    """Shut down the singleton MCP client."""
    global _mcp_client
    if _mcp_client is not None:
        _mcp_client.stop()
        _mcp_client = None


# ============================================================================
# LangChain BaseTool wrappers — adapt MCP tools for LangChain/LangGraph
# ============================================================================

def _json_type_to_python(json_type: str) -> type:
    """Convert JSON Schema type to Python type."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


def _build_args_schema(tool_def: dict) -> type[BaseModel]:
    """Build a Pydantic model from MCP tool inputSchema."""
    schema = tool_def.get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields = {}
    for prop_name, prop_def in properties.items():
        prop_type = _json_type_to_python(prop_def.get("type", "string"))
        description = prop_def.get("description", "")

        if prop_name in required:
            fields[prop_name] = (prop_type, Field(..., description=description))
        else:
            default_val = prop_def.get("default")
            if isinstance(default_val, str):
                fields[prop_name] = (prop_type, Field(default=default_val, description=description))
            elif isinstance(default_val, (int, float)):
                fields[prop_name] = (prop_type, Field(default=default_val, description=description))
            else:
                fields[prop_name] = (prop_type, Field(default=None, description=description))

    model_name = f"{tool_def['name'].replace('_', ' ').title().replace(' ', '')}Input"
    return create_model(model_name, **fields)  # type: ignore


class MCPToolWrapper(BaseTool):
    """A LangChain BaseTool that calls an MCP server tool via the global MCP client."""

    mcp_tool_name: str = ""

    def __init__(self, tool_def: dict, **kwargs):
        schema = _build_args_schema(tool_def)
        super().__init__(
            name=tool_def["name"],
            description=tool_def["description"],
            args_schema=schema,
            **kwargs,
        )
        self.mcp_tool_name = tool_def["name"]

    def _run(self, **kwargs) -> str:
        """Synchronous tool execution through MCP protocol."""
        try:
            return get_mcp_client().call_tool(self.mcp_tool_name, kwargs)
        except Exception as e:
            logger.error(f"[MCPTool] {self.name} failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _arun(self, **kwargs) -> str:
        """Async tool execution — delegates to sync (MCP stdio is inherently blocking)."""
        return self._run(**kwargs)


# ============================================================================
# Discovery — create LangChain tools from MCP server
# ============================================================================

def discover_mcp_tools(client: MCPClientManager | None = None) -> list[BaseTool]:
    """
    Connect to the MCP server and create LangChain-compatible BaseTool wrappers
    for all tools exposed by the MCP server.

    Returns a list of BaseTool instances ready for llm.bind_tools() and ToolNode.
    """
    if client is None:
        client = get_mcp_client()

    definitions = client.get_tool_definitions()
    tools = []

    for tool_def in definitions:
        wrapper = MCPToolWrapper(tool_def=tool_def)
        tools.append(wrapper)
        logger.info(f"  MCP tool loaded: {wrapper.name}")

    return tools
