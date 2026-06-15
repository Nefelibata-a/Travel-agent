"""
MCP Client — connects to SmartTrip MCP Server and provides LangChain-compatible tools.

ASYNC EDITION — concurrent tool calls via request ID routing instead of lock.

Key changes from sync version:
  - threading.Lock removed
  - Dedicated reader thread routes responses by request ID (concurrent.futures.Future)
  - Writes are atomic (write_lock), reads are routed (no global lock)
  - 4 concurrent call_tool() → 4 concurrent SerpAPI HTTP on the async MCP server

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
from concurrent.futures import Future
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model
from loguru import logger


# ============================================================================
# MCP Client Manager — concurrent via reader thread + request ID routing
# ============================================================================

class MCPClientManager:
    """
    Manages a single MCP server subprocess with concurrent JSON-RPC calls.

    Architecture:
      caller_thread_1 ─→ write(stdin) ─→ Future.wait()
      caller_thread_2 ─→ write(stdin) ─→ Future.wait()
      caller_thread_3 ─→ write(stdin) ─→ Future.wait()
      caller_thread_4 ─→ write(stdin) ─→ Future.wait()
              │                              ▲
              └─ async MCP Server ── stdout ─┘
                       │
              reader_thread: reads stdout, matches by request_id,
                             resolves the correct Future

    The reader thread is the key: it reads ALL responses from stdout and
    dispatches them by request ID, so no caller thread needs to hold a lock
    while waiting for its response.
    """

    _next_id_lock = threading.Lock()
    _next_id = 0

    @classmethod
    def _gen_id(cls) -> int:
        with cls._next_id_lock:
            cls._next_id += 1
            return cls._next_id

    def __init__(self, server_script: str | None = None):
        if server_script is None:
            server_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "tools", "mcp_server.py"
            )
        self._server_script = server_script
        self._process: subprocess.Popen | None = None
        self._tools_cache: list[dict] | None = None

        # Write lock — only held briefly during stdin.write()
        self._write_lock = threading.Lock()

        # Response routing — reader thread populates, callers wait on Futures
        self._pending: dict[int, Future] = {}
        self._pending_lock = threading.Lock()

        # Reader thread
        self._reader_thread: threading.Thread | None = None
        self._running = False

    # ---- Lifecycle ----

    def start(self):
        """Start MCP server and reader thread."""
        if self._process is not None:
            return

        logger.info(f"Starting MCP server: {self._server_script}")
        python = sys.executable

        self._process = subprocess.Popen(
            [python, self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",  # MCP 协议要求 UTF-8，Windows 默认 gbk 会炸
            errors="replace",  # 兜底：遇到坏字符用 ? 代替
            bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # 确保子进程也走 UTF-8
        )

        self._running = True

        # Start reader thread — routes stdout responses by request ID
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._reader_thread.start()

        # Initialize
        init_result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "SmartTrip Agent (Async)", "version": "2.0.0"},
            "capabilities": {},
        })
        logger.info(f"MCP server initialized: {init_result.get('serverInfo', {})}")

        # Discover tools
        tools_result = self._send_request("tools/list", {})
        self._tools_cache = tools_result.get("tools", [])
        logger.info(f"Discovered {len(self._tools_cache)} MCP tools: "
                    f"{[t['name'] for t in self._tools_cache]}")

    def stop(self):
        """Stop MCP server and reader thread."""
        self._running = False
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
        if self._tools_cache is None:
            self.start()
        return self._tools_cache or []

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call an MCP tool — CONCURRENT CAPABLE.

        Multiple threads can call this simultaneously. Each call:
        1. Writes its JSON-RPC request to stdin (write_lock, brief)
        2. Registers a Future with its request ID
        3. Waits on the Future (no global lock held)
        4. Reader thread resolves the Future when response arrives
        """
        if not self.is_running():
            self.start()

        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        content = result.get("content", [])
        if not content:
            if result.get("isError"):
                raise RuntimeError(f"MCP tool '{name}' returned an error")
            return ""

        texts = [c["text"] for c in content if c.get("type") == "text"]
        combined = "\n".join(texts)

        if result.get("isError"):
            raise RuntimeError(f"MCP tool '{name}' error: {combined}")

        return combined

    # ----------------------------------------------------------------
    # JSON-RPC over stdio — concurrent via request ID routing
    # ----------------------------------------------------------------

    def _send_request(self, method: str, params: dict) -> dict:
        """
        Send a request and wait for the response — concurrent capable.

        1. Assign a unique request ID
        2. Create a Future, store it in _pending
        3. Write the request (brief write_lock)
        4. Wait on the Future (no lock — multiple callers wait simultaneously)
        """
        req_id = self._gen_id()
        fut: Future = Future()

        with self._pending_lock:
            self._pending[req_id] = fut

        request_text = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }, ensure_ascii=False) + "\n"

        # Write — brief lock, just for atomic write
        with self._write_lock:
            try:
                self._process.stdin.write(request_text)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                raise RuntimeError(f"MCP server pipe broken: {e}") from e

        # Wait for the reader thread to resolve our Future
        try:
            return fut.result(timeout=60)
        except TimeoutError:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP '{method}' timed out after 60s")
        except Exception:
            raise

    def _reader_loop(self):
        """
        Background thread: reads stdout, matches by request ID, resolves Futures.
        """
        while self._running and self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue

                req_id = response.get("id")
                if req_id is None:
                    continue

                with self._pending_lock:
                    fut = self._pending.pop(req_id, None)

                if fut is not None and not fut.done():
                    if "error" in response:
                        err = response["error"]
                        fut.set_exception(
                            RuntimeError(f"MCP error [{err.get('code')}]: {err.get('message')}")
                        )
                    else:
                        fut.set_result(response.get("result", {}))
            except Exception:
                if self._running:
                    logger.exception("Reader thread error, retrying...")
                time.sleep(0.1)

        # Clean up unresolved futures on shutdown
        with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("MCP server closed"))
            self._pending.clear()


# ============================================================================
# Singleton MCP client
# ============================================================================

_mcp_client: MCPClientManager | None = None


def get_mcp_client() -> MCPClientManager:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClientManager()
        _mcp_client.start()
    return _mcp_client


def shutdown_mcp_client():
    global _mcp_client
    if _mcp_client is not None:
        _mcp_client.stop()
        _mcp_client = None


# ============================================================================
# LangChain BaseTool wrappers
# ============================================================================

def _json_type_to_python(json_type: str) -> type:
    mapping = {
        "string": str, "integer": int, "number": float,
        "boolean": bool, "array": list, "object": dict,
    }
    return mapping.get(json_type, str)


def _build_args_schema(tool_def: dict) -> type[BaseModel]:
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
            if isinstance(default_val, (str, int, float)):
                fields[prop_name] = (prop_type, Field(default=default_val, description=description))
            else:
                fields[prop_name] = (prop_type, Field(default=None, description=description))

    model_name = f"{tool_def['name'].replace('_', ' ').title().replace(' ', '')}Input"
    return create_model(model_name, **fields)


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
        t0 = time.perf_counter()
        try:
            result = get_mcp_client().call_tool(self.mcp_tool_name, kwargs)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"  ↳ {self.mcp_tool_name} {ms:.0f}ms")
            return result
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            logger.error(f"  ↳ {self.mcp_tool_name} {ms:.0f}ms FAILED: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ============================================================================
# Discovery
# ============================================================================

def discover_mcp_tools(client: MCPClientManager | None = None) -> list[BaseTool]:
    if client is None:
        client = get_mcp_client()

    definitions = client.get_tool_definitions()
    tools = []

    for tool_def in definitions:
        wrapper = MCPToolWrapper(tool_def=tool_def)
        tools.append(wrapper)
        logger.info(f"  MCP tool loaded: {wrapper.name}")

    return tools
