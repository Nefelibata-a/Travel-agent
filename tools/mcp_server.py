"""
SmartTrip MCP Server — exposes 5 travel tools via MCP protocol (JSON-RPC 2.0 over stdio).

ASYNC EDITION — uses asyncio event loop for concurrent HTTP requests.

Key change from sync version:
  - for line in sys.stdin → async for line in reader  (不阻塞)
  - httpx.get() → await httpx.AsyncClient.get()        (释放事件循环)
  - handle_request() → asyncio.create_task(handle())    (并发执行)

Tools:
  flight_search, hotel_search, attraction_search, weather_check, budget_calculator
"""

import sys
import os
import json
import asyncio
import traceback
from loguru import logger

# ── 修复 Windows 控制台 gbk 编码问题 ──
# 默认 sys.stdout.encoding 在 Windows 上是 'gbk'，写入非 ASCII 字符（如 ™）会崩溃
# MCP 协议通过 stdout 传 JSON，必须是 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.registry import TOOL_REGISTRY, get_tool_definitions

# ---------------------------------------------------------------------------
# MCP Server — Async JSON-RPC 2.0 over stdio
# ---------------------------------------------------------------------------

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

SERVER_NAME = "SmartTrip MCP Server (Async)"
SERVER_VERSION = "2.0.0"

_initialized = False
_write_lock = asyncio.Lock()  # stdout is shared, need lock for writing


async def _send_response(response_id, result):
    async with _write_lock:
        msg = json.dumps({
            "jsonrpc": "2.0", "id": response_id, "result": result,
        }, ensure_ascii=False)
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


async def _send_error(response_id, code, message, data=None):
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    async with _write_lock:
        msg = json.dumps({
            "jsonrpc": "2.0", "id": response_id, "error": error,
        }, ensure_ascii=False)
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_initialize(req_id, params):
    global _initialized
    _initialized = True
    logger.info(f"Client initialized: {params.get('clientInfo', {}).get('name', 'unknown')}")
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {}},
    }


async def handle_tools_list(req_id, params):
    logger.info("tools/list requested")
    return {"tools": get_tool_definitions()}


async def handle_tools_call(req_id, params):
    """
    Async tool call — the KEY concurrency point.

    Unlike the sync version where each tool call blocked the entire server,
    here each tool call is an independent asyncio task.
    4 concurrent tools/call requests → 4 concurrent SerpAPI HTTP calls.
    """
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    logger.info(f"tools/call → {tool_name}({arguments})")

    tool_fn = None
    for t in TOOL_REGISTRY:
        if t["name"] == tool_name:
            tool_fn = t["function"]
            break

    if tool_fn is None:
        return {
            "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
            "isError": True,
        }

    try:
        result = await tool_fn(**arguments)
        return {
            "content": [{"type": "text", "text": result}],
        }
    except TypeError as e:
        logger.error(f"Argument error in {tool_name}: {e}")
        return {
            "content": [{"type": "text", "text": json.dumps({"error": f"Invalid arguments: {str(e)}"})}],
            "isError": True,
        }
    except Exception as e:
        logger.error(f"Error calling {tool_name}: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
            "isError": True,
        }


_METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


async def handle_request(request: dict):
    """Route to handler, then write async response to stdout."""
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    try:
        handler = _METHOD_HANDLERS.get(method)
        if handler is None:
            await _send_error(req_id, -32601, f"Method not found: {method}")
            return

        result = await handler(req_id, params)
        if req_id is not None:
            await _send_response(req_id, result)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc(file=sys.stderr)
        if req_id is not None:
            await _send_error(req_id, -32603, f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# Main async event loop
# ---------------------------------------------------------------------------

async def run_server():
    """Async main loop — reads stdin via thread, dispatches to concurrent tasks."""
    logger.info(f"{SERVER_NAME} v{SERVER_VERSION} starting on stdio...")
    logger.info(f"Available tools: {[t['name'] for t in TOOL_REGISTRY]}")

    while True:
        # asyncio.to_thread: read stdin in a dedicated thread, don't block event loop
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            logger.info("stdin closed, shutting down")
            break

        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            continue

        # KEY: asyncio.create_task() — 不阻塞，立即读下一条 stdin
        # 4 个 tools/call 同时到达 → 4 个并发 SerpAPI 调用
        asyncio.create_task(handle_request(request))


if __name__ == "__main__":
    asyncio.run(run_server())
