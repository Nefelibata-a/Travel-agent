"""
SmartTrip MCP Server — exposes 5 travel tools via MCP protocol (JSON-RPC 2.0 over stdio).

Tools:
  flight_search, hotel_search, attraction_search, weather_check, budget_calculator

Run this as a standalone process:
  python tools/mcp_server.py

The Agent connects to this server via stdio and calls tools through MCP protocol.
"""

import sys
import os
import json
import traceback
from loguru import logger

# Ensure project root is on sys.path (needed when run as standalone subprocess)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.registry import TOOL_REGISTRY, get_tool_definitions

# ---------------------------------------------------------------------------
# MCP Server — JSON-RPC 2.0 over stdio
# ---------------------------------------------------------------------------

# Remove default loguru handler and use stderr for all logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

SERVER_NAME = "SmartTrip MCP Server"
SERVER_VERSION = "1.0.0"

# Track initialization state
_initialized = False


def _send_response(response_id, result):
    """Send a JSON-RPC success response to stdout."""
    msg = json.dumps({
        "jsonrpc": "2.0",
        "id": response_id,
        "result": result,
    }, ensure_ascii=False)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _send_error(response_id, code, message, data=None):
    """Send a JSON-RPC error response to stdout."""
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    msg = json.dumps({
        "jsonrpc": "2.0",
        "id": response_id,
        "error": error,
    }, ensure_ascii=False)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _send_notification(method, params=None):
    """Send a JSON-RPC notification (no id)."""
    msg = json.dumps({
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }, ensure_ascii=False)
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def handle_initialize(req_id, params):
    """Handle initialize handshake."""
    global _initialized
    _initialized = True
    logger.info(f"Client initialized: {params.get('clientInfo', {}).get('name', 'unknown')}")
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "capabilities": {
            "tools": {},
        },
    }


def handle_tools_list(req_id, params):
    """List all available tools."""
    logger.info("tools/list requested")
    return {"tools": get_tool_definitions()}


def handle_tools_call(req_id, params):
    """Call a specific tool with arguments."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    logger.info(f"tools/call → {tool_name}({arguments})")

    # Find the tool function
    tool_entry = None
    for t in TOOL_REGISTRY:
        if t["name"] == tool_name:
            tool_entry = t
            break

    if tool_entry is None:
        return {
            "content": [
                {"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}
            ],
            "isError": True,
        }

    # Call the tool function
    try:
        fn = tool_entry["function"]
        result = fn(**arguments)
        return {
            "content": [
                {"type": "text", "text": result}
            ],
        }
    except TypeError as e:
        logger.error(f"Argument error in {tool_name}: {e}")
        return {
            "content": [
                {"type": "text", "text": json.dumps({"error": f"Invalid arguments for {tool_name}: {str(e)}"})}
            ],
            "isError": True,
        }
    except Exception as e:
        logger.error(f"Error calling {tool_name}: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            "content": [
                {"type": "text", "text": json.dumps({"error": str(e)})}
            ],
            "isError": True,
        }


# Route table
_METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


def handle_request(request: dict):
    """Route a JSON-RPC request to the appropriate handler."""
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    try:
        handler = _METHOD_HANDLERS.get(method)
        if handler is None:
            _send_error(req_id, -32601, f"Method not found: {method}")
            return

        result = handler(req_id, params)
        if req_id is not None:
            _send_response(req_id, result)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc(file=sys.stderr)
        if req_id is not None:
            _send_error(req_id, -32603, f"Internal error: {str(e)}")


def run_server():
    """Main loop — read JSON-RPC messages from stdin, write responses to stdout."""
    logger.info(f"{SERVER_NAME} v{SERVER_VERSION} starting on stdio...")
    logger.info(f"Available tools: {[t['name'] for t in TOOL_REGISTRY]}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Fatal: {e}")
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    run_server()
