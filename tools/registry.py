"""
Travel tool registry — 5 tool functions with Pydantic input validation.

Tools:
  flight_search   — search flights (SerpAPI + Bing engine)
  hotel_search    — search hotels (SerpAPI + Bing engine)
  attraction_search — search attractions & restaurants (SerpAPI + Bing engine)
  weather_check   — lookup weather for destination/dates (SerpAPI + Bing engine)
  budget_calculator — run Python code to compute trip cost breakdown

These are pure functions — NOT decorated with @tool.
The MCP server wraps them as MCP tools; the agent calls them through MCP protocol.
"""

import subprocess
import json
import os
import asyncio
import httpx
from pydantic import BaseModel, Field
from loguru import logger


# ============================================================================
# Input schemas (Pydantic — shared between MCP server and documentation)
# ============================================================================

class FlightSearchInput(BaseModel):
    origin: str = Field(..., min_length=1, description="Departure city, e.g. 'Beijing'")
    destination: str = Field(..., min_length=1, description="Arrival city, e.g. 'Chengdu'")
    date: str = Field(..., min_length=1, description="Departure date in YYYY-MM-DD format")


class HotelSearchInput(BaseModel):
    city: str = Field(..., description="City name, e.g. 'Chengdu'")
    check_in: str = Field(..., description="Check-in date YYYY-MM-DD")
    check_out: str = Field(..., description="Check-out date YYYY-MM-DD")
    max_price_per_night: int = Field(default=500, description="Max budget per night in CNY")


class AttractionSearchInput(BaseModel):
    city: str = Field(..., description="City name")
    category: str = Field(default="all", description="attractions | food | shopping | all")
    count: int = Field(default=5, ge=1, le=10)


class WeatherCheckInput(BaseModel):
    city: str = Field(..., description="City name")
    date: str = Field(default="", description="Date YYYY-MM-DD, or empty for current")


class BudgetCalculatorInput(BaseModel):
    code: str = Field(..., description="Python code that defines a dict 'cost' and prints breakdown")
    timeout_seconds: int = Field(default=30, ge=1, le=30)


# ============================================================================
# Shared SerpAPI helper (supports Google / Bing / Baidu engines)
# ============================================================================

_SERPAPI_URL = "https://serpapi.com/search"


def _serpapi_search(query: str, num: int = 5) -> list[dict]:
    """同步版本 — 保留向后兼容。新代码使用 _serpapi_search_async。"""
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return [{"error": "SERPAPI_API_KEY not configured"}]
    try:
        resp = httpx.get(
            _SERPAPI_URL,
            params={
                "q": query,
                "engine": "bing",
                "api_key": api_key,
                "num": num,
            },
            timeout=30,
        )
        data = resp.json()
        results = data.get("organic_results", [])[:num]
        return [
            {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
            for r in results
        ]
    except Exception as e:
        logger.error(f"[serpapi] {e}")
        return [{"error": str(e)}]


async def _serpapi_search_async(query: str, num: int = 5) -> list[dict]:
    """异步版本 — MCP Server 使用此版本实现并发 HTTP 请求。
    
    与同步版的关键区别：
    - httpx.AsyncClient + await → 等待响应时不阻塞事件循环
    - 4 个并发调用时，总耗时 = max(4个), 而非 sum(4个)
    """
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return [{"error": "SERPAPI_API_KEY not configured"}]
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                _SERPAPI_URL,
                params={
                    "q": query,
                    "engine": "bing",
                    "api_key": api_key,
                    "num": num,
                },
            )
            data = resp.json()
            results = data.get("organic_results", [])[:num]
            return [
                {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
                for r in results
            ]
    except Exception as e:
        logger.error(f"[serpapi async] {e}")
        return [{"error": str(e)}]


# ============================================================================
# Pure tool function implementations
# ============================================================================

def flight_search(origin: str, destination: str, date: str) -> str:
    """
    Search for flights between two cities on a specific date.
    Returns top results with price estimates, airlines, and departure info.
    """
    query = f"{origin} to {destination} flight {date} price"
    logger.info(f"[flight_search] {query}")
    results = _serpapi_search(query, num=5)
    return json.dumps(
        {"query": f"{origin} -> {destination} on {date}", "flights": results},
        ensure_ascii=False,
    )


def hotel_search(city: str, check_in: str, check_out: str, max_price_per_night: int = 500) -> str:
    """
    Search for hotels in a city for given dates and budget.
    Returns top options with ratings, prices, and location.
    """
    query = f"{city} hotel {check_in} to {check_out} under {max_price_per_night} CNY per night"
    logger.info(f"[hotel_search] {query}")
    results = _serpapi_search(query, num=5)
    return json.dumps(
        {
            "city": city,
            "dates": f"{check_in} ~ {check_out}",
            "max_price_per_night": max_price_per_night,
            "hotels": results,
        },
        ensure_ascii=False,
    )


def attraction_search(city: str, category: str = "all", count: int = 5) -> str:
    """
    Search for must-see attractions, restaurants, and activities in a city.
    category: 'attractions' | 'food' | 'shopping' | 'all'
    """
    cat_map = {
        "attractions": f"{city} top attractions must visit",
        "food": f"{city} best restaurants local food",
        "shopping": f"{city} shopping areas markets",
        "all": f"{city} travel guide attractions food itinerary",
    }
    query = cat_map.get(category, cat_map["all"])
    logger.info(f"[attraction_search] {query}")
    results = _serpapi_search(query, num=count)
    return json.dumps(
        {"city": city, "category": category, "results": results}, ensure_ascii=False
    )


def weather_check(city: str, date: str = "") -> str:
    """
    Look up weather forecast for a city on a specific date.
    Useful for packing advice and activity planning.
    """
    query = f"{city} weather forecast {date}" if date else f"{city} weather today"
    logger.info(f"[weather_check] {query}")
    results = _serpapi_search(query, num=3)
    return json.dumps(
        {"city": city, "date": date or "current", "weather": results}, ensure_ascii=False
    )


def budget_calculator(code: str, timeout_seconds: int = 30) -> str:
    """
    Execute Python code in a sandbox to calculate trip budget.
    Uses exec() in a thread instead of subprocess (Windows startup is too slow).
    """
    import io, threading

    logger.info(f"[budget_calculator] code_len={len(code)}")

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    result = {"rc": 0, "exc": None}

    def _run():
        try:
            with __import__("contextlib").redirect_stdout(out_buf), \
                 __import__("contextlib").redirect_stderr(err_buf):
                exec(code, {"__builtins__": __builtins__}, {})
        except Exception as e:
            result["rc"] = 1
            result["exc"] = str(e)
            print(str(e), file=err_buf)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        preview = code[:300].replace("\n", " ")
        return json.dumps({
            "error": f"Calculation timed out after {timeout_seconds}s",
            "code_preview": f"{preview}...",
            "code_len": len(code),
        })

    output = {
        "stdout": out_buf.getvalue()[:2000],
        "stderr": err_buf.getvalue()[:500],
        "returncode": result["rc"],
    }
    if result["exc"]:
        output["exception"] = result["exc"]
    return json.dumps(output)


# ============================================================================
# Async versions — used by MCP Server for concurrent execution
# ============================================================================

async def flight_search_async(origin: str, destination: str, date: str) -> str:
    query = f"{origin} to {destination} flight {date} price"
    logger.info(f"[flight_search] {query}")
    results = await _serpapi_search_async(query, num=5)
    return json.dumps(
        {"query": f"{origin} -> {destination} on {date}", "flights": results},
        ensure_ascii=False,
    )


async def hotel_search_async(city: str, check_in: str, check_out: str, max_price_per_night: int = 500) -> str:
    query = f"{city} hotel {check_in} to {check_out} under {max_price_per_night} CNY per night"
    logger.info(f"[hotel_search] {query}")
    results = await _serpapi_search_async(query, num=5)
    return json.dumps(
        {"city": city, "dates": f"{check_in} ~ {check_out}",
         "max_price_per_night": max_price_per_night, "hotels": results},
        ensure_ascii=False,
    )


async def attraction_search_async(city: str, category: str = "all", count: int = 5) -> str:
    cat_map = {
        "attractions": f"{city} top attractions must visit",
        "food": f"{city} best restaurants local food",
        "shopping": f"{city} shopping areas markets",
        "all": f"{city} travel guide attractions food itinerary",
    }
    query = cat_map.get(category, cat_map["all"])
    logger.info(f"[attraction_search] {query}")
    results = await _serpapi_search_async(query, num=count)
    return json.dumps(
        {"city": city, "category": category, "results": results}, ensure_ascii=False
    )


async def weather_check_async(city: str, date: str = "") -> str:
    query = f"{city} weather forecast {date}" if date else f"{city} weather today"
    logger.info(f"[weather_check] {query}")
    results = await _serpapi_search_async(query, num=3)
    return json.dumps(
        {"city": city, "date": date or "current", "weather": results}, ensure_ascii=False
    )


async def budget_calculator_async(code: str, timeout_seconds: int = 30) -> str:
    """budget_calculator 无 I/O，直接用同步版。"""
    return budget_calculator(code, timeout_seconds)


# ============================================================================
# Tool metadata — used by MCP server and documentation
# ============================================================================

TOOL_REGISTRY = [
    {
        "name": "flight_search",
        "description": "Search for flights between two cities on a specific date. Returns top results with price estimates, airlines, and departure info.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city, e.g. 'Beijing'"},
                "destination": {"type": "string", "description": "Arrival city, e.g. 'Chengdu'"},
                "date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
            },
            "required": ["origin", "destination", "date"],
        },
        "function": flight_search_async,  # MCP Server 使用异步版
    },
    {
        "name": "hotel_search",
        "description": "Search for hotels in a city for given dates and budget. Returns top options with ratings, prices, and location.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Chengdu'"},
                "check_in": {"type": "string", "description": "Check-in date YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "Check-out date YYYY-MM-DD"},
                "max_price_per_night": {
                    "type": "integer",
                    "description": "Max budget per night in CNY",
                    "default": 500,
                },
            },
            "required": ["city", "check_in", "check_out"],
        },
        "function": hotel_search_async,
    },
    {
        "name": "attraction_search",
        "description": "Search for must-see attractions, restaurants, and activities in a city. category: 'attractions' | 'food' | 'shopping' | 'all'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "category": {
                    "type": "string",
                    "description": "attractions | food | shopping | all",
                    "default": "all",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results (1-10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["city"],
        },
        "function": attraction_search_async,
    },
    {
        "name": "weather_check",
        "description": "Look up weather forecast for a city on a specific date. Useful for packing advice and activity planning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {
                    "type": "string",
                    "description": "Date YYYY-MM-DD, or empty for current weather",
                    "default": "",
                },
            },
            "required": ["city"],
        },
        "function": weather_check_async,
    },
    {
        "name": "budget_calculator",
        "description": "Execute Python code in a sandbox to calculate trip budget. The code MUST define a dict 'cost' and print a breakdown. Example: cost = {'flight': 1200, 'hotel': 1500, 'food': 800, 'total': 3500}; for k, v in cost.items(): print(f'{k}: {v} CNY')",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code that defines a dict 'cost' with keys and prints total breakdown",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout (1-30 seconds)",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": ["code"],
        },
        "function": budget_calculator_async,
    },
]


def get_tool_by_name(name: str):
    """Look up a tool function by name."""
    for t in TOOL_REGISTRY:
        if t["name"] == name:
            return t["function"]
    return None


def get_tool_definitions():
    """Return tool definitions list (without function references) for MCP tools/list."""
    return [
        {k: v for k, v in t.items() if k != "function"}
        for t in TOOL_REGISTRY
    ]
