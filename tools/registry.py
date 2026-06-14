"""
Travel tool registry — 5 tools with Pydantic input validation.

Tools:
  flight_search   — search flights (SerpAPI + Bing engine)
  hotel_search    — search hotels (SerpAPI + Bing engine)
  attraction_search — search attractions & restaurants (SerpAPI + Bing engine)
  weather_check   — lookup weather for destination/dates (SerpAPI + Bing engine)
  budget_calculator — run Python code to compute trip cost breakdown
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import subprocess
import json
import os
import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="Departure city, e.g. 'Beijing'")
    destination: str = Field(..., description="Arrival city, e.g. 'Chengdu'")
    date: str = Field(..., description="Departure date in YYYY-MM-DD format")


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
    code: str = Field(..., description="Python code that defines a dict 'cost' with keys: flight, hotel, food, transport, attractions, total, and prints total breakdown")
    timeout_seconds: int = Field(default=10, ge=1, le=30)


# ---------------------------------------------------------------------------
# Shared SerpAPI helper (supports Google / Bing / Baidu engines)
# ---------------------------------------------------------------------------

_SERPAPI_URL = "https://serpapi.com/search"

def _serpapi_search(query: str, num: int = 5) -> list[dict]:
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return [{"error": "SERPAPI_API_KEY not configured"}]
    try:
        resp = httpx.get(
            _SERPAPI_URL,
            params={
                "q": query,
                "engine": "bing",     # ← switch to "google" / "baidu" anytime
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


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@tool(args_schema=FlightSearchInput)
def flight_search(origin: str, destination: str, date: str) -> str:
    """
    Search for flights between two cities on a specific date.
    Returns top results with price estimates, airlines, and departure info.
    """
    query = f"{origin} to {destination} flight {date} price"
    logger.info(f"[flight_search] {query}")
    results = _serpapi_search(query, num=5)
    return json.dumps({"query": f"{origin} -> {destination} on {date}", "flights": results}, ensure_ascii=False)


@tool(args_schema=HotelSearchInput)
def hotel_search(city: str, check_in: str, check_out: str, max_price_per_night: int = 500) -> str:
    """
    Search for hotels in a city for given dates and budget.
    Returns top options with ratings, prices, and location.
    """
    query = f"{city} hotel {check_in} to {check_out} under {max_price_per_night} CNY per night"
    logger.info(f"[hotel_search] {query}")
    results = _serpapi_search(query, num=5)
    return json.dumps({
        "city": city,
        "dates": f"{check_in} ~ {check_out}",
        "max_price_per_night": max_price_per_night,
        "hotels": results,
    }, ensure_ascii=False)


@tool(args_schema=AttractionSearchInput)
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
    return json.dumps({"city": city, "category": category, "results": results}, ensure_ascii=False)


@tool(args_schema=WeatherCheckInput)
def weather_check(city: str, date: str = "") -> str:
    """
    Look up weather forecast for a city on a specific date.
    Useful for packing advice and activity planning.
    """
    query = f"{city} weather forecast {date}" if date else f"{city} weather today"
    logger.info(f"[weather_check] {query}")
    results = _serpapi_search(query, num=3)
    return json.dumps({"city": city, "date": date or "current", "weather": results}, ensure_ascii=False)


@tool(args_schema=BudgetCalculatorInput)
def budget_calculator(code: str, timeout_seconds: int = 10) -> str:
    """
    Execute Python code in a sandbox to calculate trip budget.
    The code MUST define a dict 'cost' and print a breakdown.
    Example:
      cost = {"flight": 1200, "hotel": 1500, "food": 800, "total": 3500}
      for k, v in cost.items():
          print(f"{k}: {v} CNY")
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = {
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
        logger.info(f"[budget_calculator] returncode={result.returncode}")
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Calculation timed out after {timeout_seconds}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_TOOLS = [flight_search, hotel_search, attraction_search, weather_check, budget_calculator]


def get_all_tools():
    return _ALL_TOOLS
