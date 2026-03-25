"""
WAQI API client for fetching real-time AQI data by geo-coordinates.
Endpoint: https://api.waqi.info/feed/geo:{lat};{lng}/?token=TOKEN
"""
import os
import asyncio
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "10d5f9816379aaca64f81197383aa09edc7e4a7a")
WAQI_BASE = "https://api.waqi.info/feed/geo:{lat};{lng}/?token={token}"

# Simple in-memory cache to avoid re-fetching the same coordinates
# Key: (rounded_lat, rounded_lon) → data dict
_cache: dict = {}


def _round_coord(val: float, precision: int = 2) -> float:
    """Round coordinate for cache key (0.01° ≈ 1.1 km resolution)."""
    return round(val, precision)


async def fetch_aqi_at(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
    cache_precision: int = 2
) -> Optional[dict]:
    """
    Fetch AQI data from WAQI for a single coordinate.
    Returns a clean dict with pollutant values, or None on failure.
    Uses coordinate rounding to cache nearby lookups.
    """
    cache_key = (_round_coord(lat, cache_precision), _round_coord(lon, cache_precision))
    if cache_key in _cache:
        return _cache[cache_key]

    url = WAQI_BASE.format(lat=lat, lng=lon, token=WAQI_TOKEN)
    try:
        resp = await client.get(url, timeout=8.0)
        resp.raise_for_status()
        body = resp.json()

        if body.get("status") != "ok":
            return None

        data = body["data"]
        iaqi = data.get("iaqi", {})

        result = {
            "station_name": data.get("city", {}).get("name", "Unknown"),
            "station_lat": data.get("city", {}).get("geo", [lat, lon])[0],
            "station_lon": data.get("city", {}).get("geo", [lat, lon])[1],
            "aqi": data.get("aqi"),
            "dominant_pol": data.get("dominentpol"),
            "pm25":  iaqi.get("pm25", {}).get("v"),
            "pm10":  iaqi.get("pm10", {}).get("v"),
            "no2":   iaqi.get("no2",  {}).get("v"),
            "so2":   iaqi.get("so2",  {}).get("v"),
            "co":    iaqi.get("co",   {}).get("v"),
            "o3":    iaqi.get("o3",   {}).get("v"),
            "temperature": iaqi.get("t", {}).get("v"),
            "humidity":    iaqi.get("h", {}).get("v"),
            "timestamp": data.get("time", {}).get("iso"),
        }

        _cache[cache_key] = result
        return result

    except Exception as e:
        print(f"[WAQI] Error fetching ({lat},{lon}): {e}")
        return None


async def fetch_aqi_batch(
    coordinates: list[tuple[float, float]],
    max_concurrent: int = 5
) -> list[Optional[dict]]:
    """
    Fetch AQI for multiple coordinates concurrently.
    Limits concurrency to avoid rate limits.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_sem(lat: float, lon: float, client: httpx.AsyncClient):
        async with semaphore:
            return await fetch_aqi_at(lat, lon, client)

    async with httpx.AsyncClient() as client:
        tasks = [fetch_with_sem(lat, lon, client) for lat, lon in coordinates]
        return await asyncio.gather(*tasks)
