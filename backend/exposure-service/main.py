"""
EcoShield Kerala – AQI Exposure Microservice
FastAPI backend that accepts route coordinates and returns
real-time cumulative pollutant exposure using WAQI API.

Endpoints
---------
POST /exposure          - Full exposure calculation for a route
GET  /aqi/point         - Single-point AQI lookup
GET  /aqi/stations      - Nearest stations info for a coordinate
GET  /health            - Service health check
"""
import os
import math
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx
from dotenv import load_dotenv

from waqi_client import fetch_aqi_at, fetch_aqi_batch
from exposure_calculator import calculate_exposure

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EcoShield Kerala – AQI Exposure Microservice",
    description=(
        "Calculates real-time personal PM2.5/NOx exposure along a route "
        "using WAQI geo-API. Supports EcoShield multi-objective route scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # React Native (Expo) and web clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class ExposureRequest(BaseModel):
    """
    Payload for POST /exposure

    Send the sampled waypoints along your route path —
    e.g. every 2–5 km. The service will fetch live AQI for
    each point and compute cumulative personal exposure.
    """
    coordinates: list[Coordinate] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Ordered list of route waypoints (2–50 points)"
    )
    vehicle_id: str = Field(
        default="2w_petrol_bs6",
        description=(
            "Vehicle type: 2w_petrol_bs4 | 2w_petrol_bs6 | "
            "auto_cng | car_bs6 | ev | bus"
        )
    )
    avg_speed_kmh: float = Field(
        default=35.0,
        ge=5,
        le=120,
        description="Average travel speed (km/h)"
    )
    duration_minutes: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override trip duration if known (minutes)"
    )
    sample_every_n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Sample every N-th coordinate to limit API calls"
    )

    @field_validator("vehicle_id")
    @classmethod
    def validate_vehicle(cls, v: str) -> str:
        valid = {"2w_petrol_bs4", "2w_petrol_bs6", "auto_cng", "car_bs6", "ev", "bus"}
        if v not in valid:
            raise ValueError(f"vehicle_id must be one of: {valid}")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Service liveness check."""
    return {
        "status": "ok",
        "service": "EcoShield AQI Exposure Microservice",
        "version": "1.0.0",
        "waqi_token_set": bool(os.getenv("WAQI_TOKEN")),
    }


@app.get("/aqi/point", tags=["AQI"])
async def get_aqi_single_point(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    """
    Get real-time AQI data for a single coordinate.

    Example: GET /aqi/point?lat=9.9312&lon=76.2673
    """
    async with httpx.AsyncClient() as client:
        result = await fetch_aqi_at(lat, lon, client)
    if result is None:
        raise HTTPException(status_code=503, detail="Could not fetch AQI data for this location")
    return {"status": "ok", "coordinate": {"lat": lat, "lon": lon}, "aqi_data": result}


@app.post("/exposure", tags=["Exposure"])
async def calculate_route_exposure(req: ExposureRequest):
    """
    Calculate cumulative personal AQI/pollutant exposure along a route.

    **How it works:**
    1. For each waypoint (or sampled subset), fetches live AQI from WAQI API
    2. Calculates dose-weighted exposure:
       `dose = PM2.5_concentration × time_at_segment × activity_factor`
    3. Sums across all segments to give total personal exposure
    4. Returns segment breakdown, hotspots, WHO classification, health risk score

    **Tip:** Send 10–30 evenly spaced waypoints along the route for best accuracy.
    """
    # Optionally sub-sample coordinates to reduce API calls
    coords = req.coordinates
    if req.sample_every_n > 1:
        coords = coords[::req.sample_every_n]
        # Always include the last point
        if coords[-1] != req.coordinates[-1]:
            coords = list(coords) + [req.coordinates[-1]]

    coord_tuples = [(c.lat, c.lon) for c in coords]

    # Fetch live AQI for all waypoints concurrently
    aqi_readings = await fetch_aqi_batch(coord_tuples, max_concurrent=5)

    # Calculate exposure
    coord_dicts = [{"lat": c.lat, "lon": c.lon} for c in coords]
    result = calculate_exposure(
        coordinates=coord_dicts,
        aqi_readings=aqi_readings,
        vehicle_id=req.vehicle_id,
        avg_speed_kmh=req.avg_speed_kmh,
        duration_minutes=req.duration_minutes,
    )

    return {
        "status": "ok",
        "request": {
            "total_coordinates_sent": len(req.coordinates),
            "coordinates_sampled": len(coords),
            "vehicle_id": req.vehicle_id,
            "avg_speed_kmh": req.avg_speed_kmh,
        },
        **result,
    }


@app.post("/exposure/compare", tags=["Exposure"])
async def compare_routes_exposure(
    routes: list[ExposureRequest],
):
    """
    Compare exposure across multiple route alternatives (max 3).

    Send a list of routes (Fastest, Cheapest, EcoShield) to get a
    side-by-side exposure comparison — ideal for the Pareto ranking step.
    """
    if len(routes) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 routes for comparison")
    if len(routes) == 0:
        raise HTTPException(status_code=400, detail="At least 1 route required")

    results = []
    for i, route in enumerate(routes):
        coord_tuples = [(c.lat, c.lon) for c in route.coordinates]
        aqi_readings = await fetch_aqi_batch(coord_tuples, max_concurrent=5)
        coord_dicts = [{"lat": c.lat, "lon": c.lon} for c in route.coordinates]
        exposure = calculate_exposure(
            coordinates=coord_dicts,
            aqi_readings=aqi_readings,
            vehicle_id=route.vehicle_id,
            avg_speed_kmh=route.avg_speed_kmh,
            duration_minutes=route.duration_minutes,
        )
        results.append({
            "route_index": i,
            "vehicle_id": route.vehicle_id,
            **exposure,
        })

    # Rank by health_risk score (lower = better)
    ranked = sorted(results, key=lambda r: r["health_risk"]["score_0_to_100"])
    for rank, r in enumerate(ranked):
        r["exposure_rank"] = rank + 1
        r["is_lowest_exposure"] = rank == 0

    return {"status": "ok", "routes": results, "recommended_route_index": ranked[0]["route_index"]}
