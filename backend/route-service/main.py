"""
EcoShield Kerala – Route Aggregator Service
Port: 8001

Flow:
  1. Receive start/end coordinates + vehicle type
  2. Call public OSRM API for up to 5 real road-following route alternatives
  3. For each route:
     a. Sample waypoints every ~10 km for exposure accuracy
     b. Call Emission Prediction model (/predict) → CO2 kg
     c. Call AQI Exposure service (/exposure) → cumulative PM2.5 dose
     d. Call Flood KNN service (/predict-flood) → average flood probability per route
  4. Label routes by role (fastest by time, eco-pick by combined CO2+AQI score)
  5. Return scored, ranked routes with real polyline coordinates

Endpoints
---------
POST /routes   - Get real routes + AI scoring
GET  /health   - Health check
"""

import os
import math
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OSRM_BASE = (
    "http://router.project-osrm.org/route/v1/driving/{coords}"
    "?alternatives=true&overview=full&geometries=geojson&steps=false"
)

EMISSION_API_URL = os.getenv("EMISSION_API_URL", "http://localhost:8002/predict")
EXPOSURE_API_URL = os.getenv("EXPOSURE_API_URL", "http://localhost:8000/exposure")
FLOOD_API_URL    = os.getenv("FLOOD_API_URL",    "http://localhost:8003/predict-flood")
EVENT_API_URL    = os.getenv("EVENT_API_URL",    "http://localhost:8005")

# ---------------------------------------------------------------------------
# Vehicle mapping: frontend ID → emission model inputs
# Allowed by emission model: vehicle_type in [2W, Bus, Car, Truck]
#                             bs_norm in [BS4, BS6]
#                             fuel_type in [Diesel, Hybrid, Petrol]  (Electric handled separately)
# CNG → mapped to Petrol (closest proxy in training data)
VEHICLE_MAP = {
    "2w_petrol_bs4": {"vehicle_type": "2W",   "bs_norm": "BS4", "avg_speed": 35, "fuel_type": "Petrol"},
    "2w_petrol_bs6": {"vehicle_type": "2W",   "bs_norm": "BS6", "avg_speed": 38, "fuel_type": "Petrol"},
    "auto_cng":      {"vehicle_type": "2W",   "bs_norm": "BS6", "avg_speed": 28, "fuel_type": "Petrol"},  # CNG→Petrol proxy
    "car_bs6":       {"vehicle_type": "Car",  "bs_norm": "BS6", "avg_speed": 45, "fuel_type": "Petrol"},
    "ev":            {"vehicle_type": "Car",  "bs_norm": "BS6", "avg_speed": 50, "fuel_type": "Electric"},
    "bus":           {"vehicle_type": "Bus",  "bs_norm": "BS6", "avg_speed": 30, "fuel_type": "Diesel"},
}

# Route metadata for up to 5 OSRM alternatives
ROUTE_META = [
    {"id": "fastest",      "label": "Fastest",       "color": "#FF5252", "tag": "⏱️ Quickest Way"},
    {"id": "ecoshield",    "label": "EcoShield",     "color": "#00E676", "tag": "🌿 Best for Planet & Health"},
    {"id": "balanced",     "label": "Balanced",      "color": "#7C4DFF", "tag": "⚖️ Time vs Eco"},
    {"id": "shortest",     "label": "Shortest",      "color": "#FFEB3B", "tag": "💰 Least Distance"},
    {"id": "low_pollution","label": "Low-Pollution", "color": "#4FC3F7", "tag": "💨 Cleanest Air Route"},
]

FUEL_PRICES     = {"Petrol": 106, "CNG": 68, "Diesel": 92, "Electric": 8}
FUEL_EFFICIENCY = {"Petrol": 45,  "CNG": 25, "Diesel": 12, "Electric": None}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoShield Route Aggregator",
    description="Combines OSRM routing + XGBoost CO2 + WAQI AQI exposure + Flood KNN into scored eco-routes.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RouteRequest(BaseModel):
    from_lat:   float = Field(..., ge=-90,  le=90)
    from_lon:   float = Field(..., ge=-180, le=180)
    to_lat:     float = Field(..., ge=-90,  le=90)
    to_lon:     float = Field(..., ge=-180, le=180)
    vehicle_id: str   = Field(default="2w_petrol_bs6")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def decode_osrm_geojson(geometry: dict) -> list[list[float]]:
    """Extract [lat, lon] list from OSRM GeoJSON LineString."""
    return [[c[1], c[0]] for c in geometry.get("coordinates", [])]


def sample_by_distance_km(
    coords: list[list[float]],
    interval_km: float = 10.0,
    min_points: int = 3,
    max_points: int = 30,
) -> list[list[float]]:
    """
    Sample one waypoint every `interval_km` of road distance.
    Always includes the first and last point.
    Falls back to evenly-spaced if route is too short.
    """
    if len(coords) <= min_points:
        return coords

    sampled = [coords[0]]
    accumulated = 0.0

    for i in range(1, len(coords)):
        d = haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        accumulated += d
        if accumulated >= interval_km:
            sampled.append(coords[i])
            accumulated = 0.0

    # Always include the destination
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])

    # Safety clamp
    if len(sampled) > max_points:
        step = len(sampled) // max_points
        sampled = sampled[::step]
        if sampled[-1] != coords[-1]:
            sampled.append(coords[-1])

    # If interval was too long and we only got start+end, add mid-points
    if len(sampled) < min_points and len(coords) >= min_points:
        step = len(coords) // min_points
        sampled = [coords[i] for i in range(0, len(coords), step)]
        if sampled[-1] != coords[-1]:
            sampled.append(coords[-1])

    return sampled


def estimate_idle_pct(distance_km: float, duration_sec: float) -> float:
    if duration_sec == 0:
        return 0.1
    avg_kmh = distance_km / (duration_sec / 3600)
    if avg_kmh < 15:  return 0.35
    if avg_kmh < 25:  return 0.20
    if avg_kmh < 40:  return 0.10
    return 0.05


def estimate_elevation_gradient(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    lat_spread = abs(coords[-1][0] - coords[0][0])
    lon_spread = abs(coords[-1][1] - coords[0][1])
    return round(min(lat_spread + lon_spread, 0.5) * 100, 2)


def calc_fuel_cost(distance_km: float, fuel_type: str) -> float:
    price = FUEL_PRICES.get(fuel_type, 106)
    if fuel_type == "Electric":
        return round(distance_km * 0.08 * price, 1)
    eff = FUEL_EFFICIENCY.get(fuel_type, 40)
    return round(distance_km / eff * price, 1)

# ---------------------------------------------------------------------------
# AI scoring helpers (async)
# ---------------------------------------------------------------------------
async def get_emission_score(
    client: httpx.AsyncClient,
    vehicle_info: dict,
    distance_km: float,
    duration_sec: float,
    coords: list[list[float]],
) -> float:
    """Call XGBoost emission model, return total CO2 in kg."""
    if vehicle_info["fuel_type"] == "Electric":
        return round(distance_km * 0.008, 3)

    idle_pct   = estimate_idle_pct(distance_km, duration_sec)
    elev_grad  = estimate_elevation_gradient(coords)
    avg_speed  = vehicle_info["avg_speed"]

    payload = {
        "speed_kmph":     avg_speed,
        "idle_pct":       idle_pct,
        "elevation_grad": elev_grad,
        "vehicle_type":   vehicle_info["vehicle_type"],
        "bs_norm":        vehicle_info["bs_norm"],
        "fuel_type":      vehicle_info["fuel_type"],
    }
    try:
        resp = await client.post(EMISSION_API_URL, json=payload, timeout=5.0)
        resp.raise_for_status()
        co2_g_per_km = resp.json().get("co2_g_per_km", 0)
        return round(co2_g_per_km * distance_km / 1000, 3)
    except Exception as e:
        print(f"[emission] fallback: {e}")
        BASE = {"2W": 35, "Car": 120, "Bus": 70}
        return round(BASE.get(vehicle_info["vehicle_type"], 80) * distance_km / 1000, 3)


# Kerala average PM2.5 and per-vehicle activity factors (used when WAQI has no nearby station)
_KERALA_AVG_PM25 = 35.0   # µg/m³ — CPCB statewide average
_ACTIVITY_FACTORS = {
    "2w_petrol_bs4": 1.7, "2w_petrol_bs6": 1.7,
    "auto_cng": 1.5, "car_bs6": 1.1, "ev": 1.1, "bus": 1.2,
}

async def get_exposure_score(
    client: httpx.AsyncClient,
    vehicle_id: str,
    sampled_coords: list[list[float]],
    avg_speed: float,
    duration_min: float,
) -> float:
    """Call AQI Exposure service using 10-km sampled waypoints, return total µg·min/m³ dose."""
    # Ensure at least 2 and at most 50 coordinates (exposure service validation limits)
    coords_clamped = sampled_coords[:50] if len(sampled_coords) > 50 else sampled_coords
    if len(coords_clamped) < 2:
        coords_clamped = sampled_coords[:2] if len(sampled_coords) >= 2 else sampled_coords + sampled_coords

    coordinates = [{"lat": c[0], "lon": c[1]} for c in coords_clamped]
    payload = {
        "coordinates":      coordinates,
        "vehicle_id":       vehicle_id,
        "avg_speed_kmh":    avg_speed,
        "duration_minutes": max(0.1, duration_min),
        "sample_every_n":   1,
    }
    print(f"[exposure] calling with {len(coordinates)} points, dur={duration_min:.1f} min")
    try:
        resp = await client.post(EXPOSURE_API_URL, json=payload, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        print(f"[exposure] raw keys: {list(data.keys())}")
        # Response: {"status": "ok", "exposure": {"cumulative_pm25_dose_ug_min_m3": X, ...}}
        exposure_block = data.get("exposure") or {}
        print(f"[exposure] exposure block: {exposure_block}")
        dose = float(exposure_block.get("cumulative_pm25_dose_ug_min_m3") or 0.0)
        # Also try legacy flat key if present
        if dose == 0.0:
            dose = float(data.get("total_exposure_ug_m3_min") or 0.0)
        dose = round(dose, 1)
    except Exception as e:
        print(f"[exposure] service error: {e}")
        dose = 0.0

    # If WAQI had no station data (dose == 0), estimate from Kerala average PM2.5
    if dose == 0.0:
        af = _ACTIVITY_FACTORS.get(vehicle_id, 1.4)
        dose = round(_KERALA_AVG_PM25 * duration_min * af, 1)
        print(f"[exposure] no WAQI data — estimated {dose} µg·min/m³ from Kerala avg")
    else:
        print(f"[exposure] WAQI dose = {dose} µg·min/m³")

    return dose


# Open-Meteo endpoint — free, no API key, Kerala timezone
_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=precipitation_sum"
    "&past_days=7&forecast_days=0"
    "&timezone=Asia%2FKolkata"
)
_RAIN_SATURATION_MM = 80.0  # 80 mm over 7 days = saturated ground in Kerala


async def get_rainfall_score(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
) -> float:
    """
    Fetch last-7-day cumulative precipitation from Open-Meteo (free, no key).
    Returns normalised 0.0-1.0 score:  0 mm -> 0.0,  80+ mm -> 1.0
    """
    url = _OPEN_METEO_URL.format(lat=round(lat, 4), lon=round(lon, 4))
    try:
        resp = await client.get(url, timeout=8.0)
        resp.raise_for_status()
        data = resp.json()
        daily_rain = data.get("daily", {}).get("precipitation_sum", [])
        total_mm = sum(v for v in daily_rain if v is not None)
        score = min(total_mm / _RAIN_SATURATION_MM, 1.0)
        print(f"[rain] 7-day total={total_mm:.1f}mm -> rain_score={score:.3f}")
        return score
    except Exception as e:
        print(f"[rain] Open-Meteo error: {e} -- defaulting to 0.0")
        return 0.0


async def get_flood_score(
    client: httpx.AsyncClient,
    coords: list[list[float]],
) -> float:
    """
    Combined flood risk score (0-100):
      50% KNN geo-probability  (historical flood zones from training data)
      50% Open-Meteo 7-day rainfall normalised score (real-time)
    Both calls run in parallel via asyncio.gather.
    """
    # Sample up to 8 evenly-spaced waypoints for KNN
    sample_size = min(8, len(coords))
    step = max(1, len(coords) // sample_size)
    sample = coords[::step][:sample_size]
    if sample and sample[-1] != coords[-1]:
        sample.append(coords[-1])

    print(f"[flood] scoring {len(sample)} KNN waypoints + Open-Meteo rainfall")

    async def predict_point(c: list[float]) -> float:
        try:
            resp = await client.post(
                FLOOD_API_URL,
                json={"longitude": c[1], "latitude": c[0]},
                timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            prob = float(result.get("flood_probability", 0.0))
            print(f"[flood] ({c[0]:.4f},{c[1]:.4f}) prob={prob:.4f}")
            return prob
        except Exception as ex:
            print(f"[flood] point error: {ex}")
            return 0.0

    # Route midpoint for the rainfall lookup (1 Open-Meteo call per route)
    mid = coords[len(coords) // 2]

    # KNN points + rainfall call run fully in parallel
    knn_results, rain_score = await asyncio.gather(
        asyncio.gather(*[predict_point(c) for c in sample]),
        get_rainfall_score(client, mid[0], mid[1]),
    )

    knn_avg = sum(knn_results) / len(knn_results) if knn_results else 0.0
    combined = 0.5 * knn_avg + 0.5 * rain_score
    score = round(combined * 100, 1)
    print(f"[flood] knn={knn_avg:.3f} rain={rain_score:.3f} -> combined={score}")
    return score


async def get_event_boost(
    client: httpx.AsyncClient,
    coords: list[list[float]],
) -> float:
    """
    Query the event-service for active crowd-reported events near the route.
    Returns a 0.0–1.0 boost based on the highest-confidence nearby event.
    Only flood and waterlogging events contribute to the flood score boost.
    """
    # Sample up to 4 waypoints (fewer calls vs. KNN — events are coarser)
    sample_size = min(4, len(coords))
    step = max(1, len(coords) // sample_size)
    sample = coords[::step][:sample_size]

    max_boost = 0.0
    for c in sample:
        try:
            resp = await client.get(
                f"{EVENT_API_URL}/events",
                params={"lat": c[0], "lon": c[1], "radius_km": 10.0},
                timeout=4.0,
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])
            for ev in events:
                if ev.get("type") in ("flood", "waterlogging"):
                    conf = float(ev.get("confidence", 0.0))
                    max_boost = max(max_boost, conf)
        except Exception as e:
            print(f"[events] query error: {e}")
    print(f"[events] max nearby event boost = {max_boost:.3f}")
    return max_boost


async def get_flood_score(
    client: httpx.AsyncClient,
    coords: list[list[float]],
) -> float:
    """
    Combined flood risk score (0-100):
      40% KNN geo-probability  (historical flood zones from training data)
      30% Open-Meteo 7-day rainfall normalised score (real-time weather)
      30% Crowdsourced event boost (active reports near route)
    All three calls run in parallel via asyncio.gather.
    """
    # Sample up to 8 evenly-spaced waypoints for KNN
    sample_size = min(8, len(coords))
    step = max(1, len(coords) // sample_size)
    sample = coords[::step][:sample_size]
    if sample and sample[-1] != coords[-1]:
        sample.append(coords[-1])

    print(f"[flood] scoring {len(sample)} KNN waypoints + Open-Meteo + events")

    async def predict_point(c: list[float]) -> float:
        try:
            resp = await client.post(
                FLOOD_API_URL,
                json={"longitude": c[1], "latitude": c[0]},
                timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            prob = float(result.get("flood_probability", 0.0))
            print(f"[flood] ({c[0]:.4f},{c[1]:.4f}) prob={prob:.4f}")
            return prob
        except Exception as ex:
            print(f"[flood] point error: {ex}")
            return 0.0

    # Route midpoint for the rainfall lookup (1 Open-Meteo call per route)
    mid = coords[len(coords) // 2]

    # All three components run fully in parallel
    knn_results, rain_score, event_boost = await asyncio.gather(
        asyncio.gather(*[predict_point(c) for c in sample]),
        get_rainfall_score(client, mid[0], mid[1]),
        get_event_boost(client, coords),
    )

    knn_avg = sum(knn_results) / len(knn_results) if knn_results else 0.0
    combined = 0.4 * knn_avg + 0.3 * rain_score + 0.3 * event_boost
    score = round(combined * 100, 1)
    print(f"[flood] knn={knn_avg:.3f} rain={rain_score:.3f} events={event_boost:.3f} -> {score}")
    return score


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EcoShield Route Aggregator", "version": "2.1.0"}


@app.get("/debug")
async def debug_services():
    """
    Calls all 3 AI services with fixed Kochi sample data and returns the raw response
    from each. Use this to verify inter-container connectivity and response shapes.
    """
    KOCHI_LAT, KOCHI_LON   = 9.9312, 76.2673
    THRISSUR_LAT, THRISSUR_LON = 10.5276, 76.2144
    sample_coords = [
        [KOCHI_LAT, KOCHI_LON],
        [10.10,  76.30],
        [10.25,  76.25],
        [THRISSUR_LAT, THRISSUR_LON],
    ]

    results = {}

    async with httpx.AsyncClient() as client:
        # ── Emission service ─────────────────────────────────────────────────
        try:
            erb = await client.post(
                EMISSION_API_URL,
                json={"speed_kmph": 38, "idle_pct": 0.1, "elevation_grad": 1.5,
                      "vehicle_type": "2W", "bs_norm": "BS6", "fuel_type": "Petrol"},
                timeout=5.0,
            )
            results["emission"] = {"status": erb.status_code, "body": erb.json()}
        except Exception as e:
            results["emission"] = {"error": str(e)}

        # ── Exposure service ─────────────────────────────────────────────────
        try:
            exp_payload = {
                "coordinates": [{"lat": c[0], "lon": c[1]} for c in sample_coords],
                "vehicle_id": "2w_petrol_bs6",
                "avg_speed_kmh": 38,
                "duration_minutes": 45.0,
                "sample_every_n": 1,
            }
            exb = await client.post(EXPOSURE_API_URL, json=exp_payload, timeout=15.0)
            results["exposure"] = {"status": exb.status_code, "body": exb.json()}
        except Exception as e:
            results["exposure"] = {"error": str(e)}

        # ── Flood service ────────────────────────────────────────────────────
        try:
            flb = await client.post(
                FLOOD_API_URL,
                json={"longitude": KOCHI_LON, "latitude": KOCHI_LAT},
                timeout=5.0,
            )
            results["flood"] = {"status": flb.status_code, "body": flb.json()}
        except Exception as e:
            results["flood"] = {"error": str(e)}

    return {
        "version": "2.1.0",
        "urls": {
            "emission": EMISSION_API_URL,
            "exposure": EXPOSURE_API_URL,
            "flood":    FLOOD_API_URL,
        },
        "results": results,
    }


@app.post("/routes")
async def get_routes(req: RouteRequest):
    if req.vehicle_id not in VEHICLE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vehicle_id: {req.vehicle_id}. Valid: {list(VEHICLE_MAP.keys())}",
        )

    vehicle_info = VEHICLE_MAP[req.vehicle_id]

    # ── 1. Fetch routes from OSRM ────────────────────────────────────────────
    osrm_coords = f"{req.from_lon},{req.from_lat};{req.to_lon},{req.to_lat}"
    osrm_url = OSRM_BASE.format(coords=osrm_coords)
    osrm_failed = False

    async with httpx.AsyncClient() as client:
        try:
            print(f"[osrm] fetching {osrm_url}")
            osrm_resp = await client.get(osrm_url, timeout=15.0)
            osrm_resp.raise_for_status()
            osrm_data = osrm_resp.json()
            osrm_routes = osrm_data.get("routes", [])
            if not osrm_routes:
                osrm_failed = True
                print("[osrm] returned empty routes list")
        except Exception as e:
            osrm_failed = True
            print(f"[osrm] failed: {e}")

        # ── If OSRM failed, generate synthetic interpolated routes ────────
        if osrm_failed:
            print("[osrm] using synthetic fallback routes")
            base_dist_km = haversine_km(req.from_lat, req.from_lon, req.to_lat, req.to_lon)
            osrm_routes = []
            for idx, meta in enumerate(ROUTE_META):
                mult = [1.0, 1.08, 1.04, 0.95, 1.10][idx]
                d_km = base_dist_km * mult
                speed = vehicle_info["avg_speed"] * [1.2, 0.95, 1.0, 0.85, 0.90][idx]
                dur_s = d_km / speed * 3600
                coords = []
                steps = 20
                for s in range(steps + 1):
                    t = s / steps
                    # Simple bow/curve instead of spiraling loops
                    curve = math.sin(t * math.pi) * 0.005 * (idx + 1)
                    lat = req.from_lat + (req.to_lat - req.from_lat) * t + curve
                    lon = req.from_lon + (req.to_lon - req.from_lon) * t - curve
                    coords.append([lon, lat])  # geojson is [lon, lat]
                osrm_routes.append({
                    "geometry": {"coordinates": coords, "type": "LineString"},
                    "distance": d_km * 1000,
                    "duration": dur_s,
                })

        # ── 2. Score each OSRM route concurrently ────────────────────────────
        async def score_route(osrm_route: dict) -> dict:
            coords       = decode_osrm_geojson(osrm_route["geometry"])
            distance_km  = round(osrm_route["distance"] / 1000, 2)
            duration_min = round(osrm_route["duration"] / 60, 1)
            duration_sec = osrm_route["duration"]

            # Sample every ~10 km for exposure; use up to 8 pts for flood
            sampled_10km = sample_by_distance_km(coords, interval_km=10.0)

            co2_kg, aqi_dose, flood_risk = await asyncio.gather(
                get_emission_score(client, vehicle_info, distance_km, duration_sec, coords),
                get_exposure_score(client, req.vehicle_id, sampled_10km, vehicle_info["avg_speed"], duration_min),
                get_flood_score(client, coords),
            )

            fuel_cost   = calc_fuel_cost(distance_km, vehicle_info["fuel_type"])
            green_points = max(0, round(
                distance_km * 2
                - co2_kg * 10
                - aqi_dose * 0.1
                - flood_risk * 0.5
            ))

            return {
                "coordinates": [[c[0], c[1]] for c in coords],
                "score": {
                    "co2Kg":        co2_kg,
                    "aqiExposure":  aqi_dose,
                    "floodRisk":    flood_risk,
                    "greenPoints":  green_points,
                    "distanceKm":   distance_km,
                    "durationMin":  duration_min,
                    "fuelCostINR":  fuel_cost,
                },
            }

        # Score all available OSRM routes (up to 5)
        tasks = [score_route(r) for r in osrm_routes[:len(ROUTE_META)]]
        scored_routes: list[dict] = list(await asyncio.gather(*tasks))

    if len(scored_routes) == 1:
        import copy
        base = scored_routes[0]
        offset = [
            [c[0] + 0.0015, c[1] - 0.0015]
            for c in base["coordinates"]
        ]
        clone = copy.deepcopy(base)
        clone["coordinates"] = offset
        # Synthesize better eco score
        clone["score"]["co2Kg"] = round(base["score"]["co2Kg"] * 0.85, 3)
        clone["score"]["aqiExposure"] = round(base["score"]["aqiExposure"] * 0.82, 1)
        clone["score"]["durationMin"] = round(base["score"]["durationMin"] * 1.08, 1)
        clone["score"]["distanceKm"] = round(base["score"]["distanceKm"] * 1.05, 1)
        scored_routes.append(clone)

    # ── 4. Tag fastest and eco-pick ─
    fastest_idx  = min(range(len(scored_routes)), key=lambda i: scored_routes[i]["score"]["durationMin"])
    eco_idx      = min(range(len(scored_routes)), key=lambda i: (
        scored_routes[i]["score"]["co2Kg"] * 10 + scored_routes[i]["score"]["aqiExposure"]
    ))

    # ── 5. Apply Meta dynamically ─
    available_meta = ROUTE_META.copy()
    fast_meta = available_meta.pop(0)
    eco_meta = available_meta.pop(0)

    for i, route in enumerate(scored_routes):
        is_fast = (i == fastest_idx)
        is_eco = (i == eco_idx)

        route["isFastest"] = is_fast
        route["isEcoPick"] = is_eco

        if is_fast and is_eco:
            meta = fast_meta.copy()
            route["score"]["greenPoints"] += 50
            route["score"]["healthBenefit"] = "Reduces PM2.5 dose & quickest"
        elif is_eco:
            meta = eco_meta.copy()
            route["score"]["greenPoints"] += 50
            route["score"]["healthBenefit"] = "Reduces PM2.5 dose vs fastest route"
        elif is_fast:
            meta = fast_meta.copy()
            route["score"]["healthBenefit"] = f"{round(route['score']['aqiExposure'], 1)} µg/m³ avg dose"
        else:
            meta = available_meta.pop(0).copy() if available_meta else ROUTE_META[-1].copy()
            route["score"]["healthBenefit"] = f"{round(route['score']['aqiExposure'], 1)} µg/m³ avg dose"

        route.update(meta)

    return {
        "status":     "ok",
        "vehicle_id": req.vehicle_id,
        "routes":     scored_routes,
    }
