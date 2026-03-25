"""
EcoShield Kerala – Crowdsourced Event Service
Port: 8005

Implements a Google Maps/Waze-style ephemeral event layer:
  - Reports are aggregated by 500m grid cell + type
  - Confidence grows with corroboration (quorum)
  - Events have per-type TTLs and auto-expire
  - No persistent storage — pure in-memory, self-healing

Event types: flood | waterlogging | construction | road_block | accident | debris

Endpoints
---------
POST /report     — ingest a user report
GET  /events     — list active events (optional ?lat=&lon=&radius_km=)
GET  /events/all — all active events (no filter)
DELETE /events/{event_id} — admin: manually expire an event
GET  /health
"""

import asyncio
import math
import uuid
from datetime import datetime, timedelta
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid resolution: ~500m cells (0.005 ° ≈ 555 m at the equator)
GRID_RES = 0.005

# TTL in minutes per event type
TTL_MINUTES: dict[str, int] = {
    "flood":        240,   # 4 h
    "waterlogging": 120,   # 2 h
    "accident":      60,   # 1 h
    "construction": 480,   # 8 h  (shifts don't change fast)
    "road_block":   120,   # 2 h
    "debris":        90,   # 1.5 h
}

# How many reports before confidence = 1.0
QUORUM = 3

# Colour + emoji per type (sent to frontend / Leaflet)
EVENT_META: dict[str, dict] = {
    "flood":        {"color": "#2196F3", "emoji": "🌊", "label": "Flood"},
    "waterlogging": {"color": "#4FC3F7", "emoji": "💧", "label": "Waterlogging"},
    "accident":     {"color": "#FF5252", "emoji": "🚨", "label": "Accident"},
    "construction": {"color": "#FF9800", "emoji": "🚧", "label": "Construction"},
    "road_block":   {"color": "#9C27B0", "emoji": "🚫", "label": "Road Block"},
    "debris":       {"color": "#795548", "emoji": "🪨", "label": "Debris"},
}

# Open-Meteo for rainfall corroboration
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=precipitation_sum&past_days=7&forecast_days=0"
    "&timezone=Asia%2FKolkata"
)

# ---------------------------------------------------------------------------
# In-memory event store
# ---------------------------------------------------------------------------
# Key: (grid_lat, grid_lon, event_type) — one event object per cell+type
_events: dict[str, dict] = {}   # event_id -> event dict


def _snap(lat: float, lon: float) -> tuple[float, float]:
    """Snap coordinates to 500m grid cell centre."""
    return (round(round(lat / GRID_RES) * GRID_RES, 4),
            round(round(lon / GRID_RES) * GRID_RES, 4))


def _find_cell_key(grid_lat: float, grid_lon: float, etype: str) -> Optional[str]:
    """Return existing event_id for this cell+type, or None."""
    for eid, ev in _events.items():
        if ev["grid_lat"] == grid_lat and ev["grid_lon"] == grid_lon and ev["type"] == etype:
            return eid
    return None


def _severity(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.35:
        return "medium"
    return "low"


async def _rainfall_boost(lat: float, lon: float) -> float:
    """0.0–0.3 extra confidence from Open-Meteo 7-day rainfall (floods only)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                OPEN_METEO_URL.format(lat=round(lat, 4), lon=round(lon, 4)),
                timeout=5.0,
            )
            data = resp.json()
            rain = data.get("daily", {}).get("precipitation_sum", [])
            total = sum(v for v in rain if v is not None)
            return min(total / 80.0, 1.0) * 0.3
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoShield Crowdsourced Event Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
EventType = Literal["flood", "waterlogging", "accident", "construction", "road_block", "debris"]


class ReportRequest(BaseModel):
    type: EventType
    lat:  float = Field(..., ge=8.0,  le=13.0,  description="Latitude (Kerala bounds)")
    lon:  float = Field(..., ge=74.5, le=77.5,  description="Longitude (Kerala bounds)")
    description: Optional[str] = Field(default=None, max_length=200)


class ReportResponse(BaseModel):
    event_id:      str
    type:          str
    confidence:    float
    severity:      str
    reporter_count: int
    expires_at:    str
    message:       str


# ---------------------------------------------------------------------------
# Background expiry task
# ---------------------------------------------------------------------------
async def _expire_loop():
    """Runs every 60 s and removes expired events."""
    while True:
        await asyncio.sleep(60)
        now = datetime.utcnow()
        expired = [eid for eid, ev in _events.items()
                   if datetime.fromisoformat(ev["expires_at"]) <= now]
        for eid in expired:
            print(f"[events] expired: {eid} ({_events[eid]['type']})")
            del _events[eid]


@app.on_event("startup")
async def startup():
    asyncio.create_task(_expire_loop())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "EcoShield Event Service", "active_events": len(_events)}


@app.post("/report", response_model=ReportResponse)
async def report_event(req: ReportRequest):
    """
    Ingest a crowd-sourced hazard report.
    Aggregates reports within the same 500m grid cell + type.
    Confidence grows with each corroborating report (quorum = 3).
    """
    grid_lat, grid_lon = _snap(req.lat, req.lon)
    now = datetime.utcnow()
    ttl = TTL_MINUTES.get(req.type, 60)
    expires_at = now + timedelta(minutes=ttl)
    meta = EVENT_META[req.type]

    existing_id = _find_cell_key(grid_lat, grid_lon, req.type)

    if existing_id:
        # Corroborate existing event
        ev = _events[existing_id]
        ev["reporter_count"] += 1
        # Refresh TTL so it doesn't expire while people are still reporting
        ev["expires_at"] = expires_at.isoformat()
        ev["last_report"] = now.isoformat()

        base_conf = min(ev["reporter_count"] / QUORUM, 1.0)
        rain = ev.get("rain_boost", 0.0)
        ev["confidence"] = round(min(base_conf + rain, 1.0), 3)
        ev["severity"] = _severity(ev["confidence"])
        print(f"[events] corroborate {req.type} @ ({grid_lat},{grid_lon}) → conf={ev['confidence']}")
        return ReportResponse(
            event_id=existing_id,
            type=ev["type"],
            confidence=ev["confidence"],
            severity=ev["severity"],
            reporter_count=ev["reporter_count"],
            expires_at=ev["expires_at"],
            message=f"Report corroborated. Confidence now {ev['confidence']*100:.0f}%.",
        )
    else:
        # New event
        rain_boost = 0.0
        if req.type in ("flood", "waterlogging"):
            rain_boost = await _rainfall_boost(grid_lat, grid_lon)

        base_conf = min(1 / QUORUM, 1.0)
        confidence = round(min(base_conf + rain_boost, 1.0), 3)

        event_id = str(uuid.uuid4())[:8]
        _events[event_id] = {
            "id":             event_id,
            "type":           req.type,
            "label":          meta["label"],
            "emoji":          meta["emoji"],
            "color":          meta["color"],
            "lat":            round(req.lat, 5),
            "lon":            round(req.lon, 5),
            "grid_lat":       grid_lat,
            "grid_lon":       grid_lon,
            "reporter_count": 1,
            "confidence":     confidence,
            "severity":       _severity(confidence),
            "rain_boost":     rain_boost,
            "created_at":     now.isoformat(),
            "last_report":    now.isoformat(),
            "expires_at":     expires_at.isoformat(),
            "description":    req.description or "",
        }
        print(f"[events] new {req.type} @ ({grid_lat},{grid_lon}) conf={confidence}")
        return ReportResponse(
            event_id=event_id,
            type=req.type,
            confidence=confidence,
            severity=_severity(confidence),
            reporter_count=1,
            expires_at=expires_at.isoformat(),
            message=f"Event created. Needs {QUORUM - 1} more reports to reach high confidence.",
        )


@app.get("/events")
def get_events(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 50.0,
):
    """
    Return active events. If lat/lon provided, filters to within radius_km.
    Only returns events with confidence >= 0.1 (filters ghost reports).
    """
    now = datetime.utcnow()
    results = []
    for ev in _events.values():
        if datetime.fromisoformat(ev["expires_at"]) <= now:
            continue   # already expired but not cleaned yet
        if ev["confidence"] < 0.1:
            continue
        if lat is not None and lon is not None:
            # Haversine filter
            dlat = math.radians(ev["lat"] - lat)
            dlon = math.radians(ev["lon"] - lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(ev["lat"])) * math.sin(dlon/2)**2
            dist_km = 6371 * 2 * math.asin(math.sqrt(a))
            if dist_km > radius_km:
                continue
        results.append(ev)
    return {"status": "ok", "count": len(results), "events": results}


@app.delete("/events/{event_id}")
def expire_event(event_id: str):
    """Manually expire/remove an event (moderator action)."""
    if event_id not in _events:
        raise HTTPException(status_code=404, detail="Event not found")
    ev = _events.pop(event_id)
    return {"status": "removed", "type": ev["type"]}
