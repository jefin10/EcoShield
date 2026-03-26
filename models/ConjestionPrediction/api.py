"""
api.py — EcoShield Flood Service (port 8003)
============================================
Unified FastAPI server combining:
  1. Flood prediction  — powered by FloodPrediction.py (KNN pipeline)
  2. Route recommendation — powered by routerecom.py (pick safest route)

Endpoints
---------
GET  /                     health / info
GET  /health               liveness probe (used by Docker healthcheck)
POST /predict-flood        single (lon, lat) → flood risk + probability
POST /predict-flood/batch  list of (lon, lat) → flood risks
POST /recommend-route      start + candidate routes → safest route
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ── Flood model (new pipeline from FloodPrediction.py) ──
from FloodPrediction import load_model, predict_flood_risk, predict_batch

# ── Route recommender (existing logic in routerecom.py) ──
from routerecom import pick_safest_route

app = FastAPI(
    title="EcoShield Flood Service",
    description="Flood risk prediction + flood-aware route recommendation",
    version="2.0.0",
)

# Load model once at startup
_model, _threshold = load_model()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PointInput(BaseModel):
    longitude: float
    latitude: float

class BatchInput(BaseModel):
    points: List[PointInput]

class Waypoint(BaseModel):
    longitude: float
    latitude: float

class RoutePickerInput(BaseModel):
    start: Waypoint
    routes: List[List[Waypoint]]
    flood_threshold: float = 0.5


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "EcoShield Flood Service",
        "version": "2.0.0",
        "endpoints": ["/health", "/predict-flood", "/predict-flood/batch", "/recommend-route"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-flood")
def predict_flood(data: PointInput):
    """
    Predict flood risk at a single geographic point.

    Returns flood_risk (0=safe, 1=flooded) and flood_probability (0–1).
    """
    return predict_flood_risk(data.longitude, data.latitude, model=_model, threshold=_threshold)


@app.post("/predict-flood/batch")
def predict_flood_batch(data: BatchInput):
    """
    Predict flood risk for multiple points in one call.
    """
    coords = [(p.longitude, p.latitude) for p in data.points]
    return {"results": predict_batch(coords, model=_model, threshold=_threshold)}


@app.post("/recommend-route")
def recommend(data: RoutePickerInput):
    """
    Given a starting point and candidate routes, returns the safest route
    (fewest flooded waypoints, lowest average flood probability as tiebreaker).

    Each route is a list of {longitude, latitude} waypoints.
    The start point is prepended to every route before evaluation.
    """
    candidate_routes = [
        [(wp.longitude, wp.latitude) for wp in route]
        for route in data.routes
    ]
    start = (data.start.longitude, data.start.latitude)

    return pick_safest_route(
        start=start,
        candidate_routes=candidate_routes,
        model=_model,
        scaler=None,
        flood_threshold=data.flood_threshold,
    )