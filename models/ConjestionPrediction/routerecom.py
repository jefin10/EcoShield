"""
routerecom.py — Flood-Aware Route Recommendation Logic
=======================================================
Delegates flood risk scoring to FloodPrediction.py (KNN pipeline).
This module only handles route evaluation / selection logic.
"""

import numpy as np
from FloodPrediction import predict_flood_risk


# ─── Core: Pick safest from user-provided routes ──────────────────────────────

def pick_safest_route(
    start: tuple,
    candidate_routes: list,           # list of list of (lon, lat) tuples
    model=None,
    scaler=None,                      # kept for API compatibility (ignored; scaling inside pipeline)
    flood_threshold: float = 0.5,
) -> dict:
    """
    Evaluates each candidate route for flood risk.
    Prepends the start point to every route, then picks the route with:
      1. Fewest flooded waypoints
      2. Lowest average flood probability (tiebreaker)
    Returns the winning route + full details for all candidates.
    """
    if not candidate_routes:
        return {"error": "No candidate routes provided."}

    scored = []

    for idx, waypoints in enumerate(candidate_routes):
        all_points = [start] + list(waypoints)
        point_details = []

        for lon, lat in all_points:
            # Uses FloodPrediction KNN pipeline (model arg forwarded if provided)
            risk = predict_flood_risk(lon, lat, model=model)
            point_details.append(risk)

        flooded_count = sum(
            1 for r in point_details if r["flood_probability"] >= flood_threshold
        )
        avg_prob = round(
            float(np.mean([r["flood_probability"] for r in point_details])), 4
        )

        scored.append({
            "route_index":          idx,
            "flooded_waypoints":    flooded_count,
            "avg_flood_probability": avg_prob,
            "status":               "safe" if flooded_count == 0 else "has_flood_risk",
            "waypoints":            [{"longitude": p[0], "latitude": p[1]} for p in all_points],
            "waypoint_details":     point_details,
        })

    # Pick safest: fewest flooded waypoints, then lowest avg probability
    best = min(scored, key=lambda r: (r["flooded_waypoints"], r["avg_flood_probability"]))

    return {
        "recommended_route_index": best["route_index"],
        "recommended_route":       best,
        "all_routes_scored":       scored,
    }
