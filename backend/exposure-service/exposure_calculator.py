"""
Exposure Calculator — computes cumulative personal pollutant exposure
from a set of AQI readings sampled along a route.

Methodology
-----------
Based on personal exposure modeling used in public health research:
  Exposure (µg·min/m³) = Σ [concentration_i × time_fraction_i × activity_factor]

Where:
  - concentration_i  = PM2.5 / pollutant value at waypoint i
  - time_fraction_i  = time spent near waypoint (distance_i / speed)
  - activity_factor  = breathing rate multiplier (1.0 for rest, 1.6 for riding)
  - segment_weight   = haversine distance ratio for area weighting
"""
import math
from typing import Optional


ACTIVITY_FACTORS = {
    "2w_petrol_bs4": 1.7,   # open rider, high breathing rate
    "2w_petrol_bs6": 1.7,
    "auto_cng":      1.5,   # semi-open cabin
    "car_bs6":       1.1,   # sealed cabin, some filtration
    "ev":            1.1,
    "bus":           1.2,   # large cabin, mixed ventilation
}

# WHO AQI breakpoints for health risk classification
WHO_PM25_BREAKPOINTS = [
    (0, 12,   "Good",        "Minimal impact"),
    (12, 35,  "Moderate",    "Sensitive groups may be affected"),
    (35, 55,  "Unhealthy*",  "Sensitive groups: unhealthy"),
    (55, 150, "Unhealthy",   "General population: unhealthy"),
    (150, 250,"Very Unhealthy","Health alert — avoid outdoor exposure"),
    (250, 999,"Hazardous",   "Emergency conditions"),
]

# NO2 WHO guideline: 25 µg/m³ (1-hour mean)
# CO  WHO guideline: 10 mg/m³  (8-hour mean) → ~8.7 ppm
# O3  WHO guideline: 100 µg/m³ (8-hour mean)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def classify_pm25(pm25: float) -> dict:
    for lo, hi, category, advice in WHO_PM25_BREAKPOINTS:
        if lo <= pm25 < hi:
            return {"category": category, "advice": advice}
    return {"category": "Hazardous", "advice": "Emergency conditions"}


def safe_val(readings: list, key: str) -> list[float]:
    """Extract non-None numeric values for a pollutant key."""
    return [r[key] for r in readings if r and r.get(key) is not None]


def calculate_exposure(
    coordinates: list[dict],         # [{"lat": float, "lon": float, ...}]
    aqi_readings: list[Optional[dict]],
    vehicle_id: str = "2w_petrol_bs6",
    avg_speed_kmh: float = 35.0,
    duration_minutes: Optional[float] = None,
) -> dict:
    """
    Compute cumulative exposure along a route.

    Parameters
    ----------
    coordinates    : list of {"lat", "lon"} dicts for each sampled waypoint
    aqi_readings   : WAQI response dicts (same length as coordinates), may have None
    vehicle_id     : vehicle type key for activity factor lookup
    avg_speed_kmh  : average travel speed (km/h)
    duration_minutes: override if actual duration is known

    Returns
    -------
    dict with exposure scores, hotspots, health risk, segment breakdown
    """

    n = len(coordinates)
    if n == 0:
        return {"error": "No coordinates provided"}

    # --- Compute segment distances & total distance ---
    segment_distances: list[float] = []
    for i in range(n - 1):
        d = haversine_km(
            coordinates[i]["lat"], coordinates[i]["lon"],
            coordinates[i+1]["lat"], coordinates[i+1]["lon"]
        )
        segment_distances.append(d)

    total_dist_km = sum(segment_distances) if segment_distances else 0.0
    # Edge: single point
    if total_dist_km == 0.0:
        total_dist_km = 0.001

    # --- Duration ---
    if duration_minutes is None or duration_minutes <= 0:
        duration_minutes = (total_dist_km / avg_speed_kmh) * 60.0

    activity_factor = ACTIVITY_FACTORS.get(vehicle_id, 1.4)

    # --- Segment exposure accumulation ---
    segments: list[dict] = []
    cumulative_pm25_dose = 0.0    # µg·min/m³
    cumulative_no2_dose  = 0.0
    cumulative_co_dose   = 0.0
    cumulative_o3_dose   = 0.0
    hotspots: list[dict] = []
    valid_readings = 0

    for i in range(n):
        reading = aqi_readings[i] if i < len(aqi_readings) else None
        coord = coordinates[i]

        # Time fraction for this segment
        seg_dist = segment_distances[i] if i < len(segment_distances) else (segment_distances[-1] if segment_distances else 0)
        seg_time_min = (seg_dist / total_dist_km) * duration_minutes if total_dist_km > 0 else 0.0

        pm25 = reading.get("pm25") if reading else None
        no2  = reading.get("no2")  if reading else None
        co   = reading.get("co")   if reading else None
        o3   = reading.get("o3")   if reading else None
        aqi  = reading.get("aqi")  if reading else None

        if pm25 is not None:
            valid_readings += 1
            dose_pm25 = pm25 * seg_time_min * activity_factor
            dose_no2  = (no2 or 0) * seg_time_min * activity_factor
            dose_co   = (co  or 0) * seg_time_min * activity_factor
            dose_o3   = (o3  or 0) * seg_time_min * activity_factor

            cumulative_pm25_dose += dose_pm25
            cumulative_no2_dose  += dose_no2
            cumulative_co_dose   += dose_co
            cumulative_o3_dose   += dose_o3

            # Flag hotspot if PM2.5 > 55 (Unhealthy threshold)
            if pm25 > 55:
                hotspots.append({
                    "lat": coord["lat"],
                    "lon": coord["lon"],
                    "station": reading.get("station_name", "Unknown"),
                    "pm25": pm25,
                    "aqi": aqi,
                    "classification": classify_pm25(pm25)["category"],
                })

            segments.append({
                "segment_index": i,
                "lat": coord["lat"],
                "lon": coord["lon"],
                "station": reading.get("station_name") if reading else None,
                "aqi": aqi,
                "pm25": pm25,
                "no2": no2,
                "co": co,
                "o3": o3,
                "seg_dist_km": round(seg_dist, 3),
                "seg_time_min": round(seg_time_min, 2),
                "pm25_dose": round(dose_pm25, 2),
            })
        else:
            segments.append({
                "segment_index": i,
                "lat": coord["lat"],
                "lon": coord["lon"],
                "station": None,
                "aqi": None,
                "pm25": None,
                "no2": None,
                "co": None,
                "o3": None,
                "seg_dist_km": round(seg_dist, 3),
                "seg_time_min": round(seg_time_min, 2),
                "pm25_dose": 0.0,
            })

    # --- Aggregate statistics ---
    all_pm25 = safe_val(aqi_readings, "pm25")
    all_no2  = safe_val(aqi_readings, "no2")
    all_aqi  = safe_val(aqi_readings, "aqi")

    avg_pm25 = sum(all_pm25) / len(all_pm25) if all_pm25 else 0.0
    max_pm25 = max(all_pm25) if all_pm25 else 0.0
    avg_aqi  = sum(all_aqi)  / len(all_aqi)  if all_aqi  else 0.0
    max_aqi  = max(all_aqi)  if all_aqi  else 0.0

    pm25_classification = classify_pm25(avg_pm25)

    # --- Health risk score (0–100) ---
    # Weighted: PM2.5 exposure (60%), AQI level (25%), NO2 (15%)
    pm25_score = min(100, (cumulative_pm25_dose / (duration_minutes * 35)) * 60) if duration_minutes > 0 else 0
    aqi_score  = min(100, (avg_aqi / 300) * 25)
    no2_score  = min(100, (sum(safe_val(aqi_readings, "no2")) / (len(all_no2) or 1) / 25) * 15) if all_no2 else 0
    health_risk_score = round(min(100, pm25_score + aqi_score + no2_score), 1)

    # --- Eco-health benefit vs. worst route ---
    # Placeholder: assume worst route has 1.8× higher exposure
    avoided_pm25 = round(cumulative_pm25_dose * 0.8, 2)

    return {
        "summary": {
            "total_distance_km":      round(total_dist_km, 2),
            "total_duration_min":     round(duration_minutes, 1),
            "avg_speed_kmh":          round(avg_speed_kmh, 1),
            "vehicle_id":             vehicle_id,
            "activity_factor":        activity_factor,
            "waypoints_sampled":      n,
            "waypoints_with_data":    valid_readings,
            "data_coverage_pct":      round(valid_readings / n * 100, 1) if n > 0 else 0,
        },
        "exposure": {
            "cumulative_pm25_dose_ug_min_m3":  round(cumulative_pm25_dose, 2),
            "cumulative_no2_dose_ug_min_m3":   round(cumulative_no2_dose, 3),
            "cumulative_co_dose_mg_min_m3":    round(cumulative_co_dose, 3),
            "cumulative_o3_dose_ug_min_m3":    round(cumulative_o3_dose, 3),
            "avg_pm25_ug_m3":                  round(avg_pm25, 1),
            "max_pm25_ug_m3":                  round(max_pm25, 1),
            "avg_aqi":                         round(avg_aqi, 1),
            "max_aqi":                         round(max_aqi, 1),
            "pm25_classification":             pm25_classification["category"],
            "health_advice":                   pm25_classification["advice"],
        },
        "health_risk": {
            "score_0_to_100":         health_risk_score,
            "level":                  (
                "Low" if health_risk_score < 25 else
                "Moderate" if health_risk_score < 50 else
                "High" if health_risk_score < 75 else
                "Very High"
            ),
            "avoided_pm25_if_ecoshield_ug_min_m3": avoided_pm25,
        },
        "hotspots": hotspots,
        "segments": segments,
    }
