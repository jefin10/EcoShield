package com.ecoshield

import kotlinx.serialization.Serializable

// ─── Shared data models used across agents ──────────────────────────────────

@Serializable
data class Coordinate(val latitude: Double, val longitude: Double)

@Serializable
data class RouteScore(
    val co2Kg: Double = 0.0,
    val aqiExposure: Double = 0.0,
    val floodRisk: Double = 0.0,
    val greenPoints: Int = 0,
    val healthBenefit: String = "",
    val distanceKm: Double = 0.0,
    val durationMin: Double = 0.0,
    val fuelCostINR: Double = 0.0,
)

@Serializable
data class EcoRoute(
    val id: String,
    val label: String,
    val color: String,
    val tag: String,
    val isLive: Boolean = false,
    val isFastest: Boolean = false,
    val isEcoPick: Boolean = false,
    val coordinates: List<Coordinate> = emptyList(),
    val score: RouteScore = RouteScore(),
)

@Serializable
data class RouteRequest(
    val from_lat: Double,
    val from_lon: Double,
    val to_lat: Double,
    val to_lon: Double,
    val vehicle_id: String = "2w_petrol_bs6",
)

// ─── Agent messages ─────────────────────────────────────────────────────────

// Vehicle configuration for emission model
data class VehicleInfo(
    val vehicleType: String,
    val bsNorm: String,
    val avgSpeed: Int,
    val fuelType: String,
)

val VEHICLE_MAP = mapOf(
    "2w_petrol_bs4" to VehicleInfo("2W", "BS4", 35, "Petrol"),
    "2w_petrol_bs6" to VehicleInfo("2W", "BS6", 38, "Petrol"),
    "auto_cng"      to VehicleInfo("2W", "BS6", 28, "Petrol"),
    "car_bs6"       to VehicleInfo("Car", "BS6", 45, "Petrol"),
    "ev"            to VehicleInfo("Car", "BS6", 50, "Electric"),
    "bus"           to VehicleInfo("Bus", "BS6", 30, "Diesel"),
)

// ─── Utility ────────────────────────────────────────────────────────────────

fun haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val R = 6371.0
    val dLat = Math.toRadians(lat2 - lat1)
    val dLon = Math.toRadians(lon2 - lon1)
    val a = Math.sin(dLat / 2).let { it * it } +
            Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
            Math.sin(dLon / 2).let { it * it }
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}
