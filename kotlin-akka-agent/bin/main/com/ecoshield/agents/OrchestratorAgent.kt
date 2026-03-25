package com.ecoshield.agents

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.javadsl.AbstractBehavior
import akka.actor.typed.javadsl.ActorContext
import akka.actor.typed.javadsl.AskPattern
import akka.actor.typed.javadsl.Behaviors
import akka.actor.typed.javadsl.Receive
import com.ecoshield.*
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import kotlinx.coroutines.*
import kotlinx.coroutines.future.await
import kotlinx.serialization.json.*
import java.time.Duration
import java.util.concurrent.CompletableFuture

/**
 * OrchestratorAgent — The "brain" of the Akka Agentic AI system.
 *
 * Agentic Flow:
 *  1. Receive RouteRequest (non-blocking: Akka thread freed instantly)
 *  2. Launch coroutine on dedicated IO dispatcher
 *  3. Fetch up to 5 road-following alternatives from OSRM
 *  4. Per route: fire Emission + Flood + Exposure asks IN PARALLEL (AskPattern + async)
 *  5. Aggregate scores, tag Fastest / EcoPick, complete the response future
 *
 * Key design: NO runBlocking anywhere — Akka's thread pool is NEVER blocked.
 */
class OrchestratorAgent private constructor(
    context: ActorContext<Command>,
    private val httpClient: HttpClient,
    private val emissionAgent: ActorRef<EmissionAgent.Command>,
    private val floodAgent: ActorRef<FloodAgent.Command>,
    private val exposureAgent: ActorRef<ExposureAgent.Command>,
    private val eventAgent: ActorRef<EventAgent.Command>,
) : AbstractBehavior<OrchestratorAgent.Command>(context) {

    sealed interface Command

    data class ProcessRouteRequest(
        val request: RouteRequest,
        val responseFuture: CompletableFuture<List<EcoRoute>>,
    ) : Command

    companion object {
        private val ROUTE_META = listOf(
            Triple("fastest", "Fastest", "#FF5252") to "⏱️ Quickest Way",
            Triple("ecoshield", "EcoShield", "#00E676") to "🌿 Best for Planet & Health",
            Triple("balanced", "Balanced", "#7C4DFF") to "⚖️ Time vs Eco",
            Triple("shortest", "Shortest", "#FFEB3B") to "💰 Least Distance",
            Triple("low_pollution", "Low-Pollution", "#4FC3F7") to "💨 Cleanest Air Route",
        )

        private val FUEL_PRICES = mapOf("Petrol" to 106, "CNG" to 68, "Diesel" to 92, "Electric" to 8)

        fun create(
            httpClient: HttpClient,
            emissionAgent: ActorRef<EmissionAgent.Command>,
            floodAgent: ActorRef<FloodAgent.Command>,
            exposureAgent: ActorRef<ExposureAgent.Command>,
            eventAgent: ActorRef<EventAgent.Command>,
        ): Behavior<Command> = Behaviors.setup { ctx ->
            OrchestratorAgent(ctx, httpClient, emissionAgent, floodAgent, exposureAgent, eventAgent)
        }
    }

    // Dedicated IO scope — HTTP/ML work never touches Akka's dispatcher thread pool
    private val scope     = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val scheduler = context.system.scheduler()

    override fun createReceive(): Receive<Command> = newReceiveBuilder()
        .onMessage(ProcessRouteRequest::class.java, ::onProcessRoute)
        .build()

    // ── Message handler: returns to Akka immediately; real work is in the coroutine
    private fun onProcessRoute(msg: ProcessRouteRequest): Behavior<Command> {
        val vehicleInfo = VEHICLE_MAP[msg.request.vehicle_id]
        if (vehicleInfo == null) {
            msg.responseFuture.completeExceptionally(
                IllegalArgumentException("Unknown vehicle_id: ${msg.request.vehicle_id}")
            )
            return this
        }

        scope.launch {                                    // ← Akka thread freed instantly
            try {
                val req = msg.request
                context.log.info(
                    "Orchestrator: ${req.from_lat},${req.from_lon} → ${req.to_lat},${req.to_lon} [${req.vehicle_id}]"
                )

                // Step 1: road-following routes from OSRM ─────────────────────
                val osrmRoutes = fetchOsrmRoutes(req)
                context.log.info("Orchestrator: scoring ${osrmRoutes.size} OSRM route(s)…")

                // Step 2: score every route — each fans out to 3 agents in parallel
                val scored = osrmRoutes.take(ROUTE_META.size).mapIndexed { idx, route ->
                    scoreRoute(idx, route, vehicleInfo, req.vehicle_id)
                }

                // Step 3: tag fastest / eco-pick and return ───────────────────
                msg.responseFuture.complete(tagRoutes(scored))
            } catch (e: Exception) {
                context.log.error("Orchestrator failed: ${e.message}")
                msg.responseFuture.completeExceptionally(e)
            }
        }
        return this                                       // ← always immediate
    }

    // ── Score one route: ALL THREE sub-agent asks fire at the SAME INSTANT ───
    private suspend fun scoreRoute(
        index: Int,
        osrmRoute: JsonObject,
        vehicleInfo: VehicleInfo,
        vehicleId: String,
    ): ScoredRawRoute = coroutineScope {

        val geoCoords = osrmRoute["geometry"]?.jsonObject
            ?.get("coordinates")?.jsonArray
            ?.map { Coordinate(it.jsonArray[1].jsonPrimitive.double, it.jsonArray[0].jsonPrimitive.double) }
            ?: emptyList()

        val distanceKm  = (osrmRoute["distance"]?.jsonPrimitive?.double ?: 0.0) / 1000.0
        val durationSec =  osrmRoute["duration"]?.jsonPrimitive?.double ?: 0.0
        val durationMin = durationSec / 60.0
        val sampled     = sampleByDistanceKm(geoCoords, 10.0)
        val elevGrad    = estimateElevationGradient(geoCoords)

        // TRUE parallel fan-out: all three asks are sent before any await() ───
        val co2Deferred = async {
            runCatching {
                AskPattern.ask(
                    emissionAgent,
                    { replyTo: ActorRef<EmissionAgent.EmissionResult> ->
                        EmissionAgent.PredictEmission(vehicleInfo, distanceKm, durationSec, elevGrad, replyTo)
                    },
                    Duration.ofSeconds(15), scheduler,
                ).await().co2Kg
            }.getOrElse {
                context.log.warn("Route[$index] EmissionAgent timeout: ${it.message}")
                fallbackEmission(vehicleInfo, distanceKm)
            }
        }

        val floodDeferred = async {
            runCatching {
                AskPattern.ask(
                    floodAgent,
                    { replyTo: ActorRef<FloodAgent.FloodResult> ->
                        FloodAgent.PredictFloodRisk(geoCoords, replyTo)
                    },
                    Duration.ofSeconds(20), scheduler,
                ).await().floodRiskPercent
            }.getOrElse {
                context.log.warn("Route[$index] FloodAgent timeout: ${it.message}")
                0.0
            }
        }

        val exposureDeferred = async {
            runCatching {
                AskPattern.ask(
                    exposureAgent,
                    { replyTo: ActorRef<ExposureAgent.ExposureResult> ->
                        ExposureAgent.CalculateExposure(
                            vehicleId, sampled, vehicleInfo.avgSpeed.toDouble(), durationMin, replyTo
                        )
                    },
                    Duration.ofSeconds(20), scheduler,
                ).await().aqiDose
            }.getOrElse {
                context.log.warn("Route[$index] ExposureAgent timeout: ${it.message}")
                35.0 * durationMin * 1.4
            }
        }

        // All three were running concurrently above; collect results here ─────
        val co2Kg     = co2Deferred.await()
        val floodRisk = floodDeferred.await()
        val aqiDose   = exposureDeferred.await()

        val fuelCost    = calcFuelCost(distanceKm, vehicleInfo.fuelType)
        val greenPoints = maxOf(0, (distanceKm * 2 - co2Kg * 10 - aqiDose * 0.1 - floodRisk * 0.5).toInt())

        ScoredRawRoute(
            coordinates = geoCoords,
            score = RouteScore(
                co2Kg       = co2Kg,
                aqiExposure = aqiDose,
                floodRisk   = floodRisk,
                greenPoints = greenPoints,
                distanceKm  = "%.2f".format(distanceKm).toDouble(),
                durationMin = "%.1f".format(durationMin).toDouble(),
                fuelCostINR = fuelCost,
            ),
        )
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private suspend fun fetchOsrmRoutes(req: RouteRequest): List<JsonObject> {
        val url = "http://router.project-osrm.org/route/v1/driving/" +
                  "${req.from_lon},${req.from_lat};${req.to_lon},${req.to_lat}" +
                  "?alternatives=true&overview=full&geometries=geojson&steps=false"
        return try {
            val data = Json.parseToJsonElement(httpClient.get(url).body<String>()).jsonObject
            if (data["code"]?.jsonPrimitive?.content != "Ok")
                throw Exception("OSRM: ${data["code"]?.jsonPrimitive?.content}")
            data["routes"]?.jsonArray?.map { it.jsonObject }?.takeIf { it.isNotEmpty() }
                ?: throw Exception("Empty OSRM routes")
        } catch (e: Exception) {
            context.log.warn("OSRM unavailable, synthetic fallback: ${e.message}")
            generateSyntheticRoutes(req, VEHICLE_MAP[req.vehicle_id]!!)
        }
    }

    private fun tagRoutes(raw: List<ScoredRawRoute>): List<EcoRoute> {
        if (raw.isEmpty()) return emptyList()

        val fastestIdx = raw.indices.minByOrNull { raw[it].score.durationMin } ?: 0
        val ecoIdx = raw.indices.minByOrNull { raw[it].score.co2Kg * 10 + raw[it].score.aqiExposure } ?: 0

        return raw.mapIndexed { i, r ->
            val metaIdx = minOf(i, ROUTE_META.size - 1)
            val (idLabelColor, tag) = when {
                i == fastestIdx && i == ecoIdx -> ROUTE_META[0] // fastest meta
                i == ecoIdx -> ROUTE_META[1]
                i == fastestIdx -> ROUTE_META[0]
                else -> ROUTE_META[metaIdx]
            }
            val isFast = i == fastestIdx
            val isEco = i == ecoIdx
            val health = when {
                isEco -> "Reduces PM2.5 dose vs fastest route"
                else -> "${r.score.aqiExposure} µg/m³ avg dose"
            }
            val pts = r.score.greenPoints + if (isEco) 50 else 0

            EcoRoute(
                id = idLabelColor.first,
                label = idLabelColor.second,
                color = idLabelColor.third,
                tag = tag,
                isLive = true,
                isFastest = isFast,
                isEcoPick = isEco,
                coordinates = r.coordinates,
                score = r.score.copy(greenPoints = pts, healthBenefit = health),
            )
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private data class ScoredRawRoute(
        val coordinates: List<Coordinate>,
        val score: RouteScore,
    )

    private fun generateSyntheticRoutes(req: RouteRequest, v: VehicleInfo): List<JsonObject> {
        val baseDist = haversineKm(req.from_lat, req.from_lon, req.to_lat, req.to_lon)
        return (0 until 3).map { idx ->
            val mult = listOf(1.0, 1.08, 1.04)[idx]
            val speedMult = listOf(1.2, 0.95, 1.0)[idx]
            val dKm = baseDist * mult
            val speed = v.avgSpeed * speedMult
            val durS = dKm / speed * 3600

            val coords = (0..20).map { s ->
                val t = s / 20.0
                val curve = Math.sin(t * Math.PI) * 0.005 * (idx + 1)
                val lat = req.from_lat + (req.to_lat - req.from_lat) * t + curve
                val lon = req.from_lon + (req.to_lon - req.from_lon) * t - curve
                buildJsonArray { add(lon); add(lat) }
            }

            buildJsonObject {
                put("geometry", buildJsonObject {
                    put("coordinates", buildJsonArray { coords.forEach { add(it) } })
                    put("type", "LineString")
                })
                put("distance", dKm * 1000)
                put("duration", durS)
            }
        }
    }

    private fun sampleByDistanceKm(coords: List<Coordinate>, intervalKm: Double): List<Coordinate> {
        if (coords.size <= 3) return coords
        val sampled = mutableListOf(coords[0])
        var accumulated = 0.0
        for (i in 1 until coords.size) {
            val d = haversineKm(coords[i - 1].latitude, coords[i - 1].longitude, coords[i].latitude, coords[i].longitude)
            accumulated += d
            if (accumulated >= intervalKm) {
                sampled.add(coords[i])
                accumulated = 0.0
            }
        }
        if (sampled.last() != coords.last()) sampled.add(coords.last())
        return sampled
    }

    private fun estimateElevationGradient(coords: List<Coordinate>): Double {
        if (coords.size < 2) return 0.0
        val latSpread = Math.abs(coords.last().latitude - coords[0].latitude)
        val lonSpread = Math.abs(coords.last().longitude - coords[0].longitude)
        return minOf(latSpread + lonSpread, 0.5) * 100
    }

    private fun calcFuelCost(distanceKm: Double, fuelType: String): Double {
        val price = FUEL_PRICES[fuelType] ?: 106
        return if (fuelType == "Electric") {
            "%.1f".format(distanceKm * 0.08 * price).toDouble()
        } else {
            val eff = mapOf("Petrol" to 45, "CNG" to 25, "Diesel" to 12)[fuelType] ?: 40
            "%.1f".format(distanceKm / eff * price).toDouble()
        }
    }

    private fun fallbackEmission(vehicle: VehicleInfo, distanceKm: Double): Double {
        val base = mapOf("2W" to 35.0, "Car" to 120.0, "Bus" to 70.0)
        return "%.3f".format((base[vehicle.vehicleType] ?: 80.0) * distanceKm / 1000).toDouble()
    }
}
