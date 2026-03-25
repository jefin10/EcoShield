package com.ecoshield.agents

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.javadsl.AbstractBehavior
import akka.actor.typed.javadsl.ActorContext
import akka.actor.typed.javadsl.Behaviors
import akka.actor.typed.javadsl.Receive
import com.ecoshield.VehicleInfo
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.http.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*

/**
 * EmissionAgent — Autonomous actor that calls the XGBoost Emission Prediction API.
 *
 * Agentic Behavior:
 *  - Accepts EmissionRequest messages
 *  - Autonomously calls the emission microservice
 *  - Falls back to a local heuristic if the service is unavailable
 *  - Replies with the CO₂ result
 */
class EmissionAgent private constructor(
    context: ActorContext<Command>,
    private val httpClient: HttpClient,
    private val emissionApiUrl: String,
) : AbstractBehavior<EmissionAgent.Command>(context) {

    // ─── Protocol ────────────────────────────────────────────────────────────
    sealed interface Command

    data class PredictEmission(
        val vehicleInfo: VehicleInfo,
        val distanceKm: Double,
        val durationSec: Double,
        val avgElevationGrad: Double,
        val replyTo: ActorRef<EmissionResult>,
    ) : Command

    data class EmissionResult(val co2Kg: Double)

    // ─── Factory ─────────────────────────────────────────────────────────────
    companion object {
        fun create(httpClient: HttpClient, emissionApiUrl: String): Behavior<Command> =
            Behaviors.setup { ctx -> EmissionAgent(ctx, httpClient, emissionApiUrl) }
    }

    // ─── Message Handling ────────────────────────────────────────────────────
    // Each agent owns a dedicated IO-dispatcher scope — never blocks Akka's thread pool
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun createReceive(): Receive<Command> = newReceiveBuilder()
        .onMessage(PredictEmission::class.java, ::onPredict)
        .build()

    private fun onPredict(msg: PredictEmission): Behavior<Command> {
        scope.launch {                                       // ← non-blocking: return immediately
            val co2Kg = try {
                if (msg.vehicleInfo.fuelType == "Electric") {
                    (msg.distanceKm * 0.008).round(3)
                } else {
                    val idlePct = estimateIdlePct(msg.distanceKm, msg.durationSec)
                    val payload = buildJsonObject {
                        put("speed_kmph", msg.vehicleInfo.avgSpeed)
                        put("idle_pct", idlePct)
                        put("elevation_grad", msg.avgElevationGrad)
                        put("vehicle_type", msg.vehicleInfo.vehicleType)
                        put("bs_norm", msg.vehicleInfo.bsNorm)
                        put("fuel_type", msg.vehicleInfo.fuelType)
                    }
                    val response = httpClient.post(emissionApiUrl) {
                        contentType(ContentType.Application.Json)
                        setBody(payload.toString())
                    }
                    val body = Json.parseToJsonElement(response.body<String>()).jsonObject
                    val co2GPerKm = body["co2_g_per_km"]?.jsonPrimitive?.double ?: 0.0
                    (co2GPerKm * msg.distanceKm / 1000).round(3)
                }
            } catch (e: Exception) {
                context.log.warn("EmissionAgent fallback triggered: ${e.message}")
                fallbackEmission(msg.vehicleInfo, msg.distanceKm)
            }
            context.log.info("EmissionAgent: CO₂ = $co2Kg kg for ${msg.distanceKm} km")
            msg.replyTo.tell(EmissionResult(co2Kg))        // reply from coroutine
        }
        return this                                         // ← Akka thread freed immediately
    }

    // ─── Fallback ────────────────────────────────────────────────────────────
    private fun fallbackEmission(vehicle: VehicleInfo, distanceKm: Double): Double {
        val base = mapOf("2W" to 35.0, "Car" to 120.0, "Bus" to 70.0)
        return ((base[vehicle.vehicleType] ?: 80.0) * distanceKm / 1000).round(3)
    }

    private fun estimateIdlePct(distanceKm: Double, durationSec: Double): Double {
        if (durationSec == 0.0) return 0.1
        val avgKmh = distanceKm / (durationSec / 3600)
        return when {
            avgKmh < 15 -> 0.35
            avgKmh < 25 -> 0.20
            avgKmh < 40 -> 0.10
            else -> 0.05
        }
    }

    private fun Double.round(decimals: Int): Double =
        "%.${decimals}f".format(this).toDouble()
}
