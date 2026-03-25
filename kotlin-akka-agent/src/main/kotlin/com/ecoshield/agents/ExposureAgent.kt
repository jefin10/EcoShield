package com.ecoshield.agents

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.javadsl.AbstractBehavior
import akka.actor.typed.javadsl.ActorContext
import akka.actor.typed.javadsl.Behaviors
import akka.actor.typed.javadsl.Receive
import com.ecoshield.Coordinate
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.http.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*

/**
 * ExposureAgent — Autonomous actor that calls the AQI Exposure Microservice.
 *
 * Agentic Behavior:
 *  - Sends sampled waypoints to the exposure service
 *  - Falls back to Kerala average PM2.5 estimate if service is down
 */
class ExposureAgent private constructor(
    context: ActorContext<Command>,
    private val httpClient: HttpClient,
    private val exposureApiUrl: String,
) : AbstractBehavior<ExposureAgent.Command>(context) {

    sealed interface Command

    data class CalculateExposure(
        val vehicleId: String,
        val sampledCoordinates: List<Coordinate>,
        val avgSpeedKmh: Double,
        val durationMin: Double,
        val replyTo: ActorRef<ExposureResult>,
    ) : Command

    data class ExposureResult(val aqiDose: Double)

    companion object {
        private const val KERALA_AVG_PM25 = 35.0
        private val ACTIVITY_FACTORS = mapOf(
            "2w_petrol_bs4" to 1.7, "2w_petrol_bs6" to 1.7,
            "auto_cng" to 1.5, "car_bs6" to 1.1, "ev" to 1.1, "bus" to 1.2,
        )

        fun create(httpClient: HttpClient, exposureApiUrl: String): Behavior<Command> =
            Behaviors.setup { ctx -> ExposureAgent(ctx, httpClient, exposureApiUrl) }
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun createReceive(): Receive<Command> = newReceiveBuilder()
        .onMessage(CalculateExposure::class.java, ::onCalculate)
        .build()

    private fun onCalculate(msg: CalculateExposure): Behavior<Command> {
        scope.launch {
            val dose = try {
                val coordsJson = buildJsonArray {
                    msg.sampledCoordinates.forEach { c ->
                        add(buildJsonObject {
                            put("lat", c.latitude)
                            put("lon", c.longitude)
                        })
                    }
                }
                val payload = buildJsonObject {
                    put("coordinates", coordsJson)
                    put("vehicle_id", msg.vehicleId)
                    put("avg_speed_kmh", msg.avgSpeedKmh)
                    put("duration_minutes", maxOf(0.1, msg.durationMin))
                    put("sample_every_n", 1)
                }
                val response = httpClient.post(exposureApiUrl) {
                    contentType(ContentType.Application.Json)
                    setBody(payload.toString())
                }
                val body = Json.parseToJsonElement(response.body<String>()).jsonObject
                val exposureBlock = body["exposure"]?.jsonObject
                var d = exposureBlock?.get("cumulative_pm25_dose_ug_min_m3")?.jsonPrimitive?.double ?: 0.0
                if (d == 0.0) d = body["total_exposure_ug_m3_min"]?.jsonPrimitive?.double ?: 0.0
                if (d == 0.0) {
                    val af = ACTIVITY_FACTORS[msg.vehicleId] ?: 1.4
                    d = KERALA_AVG_PM25 * msg.durationMin * af
                }
                d.round(1)
            } catch (e: Exception) {
                context.log.warn("ExposureAgent fallback triggered: ${e.message}")
                val af = ACTIVITY_FACTORS[msg.vehicleId] ?: 1.4
                (KERALA_AVG_PM25 * msg.durationMin * af).round(1)
            }
            context.log.info("ExposureAgent: AQI dose = $dose µg·min/m³")
            msg.replyTo.tell(ExposureResult(dose))
        }
        return this
    }

    private fun Double.round(decimals: Int): Double =
        "%.${decimals}f".format(this).toDouble()
}
