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
 * FloodAgent — Autonomous actor that calls the KNN Flood Prediction API.
 *
 * Agentic Behavior:
 *  - Samples waypoints along the route
 *  - Calls flood service for each point
 *  - Aggregates into a combined flood risk score (0-100)
 *  - Falls back to 0 if the service is down
 */
class FloodAgent private constructor(
    context: ActorContext<Command>,
    private val httpClient: HttpClient,
    private val floodApiUrl: String,
) : AbstractBehavior<FloodAgent.Command>(context) {

    sealed interface Command

    data class PredictFloodRisk(
        val coordinates: List<Coordinate>,
        val replyTo: ActorRef<FloodResult>,
    ) : Command

    data class FloodResult(val floodRiskPercent: Double)

    companion object {
        fun create(httpClient: HttpClient, floodApiUrl: String): Behavior<Command> =
            Behaviors.setup { ctx -> FloodAgent(ctx, httpClient, floodApiUrl) }
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun createReceive(): Receive<Command> = newReceiveBuilder()
        .onMessage(PredictFloodRisk::class.java, ::onPredict)
        .build()

    private fun onPredict(msg: PredictFloodRisk): Behavior<Command> {
        scope.launch {
            val floodRisk = try {
                val sampleSize = minOf(8, msg.coordinates.size)
                val step = maxOf(1, msg.coordinates.size / sampleSize)
                val sample = msg.coordinates.filterIndexed { i, _ -> i % step == 0 }.take(sampleSize)

                // Fan-out: query every sample point concurrently on IO pool
                val probabilities = sample.map { coord ->
                    async {
                        try {
                            val payload = buildJsonObject {
                                put("longitude", coord.longitude)
                                put("latitude", coord.latitude)
                            }
                            val response = httpClient.post(floodApiUrl) {
                                contentType(ContentType.Application.Json)
                                setBody(payload.toString())
                            }
                            val body = Json.parseToJsonElement(response.body<String>()).jsonObject
                            body["flood_probability"]?.jsonPrimitive?.double ?: 0.0
                        } catch (e: Exception) {
                            context.log.debug("FloodAgent point error: ${e.message}")
                            0.0
                        }
                    }
                }.awaitAll()

                val avgProb = if (probabilities.isNotEmpty()) probabilities.average() else 0.0
                (avgProb * 100).round(1)
            } catch (e: Exception) {
                context.log.warn("FloodAgent fallback triggered: ${e.message}")
                0.0
            }
            context.log.info("FloodAgent: flood risk = $floodRisk%")
            msg.replyTo.tell(FloodResult(floodRisk))
        }
        return this
    }

    private fun Double.round(decimals: Int): Double =
        "%.${decimals}f".format(this).toDouble()
}
