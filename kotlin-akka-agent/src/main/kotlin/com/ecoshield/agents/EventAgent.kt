package com.ecoshield.agents

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.javadsl.AbstractBehavior
import akka.actor.typed.javadsl.ActorContext
import akka.actor.typed.javadsl.Behaviors
import akka.actor.typed.javadsl.Receive
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.http.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*

/**
 * EventAgent — Autonomous actor that interacts with the Crowdsourced Event Service.
 *
 * Agentic Behavior:
 *  - Queries the event service for active crowd-reported hazards near a coordinate
 *  - Reports new hazard events
 *  - Self-heals when service is unavailable (returns empty results)
 */
class EventAgent private constructor(
    context: ActorContext<Command>,
    private val httpClient: HttpClient,
    private val eventApiUrl: String,
) : AbstractBehavior<EventAgent.Command>(context) {

    sealed interface Command

    data class FetchNearbyEvents(
        val lat: Double,
        val lon: Double,
        val radiusKm: Double = 50.0,
        val replyTo: ActorRef<EventsResult>,
    ) : Command

    data class ReportHazard(
        val type: String,
        val lat: Double,
        val lon: Double,
        val replyTo: ActorRef<ReportResult>,
    ) : Command

    data class EventsResult(val events: List<JsonObject>)
    data class ReportResult(val success: Boolean, val message: String)

    companion object {
        fun create(httpClient: HttpClient, eventApiUrl: String): Behavior<Command> =
            Behaviors.setup { ctx -> EventAgent(ctx, httpClient, eventApiUrl) }
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun createReceive(): Receive<Command> = newReceiveBuilder()
        .onMessage(FetchNearbyEvents::class.java, ::onFetchEvents)
        .onMessage(ReportHazard::class.java, ::onReportHazard)
        .build()

    private fun onFetchEvents(msg: FetchNearbyEvents): Behavior<Command> {
        scope.launch {
            val events = try {
                val response = httpClient.get("$eventApiUrl/events") {
                    parameter("lat", msg.lat)
                    parameter("lon", msg.lon)
                    parameter("radius_km", msg.radiusKm)
                }
                val body = Json.parseToJsonElement(response.body<String>()).jsonObject
                val eventsArray = body["events"]?.jsonArray ?: JsonArray(emptyList())
                eventsArray.map { it.jsonObject }
            } catch (e: Exception) {
                context.log.warn("EventAgent: could not fetch events: ${e.message}")
                emptyList()
            }
            context.log.info("EventAgent: found ${events.size} nearby events")
            msg.replyTo.tell(EventsResult(events))
        }
        return this
    }

    private fun onReportHazard(msg: ReportHazard): Behavior<Command> {
        scope.launch {
            val result = try {
                val payload = buildJsonObject {
                    put("type", msg.type)
                    put("lat", msg.lat)
                    put("lon", msg.lon)
                }
                val response = httpClient.post("$eventApiUrl/report") {
                    contentType(ContentType.Application.Json)
                    setBody(payload.toString())
                }
                val body = Json.parseToJsonElement(response.body<String>()).jsonObject
                val message = body["message"]?.jsonPrimitive?.content ?: "Report submitted"
                ReportResult(true, message)
            } catch (e: Exception) {
                context.log.warn("EventAgent: could not report hazard: ${e.message}")
                ReportResult(false, "Service unavailable")
            }
            msg.replyTo.tell(result)
        }
        return this
    }
}
