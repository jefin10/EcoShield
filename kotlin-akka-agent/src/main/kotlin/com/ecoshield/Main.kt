package com.ecoshield

import akka.actor.typed.ActorSystem
import akka.actor.typed.javadsl.Behaviors
import com.ecoshield.agents.*
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import kotlinx.coroutines.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

// ─────────────────────────────────────────────────────────────────────────────
// Top-level proxy helpers — defined OUTSIDE any Ktor server routing DSL so that
// only the CLIENT extension functions are in scope (no server shadowing).
// ─────────────────────────────────────────────────────────────────────────────

private suspend fun proxyGet(client: HttpClient, url: String): String =
    client.get(url).bodyAsText()

private suspend fun proxyPost(client: HttpClient, url: String, jsonBody: String): String =
    client.post(url) {
        contentType(ContentType.Application.Json)
        setBody(jsonBody)
    }.bodyAsText()

/**
 * EcoShield Akka Agentic AI System — Main Entry Point
 *
 * Launches:
 *  1. Akka ActorSystem with all agents
 *  2. Ktor HTTP server on port 9001
 *
 * The Ktor server exposes:
 *  - POST /routes  — proxied through the OrchestratorAgent
 *  - GET  /health  — health check
 */
fun main() {
    // ─── 1. Create shared HTTP client for agents ────────────────────────────
    val httpClient = HttpClient(CIO) {
        install(io.ktor.client.plugins.contentnegotiation.ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
        engine {
            requestTimeout = 30_000
        }
    }

    // ─── 2. Service URLs (from env or defaults) ─────────────────────────────
    val emissionUrl = System.getenv("EMISSION_API_URL") ?: "http://localhost:8002/predict"
    val exposureUrl = System.getenv("EXPOSURE_API_URL") ?: "http://localhost:8004/exposure"
    val floodUrl    = System.getenv("FLOOD_API_URL")    ?: "http://localhost:8003/predict-flood"
    val eventUrl    = System.getenv("EVENT_API_URL")     ?: "http://localhost:8005"

    // ─── 3. Start Akka Actor System ─────────────────────────────────────────
    val system = ActorSystem.create(Behaviors.setup<Any> { ctx ->
        ctx.log.info("╔══════════════════════════════════════════════════════════╗")
        ctx.log.info("║     EcoShield Akka Agentic AI System — Starting...      ║")
        ctx.log.info("╚══════════════════════════════════════════════════════════╝")

        // Spawn child agents
        val emissionAgent = ctx.spawn(EmissionAgent.create(httpClient, emissionUrl), "emission-agent")
        val floodAgent    = ctx.spawn(FloodAgent.create(httpClient, floodUrl), "flood-agent")
        val exposureAgent = ctx.spawn(ExposureAgent.create(httpClient, exposureUrl), "exposure-agent")
        val eventAgent    = ctx.spawn(EventAgent.create(httpClient, eventUrl), "event-agent")

        // Spawn the orchestrator (the "brain")
        val orchestrator = ctx.spawn(
            OrchestratorAgent.create(httpClient, emissionAgent, floodAgent, exposureAgent, eventAgent),
            "orchestrator-agent"
        )

        ctx.log.info("All agents spawned: emission, flood, exposure, event, orchestrator")

        // ─── 4. Start Ktor HTTP Server ──────────────────────────────────────
        embeddedServer(Netty, port = 9001, host = "0.0.0.0") {
            install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
            install(CORS) {
                anyHost()
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Get)
                allowHeader(HttpHeaders.ContentType)
            }

            routing {
                get("/health") {
                    val json = buildJsonObject {
                        put("status", "ok")
                        put("service", "EcoShield Akka Agentic AI")
                        put("version", "1.0.0")
                        put("agents", buildJsonArray {
                            add(kotlinx.serialization.json.JsonPrimitive("emission"))
                            add(kotlinx.serialization.json.JsonPrimitive("flood"))
                            add(kotlinx.serialization.json.JsonPrimitive("exposure"))
                            add(kotlinx.serialization.json.JsonPrimitive("event"))
                            add(kotlinx.serialization.json.JsonPrimitive("orchestrator"))
                        })
                    }
                    call.respondText(json.toString(), ContentType.Application.Json)
                }

                post("/routes") {
                    try {
                        val body = call.receiveText()
                        val request = Json.decodeFromString<RouteRequest>(body)

                        // Send to the orchestrator agent and wait for response
                        val responseFuture = CompletableFuture<List<EcoRoute>>()
                        orchestrator.tell(OrchestratorAgent.ProcessRouteRequest(request, responseFuture))

                        val routes = responseFuture.get(60, TimeUnit.SECONDS)
                        val routesJson = Json.encodeToString(routes)
                        val response = """{"status":"ok","vehicle_id":"${request.vehicle_id}","routes":$routesJson}"""
                        call.respondText(response, ContentType.Application.Json)
                    } catch (e: Exception) {
                        val msg = (e.message ?: "Unknown error").replace("\"", "'")
                        call.respondText(
                            """{"status":"error","error":"$msg"}""",
                            ContentType.Application.Json,
                            HttpStatusCode.InternalServerError
                        )
                    }
                }

                get("/agents") {
                    val json = buildJsonObject {
                        put("agents", buildJsonArray {
                            add(buildJsonObject { put("name","EmissionAgent");     put("role","XGBoost CO2 prediction (async, non-blocking)");       put("endpoint",emissionUrl) })
                            add(buildJsonObject { put("name","FloodAgent");        put("role","KNN Flood risk with concurrent point queries");        put("endpoint",floodUrl) })
                            add(buildJsonObject { put("name","ExposureAgent");     put("role","WAQI AQI cumulative exposure along route");            put("endpoint",exposureUrl) })
                            add(buildJsonObject { put("name","EventAgent");        put("role","Crowdsourced hazard events (report + fetch)");         put("endpoint",eventUrl) })
                            add(buildJsonObject { put("name","OrchestratorAgent"); put("role","Parallel AskPattern fan-out across 3 ML agents") })
                        })
                    }
                    call.respondText(json.toString(), ContentType.Application.Json)
                }

                get("/events") {
                    try {
                        val lat      = call.request.queryParameters["lat"]
                        val lon      = call.request.queryParameters["lon"]
                        val radius   = call.request.queryParameters["radius_km"] ?: "50"
                        val url = if (lat != null && lon != null)
                            "$eventUrl/events?lat=$lat&lon=$lon&radius_km=$radius"
                        else "$eventUrl/events/all"
                        val raw = proxyGet(httpClient, url)
                        call.respondText(raw, ContentType.Application.Json)
                    } catch (e: Exception) {
                        call.respondText("""{"events":[],"error":"${e.message}"}""", ContentType.Application.Json)
                    }
                }

                post("/report") {
                    try {
                        val body = call.receiveText()
                        val raw  = proxyPost(httpClient, "$eventUrl/report", body)
                        call.respondText(raw, ContentType.Application.Json)
                    } catch (e: Exception) {
                        val msg = (e.message ?: "Event service unavailable").replace("\"", "'")
                        call.respondText("""{"error":"$msg"}""", ContentType.Application.Json, HttpStatusCode.ServiceUnavailable)
                    }
                }

                // ── AQI live data proxy ──────────────────────────────────────
                get("/aqi") {
                    try {
                        val city  = call.request.queryParameters["city"] ?: "kochi"
                        val token = System.getenv("WAQI_TOKEN") ?: ""
                        val raw   = proxyGet(httpClient, "https://api.waqi.info/feed/$city/?token=$token")
                        call.respondText(raw, ContentType.Application.Json)
                    } catch (e: Exception) {
                        call.respondText("""{"error":"${e.message}"}""", ContentType.Application.Json)
                    }
                }
            }
        }.start(wait = false)

        ctx.log.info("HTTP API available at http://localhost:9001")
        ctx.log.info("  POST /routes   — Get AI-scored eco-routes")
        ctx.log.info("  GET  /health   — Health check")
        ctx.log.info("  GET  /agents   — List all agents")

        // Heartbeat — use a plain logger (not ctx.log) so it's safe to call off the actor thread
        val sysLog = org.slf4j.LoggerFactory.getLogger("EcoShieldHeartbeat")
        CoroutineScope(Dispatchers.Default).launch {
            while (isActive) {
                delay(30_000)
                sysLog.info("EcoShield Backend Heartbeat: All agents active.")
            }
        }

        Behaviors.empty()
    }, "ecoshield-agentic-system")

    println("\n🌿 EcoShield Akka Agentic AI System is running!")
    println("   API: http://localhost:9001")
    println("   Press Ctrl+C to stop.\n")

    // Keep alive
    system.whenTerminated.toCompletableFuture().join()
}
