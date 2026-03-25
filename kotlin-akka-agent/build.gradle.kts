plugins {
    kotlin("jvm") version "1.9.22"
    kotlin("plugin.serialization") version "1.9.22"
    application
}

group = "com.ecoshield"
version = "1.0.0"

repositories {
    mavenCentral()
}

val akkaVersion = "2.8.6"
val ktorVersion = "2.3.7"

dependencies {
    // Akka Typed (Scala 2.13)
    implementation("com.typesafe.akka:akka-actor-typed_2.13:$akkaVersion")
    implementation("com.typesafe.akka:akka-slf4j_2.13:$akkaVersion")

    // Ktor Client (for calling microservices)
    implementation("io.ktor:ktor-client-cio:$ktorVersion")
    implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")

    // Ktor Server (to expose our own HTTP API)
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-netty:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-server-cors:$ktorVersion")

    // Kotlin Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-jdk8:1.7.3")   // await() on CompletableFuture / Akka futures

    // Logging
    implementation("ch.qos.logback:logback-classic:1.4.14")

    // Testing
    testImplementation("com.typesafe.akka:akka-actor-testkit-typed_2.13:$akkaVersion")
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("com.ecoshield.MainKt")
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(21)
}
