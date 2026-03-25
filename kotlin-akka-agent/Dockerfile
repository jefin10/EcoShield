FROM gradle:8.5-jdk21 AS builder
WORKDIR /workspace

COPY build.gradle.kts settings.gradle.kts ./
COPY src ./src

RUN gradle --no-daemon installDist

FROM eclipse-temurin:21-jre
WORKDIR /app

COPY --from=builder /workspace/build/install/ecoshield-akka-agent /app

EXPOSE 9001

ENTRYPOINT ["/app/bin/ecoshield-akka-agent"]
