# Synthea one-shot generator image. Downloads the official Synthea fat-jar and runs it
# with a fixed seed + CSV export, writing the four tables we consume into the shared
# data volume. Used by the `synthea` compose profile (`make up-synthea`).
FROM eclipse-temurin:17-jre-jammy

# Pin to a published Synthea build for reproducibility. Override at build time with
# --build-arg SYNTHEA_JAR_URL=... to track a specific release.
ARG SYNTHEA_JAR_URL=https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

WORKDIR /opt/synthea

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && curl -fL -o synthea-with-dependencies.jar "${SYNTHEA_JAR_URL}" \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY docker/synthea-entrypoint.sh /opt/synthea/entrypoint.sh
RUN chmod +x /opt/synthea/entrypoint.sh

ENTRYPOINT ["/opt/synthea/entrypoint.sh"]
