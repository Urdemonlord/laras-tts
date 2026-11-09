# syntax=docker/dockerfile:1

# ---- build: gateway + engine native (ONNX Runtime), tanpa Python ----
FROM rust:1.83-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --features onnx

# ---- runtime: debian minimal, hanya satu binary ----
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/release/laras /usr/local/bin/laras

ENV BIND=0.0.0.0 \
    PORT=8080 \
    MODELS_DIR=/data/models \
    REF_WAV=/data/ref/suara.wav \
    CACHE_DIR=/data/cache \
    LARAS_LANG=id \
    TAKES=3 \
    TEMP=0.1 \
    EOS=-4.0

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fsS http://127.0.0.1:8080/health > /dev/null || exit 1

CMD ["laras"]
