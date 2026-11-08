# syntax=docker/dockerfile:1

# ---- build gateway (Rust) ----
FROM rust:1.83-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# ---- runtime (Python + binary) ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch dari index CPU: tanpa ini pip menarik wheel CUDA ~2,5 GB
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY worker.py norm_id.py ./
COPY --from=build /src/target/release/tts-gateway /usr/local/bin/tts-gateway

ENV BIND=0.0.0.0 \
    PORT=8080 \
    WORKER_PY=python3 \
    WORKER_SCRIPT=/app/worker.py \
    REF_WAV=/data/ref/suara.wav \
    CACHE_DIR=/data/cache \
    WORKER_OUT=/data/out \
    HF_HOME=/data/hf \
    TAKES=3 \
    USE_ASR=1

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).status==200 else 1)"

CMD ["tts-gateway"]
