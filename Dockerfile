# syntax=docker/dockerfile:1

# ---- build: gateway + engine native (ONNX Runtime), tanpa Python ----
# Rust 1.92 = MSRV sebenarnya: `ort 2.0.0-rc.15` menuntut rustc 1.92, dan
# `ureq-proto` yang ikut terbawa butuh Cargo dengan edition2024.
# Build dan runtime sama-sama trixie: runtime ONNX yang ditanam `ort` adalah
# arsip statis C++ yang butuh GLIBCXX_3.4.31+ / GLIBC_2.38+ — bookworm-slim
# (gcc 12, glibc 2.36) gagal di tahap link.
FROM rust:1.92-slim-trixie AS build
WORKDIR /src
# `ort` mengunduh runtime ONNX saat build lewat ureq → openssl-sys butuh ini.
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config libssl-dev g++ \
 && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Halaman depan di-`include_str!` dari `src/main.rs`, jadi harus ada saat build.
COPY demo/index.html ./demo/index.html
RUN cargo build --release --features onnx

# ---- runtime: debian minimal, hanya satu binary ----
FROM debian:trixie-slim

# ffmpeg bukan hiasan: referensi suara dari Common Voice berformat MP3, dan
# `read_mono_auto` mengonversinya ke WAV 24 kHz mono lewat ffmpeg. Tanpa ini,
# pengguna harus mengonversi sendiri sebelum container mau bicara.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/release/laras /usr/local/bin/laras

ENV BIND=0.0.0.0 \
    PORT=8080 \
    MODELS_DIR=/data/models \
    REF_WAV=/data/ref/suara.mp3 \
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
