# Space demo Laras — TTS Bahasa Indonesia.
#
# Model diunduh saat build, bukan saat start: pengunjung pertama tidak perlu
# menunggu 160 MB, dan kalau unduhan gagal build-nya yang merah (bukan Space
# yang diam-diam rusak).
#
# Skrip unduh dipin ke tag rilis, jadi manifest sha256-nya ikut — model yang
# masuk ke image bisa diverifikasi, bukan "curl lalu percaya".
FROM ghcr.io/urdemonlord/laras-tts:0.2.1

USER root

ARG LARAS_TAG=v0.2.1
RUN curl -fsSL \
      "https://raw.githubusercontent.com/Urdemonlord/laras-tts/${LARAS_TAG}/scripts/unduh-model.sh" \
      -o /tmp/unduh.sh \
 && bash /tmp/unduh.sh --dir /data/models --ref-dir /data/ref \
 && rm -f /tmp/unduh.sh \
 && du -sh /data/models /data/ref

# Space gratis: 2 vCPU, 16 GB RAM. Satu take sudah memadai — menambah take
# hanya membuat pengunjung menunggu tanpa perbaikan artikulasi yang sepadan.
# Batas laju dipasang karena CPU-nya dipakai bersama pengunjung lain.
ENV PORT=7860 \
    BIND=0.0.0.0 \
    MODELS_DIR=/data/models \
    REF_WAV=/data/ref/common_voice_id_19961910.mp3 \
    CACHE_DIR=/data/cache \
    LARAS_LANG=id \
    TAKES=1 \
    THREADS=2 \
    RATE_LIMIT=6 \
    QUEUE_WAIT=20 \
    REQUEST_TIMEOUT=120

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://127.0.0.1:7860/health > /dev/null || exit 1
