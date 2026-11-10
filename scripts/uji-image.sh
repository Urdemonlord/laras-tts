#!/usr/bin/env bash
# Smoke test image Laras: dipakai CI terhadap artefak yang benar-benar
# di-push, bukan build lokal. Bisa juga dipakai manual:
#
#   scripts/uji-image.sh ghcr.io/urdemonlord/laras-tts:0.2.1
#
# Yang diperiksa, dan alasannya:
#   1. dependensi dinamis  — binary ONNX statis menuntut GLIBC_2.38+ dan
#      libstdc++ baru; kalau tidak cocok, container mati sebelum sempat bicara
#   2. ffmpeg              — referensi MP3 dikonversi di dalam container
#   3. start tanpa model   — harus berhenti di "gagal memuat model", bukan di
#      "error while loading shared libraries"
set -euo pipefail

IMG="${1:?pakai: scripts/uji-image.sh <referensi-image>}"
echo "image: $IMG"
docker pull "$IMG" > /dev/null

echo "--- arsitektur"
docker image inspect "$IMG" --format '{{.Os}}/{{.Architecture}}'

echo "--- dependensi dinamis di dalam image"
docker run --rm --entrypoint sh "$IMG" -c '
  ldd /usr/local/bin/laras
  echo "glibc runtime: $(ldd --version | head -1)"
  echo "glibc diminta binary: $(grep -aoE "GLIBC_2\.[0-9]+" /usr/local/bin/laras | sort -uV | tail -1)"
  ! ldd /usr/local/bin/laras | grep -q "not found"
'

echo "--- ffmpeg tersedia dan bisa mengonversi mp3"
docker run --rm --entrypoint sh "$IMG" -c '
  set -eu
  ffmpeg -version | head -1
  ffmpeg -y -v error -f lavfi -i sine=duration=0.2 -ar 48000 -ac 1 /tmp/uji.mp3
  ffmpeg -y -v error -i /tmp/uji.mp3 -ar 24000 -ac 1 /tmp/uji.wav
  test -s /tmp/uji.wav
'

echo "--- jalankan tanpa model: harus gagal saat memuat model, bukan saat link"
mkdir -p /tmp/kosong
log="$(mktemp)"
docker run --rm -e MODELS_DIR=/tmp/kosong -e CACHE_DIR=/tmp/cache "$IMG" > "$log" 2>&1 || true
tail -5 "$log"
grep -q "gagal memuat model" "$log"
! grep -qE "GLIBC_[0-9.]+ not found|error while loading shared libraries" "$log"
rm -f "$log"

# Catatan: pengaman HTTP (401/429) tidak diuji di sini karena binary berhenti
# lebih dulu saat model tidak ada. Perilaku itu diuji di `cargo test` lewat
# Service::oneshot, yang tidak butuh model.

echo "semua pemeriksaan lulus: $IMG"
