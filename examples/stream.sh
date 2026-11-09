#!/usr/bin/env bash
# Audio mengalir per kalimat: yang pertama bisa didengar sementara yang
# terakhir masih disintesis.
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8080}"
TEKS="${TEKS:-Selamat pagi semuanya. Rapat koordinasi dimulai pukul sembilan pagi di ruang rapat lantai empat. Anggaran proyek naik tujuh koma lima persen tahun ini. Mohon setiap tim menyiapkan laporan mingguan sebelum hari Jumat.}"
BADAN="{\"text\":\"$TEKS\"}"

echo "== WAV mengalir -> berkas =="
curl -sN -D /tmp/stream.h -o hasil-stream.wav \
  -X POST "$BASE/tts/stream" -H 'content-type: application/json' -d "$BADAN"
grep -iE 'x-stream|x-chunks|x-cache|x-takes' /tmp/stream.h
echo "-> hasil-stream.wav"
if command -v ffprobe >/dev/null; then
  ffprobe -v error -show_entries format=duration -of csv=p=0 hasil-stream.wav
else
  echo "(ffprobe tidak ada; durasi = ukuran byte / 48000 detik)"
fi

echo "== PCM mentah, didengar langsung =="
if command -v ffplay >/dev/null; then
  curl -sN -X POST "$BASE/tts/stream?fmt=pcm" -H 'content-type: application/json' \
    -d "$BADAN" | ffplay -v error -nodisp -autoexit -f s16le -ar 24000 -ac 1 -
else
  curl -sN -X POST "$BASE/tts/stream?fmt=pcm" -H 'content-type: application/json' \
    -d "$BADAN" -o hasil-stream.pcm
  echo "-> hasil-stream.pcm ($(wc -c < hasil-stream.pcm) byte; putar: ffplay -f s16le -ar 24000 -ac 1 hasil-stream.pcm)"
fi

echo "== isi cache lewat /tts, lalu /tts/stream harus melayani utuh =="
curl -s -o /dev/null -X POST "$BASE/tts" -H 'content-type: application/json' -d "$BADAN"
curl -s -o /dev/null -D - -X POST "$BASE/tts/stream" \
  -H 'content-type: application/json' -d "$BADAN" \
  | grep -iE 'x-cache|x-stream'
