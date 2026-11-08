#!/usr/bin/env bash
# Contoh pemakaian gateway.
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8080}"

echo "== health =="
curl -s "$BASE/health" | python3 -m json.tool

echo "== normalizer (tanpa memuat model) =="
curl -s --get "$BASE/normalize" \
  --data-urlencode "text=Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026." \
  | python3 -m json.tool

echo "== sintesis =="
curl -s -D /tmp/h.txt -o hasil.wav \
  -X POST "$BASE/tts" -H 'content-type: application/json' \
  -d '{"text":"Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."}'
grep -iE 'x-cache|x-duration|x-wer' /tmp/h.txt
echo "-> hasil.wav"
ffprobe -v error -show_entries format=duration -of csv=p=0 hasil.wav

echo "== request identik (harus x-cache: hit) =="
curl -s -o /dev/null -D - -X POST "$BASE/tts" -H 'content-type: application/json' \
  -d '{"text":"Harga saham naik 15 persen menjadi Rp 12.500 per lembar pada 2026."}' \
  | grep -i 'x-cache'
