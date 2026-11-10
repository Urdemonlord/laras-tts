#!/usr/bin/env bash
# Unduh bobot ONNX Laras dari Hugging Face, verifikasi sha256, susun direktori
# yang siap dipakai server. Tidak ada dependensi selain curl + coreutils.
#
#   scripts/unduh-model.sh                # model + satu contoh referensi
#   scripts/unduh-model.sh --tanpa-ref    # model saja
#   scripts/unduh-model.sh --cek          # hanya verifikasi, tidak mengunduh
#   scripts/unduh-model.sh --dir /data/models --ref-dir /data/ref
#
# Keluar dengan kode 1 kalau ada berkas yang hash-nya tidak cocok, supaya bisa
# dipakai di CI.
set -euo pipefail

REPO="anak10thn/pocket-tts-indonesian"
AKAR="https://huggingface.co/${REPO}/resolve/main"

DIR="data/models"
DIR_REF="data/ref"
MODE_CEK=0
PAKSA=0
AMBIL_REF=1

# nama|ukuran byte|sha256
MANIFES=(
  "flow_lm_main_int8.onnx|75865936|dba97a71fdb9f099165298f930936d170a42d14967d1fb0ab6853a84a7fef026"
  "mimi_encoder.onnx|39259746|ec1ec959af10d161776a4385043448409e371312ceed5d7ff2ee7e47c24b766a"
  "mimi_decoder_int8.onnx|22616918|a05a81992037d3322e8945c3c7884e7a1f31b31932bdb550e5ee2861f2a6f79d"
  "text_conditioner.onnx|16388310|f3927e4c01994d1aaad306820aead1033cabc30bb3ac43469ff9919a3d2006e7"
  "flow_lm_flow_int8.onnx|9935594|5160e2a82d0ce3839c048b3da413ab8c2fa330202a7dd00b55fe81934b8e0791"
  "tokenizer.model|311347|3ae0af2658bf8f54c6720603b4d965cad97d67cb0f245636d0eb23ff4f865fd8"
  "bos_before_voice.npy|4224|2c52fe74873052f8daa1d0e6ff079c474b184ec2165c20cc4e5bca41ff9bbf0c"
  "VALIDATION.txt|2494|adaee7c240d0a5a7cdb9642e8da7be766705fb3ef7cac534fd58899ca39357ca"
  "bundle.json|497|a777975cbbd61eaab340fcfeb71b2f77f2f8f236ce296657587db1769faeb710"
)

# Contoh referensi: Common Voice 22.0 Indonesia, CC0 (domain publik), 48 kHz
# mono ±4,7 detik. Format MP3 — server mengonversinya sendiri lewat ffmpeg.
REF_NAMA="common_voice_id_19961910.mp3"
REF_UKURAN=36909
REF_SHA="3634489c1d07479c2c758e5c197a96bb66077d110e636040edaeb18a41ab7f12"
REF_SUMBER="${AKAR}/voices/${REF_NAMA}"

pakai() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="${2:?--dir butuh argumen}"; shift 2 ;;
    --ref-dir) DIR_REF="${2:?--ref-dir butuh argumen}"; shift 2 ;;
    --cek) MODE_CEK=1; shift ;;
    --paksa) PAKSA=1; shift ;;
    --tanpa-ref) AMBIL_REF=0; shift ;;
    -h|--help) pakai; exit 0 ;;
    *) echo "argumen tidak dikenal: $1 (coba --help)" >&2; exit 2 ;;
  esac
done

# sha256sum ada di Linux, shasum di macOS.
if command -v sha256sum >/dev/null 2>&1; then
  sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "butuh sha256sum atau shasum untuk verifikasi" >&2
  exit 2
fi

manusia() { awk -v b="$1" 'BEGIN { printf "%.1f MB", b / 1048576 }'; }

# unduh URL TUJUAN UKURAN SHA
unduh() {
  local url="$1" tujuan="$2" ukuran="$3" harap="$4"
  local sementara="${tujuan}.part"

  if [ -e "$tujuan" ] && [ "$PAKSA" -eq 0 ]; then
    local ada; ada=$(stat -c %s "$tujuan" 2>/dev/null || stat -f %z "$tujuan")
    if [ "$ada" = "$ukuran" ] && [ "$(sha "$tujuan")" = "$harap" ]; then
      printf '  %-26s %10s  sudah benar\n' "$(basename "$tujuan")" "$(manusia "$ada")"
      return 0
    fi
    echo "  $(basename "$tujuan"): hash/ukuran tidak cocok, unduh ulang"
  fi

  printf '  %-26s %10s  mengunduh…' "$(basename "$tujuan")" "$(manusia "$ukuran")"
  local args=(-fsSL --retry 3 --retry-delay 2 -C - -o "$sementara")
  [ -n "${HF_TOKEN:-}" ] && args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  if ! curl "${args[@]}" "$url"; then
    echo " gagal"
    rm -f "$sementara"
    echo "  → unduhan gagal. Kalau repo-nya minta login: set HF_TOKEN." >&2
    return 1
  fi

  local dapat; dapat=$(stat -c %s "$sementara" 2>/dev/null || stat -f %z "$sementara")
  if [ "$dapat" != "$ukuran" ]; then
    echo " ukuran salah ($dapat ≠ $ukuran)"
    rm -f "$sementara"
    return 1
  fi
  local dapat_sha; dapat_sha=$(sha "$sementara")
  if [ "$dapat_sha" != "$harap" ]; then
    echo " sha256 tidak cocok"
    echo "    harap: $harap" >&2
    echo "    dapat: $dapat_sha" >&2
    rm -f "$sementara"
    return 1
  fi
  mv "$sementara" "$tujuan"
  echo " sha256 ok"
}

gagal=0

if [ "$MODE_CEK" -eq 1 ]; then
  echo "Verifikasi ${DIR}"
else
  mkdir -p "$DIR"
  echo "Mengunduh bobot Laras dari ${REPO}"
fi

for baris in "${MANIFES[@]}"; do
  IFS='|' read -r nama ukuran harap <<< "$baris"
  tujuan="${DIR}/${nama}"
  if [ "$MODE_CEK" -eq 1 ]; then
    if [ ! -e "$tujuan" ]; then
      printf '  %-26s %10s  HILANG\n' "$nama" "-"
      gagal=1
    elif [ "$(sha "$tujuan")" = "$harap" ]; then
      printf '  %-26s %10s  ok\n' "$nama" "$(manusia "$ukuran")"
    else
      printf '  %-26s %10s  SHA256 SALAH\n' "$nama" "$(manusia "$ukuran")"
      gagal=1
    fi
  else
    unduh "${AKAR}/onnx/${nama}" "$tujuan" "$ukuran" "$harap" || gagal=1
  fi
done

if [ "$AMBIL_REF" -eq 1 ]; then
  if [ "$MODE_CEK" -eq 1 ]; then
    tujuan="${DIR_REF}/${REF_NAMA}"
    if [ -e "$tujuan" ]; then
      printf '  %-26s %10s  ok\n' "$REF_NAMA" "$(manusia "$REF_UKURAN")"
    else
      printf '  %-26s %10s  (opsional, belum ada)\n' "$REF_NAMA" "$(manusia "$REF_UKURAN")"
    fi
  else
    mkdir -p "$DIR_REF"
    unduh "$REF_SUMBER" "${DIR_REF}/${REF_NAMA}" "$REF_UKURAN" "$REF_SHA" || gagal=1
  fi
fi

if [ "$gagal" -ne 0 ]; then
  echo
  echo "Ada berkas yang gagal diverifikasi. Jangan dipakai — ulangi unduhan." >&2
  exit 1
fi

if [ "$MODE_CEK" -eq 1 ]; then
  echo
  echo "Semua berkas cocok dengan manifes."
  exit 0
fi

echo
echo "Selesai. Model siap di ${DIR} ($(du -sh "$DIR" | cut -f1))."
echo "Jalankan:"
echo "  cargo build --release --features onnx"
echo "  MODELS_DIR=${DIR} REF_WAV=${DIR_REF}/${REF_NAMA} ./target/release/laras"
