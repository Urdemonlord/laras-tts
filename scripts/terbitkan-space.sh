#!/usr/bin/env bash
# Terbitkan demo Laras sebagai HF Space.
#
# Space-nya dibangun dari image ghcr yang sudah terbit, jadi tidak ada build di
# mesinmu dan tidak butuh GPU. Yang dibutuhkan cuma token HF dengan izin tulis
# plus `curl`, `git`, `jq` — tanpa paket Python apa pun.
#
#   export HF_TOKEN=hf_...        # atau: hf auth login
#   scripts/terbitkan-space.sh --user <nama-akun-hf>
#
# Setelah selesai skrip mencetak URL Space dan menunggu sampai statusnya
# RUNNING (build pertama biasanya 8–15 menit).
set -euo pipefail

NAMA="laras-tts"
USER_HF="${HF_USER:-}"
TAG="0.2.1"
TUNGGU=1200 # detik menunggu build pertama

pakai() {
  cat <<'TXT'
pakai: scripts/terbitkan-space.sh --user <akun-hf> [--nama laras-tts] [--tag 0.2.1]

  --user   akun Hugging Face yang punya Space (wajib, atau set HF_USER)
  --nama   nama Space, default "laras-tts"
  --tag    tag rilis Laras yang dipakai sebagai basis image, default 0.2.1

Butuh token HF ber-izin tulis: `hf auth login` lebih dulu, atau set HF_TOKEN.
Tidak butuh paket Python — hanya curl, git, dan jq.
TXT
}

while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_HF="${2:?--user butuh nilai}"; shift 2 ;;
    --nama) NAMA="${2:?--nama butuh nilai}"; shift 2 ;;
    --tag)  TAG="${2:?--tag butuh nilai}"; shift 2 ;;
    -h|--help) pakai; exit 0 ;;
    *) echo "argumen tidak dikenal: $1" >&2; pakai >&2; exit 2 ;;
  esac
done

[ -n "$USER_HF" ] || { echo "galat: --user belum diisi" >&2; pakai >&2; exit 2; }
for alat in curl git jq; do
  command -v "$alat" >/dev/null 2>&1 || { echo "galat: '$alat' tidak ada di PATH" >&2; exit 1; }
done

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDEL="$AKAR/demo/hf-space"
for f in Dockerfile README.md; do
  [ -f "$BUNDEL/$f" ] || { echo "galat: bundel tidak lengkap, $BUNDEL/$f hilang" >&2; exit 1; }
done

# ---------------------------------------------------------------- token
if [ -z "${HF_TOKEN:-}" ] && command -v hf >/dev/null 2>&1; then
  HF_TOKEN="$(hf auth token 2>/dev/null || true)"
fi
# Jalur cadangan: CLI lama tidak punya subperintah untuk mencetak token, tapi
# huggingface_hub menyimpannya di berkas ini setelah login.
BERKAS_TOKEN="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$BERKAS_TOKEN" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$BERKAS_TOKEN")"
fi
[ -n "${HF_TOKEN:-}" ] || {
  echo "galat: tidak ada token HF." >&2
  echo "  jalankan: hf auth login   (atau: export HF_TOKEN=hf_...)" >&2
  exit 1
}

# Token tidak pernah ditaruh di URL git atau argv: ditulis ke berkas 600 lalu
# dibaca credential helper, supaya tidak terlihat di `ps`.
KERJA="$(mktemp -d)"
trap 'rm -rf "$KERJA"' EXIT
umask 077
KRED="$KERJA/cred"
printf 'https://oauth2:%s@huggingface.co\n' "$HF_TOKEN" > "$KRED"
GIT_KRED=(git -c "credential.helper=store --file=$KRED")

api() { # api <metode> <jalur> [badan-json]
  curl -fsS -X "$1" \
    -H "Authorization: Bearer $HF_TOKEN" \
    -H "Content-Type: application/json" \
    ${3:+-d "$3"} \
    "https://huggingface.co$2"
}

SIAPA="$(api GET /api/whoami-v2 | jq -r '.name // empty')" || {
  echo "galat: token HF ditolak (kedaluwarsa atau tanpa izin baca)." >&2
  exit 1
}
[ -n "$SIAPA" ] || { echo "galat: token HF tidak mengembalikan nama akun." >&2; exit 1; }

echo "token milik: $SIAPA"
if [ "$SIAPA" != "$USER_HF" ]; then
  echo "catatan: --user=$USER_HF berbeda dari pemilik token ($SIAPA)." >&2
  echo "         pastikan akun itu memang boleh menulis ke Space tersebut." >&2
fi

# ---------------------------------------------------------------- image
TAG_REPO="v${TAG#v}"
TAG_IMAGE="${TAG#v}"
echo "memeriksa image ghcr.io/urdemonlord/laras-tts:$TAG_IMAGE"
TOKEN_GHCR="$(curl -fsS \
  "https://ghcr.io/token?scope=repository:urdemonlord/laras-tts:pull&service=ghcr.io" \
  | jq -r .token)"
KODE="$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN_GHCR" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://ghcr.io/v2/urdemonlord/laras-tts/manifests/$TAG_IMAGE")"
if [ "$KODE" != "200" ]; then
  echo "galat: image tag $TAG_IMAGE tidak ada di ghcr (HTTP $KODE)." >&2
  echo "  pilih tag yang terbit dengan --tag, mis. --tag 0.2.1" >&2
  exit 1
fi

# ---------------------------------------------------------------- space
echo "membuat Space $USER_HF/$NAMA (kalau belum ada)"
BADAN="$(jq -nc --arg n "$NAMA" '{type:"space",name:$n,private:false,sdk:"docker"}')"
KODE="$(curl -s -o "$KERJA/buat.json" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$BADAN" https://huggingface.co/api/repos/create)"
case "$KODE" in
  200 | 201) echo "  dibuat" ;;
  409) echo "  sudah ada, dipakai ulang" ;;
  *)
    echo "galat: gagal membuat Space (HTTP $KODE)" >&2
    cat "$KERJA/buat.json" >&2
    exit 1
    ;;
esac

echo "mengambil isi Space"
"${GIT_KRED[@]}" clone -q "https://huggingface.co/spaces/$USER_HF/$NAMA" "$KERJA/space"

cp "$BUNDEL/Dockerfile" "$BUNDEL/README.md" "$KERJA/space/"
# Versi yang dipakai Space mengikuti --tag. Disunting di salinan, bukan di repo:
# bundel tetap menunjuk rilis terakhir yang diketahui baik.
sed -i \
  -e "s|^FROM ghcr.io/urdemonlord/laras-tts:.*|FROM ghcr.io/urdemonlord/laras-tts:${TAG_IMAGE}|" \
  -e "s|^ARG LARAS_TAG=.*|ARG LARAS_TAG=${TAG_REPO}|" \
  "$KERJA/space/Dockerfile"

cd "$KERJA/space"
git config user.name "${GIT_AUTHOR_NAME:-$(git config --global user.name || echo terbitkan-space)}"
git config user.email "${GIT_AUTHOR_EMAIL:-$(git config --global user.email || echo terbitkan-space@localhost)}"
git add -A
if git diff --cached --quiet; then
  echo "isi Space sudah sama, tidak ada yang di-push"
else
  git commit -q -m "demo Laras $TAG: halaman depan + image multi-arch"
  echo "push ke huggingface.co/spaces/$USER_HF/$NAMA"
  "${GIT_KRED[@]}" push -q origin HEAD:main
fi

# ---------------------------------------------------------------- tunggu
echo "menunggu build pertama (maks ${TUNGGU}s)…"
MULAI=$(date +%s)
TERAKHIR=""
while :; do
  J="$(api GET "/api/spaces/$USER_HF/$NAMA?fields=runtime" || echo '{}')"
  TAHAP="$(printf '%s' "$J" | jq -r '.runtime.stage // "UNKNOWN"')"
  if [ "$TAHAP" != "$TERAKHIR" ]; then
    printf '  status: %s\n' "$TAHAP"
    TERAKHIR="$TAHAP"
  fi
  case "$TAHAP" in
    RUNNING | SLEEPING) break ;;
    BUILD_ERROR | CONFIG_ERROR | RUNTIME_ERROR | DELETING)
      echo "  gagal di tahap $TAHAP; log: https://huggingface.co/spaces/$USER_HF/$NAMA" >&2
      exit 1
      ;;
  esac
  if [ $(($(date +%s) - MULAI)) -ge "$TUNGGU" ]; then
    echo "  belum selesai dalam batas waktu; build masih jalan di latar Space"
    break
  fi
  sleep 15
done

echo
echo "Space: https://huggingface.co/spaces/$USER_HF/$NAMA"
echo "Demo:  https://$USER_HF-$NAMA.hf.space"
