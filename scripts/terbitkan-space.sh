#!/usr/bin/env bash
# Terbitkan demo Laras sebagai HF Space.
#
# Space-nya dibangun dari image ghcr yang sudah terbit, jadi tidak ada build di
# mesinmu dan tidak butuh GPU. Yang dibutuhkan cuma token HF dengan izin tulis.
#
#   hf auth login          # atau: export HF_TOKEN=hf_...
#   scripts/terbitkan-space.sh --user <nama-akun-hf>
#
# Setelah selesai skrip mencetak URL Space dan menunggu sampai statusnya
# "running" (build pertama biasanya 8–12 menit).
set -euo pipefail

NAMA="laras-tts"
USER_HF="${HF_USER:-}"
TAG="0.2.1"
TUNGGU=900 # detik menunggu build pertama

pakai() {
  cat <<'TXT'
pakai: scripts/terbitkan-space.sh --user <akun-hf> [--nama laras-tts] [--tag 0.2.1]

  --user   akun Hugging Face yang punya Space (wajib, atau set HF_USER)
  --nama   nama Space, default "laras-tts"
  --tag    tag rilis Laras yang dipakai sebagai basis image, default 0.2.1

Butuh token HF ber-izin tulis: jalankan `hf auth login` lebih dulu, atau
set variabel lingkungan HF_TOKEN.
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

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDEL="$AKAR/demo/hf-space"
for f in Dockerfile README.md; do
  [ -f "$BUNDEL/$f" ] || { echo "galat: bundel tidak lengkap, $BUNDEL/$f hilang" >&2; exit 1; }
done

# ---------------------------------------------------------------- prasyarat
if ! python3 -c 'import huggingface_hub' 2>/dev/null; then
  echo "galat: paket huggingface_hub belum ada." >&2
  echo "  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && command -v hf >/dev/null 2>&1; then
  HF_TOKEN="$(hf auth token 2>/dev/null || true)"
fi
# Jalur cadangan: CLI lama tidak punya subperintah untuk mencetak token, tapi
# huggingface_hub menyimpannya di berkas ini setelah `login`.
BERKAS_TOKEN="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$BERKAS_TOKEN" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$BERKAS_TOKEN")"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "galat: tidak ada token HF." >&2
  echo "  jalankan: hf auth login   (atau: export HF_TOKEN=hf_...)" >&2
  exit 1
fi
export HF_TOKEN

SIAPA="$(python3 - <<'PY'
from huggingface_hub import HfApi
try:
    print(HfApi().whoami()["name"])
except Exception as e:
    raise SystemExit(f"token ditolak: {e}")
PY
)" || { echo "galat: token HF tidak bisa dipakai (kedaluwarsa atau tanpa izin baca)." >&2; exit 1; }

echo "token milik: $SIAPA"
if [ "$SIAPA" != "$USER_HF" ]; then
  echo "catatan: --user=$USER_HF berbeda dari pemilik token ($SIAPA)." >&2
  echo "         pastikan akun itu memang boleh menulis ke Space tersebut." >&2
fi

# ------------------------------------------------------------------- image
TAG_REPO="v${TAG#v}"
TAG_IMAGE="${TAG#v}"
echo "memeriksa image ghcr.io/urdemonlord/laras-tts:$TAG_IMAGE"
TOKEN_GHCR="$(curl -fsS \
  "https://ghcr.io/token?scope=repository:urdemonlord/laras-tts:pull&service=ghcr.io" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
KODE="$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN_GHCR" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://ghcr.io/v2/urdemonlord/laras-tts/manifests/$TAG_IMAGE")"
if [ "$KODE" != "200" ]; then
  echo "galat: image tag $TAG_IMAGE tidak ada di ghcr (HTTP $KODE)." >&2
  echo "  pilih tag yang terbit dengan --tag, mis. --tag 0.2.1" >&2
  exit 1
fi

# ------------------------------------------------------------------- space
python3 - "$USER_HF" "$NAMA" <<'PY'
import sys
from huggingface_hub import HfApi
user, nama = sys.argv[1], sys.argv[2]
api = HfApi()
url = api.create_repo(
    repo_id=f"{user}/{nama}",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False,
)
print(f"space: {url}")
PY

KERJA="$(mktemp -d)"
trap 'rm -rf "$KERJA"' EXIT

# Token ditulis ke berkas 600 dan dibaca credential helper: kalau ditaruh di
# URL atau argv, isinya ikut terlihat di `ps` selama proses berjalan.
KRED="$KERJA/cred"
umask 077
printf 'https://oauth2:%s@huggingface.co\n' "$HF_TOKEN" > "$KRED"
GIT_KRED=(git -c "credential.helper=store --file=$KRED")

echo "mengambil isi Space"
"${GIT_KRED[@]}" clone -q "https://huggingface.co/spaces/$USER_HF/$NAMA" "$KERJA/space"

cp "$BUNDEL/Dockerfile" "$BUNDEL/README.md" "$KERJA/space/"
# Versi yang dipakai Space mengikuti --tag. Disunting di salinan, bukan di repo:
# bundel tetap menunjuk rilis terakhir yang diketahui baik.
sed -i \
  -e "s|^FROM ghcr.io/urdemonlord/laras-tts:.*|FROM ghcr.io/urdemonlord/laras-tts:${TAG_IMAGE}|" \
  -e "s|^ARG LARAS_TAG=.*|ARG LARAS_TAG=${TAG_REPO}|" \
  "$KERJA/space/Dockerfile"
# Space tidak butuh riwayat repo Laras; satu commit berisi dua berkas cukup.
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

# ----------------------------------------------------------------- tunggu
URL="https://huggingface.co/spaces/$USER_HF/$NAMA"
echo "menunggu build pertama (maks ${TUNGGU}s)…"
python3 - "$USER_HF" "$NAMA" "$TUNGGU" <<'PY'
import sys, time
from huggingface_hub import HfApi
user, nama, batas = sys.argv[1], sys.argv[2], int(sys.argv[3])
api = HfApi()
mulai = time.time()
terakhir = None
while time.time() - mulai < batas:
    info = api.space_info(f"{user}/{nama}")
    tahap = getattr(getattr(info, "runtime", None), "stage", None) or "UNKNOWN"
    if tahap != terakhir:
        print(f"  status: {tahap}")
        terakhir = tahap
    if tahap in ("RUNNING", "SLEEPING"):
        print("  siap")
        break
    if tahap in ("BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "DELETING"):
        print(f"  gagal di tahap {tahap}; lihat log di {info.id}")
        raise SystemExit(1)
    time.sleep(15)
else:
    print("  belum selesai dalam batas waktu; build masih jalan di latar Space")
PY

echo
echo "Space: https://huggingface.co/spaces/$USER_HF/$NAMA"
echo "Demo:  https://$USER_HF-$NAMA.hf.space"
