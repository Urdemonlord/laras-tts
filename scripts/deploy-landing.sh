#!/usr/bin/env bash
# Terbitkan landing page Laras ke https://laras.meowlabs.id
#
# Sumbernya folder `landing/` di repo ini. Situsnya statis: tidak ada build,
# tidak ada JS eksternal, tidak ada font pihak ketiga.
#
#   scripts/deploy-landing.sh
#
# Butuh sudo untuk `nginx -t` dan reload. DNS `laras` → IP box ini, diproksikan
# Cloudflare; sertifikat origin dari Let's Encrypt (certbot dns-cloudflare).
set -euo pipefail

AKAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMBER="$AKAR/landing"
TUJUAN="${TUJUAN:-/var/www/laras.meowlabs.id}"
URL="${URL:-https://laras.meowlabs.id/}"

[ -f "$SUMBER/index.html" ] || { echo "galat: $SUMBER/index.html tidak ada" >&2; exit 1; }

echo "salin $SUMBER → $TUJUAN"
sudo mkdir -p "$TUJUAN"
sudo chown "$(id -un):$(id -gn)" "$TUJUAN"
# --delete supaya berkas yang dihapus di repo ikut hilang di server.
rsync -a --delete "$SUMBER/" "$TUJUAN/"

echo "periksa konfigurasi nginx"
sudo nginx -t
sudo systemctl reload nginx

echo "verifikasi"
KODE="$(curl -s -o /dev/null -w '%{http_code}' "$URL")"
echo "  $URL → HTTP $KODE"
[ "$KODE" = "200" ] || { echo "galat: halaman tidak menjawab 200" >&2; exit 1; }
