# Aset visual

| Berkas | Ukuran | Dipakai untuk |
|---|---|---|
| `laras-banner-dark.png` | 1280×536 | header README (tema gelap) |
| `laras-banner-light.png` | 1280×536 | header README (tema terang) |
| `laras-mark.png` | 512×512 | avatar repo/org, favicon, ikon aplikasi |
| `laras-social.png` | 1280×640 | social preview GitHub (Settings → Social preview) |

## Cara build

```bash
uv run --with pillow --with numpy python assets/build.py assets/source/lockup-original.jpg
```

Skrip hanya memetakan ulang luminansi, **tidak** menggambar ulang huruf. Jadi
tipografi asli tetap utuh; yang berubah cuma warna teks dan pemotongan margin.

## Kenapa ada dua versi banner

Lockup sumber memakai wordmark `#181818` di atas latar `#000000` — kontras
**1,34:1**, jauh di bawah ambang keterbacaan (WCAG AA butuh 4,5:1). Di GitHub
tema gelap, nama brand justru jadi elemen yang paling sulit dibaca. Versi
`dark` menaikkan wordmark ke `#F2F2F2` (18,8:1) dan tagline ke `#A8A8A8`
(8,0:1); versi `light` membalik latar supaya tetap enak di tema terang.

README memilih berkasnya otomatis lewat `<picture>` + `prefers-color-scheme`.

## Catatan

- Bar equalizer di dalam kepala mark adalah *knockout* transparan, jadi ikut
  warna latar. Itu sebabnya mark butuh PNG ber-alpha, bukan latar putih.
- Idealnya mark tersedia sebagai SVG asli dari desainer. Versi sekarang hasil
  rasterisasi ulang; cukup untuk 512 px ke bawah, tapi tidak sekelas vektor
  untuk cetak atau ukuran besar.
- Social preview **tidak bisa** diunggah lewat API GitHub. Unggah manual:
  Settings → Social preview → `laras-social.png`.
