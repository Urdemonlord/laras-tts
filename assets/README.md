# Aset visual

## Banner & mark

| Berkas | Ukuran | Dipakai untuk |
|---|---|---|
| `laras-banner-dark.png` | 1280×536 | header README (tema gelap) |
| `laras-banner-light.png` | 1280×536 | header README (tema terang) |
| `laras-mark.png` | 512×512 | avatar repo/org, sumber semua ikon |
| `laras-social.png` | 1280×640 | social preview GitHub (Settings → Social preview) |

## Ikon web (`assets/icons/`)

| Berkas | Ukuran | Dipakai untuk |
|---|---|---|
| `favicon.ico` | 16 + 32 + 48 | tab browser, bookmark, desktop shortcut |
| `favicon-16x16.png` | 16×16 | tab DPI standar |
| `favicon-32x32.png` | 32×32 | tab retina |
| `favicon-48x48.png` | 48×48 | bookmark bar, shortcut |
| `apple-touch-icon.png` | 180×180 | homescreen iOS/iPadOS (latar solid, iOS tidak suka transparan) |
| `icon-192.png` | 192×192 | PWA Android |
| `icon-512.png` | 512×512 | PWA splash / store |
| `icon-maskable-512.png` | 512×512 | PWA maskable — konten di dalam safe zone 80% |
| `contact-sheet.png` | 760×510 | QA: semua ukuran + uji safe zone |

Pasang di halaman web:

```html
<link rel="icon" href="/assets/icons/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="/assets/icons/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/assets/icons/favicon-16x16.png">
<link rel="apple-touch-icon" sizes="180x180" href="/assets/icons/apple-touch-icon.png">
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#5B2ED6">
```

`manifest.json` ada di root repo (`name`, `short_name`, ikon, `theme_color`,
`display: standalone`). Path ikon di dalamnya relatif terhadap root, jadi
`start_url` dan `scope` diset `/`.

## Cara build

```bash
# banner, mark, social preview
uv run --with pillow --with numpy python assets/build.py assets/source/lockup-original.jpg

# favicon set + ikon PWA (butuh laras-mark.png dari langkah di atas)
uv run --with pillow --with numpy python assets/build_icons.py
```

`build.py` hanya memetakan ulang luminansi, **tidak** menggambar ulang huruf.
Jadi tipografi asli tetap utuh; yang berubah cuma warna teks dan pemotongan
margin.

## Kenapa ada dua versi banner

Lockup sumber memakai wordmark `#181818` di atas latar `#000000` — kontras
**1,34:1**, jauh di bawah ambang keterbacaan (WCAG AA butuh 4,5:1). Di GitHub
tema gelap, nama brand justru jadi elemen yang paling sulit dibaca. Versi
`dark` menaikkan wordmark ke `#F2F2F2` (18,8:1) dan tagline ke `#A8A8A8`
(8,0:1); versi `light` membalik latar supaya tetap enak di tema terang.

README memilih berkasnya otomatis lewat `<picture>` + `prefers-color-scheme`.

## Kenapa favicon 16/32 px bukan mark lengkap

Mark punya busur suara tipis dan bar equalizer *knockout*. Diuji pada 16 px:
busur hilang, knockout menyatu jadi bercak gelap, dan hasilnya terbaca sebagai
blob ungu — bukan kepala, bukan audio. Tiga bar equalizer adalah bentuk paling
sederhana dari mark yang sama dan tetap tajam di 16 px, jadi:

| Ukuran | Bentuk |
|---|---|
| 16, 32 px | tiga bar equalizer (gradien brand) |
| 48 px | siluet kepala + tiga bar putih, tanpa busur |
| 180 px ke atas | mark lengkap |

Konsekuensinya 48 px tampil beda dari 192/512 px. Itu disengaja: ukuran kecil
butuh bentuk yang bertahan setelah di-downsample, ukuran besar butuh detail.

## Catatan

- Bar equalizer di dalam kepala mark adalah *knockout* transparan, jadi ikut
  warna latar. Itu sebabnya mark butuh PNG ber-alpha, bukan latar putih.
- Sumber lockup punya satu artefak: garis 32×5 px terlepas di kanan bawah
  (area 155 px dari total 105 ribu). `build_icons.py` membuang komponen di
  bawah 300 px, jadi ikon bersih. Banner tidak terpengaruh karena memakai
  gambar penuh.
- `favicon.ico` ditulis manual di `build_icons.py` (`write_ico`) karena
  Pillow hanya menyimpan satu frame saat `append_images` dipakai untuk ICO.
- Idealnya mark tersedia sebagai SVG asli dari desainer. Versi sekarang hasil
  rasterisasi ulang; cukup untuk 512 px ke bawah, tapi tidak sekelas vektor
  untuk cetak atau ukuran besar.
- Social preview **tidak bisa** diunggah lewat API GitHub. Unggah manual:
  Settings → Social preview → `laras-social.png`.
